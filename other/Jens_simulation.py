#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FEM-Eigenmoden (Maxwell, PEC) einer Pillbox-Kavität mit dolfinx + SLEPc.

Abhängigkeiten:
  pip install gmsh
  # dolfinx + petsc4py + slepc4py am besten via offiziellem dolfinx-Container/conda

Ergebnis:
  - Ausgabe der ersten Eigenfrequenzen (Hz)
  - XDMF-Datei mit E-Feld der ersten Modi zur Visualisierung (ParaView)
"""

import math
from mpi4py import MPI
import numpy as np

import gmsh
import ufl
from petsc4py import PETSc
from slepc4py import SLEPc

from dolfinx import mesh, fem, io
from dolfinx.io import gmshio
import dolfinx

# ---------- Parameter ----------
R = 0.10        # Radius [m]
L = 0.02        # Länge  [m]
lc = 0.008      # Ziel-Meshgröße [m] (feiner -> genauer, aber teurer)
nev = 6         # Anzahl gewünschter Eigenwerte
c0  = 299792458.0  # Lichtgeschwindigkeit [m/s]

# Optional: Zielwert (shift) um TM010 zu treffen: k^2 ~ (chi01/R)^2
chi01 = 2.404825557695773  # erste Nullstelle von J0'
target_k2 = (chi01 / R) ** 2

comm = MPI.COMM_WORLD
rank = comm.rank

# ---------- Gmsh-Geometrie & Mesh ----------
gmsh.initialize()
gmsh.model.add("pillbox")
# Zylinder mit Achse in z-Richtung: Mittelpunkt bei (0,0,0), Höhe L
# gmsh: Zylinder wird von z=-L/2 bis z=+L/2 erzeugt
cyl = gmsh.model.occ.addCylinder(0.0, 0.0, -L/2, 0.0, 0.0, L, R)
gmsh.model.occ.synchronize()

# Physikalische Gruppen (1: Volumen, 2: Rand)
vol_tag = 1
gmsh.model.addPhysicalGroup(3, [cyl], vol_tag)
gmsh.model.setPhysicalName(3, vol_tag, "cavity")

# Alle Randflächen des Zylinders holen und als ein physikalischer Rand gruppieren
surf_entities = gmsh.model.getBoundary([(3, cyl)], oriented=False, recursive=False)
surf_tags = [s[1] for s in surf_entities]
bnd_tag = 2
gmsh.model.addPhysicalGroup(2, surf_tags, bnd_tag)
gmsh.model.setPhysicalName(2, bnd_tag, "pec_boundary")

# Mesh-Optionen
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
gmsh.model.mesh.generate(3)

# Nach dolfinx überführen
domain, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, comm, rank, gdim=3)
gmsh.finalize()

# ---------- Funktionsraum (H(curl), Nédélec) ----------
V = fem.FunctionSpace(domain, ("N1curl", 1))  # 1. Ordnung

# ---------- PEC-Randbedingung n x E = 0 ----------
# Für Nédélec bedeutet "E=0" auf der Randspur -> tangentiale Komponente = 0.
# Wir setzen "homogene Dirichlet" auf allen Rand-Facetten der phys. Gruppe 'pec_boundary'.
facet_indices = np.array(facet_tags.find(bnd_tag), dtype=np.int32)
bdry_dofs = fem.locate_dofs_topological(V, dim=2, entities=facet_indices)
zero_vec = fem.Function(V)
zero_vec.x.array[:] = 0.0
bcs = [fem.dirichletbc(zero_vec, bdry_dofs)]

# ---------- Schwache Form ----------
E = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
dx = ufl.Measure("dx", domain=domain)

a_form = ufl.inner(ufl.curl(E), ufl.curl(v)) * dx         # Steifigkeitsoperator (curl-curl)
m_form = ufl.inner(E, v) * dx                             # Massenoperator (L2)

A = fem.petsc.assemble_matrix(fem.form(a_form), bcs=bcs)
A.assemble()
M = fem.petsc.assemble_matrix(fem.form(m_form), bcs=bcs)
M.assemble()

# ---------- Eigenwertproblem A x = lambda M x ----------
eps = SLEPc.EPS().create(comm)
eps.setOperators(A, M)
eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)  # verallg. Hermitesch
eps.setDimensions(nev, PETSc.DEFAULT, PETSc.DEFAULT)

# Spektrales Target nahe TM010, optional shift-invert für Robustheit
st = eps.getST()
st.setType(SLEPc.ST.Type.SINVERT)  # Shift-Invert
eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
eps.setTarget(target_k2)

# Toleranzen / max Iterationen (anpassbar)
eps.setTolerances(1e-9, 200)
eps.setFromOptions()  # erlaubt -eps_* Optionen zur Feinsteuerung

eps.solve()

if rank == 0:
    print(f"Solved eigenproblem: nconv = {eps.getConverged()} (requested nev={nev})")
    print("Eigenwerte lambda = (omega/c)^2; Frequenz f = omega/(2*pi)")

# ---------- Eigenpaare extrahieren & Ergebnisse ----------
nconv = eps.getConverged()
vr = A.createVecRight()
vi = A.createVecRight()

# XDMF-Ausgabe vorbereiten (Feld der ersten Modi)
with io.XDMFFile(comm, "pillbox_modes.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)

    for i in range(min(nev, nconv)):
        kr = eps.getEigenpair(i, vr, vi)
        # numerische Frequenz
        omega_over_c = math.sqrt(max(kr, 0.0))
        omega = omega_over_c * c0
        f = omega / (2.0 * math.pi)

        if rank == 0:
            print(f"[{i}] lambda={kr:.8e}  ->  f = {f:.6f} Hz")

        # Eigenvektor in Function schreiben (nur Realteil)
        # (Maxwell-Moden sind bis auf Phase definiert; vi sollte ~0 sein)
        E_mode = fem.Function(V)
        E_mode.vector.setArray(vr.array_r)  # PETSc -> dolfinx
        E_mode.name = f"E_mode_{i}"

        # In Datei (ParaView: Glyph/Arrows, Streamlines etc.)
        with io.XDMFFile(comm, "pillbox_modes.xdmf", "a") as xdmf_append:
            xdmf_append.write_function(E_mode)

if rank == 0:
    # Referenz: analytische TM010-Frequenz (ideal leitende Wand, keine Endfeld-Korrekturen)
    f_TM010 = (chi01 * c0) / (2.0 * math.pi * R)
    print("\nReferenz (ungekoppelt, ideal PEC):")
    print(f"  TM010 ~ {f_TM010:.6f} Hz  (nur radial, unabhängig von L)")

    print("\nHinweise:")
    print(" - Erhöhen Sie die Meshfeinheit (lc), um die Genauigkeit zu verbessern.")
    print(" - Die kleinsten Eigenwerte können spurious/Gradienten-Moden sein; "
          "der gesetzte Target-Shift hilft, physikalische Moden zu treffen.")
    print(" - Visualisierung: 'pillbox_modes.xdmf' in ParaView öffnen (E-Vector-Feld).")
