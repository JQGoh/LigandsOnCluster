""" Creating monomers (CH3-S-Au-S-CH3) using create_monomers() 
    method for Au13 icosahedron. 
"""
import sys
sys.path.append("../../bin/")
from surfaceatom import SurfaceAtom, monomer
from ase.io import read, write
from ase.visualize import view
from ase import Atoms
import numpy as np
import random

random.seed(1234)
np.random.seed(1234)

fname = "Au13.xyz"
system = read(fname)
print("Our system:")
print(system.get_chemical_formula(mode="hill"))

methyl = Atoms('CH3', positions=[
    (-0.530346, -1.324869, 1.143010),
    (-1.616426, -1.426324, 1.054435),
    (-0.268829, -1.019727, 2.161262),
    (-0.052434, -2.279343, 0.917096)
    ])

nanocluster = SurfaceAtom(system)
radii = dict(Au=2.0)
nanocluster.get_surf_atoms(radii)

print("The determined surface atoms: Total {}".format(len(nanocluster.surf)))
print("Surface atom indices:")
print(nanocluster.surf)

# Use the center of mass (ref_neighbor=0) as a reference, to indicate the xz-plane
param = monomer(side_group=methyl, over_r=2.0, ref_neighbor=0)
nanocluster.create_monomers(monomer=param, num=6)
write('Au13-ligands.xyz', nanocluster)
view(nanocluster)
