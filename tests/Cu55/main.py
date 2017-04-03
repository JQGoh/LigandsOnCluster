""" Attach oxygen molecule to the surface of Cu55 icosahedron cluster
    ineractively, by choosing the selected surface atoms as anchored sites.
"""
import sys
sys.path.append("../../bin")
from surfaceatom import SurfaceAtom, add_interactive
from ase.io import read
from ase import Atoms

fname = "Cu55.xyz"
system = read(fname)

ligand = Atoms('O2', positions=[
     (1.536, 0.451, 1.591),
     (2.289, 1.724, 1.591),
    ])

nanocluster = SurfaceAtom(system)
radii = dict(Cu=2.0)
nanocluster.get_surf_atoms(radii)

print("The determined surface atoms: Total {}".format(len(nanocluster.surf)))
print("Surface atom indices:")
print(nanocluster.surf)

add_interactive(nanocluster, ligand, fname)
