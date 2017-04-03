"""This module contains the utilities or general functions that help to
perform simple calculaions or operations.
"""
import numpy as np
import os
from ase.io import write


def ase_debug(atoms):
    """Use ase-gui to view the structure. It always use the same window's
    name known as ase-temp.xyz. Useful for debugging purpose.

    Parameters
    ----------
    atoms : ASE atoms object
        ASE atoms object representing the system.
    """
    fd = open('ase-temp.xyz', 'w')
    write(fd, atoms, format='xyz')
    fd.close()
    os.system('ase-gui --verbose ase-temp.xyz &')


def reflect(xyzs, symmetry="xz"):
    """Reflects the coordinates with respect to a defined symmetry plane.

    Parameters
    ----------
    xyzs : numpy.ndarray, shape [n_atoms, 3]
         The xyz positions of atoms.

    symmetry : str, optional (default=xz)
        - 'xz' xz plane
        - 'yz' yz plane
        - 'xy' xy plane

    Returns
    -------
    new : numpy.ndarray, shape [n_atoms, 3]
         The atomic positions reflected with respect to the symmetry plane.
    """
    if symmetry == "xz":
        change = 1
    elif symmetry == "yz":
        change = 0
    elif symmetry == "xy":
        change = 2
    else:
        raise ValueError("symmetry must be xz|yz|xy")

    new = xyzs 
    new[:, change] = -1*new[:, change]
    return new


def angle_vec(v0, v1):
    """"Determine the angle between the two vectors.

    Parameters
    ----------
    v0 : array, shape [3,]
        xyz components of the first vector.
    v1 : array, shape [3,]
        xyz components of the second vector.

    Returns
    -------
    angle : float
        Angle (unit radian).
    """
    mag0 = np.sqrt(np.dot(v0, v0))
    mag1 = np.sqrt(np.dot(v1, v1))
    angle = np.dot(v0, v1)/(mag0*mag1)

    # To avoid overflow of float value
    # np.arccos(-1.0) will return NaN
    if ((angle + 1.0) < 1e-6):
        angle = 3.14159265359
    else:
        angle = np.arccos(angle)

    return angle


def rotate_vec(theta, vec_axis, vec):
    """Rotate a vector with respect to a chosen axis.

    Notes
    -----
    Rotation matrix refer to
    http://mathworld.wolfram.com/RodriguesRotationFormula.html

    Parameters
    ----------
    theta : float
        The angle we wish to rotate (unit radian).

    vec_axis : array, shape [3,]
        The vector specifying the rotation axis. 

    vec : array, shape [3,]
        The vector to be rotated. 

    Returns
    -------
    rotated_vec : array, shape [3,]
        The vector rotated based on the defined vec_axis and theta.
    """
    mag = np.sqrt(np.dot(vec_axis, vec_axis))
    if abs(mag - 1.0000) > 1e-4:
        # renormalize the vec_axis to be a unit vector
        vec_axis = vec_axis/mag

    u = vec_axis[0]
    v = vec_axis[1]
    w = vec_axis[2]
    rot_matrix = np.zeros((3, 3))
    rot_matrix[0, 0] = np.cos(theta) + u*u*(1 - np.cos(theta))
    rot_matrix[0, 1] = u*v*(1 - np.cos(theta)) - w*np.sin(theta)
    rot_matrix[0, 2] = v*np.sin(theta) + u*w*(1 - np.cos(theta))
    rot_matrix[1, 0] = w*np.sin(theta) + u*v*(1 - np.cos(theta))
    rot_matrix[1, 1] = np.cos(theta) + v*v*(1 - np.cos(theta))
    rot_matrix[1, 2] = -u*np.sin(theta) + v*w*(1 - np.cos(theta))
    rot_matrix[2, 0] = -v*np.sin(theta) + u*w*(1 - np.cos(theta))
    rot_matrix[2, 1] = u*np.sin(theta) + v*w*(1 - np.cos(theta))
    rot_matrix[2, 2] = np.cos(theta) + w*w*(1 - np.cos(theta))
    rotated_vec = np.dot(rot_matrix, vec)

    return rotated_vec
