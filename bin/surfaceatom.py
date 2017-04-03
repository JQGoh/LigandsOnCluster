import numpy as np
import math
import copy
import random
import textwrap

from scipy import stats
from ase import Atoms, Atom
from ase.io import write
from ase.neighborlist import NeighborList
from ase.visualize import view
from utilities import rotate_vec, angle_vec, reflect, ase_debug


class monomer(object):
    """Define the properties of a monomer.

    Parameters
    ---------- 
    SR : float, optional (default=2.5)
        The distance between the ligand and core surface (unit angstrom).

    delta : float, optional (default=0.087266)
        The angle introduced to displace the ligand with respect to xz
        plane (unit rad, 0.087266 = 5 deg).

    std : list, optional (default=[0.2, 0.017453])
        A list of float values indicating the standard deviations for SR and
        delta, respectively (unit angstrom and radian, 0.017453 rad = 1 deg).

    symbol : list, optional (default=['S', 'Au'])
        A list of str specifying the chemical elements for ligand. 
        ['S', 'Au'] refers to the monomer motif of gold-thiol.

    side_group : list, optional (default=[])
        A list of atomic information which specify the atomic elements
        and positions of side-group attached to the monomer. For instance,
        CH3-S-Au-S-CH3 has methyl group, and we should specify the 
        atomic positions of the single carbon atom + three hydrogen atoms.
        The carbon atom is recommended to be placed adjacent to the S
        atom, assuming the S atom is positioned (0., 0., 0.).
        The format should agree with ASE Atoms instance.

    over_r : float, optional (default=1.5)
        Two atoms are considered overlapped, if the distance between them <=
        this value (unit angstrom). This prevents the created ligand 
        overlapping with the existing atoms. 

    ref_neighbor : int, optional (default=1)
        Specify the reference point used to defined the xz plane.
        Refer to parameter 'plane_atom' of add_ligand() method for details.
        
        - '0' Use the center of mass as the cluster as a reference.
        - '1' Use the neighboring atom of the anchored sites, and it will
            generate a reference point beneath the surface.

    debug : boolean, optional (default=False)
        If True, display the newly generated ligand proposed by user. Users will
        learn about the positions of the newly generated ligand, and check
        whether it is appropriate. This is important for user to be aware of the
        parameters chosen for this monomer instance. 
    """
    def __init__(self, SR=2.5, delta=0.087266, std=[0.2, 0.017453],
                 symbol=['S', 'Au'], side_group=[], over_r=1.5,
                 ref_neighbor=1, debug=False):
        self.SR = SR
        self.delta = delta
        self.std = std
        self.symbol = symbol
        self.side_group = side_group
        self.over_r = over_r
        self.ref_neighbor = ref_neighbor
        self.debug = debug 


class SurfaceAtom(Atoms):
    """Represent the surface atoms of a cluster.

    It inherits all the attributes of Atoms instance from ASE package.
    It also defines the surface atoms by providing an additional list
    of indices.

    Parameters
    ----------
    atoms : ASE Atoms instance
        Imported ASE Atoms instance representing our system.

    Attributes
    ----------
    com: numpy.ndarray, shape [3,]
        The center of mass of the cluster. 

    core_num: int
        The total number of atoms for a given system.

    Notes
    -----
    The following attributes requires the calling of get_surf_atoms()

    nb : tuple
        A tuple of size N where N is the total number of atoms.
        Each entry has the tuple of atom indices which refer to neighboring
        atoms as defined by ASE's NeighborList method.

    surf : tuple
        A tuple of integers, specify the indices of surface atoms

    avail : list
        A list of integers indicating the indices of surface atoms
        which can be anchored by a designed ligand.
        Default: same as self.surf determined by get_surf_atoms().

    nb_surf : dict
        Dictionary with surface atom indices as key values. Each entry
        has the tuple of other surface atom indices as the corresponding
        nearest neighbors indices. 
        Size of dictionary = Total number of surface atoms.

    nnb_surf : dict
        Dictionary with surface atom indices as key values. Each entry
        has the tuple of indices for surface atoms as the corresponding
        NEXT nearest neighbors indicees.
        Size of dictionary: Total number of surface atoms.
    """
    def __init__(self, atoms):
        super(SurfaceAtom, self).__init__(atoms)
        self.com = self.get_center_of_mass()
        self.core_num = self.get_number_of_atoms()

    def get_surf_atoms(self, radii, dtheta=20*math.pi/180.0,
            dphi=45*math.pi/180.0, factor=0.95):
        """Determine the surface atoms of a system.
        
        Notes
        -----
        Create the displacement vectors pointing to the points on a spherical
        surface, with respect to the list of atoms. A surface atom will have
        most of these surface points (default=95%) overlapped by the spherical
        regions of neighboring atoms.
        
        Calling this update self.nb, self.surf, self.avail,
        also excecute get_nnb().

        Parameters
        __________
        radii : dict
            Chemical symbols as keys, values specifying radii for spherical
            regions of atoms with respect to their elements.
            Radius, r = cutoffs + skin (set skin=0.0) in NeighborList.

        dtheta : float, optional (default=20 deg)
            Resolution of theta angle defining points on a spherical surface.

        dphi : float, optional (default=25 deg)
            Resolution of phi angle defining points on a spherical surface.

        factor: float, optional (default=0.95)
            Minimum proportion of points overlapped by neighboring spheres, 
            which suggest the atom as non-surface atom.
        """
        kind_pts = dict()
        portion = dict()

        if not radii:
            raise ValueError("radii not defined by users.")

        # Create the points on the surface of a sphere with radii of 
        # type of Atoms
        for kind, r in radii.items():
            pts = []
            for theta in np.arange(0.0, 1*math.pi, dtheta):
                for phi in np.arange(0.0, 2*math.pi, dphi):
                    pts.append([r*math.sin(theta)*math.cos(phi),
                        r*math.sin(theta)*math.sin(phi),
                        r*math.cos(theta)])
            kind_pts[kind] = np.vstack(pts)
            # Non-overlapped points > portion[kind], it is not a surface atom
            portion[kind] = int(math.floor((1 - factor)*len(kind_pts[kind])))

        # Create the list of surface points with respect to kind atoms
        surf_pts = []
        for i in range(self.get_number_of_atoms()):
            y = self.get_chemical_symbols()[i]
            surf_pts.append(kind_pts[y] + self.positions[i])

        # Define the list of radii with respect to kind of atoms
        list_radii = []
        for i in range(self.get_number_of_atoms()):
            y = self.get_chemical_symbols()[i]
            list_radii.append(radii[y])

        # Obtain the list of neighbors
        nl = NeighborList(
            list_radii, 0.0, self_interaction=False, bothways=True)
        nl.update(self)
        self.nb = []
        for i in range(self.get_number_of_atoms()):
            indices, offsets = nl.get_neighbors(i)
            self.nb.append(tuple([int(x) for x in indices]))
        # Set to tuple of tuple
        self.nb = tuple(self.nb)

        # Determine whether it is a surface atom
        self.surf = []
        for i in range(self.get_number_of_atoms()):
            y = self.get_chemical_symbols()[i]
            # create a copy of the surface points
            pts_copy = surf_pts[i]
            # Compare to the centers of neighboring atoms
            for j in self.nb[i]:
                remove_list = []
                # break the loop if pts_copy becomes an empty list
                if (len(pts_copy) == 0):
                    break
                for (idx, x) in enumerate(pts_copy):
                    r2 = x - self.positions[j]
                    r2 = np.dot(r2, r2)
                    if (np.sqrt(r2) < list_radii[j]):
                        remove_list.append(idx)
                # The end of checking, only modify pts_copy
                pts_copy = np.delete(pts_copy, remove_list, axis=0)
            if (len(pts_copy) > portion[y]):
                self.surf.append(i)
        self.surf = tuple(self.surf)

        # Update the available sites for the creation of ligands
        self.avail = list(self.surf)
        # Update and obtain the next nearest neighbors of the surface atoms
        self.get_nnb()

        assert len(self.nnb_surf) == len(
            self.nb_surf), 'length of self.nnb_surf != lenghth of self.nb_surf'

    def get_nnb(self):
        """Determine the nearest neighbor, next nearest neighbor surface atoms.

        Notes
        -----
        Update self.nb_surf, self.nnb_surf.
        """
        if not hasattr(self, 'surf'):
            raise AttributeError(
                "Surface Atom list is not defined. Call get_surf_atoms() first.")
        if not hasattr(self, 'nb'):
            raise AttributeError(
                "Neighbor list is not defined. get_surf_atoms() first.")

        self.nb_surf = dict()
        self.nnb_surf = dict()
        for x in self.surf:
            collect = []
            nb_surf = []
            for y in self.nb[x]:
                if y in self.surf:
                    # Collect the surface atoms into nb_surf
                    nb_surf.append(y)
                    # Loop through the second nearest neighbor
                    for z in self.nb[y]:
                        if z in self.surf and z not in self.nb[x] and z != x:
                            collect.append(z)
            # Remove the duplicate
            collect = set(collect)
            self.nnb_surf[x] = tuple(list(collect))
            self.nb_surf[x] = tuple(nb_surf)

    def check_surf(self):
        """Check whether the user has invoked get_surf_atoms() instance method.
        """
        if not hasattr(self, 'surf'):
            raise AttributeError(
                """Surface Atom list (self.surf) not defined.
                Call get_surf_atoms() first.""")
        if not hasattr(self, 'nb'):
            raise AttributeError(
                """Neighbor list (self.nb) not defined.
                Call get_surf_atoms() first.""")
        if not hasattr(self, 'nb_surf'):
            raise AttributeError(
                """Neighbor list of surface atoms (self.nb_surf) not defined.
                Call get_surf_atoms() first.""")
        if not hasattr(self, 'nnb_surf'):
            raise AttributeError(
                """Next nearest neighbor list of surface atoms (self.nbb_surf)
                not defined. Call get_surf_atoms() first.""")

    def collinear_sites(self, sites):
        """Determine whether the two surface atoms, which are second nearest
        neighbors to each other, shares a common neighbor surface atom forming
        a collinear line. This helps to prevent a monomer generated on top
        of a cluster edge.

        Parameters
        ----------
        sites : list
            A list of two integers indicating the anchored surface atoms.

        Returns
        -------
        collinear : boolean
            If True, common neighbor surface atom forms a collinear sites. 
        """
        assert isinstance(sites[0], int), 'sites[0] must be an integer'
        assert isinstance(sites[1], int), 'sites[1] must be an integer'
        assert len(sites) == 2, 'sites must be a list of two integers'

        for x in sites:
            if x not in self.surf:
                print('The invalid number of the sites: %d' % x)
                raise ValueError(
                    "The site is not a valid surface atom.")
        if sites[0] not in self.nnb_surf[sites[1]]:
                raise ValueError(
                    "The sites are not second nearest neighbor to each other.")

        collinear = False 
        # common neighbors
        common = list(set(self.nb_surf[sites[0]]).intersection(
            set(self.nb_surf[sites[1]])))
        if not common:
            # No common neighbor found
            raise ValueError(
                "No common neighbor found for the selected sites.")
        else:
            for x in common:
                v1 = self[sites[0]].position - self[x].position
                v2 = self[sites[1]].position - self[x].position
                angle = angle_vec(v1, v2)
                if angle >= math.pi*175/180:
                    collinear = True
                    break

        return collinear

    def create_monomers(self, monomer, num=1, forbidden=[], tries=50):
        """Generating monomers by randomly select two surface sites which 
        are second nearest neighbor to each other, using add_monomer().

        Notes
        -----
        This will also omit the creation of monomer on sites which are 
        collinear.

        Parameters
        ----------
        monomer : Instance
            See class monomer, which defines the generated monomer.

        num : int, optional (default=1)
            Indicate the total number of ligands required. 

        forbidden : list, optional (default=[])
            List of the surface atom indices which CANNOT
            be used as common neighboring atoms for anchored sites.
            It helps to suggest appropriate surface planes for the generation
            of ligands.

        tries : int, optional (default=50)
            Maxinum number of attempts for each number of ligand. If num=10,
            the maximum attemps = 10*num = 500 (default).
        """
        self.check_surf()

        total = num
        count = 0
        while total > 0 and count <= num*tries:
            print('Tries: {}'.format(count))
            print("Searching for potential sites: %d Ligands remain" % total)
            print('Available sites:')
            print(self.avail)
            print(' ')

            # Randomly pick a surface atom
            site1 = random.choice(self.avail)
            site2 = self.nnb_surf[site1]

            # Randomly select the next nearest neighbors of chosen site
            if list(set(site2).intersection(set(self.avail))):
                site2 = random.choice(
                        list(set(site2).intersection(set(self.avail))))
            else:
                count = count + 1
                # Continue the next iteration if no suitable nnb atom found
                continue

            # Only create monomer if the sites are non collinear
            if self.collinear_sites([site1, site2]):
                count = count + 1
                continue

            if monomer.ref_neighbor:
                # Determine the common neighboring atoms
                common = list(set(self.nb_surf[site1]).intersection(
                    set(self.nb_surf[site2])))
                # Omit the neighboring atoms forbidden by user
                inter = list(set(common).intersection(set(forbidden)))
                for x in inter:
                    common.remove(x)
                success = False 

                for x in common:
                    neighbor = x
                    success = True 
                    anchor = [site1, site2]
                    break
            else:
                # Try to generate ligands using the COM as reference
                neighbor = []
                success = True 
                anchor = [site1, site2]

            # Generate ligands
            if success:
                created = self.add_monomer(
                    monomer, sites=anchor, plane_atom=neighbor)
                if created:
                    total = total - 1
            count = count + 1

        # Warn the user on the failure attempts
        if count > num*tries:
            print("\nFail to generate the total number of ligands"
                " required for the system.")

    def add_monomer(self, monomer, sites, plane_atom=None):
        """Add a monomer (e.g. CH3-S-Au-S-CH3) to the anchored sites.

        Parameters
        ----------
        monomer : Instance
            See class monomer, which defines the generated monomer.

        sites : list
            A list of two integers indicating the anchored surface atoms.
            This defines x-axis vector on the anchored surface.

        plane_atom : int, optional (default=None)
            Specifies the index of the neighboring atom used to determine
            the anchored surface. Together with 'sites', this defines the
            xy plane, and we will generate a vector perpendicular to this 
            plane, that will then define the xz plane. 
            If None, we use the center of mass as a reference to define
            the xz plane. Thereafter, we will generate a vector perpendicular
            to this xz plane, which will define the xy plane.

        Notes
        -----
        Update the self.avail to mark the occupied sites, if the ligand is
        created.
        """
        # Check whether the provided monomer class is valid
        if not hasattr(monomer, 'SR'):
            raise AttributeError(
                "monomer instance does not have 'SR'.")
        if not hasattr(monomer, 'delta'):
            raise AttributeError(
                "monomer instance does not have 'delta'.")
        if not hasattr(monomer, 'std'):
            raise AttributeError(
                "monomer instance does not have 'std'.")
        if not hasattr(monomer, 'symbol'):
            raise AttributeError(
                "monomer instance does not have 'symbol'.")
        if not hasattr(monomer, 'side_group'):
            raise AttributeError(
                "monomer instance does not have 'side_group'.")
        if not hasattr(monomer, 'over_r'):
            raise AttributeError(
                "monomer instance does not have 'over_r'.")

        assert isinstance(sites[0], int), 'sites[0] must be an integer'
        assert isinstance(sites[1], int), 'sites[1] must be an integer'
        assert len(sites) == 2, 'sites must be a list of two integers'

        # Warn the users if the provided sites have been occupied
        if sites[0] not in self.avail or sites[1] not in self.avail:
            message = """The provided sites %s, %s have been occupied!
                Are you sure that do you like to continue?""" % (
                sites[0], sites[1])
            input(message)

        ligand = Atoms()
        # Create the random value (Norm dist.) for SR
        noise_SR = stats.norm(loc=monomer.SR, scale=monomer.std[0])
        # Attach S atom
        pos = tuple(np.array([0, 0, noise_SR.rvs()]))
        ligand.append(Atom(monomer.symbol[0], position=pos))
        # Distance between the two sites
        diff = self[sites[1]].position - self[sites[0]].position
        disp = np.sqrt(np.vdot(diff, diff))
        # Attach second S atom
        pos = tuple(np.array([disp, 0, noise_SR.rvs()]))
        ligand.append(Atom(monomer.symbol[0], position=pos))
        # Attach Au atom, mid point between the S atoms
        pos = tuple(0.5*(ligand[0].position + ligand[1].position))
        ligand.append(Atom(monomer.symbol[1], position=pos))

        # If there is a defined side_group, such as methyl, append it
        if monomer.side_group:
            num = monomer.side_group.get_number_of_atoms()
            # Decide the ligand orientation, disturb the symmetry
            if np.random.rand() >= 0.5:
                orientation_1st = monomer.side_group.positions
            else:
                # Refletion through xz plane
                orientation_1st = reflect(
                    monomer.side_group.positions, symmetry='xz')

            orientation_2nd = rotate_vec(theta=math.pi, vec_axis=(
                0, 0, 1), vec=orientation_1st.T)
            orientation_2nd = orientation_2nd.T
            # Attach side_group to the first S atom
            pos = orientation_1st + ligand[0].position
            ligand.extend(monomer.side_group)
            # Update the side_group's positions
            for (i, x) in enumerate(ligand[-num:]):
                ligand[-num + i].position = pos[i]

            # Attach side_group to the second S atom
            pos = orientation_2nd + ligand[1].position
            ligand.extend(monomer.side_group)
            for (i, x) in enumerate(ligand[-num:]):
                ligand[-num + i].position = pos[i]

            # Create a temporary copy of ligand, used to generate a ligand
            # with the opposite orientaiton if the initial orientation fails
            # This is relevant only if user suggests a side group
            ligand2 = copy.deepcopy(ligand)
            pos = reflect(ligand2.positions, symmetry='xz')
            ligand2.set_positions(pos)

        # Create a random value (Norm dist.) for noise_delta
        noise_delta = stats.norm(loc=monomer.delta, scale=monomer.std[1])
        deg_delta = noise_delta.rvs()
        # Create the sign which guides how the ligands to be displaced side way.
        if np.random.rand() >= 0.5:
            sign = 1
        else:
            sign = -1

        # Rotate the ligand group with respect to xz plane, at noise_delta deg
        ligand.rotate(v=(0, 0, 1), a=sign*deg_delta)
        # Add the ligand
        success, _ = self.add_ligand(ligand, sites=sites, over_r=monomer.over_r,
            plane_atom=plane_atom, debug=monomer.debug)
        if success:
            self.avail.remove(sites[0])
            self.avail.remove(sites[1])

        # Repeat the effort of creating the ligand with the opposite 
        # orientation defined by ligand2
        if ligand2 and success == False:
            # Rotate the whole side groups with respect to xz plane
            ligand2.rotate(v=(0, 0, 1), a=sign*deg_delta)
            # Add the ligand
            success, _ = self.add_ligand(ligand2, sites=sites, 
                    over_r=monomer.over_r, plane_atom=plane_atom, 
                    debug=monomer.debug)
            if success:
                self.avail.remove(sites[0])
                self.avail.remove(sites[1])

        return success

    def add_ligand(self, ligand, sites, over_r=1.5, 
            plane_atom=None, debug=False):
        """Attach a ligand on top of the provided sites.

        Notes
        -----
        This method evaluates the xz plane using plane_atom as a
        reference (see below). Together with the given 'sites', we
        can transform the surface to a co-ordinate system with 
        sites[0] as (0, 0, 0), sites[1] as (r_x, 0, 0) where r_x
        refers to the distance between these two anchored surface atoms.

        Warning: If the provided plane_atom is collinear with the anchored
        sites, the generated ligand will not be properly placed on the surface.

        This function will not generate a ligand, if the newly created atoms
        overlaps with the existing atoms.

        Parameters
        ----------
        ligand : ASE Atoms instance
           Atomic positions of ligands, specified with reference to 
           plane spanned by sites[0] and sites[1] (see below).

        sites : list
            A list of two integers indicating the anchored surface atoms.
            This defines x-axis vector on the anchored surface.

        over_r : float, optional (default=1.5)
            Two atoms are considered overlapped, if the distance between them <=
            this value (unit angstrom). This prevents the created ligand 
            overlapping with the existing atoms. 

        plane_atom : int, optional (default=None)
            Specifies the index of the neighboring atom used to determine
            the anchored surface. Together with 'sites', this defines the
            xy plane, and we will generate a vector perpendicular to this 
            plane, that will then define the xz plane. 
            If None, we use the center of mass as a reference to define
            the xz plane. Thereafter, we will generate a vector perpendicular
            to this xz plane, which will define the xy plane.

        debug : boolean, optional (default=False)
            If True, displays the system together with the suggested ligand.

        Returns
        -------
        success : boolean
            If True, 'ligand' is successfully attached to the system,
            provided the new 'ligand' do not overlap with the exiting atoms.

        new_ligand : ASE Atoms instance 
           The calculated atomic positions of ligands, which might be
           suitable to be attahed to the system.
        """
        assert isinstance(sites[0], int), 'sites[0] must be an integer'
        assert isinstance(sites[1], int), 'sites[1] must be an integer'
        assert len(sites) == 2, 'sites must be a list of two integers'
        # Create Atoms object for the anchored sites
        anchors = Atoms([self[sites[0]], self[sites[1]]])
        if plane_atom:
            # vectors connecting reference to anchored sites
            vr0 = anchors[0].position - self[plane_atom].position
            vr1 = anchors[1].position - self[plane_atom].position
            # vector perpendicular to the surface
            vrp = np.cross(vr0, vr1)
            # check whether if vrp pointing outward of surface or not.
            v = anchors[0].position - self.com
            if angle_vec(v, vrp) > math.pi/2:
                # pointing inward of surface, rotates it
                axisr = vr1 - vr0
                vrp = rotate_vec(theta=math.pi, vec_axis=axisr, vec=vrp)
    
            # ref becomes a point beneath the surface, and the vector
            # anchors[0].position - vrp is perpendicular to the surface.
            ref = anchors[0].position - vrp
        else:
            ref = self.com

        # Create a vector perpendicular to the line joining the
        # both anchored sites (xy axis). Ligands created will be placed on
        # top of the surface.
        # Evaluate the vector from the ref to the first atom.
        v0 = anchors[0].position - ref
        # Evaluate the vector from the ref to the second atom.
        v1 = anchors[1].position - ref
        # Vector perpendicular to the v0 and v1
        v2 = np.cross(v0, v1)
        # Axis to be rotated about
        axis = anchors[1].position - anchors[0].position
        # Finally, rotate the unnormalized vector v2 to become perpendicular
        # to the line joining anchored sites, along the plane containing
        # ref, anchored sites. vp is important to mark the 'xz' plane of anchors
        vp = rotate_vec(theta=math.pi/2, vec_axis=axis, vec=v2)
        # Transform the coordinates by having the first atom postioned
        # at (0, 0, 0)
        displace = copy.deepcopy(anchors[0].position)
        anchors.translate(-1*displace)
        # Rotate the atoms to be aligned along the x-axis
        axis = (1, 0, 0)
        deg_x = angle_vec(anchors[1].position, axis)
        # Obtain the rotation axis by doing cross product
        # This axis is not normalized to unity
        # BE CAREFUL such that anchors[1] may have already be aligned
        # to x-axis and suggest that aixs_x as (0, 0, 0).
        # Give rise to RuntimeWarning, division by zero, when we apply
        # rotate_vec.
        axis_x = np.cross(anchors[1].position, axis)
        align_x = True 
        if np.dot(axis_x, axis_x) < 1e-6:
            align_x = False 
        if align_x:
            # Apply rotation to align anchored sites with x-axis
            anchors.rotate(v=axis_x, a=deg_x)
            # Don't forget about rotation on vp as well
            vp = rotate_vec(theta=deg_x, vec_axis=axis_x, vec=vp)

        # xy plane is now spanned by sites[0], sites[1]
        # Use this vp to determine the angle which we should align the
        # system such that sites will be spanned by xz plane
        axis = (0, 0, 1)
        # Evaluate the rotation angle. Remember, axis is a unit vector
        deg_z = angle_vec(vp, axis)
        if vp[1] > 0:
            # Rotate about +x
            axis_z = (1, 0, 0)
        else:
            # Rotate about -x
            axis_z = (-1, 0, 0)

        anchors.rotate(v=axis_z, a=deg_z)
        # sites[0] and sites[1] are now placed at the
        # x-axis. Rotate the atoms such that sites[1] to be in
        # positive-x region, if neccesary
        rotate_posx = False
        if anchors[1].position[0] < 0.0:
            rotate_posx = True
            anchors.rotate(v=(0, 0, 1), a=math.pi)
        
        # Add the user defined ligand to the anchor group
        anchors.extend(ligand)
        # rotate sites[1] back to negative x region, if needed
        if rotate_posx:
            anchors.rotate(v=(0, 0, 1), a=-1*math.pi)

        # After adding the extra atoms, we rotate and displace them back
        # to the original coordinates (defined with respect to the given system)
        # Rotate back all the coordinates to the original
        anchors.rotate(v=axis_z, a=-1*deg_z)
        # Only rotate back if we have aligned anchors to x-axis
        # This avoids the potential RuntimeWarning
        if align_x:
            anchors.rotate(v=axis_x, a=-1*deg_x)

        # Displace back to the original
        anchors.translate(displace)
        # Check whether the created ligand overlap with the other atoms 
        others = self.positions[:]
        new_ligand = anchors[2:]
        success = True
        for x in others:
            # Break from the loop once we found overlapped atoms
            if success == False:
                break
            for y in new_ligand:
                overlap = x - y.position
                if np.sqrt(np.vdot(overlap, overlap)) <= over_r:
                    success = False 
                    break

        # This is for checking purpose
        if debug:
            print('success = {}'.format(success))
            temp = copy.deepcopy(self)
            temp.extend(new_ligand)
            ase_debug(temp)
            input("View the system with 'designed' ligand")

        # Attach the created ligand to our system if it does not overlap with
        # the other atoms
        if success:
            self.extend(new_ligand)

        return success, new_ligand

def add_interactive(surface_atoms, ligand, fname='system.xyz'):
    """Display the system and prompt users to add the defined
    ligand by choosing the required surface (spanned by three atoms).
    It will generate ase-temp.xyz file indicating the updated system
    processed by the users.

    Parameters
    ----------
    surface_atoms : SurfaceAtom instance
        The atomic information of our system.

    ligand : ASE Atoms instance
        The atomic information of ligand. The positions should be
        defined such that they are placed on top of a surface
        which will be spanned the chosen surface atoms.
        The first two chosen surface atoms are imagined to be 
        positioned at (0, 0, 0), (r_x, 0, 0) where r_x
        refers to the distance between the two atoms.

    fname : str, optional (default='system.xyz')
        The xyz filename of our system.
    """
    print("System: {}\n".format(
        surface_atoms.get_chemical_formula(mode="hill")))
    view(surface_atoms)

    valid = True
    while(valid):
        atom_num = ligand.get_number_of_atoms()
        try:
            message = textwrap.dedent("""\n
                Please choose one of the following actions.
                0 - Quit the program.  
                1 - Add a ligand to the surface of the cluster.
                2 - Delete the ligand from the surface of the cluster.
                3 - View the current system.\n
                Input:""")
    
            action = input(message)
            if action not in ['1', '2', '3', '0']:
                print("Not an invalid input. (1, 2, 3 or 0)")
                continue
    
            if action == '0':
                valid = False
            elif action == '1':
                # Attach a ligand
                try:
                    message = textwrap.dedent("""\n
                        Please provide a list of three integers indicating the
                        sites. sites = [a, b, c] where a and b are the numbers 
                        indicating surface atoms to be anchored on, c is the 
                        number indicating the neighboring atom used to create 
                        the ligand.\n
                        sites =:""")
                    sites = input(message)
                    sites = sites.split()
                    # Convert the three numbers from string to integer
                    sites = [int(x) for x in sites]
                    if len(sites) != 3:
                        print("Three integers needed to define the sites!\n")
                        raise ValueError
    
                    created, _ = surface_atoms.add_ligand(ligand=ligand, 
                        sites=[sites[0], sites[1]], plane_atom=sites[2])
                    print('ligand created or not: {}'.format(created))
                    if created:
                        ase_debug(surface_atoms)
                except Exception as e:
                    print(e)
            elif action == '2':
                try:
                    # Delete the ligands
                    message = textwrap.dedent("""\n
                         Please provide an integer belonging to the generated
                         ligand. This programme will delete the whole ligand.\n
                         Input:""")
                    item = int(input(message))
                    assert item >= surface_atoms.core_num, ("The specified"
                        " index belongs to the original system!")
                    # k, index indicating the group
                    k = item - surface_atoms.core_num
                    k = int(k/atom_num)
                    idx = surface_atoms.core_num + k*atom_num
                    del surface_atoms[idx:(idx + atom_num)]
                    ase_debug(surface_atoms)
                except Exception as e:
                    print(e)
            elif action == '3':
                view(surface_atoms)
        except:
            print(("Exception Error, write the current system to %s\n" 
                % outfname))
            outfname = 'exception-' + fname
            write(outfname, surface_atoms)
            break
    
    if not valid:
        outfname = "result-" + fname
        print(('Option 0 was chosen, write the system to %s\n' % outfname))
        write(outfname, surface_atoms)
