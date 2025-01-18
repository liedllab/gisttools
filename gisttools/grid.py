import numpy as np
from .utils import cartesian_product
import numba
import warnings
from textwrap import dedent
import math


class Grid:
    """
    Defines a Grid class that provides the tools to handle 3-dimensional regular grids,
    convert between linear and 3-dimensional indexing, and find voxels based on geometric
    criteria.
    """

    def __init__(self, origin, shape, delta):
        """Create new Grid instance.

        Parameters
        ----------
        origin : float or tuple of 3 floats
            Lower boundaries in x-, y-, and z-direction. If only one number is given,
            use the same in all directions.
        shape : int or tuple of 3 ints
            Number of voxels in x-, y-, and z-direction. If only one number is given,
            use the same in all directions.
        delta : float or tuple of 3 floats
            Voxel size in x-, y-, and z-direction. If only one number is given,
            use the same in all directions.

        """
        self.origin = self.broadcast_origin(origin)
        self.shape = self.broadcast_shape(shape)
        self.delta = self.broadcast_delta(delta)
        self.voxel_volume = np.prod(self.delta)
        return

    def scale(self, n):
        return self.__class__(self.origin*n, self.shape, self.delta*n)

    def coarse_grain(self, n):
        origin = self.origin + (n-1)/2 * self.delta
        shape = self.shape // n
        delta = self.delta * n
        new = self.__class__(origin, shape, delta)
        if self.size / new.size != n**3:
            raise ValueError("Old and coarse-grained grids don't have matching sizes. Probably the shape is not divisible by n.")
        return new

    @staticmethod
    def broadcast_origin(origin):
        return np.broadcast_to(np.asarray(origin, dtype=np.float64), (3,)).copy()

    @staticmethod
    def broadcast_shape(shape):
        out = np.broadcast_to(np.asarray(shape), (3,)).copy()
        if not issubclass(out.dtype.type, np.integer):
            raise TypeError(f'Grid dimensions must have an integer dtype, not {out.dtype}')
        return out

    @staticmethod
    def broadcast_delta(delta):
        return np.broadcast_to(np.asarray(delta, dtype=np.float64), (3)).copy()

    @classmethod
    def centered(cls, center, shape, delta):
        """Return a Grid instance centered at *center*

        See __init__ for Parameters
        """
        center = cls.broadcast_origin(center)
        shape = cls.broadcast_shape(shape)
        delta = cls.broadcast_delta(delta)
        extent = (shape-1) * delta
        origin = center - extent / 2.
        return cls(origin, shape, delta)

    @property
    def xyz0(self):
        warnings.warn('Grid.xyz0 has been renamed to Grid.origin', DeprecationWarning)
        return self.origin

    @property
    def xyzdim(self):
        warnings.warn('Grid.xyzdim has been renamed to Grid.shape', DeprecationWarning)
        return self.shape

    @property
    def xyzspcn(self):
        warnings.warn('Grid.xyzspcn has been renamed to Grid.delta', DeprecationWarning)
        return self.delta

    @property
    def n_voxels(self):
        warnings.warn('Grid.n_voxels has been renamed to Grid.size', DeprecationWarning)
        return self.size

    @property
    def size(self):
        return np.prod(self.shape)

    def __repr__(self):
        return f"Grid(origin={self.origin}, shape={self.shape}, delta={self.delta})"

    @property
    def xyzmax(self):
        """Return maximum XYZ values

        Examples
        --------
        >>> a = Grid(origin=0., shape=3, delta=1.)
        >>> a.xyzmax
        array([2., 2., 2.])
        """
        return self.origin + self.delta * (self.shape - 1)

    @property
    def edges(self):
        """Open grid of voxel centers.

        Returns
        -------
        A tuple of 3 np.ndarrays
        Each element is an array containing the voxel center in x, y, or z direction.

        Examples
        --------
        >>> a = Grid(origin=[0, 1, 2], shape=3, delta=1.)
        >>> a.edges
        (array([0., 1., 2.]), array([1., 2., 3.]), array([2., 3., 4.]))
        """
        return tuple(
            np.arange(ax_shape) * ax_delta + ax_origin
            for ax_shape, ax_delta, ax_origin in zip(self.shape, self.delta, self.origin)
        )

    def dxheader(self):
        """Return simple OpenDX header for a single data row."""
        origin = self.origin
        return dedent(f'''\
            object 1 class gridpositions counts {self.shape[0]} {self.shape[1]} {self.shape[2]}
            origin {origin[0]} {origin[1]} {origin[2]}
            delta {self.delta[0]} 0 0
            delta 0 {self.delta[1]} 0
            delta 0 0 {self.delta[2]}
            object 2 class gridconnections counts {self.shape[0]} {self.shape[1]} {self.shape[2]}
            object 3 class array type double rank 0 items {self.size} data follows''')

    @staticmethod
    def dxfooter(name='Unknown'):
        return dedent(f'''\

        object "{name}" class field''')

    def flat_indices(self, xyz_indices):
        """Calculate 1-dimensional voxel indices from an INTEGER array with
        shape (n, 3).

        Arguments
        ---------
        xyz_indices : np.ndarray, shape=(n, 3), dtype=int
            Grid indices in x, y, and z direction for which 1-dimensional
            indices should be calculated.

        Returns
        -------
        indices : np.ndarray, shape=(n), dtype=int

        Examples
        --------
        >>> a = Grid(origin=-100., shape=3, delta=1.)
        >>> a.flat_indices([[0, 1, 1]])
        array([4])
        >>> test = np.arange(27)
        >>> np.all(a.flat_indices(a.xyz_indices(test)) == test)
        True
        """
        xyz_indices = np.asarray(xyz_indices)
        assert np.issubdtype(
            xyz_indices.dtype, np.integer
        ), "Wrong datatype in argument. Maybe use flat_indices(closest(...))?"
        assert len(xyz_indices.shape) == 2 and xyz_indices.shape[1] == 3, \
            f"xyz_indices must have shape (n, 3), not {xyz_indices.shape}"
        xyz_indices = np.asarray(xyz_indices).reshape(-1, 3)
        return np.ravel_multi_index(xyz_indices.T, self.shape)

    def xyz_indices(self, indices):
        """Return indices in xyz directions from a linear index.

        Parameters
        ----------
        indices : np.ndarray, shape=(n_indices,)

        Returns
        -------
        np.ndarray, shape=(n_indices, 3)
            xyz indices for each input index.

        Examples
        --------
        >>> a = Grid(origin=-100., shape=3, delta=1.)
        >>> a.xyz_indices([0, 4, 8, 12])
        array([[0, 0, 0],
               [0, 1, 1],
               [0, 2, 2],
               [1, 1, 0]])
        >>> a.xyz_indices(3.)
        array([[0, 1, 0]])
        >>> try:
        ...     a.xyz_indices(27)
        ... except ValueError:
        ...     print("An error happened")
        An error happened
        """
        indices = np.asarray(indices, dtype=int)
        assert (
            len(indices.shape) < 2
        ), "Only 0- or 1-dimensional data can be used as indices."
        return np.array(np.unravel_index(indices, self.shape)).T.reshape(-1, 3)

    def xyz(self, indices):
        """Return xyz coordinates for indices.

        Parameters
        ----------
        indices : np.ndarray
            If indices is 2-dimensional, each row should be a set of xyz indices. If
            indices is 0- or 1-dimensional, calls xyz_indices first.

        Returns
        -------
        xyz : np.ndarray(n_indices, 3)
            xyz coordinates of each input index.

        Examples
        --------
        >>> a = Grid(origin=-100., shape=3, delta=1.)
        >>> a.xyz([0, 4, 8, 12])
        array([[-100., -100., -100.],
               [-100.,  -99.,  -99.],
               [-100.,  -98.,  -98.],
               [ -99.,  -99., -100.]])
        >>> a.xyz(3.)
        array([[-100.,  -99., -100.]])
        >>> a.xyz([[0, 2, 0]])
        array([[-100.,  -98., -100.]])
        >>> a.xyz(27)
        Traceback (most recent call last):
        ValueError: ...
        """
        indices = np.asarray(indices, dtype=int)
        if len(indices.shape) < 2:
            indices = self.xyz_indices(indices)
        assert indices.shape[1] == 3, "Shape of indices must be (n_indices, 3)"
        return indices * self.delta + self.origin

    def closest(self, xyz, out_of_bounds='raise'):
        """Get unraveled xyz indices of the closest voxel.

        Parameters
        ---------
        xyz : array_like, shape=(n_indices, 3)
            The coordinates to calculate voxel indices from.
        out_of_bounds : 'raise', 'closest', 'ignore' or 'dummy'
            How to handle cases where xyz values are out of the Grid boundary. 'raise' will
            raise RuntimeError, 'closest' will still find the closest voxel (like the deprecated
            always_return option), 'ignore' will return invalid voxel indices, and 'dummy' will
            replace the respective xyz indices with -1 in all 3 columns (in lack of an
            integer version of np.nan)

        Returns
        -------
        indices : np.array of shape (n_indices, 3), dtype=int
            The indices of the voxel to which the coordinates xyz belong.

        Examples
        --------
        >>> a = Grid(origin=-1., shape=3, delta=1.)
        >>> a.closest([[.2, 0, 0],
        ...            [-1.4, 1, 1.4]])
        array([[1, 1, 1],
               [0, 2, 2]])
        >>> a.closest([2., 2., 2.])
        Traceback (most recent call last):
        ValueError: Out of bounds index in Grid.closest
        >>> a.closest([2., 2., 2.], out_of_bounds='closest')
        array([[2, 2, 2]])
        """
        xyz = np.asarray(xyz).reshape(-1, 3)
        normalized = (xyz - self.origin) / self.delta
        indices = np.round(normalized).astype(int)
        if out_of_bounds == 'ignore':
            return indices
        elif out_of_bounds == 'closest':
            return np.clip(indices, 0, self.shape-1)
        elif out_of_bounds == 'raise':
            if np.any(indices < 0) or np.any(indices >= self.shape):
                raise ValueError('Out of bounds index in Grid.closest')
            return indices
        elif out_of_bounds == 'dummy':
            rows = np.any(np.logical_or(indices < 0, indices >= self.shape), axis=1)
            indices[rows] = -1
            return indices
        else:
            raise ValueError(f'Unknown value for out_of_bounds: {out_of_bounds}')

    def _assign(self, xyz):
        """Get raveled (flat) indices of each row of xyz

        Returns -1 for coordinates outside of the grid.

        Parameters
        ----------
        xyz: np.ndarray, shape=(n_atoms, 3)

        Returns
        -------
        indices: np.ndarray, shape=(n_atoms,), dtype=int
        """
        xyz = np.atleast_2d(xyz)
        assert len(xyz.shape) == 2 and xyz.shape[1] == 3, \
            "Can only assign indices to (n x 3) arrays, but shape is " + str(xyz.shape)
        vox_3d = self.closest(xyz, out_of_bounds='dummy')
        out = np.ravel_multi_index(vox_3d.T, self.shape, mode='wrap')
        invalid = np.nonzero(vox_3d == -1)[0]
        out[invalid] = -1
        return out

    def assign(self, xyz):
        xyz = np.float64(np.atleast_2d(xyz))
        origin = np.float64(self.origin)
        shape = np.int64(self.shape)
        delta = np.float64(self.delta)
        if xyz.shape[-1] != 3:
            raise ValueError("Can only assign grid indices to 3D data.")
        return assign_to_grid(xyz, origin, shape, delta)

    def surrounding_box(self, center, radius):
        """Find voxels that form a box around 'center', that contains all
        voxels that are within 'radius' from the 'center'.

        Parameters
        ----------
        center : array-like, shape=(3,)
            xyz coordinates of the box center.
        radius : float
            maximum distance to the box center.

        Returns
        -------
        indices : list of 3 arrays
            voxel indices in x, y, and z direction that define a box around 'center'

        Examples
        --------
        >>> a = Grid(origin=-1., shape=3, delta=1.)
        >>> a.surrounding_box([.5, .5, .5], .99) # rounding problems when using 1
        [array([1, 2]), array([1, 2]), array([1, 2])]
        >>> a.surrounding_box([1.5, 1.5, 1.5], 1) # center outside of the box
        [array([2]), array([2]), array([2])]
        """
        center = np.asarray(center)
        assert center.shape == (3,), "Shape of center must be (3,)"
        xyzmin = center - radius
        xyzmax = center + radius

        # Both min_indices and max_indices are arrays of shape (3,)
        min_indices, max_indices = self.closest((xyzmin, xyzmax), out_of_bounds='closest')
        return [np.arange(imin, imax + 1) for imin, imax in zip(min_indices, max_indices)]

    def surrounding_sphere(self, center, radius):
        """Find voxels that lie within a sphere of 'radius' around the 'center'.

        Parameters
        ----------
        center : array-like, shape=(3,)
            x, y and z coordinates that define the center of the sphere.
        radius : float
            radius of the sphere.

        Returns
        -------
        indices : np.ndarray, shape=(n,)
            indices of voxels that are in the sphere defined by 'center' and
            'radius'.
        distance : np.narray, shape=(n,)
            distances of the voxels to the center.

        Examples
        --------
        >>> a = Grid(origin=-1., shape=3, delta=1.)
        >>> ind, dist = a.surrounding_sphere([.5, .5, .5], .99) # rounding problems when using 1
        >>> print(ind)
        [13 14 16 17 22 23 25 26]
        >>> np.allclose(dist, .5 * np.sqrt(3))
        True
        >>> print(a.surrounding_sphere([.5, .5, .5], .7)[0])
        []
        >>> ind, dist = a.surrounding_sphere([1.5, 1.5, 1.5], 1) # center outside of the box
        >>> print(ind)
        [26]
        >>> np.allclose(dist, .5 * np.sqrt(3))
        True
        """
        ind, sqrdist = self._surrounding_sphere(center, radius)
        return ind, np.sqrt(sqrdist)

    def _surrounding_sphere(self, center, radius):
        """Find voxels that lie within a sphere of 'radius' around the 'center'.

        Parameters
        ----------
        center : array-like, shape=(3,)
            x, y and z coordinates that define the center of the sphere.
        radius : float
            radius of the sphere.

        Returns
        -------
        indices : np.ndarray, shape=(n,)
            indices of voxels that are in the sphere defined by 'center' and
            'radius'.
        sqrdist : np.narray, shape=(n,)
            squared distances of the voxels to the center.

        Notes
        -----
        This is a helper function that returns the squared distance.
        """
        center = np.asarray(center, dtype=np.float64).reshape(3)
        flat_indices, sqrdist = _surrounding_sphere(
            origin=self.origin,
            shape=self.shape,
            delta=self.delta,
            center=center,
            radius=float(radius),
        )
        return flat_indices, sqrdist

    def surrounding_sphere_np(self, center, radius):
        """Pure numpy implementation of surrounding_sphere"""
        center = np.asarray(center).reshape(3)
        box_indices = cartesian_product(*self.surrounding_box(center, radius))
        relative_coords = box_indices * self.delta + (self.origin - center)
        sqrdist = np.sum(relative_coords**2, 1)
        within_sphere = sqrdist <= radius ** 2
        return self.flat_indices(box_indices[within_sphere]), np.sqrt(sqrdist[within_sphere])

    def distance_to_centers(self, centers, rmax):
        """Find voxels that lie within 'rmax' of any row in 'centers'.

        Parameters
        ----------
        centers : array-like, shape=(m, 3)
            x, y, and z coordinates of the m requested atom positions.
        rmax : float
            maximum distance of the returned voxels to the respective center.

        Returns
        -------
        indices : np.ndarray, shape=(n,)
            indices of all voxels that are within 'rmax' of any point in
            'centers'.
        closest_center : np.ndarray, shape=(n,)
            Contains the atom number (index of 'centers') which is closest to
            each voxel in 'indices'.
        distances : np.ndarray, shape=(n,)
            For all voxels within 'rmax': the distance to the nearest atom in
            'centers'.

        Examples
        --------
        >>> a = Grid(origin=-1., shape=3, delta=1.)
        >>> testdist = .5 * np.sqrt(3)
        >>> ind1, dist = a.surrounding_sphere([.5, .5, .5], .9)
        >>> ind2, dist2 = a.surrounding_sphere([-.5, -.5, -.5], .9)
        >>> ind_full, centers, dists = a.distance_to_centers([[.5, .5, .5],
        ...                                                   [-.5, -.5, -.5]],
        ...                                                  .9)
        >>> ind1
        array([13, 14, 16, 17, 22, 23, 25, 26])
        >>> ind2
        array([ 0,  1,  3,  4,  9, 10, 12, 13])
        >>> np.all(ind_full == np.unique((ind1, ind2)))
        True
        >>> np.all(dists == testdist)
        True
        """
        centers = np.asarray(centers).reshape(-1, 3)
        current_smallest = np.full(self.size, np.inf)
        current_closest_center = np.full(self.size, -1)
        for i, center in enumerate(centers):
            ind, sqrdist = self._surrounding_sphere(center, rmax)
            smaller = sqrdist < current_smallest[ind]
            smaller_indices = ind[smaller]
            current_smallest[smaller_indices] = sqrdist[smaller]
            current_closest_center[smaller_indices] = i
        indices = np.flatnonzero(current_smallest != np.inf)
        distances = np.sqrt(current_smallest[indices])
        closest_center = current_closest_center[indices]
        assert np.all(
            closest_center != -1
        ), "No distance was calculated for at least one relevant voxel. This is likely a bug."
        return indices, closest_center, distances

    def surrounding_sphere_mult(self, *args, **kwargs):
        warnings.warn(
            'surrounding_sphere_mult has been renamed to distance_to_centers.',
            DeprecationWarning,
        )
        return self.distance_to_centers(*args, **kwargs)

    def distance_to_spheres(self, centers, rmax, radii):
        """Find voxels within 'rmax' of any sphere defined by 'centers' and 'radii'.

        Note that the returned distances can be negative for voxels that lie within one
        of the spheres. Similar to distance_to_centers, only the smallest distance is
        reported.

        Parameters
        ----------
        centers : array-like, shape=(m, 3)
            x, y, and z coordinates of the m requested atom positions.
        rmax : float
            maximum distance of the returned voxels to the respective center.
        radii : array-like
            The atomic radius will be subtracted from every computed distance, yielding
            the distance to the molecular surface instead of the distance to the
            closest atomic center.

        Returns
        -------
        indices : np.ndarray, shape=(n,)
            indices of all voxels that are within 'rmax' of any point in
            'centers'.
        closest_center : np.ndarray, shape=(n,)
            Contains the atom number (index of 'centers') which is closest to
            each voxel in 'indices'.
        distances : np.ndarray, shape=(n,)
            For all voxels within 'rmax': the distance to the nearest atom in
            'centers'.

        Examples
        --------
        >>> a = Grid(origin=-1., shape=3, delta=1.)
        >>> testdist = .5 * np.sqrt(3)
        >>> ind1, dist = a.surrounding_sphere([.5, .5, .5], .9)
        >>> ind2, dist2 = a.surrounding_sphere([-.5, -.5, -.5], .9)

        If only one atomic radius is given, the function is equivalent to (but slower
        than) distance_to_centers with a changed radius.
        >>> ind_full, centers, dists = a.distance_to_spheres([[.5, .5, .5],
        ...                                                   [-.5, -.5, -.5]],
        ...                                                   .4, 0.5)
        >>> ind1
        array([13, 14, 16, 17, 22, 23, 25, 26])
        >>> ind2
        array([ 0,  1,  3,  4,  9, 10, 12, 13])
        >>> np.all(ind_full == np.unique((ind1, ind2)))
        True
        >>> np.all(dists == testdist - 0.5)
        True
        """
        # In contrast to distance_to_centers, this function has to operate on the
        # (non-squared) distances.
        centers = np.asarray(centers).reshape(-1, 3)
        radii = np.broadcast_to(radii, centers.shape[0])
        current_smallest = np.full(self.size, np.inf, dtype=np.float64)
        current_closest_center = np.full(self.size, -1, dtype=int)
        for i, (center, radius) in enumerate(zip(centers, radii)):
            ind, sqrdist = self._surrounding_sphere(center, rmax + radius)
            sphere_dist = np.sqrt(sqrdist) - radius
            smaller = sphere_dist < current_smallest[ind]
            smaller_indices = ind[smaller]
            current_smallest[smaller_indices] = sphere_dist[smaller]
            current_closest_center[smaller_indices] = i
        indices = np.flatnonzero(current_smallest != np.inf)
        distances = current_smallest[indices]
        closest_center = current_closest_center[indices]
        assert np.all(
            closest_center != -1
        ), "No distance was calculated for at least one relevant voxel. This is likely a bug."
        return indices, closest_center, distances

    @classmethod
    def from_xyz(cls, xyz):
        """Detect grid parameters from an xyz coordinate array, then construct a new
        Grid. Assumes that shape is at least 2 in each dimension.

        Parameters
        ----------
        xyz : np.ndarray, shape=(n, 3)
            Sorted voxel coordinates (see example).

        Returns
        -------
        gridparm: Dictionary
            Dictionary containing the parameters of the GIST grid.
            Keys: 'xspacing', 'yspacing', 'zspacing', 'xdim', 'ydim', 'zdim'
            (all float)

        Examples
        --------
        >>> arr = np.array([[2., 3., 4.],
        ...                 [2., 3., 5.],
        ...                 [2., 4., 4.],
        ...                 [2., 4., 5.],
        ...                 [3., 3., 4.],
        ...                 [3., 3., 5.],
        ...                 [3., 4., 4.],
        ...                 [3., 4., 5.]])
        >>> Grid.from_xyz(arr)
        Grid(origin=[2. 3. 4.], shape=[2 2 2], delta=[1. 1. 1.])
        """
        xyz = np.asarray(xyz)
        assert len(xyz.shape) == 2 and xyz.shape[1] == 3, "xyz must have shape (n, 3)."

        z = xyz[:, 2]
        zspacing = z[1] - z[0]
        zdim = _axis_len(z)

        y = xyz[::zdim, 1]
        yspacing = y[1] - y[0]
        ydim = _axis_len(y)

        x = xyz[::ydim*zdim, 0]
        xspacing = x[1] - x[0]
        xdim = _axis_len(x)

        origin = xyz[0]
        shape = np.array([xdim, ydim, zdim])
        delta = np.array([xspacing, yspacing, zspacing])

        assert np.prod(shape) == xyz.shape[0], \
            f'shape {shape} is inconsistent with voxel count {xyz.shape[0]}.'
        return cls(origin, shape, delta)

    def save_dx(self, data, filename, colname="DATA"):
        """Save a single GIST column to an OpenDX file.

        Parameters
        ----------
        data: np.ndarray
            Data to save
        filename : str or file handle
            The output file

        Returns
        -------
        None
        """
        data = np.asarray(data).ravel()
        assert data.size == self.size, f"Data size ({data.size}) must fit Grid size ({self.size})."
        np.savetxt(
            filename,
            data,
            header=self.dxheader(),
            footer=self.dxfooter(colname),
            comments='',
            fmt='%f'
        )
        return



def grid_from_xyz(xyz):
    """Detect grid parameters from an xyz coordinate array, then construct a new
    Grid. Assumes that shape is at least 2 in each dimension.

    Parameters
    ----------
    xyz : np.ndarray, shape=(n, 3)
        Sorted voxel coordinates (see example).

    Returns
    -------
    gridparm: Dictionary
        Dictionary containing the parameters of the GIST grid.
        Keys: 'xspacing', 'yspacing', 'zspacing', 'xdim', 'ydim', 'zdim'
        (all float)

    Examples
    --------
    >>> arr = np.array([[2., 3., 4.],
    ...                 [2., 3., 5.],
    ...                 [2., 4., 4.],
    ...                 [2., 4., 5.],
    ...                 [3., 3., 4.],
    ...                 [3., 3., 5.],
    ...                 [3., 4., 4.],
    ...                 [3., 4., 5.]])
    >>> grid_from_xyz(arr)
    Grid(origin=[2. 3. 4.], shape=[2 2 2], delta=[1. 1. 1.])
    """
    return Grid.from_xyz(xyz)


def _axis_len(a):
    """Given voxel coordinates along one coordinate, return the length of this
    coordinate.

    Notes
    -----
    This just looks for the first time the values repeat themselves.

    Examples
    --------
    >>> _axis_len([0, 1, 2, 0, 1, 2])
    3
    >>> _axis_len([0, 0, 1, 1, 2, 2])  # Wrong usage leads to wrong result
    1
    >>> _axis_len([0.1, 0.2, 0.3, 0.1])  # Floats work when values are exactly equal.
    3
    >>> _axis_len([1, 2])
    2
    """
    # I found no numpy solution that avoids looping over the whole array.
    # For realistically spaced grids, the python loop is quite a lot faster. I tested
    # this on np.tile(np.arange(100), 10000), which would be a typical z axis of a Gist
    # grid.
    start = a[0]
    for i, val in enumerate(a[1:]):
        if val == start:
            break
    else:
        return len(a)
    return i + 1


def combine_grids(grids):
    """
    Combine multiple Grid instances into a single one.

    The resulting grid will be big enough to contain all voxel coordinates of the input
    grids.

    Parameters
    ----------
    grids : Iterable of Grid instances

    Examples
    --------

    >>> a = Grid(0., 2, .5)
    >>> b = Grid(2., 2, .5)
    >>> c = Grid(.3, 2, .5)
    >>> combine_grids((a, b))
    Grid(origin=[0. 0. 0.], shape=[6 6 6], delta=[0.5 0.5 0.5])
    >>> combine_grids((a, c))
    Traceback (most recent call last):
    AssertionError: Offset of origin values must be a multiple of the grid spacing.
    """
    xyzmin = np.min([grid.origin for grid in grids], axis=0)
    xyzmax = np.max([grid.xyzmax for grid in grids], axis=0)
    delta = grids[0].delta
    for grid in grids:
        assert np.allclose(
            grid.delta, delta
        ), "All input grids must have the same spacing."
        assert np.allclose(
            (grid.origin - xyzmin) % delta, [0, 0, 0]
        ), "Offset of origin values must be a multiple of the grid spacing."
    shape = ((xyzmax - xyzmin) / delta + 1).astype(int)
    return Grid(origin=xyzmin, shape=shape, delta=delta)


@numba.guvectorize(
    ['f8[:], f8[:], i8[:], f8[:], i8[:]', 'f4[:], f8[:], i8[:], f8[:], i8[:]'],
    '(three), (three), (three), (three) -> ()',
    nopython=True
)
def assign_to_grid(xyz, origin, shape, delta, out):
    assert len(xyz) == 3, "Only 3D data is supported for assign_to_grid"
    or_x, or_y, or_z = origin
    sh_x, sh_y, sh_z = shape
    d_x, d_y, d_z = delta
    x, y, z = xyz
    grid_x = (x - or_x) / d_x + 0.5
    ix = int(math.floor(grid_x))
    if ix < 0 or ix >= sh_x:
        out[0] = -1
        return
    grid_y = (y - or_y) / d_y + 0.5
    iy = int(math.floor(grid_y))
    if iy < 0 or iy >= sh_y:
        out[0] = -1
        return
    grid_z = (z - or_z) / d_z + 0.5
    iz = int(math.floor(grid_z))
    if iz < 0 or iz >= sh_z:
        out[0] = -1
        return
    out[0] = iz + shape[2] * (iy + shape[1] * ix)


@numba.vectorize(nopython=True, cache=True)
def _minimum(a, b):
    """Returns the smaller value element-wise.
    
    Replacement for np.minimum, which is not available in numba."""
    return min(a, b)

    
@numba.vectorize(nopython=True, cache=True)
def _maximum(a, b):
    "Replacement for np.maximum. Returns the larger value element-wise."
    return max(a, b)


@numba.vectorize(nopython=True, cache=True)
def _round(a):
    """Rounds to the nearest integer. In contrast to np.round, this returns an
    integer array! Numba's builtin np.round only supports the 3-argument form
    (np.round_(a, decimals, out))."""
    return round(a)


@numba.vectorize(nopython=True, cache=True)
def ensure_int(a):
    """Convert to int, and ensure that the original was an integer value or a
    floating point representation of an integer.
    """
    out = np.int_(a)
    if out != a:
        raise ValueError("Non-int value!")
    return out


@numba.njit(cache=True)
def _surrounding_box_edges(shape, delta, origin, center, radius):
    """Return a box with wall distance of radius to the center. shape, delta,
    and origin define the grid.

    Returns
    -------
    x, y, z: np.ndarray, 1D
    """
    shape = ensure_int(shape)
    xyz_min = _round((center - radius - origin) / delta)
    xyz_min = _maximum(xyz_min.astype(np.int_), np.array([0, 0, 0]))
    xyz_max = _round((center + radius - origin) / delta)
    xyz_max = _minimum(xyz_max.astype(np.int_), shape - 1)
    x_ind = np.arange(xyz_min[0], xyz_max[0] + 1, dtype=np.int_)
    y_ind = np.arange(xyz_min[1], xyz_max[1] + 1, dtype=np.int_)
    z_ind = np.arange(xyz_min[2], xyz_max[2] + 1, dtype=np.int_)
    return x_ind, y_ind, z_ind


@numba.njit(cache=True)
def _surrounding_sphere(
    origin,
    shape,
    delta,
    center,
    radius
):
    """
    For all voxels within the sphere defined by center and radius, return grid
    indices and distances to the sphere center.

    Returns
    -------
    indices : np.ndarray, shape=(n_voxels, ), dtype=int
    distances : np.ndarray, shape=(n_voxels, ), dtype=float
    """
    # surrounding_box_edges tests that shape is int
    x_indices, y_indices, z_indices = _surrounding_box_edges(shape, delta, origin, center, radius)
    squared_x_dist = (x_indices * delta[0] + (origin[0] - center[0]))**2
    squared_y_dist = (y_indices * delta[1] + (origin[1] - center[1]))**2
    squared_z_dist = (z_indices * delta[2] + (origin[2] - center[2]))**2
    squared_radius = radius ** 2
    max_output_size = len(squared_x_dist) * len(squared_y_dist) * len(squared_z_dist)
    indices = np.zeros((max_output_size), dtype=np.int_)
    squared_distances = np.zeros((max_output_size), dtype=np.float64)
    output_pos = 0
    for i, dx_2 in enumerate(squared_x_dist):
        for j, dy_2 in enumerate(squared_y_dist):
            # Precompute some stuff before the innermost loop
            # This helps a little bit, but not too much.
            dx_dy_2 = dx_2 + dy_2
            max_dz_2 = squared_radius - dx_dy_2
            xyind = x_indices[i] * shape[1] * shape[2] + y_indices[j] * shape[2]
            for k, dz_2 in enumerate(squared_z_dist):
                if dz_2 <= max_dz_2:
                    indices[output_pos] = xyind + z_indices[k]
                    squared_distances[output_pos] = dx_dy_2 + dz_2
                    output_pos += 1
    return indices[:output_pos], squared_distances[:output_pos]
