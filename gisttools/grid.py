import numpy as np
# from .utils import cartesian_product
# from numba import jitclass, int64, float64
import numba
import warnings
from numba import njit, jitclass
from textwrap import dedent


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
        self.origin = np.broadcast_to(np.asfarray(origin), (3,)).copy()
        self.shape = np.broadcast_to(np.asarray(shape), (3,)).copy()
        assert issubclass(self.shape.dtype.type, np.integer), f'Grid dimensions must have an integer dtype, not {self.shape.dtype}'
        self.delta = np.broadcast_to(np.asfarray(delta), (3)).copy()
        self.voxel_volume = np.prod(self.delta)
        self.n_voxels = np.prod(self.shape)
        return

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
        return dedent(f'''\
            object 1 class gridpositions counts {self.shape[0]} {self.shape[1]} {self.shape[2]}
            origin {self.origin[0]} {self.origin[1]} {self.origin[2]}
            delta {self.delta[0]} 0 0
            delta 0 {self.delta[1]} 0
            delta 0 0 {self.delta[2]}
            object 2 class gridconnections counts {self.shape[0]} {self.shape[1]} {self.shape[2]}
            object 3 class array type double rank 0 items {self.n_voxels} data follows''')

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

#         indices = (
#             xyz_indices[:, 2]
#             + xyz_indices[:, 1] * self.shape[2]
#             + xyz_indices[:, 0] * (self.shape[1] * self.shape[2])
#         )
#         return indices

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
        >>> try:
        ...     a.xyz(27)
        ... except ValueError:
        ...     print("An error happened")
        An error happened
        """
        indices = np.asarray(indices, dtype=int)
        if len(indices.shape) < 2:
            indices = self.xyz_indices(indices)
        assert indices.shape[1] == 3, "Shape of indices must be (n_indices, 3)"
        return indices * self.delta + self.origin

    def closest(self, xyz, always_return=False, out_of_bounds='raise'):
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
        always_return : bool
            if always_return is True, always find the closest voxel even if xyz is out
            of the grid. Otherwise, raise a RuntimeError. Deprecated and will be removed.

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
        >>> try:
        ...     a.closest([2., 2., 2.])
        ... except RuntimeError:
        ...     print("An error happened")
        An error happened
        >>> try:
        ...     a.closest([2., 2., 2.], out_of_bounds='closest')
        ... except RuntimeError:
        ...     print("An error happened")
        array([[2, 2, 2]])
        """
        xyz = np.asarray(xyz).reshape(-1, 3)
        normalized = (xyz - self.origin) / self.delta
        indices = np.round(normalized).astype(np.int64)
        if always_return:
            warnings.warn(
                'always_return has been deprecated and will be removed. Use out_of_bounds="closest" instead',
                DeprecationWarning
            )
            indices = np.minimum(np.maximum(indices, 0), self.shape - 1)
            return indices
        if out_of_bounds == 'ignore':
            return indices
        too_small = indices < 0
        too_big = indices >= self.shape
        if np.any(too_small) or np.any(too_big):
            if out_of_bounds == 'closest':
                return np.minimum(np.maximum(indices, 0), self.shape - 1)
            elif out_of_bounds == 'raise':
                rows = np.nonzero(np.logical_or(too_small, too_big))[0]
                raise RuntimeError(f"indices[{rows[0]}] is out of bounds.")
            elif out_of_bounds == 'dummy':
                rows = np.nonzero(np.logical_or(too_small, too_big))[0][:, np.newaxis]
                indices[rows] = -1
                return indices
        return indices

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
        >>> testdist = .5 * np.sqrt(3)
        >>> ind, dist = a.surrounding_sphere([.5, .5, .5], .99) # rounding problems when using 1
        >>> print(ind)
        [13 14 16 17 22 23 25 26]
        >>> np.allclose(dist, testdist)
        True
        >>> print(a.surrounding_sphere([.5, .5, .5], .7)[0])
        []
        >>> ind, dist = a.surrounding_sphere([1.5, 1.5, 1.5], 1) # center outside of the box
        >>> print(ind)
        [26]
        >>> np.allclose(dist, testdist)
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
        center = np.asfarray(center).reshape(3)
        flat_indices, sqrdist = _surrounding_sphere_helper(
            self.shape,
            self.delta,
            self.origin,
            center,
            float(radius),
        )
        return flat_indices, sqrdist
        # Alternative numpy implementation:
        # ---------------------------------
        # box_indices = cartesian_product(*self.surrounding_box(center, radius))
        # relative_coords = box_indices * self.delta + (self.origin - center)
        # sqrdist = np.sum(relative_coords**2, 1)
        # sub_indices = sqrdist < radius ** 2
        # return self.flat_indices(box_indices[sub_indices]), sqrdist[sub_indices]

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
        current_smallest = np.full(self.n_voxels, np.inf)
        current_closest_center = np.full(self.n_voxels, -1)
        for i, center in enumerate(centers):
            ind, sqrdist = self._surrounding_sphere(center, rmax)
            smaller = sqrdist < current_smallest[ind]
            smaller_indices = ind[smaller]
            current_smallest[smaller_indices] = sqrdist[smaller]
            current_closest_center[smaller_indices] = i
        indices = np.flatnonzero(current_smallest != np.inf)
        distances = np.sqrt(current_smallest[indices])
        closest_center = current_closest_center[indices]
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
        current_smallest = np.full(self.n_voxels, np.inf)
        current_closest_center = np.full(self.n_voxels, -1)
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
    # xyzmax is different from self.xyzmax by one.
    # This is correct, because otherwise we would have to add one when calculating the
    # new shape.
    xyzmax = np.max([grid.origin + grid.delta * grid.shape for grid in grids], axis=0)
    delta = grids[0].delta
    for grid in grids:
        assert np.allclose(
            grid.delta, delta
        ), "All input grids must have the same spacing."
        assert np.allclose(
            (grid.origin - xyzmin) % delta, [0, 0, 0]
        ), "Offset of origin values must be a multiple of the grid spacing."
    shape = ((xyzmax - xyzmin) / delta).astype(np.int64)
    return Grid(origin=xyzmin, shape=shape, delta=delta)


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
    integer array! This is needed because numba's np.round only supports the 
    3-argument form."""
    return round(a)


@njit((
    numba.int64[:],  # shape
    numba.float64[:],  # delta
    numba.float64[:],  # origin
    numba.float64[:],  # center
    numba.float64  # radius
), cache=True)
def _surrounding_box(shape, delta, origin, center, radius):
    """Return a box with wall distance of radius to the center. shape, delta,
    and origin define the grid.

    Returns
    -------
    x, y, z: np.ndarray, 1D
    """
    xyzmin = _round((center - radius - origin) / delta)
    xyzmin = _maximum(xyzmin.astype(np.int64), np.array([0, 0, 0]))
    xyzmax = _round((center + radius - origin) / delta)
    xyzmax = _minimum(xyzmax.astype(np.int64), shape - 1)
    x_ind = np.arange(xyzmin[0], xyzmax[0] + 1)
    y_ind = np.arange(xyzmin[1], xyzmax[1] + 1)
    z_ind = np.arange(xyzmin[2], xyzmax[2] + 1)
    return x_ind, y_ind, z_ind


@njit((
    numba.int64[:],  # shape
    numba.float64[:],  # delta
    numba.float64[:],  # origin
    numba.float64[:],  # center
    numba.float64  # radius
), cache=True)
def _surrounding_sphere_helper(
    shape,
    delta,
    origin,
    center,
    radius
):
    x_index, y_index, z_index = _surrounding_box(shape, delta, origin, center, radius)
    x_sqr = (x_index * delta[0] + (origin[0] - center[0]))**2
    y_sqr = (y_index * delta[1] + (origin[1] - center[1]))**2
    z_sqr = (z_index * delta[2] + (origin[2] - center[2]))**2
    sqrrad = radius ** 2
    # xlen = shape[0]
    ylen = shape[1]
    zlen = shape[2]
    yzlen = ylen * zlen
    max_n_indices = len(x_sqr) * len(y_sqr) * len(z_sqr)
    indices = np.zeros((max_n_indices), dtype=np.int64)
    distances = np.zeros((max_n_indices), dtype=np.float64)
    n_indices = 0
    for i, xs in enumerate(x_sqr):
        for j, ys in enumerate(y_sqr):
            # Precompute some stuff before the innermost loop
            # This helps a little bit, but not too much.
            xys = xs + ys
            zmax = sqrrad - xys
            xyind = x_index[i] * yzlen + y_index[j] * zlen
            for k, zs in enumerate(z_sqr):
                if zs < zmax:  # xs + ys + zs > sqrrad
                    indices[n_indices] = xyind + z_index[k]
                    distances[n_indices] = xys + zs
                    n_indices += 1
    return indices[:n_indices], distances[:n_indices]

# Currently unused code follows.
# 
# This will at some point be used for the distance_to_spheres method, since it
# is 2-3 times faster. For normal GIST post-processing, this does not matter,
# but when running this code over a trajectory, the speedup becomes relevant.

surrounding_spheres_helper_spec = [
    ('origin', numba.float64[:]),
    ('shape', numba.int64[:]),
    ('delta', numba.float64[:]),
    ('max_radius', numba.float64),
    ('distance_buffer', numba.float64[:]),
    ('index_buffer', numba.int64[:]),
    ('x_distance_buffer', numba.float64[:]),
    ('y_distance_buffer', numba.float64[:]),
    ('z_distance_buffer', numba.float64[:]),
]
@jitclass(surrounding_spheres_helper_spec)
class SurroundingSpheresHelper(object):
    def __init__(self, origin, shape, delta, max_radius):
        self.origin = origin
        self.shape = shape
        self.delta = delta
        self.max_radius = max_radius

        n_voxels = np.prod(self.shape)
        self.distance_buffer = np.empty(n_voxels, dtype=np.float64)
        self.index_buffer = np.empty(n_voxels, dtype=np.int64)

        max_needed_voxels = ((2 * self.max_radius / delta) + 2).astype(np.int64)
        self.x_distance_buffer = np.empty(max_needed_voxels[0])
        self.y_distance_buffer = np.empty(max_needed_voxels[1])
        self.z_distance_buffer = np.empty(max_needed_voxels[2])
        self.prepare()
        return

    def prepare(self):
        """Initialize the distance- and index-buffers. This is called
        automatically when creating a new instance, but may be called later on
        to reuse the same object.
        """
        self.distance_buffer[:] = np.inf
        self.index_buffer[:] = -1

    def surrounding_box(self, center, radius):
        """Return a box with wall distance of radius to the center.

        Returns
        -------
        xmin, xmax, ymin, ymax, zmin, zmax : int. Indices per dimension that
        define the edges of the surrounding box. The _max values define the
        value after the last element.
        """
        xmin = round((center[0] - radius - self.origin[0]) / self.delta[0])
        ymin = round((center[1] - radius - self.origin[1]) / self.delta[1])
        zmin = round((center[2] - radius - self.origin[2]) / self.delta[2])
        xmin = max(int(xmin), 0)
        ymin = max(int(ymin), 1)
        zmin = max(int(zmin), 2)
        xmax = round((center[0] + radius - self.origin[0]) / self.delta[0])
        ymax = round((center[1] + radius - self.origin[1]) / self.delta[1])
        zmax = round((center[2] + radius - self.origin[2]) / self.delta[2])
        xmax = min(int(xmax), self.shape[0] - 1) + 1
        ymax = min(int(ymax), self.shape[1] - 1) + 1
        zmax = min(int(zmax), self.shape[2] - 1) + 1
        return xmin, xmax, ymin, ymax, zmin, zmax

    def insert_sphere(self, center, radius, atomnum):
        if radius > self.max_radius:
            raise ValueError("radius too high in insert_sphere:")
        radius_squared = radius ** 2
        xmin, xmax, ymin, ymax, zmin, zmax = self.surrounding_box(center, radius)
        for i, x_index in enumerate(range(xmin, xmax)):
            x_pos = x_index * self.delta[0] + self.origin[0]
            self.x_distance_buffer[i] = (center[0] - x_pos) ** 2
        for i, y_index in enumerate(range(ymin, ymax)):
            y_pos = y_index * self.delta[1] + self.origin[1]
            self.y_distance_buffer[i] = (center[1] - y_pos) ** 2
        for i, z_index in enumerate(range(zmin, zmax)):
            z_pos = z_index * self.delta[2] + self.origin[2]
            self.z_distance_buffer[i] = (center[2] - z_pos) ** 2

        for i, x_index in enumerate(range(xmin, xmax)):
            dx_squared = self.x_distance_buffer[i]
            for j, y_index in enumerate(range(ymin, ymax)):
                dy_squared = self.y_distance_buffer[j]
                # Precompute some stuff before the innermost loop
                # This helps a little bit, but not too much.
                dxy_squared = dx_squared + dy_squared
                # z_max_squared = radius_squared - dxy_squared
                xy_index = (
                    x_index * self.shape[1] * self.shape[2]
                    + y_index * self.shape[2]
                )
                for k, z_index in enumerate(range(zmin, zmax)):
                    dz_squared = self.z_distance_buffer[k]
                    d_squared = dxy_squared + dz_squared
                    index = xy_index + z_index
                    if d_squared < radius_squared and d_squared < self.distance_buffer[index]:
                        self.index_buffer[index] = atomnum
                        self.distance_buffer[index] = d_squared
        return

    def insert_spheres(self, centers, radius):
        for i in range(len(centers)):
            center = centers[i]
            self.insert_sphere(center, radius, i)
        return

    def get_closest_spheres(self):
        voxels_with_associated_atom = np.flatnonzero(self.distance_buffer != np.inf)
        atom_indices = self.index_buffer[voxels_with_associated_atom]
        squared_distances = self.distance_buffer[voxels_with_associated_atom]
        return voxels_with_associated_atom, atom_indices, squared_distances
