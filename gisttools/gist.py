import numpy as np
import warnings
import os
from .grid import Grid, combine_grids, grid_from_xyz
from .utils import ProgressPrinter, distance_weight
import scipy.interpolate
import pandas as pd
import gzip
import re


def open_maybe_gzipped(filename):
    """Try to open a file using the gzip library. If this fails, open directly."""
    try:
        handle = gzip.open(filename)
        handle.peek(1)
    except OSError:
        handle = open(filename)
    return handle


def gist_colnames_v4(fh):
    fh.seek(0)
    next(fh)
    l2 = next(fh).strip()
    entries = l2.split()
    renamed = [
        s
        .replace('Eww-norm-unref', 'Eww_unref_norm')
        .replace('Eww-dens', 'Eww_unref_dens')
        .replace('-norm', '_norm')
        .replace('-dens', '_dens')
        .replace('coord', '')
        for s in entries
    ]
    wo_units = [
        re.sub(r'\(.*\)', '', s)
        for s in renamed
    ]
    return wo_units


def gist_colnames(fmt, fh=None):
    """Return column names of a GIST file based on the format and the first line.

    The original GIST was implemented in Amber 14. With Amber 16, the dTS-six columns
    were introduced, featuring one annoying whitespace in the header. With the GPU
    implementation, the column labels were again changed, putting the g_O and g_H (and
    potentially others, when using non-water solvents) to the end of the input file.

    Parameters
    ----------
    fmt : str
        Format of the GIST file
    fh : file handle or None
        The GIST table to read from, needed for gigist because colnames are not fixed.

    Returns
    -------
    fmt : str
        The GIST format ('amber14', 'amber16', or 'gigist')
    """
    if fmt == 'v4':
        return gist_colnames_v4(fh=fh)
    if fmt == 'amber14':
        cols = [
            'voxel', 'x', 'y', 'z', 'population', 'g_O', 'g_H',
            'dTStrans_dens', 'dTStrans_norm', 'dTSorient_dens', 'dTSorient_norm',
            'Esw_dens', 'Esw_norm', 'Eww_unref_dens', 'Eww_unref_norm', 'Dipole_x_dens',
            'Dipole_y_dens', 'Dipole_z_dens', 'Dipole_dens', 'neighbor_dens',
            'neighbor_norm', 'order_norm',
        ]
    elif fmt == 'amber16':
        cols = [
            'voxel', 'x', 'y', 'z', 'population', 'g_O', 'g_H',
            'dTStrans_dens', 'dTStrans_norm', 'dTSorient_dens', 'dTSorient_norm',
            'dTSsix_dens', 'dTSsix_norm', 'Esw_dens', 'Esw_norm', 'Eww_unref_dens',
            'Eww_unref_norm', 'Dipole_x_dens', 'Dipole_y_dens', 'Dipole_z_dens',
            'Dipole_dens', 'neighbor_dens', 'neighbor_norm', 'order_norm',
        ]
    elif fmt == 'pme':
        cols = [
            'voxel', 'x', 'y', 'z', 'population', 'g_O', 'g_H',
            'dTStrans_dens', 'dTStrans_norm', 'dTSorient_dens', 'dTSorient_norm',
            'dTSsix_dens', 'dTSsix_norm', 'Esw_dens', 'Esw_norm', 'Eww_unref_dens',
            'Eww_unref_norm', 'PME_dens', 'PME_norm', 'Dipole_x_dens', 'Dipole_y_dens',
            'Dipole_z_dens', 'Dipole_dens', 'neighbor_dens', 'neighbor_norm',
            'order_norm',
        ]
    elif fmt == 'gigist':
        assert fh is not None, 'A file handle is needed to detect gigist column names.'
        fh.readline()
        second_line = fh.readline().strip()
        # Might be a string if the file handle is a StringIO
        if not isinstance(second_line, str):
            second_line = second_line.decode('utf-8')
        cols = [
            "voxel", "x", "y", "z", "population", "dTStrans_dens", "dTStrans_norm",
            "dTSorient_dens", "dTSorient_norm", "dTSsix_dens", "dTSsix_norm",
            "Esw_dens", "Esw_norm", "Eww_unref_dens", "Eww_unref_norm", "Dipole_x_dens",
            "Dipole_y_dens", "Dipole_z_dens", "Dipole_dens", "neighbor_dens",
            "neighbor_norm", "order_norm",
        ]
        cols += second_line.split()[len(cols):]
    else:
        raise ValueError(f"Unknown GIST fileformat {fmt}.")
    return cols


def detect_gist_format(fh):
    """Detect GIST version based on the first line of a table-format GIST output file.

    The original GIST was implemented in Amber 14. With Amber 16, the dTS-six columns
    were introduced, featuring one annoying whitespace in the header. With the GPU
    implementation, the column labels were again changed, putting the g_O and g_H (and
    potentially others, when using non-water solvents) to the end of the input file.

    Parameters
    ----------
    fh : file handle
        The GIST table to read from.

    Returns
    -------
    fmt : str
        The GIST format ('amber14', 'amber16', or 'gigist')
    """
    first_line = fh.readline().strip()
    if first_line.lower().startswith("gist output v4"):
        return "v4"
    second_line = fh.readline().strip()
    # Might be a str if fh is a StringIO
    if not isinstance(second_line, str):
        second_line = second_line.decode('utf-8')
    if second_line.startswith(
        'voxel xcoord ycoord zcoord population g_O g_H dTStrans-dens(kcal/mol/A^3) dTStrans-norm(kcal/mol) dTSorient-dens(kcal/mol/A^3) dTSorient-norm(kcal/mol) Esw-dens(kcal/mol/A^3) Esw-norm(kcal/mol) Eww-dens(kcal/mol/A^3) Eww-norm-unref(kcal/mol) Dipole_x-dens(D/A^3) Dipole_y-dens(D/A^3) Dipole_z-dens(D/A^3) Dipole-dens(D/A^3) neighbor-dens(1/A^3) neighbor-norm order-norm'
    ):
        fmt = 'amber14'
    elif second_line.startswith(
        'voxel xcoord ycoord zcoord population g_O g_H dTStrans-dens(kcal/mol/A^3) dTStrans-norm(kcal/mol) dTSorient-dens(kcal/mol/A^3) dTSorient-norm(kcal/mol) dTSsix-dens(kcal/mol/A^3) dTSsix-norm (kcal/mol) Esw-dens(kcal/mol/A^3) Esw-norm(kcal/mol) Eww-dens(kcal/mol/A^3) Eww-norm-unref(kcal/mol) Dipole_x-dens(D/A^3) Dipole_y-dens(D/A^3) Dipole_z-dens(D/A^3) Dipole-dens(D/A^3) neighbor-dens(1/A^3) neighbor-norm order-norm'
    ) or second_line.startswith(
        'voxel xcoord ycoord zcoord population g_O g_H dTStrans-dens(kcal/mol/A^3) dTStrans-norm(kcal/mol) dTSorient-dens(kcal/mol/A^3) dTSorient-norm(kcal/mol) dTSsix-dens(kcal/mol/A^3) dTSsix-norm(kcal/mol) Esw-dens(kcal/mol/A^3) Esw-norm(kcal/mol) Eww-dens(kcal/mol/A^3) Eww-norm-unref(kcal/mol) Dipole_x-dens(D/A^3) Dipole_y-dens(D/A^3) Dipole_z-dens(D/A^3) Dipole-dens(D/A^3) neighbor-dens(1/A^3) neighbor-norm order-norm'

    ):
        fmt = 'amber16'
    elif second_line.startswith(
        'voxel xcoord ycoord zcoord population g_O g_H dTStrans-dens(kcal/mol/A^3) dTStrans-norm(kcal/mol) dTSorient-dens(kcal/mol/A^3) dTSorient-norm(kcal/mol) dTSsix-dens(kcal/mol/A^3) dTSsix-norm(kcal/mol) Esw-dens(kcal/mol/A^3) Esw-norm(kcal/mol) Eww-dens(kcal/mol/A^3) Eww-norm-unref(kcal/mol) PME-dens(kcal/mol/A^3) PME-norm(kcal/mol) Dipole_x-dens(D/A^3) Dipole_y-dens(D/A^3) Dipole_z-dens(D/A^3) Dipole-dens(D/A^3) neighbor-dens(1/A^3) neighbor-norm order-norm'
    ):
        fmt = 'pme'
    elif second_line.split()[:7] == [
        "voxel",
        "x",
        "y",
        "z",
        "population",
        "dTSt_d(kcal/mol)",
        "dTSt_n(kcal/mol)"
    ]:
        fmt = 'gigist'
    else:
        raise ValueError(f"Could not detect file format using the file header:\n{second_line}.")
    return fmt


class Gist:
    """
    Contains commonly used analysis functions for GIST output, like integrating within
    a region defined by vicinity to a molecule, or projecting averaged energy values
    onto atomic positions.


    Attributes
    ----------
    data : pd.DataFrame
        Contains the raw data of the GIST output file.
    grid : Grid
        Contains Grid information and indexing functionality.
    struct : None or mdtraj.Trajectory
        if not None, this will provide the default for all operations that require
        molecular coordinates.
    n_frames : float
        Number of frames that was used for the GIST calculation.
    rho0 : float
        Reference density that was used for the GIST calculation.
    eww_ref : float
        Reference energy that will get subtracted from Eww_unref_norm to obtain
        Eww_norm. Affects all columns that are derived from Eww as well.
    loc : GistLocator instance
        To allow for pandas-like data access, gist.loc[...]
    _recipes : dict
        A dict of column name : bound method, that defines how derived columns such as
        A_dens should be created. Note that columns in .data take precedence over
        recipes.
    """

    def __init__(
        self,
        data,
        grid=None,
        struct=None,
        strip_H=False,
        n_frames=None,
        rho0=None,
        eww_ref=None,
        autodetect_refcol='g_O',
    ):
        """Initialize a new Gist object.

        Parameters
        ----------
        data : pandas DataFrame
            The raw data. Can still be modified after initialization. The length of all
            data rows must be equal to grid.size
        grid : grid.Grid instance
            The Grid information.
        struct : mdtraj Trajectory
            If coord is None, use the first frame of this trajectory.
        strip_H : bool
            Whether hydrogen should be stripped from struct.
        eww_ref : float
            Reference value for Eww_norm. Default None
        autodetect_refcol : str
            The reference column to use when detecting n_frames and rho0. This HAS to
            be the column that corresponds to the population. When using water, this is
            'g_O', for chloroform it is 'g_C'.
        """
        self._recipes = {
            "Eww_norm": self._recipe_eww_norm,
            "Eww_dens": self._recipe_eww_dens,
            "Eall_norm": self._recipe_eall_norm,
            "Eall_dens": self._recipe_eall_dens,
            "dTSsix_norm": self._recipe_dTSsix_norm,
            "dTSsix_dens": self._recipe_dTSsix_dens,
            "A_norm": self._recipe_A_norm,
            "A_dens": self._recipe_A_dens,
            # for PME only
            # "Eww_unref_norm": self._recipe_eww_unref_norm,
            # "Eww_unref_dens": self._recipe_eww_unref_dens,
        }
        self.loc = _GistLocator(self)
        self.data = data
        # if "Eww_dens" in data:
        #     self.data = self.data.rename(columns={"Eww_dens": "Eww_unref_dens"})
        if grid is None:
            grid = grid_from_xyz(self[["x", "y", "z"]].values)
        self.grid = grid
        if rho0 is None:
            try:
                rho0 = self.detect_rho(refcol=autodetect_refcol)
            except KeyError:  # a column is missing from data
                rho0 = None
        self.rho0 = rho0
        if n_frames is None:
            try:
                n_frames = self.detect_frames(refcol=autodetect_refcol)
            except KeyError:
                n_frames = None
        self.n_frames = n_frames
        if eww_ref == None:
            eww_ref = 0.
            if 'Eww_unref_norm' in self.data.columns:
                warnings.warn(RuntimeWarning(
                    'eww reference is zero, but there is a Eww_unref_norm column. '
                    'All operations that rely on referenced Eww values will '
                    'generate wrong results. Explicitly set eww_ref to zero to '
                    'suppress this warning.'
                ))
        self.eww_ref = eww_ref
        self.struct = struct
        if struct is not None and strip_H:
            self.struct = self.struct.atom_slice(self.struct.top.select("symbol != H"))
        return

    @property
    def coord(self):
        """Returns coordinates from traj, converted from nm to Angstrom."""
        return self.struct.xyz[0] * 10.

    def interpolate(self, columns, xyz):
        """
        Linear interpolation of data at points xyz.

        Parameters
        ----------
        columns : list of str
            Data columns to interpolate. Must be valid keys to self.data
        xyz : np.ndarray, shape=(n, 3)
            XYZ coordinates to evaluate data at.

        Returns
        -------
        out : dict
            Dictionary of col: data, where data is the interpolated values, and col are
            the keys defined by columns.

        Examples
        --------
        >>> import pandas as pd
        >>> data = np.array([13., 4., 6., -5., -9., -1., 8., 10.])
        >>> df = pd.DataFrame({
        ...     'x': [0, 0, 0, 0, 1, 1, 1, 1],
        ...     'y': [0, 0, 1, 1, 0, 0, 1, 1],
        ...     'z': [0, 1, 0, 1, 0, 1, 0, 1],
        ...     'Eww_unref_norm': data})
        >>> a = Gist(df, eww_ref=-9.533, n_frames=1, rho0=0.003)
        >>> a.interpolate(['Eww_unref_norm'], np.array([[.5, .5, .5]]))
        {'Eww_unref_norm': array([3.25])}
        >>> np.average(data)
        3.25
        >>> a.interpolate(['Eww_unref_norm'], np.array([[0, 0, .5]]))
        {'Eww_unref_norm': array([8.5])}
        >>> a.interpolate(['Eww_unref_norm'], np.array([[0, 1, .5], [3, 2, 1]]))
        {'Eww_unref_norm': array([0.5, nan])}
        """
        out = {
            col: scipy.interpolate.interpn(
                points=self.grid.edges,
                values=self[col].values.reshape(self.grid.shape),
                xi=xyz,
                fill_value=np.nan,
                bounds_error=False
            ) for col in columns
        }
        return out

    @property
    def num_frames(self):
        warnings.warn('Gist.num_frames has been deprecated. Use Gist.n_frames')
        return self.n_frames

    @classmethod
    def from_dataframe(cls, *args, **kwargs):
        warnings.warn('Gist.from_dataframe has been deprecated. Just use Gist(df)')
        return Gist(*args, **kwargs)

    def detect_rho(self, refcol='g_O'):
        """Automatically detect the number of frames and the reference density
        from data points.

        Parameters
        ----------
        refcol : str
            The reference column to use. This HAS to be the column that corresponds to
            the population. When using water, this is 'g_O', for chloroform it is
            'g_C'.

        Returns
        -------
        rho0 : float
            Reference density, e.g. 0.0329 for TIP3P (limited accuracy because of the
            GIST output file).

        Examples
        --------
        >>> import pandas as pd
        >>> a = pd.DataFrame({
        ...     'population': [25],
        ...     'g_O': [1.21581],
        ...     'Eww_unref_dens': [-0.399056],
        ...     'Eww_unref_norm': [-9.9764]
        ... })
        >>> gist = Gist(a, grid=Grid(0, 10, 0.5), eww_ref=0)
        >>> rho0 = gist.detect_rho()
        >>> f"{rho0:.4f}"
        '0.0329'

        """
        # I used to use self['population'] here. This is unsafe when using GIST
        # with ions, since there can be a voxel with no water but a lot of ions.
        # The calculated rho0 with ions is WRONG. But this way at least n_frames
        # works...
        # highest_pop_index = self[refcol].idxmax()
        # highest_pop = self.loc[highest_pop_index]
        rho0 = (self["Eww_unref_dens"] / self["Eww_unref_norm"]).sum(0) / self[refcol].sum(0)
        return rho0

    def has_pme(self):
        return 'PME_norm' in self.data.columns

    def detect_frames(self, refcol='g_O'):
        """Automatically detect the number of frames and the reference density
        from data points.

        Parameters
        ----------
        refcol : str
            The reference column to use. This HAS to be the column that corresponds to
            the population. When using water, this is 'g_O', for chloroform it is
            'g_C'.

        Returns
        -------
        n_frames : int
            Number of frames in the GIST simulation. Usually rounded to the exact
            number.

        Examples
        --------
        >>> import pandas as pd
        >>> a = pd.DataFrame({
        ...     'population': [25],
        ...     'g_O': [1.21581],
        ...     'Eww_unref_dens': [-0.399056],
        ...     'Eww_unref_norm': [-9.9764]
        ... })
        >>> gist = Gist(a, grid=Grid(0, 10, 0.5), eww_ref=0)
        >>> frames = gist.detect_frames()
        >>> f"{frames}"
        '5000'

        """
        voxel_volume = self.grid.voxel_volume
        # I used to use self['population'] here. This is unsafe when using GIST
        # with ions, since there can be a voxel with no water but a lot of ions.
        # highest_pop_index = self[refcol].idxmax()
        # highest_pop = self.loc[highest_pop_index]
        rho0 = self.rho0
        if pd.isna(rho0):
            rho0 = self.detect_rho(refcol=refcol)
        if pd.isna(rho0):
            raise ValueError('Cannot detect number of frames because rho0 is NaN and cannot be detected.')
        n_frames = np.int_(
            np.round(self["population"].sum() / rho0 / voxel_volume / self[refcol].sum())
        )
        return n_frames

    def distance_to_spheres(self, centers='struct', rmax=5., atomic_radii=None):
        """Uses grid.distance_to_spheres to find voxels within a given distance to
        centers. If centers are None, self.coord is used.

        Note that the returned distances can be negative for voxels that lie within one
        of the spheres. Similar to distance_to_centers, only the smallest distance is
        reported.

        Parameters
        ----------
        centers : array-like, shape=(m, 3), or 'struct'
            x, y, and z coordinates of the m requested atom positions. If None,
            self.coord is used. (Default None)
        rmax : float, default 5.
            maximum distance of the returned voxels to the respective center.
        atomic_radii : array-like, None, or 'struct'
            The atomic radius will be subtracted from every computed distance, yielding
            the distance to the molecular surface instead of the distance to the
            closest atomic center. If 'struct', uses the radii of self.struct. If None,
            calculate the distance to the centers instead of to the atomic surface.
            (Default None)

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
        """
        if isinstance(centers, str) and centers == 'struct':
            centers = self.coord
        centers = np.asarray(centers).reshape(-1, 3)
        if isinstance(atomic_radii, str) and atomic_radii == 'struct':
            atomic_radii = np.array([a.element.radius for a in self.struct.top.atoms]) * 10.
        if len(centers) == 1:
            ind, dist = self.grid.surrounding_sphere(centers, rmax)
            if atomic_radii is not None:
                dist -= atomic_radii
            return ind, np.array([0]), dist
        else:
            if atomic_radii is None:
                return self.grid.distance_to_centers(centers, rmax)
            return self.grid.distance_to_spheres(centers, rmax, atomic_radii)

    def detect_reference_value(
        self,
        columns='Eww_unref',
        col_suffix='_dens',
        centers='struct',
        dlim=(12, 16),
        n_bins=10,
        min_relative_population=0.2,
        max_spread=0.01,
    ):
        """
        Detects the reference value for col using a mean over voxels that are (hopefully)
        sufficiently far away from the molecule defined by centers.

        Be careful when subtracting the return value of detect_reference_value from a
        _norm column. This function returns a Series of length len(columns), which
        cannot directly be subtracted from a Gist column, which is a Series of length
        n_voxels. Use detect_reference_value(...).values[0] instead!

        Parameters
        ----------
        col : str
            For which GIST column the reference value should be calculated.
        col_suffix : str, default '_dens'
            Will be added to the column label. The default is _dens because this function
            is only correct with normed columns, even though the output is used to
            reference the _norm column.
        centers : np.ndarray, shape=(n, 3)
            Positions of n atoms to project GIST data to. If 'struct', uses self.coord.
            Default 'struct'.
        dlim : tuple of 2 positive floats, where dlim[0] < dlim[1]
            Upper and lower limit of the rdf-slice that is used to detect the reference
            value.
        n_bins : int
            Number of bins for the rdf. This value does not affect the
            calculation, but it does affect the density maximum, which is taken
            as reference for min_relative_population. Must be even because the
            sanity checks split the distance range in 2 equal groups.
        min_relative_population : float
            Assert that the ratio between the mean bin population in the first half of
            the range and the population maximum in the histogram, as well as the same
            ratio for the second half, is below min_relative_population.
        max_spread : float
            Asserts that the difference in the reference value obtained from the first
            and second half of the range is below max_spread. Raises RuntimeError
            otherwise.

        Examples
        --------
        >>> # Reference Eww and dTSsix columns of a Gist object called gf
        >> eww_ref, dts_ref = gf.detect_reference_value(
        ..     ['Eww_unref', 'dTSsix'],
        ..     dlim=(16, 24)
        .. ).values
        >> gf.eww_ref = eww_ref
        >> # There is no built-in referencing for columns other than Eww, but we can re-calculate them.
        >> gf['dTSsix_norm'] -= dts_ref
        >> gf['dTSsix_dens'] = gf.norm2dens(gf['dTSsix_norm'])

        Returns
        -------
        A pandas.Series object, where the columns are the index, and the respective
        reference values are the values.
        """
        assert dlim[1] >= dlim[0], f'dlim[1] must be >= dlim[0], but dlim is {dlim}'
        if isinstance(columns, str):
            columns = [columns]
        columns_with_suffix = [c + col_suffix for c in columns]
        # There must be an even number of bins, so that we can split the range
        # in half for the sanity checks.
        assert (n_bins % 2) == 0, "n_bins must be even."
        bin_edges = np.linspace(dlim[0], dlim[1], n_bins, endpoint=False)
        _, (pop_hist, *histograms) = self.multiple_rdfs(
            ['population'] + columns_with_suffix,
            rmax=dlim[1],
            bins=bin_edges,
            col_suffix='',
        )
        max_pop = np.max(pop_hist)
        first_half = np.zeros(n_bins, dtype=bool)
        first_half[:n_bins//2] = True
        second_half = ~first_half
        # Should never be false since we already checked that there is an even
        # number of bins.
        assert np.sum(first_half) == np.sum(second_half), \
            "Uneven split in sanity check in detect_reference_value. This should never happen!"
        # Sanity check Nr. 1:
        # Check that there is a significant amount of population (water count)
        # both in the first and in the second half of the distance range
        if np.average(pop_hist[first_half]) / max_pop < min_relative_population:
            raise RuntimeError(
                'Not enough population in the first half of dlim, '
                f'{np.average(pop_hist[first_half])} / {max_pop} < {min_relative_population}'
            )
        if np.average(pop_hist[second_half]) / max_pop < min_relative_population:
            raise RuntimeError(
                'Not enough population in the second half of dlim, '
                f'{np.average(pop_hist[second_half])} / {max_pop} < {min_relative_population}'
            )

        ref_values = pd.Series()
        for col, hist in zip(columns, histograms):
            # Normalize hist by the population, so that it converges towards the per_molecule reference value.
            hist /= (pop_hist / (self.n_frames * self.grid.voxel_volume))
            ref_values[col] = np.average(hist, weights=pop_hist)
            # Sanity check Nr. 2:
            # Check that the average reference value in the first and second half
            # of the distance range is about equal (within max_spread)
            if abs(
                np.average(hist[first_half], weights=pop_hist[first_half])
                - np.average(hist[second_half], weights=pop_hist[second_half])
            ) > max_spread:
                raise RuntimeError(
                    'Too much difference between first and second half of dlim: '
                    f'{np.average(hist[first_half], weights=pop_hist[first_half])}'
                    f' != {np.average(hist[second_half], weights=pop_hist[second_half])}'
                )
        return ref_values

    def reference_mixture(
        self,
        non_water_density_cols,
        energy_col='Eww_unref_dens',
        pop_col='population',
        centers='struct',
        rmin=16.,
        atomic_radii=None,
    ):
        voxels_within_rmin, _, _ = self.distance_to_spheres(
            centers=centers,
            rmax=rmin,
            atomic_radii=atomic_radii
        )
        far_away = np.ones(self.grid.size, dtype=bool)
        far_away[voxels_within_rmin] = False
        if isinstance(non_water_density_cols, str):
            non_water_density_cols = [non_water_density_cols]
        # A density column based on the population
        # In contrast to the g_ columns, the population is based on the centers
        # of mass (at least if the com option for gigist was used), as well as
        # ions.
        g_population = self[pop_col].values / (self.rho0 * self.n_frames * self.grid.voxel_volume)
        g_com = g_population - self[non_water_density_cols].sum(1).values
        # g_com now contains the density of the water centers of mass.
        # densities = gf.loc[far_away, x_components].values
        densities = np.concatenate((
            self[non_water_density_cols].values,
            g_com.reshape(-1, 1)
        ), axis=1)
        ref_dens = densities[far_away]
        ref_ener = self.loc[far_away, [energy_col]].values
        assert len(ref_dens.shape) == 2 and ref_dens.shape[1] == len(non_water_density_cols) + 1, \
            "Wrong shape for ref_dens: " + str(ref_dens.shape)
        assert len(ref_ener.shape) == 2 and ref_ener.shape[1] == 1, \
            "Wrong shape for ref_ener: " + str(ref_ener.shape)
        assert len(ref_dens) == len(ref_ener), "Length of ref_dens and ref_ener does not match"
        assert ref_dens.shape[0] > 10000, \
            "Insufficient voxel number for ref_dens: " + str(ref_dens.shape[0])
        ## OLS
        refvals = np.linalg.inv(ref_dens.T @ ref_dens) @ ref_dens.T @ ref_ener
        refvals = refvals.reshape(1, len(non_water_density_cols) + 1)
        ref_energy = (densities * refvals).sum(1)
        return ref_energy
        

    def integrate_around(
        self,
        columns,
        rmax=5.,
        centers='struct',
        extra_weights=1,
        col_suffix='_dens',
        weighting_method=None,
        weighting_options=None,
    ):
        """Integrate the given columns around the given centers. If no centers are
        given, defaults to self.coord.

        Notes
        -----
        This function only works with density-weighted columns, therefore _dens is
        added to all columns by default. You can generate density-weighted columns from
        normed columns using the dens2norm method. You can also override this behavior
        by setting col_suffix=''.

        Parameters
        ----------
        columns : list of str
            column indices to project.  For each col, col + '_dens' must be valid
            column index.
        rmax : float
            Radius of the sphere from which voxels are to be projected onto the atoms,
            in angstrom.
        centers : np.ndarray, shape=(n, 3), or 'struct'.
            Positions of n atoms to project GIST data to. If 'struct' is given, uses
            self.coord.
        col_suffix : str, default '_dens'
            Will be added to all column labels. The default is _dens because this
            functions is only correct with density-weighted columns.
        weighting_method : str or None
            Used to weight voxels by their distance to the nearest atom. Can be
            'piecewise_linear', 'gaussian', 'logistic', None, or a callable.
        weighting_options : dict
            Options to pass on to the weighting function. See util.weight_... for
            options to the different weighting functions.

        Returns
        -------
        A pandas.Series object with the columns as keys.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns = [c + col_suffix for c in columns]
        rename = lambda text : text[:-len(col_suffix)]
        ind, _, dist = self.distance_to_spheres(centers, rmax)
        weights = extra_weights * self.grid.voxel_volume
        if weighting_options is None:
            weighting_options = {}
        weights = (
            extra_weights \
            * self.grid.voxel_volume \
            * distance_weight(dist, weighting_method, **weighting_options)
        )
        out = pd.Series(dtype=float)
        for col in columns:
            out[col] = np.sum((self.loc[ind, col]*weights).values)
        return out.rename(rename)

    def projection_mean(
        self,
        columns,
        rmax=5,
        centers='struct',
        residues=None,
        atomic_radii=None,
        col_suffix='_dens',
        weighting_method=None,
        weighting_options=None,
    ):
        """Project GIST data to the nearest atoms using the algorithm from Michi's
        python scripts.

        Notes
        -----
        This function only works with density-weighted columns, therefore _dens is
        added to all columns by default. You can generate density-weighted columns from
        normed columns using the dens2norm method. You can also override this behavior
        by setting col_suffix=''.

        Parameters
        ----------
        columns : list of str
            column indices to project.  For each col, col + '_dens' must be valid
            column index.
        rmax : float
            Radius of the sphere from which voxels are to be projected onto the atoms,
            in angstrom.
        centers : np.ndarray, shape=(n, 3)
            Positions of n atoms to project GIST data to. If 'struct', uses self.coord.
            Default 'struct'.
        residues : iterable, len(residues) = n
            Residue numbers of all atoms.  Atoms with the same number will be treated
            as a residue.  If residues is None, treat all atoms separately.
        atomic_radii : np.ndarray or None or 'struct'.
            The atomic radius will be subtracted from every computed distance, yielding
            the distance to the molecular surface instead of the distance to the
            closest atomic center. If 'struct', uses the radii of self.struct. If None,
            calculate the distance to the centers instead of to the atomic surface.
            (Default None)
        col_suffix : str, default '_dens'
            Will be added to all column labels. The default is _dens because this
            functions is only correct with density-weighted columns.
        weighting_method : str or None
            Used to weight voxels by their distance to the nearest atom. Can be
            'piecewise_linear', 'gaussian', 'logistic', None, or a callable. 
        weighting_options : dict
            Options to pass on to the weighting function. See util.weight_... for
            options to the different weighting functions.

        Returns
        -------
        out : dict
            Each element contains the projection of the data in the corresponding
            column to the atoms in the reference structure.  The keys are equal to
            those given in columns.

        """
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(centers, str) and centers == 'struct':
            centers = self.coord
        if residues is None:
            residues = np.arange(centers.shape[0])
        if weighting_options is None:
            weighting_options = {}

        unique_res = np.unique(residues)
        out = pd.DataFrame(index=unique_res, columns=columns)
        pop = self['population'].values
        temp_data = {c: self[c + col_suffix].values for c in columns}

        with ProgressPrinter('{:.0f} % of atoms processed.', len(unique_res)) as progress:
            for resnum in unique_res:
                # nonzero returns a list of arrays.
                res_ind = np.nonzero(residues == resnum)[0]
                res_coords = centers[res_ind]

                ind, _, dist = self.distance_to_spheres(res_coords, rmax=rmax, atomic_radii=atomic_radii)
                weights = distance_weight(dist, weighting_method, **weighting_options)
                normalization = np.sum(weights * pop[ind]) / (self.n_frames * self.grid.voxel_volume)

                for col in columns:
                    out.loc[res_ind, col] = np.sum(
                        temp_data[col][ind] * weights
                    ) / normalization
                progress.tick()
        return out


    def projection_nearest(
        self,
        columns,
        centers='struct',
        cutoff_E=0,
        rmax=7.0,
        atomic_radii=None,
        col_suffix='_dens',
        weighting_method=None,
        weighting_options=None,
    ):
        """Project data of each GIST voxel to the respective nearest atom.

        Parameters
        ----------
        columns : str or list of str
            column indices to project.  Must be valid indices for GistFile.data
        centers : np.ndarray, shape=(n, 3)
            Positions of n atoms to project GIST data to. If 'struct', uses self.coord.
            Default 'struct'.
        cutoff_E : float
            Energy density, that a voxel (in kcal/mol) needs to have in order to be
            included in the projection. No normalization is performed prior to applying
            this cutoff.
        rmax : float
            Maximum distance, in angstrom, of voxel centers to the nearest atom
            (defined by its center or its atomic surface, if atomic_radii is given)
        atomic_radii : np.ndarray or None or 'struct'
            The atomic radius will be subtracted from every computed distance, yielding
            the distance to the molecular surface instead of the distance to the
            closest atomic center. If 'struct', uses the radii of self.struct. If None,
            calculate the distance to the centers instead of to the atomic surface.
            (Default None)
        col_suffix : str, default '_dens'
            Will be added to all column labels. The default is _dens because this
            function is only correct with density-weighted columns.
        weighting_method : str or None
            Used to weight voxels by their distance to the nearest atom.
        weighting_options : dict
            Options to pass on to the weighting function.

        Returns
        -------
        atom_projections : pandas.Series
            Each element contains the projection of the data in the
            corresponding column to the atoms in the reference structure.  The
            keys are equal to those given in columns.

        """
        if isinstance(columns, str):
            columns = [columns]

        ind, closest, dist = self.distance_to_spheres(centers, rmax=rmax, atomic_radii=atomic_radii)

        if weighting_options is None:
            weighting_options = {}
        if weighting_method is not None:
            weights = distance_weight(dist, weighting_method, **weighting_options)

        normalized_data = {}
        for col in columns:
            if col == 'voxels':
                values = np.ones_like(ind)
            else:
                values = self.loc[ind, col + col_suffix].values
            if cutoff_E != 0:
                values[np.abs(values) < cutoff_E] = 0
            values *= self.grid.voxel_volume
            if weighting_method is not None:
                values *= weights
            normalized_data[col] = values

        normalized_data['atom_index'] = closest
        df = pd.DataFrame.from_dict(normalized_data)
        out = df.groupby('atom_index').sum()
        return out

    def multiple_per_atom_rdfs(
        self,
        columns,
        centers='struct',
        rmax=5.,
        bins=20,
        atomic_radii=None,
        col_suffix='_dens',
        normalize='dens',
    ):
        """Create a radial distribution function (rdf) of GIST quantities around
        centers.

        Parameters
        ----------
        columns : str or list of str
            column indices to project.  Must be valid indices for GistFile.data. If
            'voxels' is given as a column name, returns the voxel count instead.
        centers : np.ndarray, shape=(n, 3)
            Positions of n atoms to project GIST data to. If 'struct', uses self.coord.
            Default 'struct'.
        rmax : float
            Radius of the sphere from which voxels are to be projected onto the atoms,
            in angstrom.
        bins : int or 1D array-like
            Left bin edges (if array-like) or the number of bins that should be
            created for the range [0:rmax]
        atomic_radii : np.ndarray or None or 'struct'
            The atomic radius will be subtracted from every computed distance, yielding
            the distance to the molecular surface instead of the distance to the
            closest atomic center. If 'struct', uses the radii of self.struct. If None,
            calculate the distance to the centers instead of to the atomic surface.
            (Default None)
        col_suffix : str, default '_dens'
            Will be added to all column labels. The default is _dens because this
            function is only correct with density-weighted columns.
        normalize : str or list of str
            How to normalize the rdfs ("none", "dens", or "norm"). Default: "dens"

        Returns
        -------
        bins : np.ndarray(shape=(bins,))
            The left edge of each distance bin.
        rdfs : list of pandas.DataFrame objects
            each element contains the per-atom rdf of one column
        """
        if isinstance(centers, str) and centers == 'struct':
            centers = self.coord
        centers = np.asarray(centers).reshape(-1, 3)
        if isinstance(columns, str):
            columns = [columns]
        bins = np.asarray(bins)
        if len(bins.shape) == 0:
            bins = np.linspace(0, rmax, bins, endpoint=False)
        if isinstance(normalize, str):
            normalize = [normalize] * len(columns)

        ind, closest_atom, dist = self.distance_to_spheres(centers, rmax=rmax, atomic_radii=atomic_radii)

        distance_bins = np.digitize(dist, bins)
        # This is not the final shape of the output dataframe, since we exclude
        # the first distance bin later on...
        df_shape = (len(bins)+1, len(centers))
        distance_and_atom_bin = np.ravel_multi_index(
            (distance_bins, closest_atom), dims=df_shape
        )

        rdfs = []
        for col, norm_by in zip(columns, normalize):
            if col == 'voxels':
                coldata = np.ones_like(ind)
            else:
                coldata = self.loc[ind, col + col_suffix].values
            normalized = self._normalize_values(coldata, ind, norm_by)
            integrals = np.bincount(
                distance_and_atom_bin,
                weights=normalized,
                minlength=np.prod(df_shape)
            )
            # The reason I set it up so that I have to transpose is that this
            # way we get a Fortran-contiguous array, which is the default for
            # DataFrames. There is probably another way to do this, but this
            # works nicely.
            # Now atoms in rows and bins in columns.
            integrals_per_atom = integrals.reshape(df_shape).T
            # Exclude the first and last column (bins for dist < 0 and dist > rmax)
            integrals_per_atom = integrals_per_atom[:, 1:]
            rdfs.append(pd.DataFrame(integrals_per_atom))
        return bins, rdfs

    def _normalize_values(self, values, voxels, by):
        """Return normalized values, depending on "by".

        If by equals
            * "none": do nothing
            * "dens": multiple with the voxel volume
            * "norm": divide by the average number of molecules in each voxel
        """
        if by == "none":
            return values
        elif by == "dens":
            return values * self.grid.voxel_volume
        elif by == "norm":
            n_solvent = self.loc[voxels, "population"] / self.n_frames
            return values / n_solvent

    def rdf(
        self,
        column,
        centers='struct',
        rmax=5.,
        bins=20,
        atomic_radii=None,
        col_suffix='_dens',
        normalize='dens'
    ):
        """Create a radial distribution function (rdf) of a single GIST
        quantities around centers.

        Parameters
        ----------
        column : str
            column index to project.  Must be valid index for GistFile.data. If
            'voxels' is given as a column name, returns the voxel count instead.
        centers : np.ndarray, shape=(n, 3)
            Positions of n atoms to project GIST data to. If 'struct', uses self.coord.
            Default 'struct'.
        rmax : float
            Radius of the sphere from which voxels are to be projected onto the atoms,
            in angstrom.
        bins : int or 1D array-like
            Histogram edges (if array-like) or the number of bins that should be
            created for the range [0:rmax]
        atomic_radii : np.ndarray or None or 'struct'
            The atomic radius will be subtracted from every computed distance, yielding
            the distance to the molecular surface instead of the distance to the
            closest atomic center. If 'struct', uses the radii of self.struct. If None,
            calculate the distance to the centers instead of to the atomic surface.
            (Default None)
        col_suffix : str, default '_dens'
            Will be added to all column labels. The default is _dens because this
            function is only correct with density-weighted columns.
        normalize : str or list of str
            How to normalize the rdfs ("none", "dens", or "norm"). Default: "dens"

        Returns
        -------
        bins : np.ndarray(shape=(bins,))
            The left edge of each distance bin.
        rdf : pd.Series.
            The sum of the respective column voxels, summed per distance bin.
        """
        bins, (rdf, ) = self.multiple_per_atom_rdfs(
            columns=[column],
            centers=centers,
            rmax=rmax,
            bins=bins,
            atomic_radii=atomic_radii,
            col_suffix=col_suffix,
            normalize=normalize,
        )
        return bins, rdf.sum(0).rename(column)

    def per_atom_rdf(
        self,
        column,
        centers='struct',
        rmax=5.,
        bins=20,
        atomic_radii=None,
        col_suffix='_dens',
        normalize='dens',
    ):
        """Create a radial distribution function (rdf) of a single GIST
        quantities around centers.

        Parameters
        ----------
        column : str
            column index to project.  Must be valid index for GistFile.data. If
            'voxels' is given as a column name, returns the voxel count instead.
        centers : np.ndarray, shape=(n, 3)
            Positions of n atoms to project GIST data to. If 'struct', uses self.coord.
            Default 'struct'.
        rmax : float
            Radius of the sphere from which voxels are to be projected onto the atoms,
            in angstrom.
        bins : int or 1D array-like
            Histogram edges (if array-like) or the number of bins that should be
            created for the range [0:rmax]
        atomic_radii : np.ndarray or None or 'struct'
            The atomic radius will be subtracted from every computed distance, yielding
            the distance to the molecular surface instead of the distance to the
            closest atomic center. If 'struct', uses the radii of self.struct. If None,
            calculate the distance to the centers instead of to the atomic surface.
            (Default None)
        col_suffix : str, default '_dens'
            Will be added to all column labels. The default is _dens because this
            function is only correct with density-weighted columns.

        Returns
        -------
        bins : np.ndarray(shape=(bins,))
            The left edge of each distance bin.
        rdf : pandas.DataFrame
            The sum of the respective column voxels, summed per atom (in rows)
            and per distance bin (columns).
        """
        bins, (rdf, ) = self.multiple_per_atom_rdfs(
            columns=[column],
            centers=centers,
            rmax=rmax,
            bins=bins,
            atomic_radii=atomic_radii,
            col_suffix=col_suffix,
            normalize=normalize,
        )
        return bins, rdf

    def multiple_rdfs(
        self,
        columns,
        centers='struct',
        rmax=5.,
        bins=20,
        atomic_radii=None,
        col_suffix='_dens',
        normalize='dens',
    ):
        """Create a radial distribution function (rdf) of a single GIST
        quantities around centers.

        Parameters
        ----------
        columns : str or list of str
            column indices to project.  Must be valid indices for self. If
            'voxels' is given as a column name, returns the voxel count instead.
        centers : np.ndarray, shape=(n, 3)
            Positions of n atoms to project GIST data to. If 'struct', uses self.coord.
            Default 'struct'.
        rmax : float
            Radius of the sphere from which voxels are to be projected onto the atoms,
            in angstrom.
        bins : int or 1D array-like
            Histogram edges (if array-like) or the number of bins that should be
            created for the range [0:rmax]
        atomic_radii : np.ndarray or None or 'struct'
            The atomic radius will be subtracted from every computed distance, yielding
            the distance to the molecular surface instead of the distance to the
            closest atomic center. If 'struct', uses the radii of self.struct. If None,
            calculate the distance to the centers instead of to the atomic surface.
            (Default None)
        col_suffix : str, default '_dens'
            Will be added to all column labels. The default is _dens because this
            function is only correct with density-weighted columns.

        Returns
        -------
        bins : np.ndarray(shape=(bins,))
            The left edge of each distance bin.
        rdfs : List of pd.Series
            The sum of the respective column voxels, summed per atom (in rows)
            for each column in columns.
        """
        if isinstance(columns, str):
            columns = [columns]
        bins, rdfs = self.multiple_per_atom_rdfs(
            columns=columns,
            centers=centers,
            rmax=rmax,
            bins=bins,
            atomic_radii=atomic_radii,
            col_suffix=col_suffix,
            normalize=normalize,
        )
        return bins, [rdf.sum(0).rename(col) for rdf, col in zip(rdfs, columns)]

    def norm2dens(self, data, index=slice(None)):
        """Convert an arbitrary data column from a _norm quantity to a _dens quanity."""
        out = (
            data
            * self.loc[index, 'population']
            / (self.grid.voxel_volume * self.n_frames)
        )
        out[pd.isna(out)] = 0.
        return out

    def dens2norm(self, data, index=slice(None)):
        """Convert an arbitrary data column from a _dens quantity to a _norm quanity."""
        return (
            data
            / self.loc[index, 'population']
            * self.grid.voxel_volume
            * self.n_frames
        )

    def save_dx(self, column, filename):
        """Save a single GIST column to an OpenDX file.

        Parameters
        ----------
        column : str
            Must be a valid index for self.__getitem__
        filename : str or file handle
            The output file

        Returns
        -------
        None
        """
        assert isinstance(column, str), 'save_dx requires a single column name as input.'
        data = self[column].values
        self.grid.save_dx(data, filename, column)
        return

    # def _recipe_eww_unref_norm(self, index):
    #     """When using PME, there is no Eww_unref_norm. Use PME_norm instead."""
    #     return self.loc[index, 'PME_norm']

    # def _recipe_eww_unref_dens(self, index):
    #     """When using PME, there is no Eww_unref_dens. Use PME_dens instead.

    #     In contrast to other _dens recipes, this does not use norm2dens. This
    #     is because eww_unref_dens is used to detect the number of frames and
    #     rho0."""
    #     return self.loc[index, 'PME_dens']

    def _recipe_eww_norm(self, index):
        """Create Eww_norm."""
        return self.loc[index, 'Eww_unref_norm'] - self.eww_ref

    def _recipe_eww_dens(self, index):
        """Create Eww_dens."""
        return self.norm2dens(self.loc[index, 'Eww_norm'], index)

    def _recipe_eall_norm(self, index):
        """Create Eall_norm."""
        return self.loc[index, 'Eww_norm'] + self.loc[index, 'Esw_norm']

    def _recipe_eall_dens(self, index):
        """Create Eall_dens."""
        return self.norm2dens(self.loc[index, 'Eall_norm'], index)

    def _recipe_dTSsix_norm(self, index):
        """Create dTStrans_norm."""
        return self.loc[index, 'dTStrans_norm'] + self.loc[index, "dTSorient_norm"]

    def _recipe_dTSsix_dens(self, index):
        """Create dTSsix_dens."""
        return self.norm2dens(self.loc[index, 'dTSsix_norm'], index)

    def _recipe_A_norm(self, index):
        """Create A_norm."""
        return (
            self.loc[index, "Eww_norm"]
            + self.loc[index, "Esw_norm"]
            - self.loc[index, 'dTSsix_norm']
        )

    def _recipe_A_dens(self, index):
        """Create A_dens."""
        return self.norm2dens(self.loc[index, 'A_norm'], index)

    def __getitem__(self, key):
        if isinstance(key, list):  # A list was passed
            return pd.DataFrame.from_dict({
                col: self[col] for col in key
            })
        try:
            return self.data[key]
        except KeyError as e:
            try:
                return self._recipes[key](slice(None, None, None))
            except:
                raise e

    def __setitem__(self, key, value):
        # Hopefully safe way to check if a list was passed instead of a column name.
        if isinstance(key, tuple) and len(key) >= 2 and isinstance(key[1], list):
            raise TypeError('Indexing using a list of columns is not supported by __setitem__')
        self.data[key] = value

    def __repr__(self):
        return f"Gist(\n    columns={list(self.data)},\n    grid={self.grid},\n    struct={self.struct}\n)"


class _GistLocator:
    """Class that defines the __getitem__ method for a Gist object's .loc property.

    3 ways of indexing are supported:
        * row index only:
            if no column is given, the index will be passed to pandas' .loc. In this
            case, the index should be a valid voxel index or a list or 1-d array. No
            derived columns will be returned.
        * row index and one column index.
            If one column index is given, it will be passed to pandas' .loc. It should
            be a valid index for a GIST data row, or a row derived from the original
            data rows such as A_dens. Derived rows are defined by the Gist object's
            ._recipes dict.
        * row index and multiple column indices.
            If multiple column indices are given in a list, a new DataFrame will be
            created holding the intersection of the selected rows and columns.
            """
    def __init__(self, gist):
        """Assign a Gist object."""
        self.gist = gist

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) == 1:  # only one argument was passed
            return self.gist.data.loc[key]
        assert len(key) == 2, "Only 1D and 2D indexing is supported by _GistLocator"
        if isinstance(key[1], list):  # A list was passed
            return pd.DataFrame.from_dict({
                col: self.gist.loc[key[0], col] for col in key[1]
            })
        else:  # Assumes (!) that a single column name was passed.
            try:
                return self.gist.data.loc[key]
            except KeyError as e:
                try:
                    return self.gist._recipes[key[1]](key[0])
                except:
                    raise e

    def __setitem__(self, key, value):
        # Hopefully safe way to check if a list was passed instead of a column name.
        if isinstance(key, tuple) and len(key) >= 2 and isinstance(key[1], list):
            raise TypeError('Indexing using a list of columns is not supported by __setitem__')
        self.gist.data.loc[key] = value
        return


# Possibly will use keyword-only arguments later on
def load_gist_file(
    filename,
    n_frames=None,
    rho0=None,
    struct=None,
    strip_H=False,
    eww_ref=None,
    format=None,
    autodetect_refcol='g_O',
):
    """Return a Gist instance by loading a GIST output file from disk.

    Arguments
    ---------
    filename : str
        Cpptraj-generated gist output file.
    n_frames : int
        Number of frames that were used for the GIST calculation. If None, will
        be autodetected.
    rho0 : float
        Density of pure solvent. For TIP3P, should be 0.0329. If None, will be
        autodetected.
    struct : str
        Filename of reference structure, in a file format that is recognised by mdtraj.
    strip_H : bool
        Whether hydrogen should be stripped from struct.
    eww_ref : float
        Reference water-water interaction energy. Should be -9.533 for TIP3P.
        This has to be supplied to get useful results!!!
    format : str or None
        File-format of the GIST input file. Supported options are 'amber14',
        'amber16', or 'gigist'.

    Returns
    -------
    GistFile object.
    """
    import mdtraj as md

    if format is None:
        with open_maybe_gzipped(filename) as f:
            format = detect_gist_format(f)
    with open_maybe_gzipped(filename) as f:
        colnames = gist_colnames(format, f)

    df = pd.read_csv(
        filename,
        delim_whitespace=True,
        names=colnames,
        header=0,
        skiprows=1,
        index_col=False,
    )
    if struct is not None:
        struct_md = md.load_frame(struct, 0)
    else:
        struct_md = None
    return Gist(
        df,
        n_frames=n_frames,
        rho0=rho0,
        struct=struct_md,
        strip_H=strip_H,
        eww_ref=eww_ref,
        autodetect_refcol=autodetect_refcol,
    )


def load_dx(
    filename,
    n_frames=None,
    rho0=None,
    struct=None,
    strip_H=False,
    eww_ref=None,
    colname=None,
    file_format='dx',
):
    """Return a Gist instance by loading an OpenDX file from disk.

    Arguments
    ---------
    filename : str
        Grid data in a format recognized by gridData (usually OpenDX)
    n_frames : int
        Number of frames that were used for the GIST calculation. If None, will
        be autodetected.
    rho0 : float
        Density of pure solvent. For TIP3P, should be 0.0329. If None, will be
        autodetected.
    struct : str
        Filename of reference structure, in a file format that is recognised by mdtraj.
    strip_H : bool
        Whether hydrogen should be stripped from struct.
    eww_ref : float
        Reference water-water interaction energy. Should be -9.533 for TIP3P.
        This has to be supplied to get useful results!!!
    colname : str or None
        How the single column in the Gist object should be called. Default: the
        basename of the filename.
    file_format : str or None
        Will be passed to gridData.Grid constructor.

    Returns
    -------
    A Gist object.
    """
    import gridData as gd
    import mdtraj as md
    infile = gd.Grid(filename, file_format=file_format)
    grid = Grid(
        origin=infile.origin,
        shape=np.array([len(x) for x in infile.edges])-1,
        delta=infile.delta
    )
    if colname is None:
        colname = os.path.splitext(os.path.basename(filename))[0]
    data = pd.DataFrame.from_dict({colname: np.reshape(infile.grid, (-1))})
    if struct is not None:
        struct_md = md.load_frame(struct, 0)
    else:
        struct_md = None
    return Gist(
        data=data,
        grid=grid,
        n_frames=n_frames,
        rho0=rho0,
        struct=struct_md,
        strip_H=strip_H,
        eww_ref=eww_ref,
    )


def combine_gists(gists):
    """Combine multiple GistFile objects.

    Arguments
    ---------
    gists : list of Gist objects

    Returns
    -------
    a new Gist object.
        The grids are combined and all default columns are written to the new grid.
        This function has only been tested with grids that are correctly aligned, i.e.
        the x, y, and z offsets are multiples of the grid spacing.

    """
    if len(gists) == 1:
        return gists[0]
    new_grid = combine_grids([gist.grid for gist in gists])
    new_data = pd.DataFrame(index=np.arange(new_grid.size))
    for gist in gists:
        xyz = gist.grid.xyz(np.arange(gist.grid.size))
        indices = new_grid.flat_indices(new_grid.closest(xyz))
        for key in gist.data.keys():
            new_data.loc[indices, key] = gist.data[key]
    return Gist(
        data=new_data,
        grid=new_grid,
        struct=gists[0].struct,
        n_frames=gists[0].n_frames,
        rho0=gists[0].rho0,
        eww_ref=gists[0].eww_ref,
    )
