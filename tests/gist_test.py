#!/usr/bin/env python3
import gisttools.gist as gist
from io import StringIO
from textwrap import dedent
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pandas as pd
import pytest

def test_combine_gists():
    # with pytest.warns(RuntimeWarning):
    example1 = gist.load_gist_file('tests/example_gist_5000frames.dat', n_frames=5000, eww_ref=-9.533)
    example2 = gist.load_gist_file('tests/example_gist_5000frames.dat', n_frames=5000, eww_ref=-9.533)
    example2.grid.origin[2] += 1.
    # with pytest.warns(RuntimeWarning):
        # Warns because there is no eww_ref. Maybe I should change the behaviour to use
        # oneo the values from the input Gist objects.
    combined = gist.combine_gists([example1, example2])
    expected_shape = np.array([2, 3, 4])
    expected_z_column = np.array([-0.25, 0.25, 0.75, 1.25]*6)
    assert_array_equal(combined.grid.shape, expected_shape)
    assert_allclose(combined.grid.xyz(np.arange(combined.grid.n_voxels))[:, 2], expected_z_column)
    print(combined.data)
    print(combined.data.keys())
    assert not np.any(combined.data['Esw_dens'] == np.nan)
    return

def test_gist_colnames():
    assert len(gist.gist_colnames('amber14')) == 22
    assert len(gist.gist_colnames('amber16')) == 24
    gigist_test = dedent("""\
        GIST calculation output.
        voxel        x          y          z         population     dTSt_d(kcal/mol)  dTSt_n(kcal/mol)  dTSo_d(kcal/mol)  dTSo_n(kcal/mol)  dTSs_d(kcal/mol)  dTSs_n(kcal/mol)   Esw_d(kcal/mol)   Esw_n(kcal/mol)   Eww_d(kcal/mol)   Eww_n(kcal/mol)    dipoleX    dipoleY    dipoleZ    dipole    neighbour_d    neighbour_n    order_n    g_O    g_H
        0 -27.25 -27.25 -29.75 41 0 0 0.00224294 0.0683823 0 0 -0.00595876 -0.18167 -0.316297 -9.64321 0.00445662 -0.0110353 0.00974964 0.0153849 0.1736 5.29268 0 0.99696 1.11854
        """)
    out = gist.gist_colnames('gigist', StringIO(gigist_test))
    assert len(out) == 24
    assert out[-2:] == ['g_O', 'g_H']

def test_detect_gist_format_amber14():
    gist14_test = dedent("""\
        GIST Output, information printed per voxel
        voxel xcoord ycoord zcoord population g_O g_H dTStrans-dens(kcal/mol/A^3) dTStrans-norm(kcal/mol) dTSorient-dens(kcal/mol/A^3) dTSorient-norm(kcal/mol) Esw-dens(kcal/mol/A^3) Esw-norm(kcal/mol) Eww-dens(kcal/mol/A^3) Eww-norm-unref(kcal/mol) Dipole_x-dens(D/A^3) Dipole_y-dens(D/A^3) Dipole_z-dens(D/A^3) Dipole-dens(D/A^3) neighbor-dens(1/A^3) neighbor-norm order-norm
        0 -14.75 -14.75 -14.75 19 0.924012 1.09422 0.00143228 0.0471146 0.00528766 0.173936 -0.00161158 -0.0530126 -0.280001 -9.21056 0.00518864 -0.0214928 -0.00779186 0.023443 0.1664 5.47368 0
        """)
    assert gist.detect_gist_format(StringIO(gist14_test)) == 'amber14'
    return

def test_detect_gist_format_amber16():
    gist16_test = dedent("""\
        GIST Output, information printed per voxel
        voxel xcoord ycoord zcoord population g_O g_H dTStrans-dens(kcal/mol/A^3) dTStrans-norm(kcal/mol) dTSorient-dens(kcal/mol/A^3) dTSorient-norm(kcal/mol) dTSsix-dens(kcal/mol/A^3) dTSsix-norm (kcal/mol) Esw-dens(kcal/mol/A^3) Esw-norm(kcal/mol) Eww-dens(kcal/mol/A^3) Eww-norm-unref(kcal/mol) Dipole_x-dens(D/A^3) Dipole_y-dens(D/A^3) Dipole_z-dens(D/A^3) Dipole-dens(D/A^3) neighbor-dens(1/A^3) neighbor-norm order-norm
        0 -9.75 -9.75 -9.75 20 0.958084 1.1976 0 0 -0.0012033 -0.0376033 0 0 0 0 -0.307576 -9.61176 0.00393923 -0.0042214 0.00845889 0.0102416 0.1712 5.35 0
        """)
    assert gist.detect_gist_format(StringIO(gist16_test)) == 'amber16'
    return

def test_detect_gist_format_gigist():
    gigist_test = dedent("""\
        GIST calculation output.
        voxel        x          y          z         population     dTSt_d(kcal/mol)  dTSt_n(kcal/mol)  dTSo_d(kcal/mol)  dTSo_n(kcal/mol)  dTSs_d(kcal/mol)  dTSs_n(kcal/mol)   Esw_d(kcal/mol)   Esw_n(kcal/mol)   Eww_d(kcal/mol)   Eww_n(kcal/mol)    dipoleX    dipoleY    dipoleZ    dipole    neighbour_d    neighbour_n    order_n    g_O    g_H
        0 -27.25 -27.25 -29.75 41 0 0 0.00224294 0.0683823 0 0 -0.00595876 -0.18167 -0.316297 -9.64321 0.00445662 -0.0110353 0.00974964 0.0153849 0.1736 5.29268 0 0.99696 1.11854
        """)
    assert gist.detect_gist_format(StringIO(gigist_test)) == 'gigist'
    return

def test_integrate_around():
    # voxel_volume is 1/4 and n_frames is 2500 because I halved the spacing in one
    # direction of the example file. The "magic" numbers in expected are the sums of
    # the respective voxels in the _dens columns.
    with pytest.warns(RuntimeWarning):
        example = gist.load_gist_file('tests/example_gist_5000frames.dat', n_frames=2500, eww_ref=-0)
    integrals = example.integrate_around(
        ["A", "Esw"],
        centers=[[-0.5, -0.5, -0.25]],
        rmax=0.51,
    )
    expected = pd.DataFrame.from_dict({"A_dens": [-0.971784722/4], "Esw_dens": [0.008096718/4]})
    assert np.allclose(integrals.values, expected.values)

def test_distance_to_spheres():
    gf = gist.load_gist_file('tests/example_gist_5000frames.dat', eww_ref=-9.533)
    # In this case, the input coordinates are out of the grid, so there are not
    # much voxels found.
    ind, center, dist = gf.distance_to_spheres(
        [[1.5, 0, 0], [-1.5, 0, 0]],
        rmax=0.6,
        atomic_radii=[0.3, 0.5]
    )
    assert np.all(ind == np.array([2, 3]))
    assert np.all(center == np.array([1, 1]))
    assert np.allclose(
        dist + 0.5,
        np.sqrt(np.sum(np.array([1., 0., 0.25])**2))
    )
print("Running doctests ...")
import doctest
doctest.testmod(gist)

print("Finished running doctests.")