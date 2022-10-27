#!/usr/bin/env python3
import gisttools.gist as gist
import gisttools.grid as grid
from io import StringIO
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pandas as pd
import pytest

CPPTRAJ_V4_OUT = """GIST Output v4 spacing=0.5000 center=0.000000,0.000000,0.000000 dims=80,80,80 
voxel xcoord ycoord zcoord population g_O g_H dTStrans-dens(kcal/mol/A^3) \
dTStrans-norm(kcal/mol) dTSorient-dens(kcal/mol/A^3) dTSorient-norm(kcal/mol) \
dTSsix-dens(kcal/mol/A^3) dTSsix-norm(kcal/mol) Esw-dens(kcal/mol/A^3) \
Esw-norm(kcal/mol) Eww-dens(kcal/mol/A^3) Eww-norm-unref(kcal/mol) \
Dipole_x-dens(D/A^3) Dipole_y-dens(D/A^3) Dipole_z-dens(D/A^3) \
Dipole-dens(D/A^3) neighbor-dens(1/A^3) neighbor-norm order-norm
"""

GIGIST_OUTPUT = """\
GIST calculation output.
voxel        x          y          z         population     dTSt_d(kcal/mol)  \
dTSt_n(kcal/mol)  dTSo_d(kcal/mol)  dTSo_n(kcal/mol)  dTSs_d(kcal/mol)  \
dTSs_n(kcal/mol)   Esw_d(kcal/mol)   Esw_n(kcal/mol)   Eww_d(kcal/mol)   \
Eww_n(kcal/mol)    dipoleX    dipoleY    dipoleZ    dipole    neighbour_d    \
neighbour_n    order_n    g_O    g_H
0 -27.25 -27.25 -29.75 41 0 0 0.00224294 0.0683823 0 0 -0.00595876 -0.18167 \
-0.316297 -9.64321 0.00445662 -0.0110353 0.00974964 0.0153849 0.1736 5.29268 0 0.99696 1.11854
"""

PME_GIST_OUTPUT = """ \
GIST Output v3 spacing=0.5000 center=0.000000,0.000000,0.000000 dims=80,80,80 
voxel xcoord ycoord zcoord population g_O g_H dTStrans-dens(kcal/mol/A^3) \
dTStrans-norm(kcal/mol) dTSorient-dens(kcal/mol/A^3) dTSorient-norm(kcal/mol) \
dTSsix-dens(kcal/mol/A^3) dTSsix-norm(kcal/mol) Esw-dens(kcal/mol/A^3) \
Esw-norm(kcal/mol) Eww-dens(kcal/mol/A^3) Eww-norm-unref(kcal/mol) \
PME-dens(kcal/mol/A^3) PME-norm(kcal/mol) Dipole_x-dens(D/A^3) \
Dipole_y-dens(D/A^3) Dipole_z-dens(D/A^3) Dipole-dens(D/A^3) \
neighbor-dens(1/A^3) neighbor-norm order-norm
0 -19.75 -19.75 -19.75 42 1.02206 1.1194 0 0 0.00677163 0.201536 0 0 -1.67165e-06 \
-4.97515e-05 -0.323573 -9.63014 -0.323574 -9.63017 0.0044711 0.00969178 -0.0037346 \
0.0113079 0.1816 5.40476 0 
"""

GIST16_OUTPUT = """\
GIST Output, information printed per voxel
voxel xcoord ycoord zcoord population g_O g_H dTStrans-dens(kcal/mol/A^3) \
dTStrans-norm(kcal/mol) dTSorient-dens(kcal/mol/A^3) dTSorient-norm(kcal/mol) \
dTSsix-dens(kcal/mol/A^3) dTSsix-norm (kcal/mol) Esw-dens(kcal/mol/A^3) \
Esw-norm(kcal/mol) Eww-dens(kcal/mol/A^3) Eww-norm-unref(kcal/mol) \
Dipole_x-dens(D/A^3) Dipole_y-dens(D/A^3) Dipole_z-dens(D/A^3) \
Dipole-dens(D/A^3) neighbor-dens(1/A^3) neighbor-norm order-norm
0 -9.75 -9.75 -9.75 20 0.958084 1.1976 0 0 -0.0012033 -0.0376033 0 0 0 0 -0.307576 \
-9.61176 0.00393923 -0.0042214 0.00845889 0.0102416 0.1712 5.35 0
"""

GIST14_OUTPUT = """\
GIST Output, information printed per voxel
voxel xcoord ycoord zcoord population g_O g_H dTStrans-dens(kcal/mol/A^3) \
dTStrans-norm(kcal/mol) dTSorient-dens(kcal/mol/A^3) dTSorient-norm(kcal/mol) \
Esw-dens(kcal/mol/A^3) Esw-norm(kcal/mol) Eww-dens(kcal/mol/A^3) \
Eww-norm-unref(kcal/mol) Dipole_x-dens(D/A^3) Dipole_y-dens(D/A^3) \
Dipole_z-dens(D/A^3) Dipole-dens(D/A^3) neighbor-dens(1/A^3) neighbor-norm order-norm
0 -14.75 -14.75 -14.75 19 0.924012 1.09422 0.00143228 0.0471146 0.00528766 0.173936 -0.00161158 \
-0.0530126 -0.280001 -9.21056 0.00518864 -0.0214928 -0.00779186 0.023443 0.1664 5.47368 0
"""

cols_wo_pme = ('voxel x y z population g_O g_H dTStrans_dens dTStrans_norm'
    ' dTSorient_dens dTSorient_norm dTSsix_dens dTSsix_norm Esw_dens Esw_norm'
    ' Eww_unref_dens Eww_unref_norm Dipole_x_dens Dipole_y_dens Dipole_z_dens'
    ' Dipole_dens neighbor_dens neighbor_norm order_norm').split()

cols_w_pme = ('voxel x y z population g_O g_H dTStrans_dens dTStrans_norm'
    ' dTSorient_dens dTSorient_norm dTSsix_dens dTSsix_norm Esw_dens Esw_norm'
    ' Eww_unref_dens Eww_unref_norm PME_dens PME_norm Dipole_x_dens Dipole_y_dens'
    ' Dipole_z_dens Dipole_dens neighbor_dens neighbor_norm order_norm').split()

@pytest.fixture
def small_outfile():
    return gist.load_gist_file('tests/example_gist_5000frames.dat', n_frames=5000, eww_ref=-9.533)

@pytest.fixture
def outfile_x_stretched():
    return gist.load_gist_file('tests/example_gist_5000frames_x_stretched.dat', n_frames=2500, eww_ref=-9.533)

@pytest.fixture
def outfile_x_stretched_shifted_origin():
    gf = gist.load_gist_file('tests/example_gist_5000frames_x_stretched.dat', n_frames=5000, eww_ref=-9.533)
    gf.grid.origin[2] += 1.
    return gf

@pytest.fixture
def dummy_gist():
    gd = grid.Grid.centered(0, 10, 0.5)
    np.random.seed(0)
    pop = np.random.randint(0, 20, 1000)
    values = np.random.random(1000)
    values[pop == 0] = 0.
    data = pd.DataFrame({'population': pop, 'val_dens': values/gd.voxel_volume})
    class dummy_traj:
        xyz = np.array([[0, 0, 0]]) / 10
    gf = gist.Gist(data, grid=gd, n_frames=1, struct=dummy_traj)
    gf['val_norm'] = gf.dens2norm(gf['val_dens'])
    return gf

def test_combine_gists(outfile_x_stretched, outfile_x_stretched_shifted_origin):
    combined = gist.combine_gists([outfile_x_stretched, outfile_x_stretched_shifted_origin])
    expected_shape = np.array([2, 3, 4])
    expected_z_column = np.array([-0.25, 0.25, 0.75, 1.25]*6)
    assert_array_equal(combined.grid.shape, expected_shape)
    assert_allclose(combined.grid.xyz(np.arange(combined.grid.size))[:, 2], expected_z_column)
    print(combined.data)
    print(combined.data.keys())
    assert not np.any(combined.data['Esw_dens'] == np.nan)
    return

def test_gist_colnames_v4():
    out = gist.gist_colnames('v4', StringIO(CPPTRAJ_V4_OUT))
    assert len(out) == 24
    assert out == cols_wo_pme

def test_gist_colnames():
    assert len(gist.gist_colnames('amber14')) == 22
    assert gist.gist_colnames('amber16') == cols_wo_pme
    assert gist.gist_colnames('pme') == cols_w_pme
    out = gist.gist_colnames('gigist', StringIO(GIGIST_OUTPUT))
    assert len(out) == 24
    assert out[-2:] == ['g_O', 'g_H']

def test_detect_gist_format_amber14():
    assert gist.detect_gist_format(StringIO(GIST14_OUTPUT)) == 'amber14'
    return

def test_detect_gist_format_amber16():
    assert gist.detect_gist_format(StringIO(GIST16_OUTPUT)) == 'amber16'
    return

def test_detect_gist_format_gigist():
    assert gist.detect_gist_format(StringIO(GIGIST_OUTPUT)) == 'gigist'
    return

def test_detect_gist_format_pme():
    assert gist.detect_gist_format(StringIO(PME_GIST_OUTPUT)) == 'pme'
    return

def test_load_without_refdens_warns():
    with pytest.warns(RuntimeWarning):
        gist.load_gist_file('tests/example_gist_5000frames.dat', n_frames=2500)

def test_integrate_around(outfile_x_stretched):
    # voxel_volume is 1/4 and n_frames is 2500 because I doubled the spacing in
    # the x direction of the example file. The "magic" numbers in expected are
    # the sums of the respective voxels in the _dens columns.
    outfile_x_stretched.eww_ref = 0.
    integrals = outfile_x_stretched.integrate_around(
        ["A_dens", "Esw_dens"],
        centers=[[-0.5, -0.5, -0.25]],
        rmax=0.51,
    )
    expected = pd.DataFrame.from_dict({"A_dens": [-0.971784722/4], "Esw_dens": [0.008096718/4]})
    assert np.allclose(integrals.values, expected.values)

def test_dens2norm(dummy_gist):
    assert np.allclose(
        dummy_gist.dens2norm(dummy_gist['val_dens']),
        dummy_gist['val_norm']
    )
    index_lst = [0, 10, 40]
    index_arr = np.array(index_lst)
    index_series = pd.Series(np.full(dummy_gist.grid.size, False, dtype=bool))
    index_series.loc[index_arr] = True
    for index in [index_lst, index_arr, index_series]:
        print(index)
        assert np.allclose(
            dummy_gist.dens2norm(dummy_gist.loc[index, 'val_dens'], index=index),
            dummy_gist.loc[index, 'val_norm']
        )
    with pytest.warns(UserWarning):
        dummy_gist.loc[0, 'population'] = 0
        dummy_gist.dens2norm(dummy_gist['val_dens'])

def test_distance_to_spheres(outfile_x_stretched):
    # In this case, the input coordinates are out of the grid, so there are not
    # much voxels found.
    ind, center, dist = outfile_x_stretched.distance_to_spheres(
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

@pytest.fixture
def coords_3x3x100():
    xyz = np.stack((
        np.repeat([-1, 0, 1], 300),
        np.tile(np.repeat([-1, 0, 1], 100), 3),
        np.tile(np.arange(100), 9)
    )).T
    return xyz

def test_3x3x100_coords(coords_3x3x100):
    np.testing.assert_array_equal(coords_3x3x100[:2], [[-1, -1, 0], [-1, -1, 1]])
    np.testing.assert_array_equal(coords_3x3x100[-2:], [[1, 1, 98], [1, 1, 99]])

def test_gist_from_dataframe(coords_3x3x100):
    gf = gist.Gist(pd.DataFrame({
        'x': coords_3x3x100[:, 0],
        'y': coords_3x3x100[:, 1],
        'z': coords_3x3x100[:, 2],
        'TESTCOL': np.random.random(len(coords_3x3x100)),
    }))
    np.testing.assert_allclose(gf.grid.origin, [-1, -1, 0])
    np.testing.assert_allclose(gf.grid.shape, [3, 3, 100])
    np.testing.assert_allclose(gf.grid.delta, [1, 1, 1])

def test_projection_nearest_no_weight(coords_3x3x100):
    class mock_traj:
        """Used as dummy object"""
    mock_traj.xyz = np.array([[
        [-1, -1, 0],
        [-1,  0, 0],
        [-1,  1, 0],
        [ 0, -1, 0],
        [ 0,  0, 0],
        [ 0,  1, 0],
        [ 1, -1, 0],
        [ 1,  0, 0],
        [ 1,  1, 0],
    ]]) * 0.1  # mdtraj calculates in nm.
    gf = gist.Gist(pd.DataFrame({
        'x': coords_3x3x100[:, 0],
        'y': coords_3x3x100[:, 1],
        'z': coords_3x3x100[:, 2],
        'TEST_dens': np.ones(len(coords_3x3x100)),
        'population': np.ones(len(coords_3x3x100)),
    }), struct=mock_traj, rho0=1., n_frames=1)
    for dist in range(4):
        proj = gf.projection_nearest(['TEST_dens', 'voxels'], rmax=dist)
        np.testing.assert_allclose(proj.TEST_dens.values, dist + 1)

def test_rdf(dummy_gist):
    gf = dummy_gist
    pop = gf['population']
    # rmax = 10 => contains the whole grid
    bins, rdf_none = gf.rdf('population', rmax=10, bins=1, normalize='none')
    bins, rdf_dens = gf.rdf('population', rmax=10, bins=1, normalize='dens')
    bins, rdf_norm = gf.rdf('population', rmax=10, bins=10, normalize='norm')
    assert rdf_norm.shape == (10,)
    assert np.isclose(rdf_none[0], pop.sum())
    print(np.sum(gf['population']))
    assert np.isclose(rdf_dens[0], np.sum(gf['population']) / (gf.grid.size * gf.grid.voxel_volume))
    assert (~np.isnan(rdf_norm)).sum() > 3
    np.testing.assert_allclose(rdf_norm[~np.isnan(rdf_norm)], 1)

    not_origin = [1, 0, 0]
    integral = gf.integrate_around('val_dens', rmax=2, centers=not_origin)
    assert np.isclose(gf.rdf('val_dens', rmax=2, bins=1, centers=not_origin, normalize='none')[1][0], integral)

def test_detect_reference_value(dummy_gist):
    expected_refval = 5.
    n_solvents = dummy_gist['population'] / dummy_gist.n_frames
    normed = np.full(dummy_gist.grid.size, expected_refval)
    ind, _, _ = dummy_gist.distance_to_spheres(rmax=1)
    normed[ind] = 2*5.
    dens = normed * n_solvents / dummy_gist.grid.voxel_volume
    dummy_gist['to_ref_norm'] = normed
    dummy_gist['to_ref_dens'] = dens
    refval = dummy_gist.detect_reference_value('to_ref_norm', dlim=(2, 3.5))
    assert isinstance(refval, float)
    assert refval == pytest.approx(expected_refval)

def test_get_total(dummy_gist):
    vvox = dummy_gist.grid.voxel_volume
    for voxels in [None, [0, 1, 32, 345]]:
        indices = voxels or slice(None)
        expected = dummy_gist.loc[indices, 'val_dens'] * vvox
        for type in ['dens', 'norm']:
            tot = dummy_gist.get_total(f'val_{type}', index=voxels)
            np.testing.assert_allclose(tot, expected)
        expected_voxels = dummy_gist.grid.size if voxels is None else len(voxels)
        n_vox = dummy_gist.get_total('voxels', index=indices)
        assert n_vox.sum() == expected_voxels

def test_get_total_referenced(small_outfile):
    col = "dTStrans"
    ref = 1.
    dens = small_outfile[col + "_dens"]
    norm = small_outfile[col + "_norm"]
    dens_ref_tot = small_outfile.get_total_referenced(col + "_dens", ref)
    norm_ref_tot = small_outfile.get_total_referenced(col + "_norm", ref)
    assert_allclose(dens_ref_tot, norm_ref_tot, rtol=1e-6)
    waters = small_outfile['population'] / small_outfile.n_frames
    assert_allclose((norm - ref) * waters, norm_ref_tot, rtol=1e-6)

def test_get_referenced(small_outfile):
    col = "dTStrans"
    ref = 1.
    tot = small_outfile.get_total_referenced(col + "_dens", ref)
    dens = small_outfile.get_referenced(col + "_dens", ref)
    norm = small_outfile.get_referenced(col + "_norm", ref)
    assert_allclose(dens, tot / small_outfile.grid.voxel_volume)
    assert_allclose(norm, tot / small_outfile['population'] * small_outfile.n_frames, rtol=1e-6)

def test_guess_column_type():
    assert gist.as_gist_quantity('something_dens').normalization == gist.VoxelNormalization.dens
    assert gist.as_gist_quantity('something_norm').normalization == gist.VoxelNormalization.norm
    with pytest.raises(ValueError):
        gist.as_gist_quantity('something')

print("Running doctests ...")
import doctest
doctest.testmod(gist)

print("Finished running doctests.")
