#!/usr/bin/env python3

import gisttools.grid as grid
import numpy as np
import pytest
import numba

example_grid_100 = grid.Grid(-50., 101, 1)
example_grid_long = grid.Grid(0, [1000, 1, 1], 1.)

def test_grid_closest():
    gd = grid.Grid(-50, 101, 1)
    assert np.allclose(gd.closest([.4, .9, .3]), [[50, 51, 50]])
    oob = np.array([51., 0., 0.])
    with pytest.raises(ValueError):
        gd.closest(oob)
    assert np.allclose(gd.closest(oob, out_of_bounds='closest'), [[100, 50, 50]])
    assert np.allclose(gd.closest(oob, out_of_bounds='dummy'), [[-1, -1, -1]])
    assert np.allclose(gd.closest(oob, out_of_bounds='ignore'), [[101, 50, 50]])

def test_grid_coarse_grain():
    gd = grid.Grid(-45, 102, 1)
    for i in (1, 2, 3):
        cg = gd.coarse_grain(i)
        # same grid volume
        assert np.isclose(gd.size * gd.voxel_volume, cg.size * cg.voxel_volume)
        # same center
        assert np.allclose(gd.origin + gd.xyzmax, cg.origin + cg.xyzmax)
        # shifted origin
        assert np.allclose(gd.origin - gd.delta / 2, cg.origin - cg.delta / 2)
    with pytest.raises(ValueError):
        gd.coarse_grain(4)

def test_init_checks_int_for_shape():
    with pytest.raises(TypeError):
        wrong_shape = 5.
        grid.Grid(0, wrong_shape, 1)

def test_surrounding_box_with_long_example_with_extra_space():
    for boxlen in range(10):
        # Adding 0.01 to avoid the border being exacly on a voxel center. 
        x, y, z = example_grid_long.surrounding_box([0, 0, 0], boxlen + 0.01)
        np.testing.assert_allclose(x, np.arange(boxlen + 1))
        np.testing.assert_allclose(y, [0])
        np.testing.assert_allclose(z, [0])

def test_surrounding_box_with_long_example_no_extra_space():
    for boxlen in range(10):
        # This time not adding the 0.01. Therefore, rounding errors might
        # have an impact, but it seems they don't. 
        x, y, z = example_grid_long.surrounding_box([0, 0, 0], boxlen)
        np.testing.assert_allclose(x, np.arange(boxlen + 1))
        np.testing.assert_allclose(y, [0])
        np.testing.assert_allclose(z, [0])

def test_surrounding_sphere_with_long_example():
    for boxlen in range(10):
        # This time not adding the 0.01. Therefore, rounding errors might
        # have an impact, but it seems they don't. 
        x, y, z = example_grid_long.surrounding_box([0, 0, 0], boxlen)
        np.testing.assert_allclose(x, np.arange(boxlen + 1))
        np.testing.assert_allclose(y, [0])
        np.testing.assert_allclose(z, [0])

def test_surrounding_sphere_np_and_numba_equal():
    test_points = (np.random.random(3000) * 120 - 60).reshape(1000, 3)
    test_radii = np.random.random(1000) * 10
    for test_point, test_radius in zip(test_points, test_radii):
        numba_result = example_grid_100.surrounding_sphere(test_point, test_radius)
        np_result = example_grid_100.surrounding_sphere_np(test_point, test_radius)
        np.testing.assert_array_equal(numba_result[0], np_result[0])
        np.testing.assert_allclose(numba_result[1], np_result[1])

def test_assign():
    gd = grid.Grid(0, 4, 1)
    assert gd._assign([0, 1, 0])[0] == 4
    assert gd._assign([1, 0, 0])[0] == 16
    assert gd._assign([1, 3.6, 0])[0] == -1
    assert gd._assign([-0.6, 0, 0])[0] == -1

def test_numba_assign():
    gd = grid.Grid(0, 4, 1)
    assert gd.assign([0, 1, 0])[0] == 4
    assert gd.assign([1, 0, 0])[0] == 16
    assert gd.assign([0.49, 0, 0])[0] == 0
    assert gd.assign([0.51, 0, 0])[0] == 16
    assert gd.assign([1, 3.6, 0])[0] == -1
    assert gd.assign([-0.6, 0, 0])[0] == -1
    gd2 = grid.Grid(0, [2, 3, 4], 1)
    assert np.all(gd2.shape == [2, 3, 4])
    assert gd2.assign([1, 2, 3])[0] == 23
    assert gd2.assign([2, 2, 3]) == -1
    assert gd2.assign([1, 3, 3]) == -1
    assert gd2.assign([1, 2, 4]) == -1
    assert gd2.assign([1, 0, 0])[0] == 12
    assert gd2.assign([0, 1, 0])[0] == 4

def test_ensure_int():
    a = np.array([5.0, 4.0, -2.0])
    out_a = grid.ensure_int(a)
    assert np.issubdtype(out_a.dtype, np.integer)
    np.testing.assert_array_equal(a, out_a)

    b = np.array([5.0, 4.1, -2.0])
    with pytest.raises(ValueError, match="Non-int value"):
        grid.ensure_int(b)


def test_numba_int_div_returns_float():
    @numba.njit()
    def divide(a, b):
        return a / b
    out = divide(3, 2)
    assert out == pytest.approx(1.5)
    out_b = divide(np.array([1, 2, 3]), np.array([2, 2, 2]))
    np.testing.assert_allclose(out_b, np.array([0.5, 1, 1.5]))

def test_centered_grid():
    gd = grid.Grid.centered(0, 11, 1.)
    assert np.allclose(-gd.origin, gd.xyzmax)
