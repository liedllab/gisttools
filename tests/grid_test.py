#!/usr/bin/env python3

import gisttools.grid as grid
import numpy as np

example_grid_100 = grid.Grid(-50., 101, 1)
example_grid_long = grid.Grid(0, [1000, 1, 1], 1.)

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

print("Running doctests ...")
import doctest
doctest.testmod(grid)

print("Finished running doctests.")
