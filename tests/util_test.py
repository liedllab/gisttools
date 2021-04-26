#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip
from gisttools.utils import open_maybe_gzipped, distance_weight, cartesian_product, scale_to
import os
import numpy as np

def test_open_maybe_zipped():
    with open('__test_not_zipped.txt', 'wt') as f:
        f.write("Hello World")
    with gzip.open('__test_gzipped.gz', 'wt') as f:
        f.write("Hello World")
    with open_maybe_gzipped('__test_not_zipped.txt') as f:
        assert f.read() == "Hello World"
    with open_maybe_gzipped('__test_gzipped.gz') as f:
        assert f.read() == "Hello World"
    os.remove('__test_not_zipped.txt')
    os.remove('__test_gzipped.gz')


def test_distance_weight():
    test_arr = np.arange(10)
    weighted_linear = distance_weight(test_arr, method='piecewise_linear', constant=3., cutoff=5.)
    weighted_gaussian = distance_weight(test_arr, method='gaussian', sigma=3.)
    weighted_logistic = distance_weight(test_arr, method='logistic', k=3., x0=3.)
    weighted_none = distance_weight(test_arr, method=None)
    square = lambda x: x**2
    weighted_square = distance_weight(test_arr, method=square)
    assert np.allclose(weighted_linear, [1., 1., 1., 1., 0.5, 0., 0., 0., 0., 0.])
    assert np.allclose(weighted_gaussian, np.exp(-test_arr**2 / (2 * 3**2)))
    assert np.allclose(weighted_logistic, 1 / (1 + np.exp(3*(test_arr - 3.))))
    assert np.allclose(weighted_square, [0, 1, 4, 9, 16, 25, 36, 49, 64, 81])
    assert np.allclose(weighted_none, np.ones(test_arr.shape))

def test_cartesian_product():
    a = np.array([-1, 1])
    b = np.array([0, 2, 4])
    c = np.array([3, 6])
    ab = cartesian_product(a, b)
    abc = cartesian_product(a, b, c)
    ab_expected = np.array([[-1, 0], [-1, 2], [-1, 4], [1, 0], [1, 2], [1, 4]])
    abc_expected = np.hstack(
        (np.repeat(ab_expected, 2, axis=0), np.tile(c, 6).reshape(-1, 1))
    )
    np.testing.assert_array_equal(ab, ab_expected)
    np.testing.assert_array_equal(abc, abc_expected)

def test_scale_to():
    a = np.array([-40, 50])
    scale = [3, 4]
    assert np.allclose(scale_to(a, scale), scale)
