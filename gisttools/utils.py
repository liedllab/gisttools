import numpy as np
import sys
import random
import tempfile
import string
import os
import gzip
from typing import Tuple


def open_maybe_gzipped(filename, mode='rt'):
    """Open a file with gzip if it is compressed, and directly if it is not.

    Be careful when changing mode, because the default for gzip.open is bytes and the
    default for open is text mode.
    """
    # magic_number = open(filename, 'rb').read(2)
    # return gzip.open(filename, mode) if magic_number == b'\x1f\x8b' else open(filename, mode)
    opener = gzip.open if filename.endswith('.gz') else open
    return opener(filename, mode)


def preview_dataset(struct, dataset, crange='scale', view=None, cmap="YlGnBu", reverse=True):
    """Create a colored surface in a nglview widget.  This function writes a
    temporary .pdb file (GISTFILE_TEMP.pdb) in the current folder, with the
    dataset in the bfactor.  It has less control of the coloring than
    preview_dataset_slow.

    Parameters
    ----------
    dataset : numpy array
        Numpy array with numbers to color the surface from.  Could be a
        column of the output of schauperl_projection.
    view : nglview widget
        If None, a new widget is created.
    crange : 'scale' or tuple
        If crange is a tuple, it defines the minimum and maximum value for the
        coloring. If is is 'scale' (the default), scales the dataset to the
        limits for PDB B-Factors (-9.99 to 99.99)
    cmap : str
        Colormap name. Will be passed to nglview.
    reverse : bool
        Whether to reverse the colormap.

    Returns
    -------
    updated_view : nglview.NGLWidget

    """
    import nglview as _nv
    if view is None:
        view = _nv.NGLWidget()
    if crange == 'scale':
        crange = (-9.99, 99.99)
        dataset = scale_to(dataset, crange)
    else:
        if crange[0] < -9.99 or crange[1] > 99.99:
            raise ValueError("crange is out of the PDB limitations for B-Factors")
        dataset = np.clip(dataset, -9.99, 99.99)
    with tempfile.TemporaryDirectory() as tmp:
        filename = os.path.join(tmp, 'tmp.pdb')
        struct.save_pdb(filename, bfactors=dataset)
        view.add_component(filename)
    view.set_representations(representations=[{
        'type': 'surface',
        'params': {
            'colorScheme': 'bfactor',
            'colorScale': cmap,
            'colorReverse': reverse,
            'colorDomain': crange,
        }
    }])
    return view


def scale_to(a, domain: Tuple[float]) -> np.ndarray:
    """Scale array a to fit into scale."""
    a = np.asarray(a)
    d_min, d_max = domain
    a_min, a_max = np.min(a), np.max(a)
    scale = (d_max - d_min) / (a_max - a_min)
    offset = -a_min * scale + d_min
    return a * scale + offset


class ProgressPrinter():

    def __init__(self, progress_string, n_steps):
        self.progress_string = progress_string
        self.previous_output = ''
        self.n_steps = n_steps
        self.step = 0
        self.finalized = False
        return None

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.finalize()

    def tick(self, number=1):
        if self.finalized:
            return
        self.step += number
        percentage = self.step / self.n_steps * 100.
        output = self.progress_string.format(percentage)
        # Only print if the string changed. Printing takes time...
        if output != self.previous_output:
            sys.stdout.write(output + '\r')
            sys.stdout.flush()
            self.previous_output = output
        if self.step >= self.n_steps:
            self.finalize()
        return None

    def finalize(self):
        if self.finalized:
            return
        self.finalized = True
        sys.stdout.write('\n')
        sys.stdout.flush()
        return None


def cartesian_product(*xi):
    """Given multiple 1D arrays, return the cartesian product.

    Parameters
    ----------
    x1, x2,..., xn : array-like, 1D

    Returns
    -------
    cartesian_product : np.ndarray, shape=(N, n)
        n is the number of input arrays, N is the product of their lengths.

    Examples
    --------
    >>> cartesian_product([1, 2])
    array([[1],
           [2]])
    >>> cartesian_product([1, 2], [3, 4, 5])
    array([[1, 3],
           [1, 4],
           [1, 5],
           [2, 3],
           [2, 4],
           [2, 5]])
    """
    ndim = len(xi)
    return np.stack(np.meshgrid(*xi, indexing='ij'), -1).reshape(-1, ndim)


def weight_piecewise_linear(a, constant, cutoff):
    """Weighting function which is 1 at first and then linearly decreases to 0.

    Parameters
    ----------
    constant : float
        For all values < constant, the weight will be 1.
    cutoff : float
        For all values > cutoff, the weight will be 0. Between constant and cutoff, the
        weight will decrease linearly.

    Examples
    --------
    >>> weight_piecewise_linear([0, 2, 3, 4, 5], 2, 4)
    array([1. , 1. , 0.5, 0. , 0. ])
    """
    a = np.asarray(a)
    assert constant < cutoff, \
        f'constant must be < cutoff, but they are {constant} and {cutoff}.'
    k = -1. / (cutoff - constant)
    d = -(cutoff * k)
    return np.minimum(
        np.maximum(a * k + d, 0.),
        1.
    )


def weight_gaussian(a, sigma):
    """Gaussian weighting function with height 1 and given sigma, centered at 0.

    Examples
    --------
    >>> weight_gaussian([0, 1, 2, 3, 4], 2) # doctest: +ELLIPSIS
    array([1. ..., 0.882..., 0.606..., 0.324..., 0.135...])
    """
    a = np.asarray(a)
    return np.exp(-a**2 / (2 * sigma**2))


def weight_logistic(a, k, x0):
    """Logistic weighting function that transitions from 1 to 0.

    See https://en.wikipedia.org/wiki/Logistic_function. Compared to the definition
    here, this function is mirrored, i.e., the values transition from 1 to 0 and not
    the other way around. The same could be achieved by multiplying k by -1.

    Parameters
    ----------
    k : float
        Steepness parameter of the logistic curve.
    x0 : float
        Turning point of the logistic curve

    Returns
    -------
    weights : np.ndarray, shape=a.shape

    Examples
    --------
    >>> weight_logistic([0, 1, 2, 3, 4], 2, 2) # doctest: +ELLIPSIS
    array([0.982..., 0.880..., 0.5 ..., 0.119..., 0.017...])
    """
    a = np.asarray(a)
    return 1. / (1. + np.exp(k * (a - x0)))


weight_functions = {
    'piecewise_linear': weight_piecewise_linear,
    'gaussian': weight_gaussian,
    'logistic': weight_logistic,
}


def distance_weight(a, method, **kwargs):
    """Calculate weights for a using the specified weighting method

    Parameters
    ----------
    a : np.ndarray
        Distances to calculate weights from.
    method : str or callable
        Can be 'piecewise_linear', 'gaussian', 'logistic', None, or a callable. For
        instance, 'gaussian' will call the weight_gaussian function with a and all
        kwargs. If None, returns 1 for each weight.
    **kwargs : Will be passed on to the weighting function.

    Returns
    -------
    weights : np.ndarray, shape=a.shape
    """
    a = np.asarray(a)
    if method is None:
        return np.ones_like(a)
    try:
        weight_function = weight_functions[method]
    except KeyError:
        if callable(method):
            weight_function = method
        else:
            raise KeyError('method must be "piecewise_linear", "gaussian", "logistic", None, or a callable.')
    return weight_function(a, **kwargs)
