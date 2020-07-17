import numpy as np
import sys
import random
import tempfile
import string
import os
import gzip


def open_maybe_gzipped(filename, mode='rt'):
    """Open a file with gzip if it is compressed, and directly if it is not.

    Be careful when changing mode, because the default for gzip.open is bytes and the
    default for open is text mode.
    """
    # magic_number = open(filename, 'rb').read(2)
    # return gzip.open(filename, mode) if magic_number == b'\x1f\x8b' else open(filename, mode)
    opener = gzip.open if filename.endswith('.gz') else open
    return opener(filename, mode)


def preview_dataset_slow(struct, dataset, view=None, crange=None, cmap='coolwarm', mode='atom', indices=None, label='preview'):
    """Create a colored surface in a nglview widget.  This function creates a
    new _ColorScheme object.  For large systems, this takes a while.

    Parameters
    ----------
    dataset : numpy array
        Numpy array with numbers to color the surface from.  Could be a
        column of the output of schauperl_projection.
    view : nglview widget
        If None, a new widget is created.
    crange : tuple
        Minimum and maximum value for the coloring.  Default: smallest and
        largest values of the dataset.
    cmap : str
        String representation of a matplotlib color map, that is used to define
        the surface colors.
    label : str
        Label for the nglview _ColorScheme

    Returns
    -------
    updated_view : nglview.NGLWidget

    """
    import nglview as _nv
    from matplotlib.cm import ScalarMappable
    if crange is None:
        cmin = np.min(dataset)
        cmax = np.max(dataset)
    else:
        cmin, cmax = crange
    if view is None:
        view = _nv.show_mdtraj(struct, gui=True)
    colors = ScalarMappable(cmap=cmap)
    colors.set_clim(cmin, cmax)
    rgbvals = colors.to_rgba(dataset)
    if mode == 'atom':
        if indices is None:
            indices = range(dataset.shape[0])
        rgbtext = [
            [
                '#{:02X}{:02X}{:02X}'.format(int(rgbvals[i, 0]*256),
                                             int(rgbvals[i, 1]*256),
                                             int(rgbvals[i, 2]*256)),
                '@{}'.format(index)
            ] for i, index in enumerate(indices)
        ]
    elif mode == 'residue':
        if indices is None:
            indices = range(struct.top.n_residues)
        rgbtext = [
            [
                '#{:02X}{:02X}{:02X}'.format(int(rgbvals[i, 0]*256),
                                             int(rgbvals[i, 1]*256),
                                             int(rgbvals[i, 2]*256)),
                '{}'.format(index)
            ] for i, index in enumerate(indices)
        ]
    else:
        raise ValueError('Mode is {}, should be \'atom\' or \'residue\''.format(mode))
    rgbtext.append(['white', '*'])
    scheme = _nv.color._ColorScheme(rgbtext, label=label)
    view.add_surface(color=scheme)
    return view


def preview_dataset_fast(struct, dataset, maximal_crange=None, view=None):
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
    maximal_crange : tuple
        Minimum and maximum value for the coloring.  Default: smallest and
        largest values of the dataset.  Note: the actual color range can still
        be smaller, but not larger than maximal_crange.

    Returns
    -------
    updated_view : nglview.NGLWidget

    """
    import nglview as _nv
    if view is None:
        view = _nv.NGLWidget(gui=True)

    if maximal_crange is not None:
        cut_min, cut_max = maximal_crange
        dataset = np.maximum(dataset, cut_min)
        dataset = np.minimum(dataset, cut_max)

    minval, maxval = np.min(dataset), np.max(dataset)
    scale1, scale2 = 1, 1
    if minval < -9.99:
        scale1 = minval / -9.99
    if maxval > 99.99:
        scale2 = maxval / 99.99
    scale = np.max((scale1, scale2))
    dataset = dataset / scale
    print("GistFile.preview: Color range from {:.2f} to {:.2f}.".format(minval, maxval))

    tempfile = TemporaryFileName(prefix='PREVIEW-', suffix='.pdb')
    struct.save_pdb(tempfile.filename, force_overwrite=True, bfactors=dataset)
    view.add_component(tempfile.filename)
    # view.add_surface(color='bfactor')
    view.set_representations(representations=[{'type': 'surface', 'params': {'colorScheme': 'bfactor', 'colorScale': 'RdYlBu', 'colorReverse': True}}])
    # del tempfile
    return view


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


class TemporaryFileName():
    """Creates a random filename that can be used for a temporary file.  When
    the object is deleted, checks if the file exists and deletes it.  This is
    necessary mainly because mdtraj.save_pdb only uses filenames, no file
    objects."""

    def __init__(self, prefix='', suffix='', size=6, chars=string.ascii_uppercase + string.digits):
        dirname = tempfile.gettempdir()
        random_name = ''.join(random.choice(chars) for _ in range(size))
        self.filename = dirname + '/' + prefix + random_name + suffix
        return None

    def __del__(self):
        if os.path.isfile(self.filename):
            os.remove(self.filename)
        return None

    def __repr__(self):
        return "TemporaryFileName: {}".format(self.filename)

    def __str__(self):
        return self.filename


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
