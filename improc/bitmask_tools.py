import numpy as np
from scipy import ndimage


def dilate_bitmask(array, structure=None, iterations=1):
    """Dilate a bitmask array with a kernel. Each bit is dilated separately.

    Parameters
    ----------
    array : ndarray of integers
        The array to dilate.
    structure : ndarray of booleans
        The kernel to use for dilation.
        If None (default), will use a structure with square
        connectivity equal to one.
    iterations : int
        The number of iterations to dilate. Default is 1.

    Returns
    -------
    output : ndarray of integers
        The dilated array, same shape and type as the input array.
    """
    output = np.zeros_like(array)
    b = array.max()
    while b:
        b2 = np.array(b, dtype=array.dtype)
        new_mask = ndimage.binary_dilation(array & b2, structure=structure, iterations=iterations)
        output += new_mask * b2
        b >>= 1

    return output

