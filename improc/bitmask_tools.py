import numpy as np
from scipy import ndimage


def dilate_bitmask(array, iterations=1, structure=None):
    """Dilate a bitmask array with a kernel. Each bit is dilated separately.

    Parameters
    ----------
    array : ndarray of integers
        The array to dilate.
    iterations : int
        The number of iterations to dilate. Default is 1.
    structure : ndarray of booleans
        The kernel to use for dilation.
        If None (default), will use a structure with square
        connectivity equal to one.

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


def make_saturated_mask(imdata, saturation=50000, iterations=2, structure=None):
    """Create a mask around saturated pixels.

    Will dilate the mask by the given number of iterations, using the given structure (use the scipy default if None).

    Parameters
    ----------
    imdata: ndarray
        The image data.
    saturation: float
        The saturation level. Default is 50000.
    iterations : int
        The number of iterations to dilate. Default is 1.
    structure : ndarray of booleans
        The kernel to use for dilation.
        If None (default), will use a structure with square
        connectivity equal to one.

    Returns
    -------
    boolean mask
    """
    mask = imdata >= saturation

    return ndimage.binary_dilation(mask, iterations=iterations, structure=structure)