import numpy as np
import numba
from functools import partial
from math import sqrt

# constants
pi = np.pi
dtype = np.complex64

try:
    import numba
    mode = 'numba'
except Exception:
    print("Numba is not available, performance will be slower")
    mode = None


def conv_db(x):
    return 10*np.log10(x)


def smoothing_filter(x, N=20):
    """
    Applies an N-point smoothing filter to data, generally for plotting. Data at the end of the array has less smoothing
    :param x: 1D data
    :param N: Number of samples to use for the moving filter
    :return: smoothed 1D data of equal size
    """
    xcopy = x.copy()
    xcopy[:-N+1] = np.convolve(x, np.ones(N)/N, mode='valid')
    for i in range(1, N):
        xcopy[-i] = np.mean(xcopy[-N+i:])
    return xcopy


def largest_indices(ary, N):
    """
    Returns the N largest indices from an array. The array can be either 1D or larger and the return shape is the same
    :param ary: ndarray to choose the largest indices from
    :param N: The number of indices to select
    :return: Either a 1D or higher with the same number of dimensions as the input
    """
    flat = ary.flatten()
    indices = np.argpartition(flat, -N)[-N:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


if mode == 'numba':
    @numba.jit(nopython=True)
    def nb_mean_0(tensor):
        units = tensor.shape
    #     tensor = tensor.transpose((axis) + units[:axis] + units[axis+1:])
    #     tensor = np.moveaxis(tensor, axis, 0)
        outputs = np.zeros((units[1:]), dtype=np.complex64)
        for i in numba.prange(units[0]):
            outputs = outputs + tensor[i]
        return outputs

    @numba.jit(nopython=True, parallel=True)
    def nb_svd_s(H):
        units = H.shape
        U = np.zeros((units[0], units[1], units[1]), dtype=np.complex64)
        S = np.zeros((units[0], units[2]), dtype=np.complex64)
        V = np.zeros((units[0], units[2], units[2]), dtype=np.complex64)
        for i in numba.prange(len(H)):
            U[i, :, :], S[i, :], V[i, :, :] = np.linalg.svd(H[i])
        return S

    @numba.jit(parallel=True, nopython=True)
    def nb_det(tensor):
        outputs = np.zeros((len(tensor)), dtype=np.complex64)
        for i in numba.prange(len(tensor)):
            outputs[i] = np.linalg.det(tensor[i])
        return outputs


    @numba.jit(nopython=True, parallel=True)
    def nb_svd(H):
        units = H.shape
        U = np.zeros((units[0], units[1], units[1]), dtype=np.complex64)
        S = np.zeros((units[0], units[1]), dtype=np.complex64)
        V = np.zeros((units[0], units[2], units[2]), dtype=np.complex64)
        for i in numba.prange(len(H)):
            U[i, :, :], S[i, :], V[i, :, :] = np.linalg.svd(H[i])
        return U, S, V


else:
    def nb_mean_0(tensor):
        units = tensor.shape
    #     tensor = tensor.transpose((axis) + units[:axis] + units[axis+1:])
    #     tensor = np.moveaxis(tensor, axis, 0)
        outputs = np.zeros((units[1:]), dtype=np.complex64)
        for i in range(units[0]):
            outputs = outputs + tensor[i]
        return outputs

    def nb_svd_s(H):
        units = H.shape
        U = np.zeros((units[0], units[1], units[1]), dtype=np.complex64)
        S = np.zeros((units[0], units[2]), dtype=np.complex64)
        V = np.zeros((units[0], units[2], units[2]), dtype=np.complex64)
        for i in range(len(H)):
            U[i, :, :], S[i, :], V[i, :, :] = np.linalg.svd(H[i])
        return S

    def nb_det(tensor):
        outputs = np.zeros((len(tensor)), dtype=np.complex64)
        for i in range(len(tensor)):
            outputs[i] = np.linalg.det(tensor[i])
        return outputs


    def nb_svd(H):
        units = H.shape
        U = np.zeros((units[0], units[1], units[1]), dtype=np.complex64)
        S = np.zeros((units[0], units[1]), dtype=np.complex64)
        V = np.zeros((units[0], units[2], units[2]), dtype=np.complex64)
        for i in range(len(H)):
            U[i, :, :], S[i, :], V[i, :, :] = np.linalg.svd(H[i])
        return U, S, V
