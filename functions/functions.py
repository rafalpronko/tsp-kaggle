from numba import njit, prange
from sympy import primerange
import numpy as np

@njit
def dist_between_two_points(X, Y, alpha=1.0):
    dst = np.hypot(X[1] - X[0],
                   Y[1] - Y[0])
    return dst * alpha


@njit(parallel=True)
def score_tour_numba(tour_, X, Y, primes):
    """Score for path. X and Y in the node id order.

    Arguments:
        tour_ {numpy array} -- path
        X {numpy array} -- X coordynate for the node
        Y {numpy array} -- Y coordynate for the node
        primes {list} -- list of primes town

    Returns:
        float -- the score of the path.
    """

    full = 0.0
    for i in prange(0, len(tour_)-1):
        alpha = 1.0
        dst = np.hypot(X[tour_[i]] - X[tour_[i+1]],
                       Y[tour_[i]] - Y[tour_[i+1]])
        if i % 10 == 9 and primes[tour_[i]] == 0:
            alpha = 1.1
        full += alpha * dst
    return full


def generate_primes(max_number):
    return list(primerange(0, max_number))


@njit
def reverse_subpath(path, i, k):
    """Create the reverse subpath.

    Arguments:
        path {numpy array} -- Path 
        i {int} -- id node to start from
        k {int} -- id node to finish

    Returns:
        numpy array -- new path
    """

    while i < k:
        path[i], path[k] = path[k], path[i]
        i += 1
        k -= 1
    return path


def local2opt_swap(path, i, k):
    path[i], path[k] = path[k], path[i]
    return path


@njit
def local2opt_swap_numba(path, i, k):
    """Make a swap between two nodes in the path in numba.

    Args:
      path: list of the nodes
      i: int first node to swap
      k: int second node to swap
    Returns:
      New path.
    """
    path[i], path[k] = path[k], path[i]
    return path
