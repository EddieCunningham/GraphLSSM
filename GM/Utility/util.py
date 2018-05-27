import numpy as np
from scipy.linalg import lapack
import copy

__all__ = [ 'invPsd', 'is_outlier', 'fullyRavel', 'randomStep', 'deepCopy', 'stabilize' ]

def invPsd( A, AChol=None, returnChol=False ):
    # https://github.com/mattjj/pybasicbayes/blob/9c00244b2d6fd767549de6ab5d0436cec4e06818/pybasicbayes/util/general.py
    L = np.linalg.cholesky( A ) if AChol is None else AChol
    Ainv = lapack.dpotri( L, lower=True )[ 0 ]
    Ainv += np.tril( Ainv, k=-1 ).T
    if returnChol:
        return Ainv, L
    return Ainv

def stabilize( A ):
    w, v = np.linalg.eig( A )
    badIndices = ( w < 0 ) | ( w > 1 )
    w[ badIndices ] = np.random.random( badIndices.sum() )
    return v @ np.diag( w ) @ np.linalg.inv( v )

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.

        https://stackoverflow.com/a/11886564/7479938
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    if( np.isclose( med_abs_deviation, 0.0 ) ):
        return np.zeros_like( diff, dtype=bool )

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def fullyRavel( x ):


    y = copy.deepcopy( x )
    def recurse( y ):
        if( isinstance( y, tuple ) or isinstance( y, list ) ):
            for _y in y:
                y = np.hstack( [ recurse( _y ) for _y in y ] )
        else:
            y = y.ravel()
        return y
    return recurse( y )

    # return y


    # if( isinstance( x, tuple ) or isinstance( x, list ) ):
    #     x = np.hstack( [ _x.ravel() for _x in x ] )
    # return x.ravel()

def randomStep( x ):

    y = copy.deepcopy( x )
    def recurse( y ):
        if( isinstance( y, tuple ) or isinstance( y, list ) ):
            for _y in y:
                recurse( _y )
        else:
            y += np.random.standard_normal( size=y.shape )
    recurse( y )

    return y

def deepCopy( x ):
    return copy.deepcopy( x )


def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(xrange(n), r))
    return tuple(pool[i] for i in indices)