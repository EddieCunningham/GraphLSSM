import numpy as np
from scipy.linalg import lapack
import copy
import autograd
from scipy.special import digamma
from functools import partial

__all__ = [ 'multigammalnDerivative',
            'invPsd',
            'is_outlier',
            'fullyRavel',
            'randomStep',
            'deepCopy',
            'stabilize',
            'cheatPrecisionHelper' ]

##########################################################################

def multigammalnDerivative( d, x ):
    return digamma( x + ( 1 - np.arange( 1, d + 1 ) ) / 2 ).sum()

##########################################################################

def invPsd( A, AChol=None, returnChol=False ):
    # https://github.com/mattjj/pybasicbayes/blob/9c00244b2d6fd767549de6ab5d0436cec4e06818/pybasicbayes/util/general.py
    L = np.linalg.cholesky( A ) if AChol is None else AChol
    Ainv = lapack.dpotri( L, lower=True )[ 0 ]
    Ainv += np.tril( Ainv, k=-1 ).T
    if returnChol:
        return Ainv, L
    return Ainv

##########################################################################

def stabilize( A ):
    w, v = np.linalg.eig( A )
    badIndices = ( w < 0.1 ) | ( w > 0.9 )
    w[ badIndices ] = np.random.random( badIndices.sum() ) * 0.9 + 0.05
    return v @ np.diag( w ) @ np.linalg.inv( v )

##########################################################################

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

##########################################################################

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

##########################################################################

def randomStep( x, stepSize=1.0 ):

    y = copy.deepcopy( x )
    def recurse( y ):
        if( isinstance( y, tuple ) or isinstance( y, list ) ):
            for _y in y:
                recurse( _y )
        else:
            y += np.random.standard_normal( size=y.shape ) * stepSize
    recurse( y )

    return y

##########################################################################

def deepCopy( x ):
    return copy.deepcopy( x )

##########################################################################

def cheatPrecisionHelper( x, N ):
    # A quick fix for numerical precision issues.  In the future, don't
    # use this and use a more stable algorithm.
    # Assumes that x is psd matrix
    x = ( x + x.T ) / 2.0
    x[ np.diag_indices( N ) ] += np.ones( N ) * 1e-8
    return x