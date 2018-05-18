import numpy as np
from scipy.linalg import lapack
import copy
from functools import wraps

__all__ = [ 'invPsd', 'is_outlier', 'fullyRavel', 'randomStep', 'deepCopy', 'doublewrap', 'multiParamSample', 'multiSampleLikelihood' ]

def invPsd( A, AChol=None, returnChol=False ):
    # https://github.com/mattjj/pybasicbayes/blob/9c00244b2d6fd767549de6ab5d0436cec4e06818/pybasicbayes/util/general.py
    L = np.linalg.cholesky( A ) if AChol is None else AChol
    Ainv = lapack.dpotri( L, lower=True )[ 0 ]
    Ainv += np.tril( Ainv, k=-1 ).T
    if returnChol:
        return Ainv, L
    return Ainv

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


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
    """
    @wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator

##########################################################################

@doublewrap
def autoRavel( func ):
    @wraps( func )
    def wrapper( self, *args, ravel=False, **kwargs ):

        ans = func( self, *args, **kwargs )
        if( ravel ):
            if( isinstance( ans, list ) or isinstance( and, tuple ) ):
                ans = np.hstack( [ x.ravel() for x in ans ] )
            else:
                ans = ans.ravel()
        return ans
    return wrapper

@doublewrap
def multiCall( func ):
    @wraps( func )
    def wrapper( self, *args, size=1, **kwargs ):
        if( size > 1 ):
            ans = [ func( self, *args, **kwargs ) for _ in range( size ) ]
        else:
            ans = func( self, *args, **kwargs )
        return ans
    return wrapper

@doublewrap
def autoUnRavel( func ):
    @wraps( func )
    def wrapper( self, x, *args, params=None, ravel=False, **kwargs ):
        x = self.unravelSample( x, params ) if ravel else x
        return func( self, x, *args, params=params, **kwargs )
    return wrapper

@doublewrap
def multiCallOnInput( func ):
    @wraps( func )
    def wrapper( self, x, *args, **kwargs ):
        size = self.dataN( x )
        if( size > 1 ):
            return sum( ( func( self, _x, *args, **kwargs ) for _x in x ) )
        return func( self, x, *args, **kwargs )
    return wrapper


# @doublewrap
# def multiParamSample( func ):
#     # This is a wrapper for classes whose sample function returns a tuple.
#     # Will give capability of flattening the output into one vector.
#     # Assumes that the sample function returns a single sample

#     @wraps( func )
#     def wrapper( self, *args, **kwargs ):
#         assert 'ravel' in kwargs and 'size' in kwargs, kwargs
#         size = kwargs[ 'size' ]
#         ravel = kwargs[ 'ravel' ]
#         kwargs[ 'size' ] = 1

#         ans = [ None for _ in range( size ) ]
#         for i in range( size ):
#             sample = func( self, *args, **kwargs )
#             assert isinstance( sample, tuple )
#             if( ravel ):
#                 raveled = []
#                 for s in sample:
#                     raveled.append( s.ravel() )
#                 ans[ i ] = np.hstack( raveled )
#             else:
#                 ans[ i ] = sample

#         if( ravel ):
#             return np.array( ans ) if size > 1 else np.array( ans[ 0 ] )
#         return ans if size > 1 else ans[ 0 ]

#     return wrapper

# ##########################################################################

# @doublewrap
# def multiSampleLikelihood( func ):
#     # Like above except for likelihood

#     @wraps( func )
#     def wrapper( self, x, params, **kwargs ):
#         assert 'ravel' in kwargs, args
#         ravel = kwargs[ 'ravel' ]

#         size = self.dataN( x, ravel=ravel )
#         if( size > 0 ):
#             return sum( [ func( self, _x, **kwargs ) for _x in x ] )

#         if( ravel == True ):
#             x = self.unravelSample( x, params )

#         kwargs[ 'ravel' ] = False
#         return func( self, x, **kwargs )

#     return wrapper

##########################################################################

@doublewrap
def checkExpFamArgs( func, allowNone=False ):

    @wraps( func )
    def wrapper( *args, **kwargs ):

        if( 'params' in kwargs and 'natParams' in kwargs ):
            params = kwargs[ 'params' ]
            natParams = kwargs[ 'natParams' ]
            if( allowNone ):
                if( not( params is None and natParams is None ) ):
                    assert ( params is None ) ^ ( natParams is None ), kwargs
            else:
                assert ( params is None ) ^ ( natParams is None ), kwargs

        if( 'priorParams' in kwargs and 'priorNatParams' in kwargs ):
            priorParams = kwargs[ 'priorParams' ]
            priorNatParams = kwargs[ 'priorNatParams' ]
            assert ( priorParams is None ) ^ ( priorNatParams is None ), kwargs

        return func( *args, **kwargs )

    return wrapper
