import numpy as np
from scipy.linalg import lapack
import copy
from functools import wraps
import inspect

__all__ = [ 'invPsd',
            'is_outlier',
            'fullyRavel',
            'randomStep',
            'deepCopy',
            'doublewrap',
            'fullSampleSupport',
            'fullLikelihoodSupport',
            'checkExpFamArgs',
            'checkArgs',
            'multiCall',
            'multiCallOnInput' ]

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
            y = np.hstack( [ recurse( _y ) for _y in y ] )
        else:
            y = y.ravel() if isinstance( y, np.ndarray ) else y
        return y
    return recurse( y )

def randomStep( x ):
    x += np.random.standard_normal( size=x.shape )
    return x

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

def bArgs( func, *args, **kwargs ):
    boundArgs = inspect.signature( func ).bind( *args )
    boundArgs.apply_defaults()
    d = boundArgs.arguments
    d.update( kwargs )
    return d

def extractArg( func, name, default, *args, **kwargs ):
    inputArgs = bArgs( func, *args, **kwargs )
    if( name in inputArgs ):
        ans = inputArgs[ name ]
    else:
        ans = default
    return ans

##########################################################################

@doublewrap
def autoRavel( func, calledFunc=None ):

    @wraps( func )
    def autoRavelWrapper( *args, **kwargs ):
        ravel = extractArg( calledFunc, 'ravel', False, *args, **kwargs )
        if( 'ravel' in kwargs ):
            del kwargs[ 'ravel' ]

        ans = func( *args, **kwargs )
        if( ravel ):
            if( isinstance( ans, list ) or isinstance( ans, tuple ) ):
                ans = np.hstack( [ x.ravel() for x in ans ] )
            else:
                ans = ans.ravel()
        return ans
    return autoRavelWrapper

@doublewrap
def multiCall( func, calledFunc=None ):

    @wraps( func )
    def multiCallWrapper( *args, **kwargs ):
        size = extractArg( calledFunc, 'size', 1, *args, **kwargs )
        if( 'size' in kwargs ):
            del kwargs[ 'size' ]

        if( size > 1 ):
            ans = [ func( *args, **kwargs ) for _ in range( size ) ]
        else:
            ans = func( *args, **kwargs )
            assert isinstance( ans, list ) == False
        return ans
    return multiCallWrapper

##########################################################################

@doublewrap
def autoUnRavel( func ):
    @wraps( func )
    def autoUnRavelWrapper( self, x, *args, params=None, natParams=None, ravel=False, **kwargs ):
        if( params is None ):
            assert natParams is not None
            params = self.natToStandard( *natParams )
        x = self.unravelSample( x, params ) if ravel else x
        return func( self, x, *args, params=params, **kwargs )
    return autoUnRavelWrapper

@doublewrap
def multiCallOnInput( func ):
    @wraps( func )
    def multiCallOnInputWrapper( self, x, *args, **kwargs ):
        size = self.dataN( x )
        if( size > 1 ):
            return sum( ( func( self, _x, *args, **kwargs ) for _x in x ) )
        return func( self, x, *args, **kwargs )
    return multiCallOnInputWrapper

##########################################################################

@doublewrap
def fullSampleSupport( func ):
    @multiCall( calledFunc=func )
    @autoRavel( calledFunc=func )
    def fullSampleSupportWrapper( *args, **kwargs ):
        return func( *args, **kwargs )
    return fullSampleSupportWrapper

@doublewrap
def fullLikelihoodSupport( func ):
    @multiCallOnInput
    @autoUnRavel
    def fullLikelihoodSupportWrapper( *args, **kwargs ):
        return func( *args, **kwargs )
    return fullLikelihoodSupportWrapper

##########################################################################

@doublewrap
def checkArgs( func, allowNone=False ):

    @wraps( func )
    def wrapper( clsOrSelf, *args, **kwargs ):

        args = ( clsOrSelf, ) + args

        params = extractArg( func, 'params', None, *args, **kwargs )
        priorParams = extractArg( func, 'priorParams', None, *args, **kwargs )

        if( params is None ):
            assert allowNone
            D = extractArg( func, 'D', None, *args, **kwargs )
            if( D is None ):
               D_in = extractArg( func, 'D_in', None, *args, **kwargs )
               D_out = extractArg( func, 'D_out', None, *args, **kwargs )
               assert D_in is not None and D_out is not None
               params = clsOrSelf.easyParamSample( **{ 'D_in': D_in, 'D_out': D_out } )

               del kwargs[ 'D_in' ]
               del kwargs[ 'D_out' ]
            else:
               params = clsOrSelf.easyParamSample( **{ 'D': D } )
               del kwargs[ 'D' ]

            kwargs[ 'params' ] = params

        return func( *args, **kwargs )

    return wrapper


##########################################################################

@doublewrap
def checkExpFamArgs( func, allowNone=False ):

    @wraps( func )
    def wrapper( clsOrSelf, *args, **kwargs ):

        args = ( clsOrSelf, ) + args

        params = extractArg( func, 'params', None, *args, **kwargs )
        natParams = extractArg( func, 'natParams', None, *args, **kwargs )
        priorParams = extractArg( func, 'priorParams', None, *args, **kwargs )
        priorNatParams = extractArg( func, 'priorNatParams', None, *args, **kwargs )

        if( params is not None or natParams is not None ):
            assert ( params is None ) ^ ( natParams is None ), kwargs
        elif( allowNone ):
            D = extractArg( func, 'D', None, *args, **kwargs )
            if( D is None ):
               D_in = extractArg( func, 'D_in', None, *args, **kwargs )
               D_out = extractArg( func, 'D_out', None, *args, **kwargs )
               assert D_in is not None and D_out is not None
               params = clsOrSelf.easyParamSample( **{ 'D_in': D_in, 'D_out': D_out } )

               del kwargs[ 'D_in' ]
               del kwargs[ 'D_out' ]
            else:
               params = clsOrSelf.easyParamSample( **{ 'D': D } )
               del kwargs[ 'D' ]

            kwargs[ 'params' ] = params

        if( priorParams is not None and priorNatParams is not None ):
            assert ( priorParams is None ) ^ ( priorNatParams is None ), kwargs

        return func( *args, **kwargs )

    return wrapper
