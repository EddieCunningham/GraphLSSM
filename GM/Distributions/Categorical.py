import numpy as np
from Base import Exponential
from Dirichlet import Dirichlet


class Categorical( Exponential ):

    priorClass = Dirichlet

    def __init__( self, p=None, prior=None, hypers=None ):
        super( Categorical, self ).__init__( p, prior=prior, hypers=hypers )

    ##########################################################################

    @classmethod
    def standardToNat( cls, p ):
        n = np.log( p )
        return ( n, )

    @classmethod
    def natToStandard( cls, n ):
        p = np.exp( n )
        return ( p, )

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, D=None, forPost=False ):
        # Compute T( x )
        assert isinstance( x, np.ndarray ) and x.ndim == 1
        assert D is not None
        t1 = np.bincount( x, minlength=D )
        return ( t1, )

    @classmethod
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( natParams is None )
        if( split ):
            return ( 0, )
        return 0

    ##########################################################################

    @classmethod
    def sample( cls, params=None, natParams=None, size=1 ):
        # Sample from P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )
        if( params is not None ):
            if( not isinstance( params, tuple ) or \
                not isinstance( params, list ) ):
                params = ( params, )

        ( p, ) = params if params is not None else cls.natToStandard( *natParams )
        return np.random.choice( p.shape[ 0 ], size, p=p )

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )
        ( p, ) = params if params is not None else cls.natToStandard( *natParams )
        if( isinstance( x, np.ndarray ) ):
            assert x.size == 1
            x = x[ 0 ]
        return np.log( p[ x ] )