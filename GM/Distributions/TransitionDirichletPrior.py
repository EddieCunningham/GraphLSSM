import numpy as np
from GenModels.GM.Distributions.Base import ExponentialFam
from scipy.special import gammaln
from GenModels.GM.Distributions import Dirichlet, Transition

__all__ = [ 'TransitionDirichletPrior' ]

class TransitionDirichletPrior( ExponentialFam ):

    def __init__( self, alpha=None, prior=None, hypers=None ):
        super( TransitionDirichletPrior, self ).__init__( alpha, prior=prior, hypers=hypers )

    ##########################################################################

    @property
    def alpha( self ):
        return self._params[ 0 ]

    ##########################################################################

    @classmethod
    def dataN( cls, x ):
        if( x.ndim == 2 ):
            return 1
        return x.shape[ 0 ]

    ##########################################################################

    @classmethod
    def standardToNat( cls, alpha ):
        return ( alpha - 1, )

    @classmethod
    def natToStandard( cls, n ):
        return ( n + 1, )

    ##########################################################################

    @property
    def constParams( self ):
        return None

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, constParams=None, forPost=False ):
        # Compute T( x )
        if( cls.dataN( x ) > 1 ):
            t = ( 0, 0 )
            for _x in x:
                t = np.add( t, cls.sufficientStats( _x, forPost=forPost ) )
            return t

        t1, = Transition.standardToNat( x )
        t2, = Transition.log_partition( params=( x, ), split=True )
        return t1, -t2

    @classmethod
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( natParams is None )
        ( alpha, ) = params if params is not None else cls.natToStandard( *natParams )
        return sum( [ Dirichlet.log_partition( params=( a, ) ) for a in alpha ] )

    ##########################################################################

    @classmethod
    def sample( cls, params=None, natParams=None, size=1 ):
        # Sample from P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )

        if( params is not None ):
            if( not isinstance( params, tuple ) or \
                not isinstance( params, list ) ):
                params = ( params, )

        ( alpha, ) = params if params is not None else cls.natToStandard( *natParams )

        ans = np.swapaxes( np.array( [ Dirichlet.sample( params=( a, ), size=size ) for a in alpha ] ), 0, 1 )
        return ans

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )
        ( alpha, ) = params if params is not None else cls.natToStandard( *natParams )
        if( isinstance( x, tuple ) ):
            assert len( x ) == 1
            x, = x
        assert isinstance( x, np.ndarray )
        if( x.ndim == 3 ):
            return sum( [ TransitionDirichletPrior.log_likelihood( _x, params=( alpha, ) ) for _x in x ] )

        assert isinstance( x, np.ndarray ) and x.ndim == 2
        return sum( [ Dirichlet.log_likelihood( _x, params=( a, ) ) for _x, a in zip( x, alpha ) ] )
