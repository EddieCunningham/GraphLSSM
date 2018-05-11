import numpy as np
from GenModels.GM.Distributions.Base import ExponentialFam

__all__ = [ 'Categorical' ]

def definePrior():
    # Because of circular dependency
    from GenModels.GM.Distributions.Dirichlet import Dirichlet
    Categorical.priorClass = Dirichlet

class Categorical( ExponentialFam ):

    priorClass = None

    def __init__( self, p=None, prior=None, hypers=None ):
        definePrior()
        super( Categorical, self ).__init__( p, prior=prior, hypers=hypers )
        self.D = self.p.shape[ 0 ]

    ##########################################################################

    @property
    def p( self ):
        return self._params[ 0 ]

    ##########################################################################

    @classmethod
    def dataN( cls, x ):
        return x.shape[ 0 ]

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

    @property
    def constParams( self ):
        return self.D

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, constParams=None, forPost=False ):
    # def sufficientStats( cls, x, D=None, constParams=None, forPost=False ):
        # Compute T( x )
        assert isinstance( x, np.ndarray ) and x.ndim == 1
        # assert D is not None
        D = constParams
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
        if( params is not None ):
            if( not isinstance( params, tuple ) ):
                params = ( params, )
        assert ( params is None ) ^ ( natParams is None )
        ( p, ) = params if params is not None else cls.natToStandard( *natParams )
        return np.random.choice( p.shape[ 0 ], size, p=p )

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )
        ( p, ) = params if params is not None else cls.natToStandard( *natParams )
        assert isinstance( x, np.ndarray )
        assert x.ndim == 1
        return np.log( p[ x ] ).sum()
