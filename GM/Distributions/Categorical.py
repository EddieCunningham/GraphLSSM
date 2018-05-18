import numpy as np
from GenModels.GM.Distributions.Base import ExponentialFam, checkExpFamArgs, multiSampleLikelihood

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
    def dataN( cls, x, ravel=False ):
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
        assert isinstance( x, np.ndarray ) and x.ndim == 1, x
        D = constParams
        assert D is not None
        t1 = np.bincount( x, minlength=D )
        return ( t1, )

    @classmethod
    @checkExpFamArgs
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        if( split ):
            return ( 0, )
        return 0

    ##########################################################################

    @classmethod
    @checkExpFamArgs
    def sample( cls, params=None, natParams=None, D=None, size=1, ravel=False ):
        # Sample from P( x | Ѳ; α )
        if( params is not None ):
            if( not isinstance( params, tuple ) ):
                params = ( params, )
        elif( natParams is None ):
            assert D is not None
            params = ( np.ones( D ) / D, )
        ( p, ) = params if params is not None else cls.natToStandard( *natParams )
        if( p.ndim > 1 ):
            assert p.size == p.squeeze().size
            p = p.squeeze()
        assert p.ndim == 1, p
        ans = np.random.choice( p.shape[ 0 ], size, p=p )
        return ans

    ##########################################################################

    @classmethod
    @checkExpFamArgs
    def log_likelihood( cls, x, params=None, natParams=None, ravel=False ):
        # Compute P( x | Ѳ; α )
        ( p, ) = params if params is not None else cls.natToStandard( *natParams )
        assert isinstance( x, np.ndarray )
        assert x.ndim == 1
        return np.log( p[ x ] ).sum()
