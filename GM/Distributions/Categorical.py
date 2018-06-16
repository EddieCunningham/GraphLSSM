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
        cls.checkShape( x )
        if( x.ndim == 2 ):
            return x.shape[ 0 ]
        return 1

    @classmethod
    def unpackSingleSample( cls, x ):
        return x[ 0 ]

    @classmethod
    def sampleShapes( cls ):
        # ( Sample #, dim )
        return ( None, None )

    def isampleShapes( cls ):
        return ( None, self.D )

    @classmethod
    def checkShape( cls, x ):
        assert isinstance( x, np.ndarray )
        assert x.ndim == 2 or x.ndim == 1

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
    def sufficientStats( cls, x, constParams=None ):
        # Compute T( x )
        assert ( isinstance( x, np.ndarray ) and x.ndim == 1 ) or isinstance( x, list ), x
        D = constParams
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

    @classmethod
    def log_partitionGradient( cls, params=None, natParams=None, split=False ):
        return ( 0, ) if split == False else ( ( 0, ), ( 0, ) )

    def _testLogPartitionGradient( self ):
        # Don't need to test this
        pass

    ##########################################################################

    @classmethod
    def generate( cls, D=2, size=1 ):
        params = ( np.ones( D ) / D, )
        samples = cls.sample( params=params, size=size )
        return samples if size > 1 else cls.unpackSingleSample( samples )

    @classmethod
    def sample( cls, params=None, natParams=None, size=1 ):
        # Sample from P( x | Ѳ; α )

        assert ( params is None ) ^ ( natParams is None )
        ( p, ) = params if params is not None else cls.natToStandard( *natParams )

        ans = np.random.choice( p.shape[ 0 ], size, p=p )
        cls.checkShape( ans )
        return ans

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )
        assert isinstance( x, np.ndarray )
        assert x.ndim == 1

        if( params is not None ):
            ( p, ) = params
            return np.log( p[ x ] ).sum()
        else:
            ( n, ) = natParams
            return n[ x ].sum()

    ##########################################################################

    @classmethod
    def mode( cls, params=None, natParams=None ):
        assert ( params is None ) ^ ( natParams is None )
        if( params is not None ):
            return np.argmax( params )
        return ( np.argmax( natParams ), )

    ##########################################################################

    @classmethod
    def maxLikelihoodFromStats( cls, t ):
        counts, = t
        p = counts - np.logaddexp.reduce( counts )
        return ( p, )

    @classmethod
    def maxLikelihood( cls, x ):
        t = cls.sufficientStats( x )
        return cls.maxLikelihoodFromStats( t )
