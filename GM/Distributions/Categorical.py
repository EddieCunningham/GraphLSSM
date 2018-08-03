import numpy as np
from GenModels.GM.Distributions.Base import ExponentialFam

__all__ = [ 'Categorical' ]

def definePrior():
    # Because of circular dependency
    from GenModels.GM.Distributions.Dirichlet import Dirichlet
    Categorical.priorClass = Dirichlet

class Categorical( ExponentialFam ):

    priorClass = None

    def __init__( self, pi=None, prior=None, hypers=None ):
        definePrior()
        super( Categorical, self ).__init__( pi, prior=prior, hypers=hypers )
        self.D = self.pi.shape[ 0 ]

    ##########################################################################

    @property
    def pi( self ):
        return self._params[ 0 ]

    @property
    def nat_pi( self ):
        return self._nat_params[ 0 ]

    @property
    def mf_nat_pi( self ):
        return self._mf_nat_params[ 0 ]

    ##########################################################################

    @classmethod
    def dataN( cls, x, constParams=None ):
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
    def standardToNat( cls, pi ):
        n = np.log( pi )
        return ( n, )

    @classmethod
    def natToStandard( cls, n ):
        pi = np.exp( n )
        return ( pi, )

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
    def log_partition( cls, x=None, params=None, nat_params=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( nat_params is None )
        if( split ):
            return ( 0, )
        return 0

    @classmethod
    def log_partitionGradient( cls, params=None, nat_params=None, split=False ):
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
    def sample( cls, params=None, nat_params=None, size=1 ):
        # Sample from pi( x | Ѳ; α )

        assert ( params is None ) ^ ( nat_params is None )
        ( p, ) = params if params is not None else cls.natToStandard( *nat_params )

        ans = np.random.choice( p.shape[ 0 ], size, p=p )
        cls.checkShape( ans )
        return ans

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, nat_params=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( nat_params is None )
        assert isinstance( x, np.ndarray )
        assert x.ndim == 1

        if( params is not None ):
            ( p, ) = params
            return np.log( p[ x ] ).sum()
        else:
            ( n, ) = nat_params
            return n[ x ].sum()

    ##########################################################################

    @classmethod
    def mode( cls, params=None, nat_params=None ):
        assert ( params is None ) ^ ( nat_params is None )
        if( params is not None ):
            return np.argmax( params )
        return ( np.argmax( nat_params ), )

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
