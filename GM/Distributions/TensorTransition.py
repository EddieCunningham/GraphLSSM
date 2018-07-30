import numpy as np
from GenModels.GM.Distributions.Base import ExponentialFam
from GenModels.GM.Distributions.Categorical import Categorical

__all__ = [ 'TensorTransition' ]

def definePrior():
    from GenModels.GM.Distributions.TensorTransitionDirichletPrior import TensorTransitionDirichletPrior
    TensorTransition.priorClass = TensorTransitionDirichletPrior

class TensorTransition( ExponentialFam ):

    priorClass = None

    def __init__( self, pi=None, prior=None, hypers=None ):
        definePrior()
        super( TensorTransition, self ).__init__( pi, prior=prior, hypers=hypers )
        self.Ds = self.pi.shape

    ##########################################################################

    @property
    def pi( self ):
        return self._params[ 0 ]

    ##########################################################################

    @classmethod
    def dataN( cls, x, constParams=None ):
        cls.checkShape( x )
        return x[ 0 ].shape[ 0 ]

    @classmethod
    def unpackSingleSample( cls, x ):
        return np.array( [ _x[ 0 ] for _x in x ] )

    @classmethod
    def sampleShapes( cls, Ds ):
        # ( Sample #, dim )
        return tuple( [ None for _ in len( Ds ) + 1 ] )

    def isampleShapes( cls, Ds ):
        return ( None, *Ds )

    @classmethod
    def checkShape( cls, x ):
        assert isinstance( x, list ) or isinstance( x, tuple )
        ndim = x[ 0 ].shape[ 0 ]
        assert sum( [ _x.shape[ 0 ] - ndim for _x in x ] ) == 0

    ##########################################################################

    @classmethod
    def standardToNat( cls, pi ):
        if( np.any( np.isclose( pi, 0.0 ) ) ):
            n = np.empty_like( pi )
            n[ ~np.isclose( pi, 0.0 ) ] = np.log( pi[ ~np.isclose( pi, 0.0 ) ] )
            n[ np.isclose( pi, 0.0 ) ] = np.NINF
        else:
            n = np.log( pi )
        return ( n, )

    @classmethod
    def natToStandard( cls, n ):
        pi = np.exp( n )
        return ( pi, )

    ##########################################################################

    @property
    def constParams( self ):
        return self.Ds

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, constParams=None ):
        # Compute T( x )
        Ds = constParams
        bins = np.zeros( Ds, dtype=int )
        # histogramdd won't work for some reason and
        # can't do bins[x]+=1
        for index in zip( *x ):
            bins[ index ] += 1

        assert bins.sum() == cls.dataN( x )
        return ( bins, )

    @classmethod
    def log_partition( cls, x=None, params=None, nat_params=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( nat_params is None )
        ( pi, ) = params if params is not None else cls.natToStandard( *nat_params )
        if( split ):
            return ( 0, )
        return 0

    @classmethod
    def log_partitionGradient( cls, params=None, nat_params=None ):
        return ( 0, ) if split == False else ( ( 0, ), ( 0, ) )

    def _testLogPartitionGradient( self ):
        # Don't need to test this
        pass

    ##########################################################################

    @classmethod
    def generate( cls, Ds=[ 2, 3, 4 ], size=1 ):
        params = ( np.ones( Ds ) / prod( Ds ), )
        samples = cls.sample( params=params, size=size )
        return samples if size > 1 else cls.unpackSingleSample( samples )

    @classmethod
    def sample( cls, params=None, nat_params=None, size=1 ):
        # Sample from P( x | Ѳ; α )
        assert ( params is None ) ^ ( nat_params is None )
        ( pi, ) = params if params is not None else cls.natToStandard( *nat_params )

        parents = [ np.random.choice( s, size ) for s in pi.shape[ :-1 ] ]
        child = np.hstack( [ np.random.choice( pi.shape[ -1 ], 1, p=p ) for p in pi[ parents ] ] )

        ans = parents + [ child ]
        cls.checkShape( ans )
        return ans

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, nat_params=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( nat_params is None )

        if( params is not None ):
            ( pi, ) = params
            return np.log( pi[ x ] ).sum()
        else:
            ( n, ) = nat_params
            return n[ x ].sum()

    ##########################################################################

    @classmethod
    def mode( cls, params=None, nat_params=None ):
        assert ( params is None ) ^ ( nat_params is None )
        assert 0
        if( params is not None ):
            return np.argmax( params, axis=1 )
        return ( np.argmax( nat_params, axis=1 ), )

    ##########################################################################

    @classmethod
    def maxLikelihoodFromStats( cls, t ):
        assert 0
        counts, = t
        p = counts - np.logaddexp.reduce( counts, axis=1 )[ :, None ]
        return ( p, )

    @classmethod
    def maxLikelihood( cls, x ):
        t = cls.sufficientStats( x )
        return cls.maxLikelihoodFromStats( t )
