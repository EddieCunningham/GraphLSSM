import numpy as np
from GenModels.GM.Distributions.Base import ExponentialFam
from GenModels.GM.Distributions.Categorical import Categorical

__all__ = [ 'Transition' ]

def definePrior():
    from GenModels.GM.Distributions.TransitionDirichletPrior import TransitionDirichletPrior
    Transition.priorClass = TransitionDirichletPrior

class Transition( ExponentialFam ):

    priorClass = None

    def __init__( self, pi=None, prior=None, hypers=None ):
        definePrior()
        super( Transition, self ).__init__( pi, prior=prior, hypers=hypers )
        self.D_in = self.pi.shape[ 0 ]
        self.D_out = self.pi.shape[ 1 ]

    ##########################################################################

    @property
    def pi( self ):
        return self._params[ 0 ]

    ##########################################################################

    @classmethod
    def dataN( cls, x ):
        cls.checkShape( x )
        x, y = x
        return x.shape[ 0 ]

    @classmethod
    def unpackSingleSample( cls, x ):
        x, y = x
        return x[ 0 ], y[ 0 ]

    @classmethod
    def sampleShapes( cls ):
        # ( ( Sample #, dim1 ), ( Sample #, dim2 ) )
        return ( ( None, None ), ( None, None ) )

    def isampleShapes( cls ):
        return ( ( None, self.D_in ), ( None, self.D_out ) )

    @classmethod
    def checkShape( cls, x ):
        assert isinstance( x, tuple )
        x, y = x
        assert isinstance( x, np.ndarray ) and isinstance( y, np.ndarray )
        assert x.ndim == 1 and y.ndim == 1
        assert x.shape[ 0 ] == y.shape[ 0 ]

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
        return self.D_in, self.D_out

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, constParams=None ):
        # Compute T( x )
        x, y = x

        assert isinstance( x, np.ndarray )
        assert isinstance( y, np.ndarray )
        D_in, D_out = constParams
        assert D_in is not None and D_out is not None

        x = x.squeeze()
        y = y.squeeze()

        if( y.ndim > 1 and x.ndim == 1 ):
            # Mutiple ys for this x
            t, _, _ = np.histogram2d( x, y[ 0 ], bins=( range( D_in + 1 ), range( D_out + 1 ) ) )
            for _y in y[ 1: ]:
                _t, _, _ = np.histogram2d( x, _y, bins=( range( D_in + 1 ), range( D_out + 1 ) ) )
                t += _t
            return ( t, )

        elif( y.ndim > 1 and x.ndim > 1 ):
            assert y.shape[ 0 ] == x.shape[ 0 ]
            # Mutiple x, y pairs
            t, _, _ = np.histogram2d( x[ 0 ], y[ 0 ], bins=( range( D_in + 1 ), range( D_out + 1 ) ) )
            for _x, _y in zip( x[ 1: ], y[ 1: ] ):
                _t, _, _ = np.histogram2d( _x, _y, bins=( range( D_in + 1 ), range( D_out + 1 ) ) )
                t += _t
            return ( t, )
        else:
            # A single x, y pair
            t, _, _ = np.histogram2d( x, y, bins=( range( D_in + 1 ), range( D_out + 1 ) ) )
            return ( t, )

    @classmethod
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( natParams is None )
        ( pi, ) = params if params is not None else cls.natToStandard( *natParams )
        if( split ):
            return ( 0, )
        return 0

    @classmethod
    def log_partitionGradient( cls, params=None, natParams=None ):
        return ( 0, ) if split == False else ( ( 0, ), ( 0, ) )

    def _testLogPartitionGradient( self ):
        # Don't need to test this
        pass

    ##########################################################################

    @classmethod
    def generate( cls, D_in=3, D_out=2, size=1 ):
        params = ( np.ones( ( D_in, D_out ) ) / ( D_in * D_out ), )
        samples = cls.sample( params=params, size=size )
        return samples if size > 1 else cls.unpackSingleSample( samples )

    @classmethod
    def sample( cls, params=None, natParams=None, size=1 ):
        # Sample from P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )
        ( pi, ) = params if params is not None else cls.natToStandard( *natParams )
        x = np.random.choice( pi.shape[ 0 ], size )
        y = np.array( [ np.random.choice( pi.shape[ 1 ], 1, p=pi[ _x ] ) for _x in x ] ).ravel()

        ans = ( x, y )
        cls.checkShape( ans )
        return ans

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )
        x, y = x
        assert isinstance( x, np.ndarray )
        assert x.ndim == 1 and y.ndim == 1

        if( params is not None ):
            ( pi, ) = params
            return np.log( pi[ x, y ] ).sum()
        else:
            ( n, ) = natParams
            return n[ x, y ].sum()

    ##########################################################################

    @classmethod
    def mode( cls, params=None, natParams=None ):
        assert ( params is None ) ^ ( natParams is None )
        if( params is not None ):
            return np.argmax( params, axis=1 )
        return ( np.argmax( natParams, axis=1 ), )

    ##########################################################################

    @classmethod
    def maxLikelihoodFromStats( cls, t ):
        counts, = t
        p = counts - np.logaddexp.reduce( counts, axis=1 )[ :, None ]
        return ( p, )

    @classmethod
    def maxLikelihood( cls, x ):
        t = cls.sufficientStats( x )
        return cls.maxLikelihoodFromStats( t )
