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
        x, y = x
        return x.shape[ 0 ]

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
        return self.D_in, self.D_out

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, constParams=None, forPost=False ):
        # Compute T( x )
        x, y = x
        assert isinstance( x, np.ndarray ) and x.ndim == 1, x
        assert isinstance( y, np.ndarray ) and y.ndim == 1, y
        D_in, D_out = constParams
        assert D_in is not None and D_out is not None

        xy = np.vstack( ( x, y ) )
        t, _, _ = np.histogram2d( x, y, bins=( range( D_in + 1 ), range( D_out + 1 ) ) )
        return ( t, )

        t1 = np.bincount( x, minlength=D_in )
        t2 = np.bincount( y, minlength=D_out )
        return ( np.outer( t1, t2 ), )

    @classmethod
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( natParams is None )
        ( pi, ) = params if params is not None else cls.natToStandard( *natParams )
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
        ( pi, ) = params if params is not None else cls.natToStandard( *natParams )
        x = np.random.choice( pi.shape[ 0 ], size )
        y = np.array( [ np.random.choice( pi.shape[ 1 ], 1, p=pi[ _x ] ) for _x in x ] ).ravel()
        return ( x, y )

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
