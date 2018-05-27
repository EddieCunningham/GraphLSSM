from GenModels.GM.States.StandardStates.StateBase import StateBase
from GenModels.GM.States.MessagePassing.ForwardBackward import *
from GenModels.GM.Distributions import Categorical, Transition
import numpy as np

__all__ = [ 'HMMState' ]

def definePrior():
    from GenModels.GM.ModelPriors.HMMDirichletPrior import HMMDirichletPrior
    HMMState.priorClass = HMMDirichletPrior

class HMMState( CategoricalForwardBackward, StateBase ):

    priorClass = None

    def __init__( self, initialDist=None, transDist=None, emissionDist=None, prior=None, hypers=None ):
        definePrior()
        super( HMMState, self ).__init__( initialDist, transDist, emissionDist, prior=prior, hypers=hypers )

    @property
    def params( self ):
        return self._params

    @params.setter
    def params( self, val ):
        self.standardChanged = True
        initialDist, transDist, emissionDist = val
        self.updateParams( initialDist, transDist, emissionDist )
        self._params = val

    ######################################################################

    @property
    def initialDist( self ):
        return self._params[ 0 ]

    @property
    def transDist( self ):
        return self._params[ 1 ]

    @property
    def emissionDist( self ):
        return self._params[ 2 ]

    ######################################################################

    @property
    def constParams( self ):
        return self.stateSize, self.emissionSize

    @classmethod
    def dataN( cls, x ):
        x, y = x
        if( x.ndim == 1 ):
            return 1
        return x.shape[ 0 ]

    ######################################################################

    @classmethod
    def standardToNat( cls, initialDist, transDist, emissionDist ):
        n1, = Categorical.standardToNat( initialDist )
        n2, = Transition.standardToNat( transDist )
        n3, = Transition.standardToNat( emissionDist )
        return n1, n2, n3

    @classmethod
    def natToStandard( cls, n1, n2, n3 ):
        initialDist, = Categorical.natToStandard( n1 )
        transDist, = Transition.natToStandard( n2 )
        emissionDist, = Transition.natToStandard( n3 )
        return initialDist, transDist, emissionDist

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, constParams=None ):
        # Compute T( x )
        ( x, ys ) = x
        assert constParams is not None
        K, L = constParams
        t1, = Categorical.sufficientStats( [ x[ 0 ] ] , constParams=K )
        t2, = Transition.sufficientStats( ( x[ :-1 ], x[ 1: ] ), constParams=( K, K ) )
        t3, = Transition.sufficientStats( ( x, ys.squeeze() ), constParams=( K, L ) )
        return t1, t2, t3

    @classmethod
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( natParams is None )
        initialDist, transDist, emissionDist = params if params is not None else cls.natToStandard( *natParams )
        A1 = Categorical.log_partition( params=( initialDist, ), split=split )
        A2 = Transition.log_partition( params=( transDist, ), split=split )
        A3 = Transition.log_partition( params=( emissionDist, ), split=split )
        return A1 + A2 + A3

    ######################################################################

    def genStates( self ):
        return np.empty( self.T, dtype=int )

    ######################################################################

    def sampleEmissions( self, x ):
        # Sample from P( y | x, ϴ )
        assert x.ndim == 1

        def sampleStep( _x ):
            p = self.emissionDist[ _x ]
            return Categorical.sample( params=( p, ) )

        return np.apply_along_axis( sampleStep, -1, x.reshape( ( -1, 1 ) ) ).ravel()[ None ]

    def emissionLikelihood( self, x, ys ):
        # Compute P( y | x, ϴ )
        return self._L[ x, ys ].sum()

    ######################################################################

    def sampleStep( self, p ):
        return int( Categorical.sample( natParams=( p, ) )[ 0 ] )

    def likelihoodStep( self, x, p ):
        return Categorical.log_likelihood( np.array( [ x ] ), natParams=( p, ) )

    ######################################################################

    def forwardArgs( self, t, beta, prevX ):
        # P( x_t+1 | x_t, y_t+1:T ) = P( y_t+1 | x_t+1 ) * P( x_t+1 | x_t ) * P( y_t+2:T | x_t+1 ) / P( y_t+1:T | x_t )
        #                           ∝ P( y_t+1 | x_t+1 ) * P( x_t+1 | x_t ) * P( y_t+2:T | x_t+1 )

        if( beta is None ):
            return ( self.pi[ prevX ], ) if t > 0 else ( self.pi0, )

        if( t == 0 ):
            unNormalized = self.pi0 + self.L[ t ] + beta
        else:
            unNormalized = self.pi[ prevX ] + self.L[ t ] + beta

        return ( unNormalized - np.logaddexp.reduce( unNormalized ), )

    def backwardArgs( self, t, alpha, prevX ):
        # P( x_t | x_t+1, y_1:t ) = P( x_t+1 | x_t ) * P( x_t, y_1:t ) / sum_{ z_t }[ P( x_t+1 | z_t ) * P( z_t, y_1:t ) ]
        #                         ∝ P( x_t+1 | x_t ) * P( x_t, y_1:t )

        unNormalized = alpha + self.pi[ prevX ] if t < self.T - 1 else alpha
        return ( unNormalized - np.logaddexp.reduce( unNormalized ), )
