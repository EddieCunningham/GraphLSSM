from GenModels.GM.States.StandardStates.StateBase import StateBase
from GenModels.GM.States.MessagePassing.ForwardBackward import *
from GenModels.GM.Distributions import Categorical, Transition, Dirichlet, TransitionDirichletPrior
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
        return self.D_latent, self.D_obs

    @classmethod
    def dataN( cls, x, conditionOnY=False, checkY=False ):
        cls.checkShape( x, conditionOnY=conditionOnY, checkY=checkY )
        if( conditionOnY == False ):
            x, y = x
        if( isinstance( x, tuple ) ):
            return len( x )
        return 1

    @classmethod
    def sequenceLength( cls, x, conditionOnY=False, checkY=False ):
        cls.checkShape( x, conditionOnY=conditionOnY, checkY=checkY )

        if( cls.dataN( x, conditionOnY=conditionOnY, checkY=checkY ) == 1 ):
            if( conditionOnY == False ):
                xs, ys = x
                return xs.shape[ 0 ]

            if( checkY == False ):
                return x.shape[ 0 ]
            else:
                return x.shape[ 1 ]
        else:
            assert 0, 'Only pass in a single example'

    @classmethod
    def unpackSingleSample( cls, x, conditionOnY=False, checkY=False ):
        if( conditionOnY == False ):
            xs, ys = x
            return xs[ 0 ], ys[ 0 ]
        return x[ 0 ]

    @classmethod
    def sampleShapes( cls, conditionOnY=False ):
        # We can have multiple measurements for the same latent state
        # ( ( Sample #, time ), ( Sample #, measurement #, time ) )
        if( conditionOnY == False ):
            return ( ( None, None ), ( None, None, None ) )
        return ( None, None )

    def isampleShapes( cls, conditionOnY=False ):
        if( conditionOnY == False ):
            return ( ( None, self.T ), ( None, self.T, None ) )
        return ( None, self.T )

    @classmethod
    def checkShape( cls, x, conditionOnY=False, checkY=False ):
        if( conditionOnY == False ):
            xs, ys = x
            if( isinstance( xs, tuple ) ):
                assert isinstance( ys, tuple )
                assert len( xs ) == len( ys )
                for x, y in zip( xs, ys ):
                    assert x.shape[ 0 ] == y.shape[ 1 ]
            else:
                assert isinstance( xs, np.ndarray )
                assert isinstance( ys, np.ndarray )
                assert xs.ndim == 1
                assert ys.ndim == 1
                assert xs[ 0 ].shape == ys[ 1 ].shape
        else:
            if( isinstance( x, tuple ) ):
                for _x in x:
                    if( checkY == True ):
                        assert _x.ndim == 2 or _x.ndim == 1
                    else:
                        assert _x.ndim == 1
            else:
                if( checkY == True ):
                    assert x.ndim == 2 or x.ndim == 1
                else:
                    assert x.ndim == 1

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
            _x = _x[ 0 ]
            p = self.emissionDist[ _x ]
            return Categorical.sample( params=( p, ) )

        return np.apply_along_axis( sampleStep, -1, x.reshape( ( -1, 1 ) ) ).ravel()[ None ]

    def emissionLikelihood( self, x, ys ):
        # Compute P( y | x, ϴ )
        return self._L[ x, ys ].sum()

    ######################################################################

    def sampleStep( self, p ):
        return int( Categorical.sample( natParams=( p, ) ) )

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
        normalized = unNormalized - np.logaddexp.reduce( unNormalized )
        return ( normalized, )

    ######################################################################

    def conditionedExpectedSufficientStats( self, alphas, betas ):

        t = np.zeros_like( self.pi )

        for t in range( self.T - 1 ):
            L = self.emissionProb( t + 1 )
            pi = self.transitionProb( t, t + 1 )
            jointPi = self.multiplyTerms( [ L, pi, alphas[ t ], betas[ t + 1 ] ] )
            t += jointPi

        return t

    ######################################################################

    @classmethod
    def generate( cls, measurements=4, T=5, D_latent=3, D_obs=2, size=1 ):
        initialDist = Dirichlet.generate( D=D_latent )
        transDist = TransitionDirichletPrior.generate( D_in=D_latent, D_out=D_latent )
        emissionDist = TransitionDirichletPrior.generate( D_in=D_latent, D_out=D_obs )

        dummy = cls( initialDist, transDist, emissionDist )
        return dummy.isample( measurements=measurements, T=T, size=size )