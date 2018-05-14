from GenModels.GM.States.StandardStates.StateBase import StateBase
from GenModels.GM.States.MessagePassing.ForwardBackward import *
from GenModels.GM.Distributions import Categorical

__all__ = [ 'HMMState' ]

class HMMState( StateBase, ForwardBackward ):

    def __init__( self ):
        pass

    def genState( self ):
        return np.empty( ( self.T, self.K ) )

    ######################################################################

    def sampleStep( self, p ):
        return Categorical.sample( natParams=( p, ) )

    def likelihoodStep( self, x, p ):
        return Categorical.log_likelihood( x, natParams=( p, ) )

    ######################################################################

    def forwardArgs( self, t, beta, prevX ):
        # P( x_t+1 | x_t, y_t+1:T ) = P( y_t+1 | x_t+1 ) * P( x_t+1 | x_t ) * P( y_t+2:T | x_t+1 ) / P( y_t+1:T | x_t )
        #                           ∝ P( y_t+1 | x_t+1 ) * P( x_t+1 | x_t ) * P( y_t+2:T | x_t+1 )

        if( beta is None ):
            return self.pi[ prevX ] if t > 0 else self.pi0

        if( t == 0 ):
            unNormalized = self.pi0 + self.L + beta
        else:
            unNormalized = self.pi[ prevX ] + self.L + beta

        return unNormalized / np.logadexp.reduce( unNormalized )

    def backwardArgs( self, t, alpha, prevX ):
        # P( x_t | x_t+1, y_1:t ) = P( x_t+1 | x_t ) * P( x_t, y_1:t ) / sum_{ z_t }[ P( x_t+1 | z_t ) * P( z_t, y_1:t ) ]
        #                         ∝ P( x_t+1 | x_t ) * P( x_t, y_1:t )

        unNormalized = alpha + self.pi[ prevX ] if t < self.T - 1 else alpha

        return unNormalized / np.logadexp.reduce( unNormalized )
