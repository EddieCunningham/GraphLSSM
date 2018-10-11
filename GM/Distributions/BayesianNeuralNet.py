import autograd.numpy as np
from GenModels.GM.Distributions.Normal import Normal
from GenModels.GM.Distributions.TensorNormal import TensorNormal
from GenModels.GM.Utility import logsumexp, extendAxes, logMultiplyTerms, logIntegrate

__all__ = [ 'BayesianNN' ]

# This is kind of ugly.  Will re-implement better in the future

def unitCov( b ):
    return np.eye( b.shape[ 0 ] )
def unitCovs( W ):
    return np.eye( W.shape[ 0 ] ), np.eye( W.shape[ 1 ] )
def covsHelp( W ):
    return { 'params1': ( W, unitCovs( W ) ), 'params2':( np.zeros_like( W ), unitCovs( W ) ) }
def covHelp( b ):
    return { 'params1': ( b, unitCov( b ) ), 'params2':( np.zeros_like( b ), unitCov( b ) ) }

######################################################################

def logadd( log_a, log_b ):
    max_a = np.max( log_a )
    max_b = np.max( log_b )
    maximum = np.max( [ max_a, max_b ] )
    return np.log( np.exp( log_a - maximum ) + np.exp( log_b - maximum ) ) + maximum

######################################################################

class BayesianNN():

    # Assume that there is a unit gaussian over every weight

    def __init__( self, d_in, d_out, recognizer_hidden_size=4, generative_hidden_size=4 ):

        self.d_in = d_in
        self.d_out = d_out

        Wr1 = TensorNormal.generate( Ds=( recognizer_hidden_size, d_out + 3 ) )[ 0 ]
        br1 = Normal.generate( D=recognizer_hidden_size )
        Wr2 = TensorNormal.generate( Ds=( d_in, recognizer_hidden_size ) )[ 0 ]
        br2 = Normal.generate( D=d_in )

        Wg1 = TensorNormal.generate( Ds=( generative_hidden_size, d_in + 3 ) )[ 0 ]
        bg1 = Normal.generate( D=generative_hidden_size )
        Wg2 = TensorNormal.generate( Ds=( d_out, generative_hidden_size ) )[ 0 ]
        bg2 = Normal.generate( D=d_out )

        self.recognizer_params = [ ( Wr1, br1 ), ( Wr2, br2 ) ]
        self.generative_hyper_params = [ ( Wg1, bg1 ), ( Wg2, bg2 ) ]

    def recognize( self, y, cond, recognizer_params=None ):
        assert recognizer_params is not None

        sex, age, affected, n_above, n_below = cond

        if( age == -1 ):
            age = 0

        # Turn y into a one hot vector
        y_one_hot = np.zeros( ( y.shape[ 0 ], self.d_out ) )
        y_one_hot[ np.arange( y.shape[ 0 ] ), y ] = 1.0

        # Stack all of the inputs
        input_layer = np.hstack( ( y_one_hot, [ [ age, n_above, n_below ] ] ) )

        last_layer = np.log( input_layer )

        for W, b in recognizer_params[ :-1 ]:
            # last_layer = np.tanh( np.einsum( 'ij,tj->ti', W, last_layer ) + b[ None ] )
            k = logsumexp( W[ None, :, : ] + last_layer[ :, None, : ], axis=2 )
            last_layer = logadd( k, b[ None ] )
            last_layer = last_layer - logsumexp( last_layer, axis=1 )[ None ]

        W, b = recognizer_params[ -1 ]

        # last_layer = np.einsum( 'ij,tj->i', W, last_layer ) + b
        last_layer = logsumexp( W[ None, :, : ] + last_layer[ :, None, : ], axis=2 )
        last_layer = logsumexp( last_layer, axis=0 )
        last_layer = logadd( last_layer, b )

        return last_layer

    def sampleGenerativeParams( self, generative_hyper_params ):
        ( Wg1, bg1 ), ( Wg2, bg2 ) = generative_hyper_params

        w1 = TensorNormal.sample( params=( Wg1, unitCovs( Wg1 ) ), size=1 )[ 0 ]
        b1 = Normal.sample( params=( bg1, unitCov( bg1 ) ), size=1 )[ 0 ]
        w2 = TensorNormal.sample( params=( Wg2, unitCovs( Wg2 ) ), size=1 )[ 0 ]
        b2 = Normal.sample( params=( bg2, unitCov( bg2 ) ), size=1 )[ 0 ]

        return ( w1, b1 ), ( w2, b2 )

    def KLPQ( self, q_params ):
        ( Wg1, bg1 ), ( Wg2, bg2 ) = q_params

        ans = 0.0
        ans += TensorNormal.KLDivergence( **covsHelp( Wg1 ) )
        ans += Normal.KLDivergence( **covHelp( bg1 ) )
        ans += TensorNormal.KLDivergence( **covsHelp( Wg2 ) )
        ans += Normal.KLDivergence( **covHelp( bg2 ) )

        return ans

    def log_likelihood( self, x, y, cond, generative_params ):
        sex, age, affected, n_above, n_below = cond

        # This is ok because x is an array of logits
        last_layer = np.exp( x )
        last_layer = np.hstack( ( last_layer, age, n_above, n_below ) )

        for W, b in generative_params[ :-1 ]:
            last_layer = np.tanh( np.einsum( 'ij,j->i', W, last_layer ) + b )

        W, b = generative_params[ -1 ]

        last_layer = np.einsum( 'ij,j->i', W, last_layer ) + b
        logits = last_layer - logsumexp( last_layer, axis=0 )

        y_one_hot = np.zeros( ( y.shape[ 0 ], self.d_out ) )
        y_one_hot[ np.arange( y.shape[ 0 ] ), y ] = 1.0

        return np.einsum( 'ti,i->', y_one_hot, logits )
