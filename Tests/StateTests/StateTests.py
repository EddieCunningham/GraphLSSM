import numpy as np
from GenModels.GM.States.StandardStates import *
from GenModels.GM.Distributions import *
import time
import scipy

__all__ = [ 'stateTests' ]

######################################################################

def testHMMBasic():
    with np.errstate( under='ignore', divide='raise', over='raise', invalid='raise' ):
        T = 40
        D_latent = 3
        D_obs = 2
        meas = 4
        size = 5

        initialDist = Dirichlet.generate( D=D_latent )
        transDist = TransitionDirichletPrior.generate( D_in=D_latent, D_out=D_latent )
        emissionDist = TransitionDirichletPrior.generate( D_in=D_latent, D_out=D_obs )

        state = HMMState( initialDist=initialDist, transDist=transDist, emissionDist=emissionDist )

        _, ys = HMMState.generate( measurements=meas, T=T, D_latent=D_latent, D_obs=D_obs, size=size )

        xNoCond  , ysNoCond  = state.isample( T=T, measurements=meas, size=size )
        xForward , yForward  = state.isample( ys=ys )
        xBackward, yBackward = state.isample( ys=ys, forwardFilter=False )

        print( state.ilog_likelihood( ( xNoCond, ysNoCond ) ) )
        print( state.ilog_likelihood( ( xForward, yForward ), forwardFilter=False ) )
        print( state.ilog_likelihood( ( xBackward, yBackward ) ) )

        print( state.ilog_likelihood( ( xNoCond, ysNoCond ), conditionOnY=True ) )
        print( state.ilog_likelihood( ( xForward, yForward ), forwardFilter=False, conditionOnY=True ) )
        print( state.ilog_likelihood( ( xBackward, yBackward ), conditionOnY=True ) )

        print( 'Done with basic HMM state test' )

def testLDSBasic():

    with np.errstate( all='raise' ), scipy.special.errstate( all='raise' ):

        T = 100
        D_latent = 7
        D_obs = 3
        D = 4

        u = np.random.random( ( T, D_latent ) )
        A, sigma = MatrixNormalInverseWishart.sample( D_in=D_latent, D_out=D_latent )

        C, R = MatrixNormalInverseWishart.sample( D_in=D_latent, D_out=D_obs )
        ys = [ Regression.sample( params=( C, R ), size=T )[ 1 ] for _ in range( D ) ]

        mu0, sigma0 = NormalInverseWishart.sample( D=D_latent )

        state = LDSState( A, sigma, C, R, mu0, sigma0 )

        xNoCond  , ysNoCond  = state.isample( u=u, T=T ) # This can get unstable for long sequence lengths
        xForward , yForward  = state.isample( u=u, ys=ys )
        xBackward, yBackward = state.isample( u=u, ys=ys, forwardFilter=False )

        state.ilog_likelihood( ( xNoCond, ysNoCond[ 0 ][ None ] ), u=u )
        state.ilog_likelihood( ( xForward, yForward[ 0 ][ None ] ), u=u, forwardFilter=False )
        state.ilog_likelihood( ( xBackward, yBackward[ 0 ][ None ] ), u=u )

        state.ilog_likelihood( ( xNoCond, ysNoCond[ 0 ][ None ] ), u=u, conditionOnY=True )
        state.ilog_likelihood( ( xForward, yForward[ 0 ][ None ] ), u=u, forwardFilter=False, conditionOnY=True )
        state.ilog_likelihood( ( xBackward, yBackward[ 0 ][ None ] ), u=u, conditionOnY=True )

        print( 'Done with basic LDS state test' )

def stateTests():
    testHMMBasic()
    testLDSBasic()