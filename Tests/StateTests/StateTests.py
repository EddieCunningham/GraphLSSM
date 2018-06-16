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

        kS = int( np.random.random() * T / 10 ) + 2
        knownStates = np.random.choice( T, kS )
        knownStates = np.vstack( ( knownStates, np.random.choice( D_latent, knownStates.shape[ 0 ] ) ) ).reshape( ( 2, -1 ) ).T

        # Sort and remove duplicates
        knownStates = np.array( sorted( knownStates, key=lambda x: x[ 0 ] ) )
        knownStates = knownStates[ 1: ][ ~( np.diff( knownStates[ :, 0 ] ) == 0 ) ]

        xNoCond  , ysNoCond  = state.isample( T=T, measurements=meas, size=size )
        xForward , yForward  = state.isample( ys=ys, knownLatentStates=knownStates )
        xBackward, yBackward = state.isample( ys=ys, forwardFilter=False )

        state.ilog_likelihood( ( xNoCond, ysNoCond ) )
        state.ilog_likelihood( ( xForward, yForward ) )
        state.ilog_likelihood( ( xBackward, yBackward ), forwardFilter=False )

        state.ilog_likelihood( ( xNoCond, ysNoCond ), conditionOnY=True )
        state.ilog_likelihood( ( xForward, yForward ), knownLatentStates=knownStates, conditionOnY=True )
        state.ilog_likelihood( ( xBackward, yBackward ), forwardFilter=False, conditionOnY=True )

        print( 'Done with basic HMM state test' )

def testLDSBasic():

    with np.errstate( all='raise' ), scipy.special.errstate( all='raise' ):

        T = 40
        D_latent = 3
        D_obs = 2
        meas = 4
        size = 5

        A, sigma = MatrixNormalInverseWishart.generate( D_in=D_latent, D_out=D_latent )
        C, R = MatrixNormalInverseWishart.generate( D_in=D_latent, D_out=D_obs )
        mu0, sigma0 = NormalInverseWishart.generate( D=D_latent )

        state = LDSState( A=A, sigma=sigma, C=C, R=R, mu0=mu0, sigma0=sigma0 )

        u = np.random.random( ( T, D_latent ) )
        nBad = int( np.random.random() * T )
        badMask = np.random.choice( T, nBad )
        u[ badMask ] = np.nan

        _, ys = LDSState.generate( measurements=meas, T=T, D_latent=D_latent, D_obs=D_obs, size=size, stabilize=True )

        xNoCond  , ysNoCond  = state.isample( u=u, T=T, measurements=meas, size=size, stabilize=True )
        xForward , yForward  = state.isample( u=u, ys=ys )
        xBackward, yBackward = state.isample( u=u, ys=ys, forwardFilter=False )

        state.ilog_likelihood( ( xNoCond, ysNoCond ), u=u )
        state.ilog_likelihood( ( xForward, yForward ), u=u, forwardFilter=False )
        state.ilog_likelihood( ( xBackward, yBackward ), u=u )

        state.ilog_likelihood( ( xNoCond, ysNoCond ), u=u, conditionOnY=True )
        state.ilog_likelihood( ( xForward, yForward ), u=u, forwardFilter=False, conditionOnY=True )
        state.ilog_likelihood( ( xBackward, yBackward ), u=u, conditionOnY=True )

        print( 'Done with basic LDS state test' )

def stateTests():
    testHMMBasic()
    testLDSBasic()
