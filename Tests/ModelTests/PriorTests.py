import autograd.numpy as np
from GenModels.GM.ModelPriors import *
from GenModels.GM.States.StandardStates import *
from GenModels.GM.Distributions import *
import time
import scipy

__all__ = [ 'priorTests' ]

######################################################################

def testHMMDirichletPriorBasic():
    with np.errstate( under='ignore', divide='raise', over='raise', invalid='raise' ):
        T = 40
        D_latent = 3
        D_obs = 2
        meas = 4
        size = 5

        alpha_0 = np.random.random( D_latent ) + 1
        alpha = np.random.random( ( D_latent, D_latent ) ) + 1
        L = np.random.random( ( D_latent, D_obs ) ) + 1

        prior = HMMDirichletPrior( alpha_0, alpha, L )
        state = HMMState( prior=prior )

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

def testLDSMNIWPriorBasic():

    with np.errstate( all='raise' ), scipy.special.errstate( all='raise' ):
        T = 10
        D_latent = 7
        D_obs = 3
        D = 4
        meas = 4
        size = 5

        LDSParams = {
            'mu_0': np.random.random( D_latent ),
            'kappa_0': np.random.random() * D_latent,
            'psi_0': InverseWishart.generate( D=D_latent ),
            'nu_0': D_latent,

            'M_trans': np.random.random( ( D_latent, D_latent ) ),
            'V_trans': InverseWishart.generate( D=D_latent ),
            'psi_trans': InverseWishart.generate( D=D_latent ),
            'nu_trans': D_latent,

            'M_emiss': np.random.random( ( D_obs, D_latent ) ),
            'V_emiss': InverseWishart.generate( D=D_latent ),
            'psi_emiss': InverseWishart.generate( D=D_obs ),
            'nu_emiss': D_obs
        }

        prior = LDSMNIWPrior( **LDSParams )
        state = LDSState( prior=prior )

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

        print( 'Done with basic LDS prior test' )

def priorTests():
    testHMMDirichletPriorBasic()
    testLDSMNIWPriorBasic()