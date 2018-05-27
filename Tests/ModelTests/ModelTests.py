import numpy as np
from GenModels.GM.ModelPriors import *
from GenModels.GM.States.StandardStates import *
from GenModels.GM.Distributions import *
import time
import scipy

__all__ = [ 'modelTests' ]

######################################################################

def testHMMDirichletPriorBasic():
    with np.errstate( under='ignore', divide='raise', over='raise', invalid='raise' ):
        T = 1000
        K = 20
        obsDim = 40
        D = 4

        alpha_0 = np.random.random( K ) + 1
        alpha = np.random.random( ( K, K ) ) + 1
        L = np.random.random( ( K, obsDim ) ) + 1

        prior = HMMDirichletPrior( alpha_0, alpha, L )

        state = HMMState( prior=prior )

        ( p, ) = Dirichlet.sample( params=np.ones( obsDim ) )
        ys = [ Categorical.sample( params=p, size=T ) for _ in range( D ) ]
        xNoCond  , ysNoCond  = state.isample( T=10 )
        xForward , yForward  = state.isample( ys=ys )
        xBackward, yBackward = state.isample( ys=ys, forwardFilter=False )

        state.ilog_likelihood( ( xNoCond, ysNoCond[ 0 ][ None ] ) )
        state.ilog_likelihood( ( xForward, yForward[ 0 ][ None ] ), forwardFilter=False )
        state.ilog_likelihood( ( xBackward, yBackward[ 0 ][ None ] ) )

        state.ilog_likelihood( ( xNoCond, ysNoCond[ 0 ][ None ] ), conditionOnY=True )
        state.ilog_likelihood( ( xForward, yForward[ 0 ][ None ] ), forwardFilter=False, conditionOnY=True )
        state.ilog_likelihood( ( xBackward, yBackward[ 0 ][ None ] ), conditionOnY=True )

        print( 'Done with basic HMM prior test' )

def testLDSMNIWPriorBasic():

    with np.errstate( all='raise' ), scipy.special.errstate( all='raise' ):
        T = 10
        D_latent = 7
        D_obs = 3
        D = 4

        LDSParams = {
            'mu_0': np.random.random( D_latent ),
            'kappa_0': np.random.random() * D_latent,
            'psi_0': InverseWishart.sample( D=D_latent ),
            'nu_0': D_latent,

            'M_trans': np.random.random( ( D_latent, D_latent ) ),
            'V_trans': InverseWishart.sample( D=D_latent ),
            'psi_trans': InverseWishart.sample( D=D_latent ),
            'nu_trans': D_latent,

            'M_emiss': np.random.random( ( D_obs, D_latent ) ),
            'V_emiss': InverseWishart.sample( D=D_latent ),
            'psi_emiss': InverseWishart.sample( D=D_obs ),
            'nu_emiss': D_obs
        }

        prior = LDSMNIWPrior( **LDSParams )
        state = LDSState( prior=prior )

        u = np.random.random( ( T, D_latent ) )
        C, R = MatrixNormalInverseWishart.sample( D_in=D_latent, D_out=D_obs )
        ys = [ Regression.sample( params=( C, R ), size=T )[ 1 ] for _ in range( D ) ]

        xNoCond  , ysNoCond  = state.isample( u=u, T=T ) # This can get unstable for long sequence lengths
        xForward , yForward  = state.isample( u=u, ys=ys )
        xBackward, yBackward = state.isample( u=u, ys=ys, forwardFilter=False )

        state.ilog_likelihood( ( xNoCond, ysNoCond[ 0 ][ None ] ), u=u )
        state.ilog_likelihood( ( xForward, yForward[ 0 ][ None ] ), u=u, forwardFilter=False )
        state.ilog_likelihood( ( xBackward, yBackward[ 0 ][ None ] ), u=u )

        state.ilog_likelihood( ( xNoCond, ysNoCond[ 0 ][ None ] ), u=u, conditionOnY=True )
        state.ilog_likelihood( ( xForward, yForward[ 0 ][ None ] ), u=u, forwardFilter=False, conditionOnY=True )
        state.ilog_likelihood( ( xBackward, yBackward[ 0 ][ None ] ), u=u, conditionOnY=True )

        print( 'Done with basic LDS prior test' )

def modelTests():
    testHMMDirichletPriorBasic()
    testLDSMNIWPriorBasic()