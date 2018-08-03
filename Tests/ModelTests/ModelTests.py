import numpy as np
from GenModels.GM.Models import *
from GenModels.GM.Distributions import *
import time
import scipy

__all__ = [ 'modelTests' ]

######################################################################

def HMMModelTest():

    with np.errstate( under='ignore', divide='raise', over='raise', invalid='raise' ):

        T = 10
        D_latent = 5
        D_obs = 4
        meas = 2
        size = 3

        alpha_0 = np.random.random( D_latent ) + 1
        alpha = np.random.random( ( D_latent, D_latent ) ) + 1
        L = np.random.random( ( D_latent, D_obs ) ) + 1

        params = {
            'alpha_0': alpha_0,
            'alpha': alpha,
            'L': L
        }

        hmm = HMMModel( **params )

        _, ys = HMMModel.generate( T=T, latentSize=D_latent, obsSize=D_obs, measurements=meas, size=size )

        hmm.fit( ys=ys, method='gibbs', nIters=500, burnIn=200, skip=2, verbose=True )
        marginal = hmm.state.ilog_marginal( ys )
        print( '\nParams' )
        for p in hmm.state.params:
            print( np.round( p, decimals=3 ) )
            print()
        print( 'MARGNIAL', marginal )

        hmm.fit( ys=ys, method='EM', nIters=1000, monitorMarginal=10, verbose=False )
        marginal = hmm.state.ilog_marginal( ys )
        print( '\nParams' )
        for p in hmm.state.params:
            print( np.round( p, decimals=3 ) )
            print()
        print( 'MARGNIAL', marginal )

        hmm.fit( ys=ys, method='cavi', maxIters=1000, verbose=False )
        elbo = hmm.state.iELBO( ys )
        print( '\nPrior mean field params' )
        for p in hmm.state.prior.mf_params:
            print( np.round( p, decimals=3 ) )
            print()
        print( 'ELBO', elbo )

######################################################################

def LDSModelTest():
    with np.errstate( all='raise' ), scipy.special.errstate( all='raise' ):
        T = 10
        D_latent = 2
        D_obs = 3
        meas = 1
        size = 1

        lds = LDSModel( **LDSModel._genericParams( D_latent, D_obs ) )

        u = np.random.random( ( T, D_latent ) )
        nBad = int( np.random.random() * T )
        badMask = np.random.choice( T, nBad )
        u[ badMask ] = np.nan
        u = None

        ( _, ys ), tru = LDSModel.generate( T=T, latentSize=D_latent, obsSize=D_obs, measurements=meas, size=size, stabilize=True, returnTrueParams=True )

        # Have abosultely no idea whats wrong with gibbs sampling here....
        # Going to move on to EM and CAVI and see if I can find the bug
        # lds.fit( ys=ys,u=u, method='gibbs', nIters=500, burnIn=200, skip=2, verbose=True )
        # marginal = lds.state.ilog_marginal( ys )
        # print( '\nParams' )
        # for p in lds.state.params:
        #     print( np.round( p, decimals=3 ) )
        #     print()
        # print( 'MARGNIAL', marginal )

        # fuck this shit
        # lds.fit( ys=ys, u=u, method='EM', nIters=100000, monitorMarginal=10, verbose=False )
        # marginal = lds.state.ilog_marginal( ys )
        # print( '\nParams' )
        # for p in lds.state.params:
        #     print( np.round( p, decimals=3 ) )
        #     print()
        # print( 'MARGNIAL', marginal )

        lds.fit( ys=ys, u=u, method='cavi', maxIters=1000, verbose=False )
        elbo = lds.state.iELBO( ys )
        print( '\nPrior mean field params' )
        for p in lds.state.prior.mf_params:
            print( np.round( p, decimals=3 ) )
            print()
        print( 'ELBO', elbo )

######################################################################

def modelTests():
    # HMMModelTest()
    LDSModelTest()
