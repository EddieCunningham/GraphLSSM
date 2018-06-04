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
        # K = 20
        # obsDim = 40
        # D = 4

        K = 3
        obsDim = 2
        D = 10

        alpha_0 = np.random.random( K ) + 1
        alpha = np.random.random( ( K, K ) ) + 1
        L = np.random.random( ( K, obsDim ) ) + 1

        params = {
            'alpha_0': alpha_0,
            'alpha': alpha,
            'L': L
        }

        hmm = HMMModel( **params )

        ys = HMMModel.generate( T=T, latentSize=K, obsSize=obsDim, size=D )
        print( ys )
        assert 0

        hmm.fit( ys=ys, method='gibbs', verbose=False )
        hmm.predict( T=10 )

        hmm.fit( ys=ys, method='cavi', verbose=False )
        hmm.predict( T=10 )

######################################################################

def modelTests():
    HMMModelTest()
