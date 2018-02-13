import numpy as np
np.random.seed(2)
import sys
sys.path.append( '/Users/Eddie/GenModels' )

from GM.States.StandardStates.MessagePassing import MessagePassingBase, \
                                                    CategoricalForwardBackward, \
                                                    KalmanFilter
from GM.Distributions import MatrixNormalInverseWishart, \
                             NormalInverseWishart, \
                             Dirichlet, \
                             Categorical, \
                             Regression, \
                             Normal
from scipy.stats import dirichlet
import time

def testForwardBackward():

    T = 100000
    K = 10
    obsDim = 20

    mp = CategoricalForwardBackward( T, K )

    onesK = np.ones( K )
    onesObs = np.ones( obsDim )

    y = Categorical.sample( params=Dirichlet.sample( params=onesK ), size=T )
    initialDist = Dirichlet.sample( params=onesK )
    transDist = Dirichlet.sample( params=onesK, size=K )
    emissionDist = Dirichlet.sample( params=onesObs, size=K )

    mp.updateParams( y, initialDist, transDist, emissionDist )

    start = time.time()

    alphas = mp.forwardFilter()
    betas = mp.backwardFilter()

    end = time.time()
    print( end - start )

    marginal = np.logaddexp.reduce( alphas[ -1 ] )

    for a, b in zip( alphas, betas ):
        comp = np.logaddexp.reduce( a + b )
        assert np.isclose( comp, marginal ), comp - marginal

    print( 'Passed the forward backward marginal test!' )

def testKalmanFilter():

    T = 10000
    D_latent = 20
    D_obs = 20

    mp = KalmanFilter( T, D_latent, D_obs )

    u = np.random.random( ( T, D_latent ) )
    A, sigma = MatrixNormalInverseWishart.basicSample( D_latent, D_latent )

    C, R = MatrixNormalInverseWishart.basicSample( D_obs, D_latent )
    _, y = Regression.sample( params=( C, R ), size=T )

    mu0, sigma0 = NormalInverseWishart.basicSample( D_latent )

    mp.updateParams( y, u, A, sigma, C, R, mu0, sigma0 )

    start = time.time()

    alphas = mp.forwardFilter()
    betas = mp.backwardFilter()

    end = time.time()
    print( end - start )

    Ja, ha, log_Za = alphas[ -1 ]
    Jb, hb, log_Zb = betas[ -1 ]

    marginal = Normal.log_partition( natParams=( -0.5*Ja, ha ) ) - log_Za

    last=None
    for a, b in zip( alphas, betas ):
        Ja, ha, log_Za = a
        Jb, hb, log_Zb = b

        comp = Normal.log_partition( natParams=( -0.5*( Ja + Jb ), ( ha + hb ) ) ) - ( log_Za + log_Zb )

        if( last is None ):
            last = comp

        assert np.isclose( comp, marginal ), comp - marginal

    print( 'Passed the kalman filter marginal test!' )

testForwardBackward()
testKalmanFilter()


