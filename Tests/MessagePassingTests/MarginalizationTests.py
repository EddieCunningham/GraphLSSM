import numpy as np
np.random.seed(2)
import sys
sys.path.append( '/Users/Eddie/GenModels' )

from GM.States.StandardStates.MessagePassing import MessagePassingBase, \
                                                    CategoricalForwardBackward, \
                                                    KalmanFilter, \
                                                    SwitchingKalmanFilter, \
                                                    GaussianForwardBackward
from GM.Distributions import MatrixNormalInverseWishart, \
                             NormalInverseWishart, \
                             Dirichlet, \
                             Categorical, \
                             Regression, \
                             Normal
from scipy.stats import dirichlet
import time

######################################################################

def testCategoricalForwardBackward():

    T = 1000
    K = 10
    obsDim = 20

    mp = CategoricalForwardBackward( T, K )

    onesK = np.ones( K )
    onesObs = np.ones( obsDim )

    ( p, ) = Dirichlet.sample( params=onesK )
    y = Categorical.sample( params=p, size=T )
    ( initialDist, ) = Dirichlet.sample( params=onesK )
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

    print( 'Passed the categorical forward backward marginal test!' )

######################################################################

def testKalmanFilter():

    T = 1000
    D_latent = 4
    D_obs = 2

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

######################################################################

def testSwitchingKalmanFilter():

    T = 1000
    D_latent = 4
    D_obs = 3
    K = 50

    mp = SwitchingKalmanFilter( T, D_latent, D_obs )

    u = np.random.random( ( T, D_latent ) )
    ASigmas = [ MatrixNormalInverseWishart.basicSample( D_latent, D_latent ) for _ in range( K ) ]
    As = [ A for A, sigma in ASigmas ]
    sigmas = [ sigma for A, sigma in ASigmas ]

    C, R = MatrixNormalInverseWishart.basicSample( D_obs, D_latent )
    _, y = Regression.sample( params=( C, R ), size=T )

    ( p, ) = Dirichlet.sample( params=np.ones( K ) )
    z = Categorical.sample( params=p, size=T )

    mu0, sigma0 = NormalInverseWishart.basicSample( D_latent )

    mp.updateParams( y, u, z, As, sigmas, C, R, mu0, sigma0 )

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

    print( 'Passed the switching kalman filter marginal test!' )

######################################################################

def testGaussianForwardBackward():

    T = 1000
    K = 10
    obsDim = 20

    mp = GaussianForwardBackward( T, K )

    onesK = np.ones( K )

    ( p, ) = Dirichlet.sample( params=onesK )
    y = np.random.random( ( T, obsDim ) )
    ( initialDist, ) = Dirichlet.sample( params=onesK )
    transDist = Dirichlet.sample( params=onesK, size=K )

    muSigmas = [ NormalInverseWishart.basicSample( obsDim ) for _ in range( K ) ]
    mus = [ mu for mu, sigma in muSigmas ]
    sigmas = [ sigma for mu, sigma in muSigmas ]

    mp.updateParams( y, initialDist, transDist, mus, sigmas )

    start = time.time()

    alphas = mp.forwardFilter()
    betas = mp.backwardFilter()

    end = time.time()
    print( end - start )

    marginal = np.logaddexp.reduce( alphas[ -1 ] )

    for a, b in zip( alphas, betas ):
        comp = np.logaddexp.reduce( a + b )
        assert np.isclose( comp, marginal ), comp - marginal

    print( 'Passed the gaussian forward backward marginal test!' )

######################################################################

######################################################################


testCategoricalForwardBackward()
testKalmanFilter()
testSwitchingKalmanFilter()
testGaussianForwardBackward()


