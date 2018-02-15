import numpy as np
# np.random.seed(2)
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
    K = 200
    obsDim = 40
    D = 40

    mp = CategoricalForwardBackward( T, K )

    onesK = np.ones( K )
    onesObs = np.ones( obsDim )

    ( p, ) = Dirichlet.sample( params=onesObs )
    ys = [ Categorical.sample( params=p, size=T ) for _ in range( D ) ]
    ( initialDist, ) = Dirichlet.sample( params=onesK )
    transDist = Dirichlet.sample( params=onesK, size=K )
    emissionDist = Dirichlet.sample( params=onesObs, size=K )

    start = time.time()
    mp.updateParams( ys, initialDist, transDist, emissionDist )
    end = time.time()
    print( 'Preprocess: ', end - start )


    start = time.time()
    alphas = mp.forwardFilter()
    betas = mp.backwardFilter()
    end = time.time()
    print( 'Both filters: ', end - start )

    marginal = np.logaddexp.reduce( alphas[ -1 ] )

    for a, b in zip( alphas, betas ):
        comp = np.logaddexp.reduce( a + b )
        assert np.isclose( comp, marginal ), comp - marginal

    print( 'Passed the categorical forward backward marginal test!\n\n' )

######################################################################

def testGaussianForwardBackward():

    T = 1000
    K = 200
    obsDim = 40
    D = 40

    mp = GaussianForwardBackward( T, K )

    onesK = np.ones( K )

    ( p, ) = Dirichlet.sample( params=onesK )
    ys = np.random.random( ( D, T, obsDim ) )
    ( initialDist, ) = Dirichlet.sample( params=onesK )
    transDist = Dirichlet.sample( params=onesK, size=K )

    muSigmas = [ NormalInverseWishart.basicSample( obsDim ) for _ in range( K ) ]
    mus = [ mu for mu, sigma in muSigmas ]
    sigmas = [ sigma for mu, sigma in muSigmas ]

    start = time.time()
    mp.updateParams( ys, initialDist, transDist, mus, sigmas )
    end = time.time()
    print( 'Preprocess: ', end - start )


    start = time.time()
    alphas = mp.forwardFilter()
    betas = mp.backwardFilter()
    end = time.time()
    print( 'Both filters: ', end - start )

    marginal = np.logaddexp.reduce( alphas[ -1 ] )

    for a, b in zip( alphas, betas ):
        comp = np.logaddexp.reduce( a + b )
        assert np.isclose( comp, marginal ), comp - marginal

    print( 'Passed the gaussian forward backward marginal test!\n\n' )

######################################################################

def testKalmanFilter():

    T = 1000
    D_latent = 200
    D_obs = 40
    D = 40

    mp = KalmanFilter( T, D_latent, D_obs )

    u = np.random.random( ( T, D_latent ) )
    A, sigma = MatrixNormalInverseWishart.basicSample( D_latent, D_latent )

    C, R = MatrixNormalInverseWishart.basicSample( D_obs, D_latent )
    ys = [ Regression.sample( params=( C, R ), size=T )[ 1 ] for _ in range( D ) ]

    mu0, sigma0 = NormalInverseWishart.basicSample( D_latent )

    start = time.time()
    mp.updateParams( ys, u, A, sigma, C, R, mu0, sigma0 )
    end = time.time()
    print( 'Preprocess: ', end - start )

    start = time.time()
    alphas = mp.forwardFilter()
    betas = mp.backwardFilter()
    end = time.time()
    print( 'Both filters: ', end - start )

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

    print( 'Passed the kalman filter marginal test!\n\n' )

######################################################################

def testSwitchingKalmanFilter():

    T = 1000
    D_latent = 200
    D_obs = 40
    D = 40
    K = 50

    mp = SwitchingKalmanFilter( T, D_latent, D_obs )

    u = np.random.random( ( T, D_latent ) )
    ASigmas = [ MatrixNormalInverseWishart.basicSample( D_latent, D_latent ) for _ in range( K ) ]
    As = [ A for A, sigma in ASigmas ]
    sigmas = [ sigma for A, sigma in ASigmas ]

    C, R = MatrixNormalInverseWishart.basicSample( D_obs, D_latent )
    ys = [ Regression.sample( params=( C, R ), size=T )[ 1 ] for _ in range( D ) ]

    ( p, ) = Dirichlet.sample( params=np.ones( K ) )
    z = Categorical.sample( params=p, size=T )

    mu0, sigma0 = NormalInverseWishart.basicSample( D_latent )

    start = time.time()
    mp.updateParams( ys, u, z, As, sigmas, C, R, mu0, sigma0 )
    end = time.time()
    print( 'Preprocess: ', end - start )

    start = time.time()
    alphas = mp.forwardFilter()
    betas = mp.backwardFilter()
    end = time.time()
    print( 'Both filters: ', end - start )

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

    print( 'Passed the switching kalman filter marginal test!\n\n' )

######################################################################

testCategoricalForwardBackward()
testGaussianForwardBackward()
testKalmanFilter()
testSwitchingKalmanFilter()


