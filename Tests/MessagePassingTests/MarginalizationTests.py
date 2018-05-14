import numpy as np
from GenModels.GM.States.MessagePassing import *
from GenModels.GM.Distributions import *
import time

__all__ = [ 'marginalizationTest' ]

######################################################################

def testCategoricalForwardBackward():

    T = 1000
    K = 20
    obsDim = 40
    D = 4

    mp = CategoricalForwardBackward()

    onesK = np.ones( K )
    onesObs = np.ones( obsDim )

    ( p, ) = Dirichlet.sample( params=onesObs )
    ys = [ Categorical.sample( params=p, size=T ) for _ in range( D ) ]
    ( initialDist, ) = Dirichlet.sample( params=onesK )
    transDist = Dirichlet.sample( params=onesK, size=K )
    emissionDist = Dirichlet.sample( params=onesObs, size=K )

    start = time.time()
    mp.updateParams( initialDist, transDist, emissionDist, ys )
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

    T = 100
    K = 20
    obsDim = 40
    D = 4

    mp = GaussianForwardBackward()

    onesK = np.ones( K )

    ( p, ) = Dirichlet.sample( params=onesK )
    ys = np.random.random( ( D, T, obsDim ) )
    ( initialDist, ) = Dirichlet.sample( params=onesK )
    transDist = Dirichlet.sample( params=onesK, size=K )

    muSigmas = [ NormalInverseWishart.sample( D=obsDim ) for _ in range( K ) ]
    mus = [ mu for mu, sigma in muSigmas ]
    sigmas = [ sigma for mu, sigma in muSigmas ]

    start = time.time()
    mp.updateParams( initialDist, transDist, mus, sigmas, ys )
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

def testSLDSForwardBackward():

    T = 100
    D_latent = 20
    D_obs = 8
    D = 4

    mp = SLDSForwardBackward()

    onesK = np.ones( D_latent )

    ( p, ) = Dirichlet.sample( params=onesK )
    ys = np.random.random( ( D, T, D_latent ) )
    ( initialDist, ) = Dirichlet.sample( params=onesK )
    transDist = Dirichlet.sample( params=onesK, size=D_latent )

    u = np.random.random( ( T, D_latent ) )
    mu0, sigma0 = NormalInverseWishart.sample( D=D_latent )

    ASigmas = [ MatrixNormalInverseWishart.sample( D_in=D_latent, D_out=D_latent ) for _ in range( D_latent ) ]
    As = [ A for A, sigma in ASigmas ]
    sigmas = [ sigma for A, sigma in ASigmas ]

    start = time.time()
    mp.updateParams( initialDist, transDist, mu0, sigma0, u, As, sigmas, ys )
    end = time.time()
    print( 'Preprocess: ', end - start )


    start = time.time()
    alphas = mp.forwardFilter()
    betas = mp.backwardFilter()
    end = time.time()
    print( 'Both filters: ', end - start )


    forwardMarginal = mp.marginalForward( alphas[ -1 ] )
    backwardMarginal = mp.marginalBackward( betas[ 0 ] )

    assert np.isclose( forwardMarginal, backwardMarginal )

    print( 'Passed the SLDS forward backward marginal test!\n\n' )

######################################################################

def testKalmanFilter():

    T = 1000
    D_latent = 7
    D_obs = 3
    D = 4

    mp = KalmanFilter()

    u = np.random.random( ( T, D_latent ) )
    A, sigma = MatrixNormalInverseWishart.sample( D_in=D_latent, D_out=D_latent )

    C, R = MatrixNormalInverseWishart.sample( D_in=D_latent, D_out=D_obs )
    ys = [ Regression.sample( params=( C, R ), size=T )[ 1 ] for _ in range( D ) ]

    mu0, sigma0 = NormalInverseWishart.sample( D=D_latent )

    start = time.time()
    mp.updateParams( A, sigma, C, R, mu0, sigma0, u, ys )
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

    T = 100
    D_latent = 20
    D_obs = 7
    D = 4
    K = 5

    mp = SwitchingKalmanFilter()

    u = np.random.random( ( T, D_latent ) )
    ASigmas = [ MatrixNormalInverseWishart.sample( D_in=D_latent, D_out=D_latent ) for _ in range( K ) ]
    As = [ A for A, sigma in ASigmas ]
    sigmas = [ sigma for A, sigma in ASigmas ]

    C, R = MatrixNormalInverseWishart.sample( D_in=D_latent, D_out=D_obs )
    ys = [ Regression.sample( params=( C, R ), size=T )[ 1 ] for _ in range( D ) ]

    ( p, ) = Dirichlet.sample( params=np.ones( K ) )
    z = Categorical.sample( params=p, size=T )

    mu0, sigma0 = NormalInverseWishart.sample( D=D_latent )

    start = time.time()
    mp.updateParams( z, As, sigmas, C, R, mu0, sigma0, u, ys )
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

def marginalizationTest():

    testCategoricalForwardBackward()
    testGaussianForwardBackward()
    testSLDSForwardBackward()
    testKalmanFilter()
    testSwitchingKalmanFilter()

