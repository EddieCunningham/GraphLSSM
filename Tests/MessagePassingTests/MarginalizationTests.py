import numpy as np
from GenModels.GM.States.MessagePassing import *
from GenModels.GM.Distributions import *
import time

__all__ = [ 'marginalizationTest' ]

######################################################################

def testCategoricalForwardBackward():

    T = 400
    D_latent = 22
    D_obs = 31
    measurements = 2

    mp = CategoricalForwardBackward()

    initialDist = Dirichlet.generate( D=D_latent )
    transDist = TransitionDirichletPrior.generate( D_in=D_latent, D_out=D_latent )
    emissionDist = TransitionDirichletPrior.generate( D_in=D_latent, D_out=D_obs )

    ys = [ Categorical.generate( D=D_obs, size=T ) for _ in range( measurements ) ]

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
        # comp = np.logaddexp.reduce( a + b )
        comp = mp.log_marginalFromAlphaBeta( a, b )
        assert np.isclose( comp, marginal ), comp - marginal

    for t in range( T - 1 ):
        joint = mp.childParentJoint( t, alphas, betas )

        parentProb = np.logaddexp.reduce( joint, axis=1 )
        childProb = np.logaddexp.reduce( joint, axis=0 )

        trueParent = alphas[ t ] + betas[ t ]
        trueChild = alphas[ t + 1 ] + betas[ t + 1 ]

        assert np.allclose( parentProb, trueParent )
        assert np.allclose( childProb, trueChild )

    print( 'Passed the categorical forward backward marginal test!\n\n' )

def testCategoricalForwardBackwardWithKnownStates():

    T = 50
    K = 3
    obsDim = 2
    D = 3

    mp = CategoricalForwardBackward()

    initialDist = Dirichlet.generate( D=K )
    transDist = TransitionDirichletPrior.generate( D_in=K, D_out=K )
    emissionDist = TransitionDirichletPrior.generate( D_in=K, D_out=obsDim )

    ys = [ Categorical.generate( D=obsDim, size=T ) for _ in range( D ) ]

    start = time.time()
    mp.updateParams( initialDist, transDist, emissionDist, ys )
    end = time.time()
    print( 'Preprocess: ', end - start )

    kS = int( np.random.random() * T / 10 ) + 2
    knownStates = np.random.choice( T, kS )
    knownStates = np.vstack( ( knownStates, np.random.choice( K, knownStates.shape[ 0 ] ) ) ).reshape( ( 2, -1 ) ).T

    # Sort and remove duplicates
    knownStates = np.array( sorted( knownStates, key=lambda x: x[ 0 ] ) )
    knownStates = knownStates[ 1: ][ ~( np.diff( knownStates[ :, 0 ] ) == 0 ) ]

    # print( knownStates )

    start = time.time()
    alphas = mp.forwardFilter( knownLatentStates=knownStates )
    betas = mp.backwardFilter( knownLatentStates=knownStates )
    end = time.time()
    print( 'Both filters: ', end - start )

    marginal = np.logaddexp.reduce( alphas[ -1 ] )
    for a, b in zip( alphas, betas ):
        # print( a + b )
        # comp = np.logaddexp.reduce( a + b )
        comp = mp.log_marginalFromAlphaBeta( a, b )
        assert np.isclose( comp, marginal ), comp - marginal

    for t in range( T - 1 ):
        joint = mp.childParentJoint( t, alphas, betas )

        parentProb = np.logaddexp.reduce( joint, axis=1 )
        childProb = np.logaddexp.reduce( joint, axis=0 )

        trueParent = alphas[ t ] + betas[ t ]
        trueChild = alphas[ t + 1 ] + betas[ t + 1 ]

        assert np.allclose( parentProb, trueParent )
        assert np.allclose( childProb, trueChild )

    print( 'Passed the categorical forward backward marginal test with known states!\n\n' )

######################################################################

def testGaussianForwardBackward():

    T = 100
    K = 20
    obsDim = 40
    D = 4

    mp = GaussianForwardBackward()

    initialDist = Dirichlet.generate( D=K )
    transDist = TransitionDirichletPrior.generate( D_in=K, D_out=K )

    mus, sigmas = list( zip( *[ NormalInverseWishart.generate( D=obsDim ) for _ in range( K ) ] ) )

    ys = np.random.random( ( D, T, obsDim ) )

    start = time.time()
    mp.updateParams( initialDist, transDist, mus, sigmas, ys )
    end = time.time()
    print( 'Preprocess: ', end - start )

    kS = int( np.random.random() * T / 10 ) + 2
    knownStates = np.random.choice( T, kS )
    knownStates = np.vstack( ( knownStates, np.random.choice( K, knownStates.shape[ 0 ] ) ) ).reshape( ( 2, -1 ) ).T

    # Sort and remove duplicates
    knownStates = np.array( sorted( knownStates, key=lambda x: x[ 0 ] ) )
    knownStates = knownStates[ 1: ][ ~( np.diff( knownStates[ :, 0 ] ) == 0 ) ]

    start = time.time()
    alphas = mp.forwardFilter( knownLatentStates=knownStates )
    betas = mp.backwardFilter( knownLatentStates=knownStates )
    end = time.time()
    print( 'Both filters: ', end - start )

    marginal = np.logaddexp.reduce( alphas[ -1 ] )

    for a, b in zip( alphas, betas ):
        # comp = np.logaddexp.reduce( a + b )
        comp = mp.log_marginalFromAlphaBeta( a, b )
        assert np.isclose( comp, marginal ), comp - marginal

    for t in range( T - 1 ):
        joint = mp.childParentJoint( t, alphas, betas )

        parentProb = np.logaddexp.reduce( joint, axis=1 )
        childProb = np.logaddexp.reduce( joint, axis=0 )

        trueParent = alphas[ t ] + betas[ t ]
        trueChild = alphas[ t + 1 ] + betas[ t + 1 ]

        assert np.allclose( parentProb, trueParent )
        assert np.allclose( childProb, trueChild )

    print( 'Passed the gaussian forward backward marginal test!\n\n' )

######################################################################

def testSLDSForwardBackward():

    T = 100
    D_latent = 20
    D_obs = 8

    mp = SLDSForwardBackward()

    xs = np.random.random( ( T, D_latent ) )

    initialDist = Dirichlet.generate( D=D_latent )
    transDist = TransitionDirichletPrior.generate( D_in=D_latent, D_out=D_latent )
    mu0, sigma0 = NormalInverseWishart.generate( D=D_latent )

    u = np.random.random( ( T, D_latent ) )

    ASigmas = [ MatrixNormalInverseWishart.generate( D_in=D_latent, D_out=D_latent ) for _ in range( D_latent ) ]
    As = [ A for A, sigma in ASigmas ]
    sigmas = [ sigma for A, sigma in ASigmas ]

    start = time.time()
    mp.updateParams( initialDist, transDist, mu0, sigma0, u, As, sigmas, xs )
    end = time.time()
    print( 'Preprocess: ', end - start )


    start = time.time()
    alphas = mp.forwardFilter()
    betas = mp.backwardFilter()
    end = time.time()
    print( 'Both filters: ', end - start )

    marginal = np.logaddexp.reduce( alphas[ -1 ] )

    for a, b in zip( alphas, betas ):
        # comp = np.logaddexp.reduce( a + b )
        comp = mp.log_marginalFromAlphaBeta( a, b )
        assert np.isclose( comp, marginal ), comp - marginal

    for t in range( T - 1 ):
        joint = mp.childParentJoint( t, alphas, betas )

        parentProb = np.logaddexp.reduce( joint, axis=1 )
        childProb = np.logaddexp.reduce( joint, axis=0 )

        trueParent = alphas[ t ] + betas[ t ]
        trueChild = alphas[ t + 1 ] + betas[ t + 1 ]

        assert np.allclose( parentProb, trueParent )
        assert np.allclose( childProb, trueChild )

    print( 'Passed the SLDS forward backward marginal test!\n\n' )

######################################################################

def testKalmanFilter():

    T = 100
    D_latent = 7
    D_obs = 3
    D = 4

    mp = KalmanFilter()

    A, sigma = MatrixNormalInverseWishart.generate( D_in=D_latent, D_out=D_latent )
    C, R = MatrixNormalInverseWishart.generate( D_in=D_latent, D_out=D_obs )
    mu0, sigma0 = NormalInverseWishart.generate( D=D_latent )

    u = np.random.random( ( T, D_latent ) )
    nBad = int( np.random.random() * T )
    badMask = np.random.choice( T, nBad )
    u[ badMask ] = np.nan

    ys = np.array( [ Regression.sample( params=( C, R ), size=T )[ 1 ] for _ in range( D ) ] )

    start = time.time()
    mp.updateParams( A=A, sigma=sigma, C=C, R=R, mu0=mu0, sigma0=sigma0, u=u, ys=ys )
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

    for a, b in zip( alphas, betas ):
        Ja, ha, log_Za = a
        Jb, hb, log_Zb = b

        # _marginal = Normal.log_partition( natParams=( -0.5*( Ja + Jb ), ( ha + hb ) ) ) - ( log_Za + log_Zb )
        _marginal = mp.log_marginalFromAlphaBeta( a, b )

        assert np.isclose( _marginal, marginal ), _marginal - marginal

    for t in range( T - 1 ):

        joint = mp.childParentJoint( t, alphas, betas )

        _JsParent, _hsParent, _logZsParent = Normal.marginalizeX1( *joint )
        _JsChild, _hsChild, _logZsChild = Normal.marginalizeX2( *joint )

        JsParent, hsParent, logZsParent = np.add( alphas[ t ], betas[ t ] )
        JsChild, hsChild, logZsChild = np.add( alphas[ t + 1 ], betas[ t + 1 ] )

        assert np.allclose( _JsParent, JsParent )
        assert np.allclose( _hsParent, hsParent )
        assert np.allclose( _logZsParent, logZsParent )

        assert np.allclose( _JsChild, JsChild )
        assert np.allclose( _hsChild, hsChild )
        assert np.allclose( _logZsChild, logZsChild )

    print( 'Passed the kalman filter marginal test!\n\n' )

######################################################################

def testSwitchingKalmanFilter():

    T = 100
    D_latent = 20
    D_obs = 7
    D = 4
    K = 5

    mp = SwitchingKalmanFilter()

    As, sigmas = list( zip( *[ MatrixNormalInverseWishart.generate( D_in=D_latent, D_out=D_latent ) for _ in range( K ) ] ) )
    C, R = MatrixNormalInverseWishart.generate( D_in=D_latent, D_out=D_obs )
    mu0, sigma0 = NormalInverseWishart.generate( D=D_latent )

    z = Categorical.generate( D=K, size=T )
    u = np.random.random( ( T, D_latent ) )
    ys = np.array( [ Regression.sample( params=( C, R ), size=T )[ 1 ] for _ in range( D ) ] )

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

    for a, b in zip( alphas, betas ):
        Ja, ha, log_Za = a
        Jb, hb, log_Zb = b

        # comp = Normal.log_partition( natParams=( -0.5*( Ja + Jb ), ( ha + hb ) ) ) - ( log_Za + log_Zb )
        comp = mp.log_marginalFromAlphaBeta( a, b )

        assert np.isclose( comp, marginal ), comp - marginal

    for t in range( T - 1 ):

        joint = mp.childParentJoint( t, alphas, betas )

        _JsParent, _hsParent, _logZsParent = Normal.marginalizeX1( *joint )
        _JsChild, _hsChild, _logZsChild = Normal.marginalizeX2( *joint )

        JsParent, hsParent, logZsParent = np.add( alphas[ t ], betas[ t ] )
        JsChild, hsChild, logZsChild = np.add( alphas[ t + 1 ], betas[ t + 1 ] )

        assert np.allclose( _JsParent, JsParent )
        assert np.allclose( _hsParent, hsParent )
        assert np.allclose( _logZsParent, logZsParent )

        assert np.allclose( _JsChild, JsChild )
        assert np.allclose( _hsChild, hsChild )
        assert np.allclose( _logZsChild, logZsChild )

    print( 'Passed the switching kalman filter marginal test!\n\n' )

######################################################################

def marginalizationTest():

    testCategoricalForwardBackward()
    testCategoricalForwardBackwardWithKnownStates()
    testGaussianForwardBackward()
    testSLDSForwardBackward()
    testKalmanFilter()
    testSwitchingKalmanFilter()
