import numpy as np
from GenModels.GM.Distributions import *
from GenModels.GM.ModelPriors import *
from GenModels.GM.States.StandardStates import *
import matplotlib.pyplot as plt
from GenModels.GM.Utility import fullyRavel

# Just a note, was trying to use umap projection for geweke test, but
# it fails all the time, even with very large burn in periods
# and steps in between gibbs samples.  Would be interesting to see
# why umap is so good at separating forward sampling from gibbs sampling
# (And its not failing because the code is wrong! 100% sure of this)
# import umap

__all__ = [ 'exponentialFamilyTest' ]

def paramNaturalTest( dist ):
    params = dist.params
    params2 = dist.natToStandard( *dist.standardToNat( *params ) )
    for p1, p2 in zip( params, params2 ):
        assert np.allclose( p1, p2 )

def tensorParamNaturalTest( self ):
    params = self.params
    params2 = self.natToStandard( *self.standardToNat( *params ) )
    for p1, p2 in zip( params, params2 ):
        if( isinstance( p1, tuple ) or isinstance( p1, list ) ):
            for _p1, _p2 in zip( p1, p2 ):
                assert np.allclose( _p1, _p2 )
        else:
            assert np.allclose( p1, p2 )

def likelihoodNoPartitionTestExpFam( dist, **kwargs ):
    x = dist.isample( size=10, **kwargs )
    xOld = x

    ans1 = dist.ilog_likelihood( x, expFam=True )
    trueAns1 = dist.ilog_likelihood( x )

    x = dist.isample( size=10, **kwargs )
    ans2 = dist.ilog_likelihood( x, expFam=True )
    trueAns2 = dist.ilog_likelihood( x )

    if( not np.isclose( ans1 - ans2, trueAns1 - trueAns2 ) ):
        # Sometimes when the sufficient stats are really big numbers,
        # the test will fail even though the math is correct
        print( '\ndiff', ( ans1 - ans2 ) - ( trueAns1 - trueAns2 ) )
        print( '\nans1', ans1 )
        print( '\nans2', ans2 )
        print( '\ntrueAns1', trueAns1 )
        print( '\ntrueAns2', trueAns2 )

        print( '\nnatParams', dist.natParams )
        stats1 = dist.sufficientStats( x, constParams=dist.constParams )
        stats2 = dist.sufficientStats( xOld, constParams=dist.constParams )
        print( '\nstats1', stats1 )
        print( '\nstats2', stats2 )
        assert 0, 'Failed test'

def likelihoodTestExpFam( dist, **kwargs ):
    x = dist.isample( size=10, **kwargs )
    ans1 = dist.ilog_likelihood( x, expFam=True )
    ans2 = dist.ilog_likelihood( x )
    assert np.isclose( ans1, ans2 ), ans1 - ans2

def paramTestExpFam( dist ):
    likelihoodTestExpFam( dist.prior )

def jointTestExpFam( dist, **kwargs ):
    x = dist.isample( size=10, **kwargs )
    ans1 = dist.ilog_joint( x, expFam=True )
    ans2 = dist.ilog_joint( x )
    assert np.isclose( ans1, ans2 ), ans1 - ans2

def posteriorTestExpFam( dist, **kwargs ):
    x = dist.isample( size=10, **kwargs )
    ans1 = dist.ilog_posterior( x, expFam=True )
    ans2 = dist.ilog_posterior( x )
    assert np.isclose( ans1, ans2 ), ans1 - ans2

def testsForDistWithoutPrior( dist, tensor=False, **kwargs ):

    if( tensor == False ):
        paramNaturalTest( dist )
    else:
        tensorParamNaturalTest( dist )

    likelihoodNoPartitionTestExpFam( dist, **kwargs )
    likelihoodTestExpFam( dist, **kwargs )

def testForDistWithPrior( dist, tensor=False, **kwargs ):

    if( tensor == False ):
        paramNaturalTest( dist )
    else:
        tensorParamNaturalTest( dist )

    likelihoodNoPartitionTestExpFam( dist, **kwargs )
    likelihoodTestExpFam( dist, **kwargs )
    paramTestExpFam( dist )
    jointTestExpFam( dist, **kwargs )
    posteriorTestExpFam( dist, **kwargs )

def standardTests():

    D = 2

    D2 = 7

    iwParams = {
        'psi': InverseWishart.sample( D=D ),
        'nu': D
    }

    niwParams = {
        'mu_0': np.random.random( D ),
        'kappa': np.random.random() * D
    }
    niwParams.update( iwParams )

    mniwParams1 = {
        'M': np.random.random( ( D, D ) ),
        'V': InverseWishart.sample( D=D )
    }
    mniwParams1.update( iwParams )

    mniwParams2 = {
        'M': np.random.random( ( D, D2 ) ),
        'V': InverseWishart.sample( D=D2 )
    }
    mniwParams2.update( iwParams )

    dirParams = {
        'alpha': np.random.random( D ) + 1
    }

    transDirParams = {
        'alpha': np.random.random( ( D, D2 ) ) + 1
    }

    niw = NormalInverseWishart( **niwParams )
    norm = Normal( prior=niw )
    iw = InverseWishart( **iwParams )

    mniw = MatrixNormalInverseWishart( **mniwParams1 )
    mniw2 = MatrixNormalInverseWishart( **mniwParams2 )
    reg = Regression( prior=mniw )

    dirichlet = Dirichlet( **dirParams )
    cat = Categorical( prior=dirichlet )

    transDir = TransitionDirichletPrior( **transDirParams )
    trans = Transition( prior=transDir )

    testsForDistWithoutPrior( iw )
    testsForDistWithoutPrior( niw )
    testForDistWithPrior( norm )
    testForDistWithPrior( reg )
    testsForDistWithoutPrior( mniw )
    testsForDistWithoutPrior( mniw2 )
    testsForDistWithoutPrior( dirichlet )
    testForDistWithPrior( cat )
    testsForDistWithoutPrior( transDir )
    testForDistWithPrior( trans )

def stateAndModelTests():
    with np.errstate( all='raise' ):

        K = 20
        obsDim = 40

        D_latent = 2
        D_obs = 3

        T = 5

        HMMParams = {
            'alpha_0': np.random.random( K ) + 1,
            'alpha_pi': np.random.random( ( K, K ) ) + 1,
            'alpha_L': np.random.random( ( K, obsDim ) ) + 1
        }

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

        hmmPrior = HMMDirichletPrior( **HMMParams )
        hmmState = HMMState( prior=hmmPrior )

        ldsPrior = LDSMNIWPrior( **LDSParams )
        ldsState = LDSState( prior=ldsPrior, _stabilize=True )

        # testsForDistWithoutPrior( hmmPrior )
        # testsForDistWithoutPrior( ldsPrior )
        # testForDistWithPrior( hmmState, T=T )
        # testForDistWithPrior( ldsState, T=T )


        A, sigma, C, R, mu0, sigma0 = ldsPrior.isample()
        _A = np.copy( A )
        _sigma = np.copy( sigma )
        _C = np.copy( C )
        _R = np.copy( R )
        _mu0 = np.copy( mu0 )
        _sigma0 = np.copy( sigma0 )

        A = np.random.random( ( 2, 2 ) )*2 - 1
        sigma = InverseWishart.sample( D=2 ) * 0.01
        C = np.random.random( ( 3, 2 ) )*2 - 1
        R = InverseWishart.sample( D=3 ) * 0.01
        mu0 = np.random.random( 2 )*2 - 1
        sigma0 = InverseWishart.sample( D=2 ) * 0.01

        # print( '\n', A.ravel(), _A.ravel() )
        # print( '\n', sigma.ravel(), _sigma.ravel() )
        # print( '\n', C.ravel(), _C.ravel() )
        # print( '\n', R.ravel(), _R.ravel() )
        # print( '\n', mu0, _mu0 )
        # print( '\n', sigma0.ravel(), _sigma0.ravel() )

        # ps = {
        #     'A': A,
        #     'sigma': sigma,
        #     'C': C,
        #     'R': R,
        #     'mu0': mu0,
        #     'sigma0': sigma0,
        #     '_stabilize': True
        # }

        ps = {
            'A': _A,
            'sigma': _sigma,
            'C': _C,
            'R': _R,
            'mu0': _mu0,
            'sigma0': _sigma0,
            '_stabilize': True
        }

        ldsState = LDSState( **ps )

        x = ldsState.isample( T=5 )

        ##################################################


        print( '\n\n-----------------------------------' )
        print( 'ldsState.A is', ldsState.A )
        print( 'ldsState.sigma is', ldsState.sigma )
        print( 'ldsState.C is', ldsState.C )
        print( 'ldsState.R is', ldsState.R )

        n1, n2, n3, n4, n5, n6, n7, n8 = ldsState.natParams

        t1, t2, t3, t4, t5, t6, t7, t8 = ldsState.sufficientStats( x, constParams=ldsState.constParams )
        assert ldsState.dataN( x ) == 1
        A1, A2, A3, A4, A5, A6, A7 = ldsState.log_partition( x, natParams=ldsState.natParams, split=True )

        print( 'n1', n1 )
        print( 'n2', n2 )
        print( 'n3', n3 )
        print( 'n4', n4 )
        print( 'n5', n5 )
        print( 'n6', n6 )
        print( 't1', t1 )
        print( 't2', t2 )
        print( 't3', t3 )
        print( 't4', t4 )
        print( 't5', t5 )
        print( 't6', t6 )
        print( 'A1', A1 )
        print( 'A2', A2 )
        print( 'A3', A3 )
        print( 'A4', A4 )
        print( '-----------------------------------\n\n' )

        # print( n2, t2 )

        trans = ( n1 * t1 ).sum() + ( n2 * t2 ).sum() + ( n3 * t3 ).sum() - ( A1 + A2 )
        emiss = ( n4 * t4 ).sum() + ( n5 * t5 ).sum() + ( n6 * t6 ).sum() - ( A3 + A4 )
        init  = ( n7 * t7 ).sum() + ( n8 * t8 ).sum() - ( A5 + A6 + A7 )

        print( '\nFrom ldsState')
        print( 'trans', trans )
        print( 'emiss', emiss )
        print( 'init', init )
        print( 'total', trans + emiss + init )

        ##################################################

        ( _x, _y ) = x
        if( _y.ndim == 3 ):
            _y = _y[ 0 ]

        print( '\n\n-----------------------------------' )
        print( 'ldsState.A is', ldsState.A )
        print( 'ldsState.sigma is', ldsState.sigma )
        print( 'ldsState.C is', ldsState.C )
        print( 'ldsState.R is', ldsState.R )

        n1, n2, n3 = Regression.standardToNat( ldsState.A  , ldsState.sigma  )

        n4, n5, n6 = Regression.standardToNat( ldsState.C  , ldsState.R      )
        n7, n8     =     Normal.standardToNat( ldsState.mu0, ldsState.sigma0 )

        t1, t2, t3 = Regression.sufficientStats( x=( _x[ :-1 ], _x[ 1: ] ) )
        t4, t5, t6 = Regression.sufficientStats( x=( _x       , _y       ) )
        t7, t8     =     Normal.sufficientStats( x=( _x[ 0 ]             ) )

        n = Regression.dataN( ( _x, _y ) )

        A1, A2     = Regression.log_partition( x=( _x[ :-1 ], _x[ 1: ] ), params=( ldsState.A  , ldsState.sigma  ), split=True )
        A1 *= n - 1
        A2 *= n - 1
        A3, A4     = Regression.log_partition( x=( _x       , _y       ), params=( ldsState.C  , ldsState.R      ), split=True )
        A3 *= n
        A4 *= n
        A5, A6, A7 =     Normal.log_partition( x=( _x[ 0 ]             ), params=( ldsState.mu0, ldsState.sigma0 ), split=True )

        print( 'n1', n1 )
        print( 'n2', n2 )
        print( 'n3', n3 )
        print( 'n4', n4 )
        print( 'n5', n5 )
        print( 'n6', n6 )
        print( 't1', t1 )
        print( 't2', t2 )
        print( 't3', t3 )
        print( 't4', t4 )
        print( 't5', t5 )
        print( 't6', t6 )
        print( 'A1', A1 )
        print( 'A2', A2 )
        print( 'A3', A3 )
        print( 'A4', A4 )
        print( '-----------------------------------\n\n' )

        trans = ( n1 * t1 ).sum() + ( n2 * t2 ).sum() + ( n3 * t3 ).sum() - ( A1 + A2 )
        emiss = ( n4 * t4 ).sum() + ( n5 * t5 ).sum() + ( n6 * t6 ).sum() - ( A3 + A4 )
        init  = ( n7 * t7 ).sum() + ( n8 * t8 ).sum() - ( A5 + A6 + A7 )

        # print( n2, t2 )

        print( '\nFrom components expfam')
        print( 'trans', trans )
        print( 'emiss', emiss )
        print( 'init', init )
        print( 'total', trans + emiss + init )
        print()
        print()
        print()
        print()

        ##################################################

        trans = Regression.log_likelihood( x=( _x[ :-1 ], _x[ 1: ] ), params=( ldsState.A, ldsState.sigma ) )
        emiss = Regression.log_likelihood( x=( _x, _y ), params=( ldsState.C, ldsState.R ) )
        init  = Normal.log_likelihood( x=_x[ 0 ], params=( ldsState.mu0, ldsState.sigma0 ) )

        print( '\nFrom components ll')
        print( 'trans', trans )
        print( 'emiss', emiss )
        print( 'init', init )
        print( 'total', trans + emiss + init )
        print()
        print()
        print()
        print()

        ##################################################

        expFamLL = ldsState.ilog_likelihood( x, expFam=True )
        recursionLL = ldsState.ilog_likelihood( x )
        print( '\nexpFamLL', expFamLL )
        print( 'recursionLL', recursionLL )

        assert 0, 'Passed!'

def tensorTests():

    D1 = 2
    D2 = 3
    D3 = 4
    D4 = 5

    tnParams = {
        'M': np.random.random( ( D1, D2, D3, D4 ) ),
        'covs': ( InverseWishart.sample( D=D1 ), \
                  InverseWishart.sample( D=D2 ), \
                  InverseWishart.sample( D=D3 ), \
                  InverseWishart.sample( D=D4 ) )
    }

    tn = TensorNormal( **tnParams )
    testsForDistWithoutPrior( tn, tensor=True )

    D = 4
    N = 5

    trParams = {
        'A': TensorNormal.sample( Ds=tuple( [ D for _ in range( N ) ] ) )[ 0 ],
        'sigma': InverseWishart.sample( D=D )
    }

    tr = TensorRegression( **trParams )
    testsForDistWithoutPrior( tr, tensor=True )

    p = np.random.random( ( D1, D2, D3, D4 ) )
    tcParams = {
        'p': p / p.sum()
    }

    tc = TensorCategorical( **tcParams )
    testsForDistWithoutPrior( tc, tensor=True )

def tensorNormalMarginalizationTest():

    D = 4
    N = 3

    trParams = {
        'A': TensorNormal.sample( Ds=tuple( [ D for _ in range( N ) ] ) )[ 0 ],
        'sigma': InverseWishart.sample( D=D )
    }

    tr = TensorRegression( **trParams )
    # To test this, graph out some stats about part of a vector sampled
    # from a tensor regression distribution and compare the plots to vectors
    # sampled from the marginalized tensor regression distribution

def exponentialFamilyTest():

    # standardTests()
    # tensorTests()
    # tensorNormalMarginalizationTest()
    stateAndModelTests()
    assert 0
    print( 'Passed all of the exp fam tests!' )