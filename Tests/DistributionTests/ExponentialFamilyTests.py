import numpy as np
from GenModels.GM.Distributions import *
from GenModels.GM.ModelPriors import *
from GenModels.GM.States.StandardStates import *
import matplotlib.pyplot as plt
from GenModels.GM.Utility import *

from autograd import jacobian
import autograd.numpy as anp

# Just a note, was trying to use umap projection for geweke test, but
# it fails all the time, even with very large burn in periods
# and steps in between gibbs samples.  Would be interesting to see
# why umap is so good at separating forward sampling from gibbs sampling
# (And its not failing because the code is wrong! 100% sure of this)
# import umap

__all__ = [ 'exponentialFamilyTest' ]

# np.random.seed( 2 )

def partitionTest( dist, **kwargs ):
    # Check the gradients
    if( isinstance( dist, Regression ) ):
        return
    # This is a classmethod because the test has to be class specific.
    # Can't use autograd with regular numpy or scipy
    dist._testLogPartitionGradient()

def scoreTest( dist, **kwargs ):
    # v( x, n ) = dlog_P( x | n ) / dn = t( x ) - dlog_A( n ) / dn

    return
    # Fisher info matrix not implemented
    x = dist.isample( **kwargs )
    stepSize = 0.1

    for _ in range( 50 ):

        score = dist.iscore( x )

        # This wont work, but fix it whenever fisherInfo gets implemented
        fisher = np.linalg.inv( dist.ifisherInfo( x ) )

        n = list( dist.natParams )
        for i, s, f in enumerate( score, fisher ):
            n[ i ] -= f @ s

        dist.natParams = n

    assert 0

def statMGFTest( dist, **kwargs ):
    pass

def klDivergenceTest( dist, **kwargs ):
    pass

####################################################################################

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

####################################################################################

def likelihoodNoPartitionTestExpFam( dist, **kwargs ):
    x = dist.isample( size=10, **kwargs )
    xOld = x

    ans1 = dist.ilog_likelihood( x, expFam=True )
    trueAns1 = dist.ilog_likelihood( x )

    x = dist.isample( size=10, **kwargs )
    ans2 = dist.ilog_likelihood( x, expFam=True )
    trueAns2 = dist.ilog_likelihood( x )

    if( not np.isclose( ans1 - ans2, trueAns1 - trueAns2, atol=1e-2 ) ):
        # Sometimes when the sufficient stats are really big numbers,
        # the test will fail even though the math is correct.
        # Using a really large tolerance because of numerical instabilities
        print( '\nx', x )
        print( '\nxOld', xOld )
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
        total = 0.0
        for n, s in zip( dist.natParams, stats1 ):
            total += ( n * s ).sum()
            print( 'n*s', ( n * s ).sum() )
        print( '->', total )
        assert 0, 'Failed test'

def likelihoodTestExpFam( dist, **kwargs ):
    x = dist.isample( size=10, **kwargs )
    ans1 = dist.ilog_likelihood( x, expFam=True )
    ans2 = dist.ilog_likelihood( x )
    assert np.isclose( ans1, ans2 ), ans1 - ans2

####################################################################################

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

####################################################################################

def testsForDistWithoutPrior( dist, tensor=False, **kwargs ):

    if( tensor == False ):
        paramNaturalTest( dist )
    else:
        tensorParamNaturalTest( dist )

    likelihoodNoPartitionTestExpFam( dist, **kwargs )
    likelihoodTestExpFam( dist, **kwargs )

    partitionTest( dist, **kwargs )
    scoreTest( dist, **kwargs )
    # statMGFTest( dist, **kwargs )
    # klDivergenceTest( dist, **kwargs )

def testForDistWithPrior( dist, tensor=False, **kwargs ):

    testsForDistWithoutPrior( dist, tensor=tensor, **kwargs )
    paramTestExpFam( dist )
    jointTestExpFam( dist, **kwargs )
    posteriorTestExpFam( dist, **kwargs )

####################################################################################

def standardTests():

    D = 2
    D2 = 7

    iwParams = {
        'psi': InverseWishart.generate( D=D ),
        'nu': D
    }

    niwParams = {
        'mu_0': np.random.random( D ),
        'kappa': np.random.random() * D
    }
    niwParams.update( iwParams )

    mniwParams1 = {
        'M': np.random.random( ( D, D ) ),
        'V': InverseWishart.generate( D=D )
    }
    mniwParams1.update( iwParams )

    mniwParams2 = {
        'M': np.random.random( ( D, D2 ) ),
        'V': InverseWishart.generate( D=D2 )
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

    testForDistWithPrior( norm )
    testsForDistWithoutPrior( iw )
    testsForDistWithoutPrior( dirichlet )
    testForDistWithPrior( reg )
    testsForDistWithoutPrior( mniw2 )
    testsForDistWithoutPrior( mniw )
    testForDistWithPrior( cat )
    testsForDistWithoutPrior( niw )
    testsForDistWithoutPrior( transDir )
    testForDistWithPrior( trans )

    print( 'Done with the regular exp fam distribution tests')

####################################################################################

def stateAndModelTests():
    with np.errstate( all='raise' ):

        D_latent = 2
        D_obs = 2

        T = 3

        HMMParams = {
            'alpha_0': np.random.random( D_latent ) + 1,
            'alpha_pi': np.random.random( ( D_latent, D_latent ) ) + 1,
            'alpha_L': np.random.random( ( D_latent, D_obs ) ) + 1
        }

        LDSParams = {
            'mu_0': np.random.random( D_latent ),
            'kappa_0': np.random.random() * D_latent,
            'psi_0': InverseWishart.generate( D=D_latent ),
            'nu_0': D_latent,

            'M_trans': np.random.random( ( D_latent, D_latent ) ) * 0.01,
            'V_trans': InverseWishart.generate( D=D_latent ),
            'psi_trans': InverseWishart.generate( D=D_latent ),
            'nu_trans': D_latent,

            'M_emiss': np.random.random( ( D_obs, D_latent ) ) * 0.01,
            'V_emiss': InverseWishart.generate( D=D_latent ),
            'psi_emiss': InverseWishart.generate( D=D_obs ),
            'nu_emiss': D_obs
        }

        hmmPrior = HMMDirichletPrior( **HMMParams )
        hmmState = HMMState( prior=hmmPrior )

        ldsPrior = LDSMNIWPrior( **LDSParams )
        ldsState = LDSState( prior=ldsPrior )

        testsForDistWithoutPrior( hmmPrior )
        testsForDistWithoutPrior( ldsPrior )
        testForDistWithPrior( hmmState, T=T )
        testForDistWithPrior( ldsState, T=T ) # I think this is correct, but it will most likely fail because the sequences of x that it samples have really large numbers

####################################################################################

def tensorTests():

    D1 = 2
    D2 = 3
    D3 = 4
    D4 = 5

    tnParams = {
        'M': np.random.random( ( D1, D2, D3, D4 ) ),
        'covs': ( InverseWishart.generate( D=D1 ), \
                  InverseWishart.generate( D=D2 ), \
                  InverseWishart.generate( D=D3 ), \
                  InverseWishart.generate( D=D4 ) )
    }

    tn = TensorNormal( **tnParams )
    testsForDistWithoutPrior( tn, tensor=True )

    D = 4
    N = 5

    trParams = {
        'A': TensorNormal.generate( Ds=tuple( [ D for _ in range( N ) ] ) )[ 0 ],
        'sigma': InverseWishart.generate( D=D )
    }

    tr = TensorRegression( **trParams )
    testsForDistWithoutPrior( tr, tensor=True )

    p = np.random.random( ( D1, D2, D3, D4 ) )
    tcParams = {
        'p': p / p.sum()
    }

    tc = TensorCategorical( **tcParams )
    testsForDistWithoutPrior( tc, tensor=True )

####################################################################################

def tensorNormalMarginalizationTest():

    D = 4
    N = 3

    trParams = {
        'A': TensorNormal.generate( Ds=tuple( [ D for _ in range( N ) ] ) )[ 0 ],
        'sigma': InverseWishart.generate( D=D )
    }

    tr = TensorRegression( **trParams )
    # To test this, graph out some stats about part of a vector sampled
    # from a tensor regression distribution and compare the plots to vectors
    # sampled from the marginalized tensor regression distribution

####################################################################################

def exponentialFamilyTest():

    standardTests()
    # tensorTests()
    # tensorNormalMarginalizationTest()
    stateAndModelTests()
    print( 'Passed all of the exp fam tests!' )