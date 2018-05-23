import numpy as np
from GenModels.GM.Distributions import *
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

def likelihoodNoPartitionTestExpFam( dist ):
    x = dist.isample( size=10 )
    ans1 = dist.ilog_likelihood( x, expFam=True )
    trueAns1 = dist.ilog_likelihood( x )

    x = dist.isample( size=10 )
    ans2 = dist.ilog_likelihood( x, expFam=True )
    trueAns2 = dist.ilog_likelihood( x )
    assert np.isclose( ans1 - ans2, trueAns1 - trueAns2 ), ( ans1 - ans2 ) - ( trueAns1 - trueAns2 )

def likelihoodTestExpFam( dist ):
    x = dist.isample( size=10 )
    ans1 = dist.ilog_likelihood( x, expFam=True )
    ans2 = dist.ilog_likelihood( x )
    assert np.isclose( ans1, ans2 ), ans1 - ans2

def paramTestExpFam( dist ):
    likelihoodTestExpFam(dist.prior)

def jointTestExpFam( dist ):
    x = dist.isample( size=10 )
    ans1 = dist.ilog_joint( x, expFam=True )
    ans2 = dist.ilog_joint( x )
    assert np.isclose( ans1, ans2 ), ans1 - ans2

def posteriorTestExpFam( dist ):
    x = dist.isample( size=10 )
    ans1 = dist.ilog_posterior( x, expFam=True )
    ans2 = dist.ilog_posterior( x )
    assert np.isclose( ans1, ans2 ), ans1 - ans2

def testsForDistWithoutPrior( dist, tensor=False ):

    if( tensor == False ):
        paramNaturalTest( dist )
    else:
        tensorParamNaturalTest( dist )

    likelihoodNoPartitionTestExpFam( dist )
    likelihoodTestExpFam( dist )

def testForDistWithPrior( dist, tensor=False ):

    if( tensor == False ):
        paramNaturalTest( dist )
    else:
        tensorParamNaturalTest( dist )

    likelihoodNoPartitionTestExpFam( dist )
    likelihoodTestExpFam( dist )
    paramTestExpFam( dist )
    jointTestExpFam( dist )
    posteriorTestExpFam( dist )

def standardTests():

    D = 2

    iwParams = {
        'psi': InverseWishart.sample( D=D ),
        'nu': D
    }

    niwParams = {
        'mu_0': np.random.random( D ),
        'kappa': np.random.random() * D
    }
    niwParams.update( iwParams )

    mniwParams = {
        'M': np.random.random( ( D, D ) ),
        'V': InverseWishart.sample( D=D )
    }
    mniwParams.update( iwParams )

    dirParams = {
        'alpha': np.random.random( D ) + 1
    }

    transDirParams = {
        'alpha': np.random.random( ( D, D ) ) + 1
    }

    niw = NormalInverseWishart( **niwParams )
    norm = Normal( prior=niw )
    iw = InverseWishart( **iwParams )

    mniw = MatrixNormalInverseWishart( **mniwParams )
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
    testsForDistWithoutPrior( dirichlet )
    testForDistWithPrior( cat )
    testsForDistWithoutPrior( transDir )
    testForDistWithPrior( trans )

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
    standardTests()
    tensorTests()
    tensorNormalMarginalizationTest()
    print( 'Passed all of the exp fam tests!' )