import numpy as np
from GenModels.GM.Distributions import *
import matplotlib.pyplot as plt
from GenModels.GM.Utility import fullyRavel
import itertools
from functools import partial

# Just a note, was trying to use umap projection for geweke test, but
# it fails all the time, even with very large burn in periods
# and steps in between gibbs samples.  Would be interesting to see
# why umap is so good at separating forward sampling from gibbs sampling
# (And its not failing because the code is wrong! 100% sure of this)
# import umap

__all__ = [ 'exponentialFamilyTest' ]

def plottingTest( plotFuncs, nPlots=3 ):

    def plotFn( xs, thetas, axisChoices=None ):
        X = []
        if( thetas is not None ):
            assert len( xs ) == len( thetas )
            for x, theta in zip( xs, thetas ):
                X.append( np.hstack( ( fullyRavel( x ), fullyRavel( theta ) ) ) )
        else:
            for x in xs:
                X.append( fullyRavel( x ) )
        X = np.vstack( X )

        if( axisChoices is None ):
            axisChoices = []
            for _ in range( nPlots ):
                a = np.random.choice( X.shape[ 1 ] )
                b = a
                while( b == a ):
                    b = np.random.choice( X.shape[ 1 ] )
                axisChoices.append( ( a, b ) )
        else:
            assert len( axisChoices ) == nPlots, axisChoices

        ans = []
        for i, ( a, b ) in enumerate( axisChoices ):
            ans.append( ( X[ :, a ], X[ :, b ] ) )

        return ans, axisChoices

    N = len( plotFuncs )

    fig = plt.figure()
    axes = [ plt.subplot2grid( ( N, nPlots ), ( i, j ) ) for i, j in itertools.product( range( N ), range( nPlots ) ) ]
    it = iter( axes )
    for distAndAx in zip( plotFuncs, *[ it for _ in range( nPlots ) ] ):
        func = distAndAx[ 0 ]
        ax = distAndAx[ 1: ]

        func( ax, ax, plotFn )

    plt.show()

def metropolistHastingsTest( dists, nPlots=3 ):
    plotFn = []
    for dist in dists:
        def f( regAxes, mhAxes, plotFn ):
            return dist.sampleTest( regAxes, mhAxes, plotFn, nRegPoints=2000, nMHPoints=2000, burnIn=4000, nMHForget=10 )
        plotFn.append( f )

    plottingTest( plotFn, nPlots=nPlots )

def gewekeTest( dists, nPlots=3 ):
    plotFn = []
    for dist in dists:
        # def f( jointAxes, gibbsAxes, plotFn ):
        #     return dist.gewekeTest( jointAxes, gibbsAxes, plotFn, nJointPoints=1000, nGibbsPoints=1000, burnIn=2000, nGibbsForget=10 )

        f = partial( dist.gewekeTest, nJointPoints=1000, nGibbsPoints=1000, burnIn=2000, nGibbsForget=10 )
        plotFn.append( f )

    plottingTest( plotFn, nPlots=nPlots )

def testsForDistWithoutPrior( dist ):
    dist.paramNaturalTest()
    dist.likelihoodNoPartitionTestExpFam()
    dist.likelihoodTestExpFam()

def testForDistWithPrior( dist ):

    dist.marginalTest()
    dist.paramNaturalTest()
    dist.likelihoodNoPartitionTestExpFam()
    dist.likelihoodTestExpFam()
    dist.paramTestExpFam()
    dist.jointTestExpFam()
    dist.posteriorTestExpFam()

def standardTests():

    D = 7

    ######################################

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

    ######################################

    niw = NormalInverseWishart( **niwParams )
    norm = Normal( prior=niw )
    iw = InverseWishart( **iwParams )

    mniw = MatrixNormalInverseWishart( **mniwParams )
    reg = Regression( prior=mniw )

    dirichlet = Dirichlet( **dirParams )
    cat = Categorical( prior=dirichlet )

    ######################################

    niw.functionalityTest( D=3 )
    mniw.functionalityTest( D_in=3, D_out=4 )
    norm.functionalityTest( D=3 )
    iw.functionalityTest( D=3 )
    reg.functionalityTest( D_in=3, D_out=4 )
    dirichlet.functionalityTest( D=3 )
    cat.functionalityTest( D=3 )
    assert 0

    ######################################

    testForDistWithPrior( norm )
    testsForDistWithoutPrior( iw )
    testForDistWithPrior( reg )
    testsForDistWithoutPrior( niw )
    testsForDistWithoutPrior( mniw )
    testsForDistWithoutPrior( dirichlet )
    testForDistWithPrior( cat )

    ######################################

    # metropolistHastingsTest( [ reg ] )

    assert 0
    # gewekeTest( [ norm, cat ] )
    # gewekeTest( [ norm, reg, cat ] )

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
    testsForDistWithoutPrior( tn )

    D = 4
    N = 5

    trParams = {
        'A': TensorNormal.sample( Ds=tuple( [ D for _ in range( N ) ] ) )[ 0 ],
        'sigma': InverseWishart.sample( D=D )
    }

    tr = TensorRegression( **trParams )
    testsForDistWithoutPrior( tr )

    p = np.random.random( ( D1, D2, D3, D4 ) )
    tcParams = {
        'p': p / p.sum()
    }

    tc = TensorCategorical( **tcParams )
    testsForDistWithoutPrior( tc )

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