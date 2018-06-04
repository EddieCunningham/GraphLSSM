import numpy as np
from GenModels.GM.Distributions import *
import matplotlib.pyplot as plt
from GenModels.GM.Utility import *
import umap
import itertools
from functools import partial

__all__ = [ 'distributionTest' ]

def marginalTest( dist, N=4 ):

    if( dist.priorClass is None ):
        return

    # P( x ) should stay the same for different settings of params
    x = dist.isample( size=10 )

    dist.resample()

    marginal = dist.ilog_marginal( x )

    for _ in range( N ):
        dist.resample()
        marginal2 = dist.ilog_marginal( x )
        assert np.isclose( marginal, marginal2 ), marginal2 - marginal

######################################################################################

def sampleTest( regAxes, mhAxes, plotFn, dist=None, nRegPoints=1000, nMHPoints=1000, burnIn=3000, nMHForget=50 ):
    # Compare the MH sampler to the implemented sampler
    assert dist is not None

    # Generate the joint samples
    regSamples = dist.isample( size=nRegPoints )
    projections, axisChoices = plotFn( *regSamples ) if isinstance( regSamples, tuple ) else plotFn( regSamples )
    for ( xs, ys ), ax, ( a, b ) in zip( projections, regAxes, axisChoices ):
        mask = ~is_outlier( xs ) & ~is_outlier( ys )
        ax.scatter( xs[ mask ], ys[ mask ], s=0.5, alpha=0.08, color='red' )
        ax.set_title( 'Ax %d vs %d'%( a, b ) )

    # Generate the metropolis hastings points
    mhSamples = dist.metropolisHastings( burnIn=burnIn, skip=nMHForget, size=nMHPoints )
    projections, _ = plotFn( mhSamples, axisChoices=axisChoices )
    for ( xs, ys ), ax in zip( projections, mhAxes ):
        mask = ~is_outlier( xs ) & ~is_outlier( ys )
        ax.scatter( xs[ mask ], ys[ mask ], s=0.5, alpha=0.08, color='blue' )

def gewekeTest( jointAxes, gibbsAxes, plotFn, dist=None, nJointPoints=1000, nGibbsPoints=1000, burnIn=3000, nGibbsForget=50 ):
    # Sample from P( x, Ѳ ) using forward sampling and gibbs sampling and comparing their plots
    assert dist is not None

    dist.resample()

    # Generate the joint samples
    jointSamples = dist.ijointSample( size=nJointPoints )
    projections, axisChoices = plotFn( *jointSamples )
    for ( xs, ys ), ax, ( a, b ) in zip( projections, jointAxes, axisChoices ):
        mask = ~is_outlier( xs ) & ~is_outlier( ys )
        ax.scatter( xs[ mask ], ys[ mask ], s=0.5, alpha=0.08, color='red' )
        ax.set_title( 'Ax %d vs %d'%( a, b ) )

    # Generate the gibbs points
    gibbsSamples = dist.igibbsJointSample( burnIn=burnIn, skip=nGibbsForget, size=nGibbsPoints )
    projections, _ = plotFn( *gibbsSamples, axisChoices=axisChoices )
    for ( xs, ys ), ax in zip( projections, gibbsAxes ):
        mask = ~is_outlier( xs ) & ~is_outlier( ys )
        ax.scatter( xs[ mask ], ys[ mask ], s=0.5, alpha=0.08, color='blue' )

######################################################################################

def plottingTest( plotFuncs, nPlots=3 ):

    def plotFn( xs, thetas=None, axisChoices=None ):
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

######################################################################################

def metropolistHastingsTest( dists, nPlots=3 ):
    plotFn = []
    for dist in dists:
        f = partial( sampleTest, dist=dist, nRegPoints=2000, nMHPoints=2000, burnIn=4000, nMHForget=20 )
        plotFn.append( f )

    plottingTest( plotFn, nPlots=nPlots )

def gewekePlottingTest( dists, nPlots=3 ):
    plotFn = []
    for dist in dists:
        f = partial( gewekeTest, dist=dist, nJointPoints=1000, nGibbsPoints=1000, burnIn=2000, nGibbsForget=20 )
        plotFn.append( f )

    plottingTest( plotFn, nPlots=nPlots )

######################################################################################

def distributionTest():

    D = 2

    iwParams = {
        'psi': InverseWishart.generate( D=D ),
        'nu': D
    }

    niwParams = {
        'mu_0': np.random.random( D ),
        'kappa': np.random.random() * D
    }
    niwParams.update( iwParams )

    mniwParams = {
        'M': np.random.random( ( D, D ) ),
        'V': InverseWishart.generate( D=D )
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

    dists = [ niw, norm, iw, mniw, reg, dirichlet, cat, transDir, trans ]
    for dist in dists:
        marginalTest( dist )

    # Can really only do normal and regression because everything else has constrained outputs
    # metropolistHastingsTest( [ reg, norm ] )

    # Need a prior to do this
    # gewekePlottingTest( [ norm, reg, cat ] )
