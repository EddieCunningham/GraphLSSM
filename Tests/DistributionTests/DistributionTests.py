import numpy as np
from GenModels.GM.Distributions import *
import matplotlib.pyplot as plt
import umap
# Just a note, was trying to use umap projection for geweke test, but
# it fails all the time, even with very large burn in periods
# and steps in between gibbs samples.  Would be interesting to see
# why umap is so good at separating forward sampling from gibbs sampling
# (And its not failing because the code is wrong! 100% sure of this)
# import umap
import itertools
from functools import partial
from GenModels.GM.Utility import *

__all__ = [ 'distributionTest' ]

def distributionClassIterator( D=3, D_in=3, D_out=4 ):

    niw       = (  NormalInverseWishart,       { 'D': D } )
    norm      = (  Normal,                     { 'D': D } )
    iw        = (  InverseWishart,             { 'D': D } )
    mniw      = (  MatrixNormalInverseWishart, { 'D_in': D_in, 'D_out': D_out } )
    reg       = (  Regression,                 { 'D_in': D_in, 'D_out': D_out } )
    dirichlet = (  Dirichlet,                  { 'D': D } )
    cat       = (  Categorical,                { 'D': D } )

    dists = [ niw, norm, iw, mniw, reg, dirichlet, cat ]
    for _class in dists:
        yield _class

def distributionInstanceIterator( D=7, forMH=False ):

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

    if( forMH ):
        # Can't use MH for anything with any constraints in the output...
        dists = [ norm, reg ]
    else:
        dists = [ niw, norm, iw, mniw, reg, dirichlet, cat ]

    for inst in dists:
        yield inst

############################################################################

def generativeFunctionality( distClass, N=7, **D ):

    # Test with 1 sample
    x = distClass.sample( **D, size=1 )
    y = distClass.sample( **D, size=1, ravel=True )

    assert distClass.dataN( x ) == 1
    assert distClass.dataN( y, ravel=True ) == 1

    distClass.log_likelihood( x, params=distClass.paramSample( **D ) )
    distClass.log_likelihood( y, params=distClass.paramSample( **D ), ravel=True )

    # Test with N samples
    x = distClass.sample( **D, size=N )
    y = distClass.sample( **D, size=N, ravel=True )

    assert distClass.dataN( x ) == N
    assert distClass.dataN( y, ravel=True ) == N

    distClass.log_likelihood( x, params=distClass.paramSample( **D ) )
    distClass.log_likelihood( y, params=distClass.paramSample( **D ), ravel=True )

def jointFunctionality( distClass, **D ):
    x, params = distClass.jointSample( **D )
    distClass.log_joint( x, params=params )

def functionalityTest( distClass, **D ):
    generativeFunctionality( distClass, **D )
    jointFunctionality( distClass, **D )

############################################################################

def posteriorFunctionality( dist, N=7 ):

    if( dist.priorClass is None ):
        return

    # Test with 1 sample
    x = dist.isample( size=1 )
    params = dist.iposteriorSample( x )
    dist.log_posterior( x, constParams=dist.constParams, params=params, priorParams=dist.prior.params )
    dist.ilog_posterior( x )

    # Test with N samples
    x = dist.isample( size=N )
    params = dist.iposteriorSample( x )
    dist.log_posterior( x, constParams=dist.constParams, params=params, priorParams=dist.prior.params )
    dist.ilog_posterior( x )

def marginalTest( dist, N=7 ):
    # P( x ) should stay the same for different settings of params
    x = dist.isample( size=10 )

    dist.resample()
    marginal = dist.ilog_marginal( x )

    for _ in range( N ):
        dist.resample()
        marginal2 = dist.ilog_marginal( x )
        assert np.isclose( marginal, marginal2 ), marginal2 - marginal

def instanceFunctionalityTests( dist ):
    posteriorFunctionality( dist )
    marginalTest( dist )

############################################################################

def sampleTest( regAxes, mhAxes, plotFn, dist=None, nRegPoints=1000, nMHPoints=1000, burnIn=3000, nMHForget=50 ):
    # Compare the MH sampler to the implemented sampler
    assert dist is not None

    # Generate the joint samples
    regSamples = dist.isample( size=nRegPoints, ravel=True )

    projections, axisChoices = plotFn( regSamples )
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

# def gewekeTest( dist, jointAxes, gibbsAxes, plotFn, nJointPoints=1000, nGibbsPoints=1000, burnIn=3000, nGibbsForget=50 ):
#     # Sample from P( x, Ѳ ) using forward sampling and gibbs sampling and comparing their plots

#     dist.resample()

#     # Generate the joint samples
#     jointSamples = dist.ijointSample( size=nJointPoints )

#     projections, axisChoices = plotFn( *jointSamples )
#     for ( xs, ys ), ax, ( a, b ) in zip( projections, jointAxes, axisChoices ):
#         mask = ~is_outlier( xs ) & ~is_outlier( ys )
#         ax.scatter( xs[ mask ], ys[ mask ], s=0.5, alpha=0.08, color='red' )
#         ax.set_title( 'Ax %d vs %d'%( a, b ) )

#     # Generate the gibbs points
#     gibbsSamples = dist.igibbsJointSample( burnIn=burnIn, skip=nGibbsForget, size=nGibbsPoints )
#     projections, _ = plotFn( *gibbsSamples, axisChoices=axisChoices )
#     for ( xs, ys ), ax in zip( projections, gibbsAxes ):
#         mask = ~is_outlier( xs ) & ~is_outlier( ys )
#         ax.scatter( xs[ mask ], ys[ mask ], s=0.5, alpha=0.08, color='blue' )

def plottingTest( plotFuncs, nPlots=3 ):

    def plotFn( xs, axisChoices=None ):

        X = np.vstack( xs )

        if( axisChoices is None ):
            axisChoices = []
            for _ in range( nPlots ):
                a = np.random.choice( X.shape[ 1 ] )
                b = a
                while( b == a and X.shape[ 1 ] > 1 ):
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
    plotFuncs = []
    for dist in dists:

        f = partial( sampleTest, dist=dist, nRegPoints=1000, nMHPoints=1000, burnIn=1000, nMHForget=10 )
        plotFuncs.append( f )

    plottingTest( plotFuncs, nPlots=nPlots )

# def gewekeTest( dists, nPlots=3 ):
#     plotFn = []
#     for dist in dists:
#         f = partial( dist.gewekeTest, nJointPoints=1000, nGibbsPoints=1000, burnIn=2000, nGibbsForget=10 )
#         plotFn.append( f )

#     plottingTest( plotFn, nPlots=nPlots )

############################################################################

def distributionTest():

    D = 3
    for _class, Ds in distributionClassIterator( D=D ):
        functionalityTest( _class, **Ds )

    for inst in distributionInstanceIterator( D=D ):
        instanceFunctionalityTests( inst )

    metropolistHastingsTest( distributionInstanceIterator( D=D, forMH=True ) )