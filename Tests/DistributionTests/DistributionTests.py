import numpy as np
from GenModels.GM.Distributions import *
import matplotlib.pyplot as plt
import umap

__all__ = [ 'distributionTest' ]

def marginalTests( dist ):
    assert dist.prior is not None
    dist.marginalTest()
    dist.marginalTestMC()

def gewekeTest( dists ):

    def plotFn( xs, thetas ):
        X = []
        for x, theta in zip( xs, thetas ):
            X.append( np.hstack( ( x.ravel(), theta.ravel() ) ) )
        X = np.array( X )
        u = umap.UMAP().fit_transform( X )
        return u[ :, 0 ], u[ :, 1 ]

    N = len( dists )

    fig = plt.figure()
    axes = [ plt.subplot2grid( ( N, 2 ), ( i, j ) ) for i, j in itertools.product( range( N ), range( 2 ) ) ]
    it = iter( axes )
    for dist, jointAx, gibbsAx in zip( dists, it, it ):
        dist.gewekeTest( jointAx, gibbsAx, plotFn )

    plt.show()

def expFamTests():

    D = 4

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

    niw = NormalInverseWishart( **niwParams )
    norm = Normal( prior=niw )
    iw = InverseWishart( **iwParams )

    mniw = MatrixNormalInverseWishart( **mniwParams )
    reg = Regression( prior=mniw )

    testsForDistWithoutPrior( iw )

    testsForDistWithoutPrior( niw )
    testForDistWithPrior( norm )

    testForDistWithPrior( reg )
    testsForDistWithoutPrior( mniw )

    dirParams = {
        'alpha': np.random.random( D ) + 1
    }

    dirichlet = Dirichlet( **dirParams )
    cat = Categorical( prior=dirichlet )

    testsForDistWithoutPrior( dirichlet )
    testForDistWithPrior( cat )