import numpy as np
from GenModels.GM.Distributions import *
from scipy.stats import invwishart

__all__ = [ 'exponentialFamilyTest' ]

def testsForDistWithoutPrior( dist ):

    dist.paramNaturalTest()
    dist.likelihoodNoPartitionTest()
    dist.likelihoodTest()

def testForDistWithPrior( dist ):

    dist.paramNaturalTest()
    dist.likelihoodNoPartitionTest()
    dist.likelihoodTest()
    dist.paramTest()
    dist.jointTest()
    dist.posteriorTest()

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