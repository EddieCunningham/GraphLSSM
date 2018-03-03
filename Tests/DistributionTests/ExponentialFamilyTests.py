import numpy as np
import sys
sys.path.append( '/Users/Eddie/GenModels' )

from GM.Distributions import ExponentialFam, \
                             Normal, \
                             NormalInverseWishart, \
                             InverseWishart, \
                             Regression, \
                             MatrixNormalInverseWishart, \
                             Categorical, \
                             Dirichlet, \
                             TensorNormal, \
                             TensorRegression
from scipy.stats import invwishart

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


standardTests()
tensorTests()