import numpy as np
import sys
sys.path.append( '/Users/Eddie/GenModels' )

from GM.Distributions import Exponential, \
                             Normal, \
                             NormalInverseWishart, \
                             InverseWishart, \
                             Regression, \
                             MatrixNormalInverseWishart, \
                             Categorical, \
                             Dirichlet
from scipy.stats import invwishart


def paramNaturalTest( dist ):
    params = dist.params
    params2 = dist.natToStandard( *dist.standardToNat( *params ) )
    for p1, p2 in zip( params, params2 ):
        assert np.allclose( p1, p2 )

def likelihoodNoPartitionTest( dist, *args ):

    x = dist.isample()
    nat1 = dist.natParams
    stat1 = dist.sufficientStats( x, *args )
    ans1 = Exponential.log_pdf( nat1, stat1 )
    trueAns1 = dist.ilog_likelihood( x )

    x = dist.isample()
    nat2 = dist.natParams
    stat2 = dist.sufficientStats( x, *args )
    ans2 = Exponential.log_pdf( nat2, stat2 )
    trueAns2 = dist.ilog_likelihood( x )

    assert np.isclose( ans1 - ans2, trueAns1 - trueAns2 ), ( ans1 - ans2 ) - ( trueAns1 - trueAns2 )

def likelihoodTest( dist, x, *args ):

    nat = dist.natParams
    stat = dist.sufficientStats( x, *args )
    part = dist.ilog_partition( x, split=True )
    ans1 = Exponential.log_pdf( nat, stat, part )

    ans2 = dist.ilog_likelihood( x )

    assert np.isclose( ans1, ans2 ), ans1 - ans2

def paramTest( dist ):
    likelihoodTest( dist.prior, dist.params )

def jointTest( dist, x, *args ):

    postNatParams = dist.posteriorPriorNatParams( x, priorNatParams=dist.prior.natParams )

    stat = dist.prior.sufficientStats( dist.params, *args )
    part = dist.prior.log_partition( dist.params, natParams=dist.prior.natParams, split=True )

    ans1 = Exponential.log_pdf( postNatParams, stat, part )
    ans2 = dist.ilog_joint( x )

    assert np.isclose( ans1, ans2 ), ans1 - ans2

def posteriorTest( dist, x, *args ):

    postNatParams = dist.posteriorPriorNatParams( x, priorNatParams=dist.prior.natParams )

    stat = dist.prior.sufficientStats( dist.params, *args )
    part = dist.prior.log_partition( dist.params, natParams=postNatParams, split=True )

    ans1 = Exponential.log_pdf( postNatParams, stat, part )
    ans2 = dist.ilog_posterior( x )

    assert np.isclose( ans1, ans2 ), ans1 - ans2

def testsForDistWithoutPrior( dist ):
    x = dist.isample()
    paramNaturalTest( dist )
    likelihoodNoPartitionTest( dist )
    likelihoodTest( dist, x )

def testForDistWithPrior( dist, *statArgs ):

    x = dist.isample()

    paramNaturalTest( dist )
    likelihoodNoPartitionTest( dist, *statArgs )
    likelihoodTest( dist, x, *statArgs )
    paramTest( dist )
    if( statArgs is None ):
        # Don't want to write special cases for
        # Categorical I'm because pretty sure its right
        jointTest( dist, x )
        posteriorTest( dist, x )

def runTests():

    D = 2

    iwParams = {
        'psi': invwishart.rvs( df=D, scale=np.eye( D ), size=1 ),
        'nu': D
    }

    niwParams = {
        'mu_0': np.random.random( D ),
        'kappa': np.random.random() * D
    }
    niwParams.update( iwParams )

    mniwParams = {
        'M': np.random.random( ( D, D ) ),
        'V': invwishart.rvs( df=D, scale=np.eye( D ), size=1 )
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
    testForDistWithPrior( cat, D )

runTests()