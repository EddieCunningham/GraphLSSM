import numpy as np
from Base import ExponentialFam
from Normal import Normal
from InverseWishart import InverseWishart
from NormalInverseWishart import NormalInverseWishart
from Regression import Regression
from MatrixNormalInverseWishart import MatrixNormalInverseWishart
from scipy.stats import invwishart


def paramNaturalTest( dist ):
    params = dist.params
    params2 = dist.natToStandard( *dist.standardToNat( *params ) )
    for p1, p2 in zip( params, params2 ):
        assert np.allclose( p1, p2 )

def likelihoodNoPartitionTest( dist ):

    x = dist.isample()
    nat1 = dist.natParams
    stat1 = dist.sufficientStats( x )
    ans1 = ExponentialFam.log_pdf( nat1, stat1, [ 0 ] )
    trueAns1 = dist.ilog_likelihood( x )

    x = dist.isample()
    nat2 = dist.natParams
    stat2 = dist.sufficientStats( x )
    ans2 = ExponentialFam.log_pdf( nat2, stat2, [ 0 ] )
    trueAns2 = dist.ilog_likelihood( x )

    assert np.isclose( ans1 - ans2, trueAns1 - trueAns2 ), ( ans1 - ans2 ) - ( trueAns1 - trueAns2 )

def likelihoodTest( dist, x ):

    nat = dist.natParams
    stat = dist.sufficientStats( x )
    part = dist.ilog_partition( x, split=True )
    ans1 = ExponentialFam.log_pdf( nat, stat, part )

    ans2 = dist.ilog_likelihood( x )

    assert np.isclose( ans1, ans2 ), ans1 - ans2

def paramTest( dist ):
    likelihoodTest( dist.prior, dist.params )

def jointTest( dist, x ):

    postNatParams = dist.posteriorPriorNatParams( x, priorNatParams=dist.prior.natParams )

    stat = dist.prior.sufficientStats( dist.params )
    part = dist.prior.log_partition( dist.params, natParams=dist.prior.natParams, split=True )

    ans1 = ExponentialFam.log_pdf( postNatParams, stat, part )
    ans2 = dist.ilog_joint( x )

    assert np.isclose( ans1, ans2 ), ans1 - ans2

def posteriorTest( dist, x ):

    postNatParams = dist.posteriorPriorNatParams( x, priorNatParams=dist.prior.natParams )

    stat = dist.prior.sufficientStats( dist.params )
    part = dist.prior.log_partition( dist.params, natParams=postNatParams, split=True )

    ans1 = ExponentialFam.log_pdf( postNatParams, stat, part )
    ans2 = dist.ilog_posterior( x )

    assert np.isclose( ans1, ans2 ), ans1 - ans2

def testsForDistWithoutPrior( dist ):
    x = dist.isample()
    paramNaturalTest( dist )
    likelihoodNoPartitionTest( dist )
    likelihoodTest( dist, x )

def testForDistWithPrior( dist ):

    x = dist.isample()

    paramNaturalTest( dist )
    likelihoodNoPartitionTest( dist )
    likelihoodTest( dist, x )
    paramTest( dist )
    jointTest( dist, x )
    posteriorTest( dist, x )

def runTests():

    D = 5

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

runTests()