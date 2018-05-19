import numpy as np
from GenModels.GM.Distributions import *
import matplotlib.pyplot as plt
import umap

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

def distributionInstanceIterator( D=7 ):

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

    dists = [ norm ]
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

def jointFunctionality( distClass, N=7, **D ):

    # Test with 1 sample
    x = distClass.jointSample( **D, size=1 )
    y = distClass.jointSample( **D, size=1, ravel=True )

    distClass.log_joint( x, params=distClass.paramSample( **D ) )
    distClass.log_joint( y, params=distClass.paramSample( **D ), ravel=True )

    # Test with N samples
    x = distClass.jointSample( **D, size=N )
    y = distClass.jointSample( **D, size=N, ravel=True )

    distClass.log_joint( x, params=distClass.paramSample( **D ) )
    distClass.log_joint( y, params=distClass.paramSample( **D ), ravel=True )

def functionalityTest( distClass, N=7, **D ):

    generativeFunctionality( distClass, N=N, **D )
    jointFunctionality( distClass, N=N, **D )

############################################################################

def distributionTest():

    D = 7
    for _class, Ds in distributionClassIterator( D=D ):
        functionalityTest( _class, **Ds )