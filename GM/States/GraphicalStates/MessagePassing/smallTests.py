import numpy as np
np.random.seed(2)
import sys
sys.path.append( '/Users/Eddie/GenModels' )

from GM.States.GraphicalStates.MessagePassing import *

# Want to test multiply terms and integrate!!!!!!!!!!

def multiplyTruth( terms, K ):
    axes = [ [ i for i, s in enumerate( t.shape ) if s != 1 ] for t in terms ]
    squeezedTerms = [ t.squeeze() for t in terms ]
    ndim = max( [ max( ax ) for ax in axes ] ) + 1
    ans, returnAxes = GraphCategoricalForwardBackward.multiplyTerms_old( squeezedTerms, axes, ndim )

    newShape = np.ones( ndim, dtype=int )
    newShape[ returnAxes ] = K
    return ans.reshape( newShape )

def multiplyTest():

    K = 2
    term1 = np.random.random( ( 1, K ) )
    term2 = np.random.random( ( K, 1, 1, 1, K ) )
    term3 = np.random.random( ( K, K ) )
    term4 = np.random.random( ( 1, K, K, 1, 1, K, K, K ) )
    terms = [ term1,
              term2,
              term3,
              term4 ]

    true = multiplyTruth( terms, K )
    comp = GraphCategoricalForwardBackward.multiplyTerms( terms )

    print( ( true - comp ).sum() )

def axesExtentionTest():

    K1 = 2
    K2 = 3
    K3 = 4

    a = np.random.random( ( K1,  1, K2,  1,  1,  1, K3 ) )
    b = np.random.random( ( K1, K2,  1,  1, K3,  1,  1 ) )
    c = np.random.random( ( K1, K2,  1, K3,  1,  1,  1 ) )
    d = np.random.random( ( K1, K2,  1,  1,  1, K3,  1 ) )
    e = np.random.random( ( K1, K2,  1,  1,  1, K3,  1, 1, 1, 1, 1, 1, 1, 1, 1, K1 ) )

    a = GraphCategoricalForwardBackward.extendAxes( a, 0, 5 )
    b = GraphCategoricalForwardBackward.extendAxes( b, 1, 5 )
    c = GraphCategoricalForwardBackward.extendAxes( c, 2, 5 )
    d = GraphCategoricalForwardBackward.extendAxes( d, 3, 5 )
    e = GraphCategoricalForwardBackward.extendAxes( e, 4, 5 )

    print( 'a', a.shape )
    print( 'b', b.shape )
    print( 'c', c.shape )
    print( 'd', d.shape )
    print( 'e', e.shape )

    mult = GraphCategoricalForwardBackward.multiplyTerms( [ a, b, c, d, e ] )
    print( 'mult.shape', mult.shape )

def integrateTest():

    K1 = 2
    K2 = 3
    K3 = 4

    a = np.random.random( ( K1,  1, K2,  1,  1,  1, K3 ) )
    b = np.random.random( ( K1, K2,  1,  1, K3,  1,  1 ) )
    c = np.random.random( ( K1, K2,  1, K3,  1,  1,  1 ) )
    d = np.random.random( ( K1, K2,  1,  1,  1, K3,  1 ) )
    e = np.random.random( ( K1, K2,  1,  1,  1, K3,  1, 1, 1, 1, 1, 1, 1, 1, 1, K1 ) )

    a = GraphCategoricalForwardBackward.extendAxes( a, 0, 5 )
    b = GraphCategoricalForwardBackward.extendAxes( b, 1, 5 )
    c = GraphCategoricalForwardBackward.extendAxes( c, 2, 5 )
    d = GraphCategoricalForwardBackward.extendAxes( d, 3, 5 )
    e = GraphCategoricalForwardBackward.extendAxes( e, 4, 5 )

    mult = GraphCategoricalForwardBackward.multiplyTerms( [ a, b, c, d, e ] )
    print( mult.shape )

    integrated = GraphCategoricalForwardBackward.integrate( mult, axes=[ 0, 1, -1 ] )
    print( integrated.shape )

# multiplyTest()
# axesExtentionTest()
integrateTest()