import numpy as np
# np.random.seed(2)
import sys
sys.path.append( '/Users/Eddie/GenModels' )

from GM.States.GraphicalStates.MessagePassing import Graph, \
                                                     GraphMessagePasser, \
                                                     GraphFilter, \
                                                     GraphCategoricalForwardBackward, \
                                                     graph1, graph2, graph3

from GM.Distributions import MatrixNormalInverseWishart, \
                             NormalInverseWishart, \
                             Dirichlet, \
                             Categorical, \
                             Regression, \
                             Normal
from scipy.stats import dirichlet
import time

def testGraphCategoricalForwardBackwardNoCycle():

    graph = graph1()

    graphs = [ graph ]

    parentMasks = []
    childMasks = []
    for graph in graphs:
        pMask, cMask = graph.toMatrix()
        parentMasks.append( pMask )
        childMasks.append( cMask )

    T = len( graph.nodes )
    K = 2      # Latent state size
    obsDim = 5 # Obs state size
    N = 2      # Numb parents per child
    D = 1      # Data sets

    onesK = np.ones( K )
    onesObs = np.ones( obsDim )

    ( p, ) = Dirichlet.sample( params=onesObs )
    ys = [ Categorical.sample( params=p, size=T ) for _ in range( D ) ]
    ( initialDist, ) = Dirichlet.sample( params=onesK )
    transDist = Dirichlet.sample( params=onesK, size=( K, ) * N )
    emissionDist = Dirichlet.sample( params=onesObs, size=K )

    # asdf = 1e-5
    # initialDist = np.array( [ 1. - asdf, asdf ] )
    # emissionDist = np.array( [ asdf, 1. - asdf ]*K ).reshape( ( K, obsDim ) )

    # transDist = np.array( [ [ [ 1. - asdf, asdf      ],
    #                           [      asdf, 1. - asdf ] ] ,
    #                         [ [ 1. - asdf, asdf      ],
    #                           [      asdf, 1. - asdf ] ] ] )

    # ys = [ np.array( [ 0, 0, 0 ] ) ]

    print('\n\nys:', ys)
    print( 'emissionDist', emissionDist )
    print('\n\n')

    msg = GraphCategoricalForwardBackward( K=K, N=N )
    msg.updateParams( ys, initialDist, transDist, emissionDist, parentMasks, childMasks )

    U, V = msg.filter()

    for n, probs in zip( msg.nodes, msg.nodeSmoothed( U, V, msg.nodes ) ):
        print( 'P( %d | Y ) for'%( n ), ':', probs, '->', np.sum( probs ) )

    # for n, probs in zip( msg.nodes, msg.conditionalParentChild( U, V, msg.nodes, returnLog=False ) ):
    #     print( '\nP( x_c | x_p1..pN, Y ) for', n, ':\n', probs )


testGraphCategoricalForwardBackwardNoCycle()