import numpy as np
# np.random.seed(2)
import sys
sys.path.append( '/Users/Eddie/GenModels' )

from GM.States.GraphicalStates.MessagePassing import *

from GM.Distributions import MatrixNormalInverseWishart, \
                             NormalInverseWishart, \
                             Dirichlet, \
                             Categorical, \
                             Regression, \
                             Normal
from scipy.stats import dirichlet
import time
from collections import Iterable

def testGraphCategoricalForwardBackwardNoCycle():

    # graphs = [ graph1(), graph2(), graph3(), graph4(), graph5() ]
    # cycleGraphs = [ cycleGraph1(), cycleGraph2(), cycleGraph3(), cycleGraph4(), cycleGraph5(), cycleGraph6() ]
    graphs = [ cycleGraph1() ]


    # Check how many transition distributions we need
    allTransitionCounts = set()
    for graph in graphs:
        if( isinstance( graph, Iterable ) ):
            graph, fbs = graph
        for parents in graph.edgeParents:
            ndim = len( parents ) + 1
            allTransitionCounts.add( ndim )

    nNodes = 0
    for graph in graphs:
        if( isinstance( graph, Iterable ) ):
            graph, fbs = graph
        nNodes += len( graph.nodes )

    K = 2      # Latent state size
    obsDim = 5 # Obs state size
    D = 2      # Data sets

    onesK = np.ones( K )
    onesObs = np.ones( obsDim )

    ( p, ) = Dirichlet.sample( params=onesObs )
    ys = [ Categorical.sample( params=p, size=nNodes ) for _ in range( D ) ]
    ( initialDist, ) = Dirichlet.sample( params=onesK )
    transitionDists = []
    for ndim in allTransitionCounts:
        transitionDists.append( Dirichlet.sample( params=onesK, size=( K, ) * ( ndim - 1 ) ) )
    emissionDist = Dirichlet.sample( params=onesObs, size=K )

    # asdf = 0.1
    # fdsa = 0.2
    # initialDist = np.array( [ 1. - fdsa, fdsa ] )
    # # initialDist = np.array( [ 1. - fdsa, fdsa ] )
    # emissionDist = np.array( [ [ asdf     , 1. - asdf ],
    #                            [ 1. - asdf, asdf      ] ] )

    # transDist = np.array( [ [ [ 1. - asdf, asdf      ],
    #                           [      asdf, 1. - asdf ] ] ,
    #                         [ [ 1. - asdf, asdf      ],
    #                           [      asdf, 1. - asdf ] ] ] )

    # print( 'TRANSITION', transDist)

    # ys = [ np.array( [ 0, 0, 0 ] ) ]

    print('\n\nys:', ys)
    print( 'emissionDist', emissionDist )
    print('\n\n')

    msg = GraphCategoricalForwardBackward( K=K )
    msg.updateParamsFromGraphs( ys, initialDist, transitionDists, emissionDist, graphs )

    msg.draw()

    U, V = msg.filter()

    # Make sure that things sum to 1
    returnLog = True
    for n, probs in zip( msg.nodes, msg.nodeSmoothed( U, V, msg.nodes, returnLog=returnLog ) ):
        reduced = np.logaddexp.reduce( probs ) if returnLog else np.sum( probs )
        print( '\nP( x_%d | Y ) for'%( n ), ':', probs, '->', reduced )

    for n, probs in zip( msg.nodes, msg.conditionalParentChild( U, V, msg.nodes, returnLog=returnLog ) ):
        reduced = np.logaddexp.reduce( probs, axis=-1 ) if returnLog else probs.sum( axis=-1 )
        # print( '\nP( x_%d | x_p1..pN, Y ) for'%( n ), ':', probs, '->', reduced )
        print( '\nP( x_%d | x_p1..pN, Y ) for'%( n ), '->', reduced.sum() )


testGraphCategoricalForwardBackwardNoCycle()