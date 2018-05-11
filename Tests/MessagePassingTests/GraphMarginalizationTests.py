import numpy as np
from GenModels.GM.States.GraphicalStates.MessagePassing import *
from GenModels.GM.Distributions import *
import time
from collections import Iterable

__all__ = [ 'graphMarginalizationTest' ]

def testGraphCategoricalForwardBackward():

    # graphs = [ cycleGraph1(), cycleGraph2(), cycleGraph3(), cycleGraph7(), cycleGraph8() ]
    graphs = [ cycleGraph8() ]
    # graphs = [ cycleGraph2(), cycleGraph8() ]
    # graphs = [ graph3() ]
    # graphs = [ graph1(), graph2(), graph3(), graph4(), graph5() ]
    # graphs = [ cycleGraph2() ]
    # graphs = [ cycleGraph8() ]
    # graphs = [ cycleGraph8(), graph1(), graph2(), graph3(), graph4(), graph5() ]

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
    # fdsa = 0.1
    # initialDist = np.array( [ 1. - fdsa, fdsa ] )
    # emissionDist = np.array( [ [ asdf     , 1. - asdf ],
    #                            [ 1. - asdf, asdf      ] ] )

    # transitionDists = [ np.array( [ [ [ 1. - asdf, asdf      ],
    #                                   [      asdf, 1. - asdf ] ] ,
    #                                 [ [ 1. - asdf, asdf      ],
    #                                   [      asdf, 1. - asdf ] ] ] ),
    #                     np.array( [ [ 1. - asdf, asdf      ],
    #                                 [      asdf, 1. - asdf ] ] ) ]

    # print( 'TRANSITION', transitionDists)

    # ys = [ np.array( [ 0, 0 ] ) ]
    # # ys = [ np.array( [ 0, 0, 0 ] ) ]
    # # ys = [ np.array( [ 0, 0, 0, 0, 0 ] ) ]

    print('\n\nys:', ys)
    print( 'emissionDist', emissionDist )
    print('\n\n')

    msg = GraphCategoricalForwardBackward( K=K )
    msg.updateParamsFromGraphs( ys, initialDist, transitionDists, emissionDist, graphs )

    msg.draw()

    U, V = msg.filter()

    print( msg.nodes )

    print( 'Done with filter' )

    finalProbs = []

    # Make sure that things sum to 1
    returnLog = True

    def totalLogReduce( probs, notFirstAxis=False ):
        reduced = probs
        while( reduced.ndim >= 1 ):
            if( notFirstAxis and reduced.ndim == 1 ):
                break
            reduced = np.logaddexp.reduce( reduced, axis=-1 )

        return reduced

    print( 'Joint' )
    for probs in msg.nodeJoint( U, V, msg.nodes, returnLog=returnLog ):
        reduced = totalLogReduce( probs ) if returnLog else np.sum( probs )
        print( probs, '->', reduced )
        finalProbs.append( ( probs, reduced ) )
    finalProbs.append( ( None, '\n' ) )

    print( 'Joint parents' )
    for probs in msg.jointParents( U, V, msg.nodes, returnLog=returnLog ):
        reduced = totalLogReduce( probs ) if returnLog else np.sum( probs )
        print( probs, '->', reduced )
        finalProbs.append( ( probs, reduced ) )
    finalProbs.append( ( None, '\n' ) )

    print( 'Joint parent child' )
    for probs in msg.jointParentChild( U, V, msg.nodes, returnLog=returnLog ):
        reduced = totalLogReduce( probs ) if returnLog else np.sum( probs )
        finalProbs.append( ( probs, reduced ) )
    finalProbs.append( ( None, '\n' ) )

    print( 'Smoothed' )
    for n, probs in zip( msg.nodes, msg.nodeSmoothed( U, V, msg.nodes, returnLog=returnLog ) ):
        reduced = totalLogReduce( probs ) if returnLog else np.sum( probs )
        print( '\nP( x_%d | Y ) for'%( n ), ':', probs, '->', reduced )
        finalProbs.append( ( probs, reduced ) )
    finalProbs.append( ( None, '\n' ) )

    print( 'Child given parents' )
    for n, probs in msg.conditionalParentChild( U, V, msg.nodes, returnLog=returnLog ):
        reduced = totalLogReduce( probs, notFirstAxis=True ) if returnLog else np.sum( probs )
        print( '\nP( x_%d | x_p1..pN, Y ) for'%( n ), '->', reduced )
        finalProbs.append( ( probs, reduced ) )

    print( '\n\n' )
    print( 'Finally, all of these should look similar' )
    for f in finalProbs:
        print( f[ 1 ] )


def graphMarginalizationTest():
    testGraphCategoricalForwardBackward()