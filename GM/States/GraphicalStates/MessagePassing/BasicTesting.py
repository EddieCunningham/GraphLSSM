from GraphicalMessagePassingBase import Graph, GraphMessagePasser
import numpy as np
from scipy.sparse import coo_matrix

def graph1():
    graph = Graph()

    graph.addEdge( parents=[ 0, 1 ], children=[ 2 ] )
    return graph

def graph2():
    graph = Graph()

    graph.addEdge( parents=[ 0, 1 ], children=[ 4 ] )
    graph.addEdge( parents=[ 2, 3 ], children=[ 5 ] )
    graph.addEdge( parents=[ 4, 5 ], children=[ 6, 7 ] )

    return graph

def graph3():
    graph = Graph()

    graph.addEdge( parents=[ 0, 1 ], children=[ 3 ] )
    graph.addEdge( parents=[ 1, 2 ], children=[ 4 ] )

    return graph

def graph4():
    graph = Graph()

    graph.addEdge( parents=[ 0, 1, 2 ], children=[ 4, 5 ] )
    graph.addEdge( parents=[ 2, 3 ], children=[ 6 ] )

    return graph

def graph5():
    graph = Graph()

    graph.addEdge( parents=[ 0, 1 ], children=[ 6, 7 ] )
    graph.addEdge( parents=[ 2 ], children=[ 8 ] )
    graph.addEdge( parents=[ 3, 4, 5 ], children=[ 9, 10, 11 ] )
    graph.addEdge( parents=[ 7, 8 ], children=[ 12 ] )
    graph.addEdge( parents=[ 6, 9 ], children=[ 13 ] )
    graph.addEdge( parents=[ 15, 16 ], children=[ 17 ] )
    graph.addEdge( parents=[ 13, 17 ], children=[ 14 ] )

    return graph

def cycleGraph1():

    graph = Graph()

    graph.addEdge( parents=[ 0, 1 ], children=[ 2, 3 ] )
    graph.addEdge( parents=[ 2, 3 ], children=[ 4 ] )

    fbs = np.array( [ 2 ] )

    return graph, fbs

def cycleGraph2():

    graph = Graph()

    graph.addEdge( parents=[ 0, 1 ], children=[ 3 ] )
    graph.addEdge( parents=[ 1, 2 ], children=[ 4 ] )
    graph.addEdge( parents=[ 2, 3, 4 ], children=[ 5, 6 ] )

    fbs = np.array( [ 2, 4 ] )

    return graph, fbs

def messagePassingTest( graphs, feedbackSets=None ):
    # Simulate message passing but do nothing at the filter step
    parentMasks = []
    childMasks = []
    for graph in graphs:
        pMask, cMask = graph.toMatrix()
        parentMasks.append( pMask )
        childMasks.append( cMask )

    if( feedbackSets is not None ):
        assert len( graphs ) == len( feedbackSets )

    msg = GraphMessagePasser()
    msg.updateParams( parentMasks, childMasks, feedbackSets=feedbackSets )

    bigGraph = msg.toGraph()
    bigGraph.draw()

    nothing = lambda x: 0
    msg.messagePassing( nothing, nothing )

def noCycleTest():
    # graphs = [ graph3() ]
    graphs = [ graph1(), graph2(), graph3(), graph4(), graph5() ]
    messagePassingTest( graphs )

def cycleTest():

    graphs, fbs = zip( *[ cycleGraph1(), cycleGraph2() ] )
    # graphs, fbs = zip( *[ cycleGraph1() ] )
    messagePassingTest( graphs, feedbackSets=fbs )

def tests():
    # noCycleTest()
    cycleTest()

tests()