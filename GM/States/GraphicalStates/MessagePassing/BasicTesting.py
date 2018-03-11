from GraphicalMessagePassingBase import Graph, GraphMessagePasser
import numpy as np
from scipy.sparse import coo_matrix

def graph1():
    graph = Graph()

    graph.addEdge( parents=[ 0, 1 ], children=[ 2 ] )

    pMask, cMask = graph.toMatrix()
    fbs = np.zeros( pMask.shape[ 0 ], dtype=bool )

    parentMasks = [ pMask ]
    childMasks = [ cMask ]
    feedbackSets = [ fbs ]

    msg = GraphMessagePasser()
    msg.updateParams( parentMasks, childMasks, feedbackSets )

    return msg

def graph2():
    graph = Graph()

    graph.addEdge( parents=[ 0, 1 ], children=[ 6, 7 ] )
    graph.addEdge( parents=[ 2 ], children=[ 8 ] )
    graph.addEdge( parents=[ 3, 4, 5 ], children=[ 9, 10, 11 ] )
    graph.addEdge( parents=[ 7, 8 ], children=[ 12 ] )
    graph.addEdge( parents=[ 6, 9 ], children=[ 13 ] )
    graph.addEdge( parents=[ 15, 16 ], children=[ 17 ] )
    graph.addEdge( parents=[ 13, 17 ], children=[ 14 ] )

    pMask, cMask = graph.toMatrix()
    fbs = np.zeros( pMask.shape[ 0 ], dtype=bool )

    parentMasks = [ pMask ]
    childMasks = [ cMask ]
    feedbackSets = [ fbs ]

    msg = GraphMessagePasser()
    msg.updateParams( parentMasks, childMasks, feedbackSets )

    return msg

def messagePassingTest():
    # Simulate message passing but do nothing at the filter step

    msg = graph2()

    nothing = lambda x: 0

    msg.messagePassing( nothing, nothing )
    return

    USem, VSem = msg.countSemaphoreInit()

    uList, vList = msg.baseCaseNodes()

    msg.UDone( uList, USem, VSem )
    msg.VDone( vList, USem, VSem )

    print( '\nSemaphore count after marking nodes done' )
    print( 'USem: \n', USem )
    print( 'VSem: \n', VSem.todense() )

    uList = msg.readyForU( USem, uList )
    vList = msg.readyForV( VSem, vList )

    print( '\nNext node list' )
    print( 'uList: \n', uList )
    print( 'vList: \n', vList )

def cycleGraph1():

    graph = Graph()

    graph.addEdge( parents=[ 0, 1 ], children=[ 3 ] )
    graph.addEdge( parents=[ 1, 2 ], children=[ 4 ] )
    graph.addEdge( parents=[ 2, 3, 4 ], children=[ 5, 6 ] )

    pMask, cMask = graph.toMatrix()
    fbs = coo_matrix( pMask.shape, dtype=bool )

    parentMasks = [ pMask ]
    childMasks = [ cMask ]
    feedbackSets = [ fbs ]

    msg = GraphMessagePasser()
    msg.updateParams( parentMasks, childMasks, feedbackSets )

    return msg
def loopyMessagePassingTest():
    # Simulate message passing but do nothing at the filter step

    msg = cycleGraph1()

    USem, VSem = msg.countSemaphoreInit()

    print( '\nInitial semaphore count' )
    print( 'USem: \n', USem )
    print( 'VSem: \n', VSem.todense() )

    uList, vList = msg.baseCaseNodes()

    print( '\nInitial node list' )
    print( 'uList: \n', uList )
    print( 'vList: \n', vList )

    msg.UDone( uList, USem, VSem )
    msg.VDone( vList, None, USem, VSem )


    print( '\nSemaphore count after marking nodes done' )
    print( 'USem: \n', USem )
    print( 'VSem: \n', VSem.todense() )

    uList = msg.readyForU( USem )
    vList = msg.readyForV( VSem )

    print( '\nNext node list' )
    print( 'uList: \n', uList )
    print( 'vList: \n', vList )


def tests():
    messagePassingTest()
    # loopyMessagePassingTest()

tests()