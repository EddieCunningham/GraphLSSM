import numpy as np

from GenModels.GM.States.GraphicalMessagePassing import *

__all__ = [ 'messagePassingTest' ]

def loadGraphs( graphs, feedbackSets=None, run=True ):
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
    return msg

def allTest():
    cycleGraphs = [ cycleGraph1(), cycleGraph2(), cycleGraph3(), cycleGraph4(), cycleGraph5(), cycleGraph6() ]
    regGraphs = [ graph1(), graph2(), graph3(), graph4(), graph5() ]

    graphs, fbs = zip( *cycleGraphs )
    graphs = list( graphs ) + regGraphs
    fbs = list( fbs ) + [ None for _ in enumerate( regGraphs ) ]

    msg = loadGraphs( graphs, feedbackSets=fbs )
    msg.draw()

    def nothing( a, b ):
        return
    msg.messagePassing( nothing, nothing, debug=True )

    print( 'Done with the tests!' )

def messagePassingTest():
    allTest()

