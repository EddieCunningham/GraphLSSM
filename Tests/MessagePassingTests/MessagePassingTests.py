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

def nonFBSTest():
    graphs = [ graph1(), graph2(), graph3(), graph4(), graph5(), graph6() ]

    msg = GraphMessagePasser()
    msg.updateParamsFromGraphs( graphs )
    msg.draw( render=True )

    def nothing( a, b ):
        return
    msg.messagePassing( nothing, nothing, debug=True )

    print( 'Done with the non fbs message passing tests!' )

def fbsTests():
    cycleGraphs = [ cycleGraph1(), cycleGraph2(), cycleGraph3(), cycleGraph7(), cycleGraph8() ]

    msg = GraphMessagePasserFBS()
    msg.updateParamsFromGraphs( cycleGraphs )
    msg.draw( render=True )

    def nothing( a, b ):
        return
    msg.messagePassing( nothing, nothing, debug=True )

    print( 'Done with the fbs message passing tests!' )

def messagePassingTest():
    nonFBSTest()
    fbsTests()
