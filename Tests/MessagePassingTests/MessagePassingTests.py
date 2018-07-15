import numpy as np

from GenModels.GM.States.GraphicalMessagePassing import *

__all__ = [ 'messagePassingTest' ]

def loadGraphs( graphs, feedback_sets=None, run=True ):
    # Simulate message passing but do nothing at the filter step
    parent_masks = []
    child_masks = []
    for graph in graphs:
        p_mask, cMask = graph.toMatrix()
        parent_masks.append( p_mask )
        child_masks.append( c_mask )

    if( feedback_sets is not None ):
        assert len( graphs ) == len( feedback_sets )

    msg = GraphMessagePasser()
    msg.updateParams( parent_masks, child_masks, feedback_sets=feedback_sets )
    return msg

def nonFBSTest():
    graphs = [ graph1(), graph2(), graph3(), graph4(), graph5(), graph6() ]

    msg = GraphMessagePasser()
    msg.updateParamsFromGraphs( graphs )
    msg.draw( render=True )

    def nothing( a, b ):
        return
    msg.messagePassing( nothing, nothing )

    print( 'Done with the non fbs message passing tests!' )

def fbsTests():
    cycleGraphs = [ cycleGraph1(), cycleGraph2(), cycleGraph3(), cycleGraph7(), cycleGraph8() ]

    msg = GraphMessagePasserFBS()
    msg.updateParamsFromGraphs( cycleGraphs )
    msg.draw( render=True )

    def nothing( a, b ):
        return
    msg.messagePassing( nothing, nothing )

    print( 'Done with the fbs message passing tests!' )

def fbsTestsImproved():
    cycleGraphs = [ cycleGraph1(), cycleGraph2(), cycleGraph3(), cycleGraph7(), cycleGraph8() ]
    # cycleGraphs = [ cycleGraph1() ]

    msg = GraphMessagePasserFBSImproved()
    msg.updateParamsFromGraphs( cycleGraphs )
    msg.draw( render=True )

    def nothing( a, b ):
        return
    msg.messagePassing( nothing, nothing )

    print( 'Done with the improved fbs message passing tests!' )

def messagePassingTest():
    nonFBSTest()
    fbsTests()
    fbsTestsImproved()
    # assert 0
