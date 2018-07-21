import numpy as np

from GenModels.GM.States.GraphicalMessagePassing import *

__all__ = [ 'messagePassingTest' ]

def loadGraphs( graphs, feedback_sets=None, run=True ):
    # Simulate message passing but do nothing at the filter step
    parent_masks = []
    child_masks = []
    for graph in graphs:
        p_mask, c_mask = graph.toMatrix()
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
    msg.upDown( nothing, nothing )

    print( 'Done with the non fbs message passing tests!' )

def fbsTests():
    graphs = [ graph1(),
               graph2(),
               graph3(),
               graph4(),
               graph5(),
               graph6(),
               graph7(),
               cycleGraph1(),
               cycleGraph2(),
               cycleGraph3(),
               cycleGraph7(),
               cycleGraph8(),
               cycleGraph9(),
               cycleGraph10(),
               cycleGraph11() ]
    graphs = [ cycleGraph3() ]

    msg = GraphMessagePasserFBS()
    msg.updateParamsFromGraphs( graphs )
    msg.draw( render=True )

    def nothing( is_base_case, node_list ):
        return

    count = 0
    def loopyHasConverged():
        nonlocal count
        count += 1
        return count > 3
    msg.upDown( nothing, nothing, loopyHasConverged=loopyHasConverged )

    class worker():

        def __init__( self ):
            self.visited = []

        def __call__( self, node_list ):
            self.visited.append( node_list )

    work = worker()
    msg.forwardPass( work )

    visited = np.hstack( work.visited )
    diff = np.setdiff1d( msg.nodes, visited )
    assert diff.size == 0

    print( 'Done with the improved fbs message passing tests!' )

def messagePassingTest():

    nonFBSTest()
    fbsTests()
