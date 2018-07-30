import numpy as np
from GenModels.GM.States.GraphicalMessagePassing import *
from GenModels.GM.Distributions import *
import time
from collections import Iterable
import itertools

__all__ = [ 'graphMarginalizationTest' ]

def graphToDataGraph( graphs, dataPerNode, with_fbs=False ):
    assert isinstance( graphs, list )
    data_graphs = []
    for graph in graphs:

        if( with_fbs ):
            if( not isinstance( graph, Graph ) ):
                graph, fbs = graph
            else:
                graph, fbs = graph, np.array( [] )

        data = [ ( node, dataPerNode( node ) ) for node in graph.nodes ]
        data_graph = DataGraph.fromGraph( graph, data )

        if( with_fbs ):
            data_graphs.append( ( data_graph, fbs ) )
        else:
            data_graphs.append( data_graph )
    return data_graphs

def testGraphCategoricalForwardBackwardNoFBS():


    d_latent = 2
    d_obs = 5
    D = 2

    # Create the dataset
    graphs = [ graph1(), graph2(), graph3(), graph4(), graph5(), graph6(), graph7() ]
    def dataPerNode( node ):
        return Categorical.generate( D=d_obs, size=D )
    data_graphs = graphToDataGraph( graphs, dataPerNode, with_fbs=False )

    # Initial dist
    initial_dists = Dirichlet.generate( D=d_latent )

    # Check how many transition distributions we need
    all_transition_counts = set()
    for graph in graphs:
        if( isinstance( graph, Iterable ) ):
            graph, fbs = graph
        for parents in graph.edge_parents:
            ndim = len( parents ) + 1
            all_transition_counts.add( ndim )

    # Create the transition distribution
    transition_dists = []
    for ndim in all_transition_counts:
        shape = [ d_latent for _ in range( ndim ) ]
        trans = np.empty( shape )
        for indices in itertools.product( *[ range( s ) for s in shape[ 1: ] ] ):
            trans[ indices ] = Dirichlet.generate( D=d_latent )

        transition_dists.append( trans )

    # Emission dist
    emission_dist = Dirichlet.generate( D=d_obs, size=d_latent )

    # Create the message passer and initialize
    msg = GraphCategoricalForwardBackward()
    msg.updateParams( initial_dists, transition_dists, emission_dist, data_graphs )

    # Draw the graphs
    msg.draw()

    # Filter
    U, V = msg.filter()

    print( 'Done with filter' )

    def totalLogReduce( probs ):
        reduced = probs
        while( reduced.ndim >= 1 ):
            reduced = np.logaddexp.reduce( reduced )
        return reduced

    print( '\nJoint' )
    for n, probs in msg.nodeJoint( U, V, msg.nodes ):
        reduced = totalLogReduce( probs )
        print( 'P( x_%d, Y )'%( n ), ':', probs, '->', reduced )

    print( '\nJoint parents' )
    for n, probs in msg.jointParents( U, V, msg.nodes ):
        reduced = totalLogReduce( probs )
        print( 'P( x_p1..pN, Y ) for %d'%( n ), '->', reduced )

    print( '\nJoint parent child' )
    for n, probs in msg.jointParentChild( U, V, msg.nodes ):
        reduced = totalLogReduce( probs )
        print( 'P( x_%d, x_p1..pN, Y )'%( n ), '->', reduced )

    print( '\nSmoothed' )
    for n, probs in msg.nodeSmoothed( U, V, msg.nodes ):
        # If we reduce over the last axis, we should have everything sum to 1
        reduced = np.abs( np.logaddexp.reduce( probs, axis=-1 ) ).sum()
        print( 'P( x_%d | Y )'%( n ), ':', probs, '->', reduced )

    print( '\nChild given parents' )
    for n, probs in msg.conditionalParentChild( U, V, msg.nodes ):
        # If we reduce over the last axis, we should have everything sum to 1
        reduced = np.abs( np.logaddexp.reduce( probs, axis=-1 ) ).sum()
        print( 'P( x_%d | x_p1..pN, Y )'%( n ), '->', reduced )

    print( 'Done with the testGraphCategoricalForwardBackwardNoFBS test!!\n' )

def testGraphCategoricalForwardBackward():

    np.random.seed( 2 )

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
               cycleGraph11(),
               cycleGraph12() ]

    d_latent = 2
    d_obs = 5
    D = 2

    # Create the dataset
    def dataPerNode( node ):
        return Categorical.generate( D=d_obs, size=D )
    data_graphs = graphToDataGraph( graphs, dataPerNode, with_fbs=True )

    # Initial dist
    initial_dists = Dirichlet.generate( D=d_latent )

    # Check how many transition distributions we need
    all_transition_counts = set()
    for graph in graphs:
        if( isinstance( graph, Iterable ) ):
            graph, fbs = graph
        for parents in graph.edge_parents:
            ndim = len( parents ) + 1
            all_transition_counts.add( ndim )

    # Create the transition distribution
    transition_dists = []
    for ndim in all_transition_counts:
        shape = [ d_latent for _ in range( ndim ) ]
        trans = np.empty( shape )
        for indices in itertools.product( *[ range( s ) for s in shape[ 1: ] ] ):
            trans[ indices ] = Dirichlet.generate( D=d_latent )

        transition_dists.append( trans )

    # Emission dist
    emission_dist = Dirichlet.generate( D=d_obs, size=d_latent )

    # Create the message passer and initialize
    msg = GraphCategoricalForwardBackwardFBS()
    msg.updateParams( initial_dists, transition_dists, emission_dist, data_graphs )

    # Draw the graphs
    # msg.draw( styles={ 0:dict( style='filled', color='red' ) }, node_to_style_key=dict( [ ( n, 0 ) for n in msg.fbs ] ) )
    msg.draw( use_partial=True )

    # Filter
    U, V = msg.filter()

    print( 'Done with filter' )

    def totalLogReduce( probs ):
        reduced = probs
        while( reduced.ndim >= 1 ):
            reduced = np.logaddexp.reduce( reduced )
        return reduced

    ####################################################

    print( '\nJoint' )
    for n, probs in msg.nodeJoint( U, V, msg.nodes ):
        reduced = msg.integrate( probs, axes=range( probs.ndim ) )
        print( 'P( x_%d, Y )'%( n ), ':', probs, '->', reduced )

    # assert 0
    ####################################################

    print( '\nJoint parents' )
    for n, probs in msg.jointParents( U, V, msg.nodes ):
        reduced = msg.integrate( probs, axes=range( probs.ndim ) )
        print( 'P( x_p1..pN, Y ) for %d'%( n ), '->', probs.shape, reduced )

    ####################################################

    print( '\nJoint parents should marginalize out to joint probs' )
    for n, probs in msg.jointParents( U, V, msg.nodes ):
        parents, parent_order = msg.getParents( n, get_order=True )
        joints = msg.nodeJoint( U, V, parents )
        for i, ( ( p, j ), o ) in enumerate( zip( joints, parent_order ) ):
            # Marginalize out the other parents from probs
            int_axes = np.setdiff1d( parent_order, o )
            reduced = msg.integrate( probs, axes=int_axes )
            print( 'sum_{ parents except %d }P( x_p1..pN, Y ) for node %d - P( x_%d, Y ) : ->'%( p, n, p ), ( j - reduced ).sum() )
            assert np.allclose( reduced, j ), 'reduced: %s, j: %s'%( reduced, j )

    ####################################################

    print( '\nJoint parent child' )
    for n, probs in msg.jointParentChild( U, V, msg.nodes ):
        reduced = msg.integrate( probs, axes=range( probs.ndim ) )
        print( 'P( x_%d, x_p1..pN, Y )'%( n ), '->', probs.shape, reduced )

    ####################################################

    print( '\nJoint parent child should marginalize out to joint probs' )
    for n, probs in msg.jointParentChild( U, V, msg.nodes ):
        parents, parent_order = msg.getParents( n, get_order=True )
        n_parents = parents.shape[ 0 ]

        joints = msg.nodeJoint( U, V, parents )
        for i, ( ( p, j ), o ) in enumerate( zip( joints, parent_order ) ):
            # Marginalize out the other parents from probs
            int_axes = np.setdiff1d( np.hstack( ( n_parents, parent_order ) ), o )
            reduced = msg.integrate( probs, axes=int_axes )
            print( 'sum_{ parents except %d }P( x_%d, x_p1..pN, Y ) for node %d - P( x_%d, Y ) : ->'%( p, p, n, p ), ( j - reduced ).sum() )
            assert np.allclose( reduced, j ), 'reduced: %s, j: %s'%( reduced, j )

        ( _, joint ), = msg.nodeJoint( U, V, [ n ] )
        # Marginalize out all of the parents
        reduced = msg.integrate( probs, axes=parent_order )
        print( 'sum_{ parents }P( x_%d, x_p1..pN, Y ) - P( x_%d, Y ) : ->'%( n, n ), ( joint - reduced ).sum() )
        assert np.allclose( reduced, joint ), 'reduced: %s, j: %s'%( reduced, j )

    ####################################################

    print( '\nSmoothed' )
    for n, probs in msg.nodeSmoothed( U, V, msg.nodes ):
        # If we reduce over the last axis, we should have everything sum to 1
        reduced = np.abs( msg.integrate( probs, axes=[ -1 ] ) ).sum()
        print( 'P( x_%d | Y )'%( n ), ':', probs, '->', probs.shape, reduced )

    ####################################################

    print( '\nChild given parents' )
    for n, probs in msg.conditionalParentChild( U, V, msg.nodes ):
        # If we reduce over the last axis, we should have everything sum to 1
        reduced = np.abs( msg.integrate( probs, axes=[ -1 ] ) ).sum()
        print( 'P( x_%d | x_p1..pN, Y )'%( n ), '->', probs.shape, reduced )

def graphMarginalizationTest():
    # testGraphCategoricalForwardBackwardNoFBS()
    testGraphCategoricalForwardBackward()
    assert 0