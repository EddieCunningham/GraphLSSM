import numpy as np
from GenModels.GM.States.GraphicalMessagePassing import *
from GenModels.GM.Distributions import *
import time
from collections import Iterable
import itertools

__all__ = [ 'graphMarginalizationTest' ]

def testGraphCategoricalForwardBackwardNoFBS():

    # graphs = [ graph1() ]
    graphs = [ graph1(), graph2(), graph3(), graph4(), graph5(), graph6(), graph7() ]

    # Check how many transition distributions we need
    all_transition_counts = set()
    for graph in graphs:
        if( isinstance( graph, Iterable ) ):
            graph, fbs = graph
        for parents in graph.edge_parents:
            ndim = len( parents ) + 1
            all_transition_counts.add( ndim )

    n_nodes = 0
    for graph in graphs:
        if( isinstance( graph, Iterable ) ):
            graph, fbs = graph
        n_nodes += len( graph.nodes )

    d_latent = 2      # Latent state size
    d_obs = 5 # Obs state size
    D = 2      # Data sets

    # Dumb way to create labels.  In the future this is going to come from a graph
    ys = [ Categorical.generate( D=d_obs, size=n_nodes ) for _ in range( D ) ]

    # Initial dist
    initial_dists = Dirichlet.generate( D=d_latent )

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

    msg = GraphCategoricalForwardBackward()
    msg.updateParamsFromGraphs( ys, initial_dists, transition_dists, emission_dist, graphs )

    msg.draw()

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

    # graphs = [ graph1(), graph2(), graph3(), graph4(), graph5(), graph6(), graph7() ]
    graphs = [ graph1(), graph2(), graph3(), graph4(), graph5(), graph6(), graph7(), cycleGraph1(), cycleGraph2(), cycleGraph3(), cycleGraph7(), cycleGraph8(), cycleGraph9(), cycleGraph10() ]
    # graphs = [ cycleGraph1(), cycleGraph2(), cycleGraph3(), cycleGraph7(), cycleGraph8(), cycleGraph9(), cycleGraph10() ]
    # graphs = [ cycleGraph3() ]

    # Check how many transition distributions we need
    all_transition_counts = set()
    for graph in graphs:
        if( isinstance( graph, Iterable ) ):
            graph, fbs = graph
        for parents in graph.edge_parents:
            ndim = len( parents ) + 1
            all_transition_counts.add( ndim )

    n_nodes = 0
    for graph in graphs:
        if( isinstance( graph, Iterable ) ):
            graph, fbs = graph
        n_nodes += len( graph.nodes )

    d_latent = 2      # Latent state size
    d_obs = 5 # Obs state size
    D = 2      # Data sets

    ys = [ Categorical.generate( D=d_obs, size=n_nodes ) for _ in range( D ) ]
    initial_dists = Dirichlet.generate( D=d_latent )

    # Create the transition distribution
    transition_dists = []
    for ndim in all_transition_counts:
        shape = [ d_latent for _ in range( ndim ) ]
        trans = np.empty( shape )
        for indices in itertools.product( *[ range( s ) for s in shape[ 1: ] ] ):
            trans[ indices ] = Dirichlet.generate( D=d_latent )

        transition_dists.append( trans )

    emission_dist = Dirichlet.generate( D=d_obs, size=d_latent )

    msg = GraphCategoricalForwardBackwardFBS()
    msg.updateParamsFromGraphs( ys, initial_dists, transition_dists, emission_dist, graphs )

    msg.draw()

    U, V = msg.filter()

    print( 'Done with filter' )

    def totalLogReduce( probs ):
        reduced = probs
        while( reduced.ndim >= 1 ):
            reduced = np.logaddexp.reduce( reduced )
        return reduced

    ####################################################

    print( '\nJoint' )
    for n, probs in msg.nodeJoint( U, V, msg.full_nodes ):
        reduced = msg.integrate( probs, axes=range( probs.ndim ), use_super=True )
        print( 'P( x_%d, Y )'%( n ), ':', probs, '->', reduced )

    ####################################################

    print( '\nJoint parents' )
    for n, probs in msg.jointParents( U, V, msg.full_nodes ):
        reduced = msg.integrate( probs, axes=range( probs.ndim ), use_super=True )
        print( 'P( x_p1..pN, Y ) for %d'%( n ), '->', probs.shape, reduced )

    ####################################################

    print( '\nJoint parents should marginalize out to joint probs' )
    for n, probs in msg.jointParents( U, V, msg.nodes ):
        parents, parent_order = msg.full_parents( n, get_order=True, full_indexing=True, return_full_indexing=True )
        joints = msg.nodeJoint( U, V, parents )
        for i, ( ( p, j ), o ) in enumerate( zip( joints, parent_order ) ):
            # Marginalize out the other parents from probs
            int_axes = np.setdiff1d( parent_order, o )
            reduced = msg.integrate( probs, axes=int_axes, use_super=True )
            print( 'sum_{ parents except %d }P( x_p1..pN, Y ) for node %d - P( x_%d, Y ) : ->'%( p, n, p ), ( j - reduced ).sum() )
            assert np.allclose( reduced, j ), 'reduced: %s, j: %s'%( reduced, j )

    ####################################################

    print( '\nJoint parent child' )
    for n, probs in msg.jointParentChild( U, V, msg.full_nodes ):
        reduced = msg.integrate( probs, axes=range( probs.ndim ), use_super=True )
        print( 'P( x_%d, x_p1..pN, Y )'%( n ), '->', probs.shape, reduced )

    ####################################################

    print( '\nJoint parent child should marginalize out to joint probs' )
    for n, probs in msg.jointParentChild( U, V, msg.nodes ):
        parents, parent_order = msg.full_parents( n, get_order=True, full_indexing=True, return_full_indexing=True )
        n_parents = parents.shape[ 0 ]

        joints = msg.nodeJoint( U, V, parents )
        for i, ( ( p, j ), o ) in enumerate( zip( joints, parent_order ) ):
            # Marginalize out the other parents from probs
            int_axes = np.setdiff1d( np.hstack( ( n_parents, parent_order ) ), o )
            print( 'int_axes', int_axes, 'p', p, 'o', o )
            reduced = msg.integrate( probs, axes=int_axes, use_super=True )
            print( 'sum_{ parents except %d }P( x_%d, x_p1..pN, Y ) for node %d - P( x_%d, Y ) : ->'%( p, p, n, p ), ( j - reduced ).sum() )
            assert np.allclose( reduced, j ), 'reduced: %s, j: %s'%( reduced, j )

        ( _, joint ), = msg.nodeJoint( U, V, [ n ] )
        # Marginalize out all of the parents
        reduced = msg.integrate( probs, axes=parent_order, use_super=True )
        print( 'sum_{ parents }P( x_%d, x_p1..pN, Y ) - P( x_%d, Y ) : ->'%( n, n ), ( joint - reduced ).sum() )
        assert np.allclose( reduced, joint ), 'reduced: %s, j: %s'%( reduced, j )

    ####################################################

    print( '\nSmoothed' )
    for n, probs in msg.nodeSmoothed( U, V, msg.full_nodes ):
        # If we reduce over the last axis, we should have everything sum to 1
        reduced = np.abs( msg.integrate( probs, axes=[ -1 ], use_super=True ) ).sum()
        print( 'P( x_%d | Y )'%( n ), ':', probs, '->', probs.shape, reduced )

    ####################################################

    print( '\nChild given parents' )
    for n, probs in msg.conditionalParentChild( U, V, msg.full_nodes ):
        # If we reduce over the last axis, we should have everything sum to 1
        reduced = np.abs( msg.integrate( probs, axes=[ -1 ], use_super=True ) ).sum()
        print( 'P( x_%d | x_p1..pN, Y )'%( n ), '->', probs.shape, reduced )

def graphMarginalizationTest():
    # testGraphCategoricalForwardBackwardNoFBS()
    testGraphCategoricalForwardBackward()
    # assert 0