import numpy as np
from GenModels.GM.States.GraphicalMessagePassing import *
from GenModels.GM.Distributions import *
from GenModels.GM.Models.DiscreteGraphModels import *
import time
from collections import Iterable
import itertools

__all__ = [ 'graphMarginalizationTest' ]

def graphToDataGraph( graphs, dataPerNode, with_fbs=False, random_latent_states=False, d_latent=None ):
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

        if( random_latent_states ):
            assert d_latent is not None
            for node in data_graph.nodes:
                possible_latent_states = np.array( list( set( np.random.choice( np.arange( d_latent ), d_latent - 1 ).tolist() ) ) )
                data_graph.setPossibleLatentStates( node, possible_latent_states )

        if( with_fbs ):
            data_graphs.append( ( data_graph, fbs ) )
        else:
            data_graphs.append( data_graph )
    return data_graphs

##################################################################################################

def graphToGroupGraph( graphs, dataPerNode, groupPerNode, with_fbs=False, random_latent_states=False, d_latents=None ):
    assert isinstance( graphs, list )
    group_graphs = []
    for graph in graphs:

        if( with_fbs ):
            if( not isinstance( graph, Graph ) ):
                graph, fbs = graph
            else:
                graph, fbs = graph, np.array( [] )

        data = [ ( node, dataPerNode( node ) ) for node in graph.nodes ]
        group = [ ( node, groupPerNode( node ) ) for node in graph.nodes ]
        group_graph = GroupGraph.fromGraph( graph, data, group )

        if( random_latent_states ):
            assert d_latents is not None
            for node in group_graph.nodes:
                group = group_graph.groups[ node ]
                possible_latent_states = np.array( list( set( np.random.choice( np.arange( d_latents[ group ] ), d_latents[ group ] - 1 ).tolist() ) ) )
                group_graph.setPossibleLatentStates( node, possible_latent_states )

        if( with_fbs ):
            group_graphs.append( ( group_graph, fbs ) )
        else:
            group_graphs.append( group_graph )
    return group_graphs

##################################################################################################

class MarginalizationTester():

    def __init__( self, graphs, d_latent=3, d_obs=4, measurements=2, random_latent_states=False ):
        self.d_latent = d_latent
        self.d_obs = d_obs
        self.measurements = measurements
        self.random_latent_states = random_latent_states
        self.graphs = self.fillInGraphs( graphs )

    def fillInGraphs( self, graphs ):
        def dataPerNode( node ):
            return Categorical.generate( D=self.d_obs, size=self.measurements )
        return graphToDataGraph( graphs, dataPerNode, with_fbs=False, random_latent_states=self.random_latent_states, d_latent=self.d_latent )

    #################################################

    def generateDists( self ):
        initial_shape, transition_shapes, emission_shape = GHMM.parameterShapes( self.graphs, self.d_latent, self.d_obs )
        initial_dist = Dirichlet.generate( D=initial_shape )
        transition_dists = [ TensorTransitionDirichletPrior.generate( Ds=s ) for s in transition_shapes ]
        emission_dist = TensorTransitionDirichletPrior.generate( Ds=emission_shape )
        return initial_dist, transition_dists, emission_dist

    #################################################

    @property
    def msg( self ):
        return GraphHMM()

    #################################################

    def timeFilter( self ):
        initial_dist, transition_dists, emission_dist = self.generateDists()
        graphs = self.graphs

        msg = self.msg
        msg.updateParams( initial_dist, transition_dists, emission_dist, graphs )

        start = time.time()
        U, V = msg.filter()
        end = time.time()
        print( 'Filter took', end - start, 'seconds' )

    #################################################

    def runNodeJoint( self ):
        initial_dist, transition_dists, emission_dist = self.generateDists()
        graphs = self.graphs

        msg = self.msg
        msg.updateParams( initial_dist, transition_dists, emission_dist, graphs )
        print( 'About to filter' )
        U, V = msg.filter()

        print( '\nJoint' )
        for n, probs in msg.nodeJoint( U, V, msg.nodes ):
            reduced = msg.integrate( probs, axes=range( probs.ndim ) )
            print( 'P( x_%d, Y )'%( n ), ':', probs, '->', reduced )

    #################################################

    def runJointParents( self ):
        initial_dist, transition_dists, emission_dist = self.generateDists()
        graphs = self.graphs

        msg = self.msg
        msg.updateParams( initial_dist, transition_dists, emission_dist, graphs )
        print( 'About to filter' )
        U, V = msg.filter()

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

    #################################################

    def runJointParentChild( self ):
        initial_dist, transition_dists, emission_dist = self.generateDists()
        graphs = self.graphs

        msg = self.msg
        msg.updateParams( initial_dist, transition_dists, emission_dist, graphs )
        print( 'About to filter' )
        U, V = msg.filter()

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

    #################################################

    def run( self ):
        initial_dist, transition_dists, emission_dist = self.generateDists()
        graphs = self.graphs

        msg = self.msg
        msg.updateParams( initial_dist, transition_dists, emission_dist, graphs )
        msg.draw()
        U, V = msg.filter()

        ####################################################

        print( '\nJoint' )
        for n, probs in msg.nodeJoint( U, V, msg.nodes ):
            reduced = msg.integrate( probs, axes=range( probs.ndim ) )
            print( 'P( x_%d, Y )'%( n ), ':', probs, '->', reduced )

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
            reduced = msg.integrate( probs, axes=[ -1 ] )
            print( 'P( x_%d | Y )'%( n ), ':', probs, '->', probs.shape, reduced )
            # assert np.allclose( reduced, 0.0 )

        ####################################################

        print( '\nChild given parents' )
        for n, probs in msg.conditionalParentChild( U, V, msg.nodes ):
            # If we reduce over the last axis, we should have everything sum to 1
            reduced = msg.integrate( probs, axes=[ -1 ] )
            print( 'P( x_%d | x_p1..pN, Y )'%( n ), '->', probs.shape, reduced )
            # assert np.allclose( reduced, 0.0 )

##################################################################################################

class MarginalizationTesterFBS( MarginalizationTester ):

    @property
    def msg( self ):
        return GraphHMMFBS()

    def fillInGraphs( self, graphs ):
        def dataPerNode( node ):
            return Categorical.generate( D=self.d_obs, size=self.measurements )
        return graphToDataGraph( graphs, dataPerNode, with_fbs=True, random_latent_states=self.random_latent_states, d_latent=self.d_latent )

##################################################################################################

class MarginalizationTesterFBSParallel( MarginalizationTesterFBS ):

    @property
    def msg( self ):
        return GraphHMMFBSParallel()

##################################################################################################

class MarginalizationTesterFBSGroup( MarginalizationTesterFBS ):

    def __init__( self, graphs, d_latents=[ 2, 3, 4 ], d_obs=4, measurements=2, groups=3, random_latent_states=False ):

        self.d_latents = d_latents
        self.d_obs = d_obs
        self.measurements = measurements
        self.groups = groups
        self.random_latent_states = random_latent_states
        self.graphs = self.fillInGraphs( graphs )

    @property
    def msg( self ):
        return GraphHMMFBSGroup()

    def fillInGraphs( self, graphs ):
        def dataPerNode( node ):
            return Categorical.generate( D=self.d_obs, size=self.measurements )
        def groupPerNode( node ):
            return Categorical.generate( D=len( self.groups ) )
        return graphToGroupGraph( graphs, dataPerNode, groupPerNode, with_fbs=True, random_latent_states=self.random_latent_states, d_latents=self.d_latents )

    def generateDists( self ):
        initial_shapes, transition_shapes, emission_shapes = GroupGHMM.parameterShapes( self.graphs, self.d_latents, self.d_obs, self.groups )

        initial_dists = dict( [ ( group, Dirichlet.generate( D=shape ) ) for group, shape in initial_shapes.items() ] )

        transition_dists = {}
        for group, shapes in transition_shapes.items():
            transition_dists[ group ] = []
            for shape in shapes:
                trans = TensorTransitionDirichletPrior.generate( Ds=shape )
                transition_dists[ group ].append( trans )

        emission_dists = dict( [ ( g, TensorTransitionDirichletPrior.generate( Ds=[ self.d_latents[ g ], self.d_obs ] ) ) for g, shape in emission_shapes.items() ] )

        return initial_dists, transition_dists, emission_dists

##################################################################################################

class MarginalizationTesterGroupFBSParallel( MarginalizationTesterFBSGroup ):

    @property
    def msg( self ):
        return GraphHMMFBSGroupParallel()

##################################################################################################

def testGraphHMMNoFBS():

    d_latent = 2
    d_obs = 5
    measurements = 2

    # Create the dataset
    graphs = [ graph1(), graph2(), graph3(), graph4(), graph5(), graph6(), graph7() ]

    tester = MarginalizationTester( graphs, d_latent, d_obs, measurements )
    tester.run()

##################################################################################################

def testGraphHMM():

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
    measurements = 2

    tester = MarginalizationTesterFBS( graphs, d_latent, d_obs, measurements )
    tester.run()

##################################################################################################

def testGraphHMMParallel():

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
    measurements = 2

    tester = MarginalizationTesterFBSParallel( graphs, d_latent, d_obs, measurements, random_latent_states=True )
    tester.run()

##################################################################################################

def testGraphGroupHMM():

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

    d_obs = 5
    measurements = 2
    groups = [ 0, 1, 2 ]
    d_latents = dict( zip( groups, [ 2, 3, 4 ] ) )

    tester = MarginalizationTesterFBSGroup( graphs, d_latents, d_obs, measurements, groups, random_latent_states=True )
    tester.run()

##################################################################################################

def testGraphGroupHMMParallel():

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

    d_obs = 5
    measurements = 2
    groups = [ 0, 1, 2 ]
    d_latents = dict( zip( groups, [ 2, 3, 4 ] ) )

    tester = MarginalizationTesterGroupFBSParallel( graphs, d_latents, d_obs, measurements, groups, random_latent_states=True )
    tester.run()

##################################################################################################

def testSpeed():
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

    d_latent = 5
    d_obs = 4
    measurements = 3

    groups = [ 0, 1, 2 ]
    d_latents = dict( zip( groups, [ 2, 3, 4 ] ) )

    regular = MarginalizationTesterFBS( graphs, d_latent, d_obs, measurements )
    start_regular = time.time()
    regular.timeFilter()
    end_regular = time.time()

    parallel = MarginalizationTesterFBSParallel( graphs, d_latent, d_obs, measurements )
    start_parallel = time.time()
    parallel.run()
    end_parallel = time.time()

    group_regular = MarginalizationTesterFBSGroup( graphs, d_latents, d_obs, measurements, groups )
    start_regular_group = time.time()
    group_regular.timeFilter()
    end_regular_group = time.time()

    group_parallel = MarginalizationTesterGroupFBSParallel( graphs, d_latents, d_obs, measurements, groups )
    start_parallel_group = time.time()
    group_parallel.run()
    end_parallel_group = time.time()

    print( 'Non-group parallel:', end_parallel - start_parallel )
    print( 'Non-group regular:', end_regular - start_regular )

    print( 'Group regular:', end_regular_group - start_regular_group )
    print( 'Group parallel:', end_parallel_group - start_parallel_group )

##################################################################################################

def graphMarginalizationTest():
    # testGraphHMMNoFBS()
    # testGraphHMM()
    # testGraphHMMParallel()
    # testGraphGroupHMM()
    testGraphGroupHMMParallel()
    # testSpeed()
    # assert 0