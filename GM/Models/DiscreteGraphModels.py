from GenModels.GM.Distributions import Categorical
from GenModels.GM.States.GraphicalMessagePassing import *
from .DiscreteGraphParameters import *
from .DiscreteGraphOptimizers import *
import autograd.numpy as np
from collections import Iterable
from functools import partial
from abc import ABC, abstractmethod

__all__ = [
    'GHMM',
    'GroupGHMM',
    'GSVAE',
    'GroupGSVAE',
    'DES',
    'GroupDES' ]

class GHMMBase( ABC ):

    def __init__( self, graphs=None, msg=None, params=None, opt=None, svi_model_type=None, prior_strength=1.0, method='SVI', priors=None, **kwargs ):
        if( graphs is not None ):
            self.graphs = graphs

        self.msg = msg
        self.method = method
        self.params = params
        self.svi_model_type = svi_model_type

        if( opt is not None ):
            self.opt = opt
        else:
            # If we're doing SVI, we can't create the model until later
            self.step_size = kwargs[ 'step_size' ]
            self.minibatch_size = kwargs[ 'minibatch_size' ]
            if( graphs is not None ):
                self.setData( graphs )

        if( method != 'SVI' ):
            if( graphs is not None ):
                self.msg.preprocessData( self.graphs )

        self.initModel()

    ###########################################

    @abstractmethod
    def initModel( self ):
        pass

    def setGraphs( self, graphs ):
        self.graphs = graphs
        self.msg.updateGraphs( graphs )

    ###########################################

    def setData( self, graphs ):
        self.graphs = graphs

        if( self.method != 'SVI' ):
            self.msg.preprocessData( self.graphs )
        else:
            self.total_nodes = sum( [ len( graph.nodes ) for graph, fbs in self.graphs ] )
            minibatch_ratio = self.minibatch_size / len( self.graphs )
            self.opt = self.svi_model_type( msg=self.msg, parameters=self.params, minibatch_ratio=minibatch_ratio, step_size=self.step_size )

    ###########################################

    @abstractmethod
    def parameterShapes( self, *args, **kwargs ):
        pass

    ###########################################

    def stateUpdate( self, **kwargs ):

        if( self.method != 'SVI' ):
            return self.opt.stateUpdate()

        minibatch_indices = np.random.randint( len( self.graphs ), size=self.minibatch_size )
        minibatch = [ self.graphs[ i ] for i in minibatch_indices ]
        self.msg.preprocessData( minibatch )

        # Compute minibatch ratio
        minibatch_nodes = sum( [ len( graph.nodes ) for graph, fbs in minibatch ] )
        self.params.setMinibatchRatio( self.total_nodes / minibatch_nodes )

        return self.opt.stateUpdate()

    ###########################################

    def fitStep( self, **kwargs ):

        if( self.method != 'SVI' ):
            return self.opt.fitStep( **kwargs )

        minibatch_indices = np.random.randint( len( self.graphs ), size=self.minibatch_size )
        minibatch = [ self.graphs[ i ] for i in minibatch_indices ]
        self.msg.preprocessData( minibatch )

        # Compute minibatch ratio
        minibatch_nodes = sum( [ len( graph.nodes ) for graph, fbs in minibatch ] )
        self.params.setMinibatchRatio( self.total_nodes / minibatch_nodes )

        return self.opt.fitStep()

    ###########################################

    @abstractmethod
    def stateSampleHelper( self, *args, **kwargs ):
        pass

    def sampleStates( self, measurements=1 ):

        # Generate data
        node_states = {}
        node_emissions = {}
        self.msg.forwardPass( partial( self.stateSampleHelper, node_states=node_states, node_emissions=node_emissions, measurements=measurements ) )

        return node_states, node_emissions

    ###########################################

    def sampleParams( self ):
        self.msg.updateParams( self.params.sampleInitialDist(), self.params.sampleTransitionDist(), self.params.sampleEmissionDist() )

    ###########################################

    def marginal( self ):
        U, V = self.msg.filter()
        return self.msg.marginalProb( U, V )

    def generative( self ):
        assert self.method == 'Gibbs'
        return self.opt.generativeProbability()

    def paramProb( self ):
        return self.params.paramProb()

######################################################################

class GHMM( GHMMBase ):

    def __init__( self, graphs=None, prior_strength=1.0, method='SVI', priors=None, params=None, **kwargs ):
        assert method in [ 'EM', 'Gibbs', 'CAVI', 'SVI' ]
        assert priors is not None

        # Create the message passer
        msg = GraphHMMFBSParallel()

        # Create the parameters
        root_prior, trans_priors, emiss_prior = priors
        if( method == 'EM' ):
            params = EMParameters( root_prior, trans_priors, emiss_prior ) if params is None else params
        elif( method == 'Gibbs' ):
            params = GibbsParameters( root_prior, trans_priors, emiss_prior ) if params is None else params
        elif( method == 'CAVI' ):
            params = CAVIParameters( root_prior, trans_priors, emiss_prior ) if params is None else params
        else:
            params = SVIParameters( root_prior, trans_priors, emiss_prior ) if params is None else params

        # Generate the model objects
        if( method == 'EM' ):
            opt = EM( msg=msg, parameters=params )
        elif( method == 'Gibbs' ):
            opt = Gibbs( msg=msg, parameters=params )
        elif( method == 'CAVI' ):
            opt = CAVI( msg=msg, parameters=params )
        else:
            opt = None

        super().__init__( graphs=graphs,
                          msg=msg,
                          params=params,
                          opt=opt,
                          svi_model_type=SVI,
                          prior_strength=prior_strength,
                          method=method,
                          priors=priors,
                          **kwargs )

    ###########################################

    def initModel( self ):
        initial = self.params.initial_dist.pi
        transition = [ dist.pi for dist in self.params.transition_dists ]
        emission = self.params.emission_dist.pi
        self.msg.updateParams( initial, transition, emission )

    ###########################################

    @staticmethod
    def parameterShapes( graphs, d_latent, d_obs ):
        # Returns the shapes of the parameters that fit graph

        # Initial dist
        initial_shape = d_latent

        # Check how many transition distributions we need
        all_transition_counts = set()
        for graph in graphs:
            if( isinstance( graph, Iterable ) ):
                graph, fbs = graph
            for parents in graph.edge_parents:
                ndim = len( parents ) + 1
                all_transition_counts.add( ndim )

        # Create the transition distribution
        transition_shapes = []
        for ndim in all_transition_counts:
            shape = [ d_latent for _ in range( ndim ) ]
            transition_shapes.append( shape )

        # Emission dist
        emission_shape = [ d_latent, d_obs ]

        return initial_shape, transition_shapes, emission_shape

    ###########################################

    def stateSampleHelper( self, node_list, node_states, node_emissions, measurements=1 ):
        # Compute P( x_c | x_p1..pN ) and P( y_c | x_c )

        for node in node_list:
            parents, parent_order = self.msg.getParents( node, get_order=True )
            ndim = len( parents ) + 1
            if( len( parents ) == 0 ):
                prob = self.msg.pi0
            else:
                indices = tuple( [ [ node_states[ o ] ] for o in parent_order ] )
                prob = self.msg.pis[ ndim ][ indices ].ravel()

            state = Categorical.sample( nat_params=( prob, ) )[ 0 ]
            node_states[ node ] = state
            node_emissions[ node ] = Categorical.sample( nat_params=( self.msg.emission_dist[ state ], ), size=measurements )

######################################################################

class GroupGHMM( GHMMBase ):

    def __init__( self, graphs=None, prior_strength=1.0, method='SVI', priors=None, params=None, **kwargs ):

        assert method in [ 'EM', 'Gibbs', 'CAVI', 'SVI' ]
        assert priors is not None

        # Create the message passer
        msg = GraphHMMFBSGroupParallel()

        # Create the parameters
        root_priors, trans_priors, emiss_priors = priors
        if( method == 'EM' ):
            params = GroupEMParameters( root_priors, trans_priors, emiss_priors ) if params is None else params
        elif( method == 'Gibbs' ):
            params = GroupGibbsParameters( root_priors, trans_priors, emiss_priors ) if params is None else params
        elif( method == 'CAVI' ):
            params = GroupCAVIParameters( root_priors, trans_priors, emiss_priors ) if params is None else params
        else:
            params = GroupSVIParameters( root_priors, trans_priors, emiss_priors ) if params is None else params

        # Generate the opt objects
        if( method == 'EM' ):
            opt = GroupEM( msg=msg, parameters=params )
        elif( method == 'Gibbs' ):
            opt = GroupGibbs( msg=msg, parameters=params )
        elif( method == 'CAVI' ):
            opt = GroupCAVI( msg=msg, parameters=params )
        else:
            opt = None

        super().__init__( graphs=graphs,
                          msg=msg,
                          params=params,
                          opt=opt,
                          svi_model_type=GroupSVI,
                          prior_strength=prior_strength,
                          method=method,
                          priors=priors,
                          **kwargs )

    ###########################################

    def initModel( self ):
        initial = {}
        transition = {}
        emission = {}
        for group in self.params.initial_dists.keys():
            initial[ group ] = self.params.initial_dists[ group ].pi
            transition[ group ] = [ dist.pi for shape, dist in self.params.transition_dists[ group ].items() ]
            emission[ group ] = self.params.emission_dists[ group ].pi
        self.msg.updateParams( initial, transition, emission )

    ###########################################

    @staticmethod
    def parameterShapes( graphs, d_latents, d_obs, groups ):
        # Returns the shapes of the parameters that fit graph

        assert isinstance( d_latents, dict )

        # Initial dist
        initial_shapes = dict( [ ( g, d_latents[ g ] ) for g in groups ] )

        # Check how many transition distributions we need
        all_transition_counts = dict( [ ( group, set() ) for group in groups ] )
        for graph in graphs:
            if( isinstance( graph, Iterable ) ):
                graph, fbs = graph
            for children, parents in zip( graph.edge_children, graph.edge_parents ):
                ndim = len( parents ) + 1

                parent_groups = [ graph.groups[ parent ] for parent in parents ]

                for child in children:
                    child_group = graph.groups[ child ]
                    family_groups = parent_groups + [ child_group ]
                    shape = tuple( [ d_latents[ group ] for group in family_groups ] )
                    all_transition_counts[ child_group ].add( shape )

        # Create the transition distribution
        transition_shapes = {}
        for group in all_transition_counts:
            transition_shapes[ group ] = []
            for shape in all_transition_counts[ group ]:
                transition_shapes[ group ].append( shape )

        # Emission dist
        emission_shapes = dict( [ ( g, [ d_latents[ g ], d_obs ] ) for g in groups ] )

        return initial_shapes, transition_shapes, emission_shapes

    ###########################################

    def stateSampleHelper( self, node_list, node_states, node_emissions, measurements=1 ):
        # Compute P( x_c | x_p1..pN ) and P( y_c | x_c )

        for node in node_list:
            parents, parent_order = self.msg.getParents( node, get_order=True )
            group = self.msg.node_groups[ node ]
            if( len( parents ) == 0 ):
                prob = self.msg.pi0s[ group ]
            else:
                shape = []
                for p, _ in sorted( zip( parents, parent_order ), key=lambda po: po[ 1 ] ):
                    g = self.msg.node_groups[ p ]
                    shape.append( self.msg.pi0s[ g ].shape[ 0 ] )
                shape.append( self.msg.pi0s[ group ].shape[ 0 ] )
                shape = tuple( shape )

                indices = tuple( [ [ node_states[ p ] ] for p in parents ] )
                self.msg.pis[ group ]
                self.msg.pis[ group ][ shape ][ indices ]
                prob = self.msg.pis[ group ][ shape ][ indices ].ravel()

            state = Categorical.sample( nat_params=( prob, ) )[ 0 ]
            node_states[ node ] = state
            node_emissions[ node ] = Categorical.sample( nat_params=( self.msg.emission_dists[ group ][ state ], ), size=measurements )

######################################################################

class GSVAE():

    def __init__( self, graphs=None, prior_strength=1.0, priors=None, d_obs=None, **kwargs ):

        assert priors is not None
        root_priors, trans_priors = priors

        if( graphs is not None ):
            self.graphs = graphs

        self.msg = GraphDiscreteSVAEConditioned()

        minibatch_ratio = 1.0

        root_priors, trans_priors = priors
        assert d_obs is not None
        self.params = SVAEParameters( root_priors, trans_priors, d_obs, minibatch_ratio )

        self.opt = SVAE( self.msg, self.params, minibatch_ratio )

        if( graphs is not None ):
            self.msg.preprocessData( self.graphs )

    def fit( self, num_iters=100 ):
        return self.opt.train( num_iters=num_iters )

    @staticmethod
    def parameterShapes( graphs, d_latent, d_obs ):
        # Returns the shapes of the parameters that fit graph

        # Initial dist
        initial_shape = d_latent

        # Check how many transition distributions we need
        all_transition_counts = set()
        for graph in graphs:
            if( isinstance( graph, Iterable ) ):
                graph, fbs = graph
            for parents in graph.edge_parents:
                ndim = len( parents ) + 1
                all_transition_counts.add( ndim )

        # Create the transition distribution
        transition_shapes = []
        for ndim in all_transition_counts:
            shape = [ d_latent for _ in range( ndim ) ]
            transition_shapes.append( shape )

        return initial_shape, transition_shapes

######################################################################

class GroupGSVAE():

    def __init__( self, graphs=None, prior_strength=1.0, priors=None, d_obs=None, **kwargs ):

        assert priors is not None
        root_priors, trans_priors = priors

        if( graphs is not None ):
            self.graphs = graphs

        self.msg = GraphDiscreteGroupSVAEConditioned()

        minibatch_ratio = 1.0

        root_priors, trans_priors = priors
        assert d_obs is not None
        self.params = GroupSVAEParameters( root_priors, trans_priors, d_obs, minibatch_ratio )

        self.opt = GroupSVAE( self.msg, self.params, minibatch_ratio )

        if( graphs is not None ):
            self.msg.preprocessData( self.graphs )

    def fit( self, num_iters=100 ):
        return self.opt.train( num_iters=num_iters )

    @staticmethod
    def parameterShapes( graphs, d_latents, d_obs, groups ):
        # Returns the shapes of the parameters that fit graph

        assert isinstance( d_latents, dict )

        # Initial dist
        initial_shapes = dict( [ ( g, d_latents[ g ] ) for g in groups ] )

        # Check how many transition distributions we need
        all_transition_counts = dict( [ ( group, set() ) for group in groups ] )
        for graph in graphs:
            if( isinstance( graph, Iterable ) ):
                graph, fbs = graph
            for children, parents in zip( graph.edge_children, graph.edge_parents ):
                ndim = len( parents ) + 1

                parent_groups = [ graph.groups[ parent ] for parent in parents ]

                for child in children:
                    child_group = graph.groups[ child ]
                    family_groups = parent_groups + [ child_group ]
                    shape = tuple( [ d_latents[ group ] for group in family_groups ] )
                    all_transition_counts[ child_group ].add( shape )

        # Create the transition distribution
        transition_shapes = {}
        for group in all_transition_counts:
            transition_shapes[ group ] = []
            for shape in all_transition_counts[ group ]:
                transition_shapes[ group ].append( shape )

        return initial_shapes, transition_shapes

######################################################################

class DES( GSVAE ):
    def __init__( self, graphs=None, prior_strength=1.0, priors=None, d_obs=None, inheritance_pattern=None, **kwargs ):
        assert inheritance_pattern is not None
        super().__init__( graphs, prior_strength, priors, d_obs, **kwargs )
        self.opt = DESOpt( self.msg, self.params, inheritance_pattern )

class GroupDES( GroupGSVAE ):
    def __init__( self, graphs=None, prior_strength=1.0, priors=None, d_obs=None, inheritance_pattern=None, **kwargs ):
        assert inheritance_pattern is not None
        super().__init__( graphs, prior_strength, priors, d_obs, **kwargs )
        self.opt = GroupDESOpt( self.msg, self.params, inheritance_pattern )
