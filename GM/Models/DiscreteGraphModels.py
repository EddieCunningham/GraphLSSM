from GenModels.GM.Distributions import Categorical, Dirichlet, TensorTransition, TensorTransitionDirichletPrior
from GenModels.GM.States.GraphicalMessagePassing import GraphHMMFBS, GraphHMMFBSGroup, GraphHMMFBSParallel, GraphHMMFBSGroupParallel

import numpy as np
from collections import Iterable
from functools import partial

__all__ = [
    'GHMM',
    'GroupGHMM'
]

######################################################################

class Parameters():

    def __init__( self, root_prior, transition_priors, emission_prior ):

        # Initial dist
        self.initial_dist = Categorical( hypers=dict( alpha=root_prior ) )

        # Create the transition distribution
        self.transition_dists = [ TensorTransition( hypers=dict( alpha=trans_prior ) ) for trans_prior in transition_priors ]

        # Emission dist
        self.emission_dist = TensorTransition( hypers=dict( alpha=emission_prior ) )

    def paramProb( self ):
        ans = self.initial_dist.ilog_params()
        ans += sum( [ dist.ilog_params() for dist in trans_dists ] )
        ans += self.emission_dist.ilog_params()
        return ans

class GroupParameters():

    def __init__( self, root_priors, transition_priors, emission_priors ):

        assert isinstance( root_priors, dict )
        assert isinstance( transition_priors, dict )
        assert isinstance( emission_priors, dict )

        # Initial dist
        self.initial_dists = dict( [ ( group, Categorical( hypers=dict( alpha=prior ) ) ) for group, prior in root_priors.items() ] )

        # Create the transition distribution
        self.transition_dists = {}
        for group, priors in transition_priors.items():
            self.transition_dists[ group ] = {}
            for prior in priors:
                shape = prior.shape
                self.transition_dists[ group ][ shape ] = TensorTransition( hypers=dict( alpha=prior ) )

        # Emission dist
        self.emission_dists = dict( [ ( group, TensorTransition( hypers=dict( alpha=prior ) ) ) for group, prior in emission_priors.items() ] )

    def paramProb( self ):
        ans = 0.0
        for group in self.initial_dists.keys():
            ans += self.initial_dists[ group ].ilog_params()
            for trans_dists in self.transition_dists[ group ]:
                ans += sum( [ dist.ilog_params() for dist in trans_dists ] )
            ans += self.emission_dists[ group ].ilog_params()
        return ans

######################################################################

class TransitionBins():
    def __init__( self, msg, graph_state ):
        self.msg = msg
        self.graph_state = graph_state
        self.counts = {}

    def __call__( self, node_list ):
        for node in filter( lambda n: self.msg.nParents( n ) > 0, node_list ):
            parents, order = self.msg.getParents( node, get_order=True )
            ndim = parents.shape[ 0 ] + 1
            if( ndim not in self.counts ):
                self.counts[ ndim ] = [ [] for _ in range( ndim ) ]
            for i, p in zip( order, parents ):
                self.counts[ ndim ][ i ].append( self.graph_state.node_states[ p ] )
            self.counts[ ndim ][ -1 ].append( self.graph_state.node_states[ node ] )

class GibbsParameters( Parameters ):

    def resampleInitialDist( self, msg, graph_state ):
        root_states = np.array( [ state for node, state in graph_state.node_states.items() if msg.nParents( node ) == 0 ] )
        self.initial_dist.resample( x=root_states )

    def resampleTransitionDist( self, msg, graph_state ):
        transition_bins = TransitionBins( msg, graph_state )
        msg.forwardPass( transition_bins )
        for dist in self.transition_dists:
            ndim = dist.pi.ndim
            x = [ np.array( count ) for count in transition_bins.counts[ ndim ] ]
            dist.resample( x )

    def resampleEmissionDist( self, msg, graph_state ):
        states = []
        emissions = []
        for node in msg.nodes:
            state = graph_state.node_states[ node ]
            state_addition = state * np.ones_like( msg.ys[ node ] )
            states.extend( state_addition.tolist() )
            emissions.extend( msg.ys[ node ].tolist() )
        x = [ np.array( states ), np.array( emissions ) ]
        self.emission_dist.resample( x )

    def sampleInitialDist( self ):
        self.initial_dist.resample()
        return self.initial_dist.pi

    def sampleTransitionDist( self ):
        for dist in self.transition_dists:
            dist.resample()
        return [ dist.pi for dist in self.transition_dists ]

    def sampleEmissionDist( self ):
        self.emission_dist.resample()
        return self.emission_dist.pi

######################################################################

class GroupTransitionBins():
    def __init__( self, msg, graph_state, groups ):
        self.msg = msg
        self.graph_state = graph_state
        self.counts = dict( [ ( group, {} ) for group in groups ] )

    def __call__( self, node_list ):
        for node in filter( lambda n: self.msg.nParents( n ) > 0, node_list ):
            group = self.msg.node_groups[ node ]

            parents, order = self.msg.getParents( node, get_order=True )
            shape = []
            for p in parents:
                shape.append( self.msg.getNodeDim( p ) )
            shape.append( self.msg.getNodeDim( node ) )
            shape = tuple( shape )

            if( shape not in self.counts[ group ] ):
                self.counts[ group ][ shape ] = [ [] for _ in shape ]

            for i, p in zip( order, parents ):
                self.counts[ group ][ shape ][ i ].append( self.graph_state.node_states[ p ] )
            self.counts[ group ][ shape ][ -1 ].append( self.graph_state.node_states[ node ] )

class GroupGibbsParameters( GroupParameters ):

    def resampleInitialDist( self, msg, graph_state ):
        root_states = dict( [ ( group, [] ) for group in self.initial_dists.keys() ] )
        for node, state in graph_state.node_states.items():
            if( msg.nParents( node ) == 0 ):
                root_states[ msg.node_groups[ node ] ].append( state )

        for group in self.initial_dists:
            self.initial_dists[ group ].resample( x=np.array( root_states[ group ], dtype=int ) )

    def resampleTransitionDist( self, msg, graph_state ):
        transition_bins = GroupTransitionBins( msg, graph_state, self.transition_dists.keys() )
        msg.forwardPass( transition_bins )

        for group, count_and_shapes in transition_bins.counts.items():
            for shape, counts in count_and_shapes.items():
                dist = self.transition_dists[ group ][ shape ]
                x = [ np.array( count ) for count in counts ]
                dist.resample( x )

    def resampleEmissionDist( self, msg, graph_state ):
        states = dict( [ ( group, [] ) for group in self.emission_dists.keys() ] )
        emissions = dict( [ ( group, [] ) for group in self.emission_dists.keys() ] )
        for node in msg.nodes:
            group = msg.node_groups[ node ]
            state = graph_state.node_states[ node ]
            state_addition = state * np.ones_like( msg.ys[ node ] )
            states[ group ].extend( state_addition.tolist() )
            emissions[ group ].extend( msg.ys[ node ].tolist() )

        for group in self.emission_dists.keys():
            x = [ np.array( states[ group ] ), np.array( emissions[ group ] ) ]
            self.emission_dists[ group ].resample( x )

    def sampleInitialDist( self ):
        sample = {}
        for group, dist in self.initial_dists.items():
            sample[ group ] = dist.iparamSample()[ 0 ]
        return sample

    def sampleTransitionDist( self ):
        sample = {}
        for group, dists in self.transition_dists.items():
            sample[ group ] = [ dist.iparamSample()[ 0 ] for shape, dist in dists.items() ]
            # sample[ group ] = dict( [ ( shape, dist.iparamSample()[ 0 ] ) for shape, dist in dists.items() ] )
        return sample

    def sampleEmissionDist( self ):
        sample = {}
        for group, dist in self.emission_dists.items():
            sample[ group ] = dist.iparamSample()[ 0 ]
        return sample

######################################################################

class EMParameters( Parameters ):

    def updateInitialDist( self, msg, node_smoothed ):
        initial_dist = np.zeros_like( msg.pi0 )
        # Update the root distribution
        for root in msg.roots:
            initial_dist += node_smoothed[ root ]
        initial_dist /= msg.roots.size

        self.initial_dist.params = ( initial_dist, )

        assert np.allclose( initial_dist.sum( axis=-1 ), 1.0 )

    def updateTransitionDist( self, msg, parents_smoothed, node_parents_smoothed ):

        trans_dist_numerator = {}
        trans_dist_denominator = {}

        for ndim, val in msg.pis.items():
            trans_dist_numerator[ ndim ] = np.zeros_like( val )
            trans_dist_denominator[ ndim ] = np.zeros( val.shape[ :-1 ] )

        # Update the transition distributions
        for node in filter( lambda n: msg.nParents( n ) > 0, msg.nodes ):
            ndim = msg.nParents( node ) + 1
            trans_dist_numerator[ ndim ] += node_parents_smoothed[ node ]
            trans_dist_denominator[ ndim ] += parents_smoothed[ node ]

        for dist in self.transition_dists:
            ndim = dist.pi.ndim
            dist.params = ( trans_dist_numerator[ ndim ] / trans_dist_denominator[ ndim ][ ..., None ], )
            assert np.allclose( dist.params[ 0 ].sum( axis=-1 ), 1.0 )

    def updateEmissionDist( self, msg, node_smoothed ):

        emission_dist_numerator = np.zeros_like( msg.emission_dist )
        emission_dist_denominator = np.zeros_like( msg.pi0 )

        # Update the emission distribution
        for node, ys in zip( msg.nodes, msg.ys ):
            measurements = ys.shape[ 0 ]

            for y in ys:
                emission_dist_numerator[ :, y ] += node_smoothed[ node ]
            emission_dist_denominator += node_smoothed[ node ] * measurements

        self.emission_dist.params = ( emission_dist_numerator / emission_dist_denominator[ :, None ], )
        assert np.allclose( self.emission_dist.params[ 0 ].sum( axis=-1 ), 1.0 )

######################################################################

class GroupEMParameters( GroupParameters ):

    def updateInitialDist( self, msg, node_smoothed ):

        initial_dists = dict( [ ( group, np.zeros_like( msg.pi0s[ group ] ) ) for group in self.initial_dists.keys() ] )

        # Update the root distribution
        root_counts = dict( [ ( group, 0 ) for group in self.initial_dists.keys() ] )
        for root in msg.roots:
            group = msg.node_groups[ root ]
            initial_dists[ group ] += node_smoothed[ root ]
            root_counts[ group ] += 1

        for group in self.initial_dists.keys():
            if( root_counts[ group ] > 0 ):
                initial_dists[ group ] /= root_counts[ group ]
                self.initial_dists[ group ].params = ( initial_dists[ group ], )
                assert np.allclose( initial_dists[ group ].sum( axis=-1 ), 1.0 )

    def updateTransitionDist( self, msg, parents_smoothed, node_parents_smoothed ):

        trans_dist_numerator = {}
        trans_dist_denominator = {}
        for group, dists in self.transition_dists.items():
            trans_dist_numerator[ group ] = {}
            trans_dist_denominator[ group ] = {}
            for shape, dist in dists.items():
                trans_dist_numerator[ group ][ shape ] = np.zeros( shape )
                trans_dist_denominator[ group ][ shape ] = np.zeros( shape[ :-1 ] )

        # Update the transition distributions
        for node in filter( lambda n: msg.nParents( n ) > 0, msg.nodes ):
            group = msg.node_groups[ node ]
            shape = node_parents_smoothed[ node ].shape

            trans_dist_numerator[ group ][ shape ] += node_parents_smoothed[ node ]
            trans_dist_denominator[ group ][ shape ] += parents_smoothed[ node ]

        for group in self.transition_dists.keys():
            for shape, dist in self.transition_dists[ group ].items():
                dist.params = ( trans_dist_numerator[ group ][ shape ] / trans_dist_denominator[ group ][ shape ][ ..., None ], )
                assert np.allclose( dist.params[ 0 ].sum( axis=-1 ), 1.0 )

    def updateEmissionDist( self, msg, node_smoothed ):

        emission_dist_numerator = dict( [ ( group, np.zeros_like( msg.emission_dists[ group ] ) ) for group in self.emission_dists.keys() ] )
        emission_dist_denominator = dict( [ ( group, np.zeros_like( msg.pi0s[ group ] ) ) for group in self.emission_dists.keys() ] )

        # Update the emission distribution
        for node, ys in zip( msg.nodes, msg.ys ):
            measurements = ys.shape[ 0 ]
            group = msg.node_groups[ node ]

            for y in ys:
                emission_dist_numerator[ group ][ :, y ] += node_smoothed[ node ]
            emission_dist_denominator[ group ] += node_smoothed[ node ] * measurements

        for group in self.emission_dists.keys():
            self.emission_dists[ group ].params = ( emission_dist_numerator[ group ] / emission_dist_denominator[ group ][ :, None ], )
            assert np.allclose( self.emission_dists[ group ].params[ 0 ].sum( axis=-1 ), 1.0 )

######################################################################

class VIParameters( Parameters ):
    # Compute posterior variational natural prior parameters.
    # To do this, just add the expected stats to the intial
    # mean field natural parameters
    def __init__( self, root_prior, transition_priors, emission_prior, minibatch_ratio=1.0 ):
        super().__init__( root_prior, transition_priors, emission_prior )
        self.s = minibatch_ratio

    def updatedInitialPrior( self, msg, node_smoothed ):

        expected_initial_stats = np.zeros_like( msg.pi0 )
        # Update the root distribution
        for root in msg.roots:
            expected_initial_stats += node_smoothed[ root ]

        return ( self.initial_dist.prior.mf_nat_params[ 0 ] + self.s * expected_initial_stats, )

    def updatedTransitionPrior( self, msg, node_parents_smoothed ):

        expected_transition_stats = {}

        for ndim, val in msg.pis.items():
            expected_transition_stats[ ndim ] = np.zeros_like( val )

        # Update the transition distributions
        for node in filter( lambda n: msg.nParents( n ) > 0, msg.nodes ):
            ndim = msg.nParents( node ) + 1
            expected_transition_stats[ ndim ] += node_parents_smoothed[ node ]

        return [ ( dist.prior.mf_nat_params[ 0 ] + self.s * expected_transition_stats[ dist.pi.ndim ], ) for dist in self.transition_dists ]

    def updatedEmissionPrior( self, msg, node_smoothed ):

        expected_emission_stats = np.zeros_like( msg.emission_dist )

        # Update the emission distribution
        for node, ys in zip( msg.nodes, msg.ys ):

            for y in ys:
                expected_emission_stats[ :, y ] += node_smoothed[ node ]

        return ( self.emission_dist.prior.mf_nat_params[ 0 ] + self.s * expected_emission_stats, )

class CAVIParameters( VIParameters ):
    pass

class SVIParameters( VIParameters ):
    def setMinibatchRatio( self, s ):
        self.s = s

######################################################################

class GroupVIParameters( GroupParameters ):
    # Compute posterior variational natural prior parameters.
    # To do this, just add the expected stats to the intial
    # mean field natural parameters
    def __init__( self, root_priors, transition_priors, emission_priors, minibatch_ratio=1.0 ):
        super().__init__( root_priors, transition_priors, emission_priors )
        self.s = minibatch_ratio

    def updatedInitialPrior( self, msg, node_smoothed ):

        expected_initial_stats = dict( [ ( group, np.zeros_like( msg.pi0s[ group ] ) ) for group in self.initial_dists.keys() ] )

        # Update the root distribution
        for root in msg.roots:
            group = msg.node_groups[ root ]
            expected_initial_stats[ group ] += node_smoothed[ root ]

        return dict( [ ( group, ( dist.prior.mf_nat_params[ 0 ] + self.s * expected_initial_stats[ group ], ) ) for group, dist in self.initial_dists.items() ] )

    def updatedTransitionPrior( self, msg, node_parents_smoothed ):

        expected_transition_stats = {}
        for group, dists in self.transition_dists.items():
            expected_transition_stats[ group ] = {}
            for shape, dist in dists.items():
                expected_transition_stats[ group ][ shape ] = np.zeros( shape )

        # Update the transition distributions
        for node in filter( lambda n: msg.nParents( n ) > 0, msg.nodes ):
            group = msg.node_groups[ node ]
            shape = node_parents_smoothed[ node ].shape

            expected_transition_stats[ group ][ shape ] += node_parents_smoothed[ node ]

        ans = {}

        for group in self.transition_dists.keys():
            ans[ group ] = {}
            for shape, dist in self.transition_dists[ group ].items():
                ans[ group ][ shape ] = ( dist.prior.mf_nat_params[ 0 ] + self.s * expected_transition_stats[ group ][ shape ], )

        return ans

    def updatedEmissionPrior( self, msg, node_smoothed ):

        expected_emission_stats = dict( [ ( group, np.zeros_like( msg.emission_dists[ group ] ) ) for group in self.emission_dists.keys() ] )

        # Update the emission distribution
        for node, ys in zip( msg.nodes, msg.ys ):
            group = msg.node_groups[ node ]

            for y in ys:
                expected_emission_stats[ group ][ :, y ] += node_smoothed[ node ]

        return dict( [ ( group, ( dist.prior.mf_nat_params[ 0 ] + self.s * expected_emission_stats[ group ], ) ) for group, dist in self.emission_dists.items() ] )

class GroupCAVIParameters( GroupVIParameters ):
    pass

class GroupSVIParameters( GroupVIParameters ):
    def setMinibatchRatio( self, s ):
        self.s = s

######################################################################

class Optimizer():

    def __init__( self, msg, parameters ):
        self.msg = msg
        self.params = parameters
        self.U = None
        self.V = None

    def loadData( self, graphs ):
        self.msg.preprocessData( graphs )

    def runFilter( self ):
        self.U, self.V = self.msg.filter()

    def fitStep( self ):
        raise NotImplementedError

######################################################################

class GraphSmoothedState():

    def __init__( self, msg, U, V ):
        self.msg = msg
        self.node_states = {}
        self.U = U
        self.V = V

    def __call__( self, node_list ):
        # Compute P( x_c | x_p1..pN, Y )
        vals = self.msg.conditionalParentChild( self.U, self.V, node_list )

        for node, probs in vals:
            parents, parent_order = msg.getParents( node, get_order=True )
            if( len( parents ) == 0 ):
                prob = probs
            else:
                indices = [ [ self.node_states[ o ] ] for o in parent_order ]
                prob = probs[ indices ].ravel()

            # Sample from P( x_c | x_p1..pN, Y )
            state = Categorical.sample( nat_params=( prob, ) )[ 0 ]
            self.node_states[ node ] = state

class Gibbs( Optimizer ):

    def __init__( self, msg, parameters ):
        super().__init__( msg, parameters )
        self.graph_state = None

    def resampleStateHelper( self, node_list ):
        # Compute P( x_c | x_p1..pN, Y )
        vals = self.msg.conditionalParentChild( self.U, self.V, node_list )

        for node, probs in vals:
            parents, parent_order = self.msg.getParents( node, get_order=True )
            if( len( parents ) == 0 ):
                prob = probs
            else:
                indices = tuple( [ [ self.graph_state.node_states[ p ] ] for p in parents ] )
                prob = probs[ indices ].ravel()

            # Sample from P( x_c | x_p1..pN, Y )
            state = Categorical.sample( nat_params=( prob, ) )[ 0 ]
            self.graph_state.node_states[ node ] = state

    def resampleStates( self, return_marginal=False ):
        self.msg.updateParams( self.params.sampleInitialDist(), self.params.sampleTransitionDist(), self.params.sampleEmissionDist() )
        self.runFilter()
        self.graph_state = GraphSmoothedState( self.msg, self.U, self.V )
        self.msg.forwardPass( self.resampleStateHelper )
        if( return_marginal ):
            return self.msg.marginalProb( self.U, self.V )
        return None

    def resampleParameters( self ):
        self.params.resampleInitialDist( self.msg, self.graph_state )
        self.params.resampleTransitionDist( self.msg, self.graph_state )
        self.params.resampleEmissionDist( self.msg, self.graph_state )

    def stateUpdate( self, return_marginal=False ):
        return self.resampleStates( return_marginal=return_marginal )

    def fitStep( self, return_marginal=False ):
        ret_val = self.resampleStates( return_marginal=return_marginal )
        self.resampleParameters()
        return ret_val

    def genProbHelper( self, node_list ):
        # Compute P( X, Y | Θ )
        for node in node_list:
            node_state = self.graph_state.node_states[ node ]

            # P( X | Θ )
            if( self.msg.nParents( node ) == 0 ):
                self.gen_prob += self.params.initial_dist.ilog_likelihood( np.array( [ node_state ] ) )
            else:
                parents, parent_order = self.msg.getParents( node, get_order=True )
                states = tuple( [ np.array( [ self.graph_state.node_states[ p ] ] ) for p in parents ] )
                transition_state = states + ( np.array( [ node_state ] ), )
                self.gen_prob += self.params.transition_dist.ilog_likelihood( transition_state )

            # P( Y | X, Θ )
            emission_state = ( np.array( [ node_state ] ), np.array( [ self.msg.ys[ node ] ] ) )
            self.gen_prob += self.params.emission_dist.ilog_likelihood( emission_state )

    def generativeProbability( self ):
        self.gen_prob = 0.0
        self.msg.forwardPass( self.genProbHelper )
        return self.gen_prob

######################################################################

class GroupGibbs( Gibbs ):

    def genProbHelper( self, node_list ):
        # Compute P( X, Y | Θ )
        for node in node_list:
            node_state = self.graph_state.node_states[ node ]
            group = self.msg.node_groups[ node ]

            # P( X | Θ )
            if( self.msg.nParents( node ) == 0 ):
                self.gen_prob += self.params.initial_dists[ group ].ilog_likelihood( np.array( [ node_state ] ) )
            else:
                parents, parent_order = self.msg.getParents( node, get_order=True )
                states = tuple( [ np.array( [ self.graph_state.node_states[ p ] ] ) for p in parents ] )
                transition_state = states + ( np.array( [ node_state ] ), )
                self.gen_prob += self.params.transition_dists[ group ].ilog_likelihood( transition_state )

            # P( Y | X, Θ )
            emission_state = ( np.array( [ node_state ] ), np.array( [ self.msg.ys[ node ] ] ) )
            self.gen_prob += self.params.emission_dists[ group ].ilog_likelihood( emission_state )

######################################################################

class EM( Optimizer ):

    def EStep( self ):
        self.msg.updateParams( self.params.initial_dist.pi, [ dist.pi for dist in self.params.transition_dists ], self.params.emission_dist.pi )
        self.runFilter()

        marginal = self.msg.marginalProb( self.U, self.V )

        # Compute log P( x | Y ), log P( x_p1..pN | Y ) and log P( x_c, x_p1..pN | Y )
        node_parents_smoothed = self.msg.parentChildSmoothed( self.U, self.V, self.msg.nodes )
        parents_smoothed = self.msg.parentsSmoothed( self.U, self.V, self.msg.nodes, node_parents_smoothed )
        node_smoothed = self.msg.nodeSmoothed( self.U, self.V, self.msg.nodes, node_parents_smoothed )

        # The probabilities are normalized, so don't need them in log space anymore
        node_smoothed = [ ( n, np.exp( val ) ) for n, val in node_smoothed ]
        parents_smoothed = [ ( n, np.exp( val ) ) for n, val in parents_smoothed ]
        node_parents_smoothed = [ ( n, np.exp( val ) ) for n, val in node_parents_smoothed ]

        return dict( node_smoothed ), dict( parents_smoothed ), dict( node_parents_smoothed ), marginal

    def MStep( self, node_smoothed, parents_smoothed, node_parents_smoothed ):

        self.params.updateInitialDist( self.msg, node_smoothed )
        self.params.updateTransitionDist( self.msg, parents_smoothed, node_parents_smoothed )
        self.params.updateEmissionDist( self.msg, node_smoothed )

    def stateUpdate( self ):
        node_smoothed, parents_smoothed, node_parents_smoothed, marginal = self.EStep()
        return node_smoothed, marginal

    def fitStep( self ):
        node_smoothed, parents_smoothed, node_parents_smoothed, marginal = self.EStep()
        self.MStep( node_smoothed, parents_smoothed, node_parents_smoothed )
        return marginal

######################################################################

class GroupEM( EM ):

    def EStep( self ):
        pi0s = dict( [ ( group, dist.pi ) for group, dist in self.params.initial_dists.items() ] )
        pis = dict( [ ( group, [ dist.pi for shape, dist in dists.items() ] ) for group, dists in self.params.transition_dists.items() ] )
        Ls = dict( [ ( group, dist.pi ) for group, dist in self.params.emission_dists.items() ] )
        self.msg.updateParams( pi0s, pis, Ls )
        self.runFilter()

        marginal = self.msg.marginalProb( self.U, self.V )

        # Compute log P( x | Y ), log P( x_p1..pN | Y ) and log P( x_c, x_p1..pN | Y )
        node_smoothed = self.msg.nodeSmoothed( self.U, self.V, self.msg.nodes )
        parents_smoothed = self.msg.parentsSmoothed( self.U, self.V, self.msg.nodes )
        node_parents_smoothed = self.msg.parentChildSmoothed( self.U, self.V, self.msg.nodes )

        # The probabilities are normalized, so don't need them in log space anymore
        node_smoothed = [ ( n, np.exp( val ) ) for n, val in node_smoothed ]
        parents_smoothed = [ ( n, np.exp( val ) ) for n, val in parents_smoothed ]
        node_parents_smoothed = [ ( n, np.exp( val ) ) for n, val in node_parents_smoothed ]

        return dict( node_smoothed ), dict( parents_smoothed ), dict( node_parents_smoothed ), marginal

######################################################################

class CAVI( Optimizer ):
    # Coordinate ascent variational inference

    def __init__( self, msg, parameters, from_super=False ):
        super().__init__( msg, parameters )

        if( from_super == False ):
            # Initialize the expected mf nat params using the prior
            self.initial_prior_mfnp     = self.params.initial_dist.prior.nat_params
            self.transition_prior_mfnps = [ dist.prior.nat_params for dist in self.params.transition_dists ]
            self.emission_prior_mfnp    = self.params.emission_dist.prior.nat_params

    def ELBO( self, initial_prior_mfnp, transition_prior_mfnps, emission_prior_mfnp ):
        normalizer = self.msg.marginalProb( self.U, self.V )

        initial_kl_divergence = Dirichlet.KLDivergence( nat_params1=initial_prior_mfnp, nat_params2=self.params.initial_dist.prior.nat_params )
        transition_kl_divergence = 0
        for mfnp, dist in zip( transition_prior_mfnps, self.params.transition_dists ):
            transition_kl_divergence += TensorTransitionDirichletPrior.KLDivergence( nat_params1=mfnp, nat_params2=dist.prior.nat_params )
        emission_kl_divergence = TensorTransitionDirichletPrior.KLDivergence( nat_params1=emission_prior_mfnp, nat_params2=self.params.emission_dist.prior.nat_params )

        return normalizer - ( initial_kl_divergence + transition_kl_divergence + emission_kl_divergence )

    def variationalEStep( self, initial_prior_mfnp, transition_prior_mfnps, emission_prior_mfnp ):

        # Filter using the expected natural parameters
        expected_initial_nat_params    = self.params.initial_dist.prior.expectedSufficientStats( nat_params=initial_prior_mfnp )[ 0 ]
        expected_transition_nat_params = [ dist.prior.expectedSufficientStats( nat_params=mfnp )[ 0 ] for dist, mfnp in zip( self.params.transition_dists, transition_prior_mfnps ) ]
        expected_emission_nat_params   = self.params.emission_dist.prior.expectedSufficientStats( nat_params=emission_prior_mfnp )[ 0 ]

        self.msg.updateNatParams( expected_initial_nat_params, expected_transition_nat_params, expected_emission_nat_params, check_parameters=False )
        self.runFilter()

        elbo = self.ELBO( initial_prior_mfnp, transition_prior_mfnps, emission_prior_mfnp )

        # Compute log P( x | Y ) and log P( x_c, x_p1..pN | Y )
        node_parents_smoothed = self.msg.parentChildSmoothed( self.U, self.V, self.msg.nodes )
        node_smoothed = self.msg.nodeSmoothed( self.U, self.V, self.msg.nodes, node_parents_smoothed )

        # The probabilities are normalized, so don't need them in log space anymore
        node_smoothed = [ ( n, np.exp( val ) ) for n, val in node_smoothed ]
        node_parents_smoothed = [ ( n, np.exp( val ) ) for n, val in node_parents_smoothed ]

        return dict( node_smoothed ), dict( node_parents_smoothed ), elbo

    def variationalMStep( self, node_smoothed, node_parents_smoothed ):
        initial_prior_mfnp     = self.params.updatedInitialPrior( self.msg, node_smoothed )
        transition_prior_mfnps = self.params.updatedTransitionPrior( self.msg, node_parents_smoothed )
        emission_prior_mfnp    = self.params.updatedEmissionPrior( self.msg, node_smoothed )

        return initial_prior_mfnp, transition_prior_mfnps, emission_prior_mfnp

    def stateUpdate( self ):
        node_smoothed, node_parents_smoothed, elbo = self.variationalEStep( self.initial_prior_mfnp, self.transition_prior_mfnps, self.emission_prior_mfnp )
        return elbo

    def fitStep( self ):
        node_smoothed, node_parents_smoothed, elbo = self.variationalEStep( self.initial_prior_mfnp, self.transition_prior_mfnps, self.emission_prior_mfnp )
        self.initial_prior_mfnp, self.transition_prior_mfnps, self.emission_prior_mfnp = self.variationalMStep( node_smoothed, node_parents_smoothed )
        return elbo

######################################################################

class GroupCAVI( CAVI ):
    # Coordinate ascent variational inference

    def __init__( self, msg, parameters ):
        super().__init__( msg, parameters, from_super=True )

        # Initialize the expected mf nat params using the prior
        self.initial_prior_mfnp    = dict( [ ( group, dist.prior.nat_params ) for group, dist in self.params.initial_dists.items() ] )
        self.transition_prior_mfnps = dict( [ ( group, dict( [ ( shape, dist.prior.nat_params ) for shape, dist in dists.items() ] ) )
                                                                                                for group, dists in self.params.transition_dists.items() ] )
        self.emission_prior_mfnp   = dict( [ ( group, dist.prior.nat_params ) for group, dist in self.params.emission_dists.items() ] )

    def ELBO( self, initial_prior_mfnp, transition_prior_mfnps, emission_prior_mfnp ):
        normalizer = self.msg.marginalProb( self.U, self.V )

        initial_kl_divergence, transition_kl_divergence, emission_kl_divergence = 0, 0, 0

        for group in self.params.initial_dists.keys():
            initial_kl_divergence += Dirichlet.KLDivergence( nat_params1=initial_prior_mfnp[ group ], nat_params2=self.params.initial_dists[ group ].prior.nat_params )
            transition_kl_divergence = 0
            for shape in transition_prior_mfnps[ group ].keys():
                mfnp = transition_prior_mfnps[ group ][ shape ]
                dist = self.params.transition_dists[ group ][ shape ]
                transition_kl_divergence += TensorTransitionDirichletPrior.KLDivergence( nat_params1=mfnp, nat_params2=dist.prior.nat_params )
            emission_kl_divergence   += TensorTransitionDirichletPrior.KLDivergence( nat_params1=emission_prior_mfnp[ group ], nat_params2=self.params.emission_dists[ group ].prior.nat_params )

        return normalizer - ( initial_kl_divergence + transition_kl_divergence + emission_kl_divergence )

    def variationalEStep( self, initial_prior_mfnp, transition_prior_mfnps, emission_prior_mfnp ):

        # Filter using the expected natural parameters
        expected_initial_nat_params    = dict( [ ( group, dist.prior.expectedSufficientStats( nat_params=initial_prior_mfnp[ group ] )[ 0 ] ) for group, dist in self.params.initial_dists.items() ] )
        expected_transition_nat_params = dict( [ ( group, [ dist.prior.expectedSufficientStats( nat_params=transition_prior_mfnps[ group ][ shape ] )[ 0 ] for shape, dist in dists.items() ] )
                                                                                                   for group, dists in self.params.transition_dists.items() ] )
        expected_emission_nat_params   = dict( [ ( group, dist.prior.expectedSufficientStats( nat_params=emission_prior_mfnp[ group ] )[ 0 ] ) for group, dist in self.params.emission_dists.items() ] )

        self.msg.updateNatParams( expected_initial_nat_params, expected_transition_nat_params, expected_emission_nat_params, check_parameters=False )
        self.runFilter()

        elbo = self.ELBO( initial_prior_mfnp, transition_prior_mfnps, emission_prior_mfnp )

        # Compute log P( x | Y ) and log P( x_c, x_p1..pN | Y )
        node_smoothed = self.msg.nodeSmoothed( self.U, self.V, self.msg.nodes )
        node_parents_smoothed = self.msg.parentChildSmoothed( self.U, self.V, self.msg.nodes )

        # The probabilities are normalized, so don't need them in log space anymore
        node_smoothed = [ ( n, np.exp( val ) ) for n, val in node_smoothed ]
        node_parents_smoothed = [ ( n, np.exp( val ) ) for n, val in node_parents_smoothed ]

        return dict( node_smoothed ), dict( node_parents_smoothed ), elbo

######################################################################

class SVI( CAVI ):

    # Stochasic variational inference
    def __init__( self, msg, parameters, minibatch_ratio, step_size ):
        super().__init__( msg, parameters )
        self.s = minibatch_ratio
        self.params.setMinibatchRatio( self.s )
        assert step_size >= 0 and step_size <= 1
        self.p = step_size

    def variationalMStep( self, node_smoothed, node_parents_smoothed ):
        initial_prior_mfnp_update,    = self.params.updatedInitialPrior( self.msg, node_smoothed )
        transition_prior_mfnp_update  = self.params.updatedTransitionPrior( self.msg, node_parents_smoothed )
        emission_prior_mfnp_update,   = self.params.updatedEmissionPrior( self.msg, node_smoothed )

        # Take a natural gradient step
        initial_prior_mfnp     = ( 1 - self.p ) * self.initial_prior_mfnp[ 0 ] + self.p * initial_prior_mfnp_update
        transition_prior_mfnps = [ ( ( 1 - self.p ) * mfnp[ 0 ] + self.p * update[ 0 ], ) for mfnp, update in zip( self.transition_prior_mfnps, transition_prior_mfnp_update ) ]
        emission_prior_mfnp    = ( 1 - self.p ) * self.emission_prior_mfnp[ 0 ] + self.p * emission_prior_mfnp_update
        return ( initial_prior_mfnp, ), transition_prior_mfnps, ( emission_prior_mfnp, )

######################################################################

class GroupSVI( GroupCAVI ):
    # Stochasic variational inference
    def __init__( self, msg, parameters, minibatch_ratio, step_size ):
        super().__init__( msg, parameters )
        self.s = minibatch_ratio
        self.params.setMinibatchRatio( self.s )
        assert step_size >= 0 and step_size <= 1
        self.p = step_size

    def variationalMStep( self, node_smoothed, node_parents_smoothed ):
        initial_prior_mfnp_update    = self.params.updatedInitialPrior( self.msg, node_smoothed )
        transition_prior_mfnp_update = self.params.updatedTransitionPrior( self.msg, node_parents_smoothed )
        emission_prior_mfnp_update   = self.params.updatedEmissionPrior( self.msg, node_smoothed )

        # Take a natural gradient step
        initial_prior_mfnp, transition_prior_mfnps, emission_prior_mfnp = {}, {}, {}
        for group in initial_prior_mfnp_update.keys():

            initial_prior_mfnp[ group ] = ( ( 1 - self.p ) * self.initial_prior_mfnp[ group ][ 0 ] + self.p * initial_prior_mfnp_update[ group ][ 0 ], )

            transition_prior_mfnps[ group ] = {}
            for shape in transition_prior_mfnp_update[ group ].keys():
                update, = transition_prior_mfnp_update[ group ][ shape ]
                mfnp, = self.transition_prior_mfnps[ group ][ shape ]
                transition_prior_mfnps[ group ][ shape ] = ( ( 1 - self.p ) * mfnp + self.p * update, )

            emission_prior_mfnp[ group ] = ( ( 1 - self.p ) * self.emission_prior_mfnp[ group ][ 0 ] + self.p * emission_prior_mfnp_update[ group ][ 0 ], )

        return initial_prior_mfnp, transition_prior_mfnps, emission_prior_mfnp

######################################################################

class GHMM():

    def __init__( self, graphs=None, prior_strength=1.0, method='SVI', priors=None, params=None, **kwargs ):
        assert method in [ 'EM', 'Gibbs', 'CAVI', 'SVI' ]

        if( graphs is not None ):
            self.graphs = graphs

        self.msg = GraphHMMFBSParallel()
        self.method = method

        assert priors is not None
        # Initialize using other priors
        root_prior, trans_priors, emiss_prior = priors

        # Generate the parameters
        if( method == 'EM' ):
            self.params = EMParameters( root_prior, trans_priors, emiss_prior ) if params is None else params
        elif( method == 'Gibbs' ):
            self.params = GibbsParameters( root_prior, trans_priors, emiss_prior ) if params is None else params
        elif( method == 'CAVI' ):
            self.params = CAVIParameters( root_prior, trans_priors, emiss_prior ) if params is None else params
        else:
            self.params = SVIParameters( root_prior, trans_priors, emiss_prior ) if params is None else params

        # Generate the model objects
        if( method == 'EM' ):
            self.model = EM( msg=self.msg, parameters=self.params )
        elif( method == 'Gibbs' ):
            self.model = Gibbs( msg=self.msg, parameters=self.params )
        elif( method == 'CAVI' ):
            self.model = CAVI( msg=self.msg, parameters=self.params )
        else:
            self.step_size = kwargs[ 'step_size' ]
            self.minibatch_size = kwargs[ 'minibatch_size' ]
            if( graphs is not None ):
                self.setData( graphs )

        if( method != 'SVI' ):
            if( graphs is not None ):
                self.msg.preprocessData( self.graphs )

        self.initModel()

    ###########################################

    def initModel( self ):
        initial = self.params.initial_dist.pi
        transition = [ dist.pi for dist in self.params.transition_dists ]
        emission = self.params.emission_dist.pi
        self.msg.updateParams( initial, transition, emission )

    ###########################################

    def setGraphs( self, graphs ):
        self.graphs = graphs
        self.msg.updateGraphs( graphs )

    def setData( self, graphs ):
        self.graphs = graphs

        if( self.method != 'SVI' ):
            self.msg.preprocessData( self.graphs )
        else:
            self.total_nodes = sum( [ len( graph.nodes ) for graph, fbs in self.graphs ] )
            minibatch_ratio = self.minibatch_size / len( self.graphs )
            self.model = SVI( msg=self.msg, parameters=self.params, minibatch_ratio=minibatch_ratio, step_size=self.step_size )

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

    def stateUpdate( self, **kwargs ):

        if( self.method != 'SVI' ):
            return self.model.stateUpdate()

        minibatch_indices = np.random.randint( len( self.graphs ), size=self.minibatch_size )
        minibatch = [ self.graphs[ i ] for i in minibatch_indices ]
        self.msg.preprocessData( minibatch )

        # Compute minibatch ratio
        minibatch_nodes = sum( [ len( graph.nodes ) for graph, fbs in minibatch ] )
        self.params.setMinibatchRatio( self.total_nodes / minibatch_nodes )

        return self.model.stateUpdate()

    ###########################################

    def fitStep( self, **kwargs ):

        if( self.method != 'SVI' ):
            return self.model.fitStep( **kwargs )

        minibatch_indices = np.random.randint( len( self.graphs ), size=self.minibatch_size )
        minibatch = [ self.graphs[ i ] for i in minibatch_indices ]
        self.msg.preprocessData( minibatch )

        # Compute minibatch ratio
        minibatch_nodes = sum( [ len( graph.nodes ) for graph, fbs in minibatch ] )
        self.params.setMinibatchRatio( self.total_nodes / minibatch_nodes )

        return self.model.fitStep()

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
        return self.model.generativeProbability()

    def paramProb( self ):
        return self.params.paramProb()

######################################################################

class GroupGHMM():

    def __init__( self, graphs=None, prior_strength=1.0, method='SVI', priors=None, params=None, **kwargs ):
        assert method in [ 'EM', 'Gibbs', 'CAVI', 'SVI' ]

        if( graphs is not None ):
            self.graphs = graphs

        self.msg = GraphHMMFBSGroupParallel()
        # self.msg = GraphHMMFBSGroup()
        self.method = method

        assert priors is not None

        # Initialize using known priors
        root_priors, trans_priors, emiss_priors = priors

        # Generate the parameters objects
        if( method == 'EM' ):
            self.params = GroupEMParameters( root_priors, trans_priors, emiss_priors ) if params is None else params
        elif( method == 'Gibbs' ):
            self.params = GroupGibbsParameters( root_priors, trans_priors, emiss_priors ) if params is None else params
        elif( method == 'CAVI' ):
            self.params = GroupCAVIParameters( root_priors, trans_priors, emiss_priors ) if params is None else params
        else:
            self.params = GroupSVIParameters( root_priors, trans_priors, emiss_priors ) if params is None else params

        # Generate the model objects
        if( method == 'EM' ):
            self.model = GroupEM( msg=self.msg, parameters=self.params )
        elif( method == 'Gibbs' ):
            self.model = GroupGibbs( msg=self.msg, parameters=self.params )
        elif( method == 'CAVI' ):
            self.model = GroupCAVI( msg=self.msg, parameters=self.params )
        else:
            self.step_size = kwargs[ 'step_size' ]
            self.minibatch_size = kwargs[ 'minibatch_size' ]
            if( graphs is not None ):
                self.setData( graphs )

        if( method != 'SVI' ):
            if( graphs is not None ):
                self.msg.preprocessData( self.graphs )

        self.initModel()

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

    def setGraphs( self, graphs ):
        self.graphs = graphs
        self.msg.updateGraphs( graphs )

    def setData( self, graphs ):
        self.graphs = graphs

        if( self.method != 'SVI' ):
            self.msg.preprocessData( self.graphs )
        else:
            self.total_nodes = sum( [ len( graph.nodes ) for graph, fbs in self.graphs ] )
            minibatch_ratio = self.minibatch_size / len( self.graphs )
            self.model = GroupSVI( msg=self.msg, parameters=self.params, minibatch_ratio=minibatch_ratio, step_size=self.step_size )

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

    def stateUpdate( self, **kwargs ):

        if( self.method != 'SVI' ):
            return self.model.stateUpdate()

        minibatch_indices = np.random.randint( len( self.graphs ), size=self.minibatch_size )
        minibatch = [ self.graphs[ i ] for i in minibatch_indices ]
        self.msg.preprocessData( minibatch )

        # Compute minibatch ratio
        minibatch_nodes = sum( [ len( graph.nodes ) for graph, fbs in minibatch ] )
        self.params.setMinibatchRatio( self.total_nodes / minibatch_nodes )

        return self.model.stateUpdate()

    ###########################################

    def fitStep( self, **kwargs ):

        if( self.method != 'SVI' ):
            return self.model.fitStep( **kwargs )

        minibatch_indices = np.random.randint( len( self.graphs ), size=self.minibatch_size )
        minibatch = [ self.graphs[ i ] for i in minibatch_indices ]
        self.msg.preprocessData( minibatch )

        # Compute minibatch ratio
        minibatch_nodes = sum( [ len( graph.nodes ) for graph, fbs in minibatch ] )
        self.params.setMinibatchRatio( self.total_nodes / minibatch_nodes )

        return self.model.fitStep()

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
        return self.model.generativeProbability()

    def paramProb( self ):
        return self.params.paramProb()
