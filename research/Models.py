from GenModels.GM.Distributions import Categorical, Dirichlet, TensorTransition, TensorTransitionDirichletPrior
from GenModels.research.PedigreeWrappers import PedigreeHMMFilter, PedigreeHMMFilterSexMatters
import numpy as np

__all__ = [
    'autosomalTransitionPrior',
    'autosomalDominantEmissionPrior',
    'autosomalRecessiveEmissionPrior',
    'autosomalTransitionPrior',
    'AutosomalParametersGibbs',
    'AutosomalParametersEM',
    'AutosomalParametersCAVI',
    'Gibbs',
    'EM',
    'CAVI'
]


######################################################################

def autosomalTransitionPrior():
    return np.array([[[1.  , 0.  , 0.  , 0.  ],
                      [0.5 , 0.5 , 0.  , 0.  ],
                      [0.5 , 0.5 , 0.  , 0.  ],
                      [0.  , 1.  , 0.  , 0.  ]],

                     [[0.5 , 0.  , 0.5 , 0.  ],
                      [0.25, 0.25, 0.25, 0.25],
                      [0.25, 0.25, 0.25, 0.25],
                      [0.  , 0.5 , 0.  , 0.5 ]],

                     [[0.5 , 0.  , 0.5 , 0.  ],
                      [0.25, 0.25, 0.25, 0.25],
                      [0.25, 0.25, 0.25, 0.25],
                      [0.  , 0.5 , 0.  , 0.5 ]],

                     [[0.  , 0.  , 1.  , 0.  ],
                      [0.  , 0.  , 0.5 , 0.5 ],
                      [0.  , 0.  , 0.5 , 0.5 ],
                      [0.  , 0.  , 0.  , 1.  ]]]) + 1

def autosomalDominantEmissionPrior():
    # [ AA, Aa, aA, aa ]
    # [ Not affected, affected ]
    return np.array( [ [ 0, 1 ],
                       [ 0, 1 ],
                       [ 0, 1 ],
                       [ 1, 0 ] ] ) + 1

def autosomalRecessiveEmissionPrior():
    # [ AA, Aa, aA, aa ]
    # [ Not affected, affected ]
    return np.array( [ [ 1, 0 ],
                       [ 0, 0 ],
                       [ 0, 0 ],
                       [ 0, 1 ] ] ) + 1

######################################################################

class AutosomalParameters():

    def __init__( self, transition_prior, emission_prior ):
        assert transition_prior.shape == ( 4, 4, 4 )
        assert emission_prior.shape == ( 4, 2 )

        # Initial dist
        self.initial_dist = Categorical( hypers=dict( alpha=np.ones( 4 ) ) )

        # Create the transition distribution
        self.transition_dist = TensorTransition( hypers=dict( alpha=transition_prior ) )

        # Emission dist
        self.emission_dist = TensorTransition( hypers=dict( alpha=emission_prior ) )

######################################################################

class TransitionBins():
    def __init__( self, msg, graph_state ):
        self.msg = msg
        self.graph_state = graph_state
        self.counts = [ [] for _ in range( 3 ) ]

    def __call__( self, node_list ):
        for node in filter( lambda n: self.msg.nParents( n ) > 0, node_list ):
            parents, order = self.msg.getParents( node, get_order=True )
            for i, p in zip( order, parents ):
                self.counts[ i ].append( self.graph_state.node_states[ p ] )
            self.counts[ -1 ].append( self.graph_state.node_states[ node ] )

class AutosomalParametersGibbs( AutosomalParameters ):

    def resampleInitialDist( self, msg, graph_state ):
        root_states = np.array( [ state for node, state in graph_state.node_states.items() if msg.nParents( node ) == 0 ] )
        self.initial_dist.resample( x=root_states )

    def resampleTransitionDist( self, msg, graph_state ):
        transition_bins = TransitionBins( msg, graph_state )
        msg.forwardPass( transition_bins )
        x = [ np.array( count ) for count in transition_bins.counts ]
        self.transition_dist.resample( x )

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
        return self.initial_dist.iparamSample()[ 0 ]

    def sampleTransitionDist( self ):
        return [ self.transition_dist.iparamSample()[ 0 ] ]

    def sampleEmissionDist( self ):
        return self.emission_dist.iparamSample()[ 0 ]

######################################################################

class AutosomalParametersEM( AutosomalParameters ):

    def updateInitialDist( self, msg, node_smoothed ):
        initial_dist = np.zeros_like( msg.pi0 )
        # Update the root distribution
        for root in msg.roots:
            initial_dist += node_smoothed[ root ]
        initial_dist /= msg.roots.size

        self.initial_dist.params = ( initial_dist, )

        assert np.allclose( initial_dist.sum( axis=-1 ), 1.0 )

    def updateTransitionDist( self, msg, parents_smoothed, node_parents_smoothed ):

        trans_dist_numerator = np.zeros( ( 4, 4, 4 ) )
        trans_dist_denominator = np.zeros( ( 4, 4 ) )

        # Update the transition distributions
        for node in msg.nodes:
            n_parents = msg.nParents( node )
            if( n_parents == 0 ):
                continue

            trans_dist_numerator += node_parents_smoothed[ node ]
            trans_dist_denominator += parents_smoothed[ node ]

        self.transition_dist.params = ( trans_dist_numerator / trans_dist_denominator[ ..., None ] , )

        assert np.allclose( ( trans_dist_numerator / trans_dist_denominator[ ..., None ] ).sum( axis=-1 ), 1.0 )

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

        assert np.allclose( ( emission_dist_numerator / emission_dist_denominator[ :, None ] ).sum( axis=-1 ), 1.0 )

######################################################################

class AutosomalParametersCAVI( AutosomalParameters ):
    # Compute posterior variational natural prior parameters.
    # To do this, just add the expected stats to the intial
    # mean field natural parameters

    def updatedInitialPrior( self, msg, node_smoothed ):


        expected_initial_stats = np.zeros_like( msg.pi0 )
        # Update the root distribution
        for root in msg.roots:
            expected_initial_stats += node_smoothed[ root ]

        return ( self.initial_dist.prior.mf_nat_params[ 0 ] + expected_initial_stats, )

    def updatedTransitionPrior( self, msg, node_parents_smoothed ):

        expected_transition_stats = np.zeros( ( 4, 4, 4 ) )

        # Update the transition distributions
        for node in msg.nodes:
            n_parents = msg.nParents( node )
            if( n_parents == 0 ):
                continue

            expected_transition_stats += node_parents_smoothed[ node ]

        return ( self.transition_dist.prior.mf_nat_params[ 0 ] + expected_transition_stats, )

    def updatedEmissionPrior( self, msg, node_smoothed ):

        expected_emission_stats = np.zeros_like( msg.emission_dist )

        # Update the emission distribution
        for node, ys in zip( msg.nodes, msg.ys ):

            for y in ys:
                expected_emission_stats[ :, y ] += node_smoothed[ node ]

        return ( self.emission_dist.prior.mf_nat_params[ 0 ] + expected_emission_stats, )

######################################################################

# class XLinkedParameters():

#     def __init__( self, transition_priors, emission_priors ):
#         # Initial dist
#         self.initial_dists = { 0: Categorical( hypers=dict( alpha=np.ones( 4 ) ) ),
#                                1: Categorical( hypers=dict( alpha=np.ones( 4 ) ) ),
#                                2: Categorical( hypers=dict( alpha=np.ones( 4 ) ) ) }

#         # Create the transition distribution
#         self.transition_dists = { 0: TensorTransition( hypers=dict( alpha=transition_priors[ 0 ] ) ),
#                                   1: TensorTransition( hypers=dict( alpha=transition_priors[ 1 ] ) ),
#                                   2: TensorTransition( hypers=dict( alpha=transition_priors[ 2 ] ) ) }

#         # Emission dist
#         self.emission_dists = { 0: TensorTransition( hypers=dict( alpha=emission_priors[ 0 ] ) ),
#                                 1: TensorTransition( hypers=dict( alpha=emission_priors[ 1 ] ) ),
#                                 2: TensorTransition( hypers=dict( alpha=emission_priors[ 2 ] ) ) }

#     def sampleInitialDist( self ):
#         sample = {}
#         for group, dist in self.initial_dists.items():
#             sample[ group ] = dist.sample()[ 0 ]
#         return sample

#     def sampleTransitionDist( self ):
#         sample = {}
#         for group, dist in self.transition_dists.items():
#             sample[ group ] = [ dist.sample()[ 0 ] ]
#         return sample

#     def sampleEmissionDist( self ):
#         sample = {}
#         for group, dist in self.emission_dists.items():
#             sample[ group ] = dist.sample()[ 0 ]
#         return sample

# ######################################################################

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
                indices = [ [ self.graph_state.node_states[ p ] ] for p in parents ]
                prob = probs[ indices ].ravel()

            # Sample from P( x_c | x_p1..pN, Y )
            state = Categorical.sample( nat_params=( prob, ) )[ 0 ]
            self.graph_state.node_states[ node ] = state

    def resampleStates( self ):
        self.msg.updateParams( self.params.sampleInitialDist(), self.params.sampleTransitionDist(), self.params.sampleEmissionDist() )
        self.runFilter()
        self.graph_state = GraphSmoothedState( self.msg, self.U, self.V )
        self.msg.forwardPass( self.resampleStateHelper )

    def resampleParameters( self ):
        self.params.resampleInitialDist( self.msg, self.graph_state )
        self.params.resampleTransitionDist( self.msg, self.graph_state )
        self.params.resampleEmissionDist( self.msg, self.graph_state )

    def fitStep( self ):
        self.resampleStates()
        self.resampleParameters()

######################################################################

class EM( Optimizer ):

    def EStep( self ):
        self.msg.updateParams( self.params.initial_dist.pi, [ self.params.transition_dist.pi ], self.params.emission_dist.pi )
        self.runFilter()

        marginal = self.msg.marginalProb( self.U, self.V, 0 )

        # Compute log P( x | Y ), log P( x_p1..pN | Y ) and log P( x_c, x_p1..pN | Y )
        node_smoothed = self.msg.nodeSmoothed( self.U, self.V, self.msg.nodes )
        parents_smoothed = self.msg.parentsSmoothed( self.U, self.V, self.msg.nodes )
        node_parents_smoothed = self.msg.parentChildSmoothed( self.U, self.V, self.msg.nodes )

        # The probabilities are normalized, so don't need them in log space anymore
        node_smoothed = [ ( n, np.exp( val ) ) for n, val in node_smoothed ]
        parents_smoothed = [ ( n, np.exp( val ) ) for n, val in parents_smoothed ]
        node_parents_smoothed = [ ( n, np.exp( val ) ) for n, val in node_parents_smoothed ]

        return dict( node_smoothed ), dict( parents_smoothed ), dict( node_parents_smoothed ), marginal

    def MStep( self, node_smoothed, parents_smoothed, node_parents_smoothed ):

        self.params.updateInitialDist( self.msg, node_smoothed )
        self.params.updateTransitionDist( self.msg, parents_smoothed, node_parents_smoothed )
        self.params.updateEmissionDist( self.msg, node_smoothed )

    def fitStep( self ):
        node_smoothed, parents_smoothed, node_parents_smoothed, marginal = self.EStep()
        self.MStep( node_smoothed, parents_smoothed, node_parents_smoothed )
        return marginal

######################################################################

class CAVI( Optimizer ):
    # Coordinate ascent variational inference

    def __init__( self, msg, parameters ):
        super().__init__( msg, parameters )

        # Initialize the expected mf nat params using the prior
        self.initial_prior_mfnp    = self.params.initial_dist.prior.nat_params
        self.transition_prior_mfnp = self.params.transition_dist.prior.nat_params
        self.emission_prior_mfnp   = self.params.emission_dist.prior.nat_params

    def ELBO( self, initial_prior_mfnp, transition_prior_mfnp, emission_prior_mfnp ):
        normalizer = self.msg.marginalProb( self.U, self.V, 0 )

        initial_kl_divergence = Dirichlet.KLDivergence( nat_params1=initial_prior_mfnp, nat_params2=self.params.initial_dist.prior.nat_params )
        transition_kl_divergence = TensorTransitionDirichletPrior.KLDivergence( nat_params1=transition_prior_mfnp, nat_params2=self.params.transition_dist.prior.nat_params )
        emission_kl_divergence = TensorTransitionDirichletPrior.KLDivergence( nat_params1=emission_prior_mfnp, nat_params2=self.params.emission_dist.prior.nat_params )

        # print( 'normalizer', normalizer, 'kl', initial_kl_divergence + transition_kl_divergence + emission_kl_divergence )
        return normalizer - ( initial_kl_divergence + transition_kl_divergence + emission_kl_divergence )

    def variationalEStep( self, initial_prior_mfnp, transition_prior_mfnp, emission_prior_mfnp ):

        # Filter using the expected natural parameters
        expected_initial_nat_params    = self.params.initial_dist.prior.expectedSufficientStats( nat_params=initial_prior_mfnp )[ 0 ]
        expected_transition_nat_params = self.params.transition_dist.prior.expectedSufficientStats( nat_params=transition_prior_mfnp )[ 0 ]
        expected_emission_nat_params   = self.params.emission_dist.prior.expectedSufficientStats( nat_params=emission_prior_mfnp )[ 0 ]

        self.msg.updateNatParams( expected_initial_nat_params, [ expected_transition_nat_params ], expected_emission_nat_params, check_parameters=False )
        self.runFilter()

        elbo = self.ELBO( initial_prior_mfnp, transition_prior_mfnp, emission_prior_mfnp )

        # Compute log P( x | Y ) and log P( x_c, x_p1..pN | Y )
        node_smoothed = self.msg.nodeSmoothed( self.U, self.V, self.msg.nodes )
        node_parents_smoothed = self.msg.parentChildSmoothed( self.U, self.V, self.msg.nodes )

        # The probabilities are normalized, so don't need them in log space anymore
        node_smoothed = [ ( n, np.exp( val ) ) for n, val in node_smoothed ]
        node_parents_smoothed = [ ( n, np.exp( val ) ) for n, val in node_parents_smoothed ]

        return dict( node_smoothed ), dict( node_parents_smoothed ), elbo

    def variationalMStep( self, node_smoothed, node_parents_smoothed ):
        initial_prior_mfnp    = self.params.updatedInitialPrior( self.msg, node_smoothed )
        transition_prior_mfnp = self.params.updatedTransitionPrior( self.msg, node_parents_smoothed )
        emission_prior_mfnp   = self.params.updatedEmissionPrior( self.msg, node_smoothed )

        return initial_prior_mfnp, transition_prior_mfnp, emission_prior_mfnp

    def fitStep( self ):
        node_smoothed, node_parents_smoothed, elbo   = self.variationalEStep( self.initial_prior_mfnp, self.transition_prior_mfnp, self.emission_prior_mfnp )
        self.initial_prior_mfnp, self.transition_prior_mfnp, self.emission_prior_mfnp = self.variationalMStep( node_smoothed, node_parents_smoothed )
        return elbo

######################################################################
