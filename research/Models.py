from GenModels.GM.Distributions import Categorical, Dirichlet, TensorTransition, TensorTransitionDirichletPrior
from GenModels.research.PedigreeWrappers import PedigreeHMMFilter, PedigreeHMMFilterSexMatters
import numpy as np

__all__ = [
    'autosomalTransitionPrior',
    'autosomalDominantEmissionPrior',
    'autosomalRecessiveEmissionPrior',
    'xLinkedFemaleTransitionPrior',
    'xLinkedMaleTransitionPrior',
    'xLinkedUnknownTransitionPrior',
    'xLinkedFemaleEmissionPrior',
    'xLinkedMaleEmissionPrior',
    'xLinkedUnknownEmissionPrior',
    'AutosomalParametersGibbs',
    'XLinkedParametersGibbs',
    'AutosomalParametersEM',
    'XLinkedParametersEM',
    'AutosomalParametersCAVI',
    'XLinkedParametersCAVI',
    'AutosomalParametersSVI',
    'XLinkedParametersSVI',
    'Gibbs',
    'EM',
    'GroupEM',
    'CAVI',
    'GroupCAVI',
    'SVI',
    'GroupSVI',
    'AutosomalDominant',
    'AutosomalRecessive',
    'XLinkedRecessive',
    'autosomalDominantPriors',
    'autosomalRecessivePriors',
    'xLinkedRecessivePriors'
]

######################################################################

def autosomalTransitionPrior():
    # [ AA, Aa, aA, aa ] ( A is affected )
    # [ AA, Aa, aa ] ( A is affected )
    with_combo = np.array( [ [ [ 1.  , 0.  , 0.  , 0.   ],
                               [ 0.5 , 0.5 , 0.  , 0.   ],
                               [ 0.5 , 0.5 , 0.  , 0.   ],
                               [ 0.  , 1.  , 0.  , 0.   ] ],

                             [ [ 0.5 , 0.  , 0.5 , 0.   ],
                               [ 0.25, 0.25, 0.25, 0.25 ],
                               [ 0.25, 0.25, 0.25, 0.25 ],
                               [ 0.  , 0.5 , 0.  , 0.5  ] ],

                             [ [ 0.5 , 0.  , 0.5 , 0.   ],
                               [ 0.25, 0.25, 0.25, 0.25 ],
                               [ 0.25, 0.25, 0.25, 0.25 ],
                               [ 0.  , 0.5 , 0.  , 0.5  ] ],

                             [ [ 0.  , 0.  , 1.  , 0.   ],
                               [ 0.  , 0.  , 0.5 , 0.5  ],
                               [ 0.  , 0.  , 0.5 , 0.5  ],
                               [ 0.  , 0.  , 0.  , 1.   ] ] ] )

    without_combo = np.zeros_like( with_combo[ ..., [ 0, 1, 3 ] ] )
    without_combo[ ..., [ 0, 2 ] ] = with_combo[ ..., [ 0, 3 ] ]
    without_combo[ ..., [ 1 ] ] = with_combo[ ..., [ 1 ] ]  + with_combo[ ..., [ 2 ] ]

    without_combo = np.delete( without_combo, 1, axis=0 )
    without_combo = np.delete( without_combo, 1, axis=1 )

    return without_combo

def autosomalDominantEmissionPrior():
    # [ AA, Aa, aa ]
    # [ Not affected, affected ]
    return np.array( [ [ 0, 1 ],
                       [ 0, 1 ],
                       [ 1, 0 ] ] )

def autosomalRecessiveEmissionPrior():
    # [ AA, Aa, aa ]
    # [ Not affected, affected ]
    return np.array( [ [ 0, 1 ],
                       [ 1, 0 ],
                       [ 1, 0 ] ] )

######################################################################

def xLinkedFemaleTransitionPrior():
    # Female, male, female child
    # [ XX, Xx, xX, xx ] ( Index using males [ XY, xY ] )
    # [ XX, Xx, xx ] ( Index using males [ XY, xY ] )
    with_combo = np.array( [ [ [ 1. , 0. , 0. , 0.  ],
                               [ 0. , 0. , 1. , 0.  ] ],

                             [ [ 0.5, 0.5, 0. , 0.  ],
                               [ 0. , 0. , 0.5, 0.5 ] ],

                             [ [ 0.5, 0.5, 0. , 0.  ],
                               [ 0. , 0. , 0.5, 0.5 ] ],

                             [ [ 0. , 1. , 0. , 0.  ],
                               [ 0. , 0. , 0. , 1.  ] ] ] )

    without_combo = np.zeros_like( with_combo[ ..., [ 0, 1, 3 ] ] )
    without_combo[ ..., [ 0, 2 ] ] = with_combo[ ..., [ 0, 3 ] ]
    without_combo[ ..., [ 1 ] ] = with_combo[ ..., [ 1 ] ]  + with_combo[ ..., [ 2 ] ]

    without_combo = np.delete( without_combo, 1, axis=0 )

    return without_combo

def xLinkedMaleTransitionPrior():
    # Female, male, male child
    # [ XY, xY ] ( X is affected )
    return np.array( [ [ [ 1. , 0.  ],
                         [ 1. , 0.  ] ],

                       [ [ 0.5, 0.5 ],
                         [ 0.5, 0.5 ] ],

                       [ [ 0. , 1.  ],
                         [ 0. , 1.  ] ] ] )

def xLinkedUnknownTransitionPrior():
    # Female, male, unknown sex child
    # [ XX, Xx, xX, xx, XY, xY ]
    # [ XX, Xx, xx, XY, xY ]
    with_combo = np.array( [ [ [ 0.5 , 0.  , 0.  , 0.  , 0.5 , 0.   ] ,
                               [ 0.  , 0.  , 0.5 , 0.  , 0.5 , 0.   ] ] ,

                             [ [ 0.25, 0.25, 0.  , 0.  , 0.25, 0.25 ] ,
                               [ 0.  , 0.  , 0.25, 0.25, 0.25, 0.25 ] ] ,

                             [ [ 0.25, 0.25, 0.  , 0.  , 0.25, 0.25 ] ,
                               [ 0.  , 0.  , 0.25, 0.25, 0.25, 0.25 ] ] ,

                             [ [ 0.  , 0.5 , 0.  , 0.  , 0.  , 0.5  ] ,
                               [ 0.  , 0.  , 0.  , 0.5 , 0.  , 0.5  ] ] ] )

    without_combo = np.zeros_like( with_combo[ ..., [ 0, 1, 3, 4, 5 ] ] )
    without_combo[ ..., [ 0, 2, 3, 4 ] ] = with_combo[ ..., [ 0, 3, 4, 5 ] ]
    without_combo[ ..., [ 1 ] ] = with_combo[ ..., [ 1 ] ]  + with_combo[ ..., [ 2 ] ]

    without_combo = np.delete( without_combo, 1, axis=0 )


    return without_combo

# Going to ignore the ( unknown, unknown ) -> unknown case

def xLinkedFemaleEmissionPrior():
    # [ XX, Xx, xx ]
    # [ Not affected, affected ]
    return np.array( [ [ 0, 1 ],
                       [ 1, 0 ],
                       [ 1, 0 ] ] )

def xLinkedMaleEmissionPrior():
    # [ XY, xY ]
    # [ Not affected, affected ]
    return np.array( [ [ 0, 1 ],
                       [ 1, 0 ] ] )

def xLinkedUnknownEmissionPrior():
    # [ XX, Xx, xx, XY, xY ]
    # [ Not affected, affected ]
    return np.array( [ [ 0, 1 ],
                       [ 1, 0 ],
                       [ 1, 0 ],
                       [ 0, 1 ],
                       [ 1, 0 ] ] )

######################################################################

class AutosomalParameters():

    def __init__( self, transition_prior, emission_prior ):
        assert transition_prior.shape == ( 3, 3, 3 )
        assert emission_prior.shape == ( 3, 2 )

        # Initial dist
        self.initial_dist = Categorical( hypers=dict( alpha=np.ones( 3 ) ) )

        # Create the transition distribution
        self.transition_dist = TensorTransition( hypers=dict( alpha=transition_prior ) )

        # Emission dist
        self.emission_dist = TensorTransition( hypers=dict( alpha=emission_prior ) )

    def paramProb( self ):
        return self.initial_dist.ilog_params() + self.transition_dist.ilog_params() + self.emission_dist.ilog_params()

######################################################################

class XLinkedParameters():

    def __init__( self, transition_priors, emission_priors ):
        # Initial dist
        self.initial_dists = { 0: Categorical( hypers=dict( alpha=np.ones( 3 ) ) ),
                               1: Categorical( hypers=dict( alpha=np.ones( 2 ) ) ),
                               2: Categorical( hypers=dict( alpha=np.ones( 5 ) ) ) }

        # Create the transition distribution
        self.transition_dists = { 0: TensorTransition( hypers=dict( alpha=transition_priors[ 0 ] ) ),
                                  1: TensorTransition( hypers=dict( alpha=transition_priors[ 1 ] ) ),
                                  2: TensorTransition( hypers=dict( alpha=transition_priors[ 2 ] ) ) }

        # Emission dist
        self.emission_dists = { 0: TensorTransition( hypers=dict( alpha=emission_priors[ 0 ] ) ),
                                1: TensorTransition( hypers=dict( alpha=emission_priors[ 1 ] ) ),
                                2: TensorTransition( hypers=dict( alpha=emission_priors[ 2 ] ) ) }

    def paramProb( self ):
        ans = 0.0
        for group in self.initial_dists.keys():
            ans += self.initial_dists[ group ].ilog_params()
            ans += self.transition_dists[ group ].ilog_params()
            ans += self.emission_dists[ group ].ilog_params()
        return ans

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

class GroupTransitionBins():
    def __init__( self, msg, graph_state, groups ):
        self.msg = msg
        self.graph_state = graph_state
        self.counts = dict( [ ( group, [ [] for _ in range( 3 ) ] ) for group in groups ] )

    def __call__( self, node_list ):
        for node in filter( lambda n: self.msg.nParents( n ) > 0, node_list ):
            group = self.msg.node_groups[ node ]
            parents, order = self.msg.getParents( node, get_order=True )
            for i, p in zip( order, parents ):
                self.counts[ group ][ i ].append( self.graph_state.node_states[ p ] )
            self.counts[ group ][ -1 ].append( self.graph_state.node_states[ node ] )

class XLinkedParametersGibbs( XLinkedParameters ):

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

        for group, counts in transition_bins.counts.items():
            x = [ np.array( count ) for count in counts ]
            self.transition_dists[ group ].resample( x )

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
        for group, dist in self.transition_dists.items():
            sample[ group ] = [ dist.iparamSample()[ 0 ] ]
        return sample

    def sampleEmissionDist( self ):
        sample = {}
        for group, dist in self.emission_dists.items():
            sample[ group ] = dist.iparamSample()[ 0 ]
        return sample

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

        trans_dist_numerator = np.zeros( ( 3, 3, 3 ) )
        trans_dist_denominator = np.zeros( ( 3, 3 ) )

        # Update the transition distributions
        for node in filter( lambda n: msg.nParents( n ) > 0, msg.nodes ):
            trans_dist_numerator += node_parents_smoothed[ node ]
            trans_dist_denominator += parents_smoothed[ node ]

        self.transition_dist.params = ( trans_dist_numerator / trans_dist_denominator[ ..., None ], )
        assert np.allclose( self.transition_dist.params[ 0 ].sum( axis=-1 ), 1.0 )

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

class XLinkedParametersEM( XLinkedParameters ):

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

        trans_dist_numerator = dict( [ ( group, np.zeros_like( msg.pis[ group ][ 3 ] ) ) for group in self.transition_dists.keys() ] )
        trans_dist_denominator = dict( [ ( group, np.zeros( msg.pis[ group ][ 3 ].shape[ :-1 ] ) ) for group in self.transition_dists.keys() ] )

        # Update the transition distributions
        for node in filter( lambda n: msg.nParents( n ) > 0, msg.nodes ):
            group = msg.node_groups[ node ]
            trans_dist_numerator[ group ] += node_parents_smoothed[ node ]
            trans_dist_denominator[ group ] += parents_smoothed[ node ]

        for group in self.transition_dists.keys():
            self.transition_dists[ group ].params = ( trans_dist_numerator[ group ] / trans_dist_denominator[ group ][ ..., None ], )
            assert np.allclose( self.transition_dists[ group ].params[ 0 ].sum( axis=-1 ), 1.0 )

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

class AutosomalParametersVI( AutosomalParameters ):
    # Compute posterior variational natural prior parameters.
    # To do this, just add the expected stats to the intial
    # mean field natural parameters
    def __init__( self, transition_prior, emission_prior, minibatch_ratio=1.0 ):
        super().__init__( transition_prior, emission_prior )
        self.s = minibatch_ratio

    def updatedInitialPrior( self, msg, node_smoothed ):

        expected_initial_stats = np.zeros_like( msg.pi0 )
        # Update the root distribution
        for root in msg.roots:
            expected_initial_stats += node_smoothed[ root ]

        return ( self.initial_dist.prior.mf_nat_params[ 0 ] + self.s * expected_initial_stats, )

    def updatedTransitionPrior( self, msg, node_parents_smoothed ):

        expected_transition_stats = np.zeros( ( 3, 3, 3 ) )

        # Update the transition distributions
        for node in msg.nodes:
            n_parents = msg.nParents( node )
            if( n_parents == 0 ):
                continue

            expected_transition_stats += node_parents_smoothed[ node ]

        return ( self.transition_dist.prior.mf_nat_params[ 0 ] + self.s * expected_transition_stats, )

    def updatedEmissionPrior( self, msg, node_smoothed ):

        expected_emission_stats = np.zeros_like( msg.emission_dist )

        # Update the emission distribution
        for node, ys in zip( msg.nodes, msg.ys ):

            for y in ys:
                expected_emission_stats[ :, y ] += node_smoothed[ node ]

        return ( self.emission_dist.prior.mf_nat_params[ 0 ] + self.s * expected_emission_stats, )

class AutosomalParametersCAVI( AutosomalParametersVI ):
    pass

class AutosomalParametersSVI( AutosomalParametersVI ):
    def setMinibatchRatio( self, s ):
        self.s = s

######################################################################

class XLinkedParametersVI( XLinkedParameters ):
    # Compute posterior variational natural prior parameters.
    # To do this, just add the expected stats to the intial
    # mean field natural parameters
    def __init__( self, transition_priors, emission_priors, minibatch_ratio=1.0 ):
        super().__init__( transition_priors, emission_priors )
        self.s = minibatch_ratio

    def updatedInitialPrior( self, msg, node_smoothed ):

        expected_initial_stats = dict( [ ( group, np.zeros_like( msg.pi0s[ group ] ) ) for group in self.initial_dists.keys() ] )

        # Update the root distribution
        for root in msg.roots:
            group = msg.node_groups[ root ]
            expected_initial_stats[ group ] += node_smoothed[ root ]

        return dict( [ ( group, ( dist.prior.mf_nat_params[ 0 ] + self.s * expected_initial_stats[ group ], ) ) for group, dist in self.initial_dists.items() ] )

    def updatedTransitionPrior( self, msg, node_parents_smoothed ):

        expected_transition_stats = dict( [ ( group, np.zeros_like( msg.pis[ group ][ 3 ] ) ) for group in self.transition_dists.keys() ] )

        # Update the transition distributions
        for node in filter( lambda n: msg.nParents( n ) > 0, msg.nodes ):
            group = msg.node_groups[ node ]

            expected_transition_stats[ group ] += node_parents_smoothed[ node ]

        return dict( [ ( group, ( dist.prior.mf_nat_params[ 0 ] + self.s * expected_transition_stats[ group ], ) ) for group, dist in self.transition_dists.items() ] )

    def updatedEmissionPrior( self, msg, node_smoothed ):

        expected_emission_stats = dict( [ ( group, np.zeros_like( msg.emission_dists[ group ] ) ) for group in self.emission_dists.keys() ] )

        # Update the emission distribution
        for node, ys in zip( msg.nodes, msg.ys ):
            group = msg.node_groups[ node ]

            for y in ys:
                expected_emission_stats[ group ][ :, y ] += node_smoothed[ node ]

        return dict( [ ( group, ( dist.prior.mf_nat_params[ 0 ] + self.s * expected_emission_stats[ group ], ) ) for group, dist in self.emission_dists.items() ] )

class XLinkedParametersCAVI( XLinkedParametersVI ):
    pass

class XLinkedParametersSVI( XLinkedParametersVI ):
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

    def resampleStates( self ):
        self.msg.updateParams( self.params.sampleInitialDist(), self.params.sampleTransitionDist(), self.params.sampleEmissionDist() )
        self.runFilter()
        self.graph_state = GraphSmoothedState( self.msg, self.U, self.V )
        self.msg.forwardPass( self.resampleStateHelper )

    def resampleParameters( self ):
        self.params.resampleInitialDist( self.msg, self.graph_state )
        self.params.resampleTransitionDist( self.msg, self.graph_state )
        self.params.resampleEmissionDist( self.msg, self.graph_state )

    def stateUpdate( self ):
        self.resampleStates()

    def fitStep( self ):
        self.resampleStates()
        self.resampleParameters()

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
        pis = dict( [ ( group, [ dist.pi ] ) for group, dist in self.params.transition_dists.items() ] )
        Ls = dict( [ ( group, dist.pi ) for group, dist in self.params.emission_dists.items() ] )
        self.msg.updateParams( pi0s, pis, Ls )
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

######################################################################

class CAVI( Optimizer ):
    # Coordinate ascent variational inference

    def __init__( self, msg, parameters, from_super=False ):
        super().__init__( msg, parameters )

        if( from_super == False ):
            # Initialize the expected mf nat params using the prior
            self.initial_prior_mfnp    = self.params.initial_dist.prior.nat_params
            self.transition_prior_mfnp = self.params.transition_dist.prior.nat_params
            self.emission_prior_mfnp   = self.params.emission_dist.prior.nat_params

    def ELBO( self, initial_prior_mfnp, transition_prior_mfnp, emission_prior_mfnp ):
        normalizer = self.msg.marginalProb( self.U, self.V, 0 )

        initial_kl_divergence = Dirichlet.KLDivergence( nat_params1=initial_prior_mfnp, nat_params2=self.params.initial_dist.prior.nat_params )
        transition_kl_divergence = TensorTransitionDirichletPrior.KLDivergence( nat_params1=transition_prior_mfnp, nat_params2=self.params.transition_dist.prior.nat_params )
        emission_kl_divergence = TensorTransitionDirichletPrior.KLDivergence( nat_params1=emission_prior_mfnp, nat_params2=self.params.emission_dist.prior.nat_params )

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

    def stateUpdate( self ):
        node_smoothed, node_parents_smoothed, elbo = self.variationalEStep( self.initial_prior_mfnp, self.transition_prior_mfnp, self.emission_prior_mfnp )
        return elbo

    def fitStep( self ):
        node_smoothed, node_parents_smoothed, elbo = self.variationalEStep( self.initial_prior_mfnp, self.transition_prior_mfnp, self.emission_prior_mfnp )
        self.initial_prior_mfnp, self.transition_prior_mfnp, self.emission_prior_mfnp = self.variationalMStep( node_smoothed, node_parents_smoothed )
        return elbo

######################################################################

class GroupCAVI( CAVI ):
    # Coordinate ascent variational inference

    def __init__( self, msg, parameters ):
        super().__init__( msg, parameters, from_super=True )

        # Initialize the expected mf nat params using the prior
        self.initial_prior_mfnp    = dict( [ ( group, dist.prior.nat_params ) for group, dist in self.params.initial_dists.items() ] )
        self.transition_prior_mfnp = dict( [ ( group, dist.prior.nat_params ) for group, dist in self.params.transition_dists.items() ] )
        self.emission_prior_mfnp   = dict( [ ( group, dist.prior.nat_params ) for group, dist in self.params.emission_dists.items() ] )

    def ELBO( self, initial_prior_mfnp, transition_prior_mfnp, emission_prior_mfnp ):
        normalizer = self.msg.marginalProb( self.U, self.V, 0 )

        initial_kl_divergence, transition_kl_divergence, emission_kl_divergence = 0, 0, 0

        for group in self.params.initial_dists.keys():
            initial_kl_divergence    += Dirichlet.KLDivergence( nat_params1=initial_prior_mfnp[ group ], nat_params2=self.params.initial_dists[ group ].prior.nat_params )
            transition_kl_divergence += TensorTransitionDirichletPrior.KLDivergence( nat_params1=transition_prior_mfnp[ group ], nat_params2=self.params.transition_dists[ group ].prior.nat_params )
            emission_kl_divergence   += TensorTransitionDirichletPrior.KLDivergence( nat_params1=emission_prior_mfnp[ group ], nat_params2=self.params.emission_dists[ group ].prior.nat_params )

        return normalizer - ( initial_kl_divergence + transition_kl_divergence + emission_kl_divergence )

    def variationalEStep( self, initial_prior_mfnp, transition_prior_mfnp, emission_prior_mfnp ):

        # Filter using the expected natural parameters
        expected_initial_nat_params    = dict( [ ( group, dist.prior.expectedSufficientStats( nat_params=initial_prior_mfnp[ group ] )[ 0 ] ) for group, dist in self.params.initial_dists.items() ] )
        expected_transition_nat_params = dict( [ ( group, [ dist.prior.expectedSufficientStats( nat_params=transition_prior_mfnp[ group ] )[ 0 ] ] ) for group, dist in self.params.transition_dists.items() ] )
        expected_emission_nat_params   = dict( [ ( group, dist.prior.expectedSufficientStats( nat_params=emission_prior_mfnp[ group ] )[ 0 ] ) for group, dist in self.params.emission_dists.items() ] )

        self.msg.updateNatParams( expected_initial_nat_params, expected_transition_nat_params, expected_emission_nat_params, check_parameters=False )
        self.runFilter()

        elbo = self.ELBO( initial_prior_mfnp, transition_prior_mfnp, emission_prior_mfnp )

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
        transition_prior_mfnp_update, = self.params.updatedTransitionPrior( self.msg, node_parents_smoothed )
        emission_prior_mfnp_update,   = self.params.updatedEmissionPrior( self.msg, node_smoothed )

        # Take a natural gradient step
        initial_prior_mfnp = ( 1 - self.p ) * self.initial_prior_mfnp[ 0 ] + self.p * initial_prior_mfnp_update
        transition_prior_mfnp = ( 1 - self.p ) * self.transition_prior_mfnp[ 0 ] + self.p * transition_prior_mfnp_update
        emission_prior_mfnp = ( 1 - self.p ) * self.emission_prior_mfnp[ 0 ] + self.p * emission_prior_mfnp_update
        return ( initial_prior_mfnp, ), ( transition_prior_mfnp, ), ( emission_prior_mfnp, )

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
        initial_prior_mfnp, transition_prior_mfnp, emission_prior_mfnp = {}, {}, {}
        for group in initial_prior_mfnp_update.keys():

            initial_prior_mfnp[ group ] = ( ( 1 - self.p ) * self.initial_prior_mfnp[ group ][ 0 ] + self.p * initial_prior_mfnp_update[ group ][ 0 ], )
            transition_prior_mfnp[ group ] = ( ( 1 - self.p ) * self.transition_prior_mfnp[ group ][ 0 ] + self.p * transition_prior_mfnp_update[ group ][ 0 ], )
            emission_prior_mfnp[ group ] = ( ( 1 - self.p ) * self.emission_prior_mfnp[ group ][ 0 ] + self.p * emission_prior_mfnp_update[ group ][ 0 ], )
        return initial_prior_mfnp, transition_prior_mfnp, emission_prior_mfnp

######################################################################

def autosomalDominantPriors( prior_strength=1.0 ):
    trans_prior = autosomalTransitionPrior() * prior_strength + 1
    emiss_prior = autosomalDominantEmissionPrior() * prior_strength + 1
    return trans_prior, emiss_prior

def autosomalRecessivePriors( prior_strength=1.0 ):
    trans_prior = autosomalTransitionPrior() * prior_strength + 1
    emiss_prior = autosomalRecessiveEmissionPrior() * prior_strength + 1
    return trans_prior, emiss_prior

######################################################################

class _Autosomal():

    def __init__( self, graphs, prior_strength=1.0, method='SVI', priors=None, params=None, _dominant=True, **kwargs ):
        assert method in [ 'EM', 'Gibbs', 'CAVI', 'SVI' ]
        self.graphs = graphs
        self.msg = PedigreeHMMFilter()
        self.method = method

        # Generate the priors
        if( priors is None ):
            trans_prior, emiss_prior = autosomalDominantPriors( prior_strength ) if _dominant else autosomalRecessivePriors( prior_strength )
        else:
            # Initialize using other priors
            trans_prior, emiss_prior = priors

        # Generate the parameters
        if( method == 'EM' ):
            self.params = AutosomalParametersEM( transition_prior=trans_prior, emission_prior=emiss_prior ) if params is None else params
        elif( method == 'Gibbs' ):
            self.params = AutosomalParametersGibbs( transition_prior=trans_prior, emission_prior=emiss_prior ) if params is None else params
        elif( method == 'CAVI' ):
            self.params = AutosomalParametersCAVI( transition_prior=trans_prior, emission_prior=emiss_prior ) if params is None else params
        else:
            self.params = AutosomalParametersSVI( transition_prior=trans_prior, emission_prior=emiss_prior ) if params is None else params

        # Generate the model objects
        if( method == 'EM' ):
            self.model = EM( msg=self.msg, parameters=self.params )
        elif( method == 'Gibbs' ):
            self.model = Gibbs( msg=self.msg, parameters=self.params )
        elif( method == 'CAVI' ):
            self.model = CAVI( msg=self.msg, parameters=self.params )
        else:
            step_size = kwargs[ 'step_size' ]
            self.minibatch_size = kwargs[ 'minibatch_size' ]
            self.model = SVI( msg=self.msg, parameters=self.params, minibatch_ratio=None, step_size=step_size )
            self.total_nodes = sum( [ len( graph.nodes ) for graph, fbs in self.graphs ] )

        if( method != 'SVI' ):
            self.msg.preprocessData( self.graphs )

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

    def fitStep( self, **kwargs ):

        if( self.method != 'SVI' ):
            return self.model.fitStep()

        minibatch_indices = np.random.randint( len( self.graphs ), size=self.minibatch_size )
        minibatch = [ self.graphs[ i ] for i in minibatch_indices ]
        self.msg.preprocessData( minibatch )

        # Compute minibatch ratio
        minibatch_nodes = sum( [ len( graph.nodes ) for graph, fbs in minibatch ] )
        self.params.setMinibatchRatio( self.total_nodes / minibatch_nodes )

        return self.model.fitStep()

    def sampleParams( self ):
        self.msg.updateParams( self.params.sampleInitialDist(), self.params.sampleTransitionDist(), self.params.sampleEmissionDist() )

    def marginal( self ):
        U, V = self.msg.filter()
        return self.msg.marginalProb( U, V, 0 )

    def generative( self ):
        assert self.method == 'Gibbs'
        return self.model.generativeProbability()

    def paramProb( self ):
        return self.params.paramProb()

class AutosomalDominant( _Autosomal ):
    def __init__( self, graphs, prior_strength=1.0, method='SVI', priors=None, params=None, **kwargs ):
        super().__init__( graphs, prior_strength, method=method, priors=priors, params=params, _dominant=True, **kwargs )

class AutosomalRecessive( _Autosomal ):
    def __init__( self, graphs, prior_strength=1.0, method='SVI', priors=None, params=None, **kwargs ):
        super().__init__( graphs, prior_strength, method=method, priors=priors, params=params, _dominant=False, **kwargs )

######################################################################

def xLinkedRecessivePriors( prior_strength=1.0 ):
    female_trans_prior = xLinkedFemaleTransitionPrior() * prior_strength + 1
    male_trans_prior = xLinkedMaleTransitionPrior() * prior_strength + 1
    unknown_trans_prior = xLinkedUnknownTransitionPrior() * prior_strength + 1
    trans_priors = [ female_trans_prior, male_trans_prior, unknown_trans_prior ]

    female_emiss_prior = xLinkedFemaleEmissionPrior() * prior_strength + 1
    male_emiss_prior = xLinkedMaleEmissionPrior() * prior_strength + 1
    unknown_emiss_prior = xLinkedUnknownEmissionPrior() * prior_strength + 1
    emiss_priors = [ female_emiss_prior, male_emiss_prior, unknown_emiss_prior ]

    return trans_priors, emiss_priors

######################################################################

class XLinkedRecessive():

    def __init__( self, graphs, prior_strength=1.0, method='SVI', priors=None, params=None, **kwargs ):
        assert method in [ 'EM', 'Gibbs', 'CAVI', 'SVI' ]
        self.graphs = graphs
        self.msg = PedigreeHMMFilterSexMatters()
        self.method = method

        # Generate the priors
        if( priors is None ):
            trans_priors, emiss_priors = xLinkedRecessivePriors( prior_strength )
        else:
            # Initialize using known priors
            trans_priors, emiss_priors = priors
            female_trans_prior, male_trans_prior, unknown_trans_prior = trans_priors
            female_emiss_prior, male_emiss_prior, unknown_emiss_prior = emiss_priors

        # Generate the parameters objects
        if( method == 'EM' ):
            self.params = XLinkedParametersEM( transition_priors=trans_priors, emission_priors=emiss_priors ) if params is None else params
        elif( method == 'Gibbs' ):
            self.params = XLinkedParametersGibbs( transition_priors=trans_priors, emission_priors=emiss_priors ) if params is None else params
        elif( method == 'CAVI' ):
            self.params = XLinkedParametersCAVI( transition_priors=trans_priors, emission_priors=emiss_priors ) if params is None else params
        else:
            self.params = XLinkedParametersSVI( transition_priors=trans_priors, emission_priors=emiss_priors ) if params is None else params

        # Generate the model objects
        if( method == 'EM' ):
            self.model = GroupEM( msg=self.msg, parameters=self.params )
        elif( method == 'Gibbs' ):
            self.model = GroupGibbs( msg=self.msg, parameters=self.params )
        elif( method == 'CAVI' ):
            self.model = GroupCAVI( msg=self.msg, parameters=self.params )
        else:
            step_size = kwargs[ 'step_size' ]
            self.minibatch_size = kwargs[ 'minibatch_size' ]
            minibatch_ratio = self.minibatch_size / len( self.graphs )
            self.model = GroupSVI( msg=self.msg, parameters=self.params, minibatch_ratio=minibatch_ratio, step_size=step_size )
            self.total_nodes = sum( [ len( graph.nodes ) for graph, fbs in self.graphs ] )

        if( method != 'SVI' ):
            self.msg.preprocessData( self.graphs )

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


    def fitStep( self, **kwargs ):

        if( self.method != 'SVI' ):
            return self.model.fitStep()

        minibatch_indices = np.random.randint( len( self.graphs ), size=self.minibatch_size )
        minibatch = [ self.graphs[ i ] for i in minibatch_indices ]
        self.msg.preprocessData( minibatch )

        # Compute minibatch ratio
        minibatch_nodes = sum( [ len( graph.nodes ) for graph, fbs in minibatch ] )
        self.params.setMinibatchRatio( self.total_nodes / minibatch_nodes )

        return self.model.fitStep()

    def sampleParams( self ):
        self.msg.updateParams( self.params.sampleInitialDist(), self.params.sampleTransitionDist(), self.params.sampleEmissionDist() )

    def marginal( self ):
        U, V = self.msg.filter()
        return self.msg.marginalProb( U, V, 0 )

    def generative( self ):
        assert self.method == 'Gibbs'
        return self.model.generativeProbability()

    def paramProb( self ):
        return self.params.paramProb()