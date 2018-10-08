from GenModels.GM.Distributions import Categorical, Dirichlet, TensorTransition, TensorTransitionDirichletPrior, BayesianNN
import autograd.numpy as np

__all__ = [ 'GibbsParameters',
            'GroupGibbsParameters',
            'EMParameters',
            'GroupEMParameters',
            'VIParameters',
            'CAVIParameters',
            'SVIParameters',
            'GroupVIParameters',
            'GroupCAVIParameters',
            'GroupSVIParameters',
            'SVAEParameters' ]

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
        ans += np.sum( [ dist.ilog_params() for dist in trans_dists ] )
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
                ans += np.sum( [ dist.ilog_params() for dist in trans_dists ] )
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

class SVAEParameters():

    def __init__( self, root_prior, transition_priors, d_emission, minibatch_ratio=1.0 ):

        self.initial_dist = Categorical( hypers=dict( alpha=root_prior ) )
        self.transition_dists = [ TensorTransition( hypers=dict( alpha=trans_prior ) ) for trans_prior in transition_priors ]

        # The hyperparameters are assumed to be a unit gaussian for the moment
        d_latent = root_prior.shape[ 0 ]
        self.emission_dist = BayesianNN( d_in=d_latent, d_out=d_emission )

        self.s = minibatch_ratio

    def paramProb( self ):
        ans = self.initial_dist.ilog_params()
        ans += np.sum( [ dist.ilog_params() for dist in trans_dists ] )
        ans += self.emission_dist.ilog_params()
        return ans

    def setMinibatchRatio( self, s ):
        self.s = s

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

        new_dists = []
        for dist in self.transition_dists:

            new_dist = dist.prior.mf_nat_params[ 0 ]
            new_dist += self.s * expected_transition_stats[ dist.pi.ndim ]
            new_dists.append( new_dist, )
        return new_dists
