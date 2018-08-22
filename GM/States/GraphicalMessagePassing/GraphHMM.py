from GenModels.GM.States.GraphicalMessagePassing.GraphicalMessagePassingBase import *
from GenModels.GM.States.GraphicalMessagePassing.GraphFilterBase import *
import numpy as np
from functools import partial
from scipy.sparse import coo_matrix
from collections import Iterable
from GenModels.GM.Utility import fbsData
import joblib

__all__ = [ 'GraphHMM',
            'GraphHMMFBS',
            'GraphHMMFBSMultiGroups' ]

######################################################################

# def unwrappedUBaseCase( cls, self, node ):
#     return cls.uBaseCase( self, node )

# def unwrappedU( cls, self, U, V, node ):
#     return cls.u( self, U, V, node )

# def unwrappedVBaseCase( cls, self, node ):
#     return cls.vBaseCase( self, node )

# def unwrappedV( cls, self, U, V, node, edge ):
#     return cls.v( self, U, V, node, edge )

######################################################################

class _graphHMMMixin():

    # def uFilter( self, is_base_case, nodes, U, V, parallel=False ):
    #     # Parallel version

    #     if( parallel == True ):
    #         # self_it = itertools.repeat( self, nodes.shape[ 0 ] )
    #         new_u = joblib.Parallel( n_jobs=-1 )( [ joblib.delayed( unwrappedUBaseCase )( GraphHMM, self, node ) if is_base_case else
    #                                                 joblib.delayed( unwrappedU )( GraphHMM, self, U, V, node ) for node in nodes ] )
    #         self.updateU( nodes, new_u, U )
    #     else:
    #         super().uFilter( is_base_case, nodes, U, V )

    # def vFilter( self, is_base_case, nodes_and_edges, U, V, parallel=False ):

    #     if( parallel == True ):
    #         nodes, edges = nodes_and_edges
    #         new_v = joblib.Parallel( n_jobs=-1 )( [ joblib.delayed( unwrappedVBaseCase )( GraphHMM, self, node ) if is_base_case else
    #                                                 joblib.delayed( unwrappedV )( GraphHMM, self, U, V, node, edge ) for node, edge in zip( nodes, edges ) ] )
    #         self.updateV( nodes, edges, new_v, V )
    #     else:
    #         super().vFilter( is_base_case, nodes_and_edges, U, V )

    ######################################################################

    def genFilterProbs( self ):

        # Initialize U and V
        U = []
        for node in self.nodes:
            U.append( np.zeros( ( self.K, ) ) )

        V_row = self.pmask.row
        V_col = self.pmask.col
        V_data = []
        for node in self.pmask.row:
            V_data.append( np.zeros( ( self.K, ) ) )

        # Invalidate all data elements
        for node in self.nodes:
            U[ node ][ : ] = np.nan
            self.assignV( ( V_row, V_col, V_data ), node, np.nan, keep_shape=True )

        return U, ( V_row, V_col, V_data )

    ######################################################################

    def parameterCheck( self, log_initial_dist, log_transition_dist, log_emission_dist ):
        K = log_initial_dist.shape[ 0 ]
        assert log_initial_dist.ndim == 1
        assert log_initial_dist.shape == ( K, )
        for _transition_dist in log_transition_dist:
            assert np.allclose( np.ones( K ), np.exp( _transition_dist ).sum( axis=-1 ) ), np.exp( _transition_dist ).sum( axis=-1 )
        assert log_emission_dist.shape[ 0 ] == K
        assert np.isclose( 1.0, np.exp( log_initial_dist ).sum() )
        assert np.allclose( np.ones( K ), np.exp( log_emission_dist ).sum( axis=1 ) )
        pis = set()
        for dist in log_transition_dist:
            ndim = dist.ndim
            assert ndim not in pis
            pis.add( ndim )

    def preprocessData( self, data_graphs ):

        super( _graphHMMMixin, self ).updateGraphs( data_graphs )

        self.possible_latent_states = {}

        total_nodes = 0
        for data_graph in data_graphs:
            for node, state in data_graph.possible_latent_states.items():
                self.possible_latent_states[ total_nodes + node ] = state
            total_nodes += len( data_graph.nodes )

        ys = []
        for graph in data_graphs:
            ys.extend( [ graph.data[ node ] if graph.data[ node ] is not None else np.nan for node in graph.nodes ] )

        self.ys = ys

        if( hasattr( self, 'emission_dist' ) ):
            self.L_set = True
            ys = np.array( ys ).T
            self.L = np.array( [ self.emission_dist[ :, y ] if not np.any( np.isnan( y ) ) else np.zeros_like( self.emission_dist[ :, 0 ] )for y in ys ] ).sum( axis=0 ).T

    def updateParams( self, initial_dist, transition_dist, emission_dist, data_graphs=None, compute_marginal=True ):

        log_initial_dist = np.log( initial_dist )
        log_transition_dist = [ np.log( dist ) for dist in transition_dist ]
        log_emission_dist = np.log( emission_dist )

        self.updateNatParams( log_initial_dist, log_transition_dist, log_emission_dist, data_graphs=data_graphs, compute_marginal=compute_marginal )

    def updateNatParams( self, log_initial_dist, log_transition_dist, log_emission_dist, data_graphs=None, check_parameters=True, compute_marginal=True ):

        if( check_parameters ):
            self.parameterCheck( log_initial_dist, log_transition_dist, log_emission_dist )

        self.K = log_initial_dist.shape[ 0 ]
        self.pi0 = log_initial_dist
        self.pis = {}
        for log_dist in log_transition_dist:
            ndim = log_dist.ndim
            self.pis[ ndim ] = log_dist

        self.emission_dist = log_emission_dist
        self.L_set = False

        if( data_graphs is not None ):
            self.preprocessData( data_graphs )

        if( hasattr( self, 'ys' ) and self.L_set == False ):
            self.L_set = True
            ys = np.array( self.ys ).T
            assert ys.ndim == 2, 'If there is only 1 measurement, add an extra dim!'
            self.L = np.array( [ self.emission_dist[ :, y ] if not np.any( np.isnan( y ) ) else np.zeros_like( self.emission_dist[ :, 0 ] )for y in ys ] ).sum( axis=0 ).T

    ######################################################################

    def transitionProb( self, child ):
        parents, parent_order = self.getParents( child, get_order=True )
        ndim = len( parents ) + 1
        pi = np.copy( self.pis[ ndim ] )

        # If we know the latent state for child, then ensure that we
        # transition there
        if( int( child ) in self.possible_latent_states ):
            states = self.possible_latent_states[ int( child ) ]
            impossible_axes = np.setdiff1d( np.arange( pi.shape[ -1 ] ), states )
            pi[ ..., impossible_axes ] = np.NINF
            pi[ ..., states ] -= np.logaddexp.reduce( pi, axis=-1 )[ ..., None ]

        # Reshape pi's axes to match parent order
        assert len( parents ) + 1 == pi.ndim
        assert parent_order.shape[ 0 ] == parents.shape[ 0 ]
        pi = np.moveaxis( pi, np.arange( ndim ), np.hstack( ( parent_order, ndim - 1 ) ) )
        return pi

    ######################################################################

    def emissionProb( self, node, forward=False ):
        prob = self.L[ node ].reshape( ( -1, ) )
        return prob

    ######################################################################

    @classmethod
    def multiplyTerms( cls, terms ):
        # Basically np.einsum but in log space

        assert isinstance( terms, Iterable )

        # Remove the empty terms
        terms = [ t for t in terms if np.prod( t.shape ) > 1 ]

        ndim = max( [ len( term.shape ) for term in terms ] )

        axes = [ [ i for i, s in enumerate( t.shape ) if s != 1 ] for t in terms ]

        # Get the shape of the output
        shape = np.ones( ndim, dtype=int )
        for ax, term in zip( axes, terms ):
            shape[ np.array( ax ) ] = term.squeeze().shape

        total_elts = shape.prod()
        if( total_elts > 1e8 ):
            assert 0, 'Don\'t do this on a cpu!  Too many terms: %d'%( int( total_elts ) )

        # Build a meshgrid out of each of the terms over the right axes
        # and sum.  Doing it this way because np.einsum doesn't work
        # for matrix multiplication in log space - we can't do np.einsum
        # but add instead of multiply over indices

        ans = np.zeros( shape )
        for ax, term in zip( axes, terms ):

            for _ in range( ndim - term.ndim ):
                term = term[ ..., None ]

            ans += np.broadcast_to( term, ans.shape )

        return ans

    ######################################################################

    @classmethod
    def integrate( cls, integrand, axes ):
        # Need adjusted axes because the relative axes in integrand change as we reduce
        # over each axis
        assert isinstance( axes, Iterable )
        if( len( axes ) == 0 ):
            return integrand

        assert max( axes ) < integrand.ndim
        axes = np.array( axes )
        axes[ axes < 0 ] = integrand.ndim + axes[ axes < 0 ]
        adjusted_axes = np.array( sorted( axes ) ) - np.arange( len( axes ) )
        for ax in adjusted_axes:
            integrand = np.logaddexp.reduce( integrand, axis=ax )

        return integrand

    ######################################################################

    def uBaseCase( self, node, debug=True ):
        pi = np.copy( self.pi0 )
        if( int( node ) in self.possible_latent_states ):
            states = self.possible_latent_states[ int( node ) ]
            impossible_states = np.setdiff1d( np.arange( pi.shape[ -1 ] ), states )
            for state in impossible_states:
                pi[ state ] = np.NINF
            pi[ states ] -= np.logaddexp.reduce( pi )

        initial_dist = pi
        emission = self.emissionProb( node )
        newU = self.multiplyTerms( terms=( emission, initial_dist ) )
        return newU

    def vBaseCase( self, node, debug=True ):
        return np.zeros( self.K )

    ######################################################################

    def filter( self, **kwargs ):
        # For loopy belief propagation
        # self.total_deviation = 0.0
        return super().filter( **kwargs )

    ######################################################################

    def updateU( self, nodes, newU, U ):

        for u, node in zip( newU, nodes ):
            # self.total_deviation += np.logaddexp( U_data[ i ], -v )**2
            U[ node ] = u

    def updateV( self, nodes, edges, newV, V ):

        V_row, V_col, V_data = V

        for node, edge, v in zip( nodes, edges, newV ):
            if( edge is None ):
                continue

            dataIndices = np.in1d( V_row, node ) & np.in1d( V_col, edge )

            for i, maskValue in enumerate( dataIndices ):

                # Don't convert V_data to an np.array even though it makes this
                # step faster because it messes up when we add fbs nodes
                if( maskValue == True ):
                    # self.total_deviation += np.logaddexp( V_data[ i ], -v )**2
                    V_data[ i ] = v

######################################################################

class GraphHMM( _graphHMMMixin, GraphFilter ):
    pass

######################################################################

class GraphHMMFBS( _graphHMMMixin, GraphFilterFBS ):

    # def uFilter( self, is_base_case, nodes, U, V, parallel=False ):
    #     # Parallel version

    #     if( parallel == True ):
    #         # self_it = itertools.repeat( self, nodes.shape[ 0 ] )
    #         new_u = joblib.Parallel( n_jobs=-1 )( [ joblib.delayed( unwrappedUBaseCase )( GraphHMMFBS, self, node ) if is_base_case else
    #                                                 joblib.delayed( unwrappedU )( GraphHMMFBS, self, U, V, node ) for node in nodes ] )
    #         self.updateU( nodes, new_u, U )
    #     else:
    #         super().uFilter( is_base_case, nodes, U, V )

    # def vFilter( self, is_base_case, nodes_and_edges, U, V, parallel=False ):

    #     if( parallel == True ):
    #         nodes, edges = nodes_and_edges
    #         new_v = joblib.Parallel( n_jobs=-1 )( [ joblib.delayed( unwrappedVBaseCase )( GraphHMMFBS, self, node ) if is_base_case else
    #                                                 joblib.delayed( unwrappedV )( GraphHMMFBS, self, U, V, node, edge ) for node, edge in zip( nodes, edges ) ] )
    #         self.updateV( nodes, edges, new_v, V )
    #     else:
    #         super().vFilter( is_base_case, nodes_and_edges, U, V )

    ######################################################################

    def preprocessData( self, data_graphs ):

        super().updateGraphs( data_graphs )

        self.possible_latent_states = {}

        total_nodes = 0
        for data_graph, fbs in data_graphs:
            for node, state in data_graph.possible_latent_states.items():
                self.possible_latent_states[ total_nodes + node ] = state
            total_nodes += len( data_graph.nodes )

        ys = []
        for graph, fbs in data_graphs:
            ys.extend( [ graph.data[ node ] if graph.data[ node ] is not None else np.nan for node in graph.nodes ] )

        self.ys = ys

        if( hasattr( self, 'emission_dist' ) ):
            self.L_set = True
            ys = np.array( ys ).T
            assert ys.ndim == 2, 'If there is only 1 measurement, add an extra dim!'
            self.L = np.array( [ self.emission_dist[ :, y ] if not np.any( np.isnan( y ) ) else np.zeros_like( self.emission_dist[ :, 0 ] )for y in ys ] ).sum( axis=0 ).T

    ######################################################################

    def genFilterProbs( self ):

        # Initialize U and V
        U = []
        for node in self.partial_graph.nodes:
            U.append( fbsData( np.zeros( self.K ), -1 ) )

        V_row = self.partial_graph.pmask.row
        V_col = self.partial_graph.pmask.col
        V_data = []
        for node in self.partial_graph.pmask.row:
            V_data.append( fbsData( np.zeros( self.K ), -1 ) )

        # Invalidate all data elements
        for node in self.partial_graph.nodes:
            U[ node ][ : ] = np.nan
            self.assignV( ( V_row, V_col, V_data ), node, np.nan, keep_shape=True )

        return U, ( V_row, V_col, V_data )

    ######################################################################

    def transitionProb( self, child, is_partial_graph_index=False ):
        parents, parent_order = self.getFullParents( child, get_order=True, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=True )
        ndim = len( parents ) + 1
        pi = np.copy( self.pis[ ndim ] )
        # Reshape pi's axes to match parent order
        assert len( parents ) + 1 == pi.ndim
        assert parent_order.shape[ 0 ] == parents.shape[ 0 ]

        # Sort the parent dimensions by parent order
        pi = np.moveaxis( pi, np.arange( ndim ), np.hstack( ( parent_order, ndim - 1 ) ) )

        # If we know the latent state for child, then ensure that we
        # transition there
        node_full = self.partialGraphIndexToFullGraphIndex( child ) if is_partial_graph_index == True else child
        if( int( node_full ) in self.possible_latent_states ):
            states = self.possible_latent_states[ int( node_full ) ]
            impossible_axes = np.setdiff1d( np.arange( pi.shape[ -1 ] ), states )
            pi[ ..., impossible_axes ] = np.NINF
            pi[ ..., states ] -= np.logaddexp.reduce( pi, axis=-1 )[ ..., None ]

        # Check if there are nodes in [ child, *parents ] that are in the fbs.
        # If there are, then move their axes
        fbsOffset = lambda x: self.fbsIndex( x, is_partial_graph_index=True, within_graph=True ) + 1
        fbs_indices = [ fbsOffset( parent ) for parent in parents if self.inFeedbackSet( parent, is_partial_graph_index=True ) ]

        if( self.inFeedbackSet( child, is_partial_graph_index=is_partial_graph_index ) ):
            fbs_indices.append( self.fbsIndex( child, is_partial_graph_index=is_partial_graph_index, within_graph=True ) + 1 )

        if( len( fbs_indices ) > 0 ):
            expand_by = max( fbs_indices )
            for _ in range( expand_by ):
                pi = pi[ ..., None ]

            # If there are parents in the fbs, move them to the appropriate axes
            for i, parent in enumerate( parents ):
                if( self.inFeedbackSet( parent, is_partial_graph_index=True ) ):
                    pi = np.swapaxes( pi, i, fbsOffset( parent ) + ndim - 1 )

            if( self.inFeedbackSet( child, is_partial_graph_index=is_partial_graph_index ) ):
                # If the child is in the fbs, then move it to the appropriate axis
                pi = np.swapaxes( pi, ndim - 1, fbsOffset( child ) + ndim - 1 )

            return fbsData( pi, ndim )
        return fbsData( pi, -1 )

    ######################################################################

    def emissionProb( self, node, is_partial_graph_index=False ):
        # Access the emission matrix with the full graph indices
        node_full = self.partialGraphIndexToFullGraphIndex( node ) if is_partial_graph_index == True else node
        prob = self.L[ node_full ].reshape( ( -1, ) )
        if( self.inFeedbackSet( node_full, is_partial_graph_index=False ) ):
            return fbsData( prob, 0 )
        return fbsData( prob, -1 )

    ######################################################################

    @classmethod
    def multiplyTerms( cls, terms ):
        # Basically np.einsum but in log space

        assert isinstance( terms, Iterable )

        # Check if we should use the multiply for fbsData or for regular data
        fbs_data_count, non_fbs_data_count = ( 0, 0 )
        for t in terms:
            if( isinstance( t, fbsData ) ):
                fbs_data_count += 1
            else:
                non_fbs_data_count += 1

        # Can't mix types
        if( not ( fbs_data_count == 0 or non_fbs_data_count == 0 ) ):
            print( 'fbs_data_count', fbs_data_count )
            print( 'non_fbs_data_count', non_fbs_data_count )
            print( terms )
            for t in terms:
                if( isinstance( t, fbsData ) ):
                    print( 'this ones good', t, type( t ) )
                else:
                    print( 'this ones bad', t, type( t ) )
            assert 0

        # Use the regular multiply if we don't have fbs data
        if( fbs_data_count == 0 ):
            return GraphHMM.multiplyTerms( terms )

        # Remove the empty terms
        terms = [ t for t in terms if np.prod( t.shape ) > 1 ]

        if( len( terms ) == 0 ):
            return fbsData( np.array( [] ), 0 )

        # Separate out where the feedback set axes start and get the largest fbs_axis.
        # Need to handle case where ndim of term > all fbs axes
        # terms, fbs_axes_start = list( zip( *terms ) )
        fbs_axes_start = [ term.fbs_axis for term in terms ]
        terms = [ term.data for term in terms ]

        if( max( fbs_axes_start ) != -1 ):
            max_fbs_axis = max( [ ax if ax != -1 else term.ndim for ax, term in zip( fbs_axes_start, terms ) ] )

            if( max_fbs_axis > 0 ):
                # Pad extra dims at each term so that the fbs axes start the same way for every term
                for i, ax in enumerate( fbs_axes_start ):
                    if( ax == -1 ):
                        for _ in range( max_fbs_axis - terms[ i ].ndim + 1 ):
                            terms[ i ] = terms[ i ][ ..., None ]
                    else:
                        for _ in range( max_fbs_axis - ax ):
                            terms[ i ] = np.expand_dims( terms[ i ], axis=ax )
        else:
            max_fbs_axis = -1

        ndim = max( [ len( term.shape ) for term in terms ] )

        axes = [ [ i for i, s in enumerate( t.shape ) if s != 1 ] for t in terms ]

        # Get the shape of the output
        shape = np.ones( ndim, dtype=int )
        for ax, term in zip( axes, terms ):
            shape[ np.array( ax ) ] = term.squeeze().shape

        total_elts = shape.prod()
        if( total_elts > 1e8 ):
            assert 0, 'Don\'t do this on a cpu!  Too many terms: %d'%( int( total_elts ) )

        # Build a meshgrid out of each of the terms over the right axes
        # and sum.  Doing it this way because np.einsum doesn't work
        # for matrix multiplication in log space - we can't do np.einsum
        # but add instead of multiply over indices
        ans = np.zeros( shape )
        for ax, term in zip( axes, terms ):

            for _ in range( ndim - term.ndim ):
                term = term[ ..., None ]

            ans += np.broadcast_to( term, ans.shape )

        return fbsData( ans, max_fbs_axis )

    ######################################################################

    @classmethod
    def integrate( cls, integrand, axes ):

        # Check if we need to use the regular integrate
        if( not isinstance( integrand, fbsData ) ):
            return GraphHMM.integrate( integrand, axes )

        # Need adjusted axes because the relative axes in integrand change as we reduce
        # over each axis
        assert isinstance( axes, Iterable )
        if( len( axes ) == 0 ):
            return integrand

        integrand, fbs_axis = ( integrand.data, integrand.fbs_axis )

        assert max( axes ) < integrand.ndim
        axes = np.array( axes )
        axes[ axes < 0 ] = integrand.ndim + axes[ axes < 0 ]
        adjusted_axes = np.array( sorted( axes ) ) - np.arange( len( axes ) )
        for ax in adjusted_axes:
            integrand = np.logaddexp.reduce( integrand, axis=ax )

        if( fbs_axis > -1 ):
            fbs_axis -= len( adjusted_axes )
            # assert fbs_axis > -1, adjusted_axes

        return fbsData( integrand, fbs_axis )

    ######################################################################

    def uBaseCase( self, node ):
        pi = np.copy( self.pi0 )
        node_full = self.partialGraphIndexToFullGraphIndex( node )
        if( int( node_full ) in self.possible_latent_states ):
            states = self.possible_latent_states[ int( node_full ) ]
            impossible_states = np.setdiff1d( np.arange( pi.shape[ -1 ] ), states )
            for state in impossible_states:
                pi[ impossible_states ] = np.NINF
            pi[ states ] -= np.logaddexp.reduce( pi )

        initial_dist = fbsData( pi, -1 )
        emission = self.emissionProb( node, is_partial_graph_index=True )
        return self.multiplyTerms( terms=( emission, initial_dist ) )

    def vBaseCase( self, node ):
        return fbsData( np.zeros( self.K ), -1 )

######################################################################

class GraphHMMFBSMultiGroups( GraphHMMFBS ):

    # This variant lets the user specify which set of parameters to apply to a node

    def genFilterProbs( self ):

        # Initialize U and V
        U = []
        for node in self.partial_graph.nodes:
            group = self.node_groups[ node ]
            U.append( fbsData( np.zeros( self.Ks[ group ] ), -1 ) )

        V_row = self.partial_graph.pmask.row
        V_col = self.partial_graph.pmask.col
        V_data = []
        for node in self.partial_graph.pmask.row:
            group = self.node_groups[ node ]
            V_data.append( fbsData( np.zeros( self.Ks[ group ] ), -1 ) )

        # Invalidate all data elements
        for node in self.partial_graph.nodes:
            U[ node ][ : ] = np.nan
            self.assignV( ( V_row, V_col, V_data ), node, np.nan, keep_shape=True )

        return U, ( V_row, V_col, V_data )

    ######################################################################

    def parameterCheck( self, log_initial_dists, log_transition_dists, log_emission_dists ):

        assert len( log_initial_dists.keys() ) == len( log_transition_dists.keys() ) and len( log_transition_dists.keys() ) == len( log_emission_dists.keys() )

        for group in log_initial_dists.keys():
            log_initial_dist = log_initial_dists[ group ]
            log_transition_dist = log_transition_dists[ group ]
            log_emission_dist = log_emission_dists[ group ]

            K = log_initial_dist.shape[ 0 ]
            assert log_initial_dist.ndim == 1
            assert log_initial_dist.shape == ( K, )
            for _transition_dist in log_transition_dist:
                assert np.allclose( np.ones( _transition_dist.shape[ :-1 ] ), np.exp( _transition_dist ).sum( axis=-1 ) ), _transition_dist.sum( axis=-1 )
            assert log_emission_dist.shape[ 0 ] == K
            assert np.isclose( 1.0, np.exp( log_initial_dist ).sum() )
            assert np.allclose( np.ones( K ), np.exp( log_emission_dist ).sum( axis=1 ) )
            pis = set()
            for dist in log_transition_dist:
                ndim = dist.shape
                assert ndim not in pis
                pis.add( ndim )

    def preprocessData( self, group_graphs ):

        super().updateGraphs( group_graphs )

        self.possible_latent_states = {}
        self.node_groups = {}

        total_nodes = 0
        for group_graph, fbs in group_graphs:
            for node, state in group_graph.possible_latent_states.items():
                self.possible_latent_states[ total_nodes + node ] = state

            for node, group in group_graph.groups.items():
                self.node_groups[ total_nodes + node ] = group

            total_nodes += len( group_graph.nodes )

        # Each node must have an assigned group
        assert len( self.node_groups ) == self.nodes.shape[ 0 ]

        self.ys = []
        for graph, fbs in group_graphs:
            self.ys.extend( [ graph.data[ node ] if graph.data[ node ] is not None else np.nan for node in graph.nodes ] )

    def updateParams( self, initial_dists, transition_dists, emission_dists, group_graphs=None, compute_marginal=True ):

        assert isinstance( initial_dists, dict ), 'Make a dict that maps groups to parameters'
        assert isinstance( transition_dists, dict ), 'Make a dict that maps groups to parameters'
        assert isinstance( emission_dists, dict ), 'Make a dict that maps groups to parameters'

        # Ignore warning for when an entry of a dist is 0
        with np.errstate( divide='ignore', invalid='ignore' ):
            log_initial_dists = {}
            for group, dist in initial_dists.items():
                log_initial_dists[ group ] = np.log( dist )

            log_transition_dists = {}
            for group, dists in transition_dists.items():
                log_transition_dists[ group ] = [ np.log( dist ) for dist in dists ]

            log_emission_dists = {}
            for group, dist in emission_dists.items():
                log_emission_dists[ group ] = np.log( dist )

        self.updateNatParams( log_initial_dists, log_transition_dists, log_emission_dists, group_graphs=group_graphs, compute_marginal=compute_marginal )

    def updateNatParams( self, log_initial_dists, log_transition_dists, log_emission_dists, group_graphs=None, check_parameters=True, compute_marginal=True ):

        if( check_parameters ):
            self.parameterCheck( log_initial_dists, log_transition_dists, log_emission_dists )

        # Get the latent state sizes from the initial dist
        self.Ks = dict( [ ( group, dist.shape[ 0 ] ) for group, dist in log_initial_dists.items() ] )

        # Set the initial distributions
        self.pi0s = dict( [ ( group, dist ) for group, dist in log_initial_dists.items() ] )

        # Set the transition distributions
        self.pis = {}
        for group, log_dists in log_transition_dists.items():
            self.pis[ group ] = {}
            for log_dist in log_dists:
                shape = log_dist.shape
                self.pis[ group ][ shape ] = log_dist

        # Set the emission distributions
        self.emission_dists = dict( [ ( group, dist ) for group, dist in log_emission_dists.items() ] )
        self.L_set = False

        if( group_graphs is not None ):
            self.preprocessData( group_graphs )

    ######################################################################

    def transitionProb( self, child, is_partial_graph_index=False ):
        parents, parent_order = self.getFullParents( child, get_order=True, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=True )
        node_full = self.partialGraphIndexToFullGraphIndex( child ) if is_partial_graph_index == True else child
        ndim = len( parents ) + 1
        group = self.node_groups[ int( node_full ) ]
        shape = []
        for p, _ in sorted( zip( parents, parent_order ), key=lambda po: po[ 1 ] ):
            full_p = int( self.partialGraphIndexToFullGraphIndex( p ) )
            g = self.node_groups[ full_p ]
            shape.append( self.pi0s[ g ].shape[ 0 ] )
        shape.append( self.pi0s[ group ].shape[ 0 ] )
        shape = tuple( shape )

        pi = np.copy( self.pis[ group ][ shape ] )
        # Reshape pi's axes to match parent order
        assert len( parents ) + 1 == pi.ndim
        assert parent_order.shape[ 0 ] == parents.shape[ 0 ]

        # Sort the parent dimensions by parent order
        pi = np.moveaxis( pi, np.arange( ndim ), np.hstack( ( parent_order, ndim - 1 ) ) )

        # If we know the latent state for child, then ensure that we
        # transition there
        if( int( node_full ) in self.possible_latent_states ):
            states = self.possible_latent_states[ int( node_full ) ]
            impossible_axes = np.setdiff1d( np.arange( pi.shape[ -1 ] ), states )
            pi[ ..., impossible_axes ] = np.NINF
            pi[ ..., states ] -= np.logaddexp.reduce( pi, axis=-1 )[ ..., None ]

        # Check if there are nodes in [ child, *parents ] that are in the fbs.
        # If there are, then move their axes
        fbsOffset = lambda x: self.fbsIndex( x, is_partial_graph_index=True, within_graph=True ) + 1
        fbs_indices = [ fbsOffset( parent ) for parent in parents if self.inFeedbackSet( parent, is_partial_graph_index=True ) ]

        if( self.inFeedbackSet( child, is_partial_graph_index=is_partial_graph_index ) ):
            fbs_indices.append( self.fbsIndex( child, is_partial_graph_index=is_partial_graph_index, within_graph=True ) + 1 )

        if( len( fbs_indices ) > 0 ):
            expand_by = max( fbs_indices )
            for _ in range( expand_by ):
                pi = pi[ ..., None ]

            # If there are parents in the fbs, move them to the appropriate axes
            for i, parent in enumerate( parents ):
                if( self.inFeedbackSet( parent, is_partial_graph_index=True ) ):
                    pi = np.swapaxes( pi, i, fbsOffset( parent ) + ndim - 1 )

            if( self.inFeedbackSet( child, is_partial_graph_index=is_partial_graph_index ) ):
                # If the child is in the fbs, then move it to the appropriate axis
                pi = np.swapaxes( pi, ndim - 1, fbsOffset( child ) + ndim - 1 )

            return fbsData( pi, ndim )
        return fbsData( pi, -1 )

    ######################################################################

    def emissionProb( self, node, is_partial_graph_index=False ):
        # Access the emission matrix with the full graph indices
        node_full = self.partialGraphIndexToFullGraphIndex( node ) if is_partial_graph_index == True else node

        group = self.node_groups[ int( node_full ) ]
        y = self.ys[ int( node_full ) ]
        if( not np.any( np.isnan( y ) ) ):
            prob = self.emission_dists[ group ][ :, y ].sum( axis=-1 )
        else:
            prob = np.zeros_like( self.emission_dists[ group ][ :, 0 ] )

        if( self.inFeedbackSet( node_full, is_partial_graph_index=False ) ):
            fbs_index = self.fbsIndex( node_full, is_partial_graph_index=False, within_graph=True )
            for _ in range( fbs_index ):
                prob = prob[ None ]
            return fbsData( prob, 0 )
        return fbsData( prob, -1 )

    ######################################################################

    def uBaseCase( self, node ):
        node_full = self.partialGraphIndexToFullGraphIndex( node )
        group = self.node_groups[ int( node_full ) ]
        pi = np.copy( self.pi0s[ group ] )
        if( int( node_full ) in self.possible_latent_states ):
            states = self.possible_latent_states[ int( node_full ) ]
            impossible_states = np.setdiff1d( np.arange( pi.shape[ -1 ] ), states )
            for state in impossible_states:
                pi[ impossible_states ] = np.NINF
            pi[ states ] -= np.logaddexp.reduce( pi )

        initial_dist = fbsData( pi, -1 )
        emission = self.emissionProb( node, is_partial_graph_index=True )
        return self.multiplyTerms( terms=( emission, initial_dist ) )

    def vBaseCase( self, node ):
        node_full = self.partialGraphIndexToFullGraphIndex( node )
        group = self.node_groups[ int( node_full ) ]
        return fbsData( np.zeros( self.Ks[ group ] ), -1 )
