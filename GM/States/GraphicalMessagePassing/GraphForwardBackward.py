from GenModels.GM.States.GraphicalMessagePassing.GraphicalMessagePassingBase import *
from GenModels.GM.States.GraphicalMessagePassing.GraphFilterBase import *
import numpy as np
from functools import reduce
from scipy.sparse import coo_matrix
from collections import Iterable
from GenModels.GM.Utility import fbsData

__all__ = [ 'GraphCategoricalForwardBackward',
            'GraphCategoricalForwardBackwardFBS' ]

######################################################################

class _fowardBackwardMixin():

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
        for _transition_dist in log_transition_dist.values():
            assert np.allclose( np.ones( K ), np.exp( _transition_dist ).sum( axis=-1 ) ), _transition_dist.sum( axis=-1 )
        assert log_emission_dist.shape[ 0 ] == K
        assert np.isclose( 1.0, np.exp( log_initial_dist ).sum() )
        assert np.allclose( np.ones( K ), np.exp( log_emission_dist ).sum( axis=1 ) )
        pis = set()
        for dist in log_transition_dist.values():
            ndim = dist.ndim
            assert ndim not in pis
            pis.add( ndim )

    def preprocessData( self, data_graphs, only_load=False ):

        super( _fowardBackwardMixin, self ).updateGraphs( data_graphs )

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
            ys = np.array( ys ).T
            self.L = np.array( [ self.emission_dist[ :, y ] if not np.any( np.isnan( y ) ) else np.zeros_like( self.emission_dist[ :, 0 ] )for y in ys ] ).sum( axis=0 ).T

    def updateParams( self, initial_dist, transition_dist, emission_dist, data_graphs=None, compute_marginal=True ):

        log_initial_dist = np.log( initial_dist )
        log_transition_dist = {}
        log_emission_dist = np.log( emission_dist )
        for dist in transition_dist:
            ndim = dist.ndim
            log_transition_dist[ ndim ] = np.log( dist )

        self.updateNatParams( log_initial_dist, log_transition_dist, log_emission_dist, data_graphs=data_graphs, compute_marginal=compute_marginal )

    def updateNatParams( self, log_initial_dist, log_transition_dist, log_emission_dist, data_graphs=None, compute_marginal=True ):

        self.parameterCheck( log_initial_dist, log_transition_dist, log_emission_dist )

        self.K = log_initial_dist.shape[ 0 ]
        self.pi0 = log_initial_dist
        self.pis = {}
        for log_dist in log_transition_dist.values():
            ndim = log_dist.ndim
            self.pis[ ndim ] = log_dist

        self.emission_dist = log_emission_dist

        if( data_graphs is not None ):
            self.preprocessData( data_graphs )

    ######################################################################

    def transitionProb( self, child ):
        parents, parent_order = self.getParents( child, get_order=True )
        ndim = len( parents ) + 1
        pi = np.copy( self.pis[ ndim ] )

        # If we know the latent state for child, then ensure that we
        # transition there
        if( int( child ) in self.possible_latent_states ):
            states = self.possible_latent_states[ child ]
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

            ax = [ i for i, s in enumerate( term.shape ) if s != 1 ]

            for _ in range( ndim - term.ndim ):
                term = term[ ..., None ]

            # Build a meshgrid to correspond to the final shape and repeat
            # over axes that aren't in ax
            reps = np.copy( shape )
            reps[ np.array( ax ) ] = 1

            ans += np.tile( term, reps )

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
        initial_dist = self.pi0
        emission = self.emissionProb( node )
        newU = self.multiplyTerms( terms=( emission, initial_dist ) )
        return newU

    def vBaseCase( self, node, debug=True ):
        return np.zeros( self.K )

    ######################################################################

    def filter( self, **kwargs ):
        # For loopy belief propagation
        self.total_deviation = 0.0
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

class GraphCategoricalForwardBackward( _fowardBackwardMixin, GraphFilter ):
    pass

######################################################################

class GraphCategoricalForwardBackwardFBS( _fowardBackwardMixin, GraphFilterFBS ):

    def preprocessData( self, data_graphs, only_load=False ):

        super().updateGraphs( data_graphs )
        # super( _fowardBackwardMixin, self ).updateParams( data_graphs )

        self.possible_latent_states = {}

        total_nodes = 0
        for data_graph, fbs in data_graphs:
            for node, state in data_graph.possible_latent_states.items():
                self.possible_latent_states[ total_nodes + node ] = state
            total_nodes += len( data_graph.nodes )

        if( only_load == False ):
            ys = []
            for graph, fbs in data_graphs:
                ys.extend( [ graph.data[ node ] if graph.data[ node ] is not None else np.nan for node in graph.nodes ] )

            self.ys = ys

            if( hasattr( self, 'emission_dist' ) ):
                ys = np.array( ys ).T
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
        fbsIndices = [ fbsOffset( parent ) for parent in parents if self.inFeedbackSet( parent, is_partial_graph_index=True ) ]

        if( self.inFeedbackSet( child, is_partial_graph_index=is_partial_graph_index ) ):
            fbsIndices.append( self.fbsIndex( child, is_partial_graph_index=is_partial_graph_index, within_graph=True ) + 1 )

        if( len( fbsIndices ) > 0 ):
            expandBy = max( fbsIndices )
            for _ in range( expandBy ):
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
        assert fbs_data_count == 0 or non_fbs_data_count == 0

        # Use the regular multiply if we don't have fbs data
        if( fbs_data_count == 0 ):
            return GraphCategoricalForwardBackward.multiplyTerms( terms )

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

            ax = [ i for i, s in enumerate( term.shape ) if s != 1 ]

            for _ in range( ndim - term.ndim ):
                term = term[ ..., None ]

            # Build a meshgrid to correspond to the final shape and repeat
            # over axes that aren't in ax
            reps = np.copy( shape )
            reps[ np.array( ax ) ] = 1

            ans += np.tile( term, reps )

        return fbsData( ans, max_fbs_axis )

    ######################################################################

    @classmethod
    def integrate( cls, integrand, axes ):

        # Check if we need to use the regular integrate
        if( not isinstance( integrand, fbsData ) ):
            return GraphCategoricalForwardBackward.integrate( integrand, axes )

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
        initial_dist = fbsData( self.pi0, -1 )
        emission = self.emissionProb( node, is_partial_graph_index=True )
        return self.multiplyTerms( terms=( emission, initial_dist ) )

    def vBaseCase( self, node ):
        return fbsData( np.zeros( self.K ), -1 )

