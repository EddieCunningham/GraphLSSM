from GenModels.GM.States.GraphicalMessagePassing.GraphicalMessagePassingBase import *
# import autograd.numpy as np
import autograd.numpy as np
from scipy.sparse import coo_matrix
from functools import partial
from collections import Iterable
import itertools
from GenModels.GM.Utility import fbsData

__all__ = [ 'GraphFilter', 'GraphFilterFBS' ]

######################################################################

class _filterMixin():
    # Base message passing class for hyper graphs.
    # Will use a sparse matrix to hold graph structure

    def genFilterProbs( self ):
        assert 0

    def transitionProb( self, children, parents ):
        assert 0

    def emissionProb( self, nodes, forward=False ):
        assert 0

    def multiplyTerms( self, N, terms ):
        assert 0

    def integrate( self, integrand, axes=None ):
        assert 0

    ######################################################################

    # def uBaseCase( self, roots, U ):
    #     assert 0

    # def vBaseCase( self, leaves, V ):
    #     assert 0

    ######################################################################

    def updateU( self, nodes, new_u, U ):
        assert 0

    def updateV( self, nodes, edges, newV, V ):
        assert 0

    ######################################################################

    def vDataFromMask( self, V, mask ):
        _, _, V_data = V
        ans = []
        for i, mask_value in enumerate( mask ):
            if( mask_value == True ):
                ans.append( V_data[ i ] )
        return ans

    ######################################################################

    @classmethod
    def extendAxes( cls, node, term, target_axis, max_dim ):
        # Push the first axis out to target_axis, but don't change
        # the axes past max_dim

        # Add axes before the fbsAxes
        for _ in range( max_dim - target_axis - 1 ):
            term = np.expand_dims( term, 1 )

        # Prepend axes
        for _ in range( target_axis ):
            term = np.expand_dims( term, 0 )

        return term

    ######################################################################

    def uData( self, U, V, node ):
        # THESE MUST NOT MODIFY U OR V!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # VERY IMPORTANT!!!!!! PROBABLY ENFORCE THIS SOMEHOW IN THE FURURE
        return U[ node ]

    def vData( self, U, V, node, edges=None, ndim=None ):
        # THESE MUST NOT MODIFY U OR V!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # VERY IMPORTANT!!!!!! PROBABLY ENFORCE THIS SOMEHOW IN THE FURURE
        V_row, V_col, V_data = V

        if( ~np.any( np.in1d( V_row, node ) ) ):
            # If the node isn't in the V object (node is a leaf)
            ans = [ np.array( [] ) ]
        elif( edges is None ):
            # Return the data over all down edges
            ans = self.vDataFromMask( V, np.in1d( V_row, node ) )
        elif( isinstance( edges, Iterable ) and len( edges ) > 0 ):
            # Return over only certain edges
            mask = np.in1d( V_row, node )
            for e in edges:
                mask &= np.in1d( V_col, e )
            ans = self.vDataFromMask( V, mask )
        elif( isinstance( edges, Iterable ) and len( edges ) == 0 ):
            # This happens when we're not passed a down edge.  In this
            # case, we should return an empty v value, not a leaf v value
            return [ np.array( [] ) ]
        else:
            # Only looking at one edge
            ans = self.vDataFromMask( V, np.in1d( V_col, edges ) )

        if( len( ans ) == 0 ):
            # If we're looking at a leaf, return all 0s
            nVals = 1 if edges is None else len( edges )
            ans = []
            for _ in range( nVals ):
                ans.append( np.array( [] ) )

        assert sum( [ 0 if ~np.any( np.isnan( v ) ) else 1 for v in ans ] ) == 0, ans

        return ans

    ######################################################################

    def a( self, U, V, node, down_edge ):
        # Compute P( Y \ !( e, n )_y, n_x )
        #
        # Probability of all emissions that can be reached without going down down_edge from node.
        #
        # P( Y \ !( e, n )_y, n_x ) = U( n_x ) * prod( [ V( n_x, e ) for e in down_edges( n ) if e is not down_edge ] )
        #
        # U( n_x )
        #   - Probability of all emissions up from node, emission of node, and latent state of node
        # prod( [ V( n_x, e ) for e in down_edges( n ) if e is not down_edge ] )
        #   - Probability of all emissions down from node (but not down down_edge) given latent state of node
        #
        # Inputs:
        #   - U, V     : ans arrays
        #   - node     : n
        #   - down_edge : e
        #
        # Outputs:
        #   - term     : shape should be ( K, ) * ( F + 1 ) where K is the latent state size and F is the size of the fbs

        if( isinstance( down_edge, Iterable ) ):
            assert len( down_edge ) > 0

        # U over node
        # V over node for each down edge that isn't down_edge
        # Multiply U and all Vs
        u = self.uData( U, V, node )

        down_edges = self.getDownEdges( node, skip_edges=down_edge )
        vs = self.vData( U, V, node, edges=down_edges )

        ans = self.multiplyTerms( terms=( u, *vs ) )
        return ans

    ######################################################################

    def b( self, U, V, node ):
        # Compute P( n_y, Y \ ↑( n )_y | ↑( n )_x )
        #
        # Probability of all emissions that can be reached without going up node's up_edge
        # conditioned on the latent state of node's parents
        #
        # P( n_y, Y \ ↑( n )_y | ↑( n )_x )
        #   = integral over node latent states{ P( n_x | parents( n )_x ) * P( n_y | n_x ) * prod( [ V( n_x, e ) for e in down_edges( n ) ] ) }
        #
        # Integrate over node's latent states n_x:
        # P( n_x | parents( n )_x )
        #   - Transition probability from parent's latent states to node's latent state
        # P( n_y | n_x )
        #   - Emission probability of node's emission given node's latent state
        # prod( [ V( n_x, e ) for e in down_edges( n ) ] )
        #   - Probability of all emissions down from node given latent state of node
        #
        # Return array should be of size ( K, ) *  N where K is the latent state size
        # and N is the number of parents.

        n_parents = self.nParents( node )

        # Transition prob to node
        # Emission prob of node (aligned on last axis)
        node_transition = self.transitionProb( node )
        node_emission = self.emissionProb( node )
        node_emission = self.extendAxes( node, node_emission, n_parents, n_parents + 1 )

        # V over node for all down edges (aligned on last axis)
        vs = self.vData( U, V, node )
        vs = [ self.extendAxes( node, v, n_parents, n_parents + 1 ) for v in vs if v.size > 0 ]

        # Multiply together the transition, emission and Vs
        # Integrate over node's (last) axis unless node is in the fbs
        # Integrate over the node's latent states which is last axis
        integrand = self.multiplyTerms( terms=( node_transition, node_emission, *vs ) )
        ans = self.integrate( integrand, axes=[ n_parents ] )

        return ans

    ######################################################################

    def u( self, U, V, node ):
        # Compute P( ↑( n )_y, n_x )
        #
        # Joint probability of all emissions that can be reached by going up node's
        # up edge and node's latent state
        #
        # P( ↑( n )_y, n_x )
        #   = integral over node's parents latent states{ P( n_x | parents( n )_x ) * P( n_y | n_x )
        #                                     * prod( [ a( n_p_x, node's up_edge ) for n_p in parents( node ) ] )
        #                                     * prod( [ b( n_s, parents( node )_x ) for n_s in siblings( node ) ] ) }
        #
        # Integrate over node's parents' latent states [ n_p_x for n_p in parents( node ) ]
        # P( n_x | parents( n )_x )
        #   - Transition probability from parent's latent states to node's latent state
        # P( n_y | n_x )
        #   - Emission probability of node's emission given node's latent state
        # prod( [ a( n_p_x, node's up_edge ) for n_p in parents( node ) ] )
        #   - Probability of all emissions that can be reached by all branches from parents except
        #     this node's up_edge
        # prod( [ b( n_s, parents( node )_x ) for n_s in siblings( node ) ] )
        #   - Probability of all emissions that can be reached down every siblings down branches
        #
        # Return array should be of size ( K, ) where K is the latent state size

        # Don't use full parents here
        parents, parent_order = self.getParents( node, get_order=True )
        n_parents = self.nParents( node )

        # Use full siblings here so that we can use transition information
        siblings = self.getSiblings( node )

        # Transition prob to node
        node_transition = self.transitionProb( node )

        # A over each parent (aligned according to ordering of parents)
        parent_as = [ self.a( U, V, p, self.getUpEdges( node ) ) for p in parents ]
        parent_as = [ self.extendAxes( p, a, i, n_parents ) for p, a, i in zip( parents, parent_as, parent_order ) if a.size > 0 ]

        # B over each sibling
        sigling_bs = [ self.b( U, V, s ) for s in siblings ]

        # Multiply all of the terms together
        # Integrate out the parent latent states
        integrand = self.multiplyTerms( terms=( node_transition, *parent_as, *sigling_bs ) )

        node_terms = self.integrate( integrand, axes=parent_order )

        # Squeeze all of the left most dims.  This is because if a parent is in
        # the fbs and we integrate out the other one, there will be an empty dim
        # remaining
        while( node_terms.shape[ 0 ] == 1 ):
            node_terms = node_terms.squeeze( axis=0 )

        # Emission for this node
        node_emission = self.emissionProb( node )

        # Combine this nodes emission with the rest of the calculation
        ans = self.multiplyTerms( terms=( node_terms, node_emission ) )
        return ans

    ######################################################################

    def v( self, U, V, node, edge ):
        # Compute P( !( n, e )_y | n_x )
        #
        # Probability of all emissions reached by going down edge, conditioned on node's latent state
        #
        # P( !( n, e )_y | n_x )
        #   = integral over node's mates latent states{
        #                                       prod( [ a( n_m_x, edge ) for n_m in mates( node ) ] )
        #                                     * prod( [ b( n_c, parents( n_s )_x ) for n_c in children( node ) ] ) }
        #
        # Integrate over node's mates' latent states [ n_m_x for n_m in mates( node ) ]
        # prod( [ a( n_m_x, node's up_edge ) for n_m in mates( node ) ] )
        #   - Probability of all emissions that can be reached by all branches from mates except edge
        # prod( [ b( n_c, parents( n_c )_x ) for n_c in children( node ) ] )
        #   - Probability of all emissions that can be reached down every child's down branches
        #
        # Return array should be of size ( K, ) where K is the latent state size

        # Use the full children here because if a child is in the fbs, we should use
        # transition information
        children = self.getChildren( node, edges=edge )

        # Don't get the full mates here
        mates, mate_order = self.getMates( node, get_order=True, edges=edge )
        n_parents = self.nParents( children[ 0 ] )

        # A values over each mate (aligned according to ordering of mates)
        mate_as = [ self.a( U, V, m, edge ) for m in mates ]
        mate_as = [ self.extendAxes( m, a, i, n_parents ) for m, a, i in zip( mates, mate_as, mate_order ) if a.size > 0  ]

        # B over each child
        child_bs = [ self.b( U, V, c ) for c in children ]

        # Multiply all of the terms together
        integrand = self.multiplyTerms( terms=( *child_bs, *mate_as ) )

        # Integrate out the mates latent states
        ans = self.integrate( integrand, axes=mate_order )

        # Squeeze all of the left most dims.  This is because if a parent is in
        # the fbs and we integrate out the other one, there will be an empty dim
        # remaining
        while( ans.shape[ 0 ] == 1 ):
            ans = ans.squeeze( axis=0 )

        return ans

    ######################################################################

    def uBaseCase( self, node ):
        initial_dist = self.initialProb( node )
        emission = self.emissionProb( node )
        return self.multiplyTerms( terms=( emission, initial_dist ) )

    def vBaseCase( self, node ):
        return np.array( [] )

    ######################################################################

    def uFilter( self, is_base_case, nodes, U, V, parallel=False ):
        # Compute P( ↑( n )_y, n_x )
        # Probability of all emissions that can be reached by going up node's up edge

        assert parallel != True, 'Call the parallel version from the child class'

        new_u = []
        for node in nodes:
            if( is_base_case ):
                u = self.uBaseCase( node )
            else:
                u = self.u( U, V, node )
            new_u.append( u )

        self.updateU( nodes, new_u, U )

    def vFilter( self, is_base_case, nodes_and_edges, U, V, parallel=False ):

        nodes, edges = nodes_and_edges

        assert parallel != True, 'Call the parallel version from the child class'

        new_v = []
        for node, edge in zip( nodes, edges ):
            if( is_base_case ):
                assert edge == None
                v = self.vBaseCase( node )
            else:
                assert edge is not None
                v = self.v( U, V, node, edge )
            new_v.append( v )

        self.updateV( nodes, edges, new_v, V )

    ######################################################################

    def filter( self, parallel=False ):

        U, V = self.genFilterProbs()

        # Run the message passing algorithm over the graph.
        # We will end up with the extended U and V values for all of
        # the nodes except the feedback set nodes.
        # To get the filtered probs for the smoothed probs for the
        # feedback set nodes, marginalize over the non fbs nodes
        # at any extended smoothed prob
        self.upDown( partial( self.uFilter, parallel=parallel ), partial( self.vFilter, parallel=parallel ), U=U, V=V )

        return U, V

    ######################################################################

    def nodeJointSingleNode( self, U, V, node ):
        # P( x, Y )

        u = self.uData( U, V, node )
        vs = self.vData( U, V, node )
        vs = [ self.extendAxes( node, v, 0, 1 ) for v in vs ]

        full_joint = self.multiplyTerms( terms=( u, *vs ) )
        return full_joint

    ######################################################################

    def jointParentsSingleNode( self, U, V, node ):
        # P( x_p1..pN, Y )

        parents, parent_order = self.getParents( node, get_order=True )
        n_parents = self.nParents( node )
        siblings = self.getSiblings( node )
        up_edge = self.getUpEdges( node )

        # Down each child
        sigling_bs = [ self.b( U, V, s ) for s in siblings ]

        # Down this node
        node_term = self.b( U, V, node )

        # Out from each parent
        parent_as = [ self.a( U, V, p, up_edge ) for p in parents ]
        parent_as = [ self.extendAxes( p, a, i, n_parents ) for p, a, i in zip( parents, parent_as, parent_order ) ]

        joint = self.multiplyTerms( terms=( node_term, *parent_as, *sigling_bs ) )
        return joint

    ######################################################################

    def jointParentChildSingleNode( self, U, V, node ):
        # P( x_c, x_p1..pN, Y )

        siblings = self.getSiblings( node )
        parents, parent_order = self.getParents( node, get_order=True )
        up_edge = self.getUpEdges( node )
        n_parents = self.nParents( node )

        # Down to this node
        node_transition = self.transitionProb( node )
        node_emission = self.emissionProb( node )
        # Align this node's values on the last axis
        node_emission = self.extendAxes( node, node_emission, n_parents, n_parents + 1 )

        # Out from each sibling
        sigling_bs = [ self.b( U, V, s ) for s in siblings ]

        # Out from each parent
        parent_as = [ self.a( U, V, p, up_edge ) for p in parents ]
        parent_as = [ self.extendAxes( p, a, i, n_parents ) for p, a, i in zip( parents, parent_as, parent_order ) ]

        # Down this node
        vs = self.vData( U, V, node )
        vs = [ self.extendAxes( node, _v, n_parents, n_parents + 1 ) for _v in vs ]

        return self.multiplyTerms( terms=( *parent_as, *sigling_bs, *vs, node_transition, node_emission ) )

    ######################################################################

    def nodeJoint( self, U, V, nodes ):
        # P( x, Y )
        return [ ( node, self.nodeJointSingleNode( U, V, node ) ) for node in nodes ]

    def jointParents( self, U, V, nodes ):
        # P( x_p1..pN, Y )
        return [ ( node, self.jointParentsSingleNode( U, V, node ) ) for node in nodes if self.nParents( node ) > 0 ]

    def jointParentChild( self, U, V, nodes ):
        # P( x_c, x_p1..pN, Y )
        return [ ( node, self.jointParentChildSingleNode( U, V, node ) ) for node in nodes if self.nParents( node ) > 0 ]

    ######################################################################

    def marginalProb( self, U, V, node=None ):
        # P( Y )
        if( node is None ):
            marginal = 0.0
            for node in self.parent_graph_assignments:
                joint = self.nodeJointSingleNode( U, V, node )
                marginal += self.integrate( joint, axes=range( joint.ndim ) )
            return marginal
        joint = self.nodeJointSingleNode( U, V, node )
        return self.integrate( joint, axes=range( joint.ndim ) )

    def nodeSmoothed( self, U, V, nodes ):
        # P( x | Y )
        return [ ( node, val - self.marginalProb( U, V, node=node ) ) for node, val in self.nodeJoint( U, V, nodes ) ]

    def parentsSmoothed( self, U, V, nodes ):
        # P( x_p1..pN | Y )
        return [ ( node, val - self.marginalProb( U, V, node=node ) ) for node, val in self.jointParents( U, V, nodes ) ]

    def parentChildSmoothed( self, U, V, nodes ):
        # P( x_c, x_p1..pN | Y )
        return [ ( node, val - self.marginalProb( U, V, node=node ) ) for node, val in self.jointParentChild( U, V, nodes ) ]

    def conditionalParentChild( self, U, V, nodes ):
        # P( x_c | x_p1..pN, Y )
        ans = []
        for node in nodes:
            if( self.nParents( node ) > 0 ):

                jpc = self.jointParentChildSingleNode( U, V, node )
                jp = self.jointParentsSingleNode( U, V, node )
                _ans = self.multiplyTerms( terms=( jpc, -jp ) )

                ans.append( ( node, _ans ) )

        return ans

######################################################################
######################################################################

class __FBSFilterMixin():

    # The most important difference is use the full graph for down edge operations
    # and the partial for the up edge ones.

    ######################################################################

    def vDataFromMask( self, V, mask ):
        _, _, V_data = V
        ans = []

        indices = np.where( mask )[ 0 ]
        for i in indices:
            ans.append( V_data[ i ] )

        return ans

    ######################################################################

    def extendAxes( self, node, term, target_axis, max_dim ):
        # Push the first axis out to target_axis, but don't change
        # the axes past max_dim
        term, fbs_axis = ( term.data, term.fbs_axis )

        # Add axes before the fbsAxes
        for _ in range( max_dim - target_axis - 1 ):
            term = np.expand_dims( term, 1 )
            if( fbs_axis != -1 ):
                fbs_axis += 1

        # Prepend axes
        for _ in range( target_axis ):
            term = np.expand_dims( term, 0 )
            if( fbs_axis != -1 ):
                fbs_axis += 1

        return fbsData( term, fbs_axis )

    ######################################################################

    def uData( self, U, V, node ):
        # THESE MUST NOT MODIFY U OR V!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # VERY IMPORTANT!!!!!! PROBABLY ENFORCE THIS SOMEHOW IN THE FURURE
        return U[ node ]

    def vData( self, U, V, node, edges=None ):
        # THESE MUST NOT MODIFY U OR V!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # VERY IMPORTANT!!!!!! PROBABLY ENFORCE THIS SOMEHOW IN THE FURURE
        V_row, V_col, _ = V

        if( self.inFeedbackSet( node, is_partial_graph_index=True ) ):
            # This is going to be empty because by construction,
            # if a node is in the fbs, all the information from
            # going down the down edges can be gained by going
            # up an up edge.
            if( edges is None ):
                return []
            elif( isinstance( edges, Iterable ) ):
                return []
            else:
                return fbsData( np.array( [] ), -1 )

        if( ~np.any( np.in1d( V_row, node ) ) ):
            # If the node isn't in the V object (node is a leaf)
            ans = [ fbsData( np.array( [] ), -1 ) ]
        elif( edges is None ):
            # Return the data over all down edges
            ans = self.vDataFromMask( V, np.in1d( V_row, node ) )
        elif( isinstance( edges, Iterable ) and len( edges ) > 0 ):
            # Return over only certain edges
            mask = np.in1d( V_row, node )
            for e in edges:
                mask &= np.in1d( V_col, e )
            ans = self.vDataFromMask( V, mask )
        elif( isinstance( edges, Iterable ) and len( edges ) == 0 ):
            # This happens when we're not passed a down edge.  In this
            # case, we should return an empty v value, not a leaf v value
            return [ fbsData( np.array( [] ), -1 ) ]
        else:
            # Only looking at one edge
            ans = self.vDataFromMask( V, np.in1d( V_col, edges ) )

        if( len( ans ) == 0 ):
            # If we're looking at a leaf, return all 0s
            nVals = 1 if edges is None else len( edges )
            ans = []
            for _ in range( nVals ):
                ans.append( fbsData( np.array( [] ), -1 ) )

        for v in ans:
            data = v.data
            assert np.any( np.isnan( data ) ) == False, data

        return ans

    ######################################################################

    def a( self, U, V, node, down_edge ):

        if( isinstance( down_edge, Iterable ) ):
            assert len( down_edge ) > 0

        u = self.uData( U, V, node )

        down_edges = self.getDownEdges( node, skip_edges=down_edge,
                                              is_partial_graph_index=True,
                                              use_partial_graph=True )
        vs = self.vData( U, V, node, edges=down_edges )

        ans = self.multiplyTerms( terms=( u, *vs ) )
        return ans

    ######################################################################

    def b( self, U, V, node ):

        # For a fbs node, this is computed as P( node_y | node_x ) * P( node_x | parents_x )
        # because we are calculating the joint with the fbs nodes
        if( self.inFeedbackSet( node, is_partial_graph_index=True ) ):
            node_transition = self.transitionProb( node, is_partial_graph_index=True )
            node_emission = self.emissionProb( node, is_partial_graph_index=True )
            return self.multiplyTerms( terms=( node_transition, node_emission ) )

        n_parents = self.nParents( node, is_partial_graph_index=True,
                                         use_partial_graph=False )

        # Transition prob to node
        # Emission prob of node (aligned on last axis)
        node_transition = self.transitionProb( node, is_partial_graph_index=True )
        node_emission = self.emissionProb( node, is_partial_graph_index=True )
        node_emission = self.extendAxes( node, node_emission, n_parents, n_parents + 1 )

        # V over node for all down edges (aligned on last axis)
        vs = self.vData( U, V, node )
        vs = [ self.extendAxes( node, v, n_parents, n_parents + 1 ) for v in vs if v.size > 0 ]

        # Multiply together the transition, emission and Vs
        # Integrate over node's (last) axis unless node is in the fbs
        # Integrate over the node's latent states which is last axis
        integrand = self.multiplyTerms( terms=( node_transition, node_emission, *vs ) )
        ans = self.integrate( integrand, axes=[ n_parents ] )

        return ans

    ######################################################################

    def u( self, U, V, node ):

        # Don't use full parents here
        parents, parent_order = self.getPartialParents( node, get_order=True,
                                                              is_partial_graph_index=True,
                                                              return_partial_graph_index=True )
        n_parents = self.nParents( node, is_partial_graph_index=True,
                                         use_partial_graph=False )

        # Use full siblings here so that we can use transition information
        siblings = self.getFullSiblings( node, is_partial_graph_index=True,
                                               return_partial_graph_index=True )

        # Get the up edge on full graph
        up_edge = self.getUpEdges( node, is_partial_graph_index=True,
                                         use_partial_graph=False )

        # Transition prob to node
        node_transition = self.transitionProb( node, is_partial_graph_index=True )

        # A over each parent (aligned according to ordering of parents)
        parent_as = [ self.a( U, V, p, up_edge ) for p in parents ]
        parent_as = [ self.extendAxes( p, a, i, n_parents ) for p, a, i in zip( parents, parent_as, parent_order ) if a.size > 0 ]

        # B over each sibling
        sigling_bs = [ self.b( U, V, s ) for s in siblings ]

        # Multiply all of the terms together
        # Integrate out the parent latent states
        integrand = self.multiplyTerms( terms=( node_transition, *parent_as, *sigling_bs ) )

        node_terms = self.integrate( integrand, axes=parent_order )

        # Squeeze all of the left most dims.  This is because if a parent is in
        # the fbs and we integrate out the other one, there will be an empty dim
        # remaining
        while( node_terms.shape[ 0 ] == 1 ):
            node_terms = node_terms.squeeze( axis=0 )

        # Emission for this node
        node_emission = self.emissionProb( node, is_partial_graph_index=True )

        # Combine this nodes emission with the rest of the calculation
        ans = self.multiplyTerms( terms=( node_terms, node_emission ) )

        return ans

    ######################################################################

    def v( self, U, V, node, edge ):

        # Use the full children here because if a child is in the fbs, we should use
        # transition information
        children = self.getFullChildren( node, edges=edge,
                                               is_partial_graph_index=True,
                                               return_partial_graph_index=True )

        # Don't get the full mates here
        mates, mate_order = self.getPartialMates( node, get_order=True,
                                                        edges=edge,
                                                        is_partial_graph_index=True,
                                                        return_partial_graph_index=True )
        n_parents = self.nParents( children[ 0 ], is_partial_graph_index=True,
                                                  use_partial_graph=False )

        # A values over each mate (aligned according to ordering of mates)
        mate_as = [ self.a( U, V, m, edge ) for m in mates ]
        mate_as = [ self.extendAxes( m, a, i, n_parents ) for m, a, i in zip( mates, mate_as, mate_order ) if a.size > 0  ]

        # B over each child
        child_bs = [ self.b( U, V, c ) for c in children ]

        # Multiply all of the terms together
        integrand = self.multiplyTerms( terms=( *child_bs, *mate_as ) )

        # Integrate out the mates latent states
        ans = self.integrate( integrand, axes=mate_order )

        # Squeeze all of the left most dims.  This is because if a parent is in
        # the fbs and we integrate out the other one, there will be an empty dim
        # remaining
        while( ans.shape[ 0 ] == 1 ):
            ans = ans.squeeze( axis=0 )

        assert np.any( np.isnan( ans.data ) ) == False, ans.data
        return ans

    ######################################################################

    def uBaseCase( self, node ):
        initial_dist = self.initialProb( node, is_partial_graph_index=True )
        emission = self.emissionProb( node, is_partial_graph_index=True )
        return self.multiplyTerms( terms=( emission, initial_dist ) )

    def vBaseCase( self, node ):
        return fbsData( np.array( [] ), -1 )

    ######################################################################

    def uFilter( self, is_base_case, nodes, U, V, parallel=False ):
        # Compute P( ↑( n )_y, n_x )
        # Probability of all emissions that can be reached by going up node's up edge
        assert parallel != True, 'Call the parallel version from the child class'

        new_u = []
        for node in nodes:
            if( is_base_case ):
                u = self.uBaseCase( node )
            else:
                u = self.u( U, V, node )
            new_u.append( u )

        self.updateU( nodes, new_u, U )

    def vFilter( self, is_base_case, nodes_and_edges, U, V, parallel=False ):
        assert parallel != True, 'Call the parallel version from the child class'

        nodes, edges = nodes_and_edges

        new_v = []
        for node, edge in zip( nodes, edges ):
            if( is_base_case ):
                assert edge == None
                v = self.vBaseCase( node )
            else:
                assert edge is not None
                v = self.v( U, V, node, edge )
            new_v.append( v )

        self.updateV( nodes, edges, new_v, V )

    ######################################################################

    def filter( self, parallel=False ):

        U, V = self.genFilterProbs()

        # Run message passing over the partial graph
        self.partial_graph.upDown( partial( self.uFilter, parallel=parallel ), partial( self.vFilter, parallel=parallel ), U=U, V=V )

        return U, V

    ######################################################################

    def nodeJointSingleNodeComputation( self, U, V, node ):
        # P( x, Y )
        u = self.uData( U, V, node )
        vs = self.vData( U, V, node )
        vs = [ self.extendAxes( node, v, 0, 1 ) for v in vs ]
        full_joint = self.multiplyTerms( terms=( u, *vs ) )
        return full_joint

    ######################################################################

    def jointParentsSingleNodeComputation( self, U, V, node, is_partial_graph_index=True ):
        # P( x_p1..pN, Y )
        n_parents = self.nParents( node, is_partial_graph_index=is_partial_graph_index, use_partial_graph=False )
        joint_with_child = self.jointParentChildSingleNodeComputation( U, V, node, is_partial_graph_index=is_partial_graph_index )
        return self.integrate( joint_with_child, axes=[ n_parents ] )

    ######################################################################

    def jointParentChildSingleNodeComputation( self, U, V, node, is_partial_graph_index=True ):
        # P( x_c, x_p1..pN, Y )
        parents, parent_order = self.getPartialParents( node, get_order=True, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=True )
        n_parents = self.nParents( node, is_partial_graph_index=is_partial_graph_index, use_partial_graph=False )
        siblings = self.getFullSiblings( node, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=True )
        up_edge = self.getUpEdges( node, is_partial_graph_index=is_partial_graph_index, use_partial_graph=False )

        # Down to this node
        node_transition = self.transitionProb( node, is_partial_graph_index=is_partial_graph_index )
        node_emission = self.emissionProb( node, is_partial_graph_index=is_partial_graph_index )

        # Align this node's values on the last axis
        node_emission = self.extendAxes( node, node_emission, n_parents, n_parents + 1 )

        # Out from each sibling
        sigling_bs = [ self.b( U, V, s ) for s in siblings ]

        # Out from each parent
        parent_as = [ self.a( U, V, p, up_edge ) for p in parents ]
        parent_as = [ self.extendAxes( p, a, i, n_parents ) for p, a, i in zip( parents, parent_as, parent_order ) ]

        # Down this node
        vs = self.vData( U, V, node )
        vs = [ self.extendAxes( node, _v, n_parents, n_parents + 1 ) for _v in vs ]

        return self.multiplyTerms( terms=( *parent_as, *sigling_bs, *vs, node_transition, node_emission ) )

    ######################################################################

    def nodeJointSingleNode( self, U, V, node, method='compute', is_partial_graph_index=False ):
        # P( x, Y )
        # Having different methods is more of a sanity check than anything

        not_in_fbs = not self.inFeedbackSet( node, is_partial_graph_index=is_partial_graph_index )

        if( not_in_fbs and method == 'integrate' ):
            # Can't use integrate method if node isn't in the fbs
            method = 'compute'

        # If the fbs node is a leaf, must use the integrate method
        if( not_in_fbs == False and self.nParents( node, is_partial_graph_index=is_partial_graph_index, use_partial_graph=False ) == 0 ):
            method = 'integrate'

        # Also if the fbs node has all fbs node parents, must use integrate method
        if( not_in_fbs == False and len( [ 1 for p in self.getFullParents( node, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=False ) if not self.inFeedbackSet( p, is_partial_graph_index=False ) ] ) == 0 ):
            method = 'integrate'

        if( method == 'compute' ):
            # Need to use the partial graph index from here
            node_partial = self.fullGraphIndexToPartialGraphIndex( node ) if is_partial_graph_index == False else node

            # If the node isn't in the fbs, compute the joint the normal way.
            if( not_in_fbs ):
                u = self.uData( U, V, node_partial )
                vs = self.vData( U, V, node_partial )
                vs = [ self.extendAxes( node_partial, v, 0, 1 ) for v in vs ]
                joint_fbs = self.multiplyTerms( terms=( u, *vs ) )
                int_axes = range( 1, joint_fbs.ndim )
            else:
                # Just computing u for the fbs should, by construction,
                # cover all of the nodes in the graph.  Still need to integrate
                # out other fbs nodes in this case
                joint_fbs = self.u( U, V, node_partial )
                fbs_index = self.fbsIndex( node, is_partial_graph_index=is_partial_graph_index, within_graph=True )
                keep_axis = fbs_index + joint_fbs.fbs_axis
                int_axes = np.setdiff1d( np.arange( joint_fbs.ndim ), keep_axis )
            return self.integrate( joint_fbs, axes=int_axes ).data

        elif( method == 'integrate' ):
            # If node is in the fbs, choose another node to get the joint with the fbs node with.
            # Otherwise, calculate the joint the regular way
            if( not_in_fbs ):
                node_partial = self.fullGraphIndexToPartialGraphIndex( node ) if is_partial_graph_index == False else node
            else:
                node_partial = self._anotherNode( node, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=True )

            # Compute the joint using the regular algorithm but on the partial_graph
            joint_fbs = self.nodeJointSingleNodeComputation( U, V, node_partial )

            if( not_in_fbs ):
                # Integrate out all of the parents and fbs nodes
                int_axes = range( 1, joint_fbs.ndim )
            else:
                # Integrate out all the nodes but this one, which is somewhere in the fbs axes
                fbs_index = self.fbsIndex( node, is_partial_graph_index=is_partial_graph_index, within_graph=True )
                keep_axis = fbs_index + joint_fbs.fbs_axis
                int_axes = np.setdiff1d( np.arange( joint_fbs.ndim ), keep_axis )

            # Integrate the joint and return only the data portion
            return self.integrate( joint_fbs, axes=int_axes ).data
        else:
            assert 0, 'Invalid method'

    ######################################################################

    def jointParentsSingleNode( self, U, V, node, method='compute', is_partial_graph_index=False ):
        # P( x_p1..pN, Y )

        not_in_fbs = not self.inFeedbackSet( node, is_partial_graph_index=is_partial_graph_index )

        if( method == 'integrate' ):
            if( not_in_fbs ):
                node_partial = self.fullGraphIndexToPartialGraphIndex( node ) if is_partial_graph_index == False else node
            else:
                node_partial = self._anotherNode( node, parents=False, mates=False, children=False, siblings=True, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=True )

            # Couldn't find a sibling, so use the compute method
            if( node_partial is None ):
                method = 'compute'

            # Can't use integrate method if node isn't in the fbs
            if( not_in_fbs ):
                method = 'compute'

        if( len( [ 1 for p in self.getFullParents( node, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=False ) if not self.inFeedbackSet( p, is_partial_graph_index=False ) ] ) == 0 ):
            if( not_in_fbs ):
                node_partial = self.fullGraphIndexToPartialGraphIndex( node ) if is_partial_graph_index == False else node
            else:
                # Find any node to run the nodeJoint algorithm on
                node_partial = self._anotherNode( node, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=True )
            joint_fbs = self.nodeJointSingleNodeComputation( U, V, node_partial )

        elif( method == 'compute' ):
            # Regular computation
            node_partial = self.fullGraphIndexToPartialGraphIndex( node ) if is_partial_graph_index == False else node
            joint_fbs = self.jointParentsSingleNodeComputation( U, V, node_partial )

        elif( method == 'integrate' ):
            # Find a sibling to pass
            node_partial = self._anotherNode( node, parents=False, mates=False, children=False, siblings=True, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=True )
            joint_fbs = self.jointParentsSingleNodeComputation( U, V, node_partial )

        else:
            assert 0, 'Invalid method'

        # Loop through the parents and find the axes that they are on so we can keep them
        parents, parent_order = self.getFullParents( node, get_order=True, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=True )
        keep_axes = []
        for p, o in zip( parents, parent_order ):
            if( self.inFeedbackSet( p, is_partial_graph_index=True ) ):
                keep_axes.append( joint_fbs.fbs_axis + self.fbsIndex( p, is_partial_graph_index=True, within_graph=True ) )
            else:
                keep_axes.append( o )
        keep_axes = np.array( keep_axes )

        # Integrate out all fbs nodes that aren't the parents
        int_axes = np.setdiff1d( np.arange( joint_fbs.ndim ), keep_axes )
        ans = self.integrate( joint_fbs, axes=int_axes ).data

        # Swap the order of the axes so that the answer has the correct parent order.
        # At the moment, all of the fbs indices are on the last indices sorted by
        # fbs index
        non_fbs_order = [ o for p, o in zip( parents, parent_order ) if not self.inFeedbackSet( p, is_partial_graph_index=True ) ]
        fbs_parents = [ ( p, o ) for p, o in zip( parents, parent_order ) if self.inFeedbackSet( p, is_partial_graph_index=True ) ]
        fbs_order = sorted( fbs_parents, key=lambda x: self.fbsIndex( x[ 0 ], is_partial_graph_index=True, within_graph=True ) )
        fbs_order = [ o for p, o in fbs_order ]

        true_order = non_fbs_order + fbs_order
        transpose_axes = [ true_order.index( i ) for i in range( len( true_order ) ) ]

        return np.transpose( ans, transpose_axes )

    ######################################################################

    def jointParentChildSingleNode( self, U, V, node, method='compute', is_partial_graph_index=False ):
        # P( x_c, x_p1..pN, Y )

        not_in_fbs = not self.inFeedbackSet( node, is_partial_graph_index=is_partial_graph_index )

        if( method == 'integrate' ):
            # This fbs node has no siblings, so must compute
            if( not_in_fbs ):
                node_partial = self.fullGraphIndexToPartialGraphIndex( node ) if is_partial_graph_index == False else node
            else:
                node_partial = self._anotherNode( node, parents=False, mates=False, children=False, siblings=True, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=True )

            # Couldn't find a sibling, so use the compute method
            if( node_partial is None ):
                method = 'compute'

            # Can't use integrate method if node isn't in the fbs
            if( not_in_fbs ):
                method = 'compute'

        # If all of the parents are in the fbs and so is node, then use nodeJoint and integrate out the correct nodes.
        if( len( [ 1 for p in self.getFullParents( node, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=True ) if not self.inFeedbackSet( p, is_partial_graph_index=True ) ] ) == 0 ):
            if( not_in_fbs ):
                node_partial = self.fullGraphIndexToPartialGraphIndex( node ) if is_partial_graph_index == False else node
            else:
                # Find any node to run the nodeJoint algorithm on
                node_partial = self._anotherNode( node, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=True )
            joint_fbs = self.nodeJointSingleNodeComputation( U, V, node_partial )

        elif( method == 'compute' ):
            node_partial = self.fullGraphIndexToPartialGraphIndex( node ) if is_partial_graph_index == False else node
            joint_fbs = self.jointParentChildSingleNodeComputation( U, V, node_partial )

        elif( method == 'integrate' ):
            if( not_in_fbs ):
                node_partial = self.fullGraphIndexToPartialGraphIndex( node ) if is_partial_graph_index == False else node
            else:
                # Find a sibling to run the algorithm on
                node_partial = self._anotherNode( node, parents=False, mates=False, children=False, siblings=True, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=True )
            joint_fbs = self.jointParentChildSingleNodeComputation( U, V, node_partial )

        else:
            assert 0, 'Invalid method'

        # Loop through the parents and find the axes that they are on so we can keep them
        parents, parent_order = self.getFullParents( node, get_order=True, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=True )
        n_parents = self.nParents( node, is_partial_graph_index=is_partial_graph_index, use_partial_graph=False )
        keep_axes = []
        for p, o in zip( parents, parent_order ):
            if( self.inFeedbackSet( p, is_partial_graph_index=True ) ):
                keep_axes.append( joint_fbs.fbs_axis + self.fbsIndex( p, is_partial_graph_index=True, within_graph=True ) )
            else:
                keep_axes.append( o )

        # Keep the axis that node is on
        if( not_in_fbs ):
            n_fbs_parents = len( [ 1 for p in self.getFullParents( node, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=True ) if self.inFeedbackSet( p, is_partial_graph_index=True ) ] )
            if( n_fbs_parents == n_parents ):
                keep_axes.append( 0 )
            else:
                keep_axes.append( n_parents )
        else:
            keep_axes.append( joint_fbs.fbs_axis + self.fbsIndex( node, is_partial_graph_index=is_partial_graph_index, within_graph=True ) )
        keep_axes = np.array( keep_axes )

        # Integrate out all fbs nodes that aren't the parents
        int_axes = np.setdiff1d( np.arange( joint_fbs.ndim ), keep_axes )
        ans = self.integrate( joint_fbs, axes=int_axes ).data

        # Swap the order of the axes to that the answer has the correct parent order
        # and so that the node is on the last axis.  To do this, first find the true
        # axes that each of the current axes correspond to
        non_fbs_order = [ o for p, o in zip( parents, parent_order ) if not self.inFeedbackSet( p, is_partial_graph_index=True ) ]
        fbs_parents = [ ( p, o ) for p, o in zip( parents, parent_order ) if self.inFeedbackSet( p, is_partial_graph_index=True ) ]
        fbs_order = sorted( fbs_parents, key=lambda x: self.fbsIndex( x[ 0 ], is_partial_graph_index=True, within_graph=True ) )
        fbs_order = [ o for p, o in fbs_order ]

        # If node is in the fbs, then that means that its current axis is somewhere
        # after the fbs_axis.  So place the node's axis correctly
        if( self.inFeedbackSet( node, is_partial_graph_index=is_partial_graph_index ) ):
            insertion_index = self.fbsIndex( node, is_partial_graph_index=is_partial_graph_index, within_graph=True )
            true_order = non_fbs_order + fbs_order[ :insertion_index ] + [ len( parents ) ] + fbs_order[ insertion_index: ]
        else:
            true_order = non_fbs_order + [ len( parents ) ] + fbs_order

        # At the moment, true_order corresonds to the axis that each of the current
        # axes should map to.  np.transpose expects the opposite of this
        transpose_axes = [ true_order.index( i ) for i in range( len( true_order ) ) ]

        return np.transpose( ans, transpose_axes )

    ######################################################################

    def nodeJoint( self, U, V, nodes ):
        # P( x, Y )
        return [ ( node, self.nodeJointSingleNode( U, V, node ) ) for node in nodes ]

    def jointParents( self, U, V, nodes ):
        # P( x_p1..pN, Y )
        return [ ( node, self.jointParentsSingleNode( U, V, node, is_partial_graph_index=False ) ) for node in nodes if self.nParents( node, is_partial_graph_index=False, use_partial_graph=False ) > 0 ]

    def jointParentChild( self, U, V, nodes ):
        # P( x_c, x_p1..pN, Y )
        return [ ( node, self.jointParentChildSingleNode( U, V, node, is_partial_graph_index=False ) ) for node in nodes if self.nParents( node, is_partial_graph_index=False, use_partial_graph=False ) > 0 ]

    ######################################################################

    def marginalProb( self, U, V, node=None ):
        # P( Y )
        if( node is None ):
            marginal = 0.0
            for node in self.full_graph.parent_graph_assignments:
                joint = self.nodeJointSingleNode( U, V, node )
                marginal += self.integrate( joint, axes=range( joint.ndim ) )
            return marginal

        joint = self.nodeJointSingleNode( U, V, node )
        return self.integrate( joint, axes=range( joint.ndim ) )

    def nodeSmoothed( self, U, V, nodes ):
        # P( x | Y )
        return [ ( node, val - self.marginalProb( U, V, node=node ) ) for node, val in self.nodeJoint( U, V, nodes ) ]

    def parentsSmoothed( self, U, V, nodes ):
        # P( x_p1..pN | Y )
        return [ ( node, val - self.marginalProb( U, V, node=node ) ) for node, val in self.jointParents( U, V, nodes ) ]

    def parentChildSmoothed( self, U, V, nodes ):
        # P( x_c, x_p1..pN | Y )
        return [ ( node, val - self.marginalProb( U, V, node=node ) ) for node, val in self.jointParentChild( U, V, nodes ) ]

    def conditionalParentChild( self, U, V, nodes ):
        # P( x_c | x_p1..pN, Y )
        ans = []
        for node in nodes:
            if( self.nParents( node, is_partial_graph_index=False, use_partial_graph=False ) > 0 ):

                jpc = self.jointParentChildSingleNode( U, V, node, is_partial_graph_index=False )
                jp = self.jointParentsSingleNode( U, V, node, is_partial_graph_index=False )
                _ans = self.multiplyTerms( terms=( jpc, -jp ) )
                ans.append( ( node, _ans ) )
            else:
                smoothed = self.nodeSmoothed( U, V, [ node ] )[ 0 ][ 1 ]
                ans.append( ( node, smoothed ) )

        return ans

######################################################################

class GraphFilter( _filterMixin, GraphMessagePasser ):
    pass

class GraphFilterFBS( __FBSFilterMixin, GraphMessagePasserFBS ):
    pass

######################################################################
