from GenModels.GM.States.GraphicalMessagePassing.GraphicalMessagePassingBase import *
import numpy as np
from scipy.sparse import coo_matrix
from functools import reduce
from collections import Iterable
import itertools
import inspect
from GenModels.GM.Utility import fbsData

__all__ = [ 'GraphFilter', 'GraphFilterFBS' ]

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

    def uBaseCase( self, roots, U ):
        assert 0

    def vBaseCase( self, leaves, V ):
        assert 0

    ######################################################################

    def updateU( self, nodes, newU, U ):
        assert 0

    def updateV( self, nodes, edges, newV, V ):
        assert 0

    ######################################################################

    def assignV( self, V, node, val, keepShape=False ):
        V_row, V_col, V_data = V
        N = V_row.shape[ 0 ]
        VIndices = np.arange( N )[ np.in1d( V_row, node ) ].astype( np.int )
        for i in VIndices:
            if( keepShape is False ):
                V_data[ i ] = val
            else:
                V_data[ i ][ : ] = val

    def vDataFromMask( self, V, mask ):
        _, _, V_data = V
        ans = []
        for i, maskValue in enumerate( mask ):
            if( maskValue == True ):
                ans.append( V_data[ i ] )
        return ans

    ######################################################################

    @classmethod
    def extendAxes( cls, node, term, targetAxis, maxDim ):
        # Push the first axis out to targetAxis, but don't change
        # the axes past maxDim

        # Add axes before the fbsAxes
        for _ in range( maxDim - targetAxis - 1 ):
            term = np.expand_dims( term, 1 )

        # Prepend axes
        for _ in range( targetAxis ):
            term = np.expand_dims( term, 0 )

        return term

    ######################################################################

    def nParents( self, node, full=True ):
        if( full == True ):
            return self.full_parents( node ).shape[ 0 ]
        return self.parents( node ).shape[ 0 ]

    def nSiblings( self, node, full=True ):
        if( full == True ):
            return self.full_siblings( node ).shape[ 0 ]
        return self.siblings( node ).shape[ 0 ]

    def nMates( self, node, edges=None, full=True ):
        if( full == True ):
            return self.full_mates( node, edges=edges ).shape[ 0 ]
        return self.mates( node, edges=edges ).shape[ 0 ]

    def nChildren( self, node, edges=None, full=True ):
        if( full == True ):
            return self.full_children( node ).shape[ 0 ]
        return self.children( node ).shape[ 0 ]

    ######################################################################

    def uData( self, U, V, node ):
        return U[ node ]
        # if( len( U ) <= node ):
        #     return self.u( U, V, node )
        # else:
        #     return U[ node ]

    def vData( self, U, V, node, edges=None, ndim=None, debug=False ):
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
                ans.append( np.zeros( self.K ) )

        assert sum( [ 0 if ~np.any( np.isnan( v ) ) else 1 for v in ans ] ) == 0, ans

        return ans

    ######################################################################

    def a( self, U, V, node, downEdge, forceComputeU=False, forceComputeV=False ):
        # Compute P( Y \ !( e, n )_y, n_x )
        #
        # Probability of all emissions that can be reached without going down downEdge from node.
        #
        # P( Y \ !( e, n )_y, n_x ) = U( n_x ) * prod( [ V( n_x, e ) for e in downEdges( n ) if e is not downEdge ] )
        #
        # U( n_x )
        #   - Probability of all emissions up from node, emission of node, and latent state of node
        # prod( [ V( n_x, e ) for e in downEdges( n ) if e is not downEdge ] )
        #   - Probability of all emissions down from node (but not down downEdge) given latent state of node
        #
        # Inputs:
        #   - U, V     : ans arrays
        #   - node     : n
        #   - downEdge : e
        #
        # Outputs:
        #   - term     : shape should be ( K, ) * ( F + 1 ) where K is the latent state size and F is the size of the fbs

        if( isinstance( downEdge, Iterable ) ):
            assert len( downEdge ) > 0

        # U over node
        # V over node for each down edge that isn't downEdge
        # Multiply U and all Vs
        u = self.uData( U, V, node ) if forceComputeU == False else self.u( U, V, node )

        downEdges = self.downEdges( node, skipEdges=downEdge )
        if( forceComputeV == False ):
            vs = self.vData( U, V, node, edges=downEdges )
        else:
            vs = [ self.v( U, V, node, edge ) for edge in downEdges ]

        ans = self.multiplyTerms( terms=( u, *vs ) )
        return ans

    ######################################################################

    def b( self, U, V, node, forceComputeV=False ):
        # Compute P( n_y, Y \ ↑( n )_y | ↑( n )_x )
        #
        # Probability of all emissions that can be reached without going up node's upEdge
        # conditioned on the latent state of node's parents
        #
        # P( n_y, Y \ ↑( n )_y | ↑( n )_x )
        #   = integral over node latent states{ P( n_x | parents( n )_x ) * P( n_y | n_x ) * prod( [ V( n_x, e ) for e in downEdges( n ) ] ) }
        #
        # Integrate over node's latent states n_x:
        # P( n_x | parents( n )_x )
        #   - Transition probability from parent's latent states to node's latent state
        # P( n_y | n_x )
        #   - Emission probability of node's emission given node's latent state
        # prod( [ V( n_x, e ) for e in downEdges( n ) ] )
        #   - Probability of all emissions down from node given latent state of node
        #
        # Return array should be of size ( K, ) *  N where K is the latent state size
        # and N is the number of parents.

        nParents = self.nParents( node, full=True )

        # Transition prob to node
        # Emission prob of node (aligned on last axis)
        nodeTransition = self.transitionProb( node )
        nodeEmission = self.emissionProb( node )
        nodeEmission = self.extendAxes( node, nodeEmission, nParents, nParents + 1 )

        # V over node for all down edges (aligned on last axis)
        if( forceComputeV == False ):
            vs = self.vData( U, V, node )
        else:
            downEdges = self.downEdges( node )
            vs = [ self.v( U, V, node, edge ) for edge in downEdges ]

        vs = [ self.extendAxes( node, v, nParents, nParents + 1 ) for v in vs if v.size > 0 ]

        # Multiply together the transition, emission and Vs
        # Integrate over node's (last) axis unless node is in the fbs
        # Integrate over the node's latent states which is last axis
        integrand = self.multiplyTerms( terms=( nodeTransition, nodeEmission, *vs ) )
        ans = self.integrate( integrand, axes=[ nParents ] )

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
        #                                     * prod( [ a( n_p_x, node's upEdge ) for n_p in parents( node ) ] )
        #                                     * prod( [ b( n_s, parents( node )_x ) for n_s in siblings( node ) ] ) }
        #
        # Integrate over node's parents' latent states [ n_p_x for n_p in parents( node ) ]
        # P( n_x | parents( n )_x )
        #   - Transition probability from parent's latent states to node's latent state
        # P( n_y | n_x )
        #   - Emission probability of node's emission given node's latent state
        # prod( [ a( n_p_x, node's upEdge ) for n_p in parents( node ) ] )
        #   - Probability of all emissions that can be reached by all branches from parents except
        #     this node's upEdge
        # prod( [ b( n_s, parents( node )_x ) for n_s in siblings( node ) ] )
        #   - Probability of all emissions that can be reached down every siblings down branches
        #
        # Return array should be of size ( K, ) where K is the latent state size

        # Don't use full parents here
        parents, parentOrder = self.parents( node, getOrder=True )
        nParents = parentOrder.shape[ 0 ]

        # Use full siblings here so that we can use transition information
        siblings = self.full_siblings( node )

        # Transition prob to node
        nodeTransition = self.transitionProb( node )

        # A over each parent (aligned according to ordering of parents)
        parentAs = [ self.a( U, V, p, self.upEdges( node ) ) for p in parents ]
        parentAs = [ self.extendAxes( p, a, i, nParents ) for p, a, i in zip( parents, parentAs, parentOrder ) if a.size > 0 ]

        # B over each sibling
        siblingBs = [ self.b( U, V, s ) for s in siblings ]

        # Multiply all of the terms together
        # Integrate out the parent latent states
        integrand = self.multiplyTerms( terms=( nodeTransition, *parentAs, *siblingBs ) )

        nodeTerms = self.integrate( integrand, axes=parentOrder )

        # Squeeze all of the left most dims.  This is because if a parent is in
        # the fbs and we integrate out the other one, there will be an empty dim
        # remaining
        while( nodeTerms.shape[ 0 ] == 1 ):
            nodeTerms = nodeTerms.squeeze( axis=0 )

        # Emission for this node
        nodeEmission = self.emissionProb( node )

        # Combine this nodes emission with the rest of the calculation
        ans = self.multiplyTerms( terms=( nodeTerms, nodeEmission ) )
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
        # prod( [ a( n_m_x, node's upEdge ) for n_m in mates( node ) ] )
        #   - Probability of all emissions that can be reached by all branches from mates except edge
        # prod( [ b( n_c, parents( n_c )_x ) for n_c in children( node ) ] )
        #   - Probability of all emissions that can be reached down every child's down branches
        #
        # Return array should be of size ( K, ) where K is the latent state size

        # Use the full children here because if a child is in the fbs, we should use
        # transition information
        children = self.full_children( node, edges=edge )

        # Don't get the full mates here
        mates, mateOrder = self.mates( node, getOrder=True, edges=edge )
        nParents = self.nParents( children[ 0 ], full=True )

        # A values over each mate (aligned according to ordering of mates)
        mateAs = [ self.a( U, V, m, edge ) for m in mates ]
        mateAs = [ self.extendAxes( m, a, i, nParents ) for m, a, i in zip( mates, mateAs, mateOrder ) if a.size > 0  ]

        # B over each child
        childBs = [ self.b( U, V, c ) for c in children ]

        # Multiply all of the terms together
        integrand = self.multiplyTerms( terms=( *childBs, *mateAs ) )

        # Integrate out the mates latent states
        ans = self.integrate( integrand, axes=mateOrder )

        # Squeeze all of the left most dims.  This is because if a parent is in
        # the fbs and we integrate out the other one, there will be an empty dim
        # remaining
        while( ans.shape[ 0 ] == 1 ):
            ans = ans.squeeze( axis=0 )

        return ans

    ######################################################################

    def uFilter( self, baseCase, nodes, U, V ):
        # Compute P( ↑( n )_y, n_x )
        # Probability of all emissions that can be reached by going up node's up edge

        newU = []
        for node in nodes:
            if( self.nParents( node, full=False ) == 0 ):
                u = self.uBaseCase( node )
            else:
                u = self.u( U, V, node )
            newU.append( u )

        self.updateU( nodes, newU, U )

    def vFilter( self, baseCase, nodesAndEdges, U, V ):

        nodes, edges = nodesAndEdges

        newV = []
        for node, edge in zip( nodes, edges ):
            if( self.nChildren( node, full=False ) == 0 ):
                assert edge == None
                v = self.vBaseCase( node )
            else:
                assert edge is not None
                v = self.v( U, V, node, edge )
            newV.append( v )

        self.updateV( nodes, edges, newV, V )

    def convergence( self, nodes ):
        return False

    ######################################################################

    def filter( self ):

        U, V = self.genFilterProbs()

        # Run the message passing algorithm over the graph.
        # We will end up with the extended U and V values for all of
        # the nodes except the feedback set nodes.
        # To get the filtered probs for the smoothed probs for the
        # feedback set nodes, marginalize over the non fbs nodes
        # at any extended smoothed prob
        self.messagePassing( self.uFilter, self.vFilter, U=U, V=V )

        return U, V

    ######################################################################

    def _nodeJoint( self, U, V, node, forceComputeU=False, forceComputeV=False ):
        # P( x, Y )

        u = self.uData( U, V, node ) if forceComputeU == False else self.u( U, V, node )

        if( forceComputeV == False ):
            vs = self.vData( U, V, node )
        else:
            downEdges = self.downEdges( node )
            vs = [ self.v( U, V, node, edge ) for edge in downEdges ]

        vs = [ self.extendAxes( node, v, 0, 1 ) for v in vs ]

        fullJoint = self.multiplyTerms( terms=( u, *vs ) )
        return fullJoint

    ######################################################################

    def _jointParents( self, U, V, node ):
        # P( x_p1..pN, Y )

        parents, parentOrder = self.parents( node, getOrder=True )
        nParents = self.nParents( node, full=True )
        siblings = self.full_siblings( node )
        upEdge = self.upEdges( node )

        # Down each child
        siblingBs = [ self.b( U, V, s ) for s in siblings ]

        # Down this node
        nodeTerm = self.b( U, V, node )

        # Out from each parent
        parentAs = [ self.a( U, V, p, upEdge ) for p in parents ]
        parentAs = [ self.extendAxes( p, a, i, nParents ) for p, a, i in zip( parents, parentAs, parentOrder ) ]

        jointParents = self.multiplyTerms( terms=( nodeTerm, *parentAs, *siblingBs ) )
        return jointParents

    ######################################################################

    def _jointParentChild( self, U, V, node ):
        # P( x_c, x_p1..pN, Y )

        siblings = self.full_siblings( node )
        parents, parentOrder = self.parents( node, getOrder=True )
        upEdge = self.upEdges( node )
        nParents = self.nParents( node, full=True )

        # Down to this node
        nodeTransition = self.transitionProb( node )
        nodeEmission = self.emissionProb( node )
        # Align this node's values on the last axis
        nodeEmission = self.extendAxes( node, nodeEmission, nParents, nParents + 1 )

        # Out from each sibling
        siblingBs = [ self.b( U, V, s ) for s in siblings ]

        # Out from each parent
        parentAs = [ self.a( U, V, p, upEdge ) for p in parents ]
        parentAs = [ self.extendAxes( p, a, i, nParents ) for p, a, i in zip( parents, parentAs, parentOrder ) ]

        # Down this node
        vs = self.vData( U, V, node )
        vs = [ self.extendAxes( node, _v, nParents, nParents + 1 ) for _v in vs ]

        fullJoint = self.multiplyTerms( terms=( *parentAs, *siblingBs, *vs, nodeTransition, nodeEmission ) )
        return fullJoint

    ######################################################################

    def nodeJoint( self, U, V, nodes ):
        # P( x, Y )
        return [ ( node, self._nodeJoint( U, V, node ) ) for node in nodes ]

    def jointParents( self, U, V, nodes ):
        # P( x_p1..pN, Y )
        return [ ( node, self._jointParents( U, V, node ) ) for node in nodes if self.nParents( node, full=True ) > 0 ]

    def jointParentChild( self, U, V, nodes ):
        # P( x_c, x_p1..pN, Y )
        return [ ( node, self._jointParentChild( U, V, node ) ) for node in nodes if self.nParents( node, full=True ) > 0 ]

    ######################################################################

    def marginalProb( self, U, V, node ):
        # P( Y )
        joint = self._nodeJoint( U, V, node )
        return self.integrate( joint, axes=range( joint.ndim ) )

    def nodeSmoothed( self, U, V, nodes ):
        # P( x | Y )
        return [ ( node, val - self.marginalProb( U, V, node ) ) for node, val in self.nodeJoint( U, V, nodes ) ]

    def conditionalParentChild( self, U, V, nodes ):
        # P( x_c | x_p1..pN, Y )
        ans = []
        for node in nodes:
            if( self.nParents( node, full=True ) > 0 ):

                jpc = self._jointParentChild( U, V, node )
                jp = self._jointParents( U, V, node )
                _ans = self.multiplyTerms( terms=( jpc, -jp ) )

                ans.append( ( node, _ans ) )

        return ans

######################################################################
######################################################################

class __FBSFilterMixin():

    # The most important difference is use the full graph for down edge operations
    # and the reduced for the up edge ones.

    # VERY IMPORTANT.  DON'T USE THE NUMBER OF PARENTS FROM SELF.PARENTS TO
    # FIND THE NUMBER OF PARENTS A NODE HAS!!! USE self.nParents( node, full=True )!!!

    ######################################################################

    def nParents( self, node, full=True, fullIndexing=False ):
        if( full == True ):
            return self.full_parents( node, fullIndexing=fullIndexing ).shape[ 0 ]
        assert fullIndexing == False
        return self.parents( node ).shape[ 0 ]

    ######################################################################

    def extendAxes( self, node, term, targetAxis, maxDim ):
        # Push the first axis out to targetAxis, but don't change
        # the axes past maxDim
        term, fbsAxisStart = ( term.data, term.fbsAxis )

        # Add axes before the fbsAxes
        for _ in range( maxDim - targetAxis - 1 ):
            term = np.expand_dims( term, 1 )
            if( fbsAxisStart != -1 ):
                fbsAxisStart += 1

        # Prepend axes
        for _ in range( targetAxis ):
            term = np.expand_dims( term, 0 )
            if( fbsAxisStart != -1 ):
                fbsAxisStart += 1

        return fbsData( term, fbsAxisStart )

    ######################################################################

    def vData( self, U, V, node, edges=None, ndim=None, debug=False ):
        V_row, V_col, V_data = V

        if( self.inFBS( node, fromReduced=True ) ):
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
                ans.append( fbsData( np.zeros( self.K ), -1 ) )

        for v in ans:
            data = v.data
            assert np.any( np.isnan( data ) ) == False

        return ans

    ######################################################################

    def b( self, U, V, node ):
        # For a fbs node, this is computed as P( node_y | node_x ) * P( node_x | parents_x )
        # because we are calculating the joint with the fbs nodes
        if( not self.inFBS( node, fromReduced=True ) ):
            return super().b( U, V, node )
        else:
            nodeTransition = self.transitionProb( node )
            nodeEmission = self.emissionProb( node )
            return self.multiplyTerms( terms=( nodeTransition, nodeEmission ) )

    ######################################################################

    def _anotherNode( self, node, parents=True, mates=True, siblings=True, children=True ):
        toUse = []
        if( parents ):
            toUse.append( self.full_parents( node, fullIndexing=True, returnFullIndex=True ) )
        if( mates ):
            toUse.append( self.full_mates( node, fullIndexing=True, returnFullIndex=True ) )
        if( siblings ):
            siblings = self.full_siblings( node, fullIndexing=True, returnFullIndex=True )
            toUse.append( siblings )
        if( children ):
            toUse.append( self.full_children( node, fullIndexing=True, returnFullIndex=True ) )
        possibilities = itertools.chain( *toUse )
        for _node in possibilities:
            if( _node not in self.fbs ):
                return int( _node )
        return None

    ######################################################################

    def _nodeJoint( self, U, V, node, method='compute' ):
        # P( x, Y )
        # Having different methods is more of a sanity check than anything

        # NODE MUST BE FROM THE FULL SET OF NODES!!!!
        notInFBS = not self.inFBS( node, fromReduced=False )

        # If the fbs node is a leaf, must use the integrate method
        if( notInFBS == False and self.full_parents( node, fullIndexing=True ).size == 0 ):
            method = 'integrate'

        # Also if the fbs node has all fbs node parents, must use integrate method
        if( len( [ 1 for p in self.full_parents( node, fullIndexing=True ) if not self.inFBS( p, fromReduced=True ) ] ) == 0 ):
            method = 'integrate'

        if( method == 'compute' ):
            _node = self.fullIndexToReduced( node )

            # If the node isn't in the fbs, compute the joint the normal way.
            if( notInFBS ):
                u = self.uData( U, V, _node )
                vs = self.vData( U, V, _node )
                vs = [ self.extendAxes( _node, v, 0, 1 ) for v in vs ]
                jointFBS = self.multiplyTerms( terms=( u, *vs ) )
                intAxes = range( 1, jointFBS.ndim )
            else:
                # Just computing u for the fbs should, by construction,
                # cover all of the nodes in the graph.  Still need to integrate
                # out other fbs nodes in this case
                jointFBS = self.u( U, V, _node )
                fbsIndex = self.fbsIndex( node, fromReduced=False, withinGraph=True )
                keepAxis = fbsIndex + jointFBS.fbsAxis
                intAxes = np.setdiff1d( np.arange( jointFBS.ndim ), keepAxis )
            return self.integrate( jointFBS, axes=intAxes ).data

        elif( method == 'integrate' ):
            # If node is in the fbs, choose another node to get the joint with the fbs node with.
            # Otherwise, calculate the joint the regular way
            _node = node if notInFBS else self._anotherNode( node )
            jointFBS = super()._nodeJoint( U, V, self.fullIndexToReduced( _node ) )

            if( notInFBS ):
                intAxes = range( 1, jointFBS.ndim )
            else:
                fbsIndex = self.fbsIndex( node, fromReduced=False, withinGraph=True )
                keepAxis = fbsIndex + jointFBS.fbsAxis
                intAxes = np.setdiff1d( np.arange( jointFBS.ndim ), keepAxis )

            return self.integrate( jointFBS, axes=intAxes ).data
        else:
            assert 0, 'Invalid method'

    ######################################################################

    def _jointParents( self, U, V, node, method='compute' ):
        # P( x_p1..pN, Y )

        notInFBS = not self.inFBS( node, fromReduced=False )

        if( method == 'integrate' ):
            # This fbs node has no siblings, so must compute
            _node = node if notInFBS else self._anotherNode( node, parents=False, mates=False, children=False, siblings=True )
            if( _node is None ):
                method = 'compute'

            if( notInFBS ):
                method = 'compute'

        # If all of the parents are in the fbs, then use nodeJoint and integrate out the correct nodes
        if( len( [ 1 for p in self.full_parents( node, fullIndexing=True ) if not self.inFBS( p, fromReduced=True ) ] ) == 0 ):
            _node = node if notInFBS else self._anotherNode( node )
            _node = self.fullIndexToReduced( _node )
            jointFBS = super()._nodeJoint( U, V, _node )

        elif( method == 'compute' ):
            _node = self.fullIndexToReduced( node )
            jointFBS = super()._jointParents( U, V, _node )

        elif( method == 'integrate' ):
            _node = node if notInFBS else self._anotherNode( node, parents=False, mates=False, children=False, siblings=True )
            _node = self.fullIndexToReduced( _node )

            jointFBS = super()._jointParents( U, V, _node )

        else:
            assert 0, 'Invalid method'

        # Integrate out the nodes that aren't parents
        parents, parentOrder = self.full_parents( node, getOrder=True, fullIndexing=True, returnFullIndex=False )
        # parents, parentOrder = self.full_parents( _node, getOrder=True, fullIndexing=False, returnFullIndex=False )
        keepAxes = []
        numbFbsParents = 0
        for p, o in zip( parents, parentOrder ):
            if( self.inFBS( p, fromReduced=True ) ):
                numbFbsParents += 1
                keepAxes.append( jointFBS.fbsAxis + self.fbsIndex( p, fromReduced=True, withinGraph=True ) )
            else:
                keepAxes.append( o )

        keepAxes = np.array( keepAxes )

        # Integrate out all fbs nodes that aren't the parents
        intAxes = np.setdiff1d( np.arange( jointFBS.ndim ), keepAxes )

        ans = self.integrate( jointFBS, axes=intAxes ).data

        # Need to swap the order of the axes!!!! Right now, all of the fbs nodes
        # are on the last axes!!!
        # The axes in ans should be aligned in order of non fbs parents, then
        # by fbs index
        parents, parentOrder = self.full_parents( node, getOrder=True, fullIndexing=True, returnFullIndex=False )
        nonFBSOrder = [ o for p, o in zip( parents, parentOrder ) if not self.inFBS( p, fromReduced=True ) ]
        fbsParents = [ ( p, o ) for p, o in zip( parents, parentOrder ) if self.inFBS( p, fromReduced=True ) ]
        fbsOrder = sorted( fbsParents, key=lambda x: self.fbsIndex( x[ 0 ], fromReduced=True, withinGraph=True ) )
        fbsOrder = [ o for p, o in fbsOrder ]

        trueOrder = nonFBSOrder + fbsOrder
        transposeAxes = [ trueOrder.index( i ) for i in range( len( trueOrder ) ) ]

        return np.transpose( ans, transposeAxes )

    ######################################################################

    def _jointParentChild( self, U, V, node, method='compute' ):
        # P( x_c, x_p1..pN, Y )

        notInFBS = not self.inFBS( node, fromReduced=False )

        if( method == 'integrate' ):
            # This fbs node has no siblings, so must compute
            _node = node if notInFBS else self._anotherNode( node, parents=False, mates=False, children=False, siblings=True )
            if( _node is None ):
                method = 'compute'

        method = 'compute' if notInFBS else method

        # If all of the parents are in the fbs, then use nodeJoint and integrate out the correct nodes.
        # This also will work when node is in the fbs too
        if( len( [ 1 for p in self.full_parents( node, fullIndexing=True ) if not self.inFBS( p, fromReduced=True ) ] ) == 0 ):
            _node = node if notInFBS else self._anotherNode( node )
            _node = self.fullIndexToReduced( _node )
            jointFBS = super()._nodeJoint( U, V, _node )

        elif( method == 'compute' ):
            _node = self.fullIndexToReduced( node )
            jointFBS = super()._jointParentChild( U, V, _node )

        elif( method == 'integrate' ):
            _node = node if notInFBS else self._anotherNode( node, parents=False, mates=False, children=False, siblings=True )
            _node = self.fullIndexToReduced( _node )

            jointFBS = super()._jointParentChild( U, V, _node )

        else:
            assert 0, 'Invalid method'

        # Integrate out the nodes that aren't parents
        parents, parentOrder = self.full_parents( node, getOrder=True, fullIndexing=True, returnFullIndex=False )
        # parents, parentOrder = self.full_parents( _node, getOrder=True, fullIndexing=False, returnFullIndex=False )
        nParents = self.nParents( node, full=True, fullIndexing=True )
        # nParents = self.nParents( _node, full=True )
        keepAxes = []
        for p, o in zip( parents, parentOrder ):
            if( self.inFBS( p, fromReduced=True ) ):
                keepAxes.append( jointFBS.fbsAxis + self.fbsIndex( p, fromReduced=True, withinGraph=True ) )
            else:
                keepAxes.append( o )

        # Keep the axis that node is on
        if( notInFBS ):
            keepAxes.append( nParents )
        else:
            keepAxes.append( jointFBS.fbsAxis + self.fbsIndex( node, fromReduced=False, withinGraph=True ) )

        keepAxes = np.array( keepAxes )

        # Integrate out all fbs nodes that aren't the parents
        intAxes = np.setdiff1d( np.arange( jointFBS.ndim ), keepAxes )

        ans = self.integrate( jointFBS, axes=intAxes ).data

        # Need to swap the order of the axes!!!! Right now, all of the fbs nodes
        # are on the last axes!!!
        # The axes in ans should be aligned in order of non fbs parents, then the child, then
        # by fbs index
        parents, parentOrder = self.full_parents( node, getOrder=True, fullIndexing=True, returnFullIndex=False )
        nonFBSOrder = [ o for p, o in zip( parents, parentOrder ) if not self.inFBS( p, fromReduced=True ) ]
        fbsParents = [ ( p, o ) for p, o in zip( parents, parentOrder ) if self.inFBS( p, fromReduced=True ) ]
        fbsOrder = sorted( fbsParents, key=lambda x: self.fbsIndex( x[ 0 ], fromReduced=True, withinGraph=True ) )
        fbsOrder = [ o for p, o in fbsOrder ]

        # This is going to contain the indices of where each axis should go,
        # which is not what transpose expects!
        if( self.inFBS( node, fromReduced=False ) ):
            # If node is in the fbs, then that means that its current axis
            # is somewhere after the fbsAxis
            # So just need to place len( parents ) somewhere in fbsOrder
            insertionIndex = self.fbsIndex( node, fromReduced=False, withinGraph=True )
            trueOrder = nonFBSOrder + fbsOrder[ :insertionIndex ] + [ len( parents ) ] + fbsOrder[ insertionIndex: ]
        else:
            trueOrder = nonFBSOrder + [ len( parents ) ] + fbsOrder

        # Find the correct axes
        transposeAxes = [ trueOrder.index( i ) for i in range( len( trueOrder ) ) ]

        return np.transpose( ans, transposeAxes )

    ######################################################################

    def jointParents( self, U, V, nodes ):
        # P( x_p1..pN, Y )
        return [ ( node, self._jointParents( U, V, node ) ) for node in nodes if self.nParents( node, full=True, fullIndexing=True ) > 0 ]

    def jointParentChild( self, U, V, nodes ):
        # P( x_c, x_p1..pN, Y )
        return [ ( node, self._jointParentChild( U, V, node ) ) for node in nodes if self.nParents( node, full=True, fullIndexing=True ) > 0 ]

    ######################################################################

    def marginalProb( self, U, V, node ):
        # P( Y )
        joint = self._nodeJoint( U, V, node )
        return self.integrate( joint, axes=range( joint.ndim ), useSuper=True )

    def nodeSmoothed( self, U, V, nodes ):
        # P( x | Y )
        return [ ( node, val - self.marginalProb( U, V, node ) ) for node, val in self.nodeJoint( U, V, nodes ) ]

    def conditionalParentChild( self, U, V, nodes ):
        # P( x_c | x_p1..pN, Y )
        ans = []
        for node in nodes:
            if( self.nParents( node, full=True, fullIndexing=True ) > 0 ):

                jpc = self._jointParentChild( U, V, node )
                jp = self._jointParents( U, V, node )
                _ans = self.multiplyTerms( terms=( jpc, -jp ), useSuper=True )

                ans.append( ( node, _ans ) )

        return ans

######################################################################

class GraphFilter( _filterMixin, GraphMessagePasser ):
    pass

class GraphFilterFBS(  __FBSFilterMixin, _filterMixin, GraphMessagePasserFBS ):
    pass