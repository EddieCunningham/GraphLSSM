from GraphicalMessagePassingBase import Graph, GraphMessagePasser, dprint
import numpy as np
from scipy.sparse import coo_matrix
from functools import reduce
from collections import Iterable

class GraphFilter( GraphMessagePasser ):
    # Base message passing class for hyper graphs.
    # Will use a sparse matrix to hold graph structure

    def __init__( self ):
        super( GraphFilter, self ).__init__()

    def genFilterProbs( self ):
        assert 0

    def genWorkspace( self ):
        assert 0

    def transitionProb( self, children, parents ):
        assert 0

    def emissionProb( self, nodes, forward=False ):
        assert 0

    def multiplyTerms( self, N, terms ):
        assert 0

    def integrate( self, integrand ):
        assert 0

    def updateParams( self, parentMasks, childMasks, feedbackSets=None ):
        super( GraphFilter, self ).updateParams( parentMasks, childMasks, feedbackSets=feedbackSets )

    ######################################################################

    def uBaseCase( self, roots, U, workspace ):
        assert 0

    def vBaseCase( self, leaves, V, workspace ):
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

    def filterFeedbackSet( self, U, V, workspace ):

        uList = self.fbs
        self.fbs = np.array( [] )

        vListNodes = []
        vListEdges = []
        for node in self.fbs:
            downEdges = self.downEdges( node )
            if( len( downEdges ) == 0 ):
                vListNodes.append( node )
                vListEdges.append( None )
            else:
                for edge in downEdges:
                    vListNodes.append( node )
                    vListEdges.append( edge )

        # This might not be correct.  what if nodes in the fbs depend on each other???
        # Probably best to pass something to the message passer instead.....
        self.uFilter( False, uList, U, V, workspace )
        self.vFilter( False, [ vListNodes, vListEdges ], U, V, workspace )

        self.fbs = np.array( [] )

        return U, V

    ######################################################################

    def inFBS( self, node ):
        return np.any( np.in1d( self.fbs, node ) )

    @classmethod
    def extendAxes( cls, term, targetAxis, maxDim ):
        # Assuming term has the 1 + F dimensions, where F is the size of the relevant feedback set
        # will align the first axis with target axis and the axes along the last F axes with
        # the maxDim : maxDim + F axes

        dprint( 'shape of term:', term.shape, use=True )

        # Add axes before the fbsAxes
        for _ in range( maxDim - targetAxis - 1 ):
            term = np.expand_dims( term, 1 )

        # Prepend axes
        for _ in range( targetAxis ):
            term = np.expand_dims( term, 0 )

        dprint( 'new shape of term:', term.shape, use=True )

        return term

    ######################################################################

    def latentStateSize( self, U, V, node ):
        return U[ node ].shape[ 0 ]

    def nParents( self, node ):
        return self.parents( node ).shape[ 0 ]

    def uData( self, U, node ):
        return U[ node ]

    def vData( self, V, node, edges=None, debug=False ):
        V_row, V_col, V_data = V

        if( self.inFBS( node ) ):
            # If node is part of the skip array (if its in the fbs)
            ans = np.array( [] )
        elif( ~np.any( np.in1d( V_row, node ) ) ):
            # If the node isn't in the V object (node is a leaf)
            ans = np.array( [] )
        elif( edges is None ):
            # Return the data over all down edges
            ans = self.vDataFromMask( V, np.in1d( V_row, node ) )
            # ans = V_data[ np.in1d( V_row, node ) ]
        elif( isinstance( edges, Iterable ) and len( edges ) > 0 ):
            # Return over only certain edges
            mask = np.in1d( V_row, node )
            for e in edges:
                mask &= np.in1d( V_col, e )
            ans = self.vDataFromMask( V, mask )
            # ans = V_data[ mask ]
        elif( isinstance( edges, Iterable ) and len( edges ) == 0 ):
            # This happens when we're not passed a down edge.  In this
            # case, we should return an invalid v value, not a leaf v value
            return np.array( [] )
        else:
            # Only looking at one edge
            ans = self.vDataFromMask( V, np.in1d( V_col, edges ) )
            # ans = V_data[ np.in1d( V_col, edges ) ]

        if( len( ans ) == 0 ):
        # if( ans.size == 0 ):
            # If we're looking at a leaf, return all 0s
            nVals = 1 if edges is None else len( edges )
            ans = np.zeros( ( nVals, self.K ) )

        assert sum( [ 0 if ~np.any( np.isnan( v ) ) else 1 for v in ans ] ) == 0, ans

        return ans

    ######################################################################

    def aFBS( self, U, V, node, downEdge, debug=True ):
        # Compute a when node is in the feedback set.
        # This case is different from the regular one because we are treating
        # the node like it is cut from the graph.
        # So we can ignore the edges from node and just return a matrix that
        # is only 1 when node_x == fbs_node_x and 0 otherwise so be consistent
        # with the conditioning
        assert 0

    def a( self, U, V, node, downEdge, debug=True ):
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

        dprint( '\n\nComputing a for', node, 'at downEdge', downEdge, use=debug )

        if( node in self.fbs ):
            return self.aFBS( U, V, node, downEdge, debug=debug )

        # Get the U data for node
        u = self.uData( U, node )
        dprint( 'u:\n', u, use=debug )

        # Get the V data over all down edges except downEdge
        v = self.vData( V, node, edges=self.downEdges( node, skipEdges=downEdge ) )
        dprint( 'v:\n', v, use=debug )

        # Multiply U and all the Vs
        term = self.multiplyTerms( terms=( u, *v ) )
        dprint( 'term:\n', term, use=debug )

        assert isinstance( term, np.ndarray ) and np.any( np.isnan( term ) ) == False

        return term

    ######################################################################

    def b( self, U, V, node, debug=True ):
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
        dprint( '\n\nComputing b for', node, use=debug )

        parents, parentOrder = self.parents( node, getOrder=True )
        dprint( 'parents:\n', parents, use=debug )
        totalDim = parents.shape[ 0 ] + 1

        # Get the transition matrix between node and parents
        nodeTransition = self.transitionProb( node, parents, parentOrder )
        dprint( 'nodeTransition:\n', nodeTransition, use=debug )

        # Get the emission vector for node.  Align on laslt axis
        nodeEmission = self.emissionProb( node )
        nodeEmission = self.extendAxes( nodeEmission, totalDim - 1, totalDim )
        dprint( 'nodeEmission:\n', nodeEmission, use=debug )

        # Get the V data over all down edges from node.  Align on last axis
        v = self.vData( V, node )
        v = [ self.extendAxes( _v, totalDim - 1, totalDim ) for _v in v ]
        dprint( 'v:\n', v, use=debug )

        # Multiply together the transition, emission and Vs
        integrand = self.multiplyTerms( terms=( nodeEmission, *v, nodeTransition ) )
        dprint( 'integrand:\n', integrand, use=debug )

        # Integrate over the node's latent states which is last axis
        term = self.integrate( integrand, axes=[ totalDim - 1 ] )
        dprint( 'term:\n', term, use=debug )

        assert isinstance( term, np.ndarray ) and np.any( np.isnan( term ) ) == False

        return term

    ######################################################################

    def u( self, U, V, node, debug=True ):
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
        dprint( '\n\nComputing u for', node, use=debug )

        upEdge = self.upEdges( node )
        parents, parentOrder = self.parents( node, getOrder=True )
        siblings = self.siblings( node )
        dprint( 'upEdge:\n', upEdge, use=debug )
        dprint( 'parents:\n', parents, use=debug )
        dprint( 'siblings:\n', siblings, use=debug )

        nParents = parents.shape[ 0 ]

        # Get the transition matrix between node and parents
        nodeTransition = self.transitionProb( node, parents, parentOrder )
        dprint( 'nodeTransition:\n', np.exp( nodeTransition ), use=debug )

        # Get the a values for all parents (skip this nodes up edge)
        parentTerms = [ self.a( U, V, p, upEdge, debug=debug ) for p in parents ]
        parentTerms = [ self.extendAxes( p, i, nParents ) for p, i in zip( parentTerms, parentOrder ) ]
        dprint( 'parentTerms:\n', parentTerms, use=debug )

        # Get the b values for all siblings.  These are all over the parents' axes
        if( len( siblings ) > 0 ):
            siblingTerms = [ self.b( U, V, s, debug=debug ) for s in siblings ]
        else:
            siblingTerms = np.array( [] )
        dprint( 'siblingTerms:\n', siblingTerms, use=debug )

        # Multiply all of the terms together
        integrand = self.multiplyTerms( terms=( *siblingTerms, *parentTerms, nodeTransition ) )
        dprint( 'integrand:\n', integrand, use=debug )

        # Integrate out the parent latent states
        nodeTerms = self.integrate( integrand, axes=list( range( nParents ) ) )
        dprint( 'nodeTerms:\n', nodeTerms, use=debug )

        # Get the emission vector for node
        nodeEmission = self.emissionProb( node )
        dprint( 'nodeEmission:\n', nodeEmission, use=debug )

        # Combine this nodes emission with the rest of the calculation
        newU = self.multiplyTerms( terms=( nodeTerms, nodeEmission ) )
        dprint( 'newU:\n', newU, use=debug )

        assert isinstance( newU, np.ndarray ) and np.any( np.isnan( newU ) ) == False

        return newU

    ######################################################################

    def v( self, U, V, node, edge, debug=True ):
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
        dprint( '\n\nComputing v for', node, 'at edge', edge, use=debug )

        mates, mateOrder = self.mates( node, getOrder=True, edges=edge )
        children = self.children( node, edges=edge )
        parents, parentOrder = self.parents( children[ 0 ], getOrder=True )
        dprint( 'mates:\n', mates, use=debug )
        dprint( 'children:\n', children, use=debug )

        nMates = mates.shape[ 0 ]
        nParents = mates.shape[ 0 ] + 1

        # Get the a values for each of the mates (skip edge)
        if( len( mates ) > 0 ):
            mateTerms = [ self.a( U, V, m, edge, debug=debug ) for m in mates ]
            mateTerms = [ self.extendAxes( m, i, nParents ) for m, i in zip( mateTerms, mateOrder ) ]
        else:
            mateTerms = np.array( [] )
        dprint( 'mateTerms:\n', mateTerms, use=debug )
        dprint( 'mateTerms shapes:\n', [ m.shape for m in mateTerms ], use=debug )

        # Get the b values for each of the children.  These are all over the parents' axes
        childTerms = [ self.b( U, V, c, debug=debug ) for c in children ]
        dprint( 'childTerms:\n', childTerms, use=debug )
        dprint( 'childTerm shapes:\n', [ c.shape for c in childTerms ], use=debug )

        # Combine the terms
        integrand = self.multiplyTerms( terms=( *childTerms, *mateTerms ) )
        dprint( 'integrand:\n', integrand, use=debug )
        dprint( 'integrand.shape:\n', integrand.shape, use=debug )

        intAxes = mateOrder
        dprint( 'intAxes:\n', intAxes, use=debug )

        # Integrate out the mates latent states
        newV = self.integrate( integrand, axes=intAxes )
        dprint( 'newV:\n', newV, use=debug )

        assert isinstance( newV, np.ndarray ) and np.any( np.isnan( newV ) ) == False

        return newV

    ######################################################################

    def uFilter( self, baseCase, nodes, U, V, workspace, debug=True ):
        # Compute P( ↑( n )_y, n_x )
        # Probability of all emissions that can be reached by going up node's up edge

        dprint( '\n\nComputing U for', nodes, use=debug )

        if( baseCase ):
            newU = [ self.uBaseCase( node, debug=debug ) for node in nodes ]
        else:
            newU = [ self.u( U, V, node, debug=debug ) for node in nodes ]

        self.updateU( nodes, newU, U )

    def vFilter( self, baseCase, nodesAndEdges, U, V, workspace, debug=True ):

        nodes, edges = nodesAndEdges

        dprint( '\n\nComputing V for', nodes, 'at edges', edges, use=debug )

        if( baseCase ):
            newV = [ self.vBaseCase( node, debug=debug ) for node in nodes ]
        else:
            newV = [ self.v( U, V, n, e, debug=debug ) for n, e in zip( nodes, edges ) ]

        self.updateV( nodes, edges, newV, V )

    def convergence( self, nodes ):
        return False

    ######################################################################

    def filter( self ):

        workspace = self.genWorkspace()
        U, V = self.genFilterProbs()

        kwargs = {
            'U': U,
            'V': V,
            'workspace': workspace
        }

        debug = True

        # Run the message passing algorithm over the graph.
        # We will end up with the extended U and V values for all of
        # the nodes except the feedback set nodes.
        # To get the filtered probs for the smoothed probs for the
        # feedback set nodes, marginalize over the non fbs nodes
        # at any extended smoothed prob
        self.messagePassing( self.uFilter, self.vFilter, **kwargs )

        dprint( '\n\n------------\n\n', use=True )

        dprint( 'U\n', use=True )
        for u in U:
            dprint( u, use=True )

        dprint( '\n\n------------\n\n', use=True )

        dprint( 'V\n', use=True )
        for v in V[ 2 ]:
            dprint( v, use=True )

        dprint( '\n\n------------\n\n', use=True )


        return U, V

    ######################################################################

    def _fullJoint( self, U, V, node, debug=True ):
        # P( x, x_{feedback set}, Y )

        parents, parentOrder = self.parents( node, getOrder=True )

        u = self.uData( U, node )
        dprint( 'u:\n', u, use=debug )

        v = self.vData( V, node )
        v = [ self.extendAxes( _v, 0, 1 ) for _v in v ]
        dprint( 'v:\n', v, use=debug )

        joint =  self.multiplyTerms( terms=( u, *v ) )
        dprint( 'joint:\n', joint, use=debug )
        return joint

    def _nodeJointForFBS( self, U, V, node, debug=True ):
        # If node is a root, get a child and if it is a leaf, get a parent
        if( self.nParents( node ) == 0 ):
            useNode = self.children( node )[ 0 ]
        else:
            useNode = self.parents( node )[ 0 ]

        joint = self._fullJoint( U, V, useNode, debug=debug )
        dprint( 'joint', joint, use=debug )

        fbsIndex = self.fbsIndex.tolist().index( node )
        intAxes = np.setdiff1d( np.arange( self.fbs.shape[ 0 ] + 1 ), fbsIndex + 1 )
        dprint( 'intAxes', intAxes, use=debug )

        # Integration axes are all axes but node
        intJoint = self.integrate( joint, axes=intAxes )
        print( '\n\n\n\n\n' )
        dprint( 'intJoint:\n', intJoint, '->', np.logaddexp.reduce( intJoint ), use=debug )
        print( '\n\n\n\n\n' )
        return intJoint

    def _nodeJoint( self, U, V, node, debug=True ):
        # P( x, Y )

        if( node in self.fbs ):
            return self._nodeJointForFBS( U, V, node, debug=debug )

        joint = self._fullJoint( U, V, node, debug=debug )
        dprint( 'joint', joint, use=debug )

        intAxes = list( range( 1, self.fbs.shape[ 0 ] + 1 ) )
        dprint( 'intAxes', intAxes, use=debug )

        intJoint = self.integrate( joint, axes=intAxes )
        print( '\n\n\n\n\n' )
        dprint( 'intJoint:\n', intJoint, '->', np.logaddexp.reduce( intJoint ), use=debug )
        print( '\n\n\n\n\n' )
        return intJoint

    def _jointParentChild( self, U, V, node, debug=True ):
        # P( x_c, x_p1..pN, Y )

        siblings = self.siblings( node )
        parents, parentOrder = self.parents( node, getOrder=True )
        upEdge = self.upEdges( node )

        nParents = parents.shape[ 0 ]
        totalDim = nParents + 1

        # Down to this node
        nodeTransition = self.transitionProb( node, parents, parentOrder )
        nodeEmission = self.emissionProb( node )
        nodeEmission = self.extendAxes( nodeEmission, totalDim - 1, totalDim )

        # Out from each sibling
        siblingTerms = [ self.b( U, V, s, debug=debug )[ 0 ] for s in siblings ]

        # Out from each parent
        parentTerms = [ self.a( U, V, p, upEdge, debug=debug ) for p in parents ]
        parentTerms = [ self.extendAxes( p, i, totalDim ) for p, i in zip( parentTerms, parentOrder ) ]

        # Down this node
        v = self.vData( V, node )
        v = [ self.extendAxes( _v, totalDim - 1, totalDim ) for _v in v ]

        return self.multiplyTerms( terms=( *parentTerms, *siblingTerms, *v, nodeTransition, nodeEmission ) )

    def _jointParents( self, U, V, node, debug=True ):
        # P( x_p1..pN, Y )

        parents, parentOrder = self.parents( node, getOrder=True )
        siblings = self.siblings( node )
        upEdge = self.upEdges( node )

        nParents = parents.shape[ 0 ]
        totalDim = nParents + 1

        # Down each child
        siblingTerms = [ self.b( U, V, s, debug=debug )[ 0 ] for s in siblings ]

        # Down this node
        nodeTerm = self.b( U, V, node, debug=debug )[ 0 ]

        # Out from each parent
        parentTerms = [ self.a( U, V, p, upEdge, debug=debug ) for p in parents ]
        parentTerms = [ self.extendAxes( p, i, totalDim ) for p, i in zip( parentTerms, parentOrder ) ]

        return self.multiplyTerms( terms=( nodeTerm, *parentTerms, *siblingTerms ) )

    ######################################################################

    def nodeJoint( self, U, V, nodes ):
        # P( x, Y )
        return [ self._nodeJoint( U, V, node ) for node in nodes ]

    def jointParentChild( self, U, V, nodes ):
        # P( x_c, x_p1..pN, Y )
        return [ self._jointParentChild( U, V, node ) for node in nodes if self.nParents( node ) > 0 ]

    def jointParents( self, U, V, nodes, returnLog=False ):
        # P( x_p1..pN | Y )

        ans = []
        for node in nodes:
            if( self.nParents( node ) > 0 ):
                _ans = self._jointParents( U, V, node )

                if( returnLog == True ):
                    ans.append( _ans )
                else:
                    ans.append( np.exp( _ans ) )
        return ans

    def marginalProb( self, U, V ):
        # P( Y )
        # THIS MIGHT NOT BE TRUE.  IF WE HAVE DISJOINT GRAPHS, THEN
        # WE'D NEED A LEAF FOR EACH COMPONENT.
        # Really don't need this because this can be calculated at any
        # node, but for testing purposes keep it like this
        parentOfEdgeCount = self.pmask.getnnz( axis=1 )
        childOfEdgeCount = self.cmask.getnnz( axis=1 )
        leafIndex = self.nodes[ ( childOfEdgeCount != 0 ) & ( parentOfEdgeCount == 0 ) ][ 0 ]
        marginalProb = np.logaddexp.reduce( self._nodeJoint( U, V, leafIndex ) )
        print( 'marginalProb', marginalProb )
        return marginalProb

    def nodeSmoothed( self, U, V, nodes, returnLog=False ):
        # P( x | Y )
        marginal = self.marginalProb( U, V )
        ans = [ val - marginal for val in self.nodeJoint( U, V, nodes ) ]
        return ans if returnLog == True else np.exp( ans )

    def conditionalParentChild( self, U, V, nodes, returnLog=False ):
        # P( x_c | x_p1..pN, Y )
        ans = []
        for node in nodes:
            nParents = self.nParents( node )
            if( nParents > 0 ):
                parents, parentOrder = self.parents( node, getOrder=True )

                jpc = self._jointParentChild( U, V, node )
                jp = -self._jointParents( U, V, node )

                # print( 'node', node )
                # print( 'jpc', jpc )
                # print( 'jp', jp )

                _ans = self.multiplyTerms( terms=( jpc, jp ) )

                if( returnLog == True ):
                    ans.append( _ans )
                else:
                    ans.append( np.exp( _ans ) )

        return ans
