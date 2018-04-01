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

    def integrateOutFeedbackSet( self, U, V, workspace ):
        assert 0

    def filterFeedbackSet( self, U, V, workspace ):
        return
        uList = self.fbs
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

    ######################################################################

    def firstAxis( self ):
        return ( 0, )

    def lastAxis( self, N ):
        return ( N - 1, )

    def ithAxis( self, i ):
        return ( i, )

    def sequentialAxes( self, N, skip=None ):
        if( skip is not None ):
            return list( np.setdiff1d( np.arange( N, dtype=int ), skip ).tolist() )
        return list( np.arange( N, dtype=int ).tolist() )

    def latentStateSize( self, U, V, node ):
        return U[ node ].shape[ 0 ]

    def nParents( self, node ):
        return self.parents( node ).shape[ 0 ]

    def uData( self, U, node ):
        return U[ node ]

    def vData( self, V, node, edges=None, debug=False ):
        V_row, V_col, V_data = V

        if( np.any( np.in1d( self.fbs, node ) ) ):
            # If node is part of the skip array (if its in the fbs)
            ans = np.array( [] )
        elif( ~np.any( np.in1d( V_row, node ) ) ):
            # IF the node isn't in the V object (node is a leaf)
            ans = np.array( [] )
        elif( edges is None ):
            # Return the data over all down edges
            ans = V_data[ np.in1d( V_row, node ) ]
        elif( isinstance( edges, Iterable ) and len( edges ) > 0 ):
            # Return over only certain edges
            mask = np.in1d( V_row, node )
            for e in edges:
                mask &= np.in1d( V_col, e )
            ans = V_data[ mask ]
        elif( isinstance( edges, Iterable ) and len( edges ) == 0 ):
            # This happens when we're not passed a down edge.  In this
            # case, we should return an invalid v value, not a leaf v value
            return np.array( [] )
        else:
            # Only looking at one edge
            ans = V_data[ np.in1d( V_col, edges ) ]

        if( ans.size == 0 ):
            # If we're looking at a leaf, return all 0s
            depth = 1 if edges is None else len( edges )
            baseCase = np.zeros( ( depth, V_data[ 0 ].shape[ 0 ] ) )
            ans = baseCase

        assert np.any( np.isnan( ans ) ) == False

        return ans

    ######################################################################

    def fbsAxesCorrection( self, totalDim, nodes, axes ):
        if( isinstance( nodes, Iterable ) == False ):
            nodes = np.array( [ nodes ] )

        newAxes = np.copy( axes )
        nodeInFBS = nodes[ np.in1d( nodes, self.fbs ) ]
        for i, fbsNode in enumerate( nodeInFBS ):
            fbsIndex = np.where( self.fbs == fbsNode )[ 0 ]
            axes[ i ] = totalDim + fbsIndex
        return newAxes

    ######################################################################

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
        # Return array should be of size ( K, ) where K is the latent state size
        dprint( '\n\nComputing a for', node, 'at downEdge', downEdge, use=debug )

        firstAxis = self.firstAxis()

        # Get the U data for node
        u = self.uData( U, node )
        uAxes = firstAxis
        dprint( 'u:\n', u, use=debug )
        dprint( 'uAxes:\n', uAxes, use=debug )

        # Get the V data over all down edges except downEdge
        relevantDownEdges = self.downEdges( node, skipEdges=downEdge )
        v = self.vData( V, node, edges=relevantDownEdges )
        vAxes = [ firstAxis for _ in v ]
        dprint( 'v:\n', v, use=debug )
        dprint( 'vAxes:\n', vAxes, use=debug )

        # Multiply U and all the Vs
        term = self.multiplyTerms( ( u, *v ), axes=( uAxes, *vAxes ) )
        dprint( 'term:\n', term, use=debug )

        assert isinstance( term, np.ndarray )
        assert np.any( np.isnan( term ) ) == False

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
        # and N is the number of parents
        dprint( '\n\nComputing b for', node, use=debug )

        parents, parentOrder = self.parents( node, getOrder=True )
        dprint( 'parents:\n', parents, use=debug )
        nParents = parents.shape[ 0 ]
        totalDim = nParents + 1

        allAxes = self.sequentialAxes( N=totalDim )
        lastAxis = self.lastAxis( N=totalDim )

        # Get the transition matrix between node and parents
        nodeTransition = self.transitionProb( node, parents )
        transitionAxes = allAxes
        dprint( 'nodeTransition:\n', nodeTransition, use=debug )
        dprint( 'transitionAxes:\n', transitionAxes, use=debug )

        # Get the emission vector for node
        nodeEmission = self.emissionProb( node )
        emissionAxes = lastAxis
        dprint( 'nodeEmission:\n', nodeEmission, use=debug )
        dprint( 'emissionAxes:\n', emissionAxes, use=debug )

        # Get the V data over all down edges from node
        v = self.vData( V, node )
        vAxes = [ lastAxis for _ in v ]
        dprint( 'v:\n', v, use=debug )
        dprint( 'vAxes:\n', vAxes, use=debug )

        # # Change all of the axes that correspond to nodes in the feedback set
        # transitionAxes = self.fbsAxesCorrection( totalDim, parents, transitionAxes )
        # dprint( 'adjusted transitionAxes:\n', transitionAxes, use=debug )

        # Multiply together the transition, emission and Vs
        integrand = self.multiplyTerms( ( nodeEmission, *v, nodeTransition ), axes=( emissionAxes, *vAxes, transitionAxes ) )
        intAxes = lastAxis
        dprint( 'integrand:\n', integrand, use=debug )
        dprint( 'intAxes:\n', intAxes, use=debug )

        # Integrate over the node's latent states
        term = self.integrate( integrand, axes=intAxes )
        dprint( 'term:\n', term, use=debug )

        assert isinstance( term, np.ndarray )
        assert np.any( np.isnan( term ) ) == False

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
        totalDim = nParents + 1

        allAxes = self.sequentialAxes( N=totalDim )
        upToLastAxes = self.sequentialAxes( N=nParents )
        firstAxis = self.firstAxis()

        # Get the transition matrix between node and parents
        nodeTransition = self.transitionProb( node, parents )
        transitionAxes = allAxes
        dprint( 'nodeTransition:\n', np.exp( nodeTransition ), use=debug )
        dprint( 'transitionAxes:\n', transitionAxes, use=debug )

        # Get the a values for all parents (skip this nodes up edge)
        parentTerms = [ self.a( U, V, p, upEdge, debug=debug ) for p in parents ]
        parentAxes = [ self.ithAxis( i ) for i in parentOrder ]
        dprint( 'parentTerms:\n', np.exp( parentTerms ), use=debug )
        dprint( 'parentAxes:\n', parentAxes, use=debug )

        # Get the b values for all siblings.  These are all over the parents' axes
        siblingTerms = [ self.b( U, V, s, debug=debug ) for s in siblings ]
        siblingAxes = [ upToLastAxes for _ in siblings ]
        dprint( 'siblingTerms:\n', np.exp( siblingTerms ), use=debug )
        dprint( 'siblingAxes:\n', siblingAxes, use=debug )

        # # Change all of the axes that correspond to nodes in the feedback set
        # transitionAxes = self.fbsAxesCorrection( totalDim, parents, transitionAxes )
        # dprint( 'adjusted transitionAxes:\n', transitionAxes, use=debug )

        # for i, ( p, pAx ) in enumerate( zip( parents, parentAxes ) ):
        #     parentAxes[ i ] = self.fbsAxesCorrection( totalDim, p, pAx )
        # dprint( 'adjusted parentAxes:\n', parentAxes, use=debug )

        # for i, ( s, sAx ) in enumerate( zip( siblings, siblingAxes ) ):
        #     # The axes for b terms should be aligned to the parents
        #     siblingAxes[ i ] = self.fbsAxesCorrection( totalDim, parents, sAx )
        # dprint( 'adjusted siblingAxes:\n', siblingAxes, use=debug )

        # Multiply all of the terms together
        integrand = self.multiplyTerms( ( *siblingTerms, *parentTerms, nodeTransition ), axes=( *siblingAxes, *parentAxes, transitionAxes ) )
        intAxes = upToLastAxes
        dprint( 'integrand:\n', np.exp( integrand ), use=debug )
        dprint( 'intAxes:\n', intAxes, use=debug )

        # Integrate out the parent latent states
        nodeTerms = self.integrate( integrand, axes=intAxes )
        nodeTermAxis = firstAxis
        dprint( 'nodeTerms:\n', np.exp( nodeTerms ), use=debug )
        dprint( 'nodeTermAxis:\n', nodeTermAxis, use=debug )

        # Get the emission vector for node
        nodeEmission = self.emissionProb( node )
        nodeEmissionAxis = firstAxis
        dprint( 'nodeEmission:\n', np.exp( nodeEmission ), use=debug )
        dprint( 'nodeEmissionAxis:\n', nodeEmissionAxis, use=debug )

        # Combine this nodes emission with the rest of the calculation
        newU = self.multiplyTerms( ( nodeTerms, nodeEmission ), axes=( nodeTermAxis, nodeEmissionAxis ) )
        dprint( 'newU:\n', np.exp( newU ), use=debug )

        assert isinstance( newU, np.ndarray )
        assert np.any( np.isnan( newU ) ) == False


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
        dprint( 'mates:\n', mates, use=debug )
        dprint( 'children:\n', children, use=debug )

        nMates = mates.shape[ 0 ]
        nParents = mates.shape[ 0 ] + 1
        totalDim = nParents + 1

        thisNodesOrder = np.setdiff1d( np.arange( nParents ), mateOrder )

        allAxes = self.sequentialAxes( N=nParents )
        upToLastAxes = self.sequentialAxes( N=nParents, skip=thisNodesOrder ) # Make sure that we integrate over mates only

        # Get the a values for each of the mates (skip edge)
        mateTerms = [ self.a( U, V, m, edge, debug=debug ) for m in mates ]
        mateAxes = [ self.ithAxis( i ) for i in mateOrder ]
        dprint( 'mateTerms:\n', mateTerms, use=debug )
        dprint( 'mateAxes:\n', mateAxes, use=debug )

        # Get the b values for each of the children.  These are all over the parents' axes
        childTerms = [ self.b( U, V, c, debug=debug ) for c in children ]
        childAxes = [ allAxes for _ in children ]
        dprint( 'childTerms:\n', childTerms, use=debug )
        dprint( 'childAxes:\n', childAxes, use=debug )

        # # Change all of the axes that correspond to nodes in the feedback set
        # for i, ( m, mAx ) in enumerate( zip( mates, mateAxes ) ):
        #     mateAxes[ i ] = self.fbsAxesCorrection( totalDim, m, mAx )
        # dprint( 'adjusted mateAxes:\n', mateAxes, use=debug )


        # for i, ( c, cAx ) in enumerate( zip( children, childAxes ) ):
        #     # The axes for b terms should be aligned to the parents
        #     childAxes[ i ] = self.fbsAxesCorrection( totalDim, mates, cAx )
        # dprint( 'adjusted childAxes:\n', childAxes, use=debug )

        # Combine the terms
        integrand = self.multiplyTerms( ( *childTerms, *mateTerms ), axes=( *childAxes, *mateAxes ) )
        intAxes = upToLastAxes
        dprint( 'integrand:\n', integrand, use=debug )
        dprint( 'intAxes:\n', intAxes, use=debug )

        # Integrate out the mates latent states
        newV = self.integrate( integrand, axes=intAxes ).ravel() # Ravel here so that we end up aligned on 0th axis
        dprint( 'newV:\n', newV, use=debug )

        assert isinstance( newV, np.ndarray )
        assert np.any( np.isnan( newV ) ) == False

        return newV

    ######################################################################

    def uFilter( self, baseCase, nodes, U, V, workspace, debug=True ):
        # Compute P( ↑( n )_y, n_x )
        # Probability of all emissions that can be reached by going up node's up edge

        dprint( '\n\nComputing U for', nodes, use=debug )

        if( baseCase ):
            newU = self.uBaseCase( nodes, U, workspace, debug=debug )
        else:
            newU = [ self.u( U, V, node, debug=debug ) for node in nodes ]

        self.updateU( nodes, newU, U )

    def vFilter( self, baseCase, nodesAndEdges, U, V, workspace, debug=True ):

        nodes, edges = nodesAndEdges

        dprint( '\n\nComputing V for', nodes, 'at edges', edges, use=debug )

        if( baseCase ):
            self.vBaseCase( nodes, V, workspace )
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

        # Run the message passing algorithm over the graph
        self.messagePassing( self.uFilter, self.vFilter, **kwargs )

        # Integrate out the nodes that we cut
        self.integrateOutFeedbackSet( U, V, workspace )

        # Update the filter probs for the cut nodes
        self.filterFeedbackSet( U, V, workspace )

        return U, V

    ######################################################################

    def _nodeJoint( self, U, V, node ):
        # P( x, Y )

        u = self.uData( U, node )
        uAxes = self.firstAxis()

        v = self.vData( V, node )
        vAxes = [ self.firstAxis() for _ in v ]

        return self.multiplyTerms( ( u, *v ), axes=( uAxes, *vAxes ) )

    def _jointParentChild( self, U, V, node, debug=True ):
        # P( x_c, x_p1..pN, Y )

        siblings = self.siblings( node )
        parents, parentOrder = self.parents( node, getOrder=True )
        upEdge = self.upEdges( node )

        nParents = parents.shape[ 0 ]
        allAxes = self.sequentialAxes( N=nParents + 1 )
        upToLastAxes = self.sequentialAxes( N=nParents )
        lastAxis = self.lastAxis( N=nParents + 1 )

        # Down to this node
        nodeTransition = self.transitionProb( node, parents )
        transitionAxes = allAxes
        nodeEmission = self.emissionProb( node )
        emissionAxes = lastAxis

        # Out from each sibling
        siblingTerms = [ self.b( U, V, s, debug=debug ) for s in siblings ]
        siblingAxes = [ upToLastAxes for _ in siblings ]

        # Out from each parent
        parentTerms = [ self.a( U, V, p, upEdge, debug=debug ) for p in parents ]
        parentAxes = [ self.ithAxis( i ) for i in parentOrder ]

        # Down this node
        v = self.vData( V, node )
        vAxes = [ lastAxis for _ in v ]

        return self.multiplyTerms( ( *parentTerms, *siblingTerms, *v, nodeTransition, nodeEmission ), axes=( *parentAxes, *siblingAxes, *vAxes, transitionAxes, emissionAxes ) )

    def _jointParents( self, U, V, node, debug=True ):
        # P( x_p1..pN, Y )

        parents, parentOrder = self.parents( node, getOrder=True )
        siblings = self.siblings( node )
        upEdge = self.upEdges( node )

        nParents = parents.shape[ 0 ]
        upToLastAxes = self.sequentialAxes( N=nParents )

        # Down each child
        siblingTerms = [ self.b( U, V, s, debug=debug ) for s in siblings ]
        siblingAxes = [ upToLastAxes for s in siblings ]

        # Down this node
        nodeTerm = self.b( U, V, node, debug=debug )
        nodeAxes = upToLastAxes

        # Out from each parent
        parentTerms = [ self.a( U, V, p, upEdge, debug=debug ) for p in parents ]
        parentAxes = [ self.ithAxis( i ) for i in parentOrder ]

        return self.multiplyTerms( ( nodeTerm, *parentTerms, *siblingTerms ), axes=( nodeAxes, *parentAxes, *siblingAxes ) )

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
        u = U[ leafIndex ]
        return self.integrate( u, axes=np.array( [ 0 ] ) )

    def nodeSmoothed( self, U, V, nodes, returnLog=False ):
        # P( x | Y )
        ans = self.nodeJoint( U, V, nodes ) - self.marginalProb( U, V )
        return ans if returnLog == True else np.exp( ans )

    def conditionalParentChild( self, U, V, nodes, returnLog=False ):
        # P( x_c | x_p1..pN, Y )
        ans = []
        for node in nodes:
            nParents = self.nParents( node )
            if( nParents > 0 ):
                jpc = self._jointParentChild( U, V, node )
                jpcAxes = self.sequentialAxes( N=nParents + 1 )

                jp = -self._jointParents( U, V, node )
                jpAxes = self.sequentialAxes( N=nParents )

                _ans = self.multiplyTerms( ( jpc, jp ), axes=( jpcAxes, jpAxes ) )

                if( returnLog == True ):
                    ans.append( _ans )
                else:
                    ans.append( np.exp( _ans ) )

        return ans
