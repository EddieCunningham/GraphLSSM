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

    def condition( self, nodes ):
        assert 0

    ######################################################################

    def uBaseCase( self, roots, U, conditioning, workspace ):
        assert 0

    def vBaseCase( self, leaves, V, conditioning, workspace ):
        assert 0

    ######################################################################

    def updateU( self, nodes, newU, U, conditioning ):
        assert 0

    def updateV( self, nodes, edges, newV, V, conditioning ):
        assert 0

    ######################################################################

    def integrateOutConditioning( self, U, V, conditioning, workspace ):
        assert 0

    def filterCutNodes( self, U, V, conditioning, workspace ):
        assert 0

    ######################################################################

    @classmethod
    def firstAxis( cls ):
        return ( 0, )

    @classmethod
    def lastAxis( cls, N ):
        return ( N - 1, )

    @classmethod
    def ithAxis( cls, i ):
        return ( i, )

    @classmethod
    def sequentialAxes( cls, N ):
        return tuple( np.arange( N, dtype=int ).tolist() )

    @classmethod
    def latentStateSize( cls, U, V, node ):
        return U[ node ].shape[ 0 ]

    def nParents( self, node ):
        return self.parents( node ).shape[ 0 ]

    @classmethod
    def uData( cls, U, node ):
        return U[ node ]

    @classmethod
    def vData( cls, V, node, edges=None ):
        V_row, V_col, V_data = V
        if( edges is None ):
            # Return the data over all down edges
            return V_data[ np.in1d( V_row, node ) ]
        elif( isinstance( edges, Iterable ) ):
            # Return over only certain edges
            mask = np.zeros_like( V_data, dtype=bool )
            for e in edges:
                mask |= np.in1d( V_col, e )
            return V_data[ mask ]
        else:
            # Only looking at one edge
            return V_data[ np.in1d( V_col, edge ) ]

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

        # Get the U data for node
        u = self.uData( U, node )
        uAxes = self.firstAxis()
        dprint( 'U:\n', u, use=debug )
        dprint( 'uAxes:\n', uAxes, use=debug )

        # Get the V data over all down edges except downEdge
        relevantDownEdges = self.downEdges( node, skipEdges=downEdge )
        v = self.vData( V, node, edges=relevantDownEdges )
        vAxes = [ self.firstAxis() for _ in v ]
        dprint( 'V:\n', v, use=debug )
        dprint( 'vAxes:\n', vAxes, use=debug )

        # Multiply U and all the Vs
        term = self.multiplyTerms( ( u, *v ), axes=( uAxes, *vAxes ) )
        dprint( 'term:\n', term, use=debug )

        assert isinstance( term, np.ndarray )
        K = self.latentStateSize( U, V, node )
        assert term.shape == ( K, )

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

        parents = self.parents( node )
        nParents = parents.shape[ 0 ]

        # Get the transition matrix between node and parents
        nodeTransition = self.transitionProb( node, parents )
        transitionAxes = self.sequentialAxes( N=nParents + 1 )
        dprint( 'nodeTransition:\n', nodeTransition, use=debug )

        # Get the emission vector for node
        nodeEmission = self.emissionProb( node )
        emissionAxes = self.lastAxis( N=nParents )
        dprint( 'nodeEmission:\n', nodeEmission, use=debug )

        # Get the V data over all down edges from node
        v = self.vData( V, node )
        vAxes = [ emissionAxes for _ in v ]
        dprint( 'V for childs:\n', v, use=debug )

        # Multiply together the transition, emission and Vs
        integrand = self.multiplyTerms( ( nodeEmission, *v, nodeTransition ), axes=( emissionAxes, *vAxes, transitionAxes ) )
        dprint( 'integrand:\n', integrand, use=debug )

        # Integrate over the node's latent states
        childrenTerms = self.integrate( integrand, axes=self.lastAxis( N=nParents ) )
        dprint( 'childrenTerms:\n', childrenTerms, use=debug )

        assert isinstance( childrenTerms, np.ndarray )
        K = self.latentStateSize( U, V, node )
        assert childrenTerms.shape == ( K, ) * nParents

        return childrenTerms

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

        upEdge = self.upEdges( node )
        parents = self.parents( node )
        siblings = self.siblings( node )
        dprint( 'upEdge:\n', upEdge, use=debug )
        dprint( 'parents:\n', parents, use=debug )
        dprint( 'siblings:\n', siblings, use=debug )

        nParents = parents.shape[ 0 ]

        # Get the transition matrix between node and parents
        nodeTransition = self.transitionProb( node, parents )
        transitionAxes = self.sequentialAxes( N=nParents + 1 )
        dprint( 'nodeTransition:\n', nodeTransition, use=debug )

        # Get the b values for all siblings
        siblingTerms = [ self.b( U, V, s, debug=debug ) for s in siblings ]
        siblingAxes = [ self.sequentialAxes( N=nParents - 1 ) for _ in siblings ]
        dprint( 'siblingTerms:\n', siblingTerms, use=debug )

        # Get the a values for all parents (skip this nodes up edge)
        parentTerms = [ self.a( U, V, p, upEdge, debug=debug ) for p in parents ]
        parentAxes = [ self.ithAxis( i ) for i in range( nParents ) ]
        dprint( 'parentTerms:\n', parentTerms, use=debug )

        # Multiply all of the terms together
        integrand = self.multiplyTerms( ( *siblingTerms, *parentTerms, nodeTransition ), axes=( *siblingAxes, *parentAxes, transitionAxes ) )
        dprint( 'integrand:\n', integrand, use=debug )

        # Integrate out the parent latent states
        nodeTerms = self.integrate( integrand, axes=self.sequentialAxes( N=nParents ) )
        nodeTermAxis = self.firstAxis()
        dprint( 'nodeTerms:\n', nodeTerms, use=debug )

        # Get the emission vector for node
        nodeEmission = self.emissionProb( node )
        nodeEmissionAxis = self.firstAxis()
        dprint( 'nodeEmission:\n', nodeEmission, use=debug )

        # Combine this nodes emission with the rest of the calculation
        newU = self.multiplyTerms( ( nodeTerms, nodeEmission ), axes=( nodeTermAxis, nodeEmissionAxis ) )
        dprint( 'newU:\n', newU, use=debug )

        assert isinstance( newU, np.ndarray )
        K = self.latentStateSize( U, V, node )
        assert newU.shape == ( K, )

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

        mates = self.mates( node )
        children = self.children( node )
        dprint( 'mates:\n', mates, use=debug )
        dprint( 'children:\n', children, use=debug )

        nMates = mates.shape[ 0 ]
        nParents = mates.shape[ 0 ] + 1

        # Get the b values for each of the children
        childTerms = [ self.b( U, V, c, debug=debug ) for c in children ]
        childAxes = [ self.sequentialAxes( N=nParents ) for _ in children ]
        dprint( 'childTerms:\n', childTerms, use=debug )

        # Get the a values for each of the mates (skip edge)
        mateTerms = [ self.a( U, V, m, edge, debug=debug ) for m in mates ]
        mateAxes = [ self.ithAxis( i ) for i in np.arange( nMates ) ]
        dprint( 'mateTerms:\n', mateTerms, use=debug )

        # Combine the terms
        integrand = self.multiplyTerms( ( *childTerms, *mateTerms ), axes=( *childAxes, *mateAxes ) )
        dprint( 'integrand:\n', integrand, use=debug )

        # Integrate out the mates latent states
        newV = self.integrate( integrand, axes=self.sequentialAxes( N=nMates ) )
        dprint( 'newV:\n', newV, use=debug )

        assert isinstance( newV, np.ndarray )
        K = self.latentStateSize( U, V, node )
        assert newV.shape == ( K, )

        return newV

    ######################################################################

    def uFilter( self, baseCase, nodes, U, V, conditioning, workspace, debug=True ):
        # Compute P( ↑( n )_y, n_x )
        # Probability of all emissions that can be reached by going up node's up edge

        dprint( '\n\nComputing U for', nodes, use=debug )

        if( baseCase ):
            newU = self.uBaseCase( nodes, U, conditioning, workspace, debug=debug )
        else:
            newU = [ self.u( U, V, node, debug=debug ) for node in nodes ]

        self.updateU( nodes, newU, U, conditioning )

    def vFilter( self, baseCase, nodesAndEdges, U, V, conditioning, workspace, debug=True ):

        nodes, edges = nodesAndEdges

        dprint( '\n\nComputing V for', nodes, 'at edges', edges, use=debug )

        if( baseCase ):
            self.vBaseCase( nodes, V, conditioning, workspace )
        else:
            newV = [ self.v( U, V, n, e, debug=debug ) for n, e in zip( nodes, edges ) ]
            self.updateV( nodes, edges, newV, V, conditioning )

    def convergence( self, nodes ):
        return False

    ######################################################################

    def filter( self ):

        workspace = self.genWorkspace()
        conditioning = self.condition( self.fbsMask )
        U, V = self.genFilterProbs()

        kwargs = {
            'U': U,
            'V': V,
            'workspace': workspace,
            'conditioning': conditioning
        }

        debug = True
        dprint( 'initialDist:', self.pi0, use=debug )
        dprint( 'transition:', self.pi, use=debug )
        dprint( 'P(y|x):', self.L, use=debug )

        # Run the message passing algorithm over the graph
        self.messagePassing( self.uFilter, self.vFilter, **kwargs )

        # Integrate out the nodes that we cut
        self.integrateOutConditioning( U, V, conditioning, workspace )

        # Update the filter probs for the cut nodes
        self.filterCutNodes( U, V, conditioning, workspace )

        # print( np.exp( U ) )
        # print( np.exp( V[ 2 ] ) )

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
        parents = self.parents( node )
        upEdge = self.upEdges( node )

        nParents = parents.shape[ 0 ]

        # Down to this node
        nodeTransition = self.transitionProb( node, parents )
        transitionAxes = self.sequentialAxes( N=len( nodeTransition.shape ) )

        # Out from each sibling
        siblingTerms = [ self.b( U, V, s, debug=debug ) for s in siblings ]
        siblingAxes = [ self.sequentialAxes( N=len( nodeTransition.shape ) ) for _ in siblings ]

        # Out from each parent
        parentTerms = [ self.a( U, V, p, upEdge, debug=debug ) for p in parents ]
        parentAxes = [ self.ithAxis( i ) for i in range( nParents ) ]

        # Down this node
        v = self.vData( V, node )
        vAxes = [ self.lastAxis( N=nParents ) for _ in v ]

        return self.multiplyTerms( ( *parentTerms, *siblingTerms, *v, nodeTransition ), axes=( *parentAxes, *siblingAxes, *vAxes, transitionAxes ) )

    def _jointParents( self, U, V, node, debug=True ):
        # P( x_p1..pN | Y )

        parents = self.parents( node )
        siblings = self.siblings( node )
        upEdge = self.upEdges( node )

        # Down each child
        siblingTerms = [ self.b( U, V, s, debug=debug ) for s in siblings ]
        siblingAxes = [ self.sequentialAxes( N=parents.shape[ 0 ] ) for s in siblings ]

        # Down this node
        nodeTerm = self.b( U, V, node, debug=debug )
        nodeAxes = self.sequentialAxes( N=parents.shape[ 0 ] )

        # Out from each parent
        parentTerms = [ self.a( U, V, p, upEdge, debug=debug ) for p in parents ]
        parentAxes = [ self.ithAxis( i ) for i in range( parents.shape[ 0 ] ) ]

        return self.multiplyTerms( ( nodeTerm, *parentTerms, *siblingTerms ), axes=( nodeAxes, *parentAxes, *siblingAxes ) )

    ######################################################################

    def nodeJoint( self, U, V, nodes ):
        # P( x, Y )
        return [ self._nodeJoint( U, V, node ) for node in nodes ]

    def jointParentChild( self, U, V, nodes ):
        # P( x_c, x_p1..pN, Y )
        return [ self._jointParentChild( U, V, node ) for node in nodes ]

    def jointParents( self, U, V, nodes ):
        # P( x_p1..pN | Y )
        return [ self._jointParents( U, V, node ) for node in nodes ]

    def marginalProb( self, U, V ):
        # P( Y )
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
            if( self.nParents( node ) == 0 ):
                _ans = self.nodeSmoothed( U, V, [ node ], returnLog=True )
            else:
                jpc = self._jointParentChild( U, V, node )
                jp = self._jointParents( U, V, node )
                # print( '\n\n\n' )
                # print( 'jpc\n', jpc )
                # print( '\n\n\n' )
                # print( 'jp\n', jp )
                # assert 0
                _ans =  jpc - jp
            if( returnLog == True ):
                ans.append( _ans )
            else:
                ans.append( np.exp( _ans ) )

        return ans
