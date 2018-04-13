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

        # This isn't a valid thing to do! Can't integrate V because
        # doesn't make sense to integrate over something being conditioned on
        assert 0

        V_row, V_col, V_data = V

        # ( node, latent state )
        normalAxes = ( 0, 1 )
        allAxes = ( 0, 1 ) + self.fbsAxes( 2 )

        print( '\n-----------------------\n')
        print( '\n\n\n\nU before integration\n', U )
        print( '\n-----------------------\n')
        print( '\nV before integration\n', V_data )
        print( '\n-----------------------\n')

        newU, _ = self.integrate( U, integrandAxes=allAxes, axes=self.fbsAxes( 2 ) )
        newV, _ = self.integrate( V_data, integrandAxes=allAxes, axes=self.fbsAxes( 2 ) )

        print( '\n\n\n\nnewU', newU )
        print( 'newV', newV )

        return newU, ( V_row, V_col, newV )

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

    def fbsAxes( self, N ):
        return tuple( np.arange( len( self.fbs ), dtype=int ) + N )

    def firstAxis( self, node, N, withoutFBS=False ):
        axesForNode = tuple() if self.inFBS( node ) else ( 0, )
        return axesForNode if withoutFBS else axesForNode + self.fbsAxes( N )

    def lastAxis( self, node, N, withoutFBS=False ):
        axesForNode = tuple() if self.inFBS( node ) else ( N - 1, )
        return axesForNode if withoutFBS else axesForNode + self.fbsAxes( N )

    def ithAxis( self, node, i, N, withoutFBS=False ):
        axesForNode = tuple() if self.inFBS( node ) else ( i, )
        return axesForNode if withoutFBS else axesForNode + self.fbsAxes( N )

    def sequentialAxes( self, nodes, nodeOrder, N, withoutFBS=False, skip=None ):

        assert len( nodes ) == len( nodeOrder )
        # assert N  == len( nodes )

        fbsAxes = list( self.fbsAxes( N ) )

        axes = []
        for n, i in zip( nodes, nodeOrder ):
            if( np.any( np.in1d( skip, i ) ) ):
                continue

            # if( self.inFBS( n ) and False ):
            #     indexInFBS = self.fbs.tolist().index( n )
            #     axes.append( indexInFBS + N )
            #     del fbsAxes[ indexInFBS ]
            # else:
            axes.append( i )

        return axes if withoutFBS else axes + fbsAxes

    def axesLike( self, array ):
        return list( range( len( array.shape ) ) )

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
            nVals = 1 if edges is None else len( edges )
            depth = self.fbs.shape[ 0 ] + 1 if node not in self.fbs else self.fbs.shape[ 0 ]
            baseCase = np.zeros( ( nVals, *( ( self.K, ) * ( depth ) ) ) )
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

    def aFBS( self, U, V, node, downEdge, debug=True ):
        # Compute a when node is in the feedback set.
        # This case is different from the regular one because we are treating
        # the node like it is cut from the graph.
        # So we can ignore the edges from node and just return a matrix that
        # is only 1 when node_x == fbs_node_x and 0 otherwise so be consistent
        # with the conditioning
        assert 0

    def a( self, U, V, node, downEdge, maxDim, targetAxis, debug=True ):
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
        #   - maxDim   : The max dimension until we hit the fbs axes
        #
        # Outputs:
        #   - term     : shape should be ( K, ) * ( F + 1 ) where K is the latent state size and F is the size of the fbs
        #   - termAxes : true axes that each of term's axes spans

        dprint( '\n\nComputing a for', node, 'at downEdge', downEdge, use=debug )

        axis = self.ithAxis( node=node, i=targetAxis, N=maxDim )

        if( node in self.fbs ):
            return self.aFBS( U, V, node, downEdge, debug=debug ), ( targetAxis, ) + self.fbsAxes( maxDim )

        # Get the U data for node
        u = self.uData( U, node )
        uAxes = axis
        dprint( 'u:\n', u, use=debug )
        dprint( 'uAxes:\n', uAxes, use=debug )

        # Get the V data over all down edges except downEdge
        v = self.vData( V, node, edges=self.downEdges( node, skipEdges=downEdge ) )
        vAxes = [ axis for _ in v ]
        dprint( 'v:\n', v, use=debug )
        dprint( 'vAxes:\n', vAxes, use=debug )

        # Multiply U and all the Vs
        term, termAxes = self.multiplyTerms( terms=( u, *v ), axes=( uAxes, *vAxes ), ndim=maxDim + self.fbs.shape[ 0 ] )
        dprint( 'term:\n', term, use=debug )
        dprint( 'termAxes:\n', termAxes, use=debug )

        assert isinstance( term, np.ndarray ) and np.any( np.isnan( term ) ) == False
        assert np.all( np.array( termAxes ) == np.array( axis ) ), 'termAxes: %s axis: %s'%( termAxes, axis )
        assert len( term.shape ) == len( termAxes )

        return term, termAxes

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
        nParents = parents.shape[ 0 ]
        totalDim = nParents + 1
        finalDim = nParents + self.fbs.shape[ 0 ]

        nodeOrder = nParents if node not in self.fbs else nParents + self.fbs.tolist().index( node ) + 1

        lastAxis = self.lastAxis( node=node, N=totalDim )

        # Get the transition matrix between node and parents
        nodeTransition = self.transitionProb( node, parents )
        transitionAxes = self.sequentialAxes( nodes=np.hstack( ( parents, node ) ), \
                                              nodeOrder=np.hstack( ( parentOrder, nodeOrder ) ), \
                                              N=totalDim, \
                                              withoutFBS=True )
        dprint( 'nodeTransition:\n', nodeTransition, use=debug )
        dprint( 'transitionAxes:\n', transitionAxes, use=debug )

        # Get the emission vector for node
        nodeEmission = self.emissionProb( node )
        emissionAxes = self.lastAxis( node=node, N=totalDim, withoutFBS=( not node in self.fbs ) )
        dprint( 'nodeEmission:\n', nodeEmission, use=debug )
        dprint( 'emissionAxes:\n', emissionAxes, use=debug )

        # Get the V data over all down edges from node
        v = self.vData( V, node )
        vAxes = [ lastAxis for _ in v ]
        dprint( 'v:\n', v, use=debug )
        dprint( 'vAxes:\n', vAxes, use=debug )

        # Multiply together the transition, emission and Vs
        integrand, integrandAxes = self.multiplyTerms( terms=( nodeEmission, *v, nodeTransition ), \
                                                       axes=( emissionAxes, *vAxes, transitionAxes ), \
                                                       ndim=totalDim + self.fbs.shape[ 0 ] )
        intAxes = self.lastAxis( node=node, N=totalDim, withoutFBS=True )
        dprint( 'integrand:\n', integrand, use=debug )
        dprint( 'intAxes:\n', intAxes, use=debug )

        # Integrate over the node's latent states
        term, termAxes = self.integrate( integrand, integrandAxes, axes=intAxes )
        dprint( 'term:\n', term, use=debug )
        dprint( 'termAxes:\n', termAxes, use=debug )

        assert isinstance( term, np.ndarray ) and np.any( np.isnan( term ) ) == False

        # Want term to have a dense shape and then need to specify the
        # true axes that each axis in term spans
        assert np.any( np.array( term.shape ) == 1 ) == False
        assert len( term.shape ) == len( termAxes )

        return term, termAxes

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

        # Get the transition matrix between node and parents
        nodeTransition = self.transitionProb( node, parents )
        transitionAxes = self.sequentialAxes( nodes=np.hstack( ( parents, node ) ), \
                                              nodeOrder=np.hstack( ( parentOrder, nParents ) ), \
                                              N=totalDim, \
                                              withoutFBS=True )
        dprint( 'nodeTransition:\n', np.exp( nodeTransition ), use=debug )
        dprint( 'transitionAxes:\n', transitionAxes, use=debug )

        # Get the a values for all parents (skip this nodes up edge)
        parentTerms, parentAxes = list( zip( *[ self.a( U, V, p, upEdge, maxDim=totalDim, targetAxis=i, debug=debug ) for p, i in zip( parents, parentOrder ) ] ) )
        dprint( 'parentTerms:\n', np.exp( parentTerms ), use=debug )
        dprint( 'parentAxes:\n', parentAxes, use=debug )

        # Get the b values for all siblings.  These are all over the parents' axes
        if( len( siblings ) > 0 ):
            siblingTerms, siblingAxes = list( zip( *[ self.b( U, V, s, debug=debug ) for s in siblings ] ) )
        else:
            siblingTerms, siblingAxes = np.array( [] ), []
        dprint( 'siblingTerms:\n', np.exp( siblingTerms ), use=debug )
        dprint( 'siblingAxes:\n', siblingAxes, use=debug )

        # Multiply all of the terms together
        integrand, integrandAxes = self.multiplyTerms( terms=( *siblingTerms, *parentTerms, nodeTransition ), \
                                                       axes=( *siblingAxes, *parentAxes, transitionAxes ), \
                                                       ndim=totalDim + self.fbs.shape[ 0 ] )
        intAxes = self.sequentialAxes( nodes=parents, nodeOrder=parentOrder, N=nParents, withoutFBS=True )
        dprint( 'integrand:\n', np.exp( integrand ), use=debug )
        dprint( 'integrandAxes:\n', integrandAxes, use=debug )
        dprint( 'intAxes:\n', intAxes, use=debug )

        # Integrate out the parent latent states
        nodeTerms, nodeTermAxes = self.integrate( integrand, integrandAxes, axes=intAxes )
        nodeTermAxis = self.firstAxis( node=node, N=totalDim )
        dprint( 'nodeTerms:\n', np.exp( nodeTerms ), use=debug )
        dprint( 'nodeTermAxes:\n', nodeTermAxes, use=debug )
        dprint( 'nodeTermAxis:\n', nodeTermAxis, use=debug )

        # Get the emission vector for node
        nodeEmission = self.emissionProb( node )
        nodeEmissionAxis = self.firstAxis( node=node, N=totalDim, withoutFBS=True )
        dprint( 'nodeEmission:\n', np.exp( nodeEmission ), use=debug )
        dprint( 'nodeEmissionAxis:\n', nodeEmissionAxis, use=debug )

        # Combine this nodes emission with the rest of the calculation
        newU, newUAxes = self.multiplyTerms( terms=( nodeTerms, nodeEmission ), \
                                             axes=( nodeTermAxis, nodeEmissionAxis ), \
                                             ndim=totalDim + self.fbs.shape[ 0 ] )
        dprint( 'newU:\n', np.exp( newU ), use=debug )
        dprint( 'newUAxes:\n', newUAxes, use=debug )

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
        parents, parentOrder = self.parents( children[ 0 ], getOrder=True )
        dprint( 'mates:\n', mates, use=debug )
        dprint( 'children:\n', children, use=debug )

        nMates = mates.shape[ 0 ]
        nParents = mates.shape[ 0 ] + 1
        totalDim = nParents + 1

        thisNodesOrder = np.setdiff1d( np.arange( nParents ), mateOrder )

        allAxes = self.sequentialAxes( nodes=parents, \
                                       nodeOrder=parentOrder, \
                                       N=nParents )
        integrationAxes = self.sequentialAxes( nodes=parents, \
                                               nodeOrder=parentOrder, \
                                               N=totalDim, \
                                               skip=thisNodesOrder, \
                                               withoutFBS=True )

        # Get the a values for each of the mates (skip edge)
        if( len( mates ) > 0 ):
            mateTerms, mateAxes = list( zip( *[ self.a( U, V, m, edge, maxDim=totalDim, targetAxis=i, debug=debug ) for m, i in zip( mates, mateOrder ) ] ) )
        else:
            mateTerms, mateAxes = np.array( [] ), []
        dprint( 'mateTerms:\n', mateTerms, use=debug )
        dprint( 'mateAxes:\n', mateAxes, use=debug )

        # Get the b values for each of the children.  These are all over the parents' axes
        childTerms, childAxes = list( zip( *[ self.b( U, V, c, debug=debug ) for c in children ] ) )
        dprint( 'childTerms:\n', childTerms, use=debug )
        dprint( 'childAxes:\n', childAxes, use=debug )

        # Combine the terms
        integrand, integrandAxes = self.multiplyTerms( terms=( *childTerms, *mateTerms ), \
                                                       axes=( *childAxes, *mateAxes ), \
                                                       ndim=totalDim + self.fbs.shape[ 0 ] )
        intAxes = integrationAxes
        dprint( 'integrand:\n', integrand, use=debug )
        dprint( 'integrandAxes:\n', integrandAxes, use=debug )
        dprint( 'intAxes:\n', intAxes, use=debug )

        # Integrate out the mates latent states
        newV, newVAxes = self.integrate( integrand, integrandAxes, axes=intAxes )
        newV = newV.squeeze() # Ravel here so that we end up aligned on 0th axis
        dprint( 'newV:\n', newV, use=debug )
        dprint( 'newVAxes:\n', newVAxes, use=debug )

        assert isinstance( newV, np.ndarray )
        assert np.any( np.isnan( newV ) ) == False

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

        # Run the message passing algorithm over the graph.
        # We will end up with the extended U and V values for all of
        # the nodes except the feedback set nodes.
        # To get the filtered probs for the smoothed probs for the
        # feedback set nodes, marginalize over the non fbs nodes
        # at any extended smoothed prob
        self.messagePassing( self.uFilter, self.vFilter, **kwargs )

        return U, V

    ######################################################################

    def _fullJoint( self, U, V, node, debug=True ):
        # P( x, x_{feedback set}, Y )

        parents, parentOrder = self.parents( node, getOrder=True )
        totalDim = parents.shape[ 0 ] + 1

        firstAxis = self.firstAxis( node=node, N=totalDim )

        u = self.uData( U, node )
        uAxes = firstAxis
        dprint( 'u:\n', u, use=debug )
        dprint( 'uAxes:\n', uAxes, use=debug )

        v = self.vData( V, node )
        vAxes = [ firstAxis for _ in v ]
        dprint( 'v:\n', v, use=debug )
        dprint( 'vAxes:\n', vAxes, use=debug )

        joint, jointAxes =  self.multiplyTerms( terms=( u, *v ), \
                                                axes=( uAxes, *vAxes ), \
                                                ndim=totalDim + self.fbs.shape[ 0 ] )
        dprint( 'joint:\n', joint, use=debug )
        dprint( 'jointAxes:\n', jointAxes, use=debug )
        return joint, jointAxes

    def _nodeJointForFBS( self, U, V, node, debug=True ):
        # If node is a root, get a child and if it is a leaf, get a parent
        if( self.nParents( node ) == 0 ):
            useNode = self.children( node )[ 0 ]
        else:
            useNode = self.parents( node )[ 0 ]

        joint, jointAxes = self._fullJoint( U, V, useNode, debug=debug )

        # Integration axes are all axes but node
        fbsIndex = self.fbs.tolist().index( node )
        intAxes = [ i for i in ( 0, ) + self.fbsAxes( 1 ) if i != fbsIndex ]
        dprint( 'intAxes:\n', intAxes, use=debug )
        intJoint, _ = self.integrate( joint, jointAxes, axes=intAxes )
        print( '\n\n\n\n\n' )
        dprint( 'intJoint:\n', intJoint, '->', np.logaddexp.reduce( intJoint ), use=debug )
        print( '\n\n\n\n\n' )
        return intJoint

    def _nodeJoint( self, U, V, node, debug=True ):
        # P( x, Y )

        if( node in self.fbs ):
            return self._nodeJointForFBS( U, V, node, debug=debug )

        joint, jointAxes = self._fullJoint( U, V, node, debug=debug )

        intJoint, _ = self.integrate( joint, jointAxes, axes=self.fbsAxes( 0 ) )
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
        nodeSpan = np.hstack( ( parents, node ) )
        spanOrder = np.hstack( ( parentOrder, nParents ) )
        allAxes = self.sequentialAxes( nodes=nodeSpan, nodeOrder=spanOrder, N=totalDim )
        upToLastAxes = self.sequentialAxes( nodes=parents, nodeOrder=parentOrder, N=nParents )
        lastAxis = self.lastAxis( node=node, N=totalDim )

        # Down to this node
        nodeTransition = self.transitionProb( node, parents )
        transitionAxes = allAxes
        nodeEmission = self.emissionProb( node )
        emissionAxes = lastAxis

        # Out from each sibling
        siblingTerms = [ self.b( U, V, s, debug=debug )[ 0 ] for s in siblings ]
        siblingAxes = [ upToLastAxes for _ in siblings ]

        # Out from each parent
        parentTerms, parentAxes = list( zip( *[ self.a( U, V, p, upEdge, maxDim=totalDim, targetAxis=i, debug=debug ) for p, i in zip( parents, parentOrder ) ] ) )

        # Down this node
        v = self.vData( V, node )
        vAxes = [ lastAxis for _ in v ]

        return self.multiplyTerms( terms=( *parentTerms, *siblingTerms, *v, nodeTransition, nodeEmission ), \
                                   axes=( *parentAxes, *siblingAxes, *vAxes, transitionAxes, emissionAxes ), \
                                   ndim=totalDim )[ 0 ]

    def _jointParents( self, U, V, node, debug=True ):
        # P( x_p1..pN, Y )

        parents, parentOrder = self.parents( node, getOrder=True )
        siblings = self.siblings( node )
        upEdge = self.upEdges( node )

        nParents = parents.shape[ 0 ]
        totalDim = nParents + 1
        upToLastAxes = self.sequentialAxes( nodes=parents, nodeOrder=parentOrder, N=nParents )

        # Down each child
        siblingTerms = [ self.b( U, V, s, debug=debug )[ 0 ] for s in siblings ]
        siblingAxes = [ upToLastAxes for s in siblings ]

        # Down this node
        nodeTerm = self.b( U, V, node, debug=debug )[ 0 ]
        nodeAxes = upToLastAxes

        # Out from each parent
        parentTerms, parentAxes = list( zip( *[ self.a( U, V, p, upEdge, maxDim=totalDim, targetAxis=i, debug=debug ) for p, i in zip( parents, parentOrder ) ] ) )

        return self.multiplyTerms( terms=( nodeTerm, *parentTerms, *siblingTerms ), \
                                   axes=( nodeAxes, *parentAxes, *siblingAxes ), \
                                   ndim=nParents )[ 0 ]

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
        ans = np.logaddexp.reduce( self._nodeJoint( U, V, leafIndex ) )
        print( 'ANS', ans )
        return ans

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
                parents, parentOrder = self.parents( node, getOrder=True )
                nodeSpan = np.hstack( ( parents, node ) )
                spanOrder = np.hstack( ( parentOrder, nParents ) )
                jpc = self._jointParentChild( U, V, node )
                jpcAxes = self.sequentialAxes( nodes=nodeSpan, nodeOrder=spanOrder, N=nParents + 1 )

                jp = -self._jointParents( U, V, node )
                jpAxes = self.sequentialAxes( nodes=parents, nodeOrder=parentOrder, N=nParents )

                print( 'node', node )
                print( 'jpc', jpc )
                print( 'jp', jp )

                _ans = self.multiplyTerms( terms=( jpc, jp ), \
                                           axes=( jpcAxes, jpAxes ), \
                                           ndim=nParents+1 )[ 0 ]

                if( returnLog == True ):
                    ans.append( _ans )
                else:
                    ans.append( np.exp( _ans ) )

        return ans
