from GenModels.GM.States.GraphicalMessagePassing.GraphicalMessagePassingBase import Graph, GraphMessagePasser, dprint
import numpy as np
from scipy.sparse import coo_matrix
from functools import reduce
from collections import Iterable
import itertools
import inspect


__INDENT = 0

def debugWorkFlow( function ):

    def wrapper( *args, **kwargs  ):

        global __INDENT

        tab = '        '

        ind = tab * __INDENT
        __INDENT += 1

        sig = inspect.signature( function )

        namedArgs = []
        skip = [ 'self', 'cls', 'U', 'V', 'debug', 'nodeTransition' ]

        for name, val in zip( sig.parameters, args ):
            if( name in skip ):
                continue
            namedArgs.append( ( name, val ) )

        for name, val in kwargs.items():
            if( name in skip ):
                continue
            namedArgs.append( ( name, val ) )

        print( ind, '------- Start %s -------'%function.__name__, flush=True )

        print( ind, 'Computing', function.__name__, 'with args:', flush=True )
        for name, val in namedArgs:
            if( isinstance( val, np.ndarray ) ):
                val = str( val ).replace( '\n', '\n' + ind )
            print( ind, name, ':\n', ind, val, flush=True )

        print( ind, '------- Compute %s -------'%function.__name__, flush=True )

        returnVals, printVals = function( *args, **kwargs )

        for key, val in printVals.items():
            if( isinstance( val, np.ndarray ) ):
                val = str( val ).replace( '\n', '\n' + ind )
            print( ind, key, ':\n', ind, val, flush=True )

        print( ind, '------- End %s -------'%function.__name__, flush=True )

        __INDENT -= 1

        return returnVals

    return wrapper

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
                V_data[ i ][ 0 ][ : ] = val

    def vDataFromMask( self, V, mask ):
        _, _, V_data = V
        ans = []
        for i, maskValue in enumerate( mask ):
            if( maskValue == True ):
                ans.append( V_data[ i ] )
        return ans

    ######################################################################

    def inFBS( self, node ):
        return np.any( np.in1d( self.fbs, node ) )

    @classmethod
    @debugWorkFlow
    def extendAxes( cls, term, targetAxis, maxDim ):
        # Push the first axis out to targetAxis, but don't change
        # the axes past maxDim

        term, fbsAxisStart = term

        originalShape = term.shape

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

        returnVals = ( term, fbsAxisStart )
        printVals = {
            'originalShape': originalShape,
            'returnVals[0].shape': returnVals[ 0 ].shape,
            'returnVals[1]': returnVals[ 1 ]
        }

        printVals = {}

        return returnVals, printVals

    ######################################################################

    def nParents( self, node ):
        return self.parents( node ).shape[ 0 ]

    def nChildren( self, node ):
        return self.children( node ).shape[ 0 ]

    def uData( self, U, node ):
        return U[ node ]

    def vData( self, V, node, edges=None, ndim=None, debug=False ):
        V_row, V_col, V_data = V

        if( self.inFBS( node ) ):
            # assert ndim is not None
            # If node is part of the skip array (if its in the fbs)
            ans = [ ( np.zeros( self.K ), 0 ) ]
        elif( ~np.any( np.in1d( V_row, node ) ) ):
            # If the node isn't in the V object (node is a leaf)
            ans = [ ( np.array( [] ), -1 ) ]
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
            return [ ( np.array( [] ), -1 ) ]
        else:
            # Only looking at one edge
            ans = self.vDataFromMask( V, np.in1d( V_col, edges ) )

        if( len( ans ) == 0 ):
            # If we're looking at a leaf, return all 0s
            nVals = 1 if edges is None else len( edges )
            ans = []
            for _ in range( nVals ):
                ans.append( ( np.zeros( self.K ), -1 ) )

        assert sum( [ 0 if ~np.any( np.isnan( v ) ) else 1 for v, _ in ans ] ) == 0, ans

        return ans

    ######################################################################

    @debugWorkFlow
    def a( self, U, V, node, downEdge, purpose, debug=True ):
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
        #   - purpose  : Whether this is called for U or V.  This matters because in U we need the fbs axes to start
        #                after the dims of the parents + child for a node, but for V we only need them to start after
        #                the dims of the parents
        #
        # Outputs:
        #   - term     : shape should be ( K, ) * ( F + 1 ) where K is the latent state size and F is the size of the fbs

        if( node in self.fbs ):
            term = ( np.array( [] ), 0 )
            return term, { 'term[0].shape': term[ 0 ].shape, 'term[1]': term[ 1 ] }

        # U over node
        # V over node for each down edge that isn't downEdge
        # Multiply U and all Vs
        u = self.uData( U, node )
        vs = self.vData( V, node, edges=self.downEdges( node, skipEdges=downEdge ) )
        term = self.multiplyTerms( terms=( u, *vs ) )

        returnVals = term
        printVals = {
            'u[0].shape': u[ 0 ].shape,
            'u[1]': u[ 1 ],
            'vs[0].shape': [ v[ 0 ].shape for v in vs ],
            'vs[1]': [ v[ 1 ] for v in vs ],
            'term[0].shape': term[ 0 ].shape,
            'term[1]': term[ 1 ]
        }

        return returnVals, printVals

    ######################################################################

    @debugWorkFlow
    def b( self, U, V, node, purpose, debug=True ):
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

        parents, parentOrder = self.parents( node, getOrder=True )
        nParents = parents.shape[ 0 ]

        # Transition prob to node
        # Emission prob of node (aligned on last axis)
        nodeTransition = self.transitionProb( node, parents, parentOrder )

        nodeEmission = self.emissionProb( node )
        if( node not in self.fbs ):
            nodeEmission = self.extendAxes( nodeEmission, nParents, nParents + 1 )

        # V over node for all down edges (aligned on last axis)
        vs = self.vData( V, node )
        if( node not in self.fbs ):
            vs = [ self.extendAxes( v, nParents, nParents + 1 ) for v in vs ]
        else:
            vs = []

        # Multiply together the transition, emission and Vs
        # Integrate over node's (last) axis unless node is in the fbs
        # Integrate over the node's latent states which is last axis
        integrand = self.multiplyTerms( terms=( nodeTransition, nodeEmission, *vs ) )
        intAxes = [ nParents ] if node not in self.fbs else []
        term = self.integrate( integrand, axes=intAxes )

        returnVals = term
        printVals = {
            'parents': parents,
            'parentOrder': parentOrder,
            'nodeTransition[0].shape': nodeTransition[ 0 ].shape,
            'nodeTransition[ 1 ]': nodeTransition[ 1 ],
            'nodeEmission[ 0 ].shape': nodeEmission[ 0 ].shape,
            'nodeEmission[ 1 ]': nodeEmission[ 1 ],
            'vs[0].shape': [ v[ 0 ].shape for v in vs ],
            'vs[1]': [ v[ 1 ] for v in vs ],
            'integrand[0].shape': integrand[ 0 ].shape,
            'integrand[1]': integrand[ 1 ],
            'intAxes': intAxes,
            'term[0].shape': term[ 0 ].shape,
            'term[1]': term[ 1 ]
        }

        return returnVals, printVals

    ######################################################################

    @debugWorkFlow
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

        parents, parentOrder = self.parents( node, getOrder=True )
        siblings = self.siblings( node )

        nParents = parents.shape[ 0 ]

        # Transition prob to node
        nodeTransition = self.transitionProb( node, parents, parentOrder )

        # A over each parent (aligned according to ordering of parents)
        parentAs = [ self.a( U, V, p, self.upEdges( node ), purpose='U', debug=debug ) for p in parents ]
        parentAs = [ self.extendAxes( a, i, nParents ) for a, i in zip( parentAs, parentOrder ) if a[ 0 ].size > 0 ]

        # B over each sibling
        siblingBs = [ self.b( U, V, s, purpose='U', debug=debug ) for s in siblings ]

        # Multiply all of the terms together
        # Integrate out the parents (unless they are in the fbs)
        # Integrate out the parent latent states
        integrand = self.multiplyTerms( terms=( nodeTransition, *parentAs, *siblingBs ) )
        intAxes = [ i for i, parent in zip( parentOrder, parents ) if parent not in self.fbs ]
        nodeTerms = self.integrate( integrand, axes=intAxes )

        # Squeeze all of the left most dims.  This is because if a parent is in
        # the fbs and we integrate out the other one, there will be an empty dim
        # remaining
        while( nodeTerms[ 0 ].shape[ 0 ] == 1 ):
            nodeTerms = list( nodeTerms )
            nodeTerms[ 0 ] = nodeTerms[ 0 ].squeeze( axis=0 )
            if( nodeTerms[ 1 ] != -1 ):
                nodeTerms[ 1 ] -= 1
            nodeTerms = tuple( nodeTerms )

        # Emission for this node
        nodeEmission = self.emissionProb( node )

        # Combine this nodes emission with the rest of the calculation
        newU = self.multiplyTerms( terms=( nodeTerms, nodeEmission ) )

        returnVals = newU
        printVals = {
            'parents': parents,
            'parentOrder': parentOrder,
            'siblings': siblings,
            'nodeTransition[0].shape': nodeTransition[ 0 ].shape,
            'nodeTransition[ 1 ]': nodeTransition[ 1 ],
            'parentAs[0].shape': [ a[ 0 ].shape for a in parentAs ],
            'parentAs[1]': [ a[ 1 ] for a in parentAs ],
            'siblingBs[0].shape': [ b[ 0 ].shape for b in siblingBs ],
            'siblingBs[1]': [ b[ 1 ] for b in siblingBs ],
            'integrand[0].shape': integrand[ 0 ].shape,
            'integrand[ 1 ]': integrand[ 1 ],
            'intAxes': intAxes,
            'nodeTerms[0].shape': nodeTerms[ 0 ].shape,
            'nodeTerms[ 1 ]': nodeTerms[ 1 ],
            'nodeEmission[0].shape': nodeEmission[ 0 ].shape,
            'nodeEmission[1]': nodeEmission[ 1 ],
            'newU[ 0 ].shape': newU[ 0 ].shape,
            'newU[ 1 ]': newU[ 1 ]
        }

        return returnVals, printVals

    ######################################################################

    @debugWorkFlow
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

        mates, mateOrder = self.mates( node, getOrder=True, edges=edge )
        children = self.children( node, edges=edge )
        parents, parentOrder = self.parents( children[ 0 ], getOrder=True )

        nParents = parents.shape[ 0 ]

        # A values over each mate (aligned according to ordering of mates)
        mateAs = [ self.a( U, V, m, edge, purpose='V', debug=debug ) for m in mates ]
        mateAs = [ self.extendAxes( a, i, nParents ) for a, i in zip( mateAs, mateOrder ) if a[ 0 ].size > 0  ]

        # B over each child
        childBs = [ self.b( U, V, c, purpose='V', debug=debug ) for c in children ]

        # Multiply all of the terms together
        integrand = self.multiplyTerms( terms=( *childBs, *mateAs ) )

        # Integrate out the mates.  It is important to note that we even integrate
        # over fbs nodes.  This is because when we're computing V, we've conditioned on
        # the fbs nodes.  However for the computation of b, we condition on the different
        # configurations of the fbs nodes.  So we integrate over all possible values
        # for the fbs nodes (which is why a is 2d for the fbs mates).
        intAxes = [ i for i, mate in zip( mateOrder, mates ) if mate not in self.fbs ]

        # Integrate out the mates latent states
        newV = self.integrate( integrand, axes=intAxes )

        # Squeeze all of the left most dims.  This is because if a parent is in
        # the fbs and we integrate out the other one, there will be an empty dim
        # remaining
        while( newV[ 0 ].shape[ 0 ] == 1 ):
            newV = list( newV )
            newV[ 0 ] = newV[ 0 ].squeeze( axis=0 )
            if( newV[ 1 ] != -1 ):
                newV[ 1 ] -= 1
            newV = tuple( newV )

        returnVals = newV
        printVals = {
            'mates': mates,
            'mateOrder': mateOrder,
            'children': children,
            'mateAs[0].shape': [ a[ 0 ].shape for a in mateAs ],
            'mateAs[1]': [ a[ 1 ] for a in mateAs ],
            'childBs[0].shape': [ b[ 0 ].shape for b in childBs ],
            'childBs[1]': [ b[ 1 ] for b in childBs ],
            'integrand[0].shape': integrand[ 0 ].shape,
            'integrand[1]': integrand[ 1 ],
            'intAxes': intAxes,
            'newV[ 0 ].shape': newV[ 0 ].shape,
            'newV[ 1 ]': newV[ 1 ]
        }

        return returnVals, printVals

    ######################################################################

    def uFilter( self, baseCase, nodes, U, V, workspace, debug=True ):
        # Compute P( ↑( n )_y, n_x )
        # Probability of all emissions that can be reached by going up node's up edge

        dprint( '\n\nComputing U for', nodes, use=debug )

        newU = []
        for node in nodes:
            if( self.nParents( node ) == 0 ):
                u = self.uBaseCase( node, debug=debug )
            else:
                u = self.u( U, V, node, debug=debug )
            newU.append( u )

        self.updateU( nodes, newU, U )

    def vFilter( self, baseCase, nodesAndEdges, U, V, workspace, debug=True ):

        nodes, edges = nodesAndEdges

        dprint( '\n\nComputing V for', nodes, 'at edges', edges, use=debug )

        newV = []
        for node, edge in zip( nodes, edges ):
            if( self.nChildren( node ) == 0 ):
                assert edge == None
                v = self.vBaseCase( node, debug=debug )
            else:
                assert edge is not None
                v = self.v( U, V, node, edge, debug=debug )
            newV.append( v )

        self.updateV( nodes, edges, newV, V )

    def convergence( self, nodes ):
        return False

    ######################################################################

    @debugWorkFlow
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

        returnVals = ( U, V )
        printVals = {
            'U[0].shape': [ ( n, u[ 0 ].shape ) for n, u in enumerate( U ) ],
            'U[1]': [ ( n, u[ 1 ] ) for n, u in enumerate( U ) ],
            'V[0].shape': [ ( n, e, v[ 0 ].shape ) for n, e, v in zip( *V ) ],
            'V[1].shape': [ ( n, e, v[ 1 ] ) for n, e, v in zip( *V ) ]
        }

        return returnVals, printVals

    ######################################################################

    @debugWorkFlow
    def _statHelperForFBS( self, U, V, node, fullFunction, saveParents, saveChild, debug=True ):

        parents, parentOrder = self.parents( node, getOrder=True )
        nParents = parents.shape[ 0 ]

        if( np.all( np.in1d( parents, self.fbs ) ) and saveParents ):
            assert 0, 'fuq'

        if( node in self.fbs and saveChild ):
            useNode = None
            # Use a sibling
            for _node in self.siblings( node ):
                if( _node not in self.fbs ):
                    useNode = _node
                    break
            if( useNode is None ):
                if( saveChild == True and saveParents == False ):
                    # Can just choose from any node
                    for _node in itertools.chain( self.parents( node ), self.mates( node ), self.siblings( node ), self.children( node ) ):
                        if( _node not in self.fbs ):
                            useNode = _node
                            break
                elif( saveChild == True and saveParents == True ):
                    # Calculate jointParent instead of jointParentChild but just don't
                    # integrate out this node
                    pass

            assert useNode is not None, 'fuq!'
        else:
            useNode = node

        joint = fullFunction( U, V, useNode, debug=debug )

        # Want to integrate out everything that isn't in the fbs
        if( saveParents and saveChild ):
            if( node in self.fbs ):
                fbsIndices = joint[ 1 ] + np.hstack( ( self.fbs.tolist().index( node ), np.arange( nParents )[ np.in1d( parents, self.fbs ) ] ) )
            else:
                fbsIndices = joint[ 1 ] + np.arange( nParents )[ np.in1d( parents, self.fbs ) ]
        elif( saveChild ):
            if( node in self.fbs ):
                fbsIndices = np.array( [ joint[ 1 ] + self.fbs.tolist().index( node ) ] )
            else:
                assert 0
        else:
            fbsIndices = joint[ 1 ] + np.arange( nParents )[ np.in1d( parents, self.fbs ) ]
        intAxes = np.setdiff1d( np.arange( joint[ 0 ].ndim ), fbsIndices )

        # Also save the parents that aren't in the fbs if needed
        if( saveParents ):
            intAxes = np.setdiff1d( intAxes, np.arange( nParents - np.in1d( parents, self.fbs ).sum() ) )

        # Integrate out the irrelevant fbs axes
        intJoint = self.integrate( joint, axes=intAxes, ignoreFBSAxis=True )

        returnVals = intJoint
        printVals = {
            'useNode': useNode,
            'joint[0].shape': joint[ 0 ].shape,
            'joint[1]': joint[ 1 ],
            'intAxes': intAxes,
            'intJoint.shape': intJoint.shape
        }

        return returnVals, printVals

    @debugWorkFlow
    def _statHelper( self, U, V, node, fullFunction, saveParents, saveChild, debug=True ):

        parents, parentOrder = self.parents( node, getOrder=True )

        if( ( saveChild and node in self.fbs ) or
            ( saveParents and np.any( np.in1d( parents, self.fbs ) ) ) ):
            intJoint = self._statHelperForFBS( U, V, node, fullFunction, saveParents, saveChild, debug=debug )
            printVals = {
                'intJoint.shape': intJoint.shape
            }
        else:
            joint = fullFunction( U, V, node, debug=debug )
            start = 0
            if( saveParents ):
                start += self.parents( node ).shape[ 0 ]
            if( saveChild ):
                start += 1

            intAxes = list( range( start, joint[ 0 ].ndim ) )
            intJoint = self.integrate( joint, axes=intAxes, ignoreFBSAxis=True )
            printVals = {
                'joint[0].shape': joint[ 0 ].shape,
                'intAxes': intAxes,
                'intJoint.shape': intJoint.shape
            }

        returnVals = intJoint

        return returnVals, printVals

    ######################################################################

    @debugWorkFlow
    def _fullJoint( self, U, V, node, debug=True ):
        # P( x, x_{feedback set}, Y )


        u = self.uData( U, node )

        vs = self.vData( V, node )
        vs = [ self.extendAxes( v, 0, 1 ) for v in vs ]

        fullJoint =  self.multiplyTerms( terms=( u, *vs ) )

        returnVals = fullJoint
        printVals = {
            'u[0].shape': u[ 0 ].shape,
            'u[1]': u[ 1 ],
            'vs[0].shape': [ v[ 0 ].shape for v in vs ],
            'vs[1]': [ v[ 1 ] for v in vs ],
            'fullJoint[0].shape': fullJoint[ 0 ].shape,
            'fullJoint[1]': fullJoint[ 1 ]
        }

        return returnVals, printVals

    @debugWorkFlow
    def _nodeJoint( self, U, V, node, debug=True ):
        # P( x, Y )
        joint = self._statHelper( U, V, node, self._fullJoint, False, True, debug=debug )
        returnVals = joint
        printVals = {
                'joint.shape': joint.shape
            }

        return returnVals, printVals

    ######################################################################

    @debugWorkFlow
    def _fullJointParents( self, U, V, node, debug=True ):
        # P( x_p1..pN, x_{feedback set}, Y )

        parents, parentOrder = self.parents( node, getOrder=True )
        siblings = self.siblings( node )
        upEdge = self.upEdges( node )

        nParents = parents.shape[ 0 ]

        # Down each child
        siblingBs = [ self.b( U, V, s, purpose='V', debug=debug ) for s in siblings ]

        # Down this node
        nodeTerm = self.b( U, V, node, purpose='V', debug=debug )

        # Out from each parent
        parentAs = [ self.a( U, V, p, upEdge, purpose='V', debug=debug ) for p in parents ]
        parentAs = [ self.extendAxes( a, i, nParents ) for a, i in zip( parentAs, parentOrder ) ]

        jointParents = self.multiplyTerms( terms=( nodeTerm, *parentAs, *siblingBs ) )

        intAxes = list( range( 1, jointParents[ 0 ].ndim ) )
        intJoint = self.integrate( jointParents, axes=intAxes )

        returnVals = intJoint
        printVals = {
            'parents': parents,
            'parentOrder': parentOrder,
            'nodeTerm[0].shape': nodeTerm[ 0 ].shape,
            'nodeTerm[1]': nodeTerm[ 1 ],
            'siblingBs[0].shape': [ b[ 0 ].shape for b in siblingBs ],
            'siblingBs[1]': [ b[ 1 ] for b in siblingBs ],
            'parentAs[0].shape': [ a[ 0 ].shape for a in parentAs ],
            'parentAs[1]': [ a[ 1 ] for a in parentAs ],
            'jointParents[0].shape': jointParents[ 0 ].shape,
            'jointParents[1]': jointParents[ 1 ],
            'intJoint[0].shape': intJoint[ 0 ].shape,
            'intJoint[1]': intJoint[ 1 ]
        }

        return returnVals, printVals

    @debugWorkFlow
    def _jointParents( self, U, V, node, debug=True ):
        # P( x_p1..pN, Y )

        joint = self._statHelper( U, V, node, self._fullJointParents, True, False, debug=debug )
        returnVals = joint
        printVals = {
                'joint.shape': joint.shape
            }

        return returnVals, printVals

    ######################################################################

    @debugWorkFlow
    def _fullJointParentChild( self, U, V, node, debug=True ):
        # P( x_c, x_p1..pN, x_{feedback set}, Y )

        siblings = self.siblings( node )
        parents, parentOrder = self.parents( node, getOrder=True )
        upEdge = self.upEdges( node )

        nParents = parents.shape[ 0 ]

        # Down to this node
        nodeTransition = self.transitionProb( node, parents, parentOrder )

        nodeEmission = self.emissionProb( node )
        nodeEmission = self.extendAxes( nodeEmission, nParents, nParents + 1 )

        # Out from each sibling
        siblingBs = [ self.b( U, V, s, purpose='U', debug=debug ) for s in siblings ]

        # Out from each parent
        parentAs = [ self.a( U, V, p, upEdge, purpose='U', debug=debug ) for p in parents ]
        parentAs = [ self.extendAxes( a, i, nParents ) for a, i in zip( parentAs, parentOrder ) ]

        # Down this node
        vs = self.vData( V, node )
        vs = [ self.extendAxes( _v, nParents, nParents + 1 ) for _v in vs ]

        jointParentChild = self.multiplyTerms( terms=( *parentAs, *siblingBs, *vs, nodeTransition, nodeEmission ) )

        returnVals = jointParentChild
        printVals = {
            'parents': parents,
            'parentOrder': parentOrder,
            'nodeTransition[0].shape': nodeTransition[ 0 ].shape,
            'nodeEmission[ 0 ].shape': nodeEmission[ 0 ].shape,
            'siblingBs[0].shape': [ b[ 0 ].shape for b in siblingBs ],
            'parentAs[0].shape': [ a[ 0 ].shape for a in parentAs ],
            'vs[0].shape': [ v[ 0 ].shape for v in vs ],
            'jointParentChild[0].shape': jointParentChild[ 0 ].shape
        }

        return returnVals, printVals

    @debugWorkFlow
    def _jointParentChild( self, U, V, node, debug=True ):
        # P( x_c, x_p1..pN, Y )
        joint = self._statHelper( U, V, node, self._fullJointParentChild, True, True, debug=debug )
        returnVals = joint
        printVals = {
                'joint.shape': joint.shape
            }

        return returnVals, printVals

    ######################################################################

    def nodeJoint( self, U, V, nodes, returnLog=False ):
        # P( x, Y )
        ans = []
        for node in nodes:
            ans.append( self._nodeJoint( U, V, node ) )
        return ans

    def jointParentChild( self, U, V, nodes, returnLog ):
        # P( x_c, x_p1..pN, Y )

        ans = []
        for node in nodes:
            if( self.nParents( node ) > 0 ):
                _ans = self._jointParentChild( U, V, node )

                if( returnLog == True ):
                    ans.append( _ans )
                else:
                    ans.append( np.exp( _ans ) )
        return ans

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

    ######################################################################

    def marginalProb( self, U, V ):
        # P( Y )
        # This isn't true when there are multiple graph components
        randomNode = 0
        marginalProb = np.logaddexp.reduce( self._nodeJoint( U, V, randomNode ) )
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

                _jpc = self._jointParentChild( U, V, node )
                _jp = self._jointParents( U, V, node )
                jpc = ( _jpc, -1 )
                jp = ( -_jp, -1 )

                _ans = self.multiplyTerms( terms=( jpc, jp ) )[ 0 ]

                if( returnLog == True ):
                    ans.append( ( node, _ans ) )
                else:
                    ans.append( ( node, np.exp( _ans ) ) )

        return ans
