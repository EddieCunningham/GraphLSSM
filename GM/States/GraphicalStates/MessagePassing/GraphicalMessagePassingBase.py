import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix

def dprint( *args, use=False ):
    if( use ):
        print( *args )

class Graph():
    # This class is how we make sparse matrices

    def __init__( self ):
        self.nodes = set()
        self.edgeChildren = list()
        self.edgeParents = list()

    def addEdge( self, parents, children ):
        assert isinstance( parents, list ) or isinstance( parents, tuple )
        assert isinstance( children, list ) or isinstance( children, tuple )
        for node in parents + children:
            self.nodes.add( node )

        self.edgeChildren.append( children )
        self.edgeParents.append( parents )

    def _cooMatrixFromNodeEdge( self, nodes, edges ):

        nRows = len( nodes )
        nCols = len( edges )

        rows = []
        cols = []
        data = []

        for i, nodeGroup in enumerate( edges ):
            for node in nodeGroup:
                rowIndex = nodes.index( node )
                colIndex = i

                rows.append( rowIndex )
                cols.append( colIndex )
                data.append( True )

        mask = coo_matrix( ( data, ( rows, cols ) ), shape=( nRows, nCols ), dtype=bool )
        return mask

    def toMatrix( self ):

        nodeList = list( self.nodes )

        parentMask = self._cooMatrixFromNodeEdge( nodeList, self.edgeParents )
        childMask = self._cooMatrixFromNodeEdge( nodeList, self.edgeChildren )

        return parentMask, childMask

class GraphMessagePasser():
    # Base message passing class for hyper graphs.
    # Will use a sparse matrix to hold graph structure

    def __init__( self ):
        pass

    def genFilterProbs( self ):
        assert 0

    def genWorkspace( self ):
        assert 0

    def genChildMasks( self ):
        assert 0

    def updateParams( self, parentMasks, childMasks, feedbackSets ):
        for childMask, parentMask, feedbackSet in zip( childMasks, parentMasks, feedbackSets ):
            assert isinstance( childMask, coo_matrix )
            assert isinstance( parentMask, coo_matrix )
            assert childMask.shape == parentMask.shape and parentMask.shape[ 0 ] == feedbackSet.shape[ 0 ]
        self.pmask = sp.sparse.vstack( parentMasks )
        self.cmask = sp.sparse.vstack( childMasks )
        self.fbsMask = np.concatenate( feedbackSets )

    def transitionProb( self, t, t1 ):
        assert 0

    def emissionProb( self, t ):
        assert 0

    def combineTerms( self, *terms ):
        assert 0

    def integrate( self, integrand, outMem ):
        assert 0

    def upBaseCase( self, leaves ):
        assert 0

    def downBaseCase( self, roots ):
        assert 0

    ######################################################################

    def upEdges( self, nodes, split=False ):
        if( split ):
            return [ self.upEdges( n, split=False ) for n in nodes ]
        rows, cols = self.cmask.nonzero()
        return cols[ np.in1d( rows, nodes ) ]

    def downEdges( self, nodes, skipEdges=None, split=False ):
        if( split ):
            return [ self.downEdges( n, skipEdges=skipEdges, split=False ) for n in nodes ]
        if( skipEdges is not None ):
            return np.setdiff1d( self.downEdges( nodes, skipEdges=None, split=False ), skipEdges )
        rows, cols = self.pmask.nonzero()
        return cols[ np.in1d( rows, nodes ) ]

    def parents( self, nodes, split=False ):
        if( split ):
            return [ self.parents( n, split=False ) for n in nodes ]

        nodeMask = np.in1d( self.cmask.row, nodes )
        parentEdges = np.in1d( self.pmask.col, self.cmask.col[ nodeMask ] )
        return np.unique( self.pmask.row[ parentEdges ] )

    def children( self, nodes, edges=None, split=False ):
        if( split ):
            if( edges is None ):
                return [ self.children( n, split=False ) for n in nodes ]
            else:
                return [ self.children( n, e, split=False ) for n, e in zip( nodes, edges ) ]

        nodeMask = np.in1d( self.pmask.row, nodes )
        if( edges is not None ):
            relevantEdgeMask = np.in1d( self.pmask.col, edges ) & nodeMask
            childEdges = self.pmask.col[ relevantEdgeMask ]
        else:
            childEdges = self.pmask.col[ nodeMask ]
        edgeMask = np.in1d( self.cmask.col, childEdges )
        return np.unique( self.cmask.row[ edgeMask ] )

    def mates( self, nodes, edges=None, split=False ):
        if( split ):
            if( edges is None ):
                return [ self.mates( n, split=False ) for n in nodes ]
            else:
                return [ self.mates( n, e, split=False ) for n, e in zip( nodes, edges ) ]

        nodeMask = np.in1d( self.pmask.row, nodes )
        if( edges is not None ):
            relevantEdgeMask = np.in1d( self.pmask.col, edges ) & nodeMask
            mateEdges = self.pmask.col[ relevantEdgeMask ]
        else:
            mateEdges = self.pmask.col[ nodeMask ]
        edgeMask = np.in1d( self.pmask.col, mateEdges )
        return np.setdiff1d( self.pmask.row[ edgeMask ], nodes )

    def siblings( self, nodes, split=False ):
        if( split ):
            return [ self.siblings( n, split=False ) for n in nodes ]
        return np.setdiff1d( self.cmask.row[ np.in1d( self.cmask.col, self.cmask.col[ np.in1d( self.cmask.row, nodes ) ] ) ], nodes )

    ######################################################################

    def baseCaseNodes( self ):

        M, N = self.pmask.shape

        # Get the number of edges that each node is a parent of
        parentOfEdgeCount = self.pmask.getnnz( axis=1 )

        # Get the number of edges that each node is a child of
        childOfEdgeCount = self.cmask.getnnz( axis=1 )

        # Get the indices of leaves and roots
        rootIndices = np.arange( M )[ ( parentOfEdgeCount != 0 ) & ( childOfEdgeCount == 0 ) ]
        leafIndices = np.arange( M )[ ( childOfEdgeCount != 0 ) & ( parentOfEdgeCount == 0 ) ]

        # Explicitely get the feedback set
        fbs = np.arange( self.fbsMask.shape[ 0 ] )[ self.fbsMask ]

        # Parent of nodes in feedback set are nodes who
        fbsParents = self.parents( fbs )
        fbsChildren = self.children( fbs )

        # Generate the up and down base arrays
        uList = np.setdiff1d( np.hstack( ( rootIndices, fbsChildren ) ), fbs )
        vList = np.setdiff1d( np.hstack( ( leafIndices, fbsParents ) ), fbs )

        return uList, [ vList, None ]

    ######################################################################

    def progressInit( self ):
        uDone = np.zeros( self.pmask.shape[ 0 ], dtype=bool )
        vDone = coo_matrix( ( np.zeros_like( self.pmask.row ), ( self.pmask.row, self.pmask.col ) ), shape=self.pmask.shape, dtype=bool )
        return uDone, vDone

    ######################################################################

    def countSemaphoreInit( self, debug=False ):
        # Counting semaphores for U and V

        USemData = np.zeros( self.pmask.shape[ 0 ], dtype=int )

        for n in range( USemData.shape[ 0 ] ):
            dprint( '\nU Sem for n:', n, use=debug )
            # U:
            #  - U for all parents
            #  - V for all parents over all down edges except node's up edge
            #  - V for all siblings over all down edges
            upEdge = self.upEdges( n )
            dprint( 'upEdge:', upEdge, use=debug )
            for p in self.parents( n ):
                USemData[ n ] += 1

                downEdges = self.downEdges( p, skipEdges=upEdge )
                USemData[ n ] += downEdges.shape[ 0 ]

                dprint( 'p:', p, use=debug )
                dprint( 'downEdges:', downEdges, use=debug )

            USemData[ n ] += self.siblings( n ).shape[ 0 ]
            dprint( 'siblings:', self.siblings( n ), use=debug )

        VSemData = np.zeros_like( self.pmask.row )

        for i, ( n, e, _ ) in enumerate( zip( self.pmask.row, self.pmask.col, self.pmask.data ) ):
            dprint( '\nV Sem for n:', n, 'e:', e, use=debug )
            # V:
            #  - U for all mates from e
            #  - V for all mates over all down edges except for e
            #  - V for all children from e over all down edges
            VSemData[ i ] += self.mates( n, edges=e ).shape[ 0 ]
            dprint( 'All mates:', self.mates( n, edges=e ), use=debug )

            downEdges = self.downEdges( n, skipEdges=e )
            VSemData[ i ] += self.mates( n, edges=downEdges ).shape[ 0 ]
            dprint( 'downEdges without e:', downEdges, use=debug )
            dprint( 'Not down e mates:', self.mates( n, edges=downEdges ), use=debug )

            VSemData[ i ] += self.children( n, edges=e ).shape[ 0 ]
            dprint( 'children from e:', self.children( n, edges=e ), use=debug )

        uSem = USemData
        vSem = coo_matrix( ( VSemData, ( self.pmask.row, self.pmask.col ) ), shape=self.pmask.shape, dtype=int )

        return uSem, vSem

    ######################################################################

    def condition( self, nodes ):
        pass

    ######################################################################

    def readyForU( self, uSem, uDone, debug=False ):
        nodes = np.arange( uSem.shape[ 0 ] )
        dprint( '\nWorking on ready for U', use=debug )
        dprint( 'uSem mask', uSem == 0, use=debug )
        dprint( 'uDone', uDone, use=debug )
        dprint( 'done mask', np.logical_not( uDone ), use=debug )
        return nodes[ ( uSem == 0 ) & np.logical_not( uDone ) ]

    def readyForV( self, vSem, vDone, debug=False ):
        dprint( '\nWorking on ready for V', use=debug )
        dprint( 'vSem mask', vSem.data == 0, use=debug )
        dprint( 'vDone', vDone, use=debug )
        dprint( 'vDone', np.logical_not( vDone.data ), use=debug )
        mask = ( vSem.data == 0 ) & np.logical_not( vDone.data )
        return vSem.row[ mask ], vSem.col[ mask ]

    ######################################################################

    def UDone( self, nodes, uSem, vSem, uDone, debug=False ):

        dprint( '\nDone with U for', nodes, use=debug )

        # Decrement uSem for children
        children = self.children( nodes, split=True )
        for node, childrenForNode in zip( nodes, children ):
            uSem[ childrenForNode ] -= 1
            dprint( 'Decrementing from U for child', childrenForNode, 'from parent', node, use=debug )

        # Decrement vSem for all mates over each down edge
        mates = self.mates( nodes, split=True )
        for node, mateForNode in zip( nodes, mates ):
            vSem.data[ np.in1d( vSem.row, mateForNode ) ] -= 1
            dprint( 'Decrementing from V for mates', mateForNode, 'from node', node, use=debug )

        uDone[ nodes ] = True

    def VDone( self, nodesAndEdges, uSem, vSem, vDone, debug=False ):

        nodes, edges = nodesAndEdges

        dprint( '\nDone with V for', nodes, ' at', edges, use=debug  )
        notCurrentEdge = np.setdiff1d( vSem.col, edges )

        # Decrement uSem for children
        children = self.children( nodes, edges=notCurrentEdge, split=True )
        for node, childrenForNode in zip( nodes, children ):
            uSem[ childrenForNode ] -= 1
            dprint( 'Decrementing from U for child', children, 'from parent', node, use=debug )

        # Decrement uSem for siblings
        siblings = self.siblings( nodes, split=True )
        for node, siblingsForNode in zip( nodes, siblings ):
            uSem[ siblingsForNode ] -= 1
            dprint( 'Decrementing from U for sibling', children, 'from node', node, use=debug )

        # Decrement vSem for mates that aren't current edge
        mates = self.mates( nodes, edges=notCurrentEdge, split=False )
        downEdges = self.downEdges( nodes, skipEdges=edges, split=True )
        for m, e in zip( mates, downEdges ):
            vSem.data[ np.in1d( vSem.row, m ) & np.in1d( vSem.col, e ) ] -= 1
            dprint( 'Decrementing from V for mates', vSem.row[ np.in1d( vSem.row, m ) & np.in1d( vSem.col, e ) ], \
                                        'at edges', vSem.col[ np.in1d( vSem.row, m ) & np.in1d( vSem.col, e ) ], use=debug )

        # Decrement vSem for parents over up edges
        parents = self.parents( nodes, split=True )
        upEdges = self.upEdges( nodes, split=True )
        for p, e in zip( parents, upEdges ):
            vSem.data[ np.in1d( vSem.row, p ) & np.in1d( vSem.col, e ) ] -= 1
            dprint( 'Decrementing from V for parents', vSem.row[ np.in1d( vSem.row, p ) & np.in1d( vSem.col, e ) ], \
                                          'at edges', vSem.col[ np.in1d( vSem.row, p ) & np.in1d( vSem.col, e ) ], use=debug )

        vDone.data[ np.in1d( vDone.row, nodes ) & np.in1d( vDone.col, edges ) ] = True

    ######################################################################

    def uReady( self, nodes, uSem ):
        return nodes[ uSem[ nodes ] == 0 ], nodes[ uSem[ nodes ] != 0 ]

    def vReady( self, nodes, vSem ):
        ready = np.intersect1d( nodes, np.setdiff1d( vSem.row, vSem.nonzero()[ 0 ] ) )
        notReady = np.setdiff1d( nodes, ready )
        return ready, notReady

    ######################################################################

    def uBaseCase( self, roots, U, conditioning, workspace ):
        pass

    def vBaseCase( self, leaves, V, conditioning, workspace ):
        pass

    ######################################################################

    def uFilter( self, nodes, U, V, conditioning, workspace ):

        # Compute P( ↑( n )_y, n_x )

        upEdges = self.upEdges( nodes, split=True )
        n_p = self.parents( nodes, split=True )
        n_s = self.siblings( nodes, split=True )

        # Generate P( n_s_x | x_t-^( n_s )_x ) as a function of [ n_s_x, *x_t-^( n_s )_x ]
        transition = self.multiplyTerms( [ self.transitionProb( sibling, parents ) for sibling, parents in zip( n_s, n_p ) ] )

        # Generate P( n_s_y | n_s_x ) as a function of n_s_x
        emission = self.multiplyTerms( [ self.emissionProb( sibling ) for sibling in n_s ] )

        # Get the relevant V terms for each sibling over each down edge
        V_s = self.multiplyTerms( [ \
              self.multiplyTerms( [ self.unpackV( sibling, edge, U, V, conditioning ) \
                                                           for edge in self.downEdges( sibling ) \
                                                       ] ) for sibling in n_s ] )

        # Compute P( !( n_s, e↓( n_s ) )_y, n_s_x | ^( n_s )_x )
        siblingIntegrands = self.multiplyTerms( ( transition, emission, V_s ) )

        # Integrate out n_s_x to get P( !( n_s, e↓( n_s ) )_y | ^( n_s )_x )
        siblingTerms = self.integrate( siblingIntegrands )

        # Generate P( n_x | x_t-^( n )_x ) as a function of [ n_x, *x_t-^( n )_x ]
        transition = self.multiplyTerms( [ self.transitionProb( node, parents ) for node, parents in zip( nodes, n_p ) ] )

        # Get the relevant U terms for each parent
        U_p = self.multiplyTerms( [ self.unpackU( parent, U, V, conditioning ) for parent in n_p ] )

        # Get the relevant V terms for each parent and each down edge except the up edges of nodes
        V_p = self.multiplyTerms( [ \
              self.multiplyTerms( [ self.unpackV( parent, edge, U, V, conditioning ) \
                                                           for edge in self.downEdges( parent, skipEdge=e ) \
                                                       ] ) for parent, e in zip( n_p, upEdges ) ] )

        # Compute P( {↑( n ) \ n }_y \ n_y, ↑( n )_x, n_x )
        nodeIntegrands = self.multiplyTerms( ( transition, U_p, V_p, siblingTerms ) )

        # Integrate out ^( n )_x to get P( {↑( n ) \ n }_y \ n_y, n_x )
        nodeTerms = self.integrate( nodeIntegrands )

        # Generate P( n_s_y | n_x ) as a function of n_x
        emission = self.multiplyTerms( [ self.emissionProb( node ) for node in nodes ] )

        # Compute P( ↑( n )_y, n_x )
        newU = self.multiplyTerms( ( emission, nodeTerms ) )

        self.updateU( nodes, newU, U, conditioning )

    def vFilter( self, nodes, edges, U, V, conditioning, workspace ):

        # Compute P( !( n, e )_y | n_x )
        upEdges = self.upEdges( nodes, split=True )
        n_m = self.parents( nodes, split=True )
        n_c = self.siblings( nodes, split=True )

        transition = self.multiplyTerms( [ self.transitionProb( child, parents ) for child, parents in zip( n_c, n_m ) ] )
        emission = self.multiplyTerms( [ self.emissionProb( child ) for child in n_c ] )
        V_s = self.multiplyTerms( [ \
              self.multiplyTerms( [ self.unpackV( child, edge, U, V, conditioning ) \
                                                           for edge in self.downEdges( child ) \
                                                       ] ) for child in n_c ] )
        childIntegrands = self.multiplyTerms( ( transition, emission, V_s ) )
        childTerms = self.integrate( childIntegrands )

        U_m = self.multiplyTerms( [ self.unpackU( parent, U, V, conditioning ) for parent in n_m ] )
        V_m = self.multiplyTerms( [ \
              self.multiplyTerms( [ self.unpackV( parent, edge, U, V, conditioning ) \
                                                           for edge in self.downEdges( parent, skipEdge=e ) \
                                                       ] ) for parent, e in zip( n_m, upEdges ) ] )

        nodeIntegrands = self.multiplyTerms( ( U_m, V_m, childTerms ) )
        newU = self.integrate( nodeIntegrands )

        self.updateU( nodes, newU, U, conditioning )

    ######################################################################

    def messagePassing( self, uWork, vWork, **kwargs ):

        uDone, vDone = self.progressInit()
        uSem, vSem = self.countSemaphoreInit( debug=False )
        uList, vList = self.baseCaseNodes()

        print( '\nInitial semaphore count' )
        print( 'uSem: \n', uSem )
        print( 'vSem: \n', vSem.todense() )

        print( '\nInitial node list' )
        print( 'uList: \n', uList )
        print( 'vList: \n', vList )

        i = 1
        debug = True

        # Filter over all of the graphs
        while( uList.size > 0 or vList[ 0 ].size > 0 ):

            dprint( '\n----------------\n', 'Iteration', i, '\n----------------\n', use=True )

            # Do work for each of the nodes
            uWork( uList, **kwargs )
            vWork( vList, **kwargs )

            # Mark that we're done with the current nodes
            self.UDone( uList, uSem, vSem, uDone, debug=False )
            self.VDone( vList, uSem, vSem, vDone, debug=False )

            if( debug ):
                print( '\nSemaphore count after marking nodes done' )
                print( 'uSem: \n', uSem )
                print( 'vSem: \n', vSem.todense() )

            # Find the next nodes that are ready
            uList = self.readyForU( uSem, uDone, debug=False )
            vList = self.readyForV( vSem, vDone, debug=False )

            if( debug ):
                print( '\nNext node list' )
                print( 'uList: \n', uList )
                print( 'vList: \n', vList )

                print( '\nCompleted list' )
                print( 'uDone: \n', np.arange( uDone.shape[ 0 ] )[ uDone ] )
                print( 'vDone: \n', sorted( list( zip( vDone.row, vDone.col ) ), key=lambda x:x[0] ) )

            i += 1
            # if( i == 5 ):
                # assert 0

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

        # Run the message passing algorithm over the graph
        self.messagePassing( self.uFilter, self.vFilter, kwargs )

        # Integrate out the nodes that we cut
        self.integrateOutConditioning( U, V, conditioning, workspace )

        # Update the filter probs for the cut nodes
        self.filterCutNodes( U, V, conditioning, workspace )

        return alphas