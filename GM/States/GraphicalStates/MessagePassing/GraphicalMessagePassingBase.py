import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix
import graphviz

def dprint( *args, use=False ):
    if( use ):
        print( *args )

class Graph():
    # This class is how we make sparse matrices

    def __init__( self ):
        self.nodes = set()
        self.edgeChildren = list()
        self.edgeParents = list()

    @staticmethod
    def fromParentChildMask( pMask, cMask ):
        graph = Graph()
        assert pMask.shape == cMask.shape
        nEdges = pMask.shape[ 1 ]
        for e in range( nEdges ):
            parents = pMask.getcol( e ).nonzero()[ 0 ]
            children = cMask.getcol( e ).nonzero()[ 0 ]
            graph.addEdge( parents=parents.tolist(), children=children.tolist() )

        return graph

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

    def draw( self, render=True ):

        """ Draws the hypergraph using graphviz """
        d = graphviz.Digraph()
        for e, ( parents, children ) in enumerate( zip( self.edgeParents, self.edgeChildren ) ):
            for p in parents:
                d.edge( 'n( '+str( p )+' )', 'E( '+str( e )+' )', **{
                    'arrowhead': 'none',
                    'fixedsize': 'true'
                })
            for c in children:
                d.edge( 'E( '+str( e )+' )', 'n( '+str( c )+' )', **{
                    'arrowhead': 'none',
                    'fixedsize': 'true'
                })

            d.node('E( '+str( e )+' )', **{
                'width': '0.25',
                'height': '0.25',
                'fontcolor': 'white',
                'style': 'filled',
                'fillcolor': 'black',
                'fixedsize': 'true',
                'fontsize': '6'
            })

        if( render ):
            d.render()

        return d

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

    def toGraph( self ):
        return Graph.fromParentChildMask( self.pmask, self.cmask )

    def concatSparseMatrix( self, sparseMatrices ):
        # Builds a big block diagonal matrix where each diagonal matrix
        # is an element in sparseMatrices

        row = np.array( [], dtype=int )
        col = np.array( [], dtype=int )
        data = np.array( [], dtype=int )
        nRows = 0
        nCols = 0
        for mat in sparseMatrices:
            m, n = mat.shape
            row = np.hstack( ( row, mat.row + nRows ) )
            col = np.hstack( ( col, mat.col + nCols ) )
            data = np.hstack( ( data, mat.data ) )
            nRows += m
            nCols += n
        return coo_matrix( ( data, ( row, col ) ), shape=( nRows, nCols ), dtype=bool )

    def fbsConcat( self, feedbackSets, nodeCounts ):
        assert len( feedbackSets ) == len( nodeCounts )
        bigFBS = []
        totalN = 0
        for fbs, N in zip( feedbackSets, nodeCounts ):
            bigFBS.append( fbs + totalN )
            totalN += N
        return np.concatenate( bigFBS )

    def updateParams( self, parentMasks, childMasks, feedbackSets=None ):

        if( feedbackSets is not None ):
            assert len( parentMasks ) == len( childMasks ) == len( feedbackSets )
            for childMask, parentMask, feedbackSet in zip( childMasks, parentMasks, feedbackSets ):
                assert isinstance( childMask, coo_matrix )
                assert isinstance( parentMask, coo_matrix )
                assert childMask.shape == parentMask.shape
        else:
            assert len( parentMasks ) == len( childMasks )
            for childMask, parentMask in zip( childMasks, parentMasks ):
                assert isinstance( childMask, coo_matrix )
                assert isinstance( parentMask, coo_matrix )
                assert childMask.shape == parentMask.shape

        self.pmask = self.concatSparseMatrix( parentMasks )
        self.cmask = self.concatSparseMatrix( childMasks )

        self.nodes = np.arange( self.pmask.shape[ 0 ] )

        if( feedbackSets is not None ):
            nodeCounts = [ mat.shape[ 0 ] for mat in parentMasks ]
            self.fbsMask = np.in1d( self.nodes, self.fbsConcat( feedbackSets, nodeCounts ) )
            # self.fbsMask = np.in1d( self.nodes, np.concatenate( feedbackSets ) )
        else:
            self.fbsMask = np.zeros( self.pmask.shape[ 0 ], dtype=bool )

        fbs = self.nodes[ self.fbsMask ]
        self.fbsPMask = np.in1d( self.pmask.row, fbs )
        self.fbsCMask = np.in1d( self.cmask.row, fbs )

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
        return np.unique( cols[ np.in1d( rows, nodes ) ] )

    def downEdges( self, nodes, skipEdges=None, split=False ):
        if( split ):
            return [ self.downEdges( n, skipEdges=skipEdges, split=False ) for n in nodes ]
        if( skipEdges is not None ):
            return np.setdiff1d( self.downEdges( nodes, skipEdges=None, split=False ), skipEdges )
        rows, cols = self.pmask.nonzero()
        return np.unique( cols[ np.in1d( rows, nodes ) ] )

    def _nodesFromEdges( self, nodes, edges, getChildren=True, diffNodes=False, noFBS=False ):

        mask = self.cmask if getChildren else self.pmask
        fbsmask = self.fbsCMask if getChildren else self.fbsPMask

        edgeMask = np.in1d( mask.col, edges )

        if( noFBS == True ):
            edgeMask &= np.logical_not( fbsmask )

        if( diffNodes ):
            return np.setdiff1d( mask.row[ edgeMask ], nodes )

        return np.unique( mask.row[ edgeMask ] )

    def _nodeSelectFromEdge( self, nodes, edges=None, upEdge=False, getChildren=True, diffNodes=False, splitByEdge=False, split=False, noFBS=False ):

        if( split ):
            if( edges is None ):
                return [ self._nodeSelectFromEdge( n, edges=None, \
                                                      upEdge=upEdge, \
                                                      getChildren=getChildren, \
                                                      diffNodes=diffNodes, \
                                                      splitByEdge=splitByEdge, \
                                                      split=False, \
                                                      noFBS=noFBS ) for n in nodes ]
            else:
                return [ self._nodeSelectFromEdge( n, edges=e, \
                                                      upEdge=upEdge, \
                                                      getChildren=getChildren, \
                                                      diffNodes=diffNodes, \
                                                      splitByEdge=splitByEdge, \
                                                      split=False, \
                                                      noFBS=noFBS ) for n, e in zip( nodes, edges ) ]

        _edges = self.upEdges( nodes ) if upEdge else self.downEdges( nodes )

        if( edges is not None ):
            _edges = np.intersect1d( _edges, edges )

        if( splitByEdge == True ):
            return [ [ e, self._nodeSelectFromEdge( nodes, edges=e, \
                                                           upEdge=upEdge, \
                                                           getChildren=getChildren, \
                                                           diffNodes=diffNodes, \
                                                           splitByEdge=False, \
                                                           split=False, \
                                                           noFBS=noFBS ) ] for e in _edges ]

        return self._nodesFromEdges( nodes, _edges, getChildren=getChildren, diffNodes=diffNodes, noFBS=noFBS )

    def parents( self, nodes, split=False, noFBS=False ):
        return self._nodeSelectFromEdge( nodes, edges=None, \
                                                upEdge=True, \
                                                getChildren=False, \
                                                diffNodes=False, \
                                                splitByEdge=False, \
                                                split=split, \
                                                noFBS=noFBS )

    def siblings( self, nodes, split=False, noFBS=False ):
        return self._nodeSelectFromEdge( nodes, edges=None, \
                                                upEdge=True, \
                                                getChildren=True, \
                                                diffNodes=True, \
                                                splitByEdge=False, \
                                                split=split, \
                                                noFBS=noFBS )

    def children( self, nodes, edges=None, splitByEdge=False, split=False, noFBS=False ):
        return self._nodeSelectFromEdge( nodes, edges=edges, \
                                                upEdge=False, \
                                                getChildren=True, \
                                                diffNodes=False, \
                                                splitByEdge=splitByEdge, \
                                                split=split, \
                                                noFBS=noFBS )

    def mates( self, nodes, edges=None, splitByEdge=False, split=False, noFBS=False ):
        return self._nodeSelectFromEdge( nodes, edges=edges, \
                                                upEdge=False, \
                                                getChildren=False, \
                                                diffNodes=True, \
                                                splitByEdge=splitByEdge, \
                                                split=split, \
                                                noFBS=noFBS )

    ######################################################################

    def baseCaseNodes( self ):

        M, N = self.pmask.shape

        # Get the number of edges that each node is a parent of
        parentOfEdgeCount = self.pmask.getnnz( axis=1 )

        # Get the number of edges that each node is a child of
        childOfEdgeCount = self.cmask.getnnz( axis=1 )

        # Get the indices of leaves and roots
        rootIndices = self.nodes[ ( parentOfEdgeCount != 0 ) & ( childOfEdgeCount == 0 ) ]
        leafIndices = self.nodes[ ( childOfEdgeCount != 0 ) & ( parentOfEdgeCount == 0 ) ]

        # Explicitely get the feedback set
        fbs = self.nodes[ self.fbsMask ]

        # Nodes whose parents are all in the fbs are roots, and nodes whose
        # children are all in the fbs are leaves
        pseudoRoots = []
        pseudoLeaves = []

        fbsParents = self.parents( fbs )
        childrenOfFBSParents = self.children( fbsParents, split=True )
        for children, fbsParent in zip( childrenOfFBSParents, fbsParents ):
            if( np.all( np.in1d( children, fbs ) ) == True ):
                pseudoLeaves.append( fbsParent )

        fbsChildren = self.children( fbs )
        parentsOfFBSChildren = self.parents( fbsChildren, split=True )
        for parents, fbsChild in zip( parentsOfFBSChildren, fbsChildren ):
            if( np.all( np.in1d( parents, fbs ) ) == True ):
                pseudoRoots.append( fbsChild )

        # Generate the up and down base arrays
        uList = np.setdiff1d( np.hstack( ( rootIndices, np.array( pseudoRoots, dtype=int ) ) ), fbs )
        vList = np.setdiff1d( np.hstack( ( leafIndices, np.array( pseudoLeaves, dtype=int ) ) ), fbs )

        return uList, [ vList, [ None for _ in vList ] ]

    ######################################################################

    def progressInit( self ):
        uDone = np.copy( self.fbsMask )
        vDone = coo_matrix( ( np.zeros_like( self.pmask.row ), ( self.pmask.row, self.pmask.col ) ), shape=self.pmask.shape, dtype=bool )
        fbs = self.nodes[ self.fbsMask ]
        vDone.data[ np.in1d( vDone.row, fbs ) ] = True

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
            parents = self.parents( n, noFBS=True )

            USemData[ n ] += parents.shape[ 0 ]

            dprint( 'All parents:', parents, use=debug )
            dprint( 'upEdge:', upEdge, use=debug )
            for parent in parents:

                downEdges = self.downEdges( parent, skipEdges=upEdge )
                USemData[ n ] += downEdges.shape[ 0 ]

                dprint( 'downEdges:', downEdges, 'from parent:', parent, use=debug )

            siblings = self.siblings( n, noFBS=True )
            for sibling in siblings:
                downEdges = self.downEdges( sibling )
                USemData[ n ] += downEdges.shape[ 0 ]
                dprint( 'downEdges:', downEdges, 'from sibling:', sibling, use=debug )

        VSemData = np.zeros_like( self.pmask.row )

        for i, ( n, e, _ ) in enumerate( zip( self.pmask.row, self.pmask.col, self.pmask.data ) ):
            dprint( '\nV Sem for n:', n, 'e:', e, use=debug )
            # V:
            #  - U for all mates from e
            #  - V for all mates over all down edges for mate except for e
            #  - V for all children from e over all down edges for child

            mates = self.mates( n, edges=e, noFBS=True )

            VSemData[ i ] += mates.shape[ 0 ]
            dprint( 'All mates:', mates, use=debug )

            for mate in mates:
                downEdges = self.downEdges( mate, skipEdges=e )
                VSemData[ i ] += downEdges.shape[ 0 ]
                dprint( 'downEdges without e:', downEdges, 'from mate:', mate, use=debug )

            children = self.children( n, edges=e, noFBS=True )
            dprint( 'All children:', children, use=debug )
            for child in children:
                downEdges = self.downEdges( child )
                VSemData[ i ] += downEdges.shape[ 0 ]
                dprint( 'downEdges:', downEdges, 'from child:', child, use=debug )

        # Set the feedback set nodes to 0
        USemData[ self.fbsMask ] = 0

        fbs = self.nodes[ self.fbsMask ]
        VSemData[ np.in1d( self.pmask.row, fbs ) ] = 0

        uSem = USemData
        vSem = coo_matrix( ( VSemData, ( self.pmask.row, self.pmask.col ) ), shape=self.pmask.shape, dtype=int )

        return uSem, vSem

    ######################################################################

    def condition( self, nodes ):
        pass

    ######################################################################

    def readyForU( self, uSem, uDone, debug=False ):
        dprint( '\nWorking on ready for U', use=debug )
        dprint( 'uSem mask', uSem == 0, use=debug )
        dprint( 'uDone', uDone, use=debug )
        dprint( 'done mask', np.logical_not( uDone ), use=debug )
        return self.nodes[ ( uSem == 0 ) & np.logical_not( uDone ) ]

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
        children = self.children( nodes, split=True, noFBS=True )
        for node, childrenForNode in zip( nodes, children ):
            uSem[ childrenForNode ] -= 1
            dprint( 'Decrementing from U for child', childrenForNode, 'from parent', node, use=debug )

        # Decrement vSem for all mates over down edges that node and mate are a part of
        matesAndEdges = self.mates( nodes, splitByEdge=True, split=True, noFBS=True )
        for node, mateAndEdge in zip( nodes, matesAndEdges ):
            for e, m in mateAndEdge:
                vSem.data[ np.in1d( vSem.row, m ) & np.in1d( vSem.col, e ) ] -= 1
                dprint( 'Decrementing from V for mates', m, 'at edge', e, 'from node', node, use=debug )

        uDone[ nodes ] = True

    def VDone( self, nodesAndEdges, uSem, vSem, vDone, debug=False ):

        nodes, edges = nodesAndEdges
        edgesWithoutNone = np.array( [ e for e in edges if e is not None ] )

        dprint( '\nDone with V for', nodes, 'at', edges, use=debug  )
        notCurrentEdge = np.setdiff1d( vSem.col, edgesWithoutNone )

        # Decrement uSem for children that come from a different edge than the one computed for V
        childrenAndEdges = self.children( nodes, splitByEdge=True, split=True, noFBS=True )
        for node, edge, childAndEdge in zip( nodes, edges, childrenAndEdges ):
            for e, c in childAndEdge:
                if( e == edge ):
                    continue
                uSem[ c ] -= 1
                dprint( 'Decrementing from U for child', c, 'from parent', node, use=debug )

        # Decrement uSem for all siblings
        siblings = self.siblings( nodes, split=True, noFBS=True )
        for _e, node, siblingsForNode in zip( edges, nodes, siblings ):
            if( _e is None ):
                # If this node doesn't have a down edge, then we don't want to decrement
                continue
            uSem[ siblingsForNode ] -= 1
            dprint( 'Decrementing from U for sibling', siblingsForNode, 'from node', node, use=debug )

        # Decrement vSem for mates that aren't current edge
        matesAndEdges = self.mates( nodes, splitByEdge=True, split=True, noFBS=True )
        print( nodes )
        print( matesAndEdges )
        for node, edge, mateAndEdge in zip( nodes, edges, matesAndEdges ):
            for e, m in mateAndEdge:
                if( e == edge ):
                    continue
                vSem.data[ np.in1d( vSem.row, m ) & np.in1d( vSem.col, e ) ] -= 1
                dprint( 'Decrementing from V for mates', m, 'at edge', e, 'from node', node, use=debug )

        # Decrement vSem for parents over up edges
        parents = self.parents( nodes, split=True, noFBS=True )
        upEdges = self.upEdges( nodes, split=True )
        for _e, p, e in zip( edges, parents, upEdges ):
            if( _e is None ):
                # If this node doesn't have a down edge, then we don't want to decrement
                continue
            vSem.data[ np.in1d( vSem.row, p ) & np.in1d( vSem.col, e ) ] -= 1
            dprint( 'Decrementing from V for parents', vSem.row[ np.in1d( vSem.row, p ) & np.in1d( vSem.col, e ) ], \
                                          'at edges', vSem.col[ np.in1d( vSem.row, p ) & np.in1d( vSem.col, e ) ], use=debug )

        vDone.data[ np.in1d( vDone.row, nodes ) & np.in1d( vDone.col, edgesWithoutNone ) ] = True

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

    def convergence( self, nodes ):
        return False

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

        print( '\nInitial done list' )
        print( 'uDone: \n', uDone )
        print( 'vDone: \n', vDone )

        i = 1
        debug = True

        # Filter over all of the graphs
        while( uList.size > 0 or vList[ 0 ].size > 0 ):

            dprint( '\n----------------\n', 'Iteration', i, '\n----------------\n', use=debug )

            # Do work for each of the nodes
            uWork( uList, **kwargs )
            vWork( vList, **kwargs )

            # Mark that we're done with the current nodes
            self.UDone( uList, uSem, vSem, uDone, debug=True )
            self.VDone( vList, uSem, vSem, vDone, debug=True )

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
                print( 'uDone: \n', self.nodes[ uDone ] )
                print( 'vDone: \n', sorted( list( zip( vDone.row, vDone.col ) ), key=lambda x:x[0] ) )

            i += 1
            # if( i == 5 ):
                # assert 0

            # Check if we need to do loopy propogation belief
            if( ( uList.size == 0 and vList[ 0 ].size == 0 ) and \
                ( not np.any( uDone ) or not np.any( vDone.data ) ) ):
                loopy = True

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