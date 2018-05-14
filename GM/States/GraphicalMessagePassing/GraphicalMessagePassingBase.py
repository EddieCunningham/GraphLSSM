import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix
import graphviz
from collections import Iterable

def dprint( *args, use=False ):
    if( use ):
        print( *args, flush=True )

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
            for j, node in enumerate( nodeGroup ):
                rowIndex = nodes.index( node )
                colIndex = i

                rows.append( rowIndex )
                cols.append( colIndex )

                # Use an integer so that we can have an ordering of nodes within edges!!!!
                data.append( j + 1 )

        mask = coo_matrix( ( data, ( rows, cols ) ), shape=( nRows, nCols ), dtype=int )
        return mask

    def toMatrix( self ):

        nodeList = list( self.nodes )

        parentMask = self._cooMatrixFromNodeEdge( nodeList, self.edgeParents )
        childMask = self._cooMatrixFromNodeEdge( nodeList, self.edgeChildren )

        return parentMask, childMask

    def draw( self, render=True, cutNodes=None ):

        # Draws the graph using graphviz
        d = graphviz.Digraph()
        for e, ( parents, children ) in enumerate( zip( self.edgeParents, self.edgeChildren ) ):
            for p in parents:
                d.edge( 'n( %d )'%( p ), 'E( %d )'%( e ), **{
                    'fixedsize': 'true'
                } )
            for c in children:
                d.edge( 'E( %d )'%( e ), 'n( %d )'%( c ), **{
                    'fixedsize': 'true'
                } )

            d.node( 'E( %d )'%( e ), **{
                'width': '0.25',
                'height': '0.25',
                'fontcolor': 'white',
                'style': 'filled',
                'fillcolor': 'black',
                'fixedsize': 'true',
                'fontsize': '6'
            } )

        if( cutNodes is not None ):
            for n in cutNodes:
                print( n )
                d.node( 'n( %d )'%( n ), **{
                       'style': 'filled',
                       'fontcolor': 'white',
                       'fillcolor':'blue'
                       } )

        if( render ):
            d.render()

        return d

class GraphMessagePasser():

    def __init__( self ):
        pass

    def toGraph( self ):
        return Graph.fromParentChildMask( self.pmask, self.cmask )

    def draw( self ):
        return self.toGraph().draw( cutNodes=self.fbs )

    def concatSparseMatrix( self, sparseMatrices ):
        # Builds a big block diagonal matrix where each diagonal matrix
        # is an element in sparseMatrices

        row = np.array( [], dtype=int )
        col = np.array( [], dtype=int )
        data = np.array( [], dtype=int )
        graphAssignments = []
        nRows = 0
        nCols = 0
        for i, mat in enumerate( sparseMatrices ):
            m, n = mat.shape
            row = np.hstack( ( row, mat.row + nRows ) )
            col = np.hstack( ( col, mat.col + nCols ) )
            data = np.hstack( ( data, mat.data ) )
            nRows += m
            nCols += n
            graphAssignments.append( nRows )
        return coo_matrix( ( data, ( row, col ) ), shape=( nRows, nCols ), dtype=int ), graphAssignments

    def fbsConcat( self, feedbackSets, nodeCounts ):
        assert len( feedbackSets ) == len( nodeCounts )
        bigFBS = []
        totalN = 0
        for fbs, N in zip( feedbackSets, nodeCounts ):
            if( fbs is not None ):
                bigFBS.append( fbs + totalN )
            else:
                bigFBS.append( np.array( []) )
            totalN += N
        if( len( bigFBS ) == 0 ):
            return np.array( [] ), np.array( [] )
        return np.concatenate( bigFBS ), bigFBS

    def updateParamsFromGraphs( self, graphs ):

        parentMasks = []
        childMasks = []
        feedbackSets = []
        for graph in graphs:
            if( isinstance( graph, Iterable ) ):
                assert len( graph ) == 2
                graph, fbs = graph
            else:
                fbs = None

            pMask, cMask = graph.toMatrix()
            parentMasks.append( pMask )
            childMasks.append( cMask )
            feedbackSets.append( fbs )

        self.updateParams( parentMasks, childMasks, feedbackSets=feedbackSets )

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

        self.pmask, self.parentGraphAssignments = self.concatSparseMatrix( parentMasks )
        self.cmask, self.childGraphAssignments = self.concatSparseMatrix( childMasks )

        self.nodes = np.arange( self.pmask.shape[ 0 ] )

        if( feedbackSets is not None ):
            nodeCounts = [ mat.shape[ 0 ] for mat in parentMasks ]
            # self.feedbackSets contains all of the feedback sets with the adjusted node indices
            fbsNodes, self.feedbackSets = self.fbsConcat( feedbackSets, nodeCounts )
            self.fbsMask = np.in1d( self.nodes, fbsNodes )
        else:
            self.fbsMask = np.zeros( self.pmask.shape[ 0 ], dtype=bool )

        # All of the feedback sets together
        self.fbs = self.nodes[ self.fbsMask ]

        # Parent and child mask for feedback set nodes
        self.fbsPMask = np.in1d( self.pmask.row, self.fbs )
        self.fbsCMask = np.in1d( self.cmask.row, self.fbs )

    ######################################################################

    def getGraphAssignment( self, node ):
        pass

    ######################################################################

    @staticmethod
    def _upEdges( cmask, nodes, split=False ):
        if( split ):
            return [ GraphMessagePasser._upEdges( cmask, n, split=False ) for n in nodes ]
        rows, cols = cmask.nonzero()
        return np.unique( cols[ np.in1d( rows, nodes ) ] )

    @staticmethod
    def _downEdges( pmask, nodes, skipEdges=None, split=False ):
        if( split ):
            return [ GraphMessagePasser._downEdges( pmask, n, skipEdges=skipEdges, split=False ) for n in nodes ]
        if( skipEdges is not None ):
            return np.setdiff1d( GraphMessagePasser._downEdges( pmask, nodes, skipEdges=None, split=False ), skipEdges )
        rows, cols = pmask.nonzero()
        return np.unique( cols[ np.in1d( rows, nodes ) ] )

    ######################################################################

    def upEdges( self, nodes, split=False ):
        return GraphMessagePasser._upEdges( self.cmask, nodes, split=split )

    def downEdges( self, nodes, skipEdges=None, split=False ):
        return GraphMessagePasser._downEdges( self.pmask, nodes, skipEdges=skipEdges, split=split )

    ######################################################################

    @staticmethod
    def _nodesFromEdges( nodes, \
                         edges, \
                         cmask, \
                         pmask, \
                         skipCMask, \
                         fbsPMask, \
                         getChildren=True, \
                         diffNodes=False, \
                         noFBS=False, \
                         getOrder=False ):

        mask = cmask if getChildren else pmask
        skipmask = skipCMask if getChildren else fbsPMask

        edgeMask = np.in1d( mask.col, edges )

        if( noFBS == True ):
            edgeMask &= ~skipmask

        if( diffNodes ):
            finalMask = edgeMask & ~np.in1d( mask.row, nodes )
        else:
            finalMask = edgeMask

        if( getOrder is False ):
            return np.unique( mask.row[ finalMask ] )
        return mask.row[ finalMask ], mask.data[ finalMask ] - 1 # Subtract one to use 0 indexing

    @staticmethod
    def _nodeSelectFromEdge( cmask, \
                             pmask, \
                             skipCMask, \
                             skipPMask, \
                             nodes, \
                             edges=None, \
                             upEdge=False, \
                             getChildren=True, \
                             diffNodes=False, \
                             splitByEdge=False, \
                             split=False, \
                             noFBS=False, \
                             getOrder=False ):

        if( split ):
            if( edges is None ):
                return [ GraphMessagePasser._nodeSelectFromEdge( cmask, \
                                                                 pmask, \
                                                                 skipCMask, \
                                                                 skipPMask,
                                                                 n, \
                                                                 edges=None, \
                                                                 upEdge=upEdge, \
                                                                 getChildren=getChildren, \
                                                                 diffNodes=diffNodes, \
                                                                 splitByEdge=splitByEdge, \
                                                                 split=False, \
                                                                 noFBS=noFBS, \
                                                                 getOrder=getOrder ) for n in nodes ]
            else:
                return [ GraphMessagePasser._nodeSelectFromEdge( cmask, \
                                                                 pmask, \
                                                                 skipCMask, \
                                                                 skipPMask,
                                                                 n, \
                                                                 edges=e, \
                                                                 upEdge=upEdge, \
                                                                 getChildren=getChildren, \
                                                                 diffNodes=diffNodes, \
                                                                 splitByEdge=splitByEdge, \
                                                                 split=False, \
                                                                 noFBS=noFBS, \
                                                                 getOrder=getOrder ) for n, e in zip( nodes, edges ) ]

        _edges = GraphMessagePasser._upEdges( cmask, nodes ) if upEdge else GraphMessagePasser._downEdges( pmask, nodes )

        if( edges is not None ):
            _edges = np.intersect1d( _edges, edges )

        if( splitByEdge == True ):
            return [ [ e, GraphMessagePasser._nodeSelectFromEdge( cmask, \
                                                                  pmask, \
                                                                  skipCMask, \
                                                                  skipPMask,
                                                                  nodes, \
                                                                  edges=e, \
                                                                  upEdge=upEdge, \
                                                                  getChildren=getChildren, \
                                                                  diffNodes=diffNodes, \
                                                                  splitByEdge=False, \
                                                                  split=False, \
                                                                  noFBS=noFBS, \
                                                                  getOrder=getOrder ) ] for e in _edges ]

        return GraphMessagePasser._nodesFromEdges( nodes, \
                                                   _edges, \
                                                   cmask, \
                                                   pmask, \
                                                   skipCMask, \
                                                   skipPMask, \
                                                   getChildren=getChildren, \
                                                   diffNodes=diffNodes, \
                                                   noFBS=noFBS, \
                                                   getOrder=getOrder )

    ######################################################################

    @staticmethod
    def _parents( cmask, pmask, skipCMask, skipPMask, nodes, split=False, noFBS=False, getOrder=False ):
        return GraphMessagePasser._nodeSelectFromEdge( cmask, \
                                                       pmask, \
                                                       skipCMask, \
                                                       skipPMask, \
                                                       nodes, \
                                                       edges=None, \
                                                       upEdge=True, \
                                                       getChildren=False, \
                                                       diffNodes=False, \
                                                       splitByEdge=False, \
                                                       split=split, \
                                                       noFBS=noFBS, \
                                                       getOrder=getOrder )

    @staticmethod
    def _siblings( cmask, pmask, skipCMask, skipPMask, nodes, split=False, noFBS=False, getOrder=False ):
        return GraphMessagePasser._nodeSelectFromEdge( cmask, \
                                                       pmask, \
                                                       skipCMask, \
                                                       skipPMask, \
                                                       nodes, edges=None, \
                                                       upEdge=True, \
                                                       getChildren=True, \
                                                       diffNodes=True, \
                                                       splitByEdge=False, \
                                                       split=split, \
                                                       noFBS=noFBS, \
                                                       getOrder=getOrder )

    @staticmethod
    def _children( cmask, pmask, skipCMask, skipPMask, nodes, edges=None, splitByEdge=False, split=False, noFBS=False, getOrder=False ):
        return GraphMessagePasser._nodeSelectFromEdge( cmask, \
                                                       pmask, \
                                                       skipCMask, \
                                                       skipPMask, \
                                                       nodes, \
                                                       edges=edges, \
                                                       upEdge=False, \
                                                       getChildren=True, \
                                                       diffNodes=False, \
                                                       splitByEdge=splitByEdge, \
                                                       split=split, \
                                                       noFBS=noFBS, \
                                                       getOrder=getOrder )

    @staticmethod
    def _mates( cmask, pmask, skipCMask, skipPMask, nodes, edges=None, splitByEdge=False, split=False, noFBS=False, getOrder=False ):
        return GraphMessagePasser._nodeSelectFromEdge( cmask, \
                                                       pmask, \
                                                       skipCMask, \
                                                       skipPMask, \
                                                       nodes, \
                                                       edges=edges, \
                                                       upEdge=False, \
                                                       getChildren=False, \
                                                       diffNodes=True, \
                                                       splitByEdge=splitByEdge, \
                                                       split=split, \
                                                       noFBS=noFBS, \
                                                       getOrder=getOrder )

    ######################################################################

    def parents( self, nodes, split=False, noFBS=False, getOrder=False ):
        return GraphMessagePasser._parents( self.cmask, \
                                            self.pmask, \
                                            self.fbsCMask, \
                                            self.fbsPMask, \
                                            nodes, \
                                            split=split, \
                                            noFBS=noFBS, \
                                            getOrder=getOrder )

    def siblings( self, nodes, split=False, noFBS=False ):
        return GraphMessagePasser._siblings( self.cmask, \
                                             self.pmask, \
                                             self.fbsCMask, \
                                             self.fbsPMask, \
                                             nodes, \
                                             split=split, \
                                             noFBS=noFBS )

    def children( self, nodes, edges=None, splitByEdge=False, split=False, noFBS=False ):
        return GraphMessagePasser._children( self.cmask, \
                                             self.pmask, \
                                             self.fbsCMask, \
                                             self.fbsPMask, \
                                             nodes, \
                                             edges=edges, \
                                             splitByEdge=splitByEdge, \
                                             split=split, \
                                             noFBS=noFBS )

    def mates( self, nodes, edges=None, splitByEdge=False, split=False, noFBS=False, getOrder=False ):
        return GraphMessagePasser._mates( self.cmask, \
                                          self.pmask, \
                                          self.fbsCMask, \
                                          self.fbsPMask, \
                                          nodes, \
                                          edges=edges, \
                                          splitByEdge=splitByEdge, \
                                          split=split, \
                                          noFBS=noFBS, \
                                          getOrder=getOrder )

    ######################################################################

    def pseudoRootsAndLeaves( self ):
        # Nodes whose parents are all in the fbs are roots, and nodes whose
        # children are all in the fbs are leaves
        pseudoRoots = []
        pseudoLeaves = []

        fbsParents = self.parents( self.fbs )
        childrenOfFBSParents = self.children( fbsParents, split=True )
        for children, fbsParent in zip( childrenOfFBSParents, fbsParents ):
            if( np.all( np.in1d( children, self.fbs ) ) == True ):
                pseudoLeaves.append( fbsParent )

        fbsChildren = self.children( self.fbs )
        parentsOfFBSChildren = self.parents( fbsChildren, split=True )
        for parents, fbsChild in zip( parentsOfFBSChildren, fbsChildren ):
            if( np.all( np.in1d( parents, self.fbs ) ) == True ):
                pseudoRoots.append( fbsChild )

        return pseudoRoots, pseudoLeaves

    def baseCaseNodes( self ):

        M, N = self.pmask.shape

        # Get the number of edges that each node is a parent of
        parentOfEdgeCount = self.pmask.getnnz( axis=1 )

        # Get the number of edges that each node is a child of
        childOfEdgeCount = self.cmask.getnnz( axis=1 )

        # Get the indices of leaves and roots
        rootIndices = self.nodes[ ( parentOfEdgeCount != 0 ) & ( childOfEdgeCount == 0 ) ]
        leafIndices = self.nodes[ ( childOfEdgeCount != 0 ) & ( parentOfEdgeCount == 0 ) ]

        # Nodes whose parents are all in the fbs are roots, and nodes whose
        # children are all in the fbs are leaves
        pseudoRoots, pseudoLeaves = self.pseudoRootsAndLeaves()

        # Generate the up and down base arrays
        uList = np.setdiff1d( np.hstack( ( rootIndices, np.array( pseudoRoots, dtype=int ) ) ), self.fbs )
        vList = np.setdiff1d( np.hstack( ( leafIndices, np.array( pseudoLeaves, dtype=int ) ) ), self.fbs )

        # Make sure that fbs children have the correct down edges
        vListNodes = []
        vListEdges = []
        for v in vList:
            if( v in pseudoLeaves ):
                downEdges = self.downEdges( v )
                for e in downEdges:
                    vListNodes.append( v )
                    vListEdges.append( e )
            else:
                vListNodes.append( v )
                vListEdges.append( None )

        return uList, [ vListNodes, vListEdges ]

    ######################################################################

    def progressInit( self ):
        uDone = np.copy( self.fbsMask )
        vDone = coo_matrix( ( np.zeros_like( self.pmask.row ), ( self.pmask.row, self.pmask.col ) ), shape=self.pmask.shape, dtype=bool )
        vDone.data[ np.in1d( vDone.row, self.fbs ) ] = True

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

    def readyForU( self, uSem, uDone, debug=False ):
        dprint( '\nWorking on ready for U', use=debug )
        dprint( 'uSem mask', uSem == 0, use=debug )
        dprint( 'uDone', uDone, use=debug )
        dprint( 'done mask', ~uDone, use=debug )
        return self.nodes[ ( uSem == 0 ) & ~uDone ]

    def readyForV( self, vSem, vDone, debug=False ):
        dprint( '\nWorking on ready for V', use=debug )
        dprint( 'vSem mask', vSem.data == 0, use=debug )
        dprint( 'vDone', vDone, use=debug )
        dprint( 'vDone', ~vDone.data, use=debug )
        mask = ( vSem.data == 0 ) & ~vDone.data
        return vSem.row[ mask ], vSem.col[ mask ]

    ######################################################################

    def UDone( self, nodes, uSem, vSem, uDone, debug=False ):

        dprint( '\nDone with U for', nodes, use=debug )

        # Decrement uSem for children
        children = self.children( nodes, split=True, noFBS=True )
        for node, childrenForNode in zip( nodes, children ):
            uSem[ childrenForNode ] -= 1
            dprint( 'Decrementing from U for child', childrenForNode, 'from parent', node, use=debug )
            assert np.all( uSem[ childrenForNode ] >= 0 )

        # Decrement vSem for all mates over down edges that node and mate are a part of
        matesAndEdges = self.mates( nodes, splitByEdge=True, split=True, noFBS=True )
        for node, mateAndEdge in zip( nodes, matesAndEdges ):
            for e, m in mateAndEdge:
                vSem.data[ np.in1d( vSem.row, m ) & np.in1d( vSem.col, e ) ] -= 1
                dprint( 'Decrementing from V for mates', m, 'at edge', e, 'from node', node, use=debug )
                assert np.all( vSem.data[ np.in1d( vSem.row, m ) & np.in1d( vSem.col, e ) ] >= 0 )

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
                assert np.all( uSem[ c ] >= 0 )

        # Decrement uSem for all siblings
        siblings = self.siblings( nodes, split=True, noFBS=True )
        for _e, node, siblingsForNode in zip( edges, nodes, siblings ):
            if( _e is None ):
                # If this node doesn't have a down edge, then we don't want to decrement
                continue
            uSem[ siblingsForNode ] -= 1
            dprint( 'Decrementing from U for sibling', siblingsForNode, 'from node', node, use=debug )
            assert np.all( uSem[ siblingsForNode ] >= 0 )

        # Decrement vSem for mates that aren't current edge
        matesAndEdges = self.mates( nodes, splitByEdge=True, split=True, noFBS=True )
        for node, edge, mateAndEdge in zip( nodes, edges, matesAndEdges ):
            for e, m in mateAndEdge:
                if( e == edge ):
                    continue
                vSem.data[ np.in1d( vSem.row, m ) & np.in1d( vSem.col, e ) ] -= 1
                dprint( 'Decrementing from V for mates', m, 'at edge', e, 'from node', node, use=debug )
                assert np.all( vSem.data[ np.in1d( vSem.row, m ) & np.in1d( vSem.col, e ) ] >= 0 )

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
            assert np.all( vSem.data[ np.in1d( vSem.row, p ) & np.in1d( vSem.col, e ) ] >= 0 )

        vDone.data[ np.in1d( vDone.row, nodes ) & np.in1d( vDone.col, edgesWithoutNone ) ] = True

    ######################################################################

    def uReady( self, nodes, uSem ):
        return nodes[ uSem[ nodes ] == 0 ], nodes[ uSem[ nodes ] != 0 ]

    def vReady( self, nodes, vSem ):
        ready = np.intersect1d( nodes, np.setdiff1d( vSem.row, vSem.nonzero()[ 0 ] ) )
        notReady = np.setdiff1d( nodes, ready )
        return ready, notReady

    ######################################################################

    def messagePassing( self, uWork, vWork, debug=False, **kwargs ):

        uDone, vDone = self.progressInit()
        uSem, vSem = self.countSemaphoreInit( debug=debug )
        uList, vList = self.baseCaseNodes()

        dprint( '\nInitial semaphore count', use=debug )
        dprint( 'uSem: \n', uSem, use=debug )
        dprint( 'vSem: \n', vSem.todense(), use=debug )
        dprint( '\nInitial node list', use=debug )
        dprint( 'uList: \n', uList, use=debug )
        dprint( 'vList: \n', vList, use=debug )
        dprint( '\nInitial done list', use=debug )
        dprint( 'uDone: \n', uDone, use=debug )
        dprint( 'vDone: \n', vDone, use=debug )

        # Do work for base case nodes
        uWork( True, uList, **kwargs )
        vWork( True, vList, **kwargs )

        i = 1

        # Filter over all of the graphs
        while( uList.size > 0 or vList[ 0 ].size > 0 ):

            dprint( '\n----------------\n', 'Iteration', i, '\n----------------\n', use=debug )

            if( i > 1 ):
              # Do work for each of the nodes
              uWork( False, uList, **kwargs )
              vWork( False, vList, **kwargs )

            # Mark that we're done with the current nodes
            self.UDone( uList, uSem, vSem, uDone, debug=debug )
            self.VDone( vList, uSem, vSem, vDone, debug=debug )

            if( debug ):
                print( '\nSemaphore count after marking nodes done' )
                print( 'uSem: \n', uSem )
                print( 'vSem: \n', vSem.todense() )

            # Find the next nodes that are ready
            uList = self.readyForU( uSem, uDone, debug=debug )
            vList = self.readyForV( vSem, vDone, debug=debug )

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

            # # Check if we need to do loopy propogation belief
            # if( ( uList.size == 0 and vList[ 0 ].size == 0 ) and \
            #     ( not np.any( uDone ) or not np.any( vDone.data ) ) ):
            #     loopy = True
        assert np.any( uSem != 0 ) == False
        assert np.any( vSem.data != 0 ) == False