import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix
import graphviz
from collections import Iterable

__all__ = [ 'Graph', 'GraphMessagePasser', 'GraphMessagePasserFBS']

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
                d.edge( '%d '%( p ), '%d'%( e ), **{
                    'fixedsize': 'true'
                } )
            for c in children:
                d.edge( '%d'%( e ), '%d '%( c ), **{
                    'fixedsize': 'true'
                } )

            d.node( '%d'%( e ), **{
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
                d.node( '%d '%( n ), **{
                       'style': 'filled',
                       'fontcolor': 'white',
                       'fillcolor':'blue'
                       } )

        if( render ):
            d.render()

        return d

######################################################################

class GraphMessagePasser():

    def toGraph( self ):
        return Graph.fromParentChildMask( self.pmask, self.cmask )

    def draw( self, render=True, **kwargs ):
        return self.toGraph().draw( render=render, **kwargs )

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

    def updateParamsFromGraphs( self, graphs ):

        parentMasks = []
        childMasks = []
        for graph in graphs:
            if( isinstance( graph, Iterable ) ):
                assert len( graph ) == 2
                graph, fbs = graph
            else:
                fbs = None

            pMask, cMask = graph.toMatrix()
            parentMasks.append( pMask )
            childMasks.append( cMask )

        self.updateParams( parentMasks, childMasks )

    def updateParams( self, parentMasks, childMasks ):

        assert len( parentMasks ) == len( childMasks )
        for childMask, parentMask in zip( childMasks, parentMasks ):
            assert isinstance( childMask, coo_matrix )
            assert isinstance( parentMask, coo_matrix )
            assert childMask.shape == parentMask.shape

        self.pmask, self.parentGraphAssignments = self.concatSparseMatrix( parentMasks )
        self.cmask, self.childGraphAssignments = self.concatSparseMatrix( childMasks )

        self.nodes = np.arange( self.pmask.shape[ 0 ] )

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
    def _nodesFromEdges( nodes,
                         edges,
                         cmask,
                         pmask,
                         getChildren=True,
                         diffNodes=False,
                         getOrder=False ):

        mask = cmask if getChildren else pmask

        edgeMask = np.in1d( mask.col, edges )

        if( diffNodes ):
            finalMask = edgeMask & ~np.in1d( mask.row, nodes )
        else:
            finalMask = edgeMask

        if( getOrder is False ):
            return np.unique( mask.row[ finalMask ] )
        return mask.row[ finalMask ], mask.data[ finalMask ] - 1 # Subtract one to use 0 indexing

    @staticmethod
    def _nodeSelectFromEdge( cmask,
                             pmask,
                             nodes,
                             edges=None,
                             upEdge=False,
                             getChildren=True,
                             diffNodes=False,
                             splitByEdge=False,
                             split=False,
                             getOrder=False ):

        if( split ):
            if( edges is None ):
                return [ GraphMessagePasser._nodeSelectFromEdge( cmask,
                                                                 pmask,
                                                                 n,
                                                                 edges=None,
                                                                 upEdge=upEdge,
                                                                 getChildren=getChildren,
                                                                 diffNodes=diffNodes,
                                                                 splitByEdge=splitByEdge,
                                                                 split=False,
                                                                 getOrder=getOrder ) for n in nodes ]
            else:
                return [ GraphMessagePasser._nodeSelectFromEdge( cmask,
                                                                 pmask,
                                                                 n,
                                                                 edges=e,
                                                                 upEdge=upEdge,
                                                                 getChildren=getChildren,
                                                                 diffNodes=diffNodes,
                                                                 splitByEdge=splitByEdge,
                                                                 split=False,
                                                                 getOrder=getOrder ) for n, e in zip( nodes, edges ) ]

        _edges = GraphMessagePasser._upEdges( cmask, nodes ) if upEdge else GraphMessagePasser._downEdges( pmask, nodes )

        if( edges is not None ):
            _edges = np.intersect1d( _edges, edges )

        if( splitByEdge == True ):
            return [ [ e, GraphMessagePasser._nodeSelectFromEdge( cmask,
                                                                  pmask,
                                                                  nodes,
                                                                  edges=e,
                                                                  upEdge=upEdge,
                                                                  getChildren=getChildren,
                                                                  diffNodes=diffNodes,
                                                                  splitByEdge=False,
                                                                  split=False,
                                                                  getOrder=getOrder ) ] for e in _edges ]

        return GraphMessagePasser._nodesFromEdges( nodes,
                                                   _edges,
                                                   cmask,
                                                   pmask,
                                                   getChildren=getChildren,
                                                   diffNodes=diffNodes,
                                                   getOrder=getOrder )

    ######################################################################

    @staticmethod
    def _parents( cmask, pmask, nodes, split=False, getOrder=False ):
        return GraphMessagePasser._nodeSelectFromEdge( cmask,
                                                       pmask,
                                                       nodes,
                                                       edges=None,
                                                       upEdge=True,
                                                       getChildren=False,
                                                       diffNodes=False,
                                                       splitByEdge=False,
                                                       split=split,
                                                       getOrder=getOrder )

    @staticmethod
    def _siblings( cmask, pmask, nodes, split=False, getOrder=False ):
        return GraphMessagePasser._nodeSelectFromEdge( cmask,
                                                       pmask,
                                                       nodes, edges=None,
                                                       upEdge=True,
                                                       getChildren=True,
                                                       diffNodes=True,
                                                       splitByEdge=False,
                                                       split=split,
                                                       getOrder=getOrder )

    @staticmethod
    def _children( cmask, pmask, nodes, edges=None, splitByEdge=False, split=False, getOrder=False ):
        return GraphMessagePasser._nodeSelectFromEdge( cmask,
                                                       pmask,
                                                       nodes,
                                                       edges=edges,
                                                       upEdge=False,
                                                       getChildren=True,
                                                       diffNodes=False,
                                                       splitByEdge=splitByEdge,
                                                       split=split,
                                                       getOrder=getOrder )

    @staticmethod
    def _mates( cmask, pmask, nodes, edges=None, splitByEdge=False, split=False, getOrder=False ):
        return GraphMessagePasser._nodeSelectFromEdge( cmask,
                                                       pmask,
                                                       nodes,
                                                       edges=edges,
                                                       upEdge=False,
                                                       getChildren=False,
                                                       diffNodes=True,
                                                       splitByEdge=splitByEdge,
                                                       split=split,
                                                       getOrder=getOrder )

    ######################################################################

    def parents( self, nodes, split=False, getOrder=False ):
        return GraphMessagePasser._parents( self.cmask,
                                            self.pmask,
                                            nodes,
                                            split=split,
                                            getOrder=getOrder )

    def siblings( self, nodes, split=False ):
        return GraphMessagePasser._siblings( self.cmask,
                                             self.pmask,
                                             nodes,
                                             split=split )

    def children( self, nodes, edges=None, splitByEdge=False, split=False ):
        return GraphMessagePasser._children( self.cmask,
                                             self.pmask,
                                             nodes,
                                             edges=edges,
                                             splitByEdge=splitByEdge,
                                             split=split )

    def mates( self, nodes, edges=None, splitByEdge=False, split=False, getOrder=False ):
        return GraphMessagePasser._mates( self.cmask,
                                          self.pmask,
                                          nodes,
                                          edges=edges,
                                          splitByEdge=splitByEdge,
                                          split=split,
                                          getOrder=getOrder )

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

        # Generate the up and down base arrays
        uList = rootIndices
        vList = leafIndices

        vListNodes = []
        vListEdges = []
        for v in vList:
            vListNodes.append( v )
            vListEdges.append( None )

        return uList, [ vListNodes, vListEdges ]

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
            # U:
            #  - U for all parents
            #  - V for all parents over all down edges except node's up edge
            #  - V for all siblings over all down edges
            upEdge = self.upEdges( n )
            parents = self.parents( n )

            USemData[ n ] += parents.shape[ 0 ]

            for parent in parents:

                downEdges = self.downEdges( parent, skipEdges=upEdge )
                USemData[ n ] += downEdges.shape[ 0 ]

            siblings = self.siblings( n )
            for sibling in siblings:
                downEdges = self.downEdges( sibling )
                USemData[ n ] += downEdges.shape[ 0 ]

        VSemData = np.zeros_like( self.pmask.row )

        for i, ( n, e, _ ) in enumerate( zip( self.pmask.row, self.pmask.col, self.pmask.data ) ):
            # V:
            #  - U for all mates from e
            #  - V for all mates over all down edges for mate except for e
            #  - V for all children from e over all down edges for child

            mates = self.mates( n, edges=e )

            VSemData[ i ] += mates.shape[ 0 ]

            for mate in mates:
                downEdges = self.downEdges( mate, skipEdges=e )
                VSemData[ i ] += downEdges.shape[ 0 ]

            children = self.children( n, edges=e )
            for child in children:
                downEdges = self.downEdges( child )
                VSemData[ i ] += downEdges.shape[ 0 ]

        uSem = USemData
        vSem = coo_matrix( ( VSemData, ( self.pmask.row, self.pmask.col ) ), shape=self.pmask.shape, dtype=int )

        return uSem, vSem

    ######################################################################

    def readyForU( self, uSem, uDone, debug=False ):
        return self.nodes[ ( uSem == 0 ) & ~uDone ]

    def readyForV( self, vSem, vDone, debug=False ):
        mask = ( vSem.data == 0 ) & ~vDone.data
        return vSem.row[ mask ], vSem.col[ mask ]

    ######################################################################

    def UDone( self, nodes, uSem, vSem, uDone, debug=False ):

        # Decrement uSem for children
        children = self.children( nodes, split=True )
        for node, childrenForNode in zip( nodes, children ):
            uSem[ childrenForNode ] -= 1
            assert np.all( uSem[ childrenForNode ] >= 0 )

        # Decrement vSem for all mates over down edges that node and mate are a part of
        matesAndEdges = self.mates( nodes, splitByEdge=True, split=True )
        for node, mateAndEdge in zip( nodes, matesAndEdges ):
            for e, m in mateAndEdge:
                vSem.data[ np.in1d( vSem.row, m ) & np.in1d( vSem.col, e ) ] -= 1
                assert np.all( vSem.data[ np.in1d( vSem.row, m ) & np.in1d( vSem.col, e ) ] >= 0 )

        uDone[ nodes ] = True

    def VDone( self, nodesAndEdges, uSem, vSem, vDone, debug=False ):

        nodes, edges = nodesAndEdges
        edgesWithoutNone = np.array( [ e for e in edges if e is not None ] )

        notCurrentEdge = np.setdiff1d( vSem.col, edgesWithoutNone )

        # Decrement uSem for children that come from a different edge than the one computed for V
        childrenAndEdges = self.children( nodes, splitByEdge=True, split=True )
        for node, edge, childAndEdge in zip( nodes, edges, childrenAndEdges ):
            for e, c in childAndEdge:
                if( e == edge ):
                    continue
                uSem[ c ] -= 1
                assert np.all( uSem[ c ] >= 0 )

        # Decrement uSem for all siblings
        siblings = self.siblings( nodes, split=True )
        for _e, node, siblingsForNode in zip( edges, nodes, siblings ):
            if( _e is None ):
                # If this node doesn't have a down edge, then we don't want to decrement
                continue
            uSem[ siblingsForNode ] -= 1
            assert np.all( uSem[ siblingsForNode ] >= 0 )

        # Decrement vSem for mates that aren't current edge
        matesAndEdges = self.mates( nodes, splitByEdge=True, split=True )
        for node, edge, mateAndEdge in zip( nodes, edges, matesAndEdges ):
            for e, m in mateAndEdge:
                if( e == edge ):
                    continue
                vSem.data[ np.in1d( vSem.row, m ) & np.in1d( vSem.col, e ) ] -= 1
                assert np.all( vSem.data[ np.in1d( vSem.row, m ) & np.in1d( vSem.col, e ) ] >= 0 )

        # Decrement vSem for parents over up edges
        parents = self.parents( nodes, split=True )
        upEdges = self.upEdges( nodes, split=True )
        for _e, p, e in zip( edges, parents, upEdges ):
            if( _e is None ):
                # If this node doesn't have a down edge, then we don't want to decrement
                continue
            vSem.data[ np.in1d( vSem.row, p ) & np.in1d( vSem.col, e ) ] -= 1
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

        # Do work for base case nodes
        uWork( True, uList, **kwargs )
        vWork( True, vList, **kwargs )

        i = 1

        # Filter over all of the graphs
        while( uList.size > 0 or vList[ 0 ].size > 0 ):

            if( i > 1 ):
              # Do work for each of the nodes
              uWork( False, uList, **kwargs )
              vWork( False, vList, **kwargs )

            # Mark that we're done with the current nodes
            self.UDone( uList, uSem, vSem, uDone, debug=debug )
            self.VDone( vList, uSem, vSem, vDone, debug=debug )

            # Find the next nodes that are ready
            uList = self.readyForU( uSem, uDone, debug=debug )
            vList = self.readyForV( vSem, vDone, debug=debug )

            i += 1

            # # Check if we need to do loopy propogation belief
            # if( ( uList.size == 0 and vList[ 0 ].size == 0 ) and \
            #     ( not np.any( uDone ) or not np.any( vDone.data ) ) ):
            #     loopy = True
        assert np.any( uSem != 0 ) == False
        assert np.any( vSem.data != 0 ) == False

    ######################################################################

    def full_parents( self, nodes, split=False, getOrder=False ):
        return self.parents( nodes, split=split, getOrder=getOrder )

    def full_siblings( self, nodes, split=False ):
        return self.siblings( nodes, split=split )

    def full_children( self, nodes, edges=None, splitByEdge=False, split=False ):
        return self.children( nodes, edges=edges, splitByEdge=splitByEdge, split=split )

    def full_mates( self, nodes, edges=None, splitByEdge=False, split=False, getOrder=False ):
        return self.mates( nodes, edges=edges, splitByEdge=splitByEdge, split=split, getOrder=getOrder )

##########################################################################################################
##########################################################################################################

class __FBSMessagePassingMixin():

    def toGraph( self, usePartial=False ):
        if( usePartial ):
            return Graph.fromParentChildMask( self.pmask, self.cmask )
        return Graph.fromParentChildMask( self.full_pmask, self.full_cmask )

    def draw( self, usePartial=False, render=True ):
        if( usePartial ):
            return self.toGraph( usePartial=True ).draw( render=render )
        return self.toGraph().draw( render=render, cutNodes=self.fbs )

    def fbsConcat( self, feedbackSets, nodeCounts ):
        assert len( feedbackSets ) == len( nodeCounts )
        bigFBS = []
        totalN = 0
        for fbs, N in zip( feedbackSets, nodeCounts ):
            if( fbs is not None ):
                bigFBS.append( fbs + totalN )
            else:
                bigFBS.append( np.array( [] ) )
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

        # Save off the full pmask, cmask and node set and only use the graph without
        # the fbs nodes for message passing.

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

        self.full_pmask, self.parentGraphAssignments = self.concatSparseMatrix( parentMasks )
        self.full_cmask, self.childGraphAssignments = self.concatSparseMatrix( childMasks )

        self.full_nodes = np.arange( self.full_pmask.shape[ 0 ] )

        if( feedbackSets is not None ):
            nodeCounts = [ mat.shape[ 0 ] for mat in parentMasks ]
            # self.feedbackSets contains all of the feedback sets with the adjusted node indices
            fbsNodes, self.feedbackSets = self.fbsConcat( feedbackSets, nodeCounts )
            self.fbsMask = np.in1d( self.full_nodes, fbsNodes )
        else:
            self.fbsMask = np.zeros( self.full_pmask.shape[ 0 ], dtype=bool )

        # All of the feedback sets together
        self.fbs = self.full_nodes[ self.fbsMask ]

        # Parent and child mask for feedback set nodes
        self.fbsPMask = np.in1d( self.full_pmask.row, self.fbs )
        self.fbsCMask = np.in1d( self.full_cmask.row, self.fbs )

        # Create a mapping from the full indices to the new indices
        nonFBS = self.full_nodes[ ~self.fbsMask ]
        nonFBSReIndexed = ( self.full_nodes - self.fbsMask.cumsum() )[ ~self.fbsMask ]
        indexMap = dict( zip( nonFBS, nonFBSReIndexed ) )
        indexMapReverse = dict( zip( nonFBSReIndexed, nonFBS ) )

        # Re-index the fbs nodes starting from nonFBS.size
        fbsIndexMap = dict( zip( self.fbs, np.arange( self.fbs.shape[ 0 ] ) + nonFBS.shape[ 0 ] ) )
        fbsIndexMapReverse = dict( zip( np.arange( self.fbs.shape[ 0 ] ) + nonFBS.shape[ 0 ], self.fbs ) )
        indexMap.update( fbsIndexMap )
        indexMapReverse.update( fbsIndexMapReverse )

        self.fullIndexToReduced = np.vectorize( lambda x: indexMap[ x ] )
        self.reducedIndexToFull = np.vectorize( lambda x: indexMapReverse[ x ] )

        # Create the new list of nodes
        self.nodes = self.fullIndexToReduced( nonFBS )

        # TODO: MAKE REDUCED INDICES A DIFFERENT TYPE FROM FULL INDICES!!!!!

        # Get a mask over where the fbs nodes arent and
        # make the new sparse matrix parameters
        mask = ~np.in1d( self.full_pmask.row, self.fbs )
        _pmaskRow, _pmaskCol, _pmaskData = self.full_pmask.row[ mask ], self.full_pmask.col[ mask ], self.full_pmask.data[ mask ]
        _pmaskRow = self.fullIndexToReduced( _pmaskRow )

        mask = ~np.in1d( self.full_cmask.row, self.fbs )
        _cmaskRow, _cmaskCol, _cmaskData = self.full_cmask.row[ mask ], self.full_cmask.col[ mask ], self.full_cmask.data[ mask ]
        _cmaskRow = self.fullIndexToReduced( _cmaskRow )

        # The new shape will have fewer nodes
        shape = ( self.full_pmask.shape[ 0 ] - self.fbs.shape[ 0 ], self.full_pmask.shape[ 1 ] )
        self.pmask = coo_matrix( ( _pmaskData, ( _pmaskRow, _pmaskCol ) ), shape=shape, dtype=int )

        shape = ( self.full_cmask.shape[ 0 ] - self.fbs.shape[ 0 ], self.full_cmask.shape[ 1 ] )
        self.cmask = coo_matrix( ( _cmaskData, ( _cmaskRow, _cmaskCol ) ), shape=shape, dtype=int )

    ######################################################################

    def inFBS( self, node, fromReduced=True ):
        if( fromReduced ):
            return self.reducedIndexToFull( node ) in self.fbs
        return node in self.fbs

    def fbsIndex( self, node, fromReduced=True, withinGraph=True ):
        fullNode = self.reducedIndexToFull( node ) if fromReduced else node

        if( withinGraph == False ):
            return self.fbs.tolist().index( node )

        for fbs in self.feedbackSets:
            if( fullNode in fbs ):
                return fbs.tolist().index( fullNode )

        assert 0, 'This is not a fbs node'

    ######################################################################

    def splitNodesFromFBS( self, nodes ):
        if( not isinstance( nodes, Iterable ) ):
            nodes = [ nodes ]
        if( isinstance( nodes, np.ndarray ) and nodes.ndim == 0 ):
            nodes = nodes[ None ]
        fbsNodes = [ n for n in nodes if self.inFBS( n, fromReduced=True ) ]
        nonFBSNodes = [ n for n in nodes if not self.inFBS( n, fromReduced=True ) ]
        return fbsNodes, nonFBSNodes

    def removeFBSFromNodes( self, nodes ):
        if( not isinstance( nodes, Iterable ) ):
            nodes = [ nodes ]
        return np.array( [ n for n in nodes if not self.inFBS( n, fromReduced=True ) ] )

    def removeFBSFromNodesAndOrder( self, nodes, order ):
        if( not isinstance( nodes, Iterable ) ):
            nodes = [ nodes ]
        if( nodes.size == 0 ):
            return nodes, order
        nodesOrder = zip( *[ ( n, o ) for n, o in zip( nodes, order ) if not self.inFBS( n, fromReduced=True ) ] )
        if( len( list( nodesOrder ) ) > 0 ):
            # Iterator is consumed in check above
            _nodes, _order = zip( *[ ( n, o ) for n, o in zip( nodes, order ) if not self.inFBS( n, fromReduced=True ) ] )
            return np.array( _nodes ), np.array( _order )
        else:
            return np.array( [] ), np.array( [] )

    def removeFBSFromSplitNodes( self, nodes ):
        if( not isinstance( nodes, Iterable ) ):
            nodes = [ nodes ]
        nodes_fbs = []
        for n in nodes:
            currentFBS = []
            for _n in n:
                if( not self.inFBS( _n ) ):
                    currentFBS.append( _n )
            nodes_fbs.append( currentFBS )
        return nodes_fbs

    def removeFBSFromSplitNodesAndOrder( self, nodes, order ):
        if( not isinstance( nodes, Iterable ) ):
            nodes = [ nodes ]
        nodes_fbs = []
        nodesOrder_fbs = []
        for n, o in zip( nodes, order ):
            currentFBS = []
            currentOrder = []
            for _n, _o in zip( n, o ):
                if( not self.inFBS( _n ) ):
                    currentFBS.append( _n )
                    currentOrder.append( _o )
            nodes_fbs.append( currentFBS )
            nodesOrder_fbs.append( currentOrder )
        return np.array( nodes_fbs ), np.array( nodesOrder_fbs )

    def removeFBSFromSplitEdges( self, nodes ):
        assert 0, 'Implement this'

    def removeFBSFromSplitNodesAndEdges( self, nodes ):
        assert 0, 'Implement this'

    def removeFBSFromSplitEdgesAndOrder( self, nodes, order ):
        assert 0, 'Implement this'

    def removeFBSFromSplitNodesAndEdgesAndOrder( self, nodes, order ):
        assert 0, 'Implement this'

    ######################################################################

    def parents( self, nodes, split=False, getOrder=False ):

        fbsNodes, nonFBSNodes = self.splitNodesFromFBS( nodes )

        nonFBSAns = GraphMessagePasser._parents( self.cmask,
                                                 self.pmask,
                                                 nonFBSNodes,
                                                 split=split,
                                                 getOrder=getOrder )
        if( len( fbsNodes ) == 0 ):
            return nonFBSAns

        if( getOrder == True ):
            full_parents, full_parentOrder = self.full_parents( fbsNodes, split=split, getOrder=getOrder )
            if( split == False ):
                parents_fbs, parentOrder_fbs = self.removeFBSFromNodesAndOrder( full_parents, full_parentOrder )
            else:
                parent_fbs, parentOrder_fbs = self.removeFBSFromSplitNodesAndOrder( full_parents, full_parentOrder )

            parents, parentOrder = nonFBSAns
            assert type( parents ) == type( parents_fbs ), '%s, %s'%( type( parents ), type( parents_fbs ) )
            assert type( parentOrder ) == type( parentOrder_fbs )
            return np.hstack( ( parents_fbs, parents ) ), np.hstack( ( parentOrder_fbs, parentOrder ) )
        else:
            full_parents = self.full_parents( fbsNodes, split=split, getOrder=getOrder )
            if( split == False ):
                parents_fbs = self.removeFBSFromNodes( full_parents )
            else:
                parent_fbs = self.removeFBSFromSplitNodes( full_parents )

            parents = nonFBSAns
            assert type( parents ) == type( parents_fbs )
            return np.hstack( ( parents_fbs, parents ) )

    def siblings( self, nodes, split=False ):

        fbsNodes, nonFBSNodes = self.splitNodesFromFBS( nodes )

        nonFBSAns = GraphMessagePasser._siblings( self.cmask,
                                                  self.pmask,
                                                  nonFBSNodes,
                                                  split=split )
        if( len( fbsNodes ) == 0 ):
            return nonFBSAns

        full_siblings = self.full_siblings( fbsNodes, split=split )
        if( split == False ):
            siblings_fbs = self.removeFBSFromNodes( full_siblings )
        else:
            siblings_fbs = self.removeFBSFromSplitNodes( full_siblings )

        assert type( siblings ) == type( siblings_fbs )
        return np.hstack( ( siblings_fbs, siblings ) )

    def children( self, nodes, edges=None, splitByEdge=False, split=False ):

        fbsNodes, nonFBSNodes = self.splitNodesFromFBS( nodes )

        nonFBSAns = GraphMessagePasser._children( self.cmask,
                                                  self.pmask,
                                                  nonFBSNodes,
                                                  edges=edges,
                                                  splitByEdge=splitByEdge,
                                                  split=split )
        if( len( fbsNodes ) == 0 ):
            return nonFBSAns

        full_children = self.full_children( fbsNodes, edges=edges, splitByEdge=splitByEdge, split=split )

        if( splitByEdge == False ):
            if( split == False ):
                children_fbs = self.removeFBSFromNodes( full_children )
            else:
                children_fbs = self.removeFBSFromSplitNodes( full_children )

            assert type( children ) == type( children_fbs )
            return np.hstack( ( children_fbs, children ) )
        else:
            if( split == False ):
                children_fbs = self.removeFBSFromSplitEdges( full_children )
            else:
                children_fbs = self.removeFBSFromSplitNodesAndEdges( full_children )
            assert 0

    def mates( self, nodes, edges=None, splitByEdge=False, split=False, getOrder=False ):

        fbsNodes, nonFBSNodes = self.splitNodesFromFBS( nodes )

        nonFBSAns = GraphMessagePasser._mates( self.cmask,
                                               self.pmask,
                                               nonFBSNodes,
                                               edges=edges,
                                               splitByEdge=splitByEdge,
                                               split=split,
                                               getOrder=getOrder )
        if( len( fbsNodes ) == 0 ):
            return nonFBSAns

        if( splitByEdge == False ):
            if( getOrder == True ):
                full_mates, full_mateOrder = self.full_mates( fbsNodes, edges=edges, splitByEdge=splitByEdge, split=split, getOrder=getOrder )
                if( split == False ):
                    mates_fbs, matesOrder_fbs = self.removeFBSFromNodesAndOrder( full_mates, full_mateOrder )
                else:
                    mates_fbs, matesOrder_fbs = self.removeFBSFromSplitNodesAndOrder( full_mates, full_mateOrder )

                mates, matesOrder = nonFBSAns
                assert type( mates ) == type( mates_fbs )
                assert type( matesOrder ) == type( matesOrder_fbs )
                return np.hstack( ( mates_fbs, mates ) ), np.hstack( ( matesOrder_fbs, matesOrder ) )
            else:
                full_mates = self.full_mates( fbsNodes, edges=edges, splitByEdge=splitByEdge, split=split, getOrder=getOrder )
                if( split == False ):
                    mates_fbs = self.removeFBSFromNodes( full_mates )
                else:
                    mates_fbs = self.removeFBSFromSplitNodes( full_mates )

                assert type( mates ) == type( mates_fbs )
                return np.hstack( ( mates_fbs, mates ) )

        else:
            if( getOrder == True ):
                if( split == False ):
                    mates_fbs, matesOrder_fbs = self.removeFBSFromSplitEdgesAndOrder( full_mates )
                else:
                    mates_fbs, matesOrder_fbs = self.removeFBSFromSplitNodesAndEdgesAndOrder( full_mates )
            else:
                if( split == False ):
                    mates_fbs = self.removeFBSFromSplitEdges( full_mates )
                else:
                    mates_fbs = self.removeFBSFromSplitNodesAndEdges( full_mates )
            assert 0

    ######################################################################

    def full_parents( self, nodes, split=False, getOrder=False, fullIndexing=False, returnFullIndex=False ):
        if( fullIndexing == False ):
            if( isinstance( nodes, np.ndarray ) and nodes.ndim == 0 ):
                nodes = self.reducedIndexToFull( nodes )
            else:
                nodes = [ self.reducedIndexToFull( n ) for n in nodes ] if isinstance( nodes, Iterable ) else self.reducedIndexToFull( nodes )

        ans = GraphMessagePasser._parents( self.full_cmask,
                                           self.full_pmask,
                                           nodes,
                                           split=split,
                                           getOrder=getOrder )
        if( getOrder ):
            ans, order = ans
            if( returnFullIndex == False ):
                ans = [ self.fullIndexToReduced( n ) for n in ans ]
            if( not ( split ) ):
                ans = np.array( ans )
            ans = ans, order
        else:
            if( returnFullIndex == False ):
                ans = [ self.fullIndexToReduced( n ) for n in ans ]

        if( split ):
            return ans
        return np.array( ans )

    def full_siblings( self, nodes, split=False, fullIndexing=False, returnFullIndex=False ):
        if( fullIndexing == False ):
            if( isinstance( nodes, np.ndarray ) and nodes.ndim == 0 ):
                nodes = self.reducedIndexToFull( nodes )
            else:
                nodes = [ self.reducedIndexToFull( n ) for n in nodes ] if isinstance( nodes, Iterable ) else self.reducedIndexToFull( nodes )

        ans = GraphMessagePasser._siblings( self.full_cmask,
                                            self.full_pmask,
                                            nodes,
                                            split=split )

        if( returnFullIndex == False ):
            ans = [ self.fullIndexToReduced( n ) for n in ans ]

        if( split ):
            return ans
        return np.array( ans )

    def full_children( self, nodes, edges=None, splitByEdge=False, split=False, fullIndexing=False, returnFullIndex=False ):
        if( fullIndexing == False ):
            if( isinstance( nodes, np.ndarray ) and nodes.ndim == 0 ):
                nodes = self.reducedIndexToFull( nodes )
            else:
                nodes = [ self.reducedIndexToFull( n ) for n in nodes ] if isinstance( nodes, Iterable ) else self.reducedIndexToFull( nodes )

        ans = GraphMessagePasser._children( self.full_cmask,
                                            self.full_pmask,
                                            nodes,
                                            edges=edges,
                                            splitByEdge=splitByEdge,
                                            split=split )

        if( returnFullIndex == False ):
            ans = [ self.fullIndexToReduced( n ) for n in ans ]

        if( split or splitByEdge ):
            return ans
        return np.array( ans )

    def full_mates( self, nodes, edges=None, splitByEdge=False, split=False, getOrder=False, fullIndexing=False, returnFullIndex=False ):
        if( fullIndexing == False ):
            if( isinstance( nodes, np.ndarray ) and nodes.ndim == 0 ):
                nodes = self.reducedIndexToFull( nodes )
            else:
                nodes = [ self.reducedIndexToFull( n ) for n in nodes ] if isinstance( nodes, Iterable ) else self.reducedIndexToFull( nodes )

        ans = GraphMessagePasser._mates( self.full_cmask,
                                         self.full_pmask,
                                         nodes,
                                         edges=edges,
                                         splitByEdge=splitByEdge,
                                         split=split,
                                         getOrder=getOrder )
        if( getOrder ):
            ans, order = ans
            if( returnFullIndex == False ):
                ans = [ self.fullIndexToReduced( n ) for n in ans ]
            if( not ( split or splitByEdge ) ):
                ans = np.array( ans )
            ans = ans, order
        else:
            if( returnFullIndex == False ):
                ans = [ self.fullIndexToReduced( n ) for n in ans ]

        if( split or splitByEdge ):
            return ans
        return np.array( ans )

    ######################################################################

    def upEdges( self, nodes, split=False, fromFull=True ):
        if( fromFull == False ):
            return GraphMessagePasser._upEdges( self.cmask, nodes, split=split )

        if( isinstance( nodes, np.ndarray ) and nodes.ndim == 0 ):
            nodes = self.reducedIndexToFull( nodes )
        else:
            nodes = [ self.reducedIndexToFull( n ) for n in nodes ] if isinstance( nodes, Iterable ) else self.reducedIndexToFull( nodes )

        return GraphMessagePasser._upEdges( self.full_cmask, nodes, split=split )

    def downEdges( self, nodes, skipEdges=None, split=False, fromFull=True ):
        if( fromFull == False ):
            return GraphMessagePasser._downEdges( self.pmask, nodes, skipEdges=skipEdges, split=split )

        if( isinstance( nodes, np.ndarray ) and nodes.ndim == 0 ):
            nodes = self.reducedIndexToFull( nodes )
        else:
            nodes = [ self.reducedIndexToFull( n ) for n in nodes ] if isinstance( nodes, Iterable ) else self.reducedIndexToFull( nodes )

        return GraphMessagePasser._downEdges( self.full_pmask, nodes, skipEdges=skipEdges, split=split )

##########################################################################################################

class GraphMessagePasserFBS( __FBSMessagePassingMixin, GraphMessagePasser ):
    pass