import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix
import graphviz
from collections import Iterable
from functools import partial
import itertools

__all__ = [ 'Graph', 'GraphMessagePasser', 'GraphMessagePasserFBS', 'GraphMessagePasserFBSImproved' ]

class Graph():
    # This class is how we make sparse matrices

    def __init__( self ):
        self.nodes = set()
        self.edge_children = list()
        self.edge_parents = list()

    @staticmethod
    def fromParentChildMask( p_mask, c_mask ):
        graph = Graph()
        assert p_mask.shape == c_mask.shape
        nEdges = p_mask.shape[ 1 ]
        for e in range( nEdges ):
            parents = p_mask.getcol( e ).nonzero()[ 0 ]
            children = c_mask.getcol( e ).nonzero()[ 0 ]
            graph.addEdge( parents=parents.tolist(), children=children.tolist() )

        return graph

    def addEdge( self, parents, children ):
        assert isinstance( parents, list ) or isinstance( parents, tuple )
        assert isinstance( children, list ) or isinstance( children, tuple )
        for node in parents + children:
            self.nodes.add( node )

        self.edge_children.append( children )
        self.edge_parents.append( parents )

    def cooMatrixFromNodeEdge( self, nodes, edges ):

        n_rows = len( nodes )
        n_cols = len( edges )

        rows = []
        cols = []
        data = []

        for i, node_group in enumerate( edges ):
            for j, node in enumerate( node_group ):
                row_index = nodes.index( node )
                col_index = i

                rows.append( row_index )
                cols.append( col_index )

                # Use an integer so that we can have an ordering of nodes within edges!!!!
                data.append( j + 1 )

        mask = coo_matrix( ( data, ( rows, cols ) ), shape=( n_rows, n_cols ), dtype=int )
        return mask

    def toMatrix( self ):

        node_list = list( self.nodes )

        parent_mask = self.cooMatrixFromNodeEdge( node_list, self.edge_parents )
        child_mask = self.cooMatrixFromNodeEdge( node_list, self.edge_children )

        return parent_mask, child_mask

    def draw( self, render=True, cut_nodes=None ):

        # Draws the graph using graphviz
        d = graphviz.Digraph()
        for e, ( parents, children ) in enumerate( zip( self.edge_parents, self.edge_children ) ):
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

        if( cut_nodes is not None ):
            for n in cut_nodes:
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

    def concatSparseMatrix( self, sparse_matrices ):
        # Builds a big block diagonal matrix where each diagonal matrix
        # is an element in sparse_matrices

        row = np.array( [], dtype=int )
        col = np.array( [], dtype=int )
        data = np.array( [], dtype=int )
        graph_assigments = []
        n_rows = 0
        n_cols = 0
        for i, mat in enumerate( sparse_matrices ):
            m, n = mat.shape
            row = np.hstack( ( row, mat.row + n_rows ) )
            col = np.hstack( ( col, mat.col + n_cols ) )
            data = np.hstack( ( data, mat.data ) )
            n_rows += m
            n_cols += n
            graph_assigments.append( n_rows )
        return coo_matrix( ( data, ( row, col ) ), shape=( n_rows, n_cols ), dtype=int ), graph_assigments

    def updateParamsFromGraphs( self, graphs ):

        parent_masks = []
        child_masks = []
        for graph in graphs:
            if( isinstance( graph, Iterable ) ):
                assert len( graph ) == 2
                graph, fbs = graph
            else:
                fbs = None

            p_mask, c_mask = graph.toMatrix()
            parent_masks.append( p_mask )
            child_masks.append( c_mask )

        self.updateParams( parent_masks, child_masks )

    def updateParams( self, parent_masks, child_masks ):

        assert len( parent_masks ) == len( child_masks )
        for child_mask, parent_mask in zip( child_masks, parent_masks ):
            assert isinstance( child_mask, coo_matrix )
            assert isinstance( parent_mask, coo_matrix )
            assert child_mask.shape == parent_mask.shape

        self.pmask, self.parent_graph_assignments = self.concatSparseMatrix( parent_masks )
        self.cmask, self.child_graph_assignments = self.concatSparseMatrix( child_masks )

        self.nodes = np.arange( self.pmask.shape[ 0 ] )

    ######################################################################

    @staticmethod
    def _upEdges( cmask, nodes, split=False ):
        if( split ):
            return [ GraphMessagePasser._upEdges( cmask, n, split=False ) for n in nodes ]
        rows, cols = cmask.nonzero()
        return np.unique( cols[ np.in1d( rows, nodes ) ] )

    @staticmethod
    def _downEdges( pmask, nodes, skip_edges=None, split=False ):
        if( split ):
            return [ GraphMessagePasser._downEdges( pmask, n, skip_edges=skip_edges, split=False ) for n in nodes ]
        if( skip_edges is not None ):
            return np.setdiff1d( GraphMessagePasser._downEdges( pmask, nodes, skip_edges=None, split=False ), skip_edges )
        rows, cols = pmask.nonzero()
        return np.unique( cols[ np.in1d( rows, nodes ) ] )

    ######################################################################

    def getUpEdges( self, nodes, split=False ):
        return GraphMessagePasser._upEdges( self.cmask, nodes, split=split )

    def getDownEdges( self, nodes, skip_edges=None, split=False ):
        return GraphMessagePasser._downEdges( self.pmask, nodes, skip_edges=skip_edges, split=split )

    ######################################################################

    @staticmethod
    def _nodesFromEdges( nodes,
                         edges,
                         cmask,
                         pmask,
                         get_children=True,
                         diff_nodes=False,
                         get_order=False ):

        # This is the logic to get different members from the same family.
        #   node         - The nodes that index into either the pmask or cmask row
        #   edges        - The edges that index into either the pmask or cmask col
        #   cmask        - The mask that contains info as to whether a node is a
        #                  child of a particular edge
        #   pmask        - Contains info as to whether a node is a parent of an edge
        #   get_children - Whether to use the pmask or cmask
        #   diff_node    - Whether or not to subtract the nodes from the result.
        #                  Useful when finding mates, however be cautious that
        #                  if nodes contains all of the parents of an edge, a call
        #                  to getMates will return an empty array
        #   get_order    - Whether or not to return the order of the parents

        mask = cmask if get_children else pmask

        edge_mask = np.in1d( mask.col, edges )

        if( diff_nodes ):
            final_mask = edge_mask & ~np.in1d( mask.row, nodes )
        else:
            final_mask = edge_mask

        if( get_order is False ):
            # Order doesn't matter
            return np.unique( mask.row[ final_mask ] )

        return mask.row[ final_mask ], mask.data[ final_mask ] - 1 # Subtract one to use 0 indexing

    @staticmethod
    def _nodeSelectFromEdge( cmask,
                             pmask,
                             nodes,
                             edges=None,
                             up_edge=False,
                             get_children=True,
                             diff_nodes=False,
                             split_by_edge=False,
                             split=False,
                             get_order=False ):

        if( split ):
            if( edges is None ):
                return [ GraphMessagePasser._nodeSelectFromEdge( cmask,
                                                                 pmask,
                                                                 n,
                                                                 edges=None,
                                                                 up_edge=up_edge,
                                                                 get_children=get_children,
                                                                 diff_nodes=diff_nodes,
                                                                 split_by_edge=split_by_edge,
                                                                 split=False,
                                                                 get_order=get_order ) for n in nodes ]
            else:
                return [ GraphMessagePasser._nodeSelectFromEdge( cmask,
                                                                 pmask,
                                                                 n,
                                                                 edges=e,
                                                                 up_edge=up_edge,
                                                                 get_children=get_children,
                                                                 diff_nodes=diff_nodes,
                                                                 split_by_edge=split_by_edge,
                                                                 split=False,
                                                                 get_order=get_order ) for n, e in zip( nodes, edges ) ]

        _edges = GraphMessagePasser._upEdges( cmask, nodes ) if up_edge else GraphMessagePasser._downEdges( pmask, nodes )

        if( edges is not None ):
            _edges = np.intersect1d( _edges, edges )

        if( split_by_edge == True ):
            return [ [ e, GraphMessagePasser._nodeSelectFromEdge( cmask,
                                                                  pmask,
                                                                  nodes,
                                                                  edges=e,
                                                                  up_edge=up_edge,
                                                                  get_children=get_children,
                                                                  diff_nodes=diff_nodes,
                                                                  split_by_edge=False,
                                                                  split=False,
                                                                  get_order=get_order ) ] for e in _edges ]

        return GraphMessagePasser._nodesFromEdges( nodes,
                                                   _edges,
                                                   cmask,
                                                   pmask,
                                                   get_children=get_children,
                                                   diff_nodes=diff_nodes,
                                                   get_order=get_order )

    ######################################################################

    @staticmethod
    def _parents( cmask, pmask, nodes, split=False, get_order=False ):
        return GraphMessagePasser._nodeSelectFromEdge( cmask,
                                                       pmask,
                                                       nodes,
                                                       edges=None,
                                                       up_edge=True,
                                                       get_children=False,
                                                       diff_nodes=False,
                                                       split_by_edge=False,
                                                       split=split,
                                                       get_order=get_order )

    @staticmethod
    def _siblings( cmask, pmask, nodes, split=False, get_order=False ):
        return GraphMessagePasser._nodeSelectFromEdge( cmask,
                                                       pmask,
                                                       nodes,
                                                       edges=None,
                                                       up_edge=True,
                                                       get_children=True,
                                                       diff_nodes=True,
                                                       split_by_edge=False,
                                                       split=split,
                                                       get_order=get_order )

    @staticmethod
    def _children( cmask, pmask, nodes, edges=None, split_by_edge=False, split=False, get_order=False ):
        return GraphMessagePasser._nodeSelectFromEdge( cmask,
                                                       pmask,
                                                       nodes,
                                                       edges=edges,
                                                       up_edge=False,
                                                       get_children=True,
                                                       diff_nodes=False,
                                                       split_by_edge=split_by_edge,
                                                       split=split,
                                                       get_order=get_order )

    @staticmethod
    def _mates( cmask, pmask, nodes, edges=None, split_by_edge=False, split=False, get_order=False ):
        return GraphMessagePasser._nodeSelectFromEdge( cmask,
                                                       pmask,
                                                       nodes,
                                                       edges=edges,
                                                       up_edge=False,
                                                       get_children=False,
                                                       diff_nodes=True,
                                                       split_by_edge=split_by_edge,
                                                       split=split,
                                                       get_order=get_order )

    ######################################################################

    def getParents( self, nodes, split=False, get_order=False ):
        return GraphMessagePasser._parents( self.cmask,
                                            self.pmask,
                                            nodes,
                                            split=split,
                                            get_order=get_order )

    def getSiblings( self, nodes, split=False ):
        return GraphMessagePasser._siblings( self.cmask,
                                             self.pmask,
                                             nodes,
                                             split=split )

    def getChildren( self, nodes, edges=None, split_by_edge=False, split=False ):
        return GraphMessagePasser._children( self.cmask,
                                             self.pmask,
                                             nodes,
                                             edges=edges,
                                             split_by_edge=split_by_edge,
                                             split=split )

    def getMates( self, nodes, edges=None, split_by_edge=False, split=False, get_order=False ):
        return GraphMessagePasser._mates( self.cmask,
                                          self.pmask,
                                          nodes,
                                          edges=edges,
                                          split_by_edge=split_by_edge,
                                          split=split,
                                          get_order=get_order )

    ######################################################################

    def baseCaseNodes( self ):

        M, N = self.pmask.shape

        # Get the number of edges that each node is a parent of
        parent_of_edge_count = self.pmask.getnnz( axis=1 )

        # Get the number of edges that each node is a child of
        child_of_edge_count = self.cmask.getnnz( axis=1 )

        # Get the indices of leaves and roots
        root_indices = self.nodes[ ( parent_of_edge_count != 0 ) & ( child_of_edge_count == 0 ) ]
        leaf_indices = self.nodes[ ( child_of_edge_count != 0 ) & ( parent_of_edge_count == 0 ) ]

        # Generate the up and down base arrays
        u_list = root_indices
        v_list = leaf_indices

        v_list_nodes = []
        v_list_edges = []
        for v in v_list:
            v_list_nodes.append( v )
            v_list_edges.append( None )

        return u_list, [ v_list_nodes, v_list_edges ]

    ######################################################################

    def progressInit( self ):
        u_done = np.zeros( self.pmask.shape[ 0 ], dtype=bool )
        v_done = coo_matrix( ( np.zeros_like( self.pmask.row ), ( self.pmask.row, self.pmask.col ) ), shape=self.pmask.shape, dtype=bool )
        return u_done, v_done

    ######################################################################

    def countSemaphoreInit( self ):
        # Counting semaphores for U and V

        u_sem_data = np.zeros( self.pmask.shape[ 0 ], dtype=int )

        for n in range( u_sem_data.shape[ 0 ] ):
            # U:
            #  - U for all parents
            #  - V for all parents over all down edges except node's up edge
            #  - V for all siblings over all down edges
            up_edges = self.getUpEdges( n )
            parents = self.getParents( n )

            u_sem_data[ n ] += parents.shape[ 0 ]

            for parent in parents:

                down_edges = self.getDownEdges( parent, skip_edges=up_edges )
                u_sem_data[ n ] += down_edges.shape[ 0 ]

            siblings = self.getSiblings( n )
            for sibling in siblings:
                down_edges = self.getDownEdges( sibling )
                u_sem_data[ n ] += down_edges.shape[ 0 ]

        v_sem_data = np.zeros_like( self.pmask.row )

        for i, ( n, e, _ ) in enumerate( zip( self.pmask.row, self.pmask.col, self.pmask.data ) ):
            # V:
            #  - U for all mates from e
            #  - V for all mates over all down edges for mate except for e
            #  - V for all children from e over all down edges for child

            mates = self.getMates( n, edges=e )

            v_sem_data[ i ] += mates.shape[ 0 ]

            for mate in mates:
                down_edges = self.getDownEdges( mate, skip_edges=e )
                v_sem_data[ i ] += down_edges.shape[ 0 ]

            children = self.getChildren( n, edges=e )
            for child in children:
                down_edges = self.getDownEdges( child )
                v_sem_data[ i ] += down_edges.shape[ 0 ]

        u_semaphore = u_sem_data
        v_semaphore = coo_matrix( ( v_sem_data, ( self.pmask.row, self.pmask.col ) ), shape=self.pmask.shape, dtype=int )

        return u_semaphore, v_semaphore

    ######################################################################

    def readyForU( self, u_semaphore, u_done ):
        return self.nodes[ ( u_semaphore == 0 ) & ~u_done ]

    def readyForV( self, v_semaphore, v_done ):
        mask = ( v_semaphore.data == 0 ) & ~v_done.data
        return v_semaphore.row[ mask ], v_semaphore.col[ mask ]

    ######################################################################

    def UDone( self, nodes, u_semaphore, v_semaphore, u_done ):

        # Decrement u_semaphore for children
        children = self.getChildren( nodes, split=True )
        for node, children_for_nodes in zip( nodes, children ):
            u_semaphore[ children_for_nodes ] -= 1
            assert np.all( u_semaphore[ children_for_nodes ] >= 0 )

        # Decrement v_semaphore for all mates over down edges that node and mate are a part of
        mates_and_edges = self.getMates( nodes, split_by_edge=True, split=True )
        for node, mate_and_edge in zip( nodes, mates_and_edges ):
            for e, m in mate_and_edge:
                v_semaphore.data[ np.in1d( v_semaphore.row, m ) & np.in1d( v_semaphore.col, e ) ] -= 1
                assert np.all( v_semaphore.data[ np.in1d( v_semaphore.row, m ) & np.in1d( v_semaphore.col, e ) ] >= 0 )

        u_done[ nodes ] = True

    def VDone( self, nodes_and_edges, u_semaphore, v_semaphore, v_done ):

        nodes, edges = nodes_and_edges
        edges_without_none = np.array( [ e for e in edges if e is not None ] )

        # Decrement u_semaphore for children that come from a different edge than the one computed for V
        children_and_edges = self.getChildren( nodes, split_by_edge=True, split=True )
        for node, edge, child_and_edge in zip( nodes, edges, children_and_edges ):
            for e, c in child_and_edge:
                if( e == edge ):
                    continue
                u_semaphore[ c ] -= 1
                assert np.all( u_semaphore[ c ] >= 0 )

        # Decrement u_semaphore for all siblings
        siblings = self.getSiblings( nodes, split=True )
        for _e, node, siblings_for_node in zip( edges, nodes, siblings ):
            if( _e is None ):
                # If this node doesn't have a down edge, then we don't want to decrement
                continue
            u_semaphore[ siblings_for_node ] -= 1
            assert np.all( u_semaphore[ siblings_for_node ] >= 0 )

        # Decrement v_semaphore for mates that aren't current edge
        mates_and_edges = self.getMates( nodes, split_by_edge=True, split=True )
        for node, edge, mate_and_edge in zip( nodes, edges, mates_and_edges ):
            for e, m in mate_and_edge:
                if( e == edge ):
                    continue
                v_semaphore.data[ np.in1d( v_semaphore.row, m ) & np.in1d( v_semaphore.col, e ) ] -= 1
                assert np.all( v_semaphore.data[ np.in1d( v_semaphore.row, m ) & np.in1d( v_semaphore.col, e ) ] >= 0 )

        # Decrement v_semaphore for parents over up edges
        parents = self.getParents( nodes, split=True )
        up_edges = self.getUpEdges( nodes, split=True )
        for _e, p, e in zip( edges, parents, up_edges ):
            if( _e is None ):
                # If this node doesn't have a down edge, then we don't want to decrement
                continue
            v_semaphore.data[ np.in1d( v_semaphore.row, p ) & np.in1d( v_semaphore.col, e ) ] -= 1
            assert np.all( v_semaphore.data[ np.in1d( v_semaphore.row, p ) & np.in1d( v_semaphore.col, e ) ] >= 0 )

        v_done.data[ np.in1d( v_done.row, nodes ) & np.in1d( v_done.col, edges_without_none ) ] = True

    ######################################################################

    def uReady( self, nodes, u_semaphore ):
        return nodes[ u_semaphore[ nodes ] == 0 ], nodes[ u_semaphore[ nodes ] != 0 ]

    def vReady( self, nodes, v_semaphore ):
        ready = np.intersect1d( nodes, np.setdiff1d( v_semaphore.row, v_semaphore.nonzero()[ 0 ] ) )
        not_ready = np.setdiff1d( nodes, ready )
        return ready, not_ready

    ######################################################################

    def messagePassing( self, uWork, vWork, **kwargs ):

        u_done, v_done = self.progressInit()
        u_semaphore, v_semaphore = self.countSemaphoreInit()
        u_list, v_list = self.baseCaseNodes()

        # Do work for base case nodes
        uWork( True, u_list, **kwargs )
        vWork( True, v_list, **kwargs )

        i = 1

        # Filter over all of the graphs
        while( u_list.size > 0 or v_list[ 0 ].size > 0 ):

            if( i > 1 ):
              # Do work for each of the nodes
              uWork( False, u_list, **kwargs )
              vWork( False, v_list, **kwargs )

            # Mark that we're done with the current nodes
            self.UDone( u_list, u_semaphore, v_semaphore, u_done )
            self.VDone( v_list, u_semaphore, v_semaphore, v_done )

            # Find the next nodes that are ready
            u_list = self.readyForU( u_semaphore, u_done )
            v_list = self.readyForV( v_semaphore, v_done )

            i += 1

            # # Check if we need to do loopy propogation belief
            # if( ( u_list.size == 0 and v_list[ 0 ].size == 0 ) and \
            #     ( not np.any( u_done ) or not np.any( v_done.data ) ) ):
            #     loopy = True
        assert np.any( u_semaphore != 0 ) == False
        assert np.any( v_semaphore.data != 0 ) == False

    ######################################################################

    def getFullParents( self, nodes, split=False, get_order=False ):
        return self.getParents( nodes, split=split, get_order=get_order )

    def getFullSiblings( self, nodes, split=False ):
        return self.getSiblings( nodes, split=split )

    def getFullChildren( self, nodes, edges=None, split_by_edge=False, split=False ):
        return self.getChildren( nodes, edges=edges, split_by_edge=split_by_edge, split=split )

    def getFullMates( self, nodes, edges=None, split_by_edge=False, split=False, get_order=False ):
        return self.getMates( nodes, edges=edges, split_by_edge=split_by_edge, split=split, get_order=get_order )

##########################################################################################################
##########################################################################################################

class __fbsMessagePassingMixin():

    def toGraph( self, use_partial=False ):
        if( use_partial ):
            return Graph.fromParentChildMask( self.pmask, self.cmask )
        return Graph.fromParentChildMask( self.full_pmask, self.full_cmask )

    def draw( self, use_partial=False, render=True ):
        if( use_partial ):
            return self.toGraph( use_partial=True ).draw( render=render )
        return self.toGraph().draw( render=render, cut_nodes=self.fbs )

    def fbsConcat( self, feedback_sets, node_counts ):
        assert len( feedback_sets ) == len( node_counts )
        big_fbs = []
        total_n = 0
        for fbs, N in zip( feedback_sets, node_counts ):
            if( fbs is not None ):
                big_fbs.append( fbs + total_n )
            else:
                big_fbs.append( np.array( [] ) )
            total_n += N
        if( len( big_fbs ) == 0 ):
            return np.array( [] ), np.array( [] )
        return np.concatenate( big_fbs ), big_fbs

    def updateParamsFromGraphs( self, graphs ):

        parent_masks = []
        child_masks = []
        feedback_sets = []
        for graph in graphs:
            if( isinstance( graph, Iterable ) ):
                assert len( graph ) == 2
                graph, fbs = graph
            else:
                fbs = None

            p_mask, c_mask = graph.toMatrix()
            parent_masks.append( p_mask )
            child_masks.append( c_mask )
            feedback_sets.append( fbs )

        self.updateParams( parent_masks, child_masks, feedback_sets=feedback_sets )

    def updateParams( self, parent_masks, child_masks, feedback_sets=None ):

        # Save off the full pmask, cmask and node set and only use the graph without
        # the fbs nodes for message passing.

        if( feedback_sets is not None ):
            assert len( parent_masks ) == len( child_masks ) == len( feedback_sets )
            for child_mask, parent_mask, feedbackSet in zip( child_masks, parent_masks, feedback_sets ):
                assert isinstance( child_mask, coo_matrix )
                assert isinstance( parent_mask, coo_matrix )
                assert child_mask.shape == parent_mask.shape
        else:
            assert len( parent_masks ) == len( child_masks )
            for child_mask, parent_mask in zip( child_masks, parent_masks ):
                assert isinstance( child_mask, coo_matrix )
                assert isinstance( parent_mask, coo_matrix )
                assert child_mask.shape == parent_mask.shape

        self.full_pmask, self.parent_graph_assignments = self.concatSparseMatrix( parent_masks )
        self.full_cmask, self.child_graph_assignments = self.concatSparseMatrix( child_masks )

        self.full_nodes = np.arange( self.full_pmask.shape[ 0 ] )

        if( feedback_sets is not None ):
            node_counts = [ mat.shape[ 0 ] for mat in parent_masks ]
            # self.feedback_sets contains all of the feedback sets with the adjusted node indices
            fbs_nodes, self.feedback_sets = self.fbsConcat( feedback_sets, node_counts )
            self.fbs_mask = np.in1d( self.full_nodes, fbs_nodes )
        else:
            self.fbs_mask = np.zeros( self.full_pmask.shape[ 0 ], dtype=bool )

        # All of the feedback sets together
        self.fbs = self.full_nodes[ self.fbs_mask ]

        # Create a mapping from the full indices to the new indices
        non_fbs = self.full_nodes[ ~self.fbs_mask ]
        non_fbs_reindexed = ( self.full_nodes - self.fbs_mask.cumsum() )[ ~self.fbs_mask ]
        index_map = dict( zip( non_fbs, non_fbs_reindexed ) )
        index_map_reversed = dict( zip( non_fbs_reindexed, non_fbs ) )

        # Re-index the fbs nodes starting from non_fbs.size
        fbs_index_map = dict( zip( self.fbs, np.arange( self.fbs.shape[ 0 ] ) + non_fbs.shape[ 0 ] ) )
        fns_index_map_reversed = dict( zip( np.arange( self.fbs.shape[ 0 ] ) + non_fbs.shape[ 0 ], self.fbs ) )
        index_map.update( fbs_index_map )
        index_map_reversed.update( fns_index_map_reversed )

        self.fullGraphIndexToPartialGraphIndex = np.vectorize( lambda x: index_map[ x ] )
        self.partialGraphIndexToFullGraphIndex = np.vectorize( lambda x: index_map_reversed[ x ] )

        # Create the new list of nodes
        self.nodes = self.fullGraphIndexToPartialGraphIndex( non_fbs )

        # TODO: MAKE REDUCED INDICES A DIFFERENT TYPE FROM FULL INDICES!!!!!

        # Get a mask over where the fbs nodes arent and
        # make the new sparse matrix parameters
        mask = ~np.in1d( self.full_pmask.row, self.fbs )
        _pmask_row, _pmask_col, _pmask_data = self.full_pmask.row[ mask ], self.full_pmask.col[ mask ], self.full_pmask.data[ mask ]
        _pmask_row = self.fullGraphIndexToPartialGraphIndex( _pmask_row )

        mask = ~np.in1d( self.full_cmask.row, self.fbs )
        _cmask_row, _cmask_col, _cmask_data = self.full_cmask.row[ mask ], self.full_cmask.col[ mask ], self.full_cmask.data[ mask ]
        _cmask_row = self.fullGraphIndexToPartialGraphIndex( _cmask_row )

        # The new shape will have fewer nodes
        shape = ( self.full_pmask.shape[ 0 ] - self.fbs.shape[ 0 ], self.full_pmask.shape[ 1 ] )
        self.pmask = coo_matrix( ( _pmask_data, ( _pmask_row, _pmask_col ) ), shape=shape, dtype=int )

        shape = ( self.full_cmask.shape[ 0 ] - self.fbs.shape[ 0 ], self.full_cmask.shape[ 1 ] )
        self.cmask = coo_matrix( ( _cmask_data, ( _cmask_row, _cmask_col ) ), shape=shape, dtype=int )

    ######################################################################

    def inFeedbackSet( self, node, from_reduced=True ):
        if( from_reduced ):
            return self.partialGraphIndexToFullGraphIndex( node ) in self.fbs
        return node in self.fbs

    def fbsIndex( self, node, from_reduced=True, within_graph=True ):
        full_node = self.partialGraphIndexToFullGraphIndex( node ) if from_reduced else node

        if( within_graph == False ):
            return self.fbs.tolist().index( node )

        for fbs in self.feedback_sets:
            if( full_node in fbs ):
                return fbs.tolist().index( full_node )

        assert 0, 'This is not a fbs node'

    ######################################################################

    def splitNodesFromFBS( self, nodes ):
        if( not isinstance( nodes, Iterable ) ):
            nodes = [ nodes ]
        if( isinstance( nodes, np.ndarray ) and nodes.ndim == 0 ):
            nodes = nodes[ None ]
        fbs_nodes = [ n for n in nodes if self.inFeedbackSet( n, from_reduced=True ) ]
        non_fbs_nodes = [ n for n in nodes if not self.inFeedbackSet( n, from_reduced=True ) ]
        return fbs_nodes, non_fbs_nodes

    def removeFBSFromNodes( self, nodes ):
        if( not isinstance( nodes, Iterable ) ):
            nodes = [ nodes ]
        return np.array( [ n for n in nodes if not self.inFeedbackSet( n, from_reduced=True ) ] )

    def removeFBSFromNodesAndOrder( self, nodes, order ):
        if( not isinstance( nodes, Iterable ) ):
            nodes = [ nodes ]
        if( nodes.size == 0 ):
            return nodes, order
        nodesOrder = zip( *[ ( n, o ) for n, o in zip( nodes, order ) if not self.inFeedbackSet( n, from_reduced=True ) ] )
        if( len( list( nodesOrder ) ) > 0 ):
            # Iterator is consumed in check above
            _nodes, _order = zip( *[ ( n, o ) for n, o in zip( nodes, order ) if not self.inFeedbackSet( n, from_reduced=True ) ] )
            return np.array( _nodes ), np.array( _order )
        else:
            return np.array( [] ), np.array( [] )

    def removeFBSFromSplitNodes( self, nodes ):
        if( not isinstance( nodes, Iterable ) ):
            nodes = [ nodes ]
        nodes_fbs = []
        for n in nodes:
            current_fbs = []
            for _n in n:
                if( not self.inFeedbackSet( _n ) ):
                    current_fbs.append( _n )
            nodes_fbs.append( current_fbs )
        return nodes_fbs

    def removeFBSFromSplitNodesAndOrder( self, nodes, order ):
        if( not isinstance( nodes, Iterable ) ):
            nodes = [ nodes ]
        nodes_fbs = []
        nodes_order_fbs = []
        for n, o in zip( nodes, order ):
            current_fbs = []
            current_order = []
            for _n, _o in zip( n, o ):
                if( not self.inFeedbackSet( _n ) ):
                    current_fbs.append( _n )
                    current_order.append( _o )
            nodes_fbs.append( current_fbs )
            nodes_order_fbs.append( current_order )
        return np.array( nodes_fbs ), np.array( nodes_order_fbs )

    def removeFBSFromSplitEdges( self, nodes ):
        assert 0, 'Implement this'

    def removeFBSFromSplitNodesAndEdges( self, nodes ):
        assert 0, 'Implement this'

    def removeFBSFromSplitEdgesAndOrder( self, nodes, order ):
        assert 0, 'Implement this'

    def removeFBSFromSplitNodesAndEdgesAndOrder( self, nodes, order ):
        assert 0, 'Implement this'

    ######################################################################

    def getParents( self, nodes, split=False, get_order=False ):

        fbs_nodes, non_fbs_nodes = self.splitNodesFromFBS( nodes )

        non_fbs_ans = GraphMessagePasser._parents( self.cmask,
                                                   self.pmask,
                                                   non_fbs_nodes,
                                                   split=split,
                                                   get_order=get_order )
        if( len( fbs_nodes ) == 0 ):
            return non_fbs_ans

        if( get_order == True ):
            full_parents, full_parent_prder = self.getFullParents( fbs_nodes, split=split, get_order=get_order )
            if( split == False ):
                parents_fbs, parent_order_fbs = self.removeFBSFromNodesAndOrder( full_parents, full_parent_prder )
            else:
                parent_fbs, parent_order_fbs = self.removeFBSFromSplitNodesAndOrder( full_parents, full_parent_prder )

            parents, parent_order = non_fbs_ans
            assert type( parents ) == type( parents_fbs ), '%s, %s'%( type( parents ), type( parents_fbs ) )
            assert type( parent_order ) == type( parent_order_fbs )
            return np.hstack( ( parents_fbs, parents ) ), np.hstack( ( parent_order_fbs, parent_order ) )
        else:
            full_parents = self.getFullParents( fbs_nodes, split=split, get_order=get_order )
            if( split == False ):
                parents_fbs = self.removeFBSFromNodes( full_parents )
            else:
                parent_fbs = self.removeFBSFromSplitNodes( full_parents )

            parents = non_fbs_ans
            assert type( parents ) == type( parents_fbs )
            return np.hstack( ( parents_fbs, parents ) )

    def getSiblings( self, nodes, split=False ):

        fbs_nodes, non_fbs_nodes = self.splitNodesFromFBS( nodes )

        non_fbs_ans = GraphMessagePasser._siblings( self.cmask,
                                                  self.pmask,
                                                  non_fbs_nodes,
                                                  split=split )
        if( len( fbs_nodes ) == 0 ):
            return non_fbs_ans

        full_siblings = self.getFullSiblings( fbs_nodes, split=split )
        if( split == False ):
            siblings_fbs = self.removeFBSFromNodes( full_siblings )
        else:
            siblings_fbs = self.removeFBSFromSplitNodes( full_siblings )

        assert type( siblings ) == type( siblings_fbs )
        return np.hstack( ( siblings_fbs, siblings ) )

    def getChildren( self, nodes, edges=None, split_by_edge=False, split=False ):

        fbs_nodes, non_fbs_nodes = self.splitNodesFromFBS( nodes )

        non_fbs_ans = GraphMessagePasser._children( self.cmask,
                                                  self.pmask,
                                                  non_fbs_nodes,
                                                  edges=edges,
                                                  split_by_edge=split_by_edge,
                                                  split=split )
        if( len( fbs_nodes ) == 0 ):
            return non_fbs_ans

        full_children = self.getFullChildren( fbs_nodes, edges=edges, split_by_edge=split_by_edge, split=split )

        if( split_by_edge == False ):
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

    def getMates( self, nodes, edges=None, split_by_edge=False, split=False, get_order=False ):

        fbs_nodes, non_fbs_nodes = self.splitNodesFromFBS( nodes )

        non_fbs_ans = GraphMessagePasser._mates( self.cmask,
                                               self.pmask,
                                               non_fbs_nodes,
                                               edges=edges,
                                               split_by_edge=split_by_edge,
                                               split=split,
                                               get_order=get_order )
        if( len( fbs_nodes ) == 0 ):
            return non_fbs_ans

        if( split_by_edge == False ):
            if( get_order == True ):
                full_mates, full_mate_order = self.getFullMates( fbs_nodes, edges=edges, split_by_edge=split_by_edge, split=split, get_order=get_order )
                if( split == False ):
                    mates_fbs, mate_order_fbs = self.removeFBSFromNodesAndOrder( full_mates, full_mate_order )
                else:
                    mates_fbs, mate_order_fbs = self.removeFBSFromSplitNodesAndOrder( full_mates, full_mate_order )

                mates, mate_order = non_fbs_ans
                assert type( mates ) == type( mates_fbs )
                assert type( mate_order ) == type( mate_order_fbs )
                return np.hstack( ( mates_fbs, mates ) ), np.hstack( ( mate_order_fbs, mate_order ) )
            else:
                full_mates = self.getFullMates( fbs_nodes, edges=edges, split_by_edge=split_by_edge, split=split, get_order=get_order )
                if( split == False ):
                    mates_fbs = self.removeFBSFromNodes( full_mates )
                else:
                    mates_fbs = self.removeFBSFromSplitNodes( full_mates )

                assert type( mates ) == type( mates_fbs )
                return np.hstack( ( mates_fbs, mates ) )

        else:
            if( get_order == True ):
                if( split == False ):
                    mates_fbs, mate_order_fbs = self.removeFBSFromSplitEdgesAndOrder( full_mates )
                else:
                    mates_fbs, mate_order_fbs = self.removeFBSFromSplitNodesAndEdgesAndOrder( full_mates )
            else:
                if( split == False ):
                    mates_fbs = self.removeFBSFromSplitEdges( full_mates )
                else:
                    mates_fbs = self.removeFBSFromSplitNodesAndEdges( full_mates )
            assert 0

    ######################################################################

    def getFullParents( self, nodes, split=False, get_order=False, full_indexing=False, return_full_indexing=False ):
        if( full_indexing == False ):
            if( isinstance( nodes, np.ndarray ) and nodes.ndim == 0 ):
                nodes = self.partialGraphIndexToFullGraphIndex( nodes )
            else:
                nodes = [ self.partialGraphIndexToFullGraphIndex( n ) for n in nodes ] if isinstance( nodes, Iterable ) else self.partialGraphIndexToFullGraphIndex( nodes )

        ans = GraphMessagePasser._parents( self.full_cmask,
                                           self.full_pmask,
                                           nodes,
                                           split=split,
                                           get_order=get_order )
        if( get_order ):
            ans, order = ans
            if( return_full_indexing == False ):
                ans = [ self.fullGraphIndexToPartialGraphIndex( n ) for n in ans ]
            if( not ( split ) ):
                ans = np.array( ans )
            ans = ans, order
        else:
            if( return_full_indexing == False ):
                ans = [ self.fullGraphIndexToPartialGraphIndex( n ) for n in ans ]

        if( split ):
            return ans
        return np.array( ans )

    def getFullSiblings( self, nodes, split=False, full_indexing=False, return_full_indexing=False ):
        if( full_indexing == False ):
            if( isinstance( nodes, np.ndarray ) and nodes.ndim == 0 ):
                nodes = self.partialGraphIndexToFullGraphIndex( nodes )
            else:
                nodes = [ self.partialGraphIndexToFullGraphIndex( n ) for n in nodes ] if isinstance( nodes, Iterable ) else self.partialGraphIndexToFullGraphIndex( nodes )

        ans = GraphMessagePasser._siblings( self.full_cmask,
                                            self.full_pmask,
                                            nodes,
                                            split=split )

        if( return_full_indexing == False ):
            ans = [ self.fullGraphIndexToPartialGraphIndex( n ) for n in ans ]

        if( split ):
            return ans
        return np.array( ans )

    def getFullChildren( self, nodes, edges=None, split_by_edge=False, split=False, full_indexing=False, return_full_indexing=False ):
        if( full_indexing == False ):
            if( isinstance( nodes, np.ndarray ) and nodes.ndim == 0 ):
                nodes = self.partialGraphIndexToFullGraphIndex( nodes )
            else:
                nodes = [ self.partialGraphIndexToFullGraphIndex( n ) for n in nodes ] if isinstance( nodes, Iterable ) else self.partialGraphIndexToFullGraphIndex( nodes )

        ans = GraphMessagePasser._children( self.full_cmask,
                                            self.full_pmask,
                                            nodes,
                                            edges=edges,
                                            split_by_edge=split_by_edge,
                                            split=split )

        if( return_full_indexing == False ):
            ans = [ self.fullGraphIndexToPartialGraphIndex( n ) for n in ans ]

        if( split or split_by_edge ):
            return ans
        return np.array( ans )

    def getFullMates( self, nodes, edges=None, split_by_edge=False, split=False, get_order=False, full_indexing=False, return_full_indexing=False ):
        if( full_indexing == False ):
            if( isinstance( nodes, np.ndarray ) and nodes.ndim == 0 ):
                nodes = self.partialGraphIndexToFullGraphIndex( nodes )
            else:
                nodes = [ self.partialGraphIndexToFullGraphIndex( n ) for n in nodes ] if isinstance( nodes, Iterable ) else self.partialGraphIndexToFullGraphIndex( nodes )

        ans = GraphMessagePasser._mates( self.full_cmask,
                                         self.full_pmask,
                                         nodes,
                                         edges=edges,
                                         split_by_edge=split_by_edge,
                                         split=split,
                                         get_order=get_order )
        if( get_order ):
            ans, order = ans
            if( return_full_indexing == False ):
                ans = [ self.fullGraphIndexToPartialGraphIndex( n ) for n in ans ]
            if( not ( split or split_by_edge ) ):
                ans = np.array( ans )
            ans = ans, order
        else:
            if( return_full_indexing == False ):
                ans = [ self.fullGraphIndexToPartialGraphIndex( n ) for n in ans ]

        if( split or split_by_edge ):
            return ans
        return np.array( ans )

    ######################################################################

    def getUpEdges( self, nodes, split=False, from_full=True ):
        if( from_full == False ):
            return GraphMessagePasser._upEdges( self.cmask, nodes, split=split )

        if( isinstance( nodes, np.ndarray ) and nodes.ndim == 0 ):
            nodes = self.partialGraphIndexToFullGraphIndex( nodes )
        else:
            nodes = [ self.partialGraphIndexToFullGraphIndex( n ) for n in nodes ] if isinstance( nodes, Iterable ) else self.partialGraphIndexToFullGraphIndex( nodes )

        return GraphMessagePasser._upEdges( self.full_cmask, nodes, split=split )

    def getDownEdges( self, nodes, skip_edges=None, split=False, from_full=True ):
        if( from_full == False ):
            return GraphMessagePasser._downEdges( self.pmask, nodes, skip_edges=skip_edges, split=split )

        if( isinstance( nodes, np.ndarray ) and nodes.ndim == 0 ):
            nodes = self.partialGraphIndexToFullGraphIndex( nodes )
        else:
            nodes = [ self.partialGraphIndexToFullGraphIndex( n ) for n in nodes ] if isinstance( nodes, Iterable ) else self.partialGraphIndexToFullGraphIndex( nodes )

        return GraphMessagePasser._downEdges( self.full_pmask, nodes, skip_edges=skip_edges, split=split )

##########################################################################################################

class __fbsMessagePassingMixinImproved():

    def toGraph( self, use_partial=False ):
        if( use_partial ):
            return Graph.fromParentChildMask( self.partial_graph.pmask, self.partial_graph.cmask )
        return Graph.fromParentChildMask( self.full_graph.pmask, self.full_graph.cmask )

    def draw( self, use_partial=False, render=True ):
        if( use_partial ):
            return self.toGraph( use_partial=True ).draw( render=render )
        return self.toGraph().draw( render=render, cut_nodes=self.fbs )

    def fbsConcat( self, feedback_sets, node_counts ):
        assert len( feedback_sets ) == len( node_counts )
        big_fbs = []
        total_n = 0
        for fbs, N in zip( feedback_sets, node_counts ):
            if( fbs is not None ):
                big_fbs.append( fbs + total_n )
            else:
                big_fbs.append( np.array( [] ) )
            total_n += N
        if( len( big_fbs ) == 0 ):
            return np.array( [] ), np.array( [] )
        return np.concatenate( big_fbs ), big_fbs

    def updateParamsFromGraphs( self, graphs ):

        parent_masks = []
        child_masks = []
        feedback_sets = []
        for graph in graphs:
            if( isinstance( graph, Iterable ) ):
                assert len( graph ) == 2
                graph, fbs = graph
            else:
                fbs = None

            p_mask, c_mask = graph.toMatrix()
            parent_masks.append( p_mask )
            child_masks.append( c_mask )
            feedback_sets.append( fbs )

        self.updateParams( parent_masks, child_masks, feedback_sets=feedback_sets )

    def updateParams( self, parent_masks, child_masks, feedback_sets=None ):

        # The main idea with this class is to have 2 different graphs.  The full
        # graph will cycles, and a graph without the feedback nodes which will
        # be acyclic.  Message passing will be done over the partial graph while
        # using computations with the full graph.  At the end of message passing,
        # the fbs nodes will be integrated out as needed.

        # This is the full directed graph.  It might have directed cycles
        self.full_graph = GraphMessagePasser()
        self.full_graph.updateParams( parent_masks, child_masks )

        if( feedback_sets is not None ):
            assert len( parent_masks ) == len( child_masks ) == len( feedback_sets )
            for child_mask, parent_mask, feedbackSet in zip( child_masks, parent_masks, feedback_sets ):
                assert isinstance( child_mask, coo_matrix )
                assert isinstance( parent_mask, coo_matrix )
                assert child_mask.shape == parent_mask.shape
        else:
            assert len( parent_masks ) == len( child_masks )
            for child_mask, parent_mask in zip( child_masks, parent_masks ):
                assert isinstance( child_mask, coo_matrix )
                assert isinstance( parent_mask, coo_matrix )
                assert child_mask.shape == parent_mask.shape

        # Extract the a mask over the fbs nodes
        if( feedback_sets is not None ):
            node_counts = [ mat.shape[ 0 ] for mat in parent_masks ]
            # Keep track of the feedback sets for each individual graph
            fbs_nodes, self.feedback_sets = self.fbsConcat( feedback_sets, node_counts )
            self.fbs_mask = np.in1d( self.full_graph.nodes, fbs_nodes )
        else:
            self.fbs_mask = np.zeros( self.full_graph.pmask.shape[ 0 ], dtype=bool )

        # Get the indices in the full graph of the feedback set
        self.fbs = self.full_graph.nodes[ self.fbs_mask ]

        # Create a mapping from the full graph indices to the partial graph indices
        non_fbs = self.full_graph.nodes[ ~self.fbs_mask ]
        non_fbs_reindexed = ( self.full_graph.nodes - self.fbs_mask.cumsum() )[ ~self.fbs_mask ]
        index_map = dict( zip( non_fbs, non_fbs_reindexed ) )
        index_map_reversed = dict( zip( non_fbs_reindexed, non_fbs ) )

        # Re-index the fbs nodes starting from non_fbs.size
        fbs_index_map = dict( zip( self.fbs, np.arange( self.fbs.shape[ 0 ] ) + non_fbs.shape[ 0 ] ) )
        fns_index_map_reversed = dict( zip( np.arange( self.fbs.shape[ 0 ] ) + non_fbs.shape[ 0 ], self.fbs ) )
        index_map.update( fbs_index_map )
        index_map_reversed.update( fns_index_map_reversed )

        # Create a functions to map:
        # full_graph -> partial_graph
        # partial_graph -> full_graph
        self.fullGraphIndexToPartialGraphIndex = np.vectorize( lambda x: index_map[ x ] )
        self.partialGraphIndexToFullGraphIndex = np.vectorize( lambda x: index_map_reversed[ x ] )

        # Create the partial graph
        mask = ~np.in1d( self.full_graph.pmask.row, self.fbs )
        _pmask_row, _pmask_col, _pmask_data = self.full_graph.pmask.row[ mask ], self.full_graph.pmask.col[ mask ], self.full_graph.pmask.data[ mask ]
        _pmask_row = self.fullGraphIndexToPartialGraphIndex( _pmask_row )

        mask = ~np.in1d( self.full_graph.cmask.row, self.fbs )
        _cmask_row, _cmask_col, _cmask_data = self.full_graph.cmask.row[ mask ], self.full_graph.cmask.col[ mask ], self.full_graph.cmask.data[ mask ]
        _cmask_row = self.fullGraphIndexToPartialGraphIndex( _cmask_row )

        # The new shape will have fewer nodes
        shape = ( self.full_graph.pmask.shape[ 0 ] - self.fbs.shape[ 0 ], self.full_graph.pmask.shape[ 1 ] )
        parital_pmask = coo_matrix( ( _pmask_data, ( _pmask_row, _pmask_col ) ), shape=shape, dtype=int )

        shape = ( self.full_graph.cmask.shape[ 0 ] - self.fbs.shape[ 0 ], self.full_graph.cmask.shape[ 1 ] )
        parital_cmask = coo_matrix( ( _cmask_data, ( _cmask_row, _cmask_col ) ), shape=shape, dtype=int )

        # This is the full graph without the feedback set nodes.  It will
        # be a directed acyclic graph
        self.partial_graph = GraphMessagePasser()
        self.partial_graph.updateParams( [ parital_pmask ], [ parital_cmask ] )

    ######################################################################

    def inFeedbackSet( self, node, is_partial_graph_index ):
        full_node = self.partialGraphIndexToFullGraphIndex( node ) if is_partial_graph_index else node
        return full_node in self.fbs

    def fbsIndex( self, node, is_partial_graph_index, within_graph=True ):
        full_node = self.partialGraphIndexToFullGraphIndex( node ) if is_partial_graph_index else node

        # Get the index either over the entire fbs or within its specific graph
        if( within_graph == False ):
            return self.fbs.tolist().index( node )

        for fbs in self.feedback_sets:
            if( full_node in fbs ):
                return fbs.tolist().index( full_node )

        assert 0, 'This is not a fbs node'

    ######################################################################

    @property
    def nodes( self ):
        return self.full_graph.nodes

    ######################################################################

    def getParents( self, nodes, split=False, get_order=False ):
        return self.full_graph.getParents( nodes, split=split, get_order=get_order )

    def getSiblings( self, nodes, split=False ):
        return self.full_graph.getSiblings( nodes, split=split, get_order=get_order )

    def getChildren( self, nodes, edges=None ):
        return self.full_graph.getChildren( nodes, edges=edges )

    def getMates( self, nodes, edges=None, split_by_edge=False, split=False, get_order=False ):
        return self.full_graph.getMates( nodes, edges=edges, split_by_edge=split_by_edge, split=split, get_order=get_order )

    ######################################################################

    def convertIndices( self, nodes, partial_to_full=True ):
        converter = self.partialGraphIndexToFullGraphIndex if partial_to_full else self.fullGraphIndexToPartialGraphIndex

        # Handle different type possibilities for nodes
        if( isinstance( nodes, np.ndarray ) and nodes.ndim == 0 ):
            nodes = converter( nodes )
        else:
            nodes = [ converter( n ) for n in nodes ] if isinstance( nodes, Iterable ) else converter( nodes )
        return nodes

    def convertResultIndices( self, nodes_maybe_order, get_order=False, split=False, split_by_edge=False, partial_to_full=True, filterFunc=None ):
        if( partial_to_full is None ):
            converter = lambda n: n
        else:
            converter = self.partialGraphIndexToFullGraphIndex if partial_to_full else self.fullGraphIndexToPartialGraphIndex
        filterFunc = filterFunc if filterFunc is not None else lambda x: True

        # Handle case where ans is split by node and/or edge, or order is included
        if( get_order ):
            if( split ):
                if( split_by_edge ):
                    nodes, order = [], []
                    for nodes_in_group in nodes_maybe_order:
                        _nodes, _order = [], []
                        for nodes_in_edge in nodes_in_group:
                            if( len( nodes_in_edge ) > 0 ):
                                __nodes__order = [ ( converter( n ), o ) for n, o in zip( *nodes_in_edge ) if filterFunc( n ) ]
                                __nodes, __order = zip( *__nodes__order ) if len( __nodes__order ) > 0 else ( [], [] )
                            else:
                                __nodes, __order = [], []
                            _nodes.append( np.array( __nodes ) )
                            _order.append( np.array( __order ) )
                        nodes.append( _nodes )
                        order.append( _order )
                    return nodes, order
                else:
                    nodes, order = [], []
                    for nodes_in_group in nodes_maybe_order:
                        if( len( nodes_in_group ) > 0 ):
                            _nodes_order = [ ( converter( n ), o ) for n, o in zip( *nodes_in_group ) if filterFunc( n ) ]
                            _nodes, _order = zip( *_nodes_order ) if len( _nodes_order ) > 0 else ( [], [] )
                        else:
                            _nodes, _order = [], []
                        nodes.append( np.array( _nodes ) )
                        order.append( np.array( _order ) )
                    return nodes, order
            else:
                if( split_by_edge ):
                    nodes, order = [], []
                    for nodes_in_edge in nodes_maybe_order:
                        if( len( nodes_in_edge ) > 0 ):
                            _nodes_order = [ ( converter( n ), o ) for n, o in zip( *nodes_in_edge ) if filterFunc( n ) ]
                            _nodes, _order = zip( *_nodes_order ) if len( _nodes_order ) > 0 else ( [], [] )
                        else:
                            _nodes, _order = [], []
                        nodes.append( np.array( _nodes ) )
                        order.append( np.array( _order ) )
                    return nodes, order
                else:
                    if( len( nodes_maybe_order ) > 0 ):
                        nodesorder = [ ( converter( n ), o ) for n, o in zip( *nodes_maybe_order ) if filterFunc( n ) ]
                        nodes, order = zip( *nodesorder ) if len( nodesorder ) > 0 else ( [], [] )
                    else:
                        nodes, order = [], []
                    return np.array( nodes ), np.array( order )
        else:
            if( split ):
                if( split_by_edge ):
                    nodes = []
                    for nodes_in_group in nodes_maybe_order:
                        _nodes = []
                        for nodes_in_edge in nodes_in_group:
                            _nodes.append( np.array( [ converter( n ) for n in nodes_in_edge if filterFunc( n ) ] ) )
                        nodes.append( _nodes )
                    return nodes
                else:
                    nodes = []
                    for nodes_in_group in nodes_maybe_order:
                        nodes.append( np.array( [ converter( n ) for n in nodes_in_group if filterFunc( n ) ] ) )
                    return nodes
            else:
                if( split_by_edge ):
                    nodes = []
                    for nodes_in_edge in nodes_maybe_order:
                        nodes.append( np.array( [ converter( n ) for n in nodes_in_edge if filterFunc( n ) ] ) )
                    return nodes
                else:
                    return np.array( [ converter( n ) for n in nodes_maybe_order if filterFunc( n ) ] )

    def shouldConvertPartialToFull( self, input_is_parital_index, output_partial_index ):
        if( input_is_parital_index == True ):
            if( output_partial_index == True ):
                # partial -> partial
                partial_to_full = None
            else:
                # full -> partial
                partial_to_full = True
        else:
            if( output_partial_index == True ):
                # parital -> full
                partial_to_full = False
            else:
                # full -> full
                partial_to_full = None
        return partial_to_full

    ######################################################################

    def fetchGraphNodes( self, nodes, fetchFunc, filterFunc=None, get_order=False, split=False, split_by_edge=False, is_partial_graph_index=False, return_partial_graph_index=False, use_partial_graph=False ):

        # Ensure that the nodes are using the correct indices before passing to fetchFunc
        partial_to_full = self.shouldConvertPartialToFull( is_partial_graph_index, use_partial_graph )
        if( partial_to_full is not None ):
            nodes = self.convertIndices( nodes, partial_to_full=partial_to_full )

        # Get the result nodes
        fetched = fetchFunc( nodes )

        # Return the nodes using the correct indices
        partial_to_full = self.shouldConvertPartialToFull( use_partial_graph, return_partial_graph_index )
        if( partial_to_full is not None or filterFunc is not None ):
            fetched = self.convertResultIndices( fetched, get_order=get_order, split=split, split_by_edge=split_by_edge, partial_to_full=partial_to_full, filterFunc=filterFunc )

        return fetched

    ######################################################################

    def getFullParents( self,
                        nodes,
                        split=False,
                        get_order=False,
                        is_partial_graph_index=False,
                        return_partial_graph_index=False ):
        fetchFunc = partial( self.full_graph.getParents, split=split, get_order=get_order )
        return self.fetchGraphNodes( nodes, fetchFunc, get_order=get_order, split=split, split_by_edge=False, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=return_partial_graph_index, use_partial_graph=False )

    def getFullSiblings( self,
                         nodes,
                         split=False,
                         is_partial_graph_index=False,
                         return_partial_graph_index=False ):
        fetchFunc = partial( self.full_graph.getSiblings, split=split )
        return self.fetchGraphNodes( nodes, fetchFunc, get_order=False, split=split, split_by_edge=False, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=return_partial_graph_index, use_partial_graph=False )

    def getFullChildren( self,
                         nodes,
                         edges=None,
                         split_by_edge=False,
                         split=False,
                         is_partial_graph_index=False,
                         return_partial_graph_index=False ):
        fetchFunc = partial( self.full_graph.getChildren, edges=edges, split_by_edge=split_by_edge, split=split )
        return self.fetchGraphNodes( nodes, fetchFunc, get_order=False, split=split, split_by_edge=split_by_edge, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=return_partial_graph_index, use_partial_graph=False )

    def getFullMates( self,
                      nodes,
                      edges=None,
                      split_by_edge=False,
                      split=False,
                      get_order=False,
                      is_partial_graph_index=False,
                      return_partial_graph_index=False ):
        fetchFunc = partial( self.full_graph.getMates, edges=edges, split_by_edge=split_by_edge, split=split, get_order=get_order )
        return self.fetchGraphNodes( nodes, fetchFunc, get_order=get_order, split=split, split_by_edge=split_by_edge, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=return_partial_graph_index, use_partial_graph=False )

    ######################################################################

    def _anotherNode( self, node, parents=True, mates=True, siblings=True, children=True, is_partial_graph_index=False, return_partial_graph_index=False ):
        to_use = []
        if( parents ):
            to_use.append( self.getFullParents( node, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=return_partial_graph_index ) )
        if( mates ):
            to_use.append( self.getFullMates( node, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=return_partial_graph_index ) )
        if( siblings ):
            to_use.append( self.getFullSiblings( node, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=return_partial_graph_index ) )
        if( children ):
            to_use.append( self.getFullChildren( node, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=return_partial_graph_index ) )
        possibilities = itertools.chain( *to_use )
        for _node in possibilities:
            if( not self.inFeedbackSet( _node, is_partial_graph_index=return_partial_graph_index ) ):
                return int( _node )
        return None

    ######################################################################

    def splitNodesFromFBS( self, nodes, is_partial_graph_index=True ):
        if( not isinstance( nodes, Iterable ) ):
            nodes = [ nodes ]
        if( isinstance( nodes, np.ndarray ) and nodes.ndim == 0 ):
            nodes = nodes[ None ]
        fbs_nodes = [ n for n in nodes if self.inFeedbackSet( n, is_partial_graph_index=is_partial_graph_index ) ]
        non_fbs_nodes = [ n for n in nodes if not self.inFeedbackSet( n, is_partial_graph_index=is_partial_graph_index ) ]
        return fbs_nodes, non_fbs_nodes

    def getPartialParents( self,
                           nodes,
                           split=False,
                           get_order=False,
                           is_partial_graph_index=False,
                           return_partial_graph_index=False ):

        # If this function is called for a fbs node, it won't exist in the partial graph.
        # So need to get the parents from the full graph and filter out the fbs nodes
        fbs_nodes, non_fbs_nodes = self.splitNodesFromFBS( nodes, is_partial_graph_index=is_partial_graph_index )

        if( len( fbs_nodes ) == 0 ):
            fetchFunc = partial( self.partial_graph.getParents, split=split, get_order=get_order )
            return self.fetchGraphNodes( non_fbs_nodes, fetchFunc, get_order=get_order, split=split, split_by_edge=False, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=return_partial_graph_index, use_partial_graph=True )

        # Otherwise, just fetch nodes from the full graph and filter out the fbs nodes
        fetchFunc = partial( self.full_graph.getParents, split=split, get_order=get_order )
        filterFunc = lambda n: not self.inFeedbackSet( n, is_partial_graph_index=False )
        return self.fetchGraphNodes( nodes, fetchFunc, filterFunc=filterFunc, get_order=get_order, split=split, split_by_edge=False, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=return_partial_graph_index, use_partial_graph=False )

    def getPartialSiblings( self,
                            nodes,
                            split=False,
                            is_partial_graph_index=False,
                            return_partial_graph_index=False ):

        # If this function is called for a fbs node, it won't exist in the partial graph.
        # So need to get the parents from the full graph and filter out the fbs nodes
        fbs_nodes, non_fbs_nodes = self.splitNodesFromFBS( nodes, is_partial_graph_index=is_partial_graph_index )

        if( len( fbs_nodes ) == 0 ):
            fetchFunc = partial( self.partial_graph.getSiblings, split=split )
            return self.fetchGraphNodes( nodes, fetchFunc, get_order=False, split=split, split_by_edge=False, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=return_partial_graph_index, use_partial_graph=True )

        fetchFunc = partial( self.full_graph.getSiblings, split=split )
        filterFunc = lambda n: not self.inFeedbackSet( n, is_partial_graph_index=False )
        return self.fetchGraphNodes( nodes, fetchFunc, get_order=False, split=split, split_by_edge=False, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=return_partial_graph_index, use_partial_graph=False )

    def getPartialChildren( self,
                            nodes,
                            edges=None,
                            split_by_edge=False,
                            split=False,
                            is_partial_graph_index=False,
                            return_partial_graph_index=False ):

        # If this function is called for a fbs node, it won't exist in the partial graph.
        # So need to get the parents from the full graph and filter out the fbs nodes
        fbs_nodes, non_fbs_nodes = self.splitNodesFromFBS( nodes, is_partial_graph_index=is_partial_graph_index )

        if( len( fbs_nodes ) == 0 ):
            fetchFunc = partial( self.partial_graph.getChildren, edges=edges, split_by_edge=split_by_edge, split=split )
            return self.fetchGraphNodes( nodes, fetchFunc, get_order=False, split=split, split_by_edge=split_by_edge, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=return_partial_graph_index, use_partial_graph=True )

        fetchFunc = partial( self.full_graph.getChildren, edges=edges, split_by_edge=split_by_edge, split=split )
        filterFunc = lambda n: not self.inFeedbackSet( n, is_partial_graph_index=False )
        return self.fetchGraphNodes( nodes, fetchFunc, get_order=False, split=split, split_by_edge=split_by_edge, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=return_partial_graph_index, use_partial_graph=False )

    def getPartialMates( self,
                         nodes,
                         edges=None,
                         split_by_edge=False,
                         split=False,
                         get_order=False,
                         is_partial_graph_index=False,
                         return_partial_graph_index=False ):

        # If this function is called for a fbs node, it won't exist in the partial graph.
        # So need to get the parents from the full graph and filter out the fbs nodes
        fbs_nodes, non_fbs_nodes = self.splitNodesFromFBS( nodes, is_partial_graph_index=is_partial_graph_index )

        if( len( fbs_nodes ) == 0 ):
            fetchFunc = partial( self.partial_graph.getMates, edges=edges, split_by_edge=split_by_edge, split=split, get_order=get_order )
            return self.fetchGraphNodes( nodes, fetchFunc, get_order=get_order, split=split, split_by_edge=split_by_edge, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=return_partial_graph_index, use_partial_graph=True )

        fetchFunc = partial( self.full_graph.getMates, edges=edges, split_by_edge=split_by_edge, split=split, get_order=get_order )
        filterFunc = lambda n: not self.inFeedbackSet( n, is_partial_graph_index=False )
        return self.fetchGraphNodes( nodes, fetchFunc, get_order=get_order, split=split, split_by_edge=split_by_edge, is_partial_graph_index=is_partial_graph_index, return_partial_graph_index=return_partial_graph_index, use_partial_graph=False )

    ######################################################################

    def getUpEdges( self, nodes, split=False, is_partial_graph_index=False, use_partial_graph=False ):

        # Ensure that the nodes are using the correct indices
        partial_to_full = self.shouldConvertPartialToFull( is_partial_graph_index, use_partial_graph )
        if( partial_to_full is not None ):
            nodes = self.convertIndices( nodes, partial_to_full=partial_to_full )

        if( use_partial_graph ):
            return self.partial_graph.getUpEdges( nodes, split=split )
        else:
            return self.full_graph.getUpEdges( nodes, split=split )

    def getDownEdges( self, nodes, skip_edges=None, split=False, is_partial_graph_index=False, use_partial_graph=False ):

        # Ensure that the nodes are using the correct indices
        partial_to_full = self.shouldConvertPartialToFull( is_partial_graph_index, use_partial_graph )
        if( partial_to_full is not None ):
            nodes = self.convertIndices( nodes, partial_to_full=partial_to_full )

        if( use_partial_graph ):
            return self.partial_graph.getDownEdges( nodes, split=split, skip_edges=skip_edges )
        else:
            return self.full_graph.getDownEdges( nodes, split=split, skip_edges=skip_edges )

    ######################################################################

    def nParents( self, node, is_partial_graph_index=False, use_partial_graph=False ):
        if( use_partial_graph == False ):
            return self.getFullParents( node, is_partial_graph_index=is_partial_graph_index ).shape[ 0 ]
        else:
            return self.getPartialParents( node, is_partial_graph_index=is_partial_graph_index ).shape[ 0 ]

    def nSiblings( self, node, is_partial_graph_index=False, use_partial_graph=False ):
        if( use_partial_graph == False ):
            return self.getFullSiblings( node, is_partial_graph_index=is_partial_graph_index ).shape[ 0 ]
        else:
            return self.getPartialSiblings( node, is_partial_graph_index=is_partial_graph_index ).shape[ 0 ]

    def nMates( self, node, edges=None, is_partial_graph_index=False, use_partial_graph=False ):
        if( use_partial_graph == False ):
            return self.getFullMates( node, edges=edges, is_partial_graph_index=is_partial_graph_index ).shape[ 0 ]
        else:
            return self.getPartialMates( node, edges=edges, is_partial_graph_index=is_partial_graph_index ).shape[ 0 ]

    def nChildren( self, node, edges=None, is_partial_graph_index=False, use_partial_graph=False ):
        if( use_partial_graph == False ):
            return self.getFullChildren( node, edges=edges, is_partial_graph_index=is_partial_graph_index ).shape[ 0 ]
        else:
            return self.getPartialChildren( node, edges=edges, is_partial_graph_index=is_partial_graph_index ).shape[ 0 ]

    ######################################################################

    def messagePassing( self, uWork, vWork, **kwargs ):
        # Message passing is from the partial graph!
        return self.partial_graph.messagePassing( uWork, vWork, **kwargs )

##########################################################################################################

class GraphMessagePasserFBS( __fbsMessagePassingMixin, GraphMessagePasser ):
    pass

##########################################################################################################

class GraphMessagePasserFBSImproved( __fbsMessagePassingMixinImproved, GraphMessagePasser ):
    pass
