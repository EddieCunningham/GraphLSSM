import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix
import graphviz
from collections import Iterable
from functools import partial
import itertools
from .Graph import Graph

__all__ = [ 'GraphMessagePasser',
            'GraphMessagePasserFBS' ]

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

    def updateGraphs( self, graphs ):

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

        self.updateMasks( parent_masks, child_masks )

    def updateMasks( self, parent_masks, child_masks ):

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

    @property
    def roots( self ):
        # Get the number of edges that each node is a parent of
        parent_of_edge_count = self.pmask.getnnz( axis=1 )

        # Get the number of edges that each node is a child of
        child_of_edge_count = self.cmask.getnnz( axis=1 )

        # Get the indices of leaves and roots
        roots = self.nodes[ ( parent_of_edge_count != 0 ) & ( child_of_edge_count == 0 ) ]

        return roots

    @property
    def leaves( self ):
        # Get the number of edges that each node is a parent of
        parent_of_edge_count = self.pmask.getnnz( axis=1 )

        # Get the number of edges that each node is a child of
        child_of_edge_count = self.cmask.getnnz( axis=1 )

        # Get the indices of leaves and roots
        leaves = self.nodes[ ( child_of_edge_count != 0 ) & ( parent_of_edge_count == 0 ) ]

        return leaves

    def baseCaseNodes( self ):

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
        for children_for_nodes in children:
            u_semaphore[ children_for_nodes ] -= 1
            assert np.all( u_semaphore[ children_for_nodes ] >= 0 )

        # Decrement v_semaphore for all mates over down edges that node and mate are a part of
        mates_and_edges = self.getMates( nodes, split_by_edge=True, split=True )
        for mate_and_edge in mates_and_edges:
            for e, m in mate_and_edge:
                v_semaphore.data[ np.in1d( v_semaphore.row, m ) & np.in1d( v_semaphore.col, e ) ] -= 1
                assert np.all( v_semaphore.data[ np.in1d( v_semaphore.row, m ) & np.in1d( v_semaphore.col, e ) ] >= 0 )

        # Update u_done
        u_done[ nodes ] = True

    def VDone( self, nodes_and_edges, u_semaphore, v_semaphore, v_done ):

        nodes, edges = nodes_and_edges

        # Decrement u_semaphore for children that come from a different edge than the one computed for V
        children_and_edges = self.getChildren( nodes, split_by_edge=True, split=True )
        for edge, child_and_edge in zip( edges, children_and_edges ):
            for e, c in child_and_edge:
                if( e == edge ):
                    continue
                u_semaphore[ c ] -= 1
                assert np.all( u_semaphore[ c ] >= 0 )

        # Decrement u_semaphore for all siblings
        siblings = self.getSiblings( nodes, split=True )
        for _e, siblings_for_node in zip( edges, siblings ):
            if( _e is None ):
                # If this node doesn't have a down edge, then we don't want to decrement
                continue
            u_semaphore[ siblings_for_node ] -= 1
            assert np.all( u_semaphore[ siblings_for_node ] >= 0 )

        # Decrement v_semaphore for mates that aren't current edge
        mates_and_edges = self.getMates( nodes, split_by_edge=True, split=True )
        for edge, mate_and_edge in zip( edges, mates_and_edges ):
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

        # Update v_done
        edges_without_none = np.array( [ e for e in edges if e is not None ] )
        v_done.data[ np.in1d( v_done.row, nodes ) & np.in1d( v_done.col, edges_without_none ) ] = True

    ######################################################################

    def forwardPass( self, work, **kwargs ):
        # Calls work at every node when its parents are ready

        # Get the number of edges that each node is a parent of
        # Get the number of edges that each node is a child of
        parent_of_edge_count = self.pmask.getnnz( axis=1 )
        child_of_edge_count = self.cmask.getnnz( axis=1 )

        # Start with the roots
        node_list = self.nodes[ ( parent_of_edge_count != 0 ) & ( child_of_edge_count == 0 ) ]

        # Semaphore is the number of parents in the nodes up edge
        parents_per_edge = self.pmask.getnnz( axis=0 )
        edge_semaphore = parents_per_edge

        traversed_edges = np.zeros( self.cmask.shape[ 1 ], dtype=bool )

        while( node_list.size > 0 ):

            # Do work on the current nodes
            work( node_list, **kwargs )

            # Subtract a value off of each of node_list's down edges
            decremented_edges = self.pmask.col[ np.in1d( self.pmask.row, node_list ) ]
            edge_counts = np.bincount( decremented_edges, minlength=self.cmask.shape[ 1 ] )
            edge_semaphore -= edge_counts

            # Select the next edges to traverse and update the traversed list
            next_edge_mask = ( edge_semaphore == 0 ) & ( ~traversed_edges )
            next_edges = np.arange( next_edge_mask.shape[ 0 ] )[ next_edge_mask ]
            traversed_edges[ next_edge_mask ] = True

            # Update node_list
            node_list = self.cmask.row[ np.in1d( self.cmask.col, next_edges ) ]

        assert np.any( edge_semaphore != 0 ) == False

    ######################################################################

    def loopyNextNodes( self, last_u_list, last_v_list ):
        last_u_nodes = last_u_list
        last_v_nodes, last_v_edges = last_v_list

        # Add all of the children from last_u_nodes and siblings from last_v_nodes
        next_u_list = np.hstack( ( self.getChildren( last_u_nodes ), self.getSiblings( last_v_nodes ) ) )

        # Add children that come from a different edge than the one computed for last_v_nodes
        children_and_edges = self.getChildren( last_v_nodes, split_by_edge=True, split=True )
        for edge, child_and_edge in zip( last_v_edges, children_and_edges ):
            for e, c in child_and_edge:
                if( e == edge ):
                    continue
                next_u_list = np.hstack( ( next_u_list, c ) )

        # Add mates over down edges that last_u_nodes and mate are a part of
        mates_and_edges = self.getMates( last_u_nodes, split_by_edge=True, split=True )
        next_v_nodes = np.array( [], dtype=int )
        next_v_edges = np.array( [], dtype=int )
        for mate_and_edge in mates_and_edges:
            for e, m in mate_and_edge:
                mask = np.in1d( self.pmask.row, m ) & np.in1d( self.pmask.col, e )
                next_v_nodes = np.hstack( ( next_v_nodes, self.pmask.row[ mask ] ) )
                next_v_edges = np.hstack( ( next_v_edges, self.pmask.col[ mask ] ) )

        # Add mates of last_v_nodes that aren't from last_v_edges
        mates_and_edges = self.getMates( last_v_nodes, split_by_edge=True, split=True )
        for edge, mate_and_edge in zip( last_v_edges, mates_and_edges ):
            for e, m in mate_and_edge:
                if( e == edge ):
                    continue
                mask = np.in1d( self.pmask.row, m ) & np.in1d( self.pmask.col, e )
                next_v_nodes = np.hstack( ( next_v_nodes, self.pmask.row[ mask ] ) )
                next_v_edges = np.hstack( ( next_v_edges, self.pmask.col[ mask ] ) )

        # Add parents over up edges for last_v_nodes
        parents = self.getParents( last_v_nodes, split=True )
        up_edges = self.getUpEdges( last_v_nodes, split=True )
        for _e, p, e in zip( last_v_edges, parents, up_edges ):
            if( _e is None ):
                # If this node doesn't have a down edge, then we don't want to decrement
                continue
            mask = np.in1d( self.pmask.row, p ) & np.in1d( self.pmask.col, e )
            next_v_nodes = np.hstack( ( next_v_nodes, self.pmask.row[ mask ] ) )
            next_v_edges = np.hstack( ( next_v_edges, self.pmask.col[ mask ] ) )

        # Remove duplicates
        next_u_list = np.array( list( set( next_u_list.tolist() ) ) )

        next_v_nodes, next_v_edges = zip( *list( set( zip( next_v_nodes, next_v_edges ) ) ) )
        next_v_nodes = np.array( next_v_nodes )
        next_v_edges = np.array( next_v_edges )

        return next_u_list, ( next_v_nodes, next_v_edges )

    ######################################################################

    def upDown( self, uWork, vWork, enable_loopy=False, loopyHasConverged=None, **kwargs ):
        # Run the up down algorithm for latent state space models

        u_done, v_done = self.progressInit()
        u_semaphore, v_semaphore = self.countSemaphoreInit()
        u_list, v_list = self.baseCaseNodes()

        # Do work for base case nodes
        uWork( True, u_list, **kwargs )
        vWork( True, v_list, **kwargs )

        i = 1
        loopy = False

        # Filter over all of the graphs
        while( u_list.size > 0 or v_list[ 0 ].size > 0 ):

            if( i > 1 ):
              # Do work for each of the nodes
              uWork( False, u_list, **kwargs )
              vWork( False, v_list, **kwargs )

            # Mark that we're done with the current nodes
            self.UDone( u_list, u_semaphore, v_semaphore, u_done )
            self.VDone( v_list, u_semaphore, v_semaphore, v_done )

            last_u_list = u_list
            last_v_list = v_list

            # Find the next nodes that are ready
            u_list = self.readyForU( u_semaphore, u_done )
            v_list = self.readyForV( v_semaphore, v_done )

            i += 1

            # Check if we need to do loopy propagation belief
            if( ( u_list.size == 0 and v_list[ 0 ].size == 0 ) and \
                ( not np.any( u_done ) or not np.any( v_done.data ) ) ):

                if( not enable_loopy ):
                    assert 0, 'Cycle encountered.  Set enable_loopy to True'
                if( loopyHasConverged is None ):
                    assert 0, 'Need to specify the convergence function loopyHasConverged!'

                print( 'Cycle encountered!  Starting loopy belief propagation...' )

                # The loopy algorithm will just forcibly keep adding mates, parents, siblings and children
                # to be computed
                u_list = last_u_list
                v_list = last_v_list

                while( loopyHasConverged() == False ):

                    u_list, v_list = self.loopyNextNodes( u_list, v_list )

                    uWork( False, u_list, **kwargs )
                    vWork( False, v_list, **kwargs )

                loopy = True
                break

        if( loopy == False ):
            assert np.any( u_semaphore != 0 ) == False
            assert np.any( v_semaphore.data != 0 ) == False

    ######################################################################

    def nParents( self, node ):
        return self.getParents( node ).shape[ 0 ]

    def nSiblings( self, node ):
        return self.getSiblings( node ).shape[ 0 ]

    def nMates( self, node, edges=None ):
        return self.getMates( node, edges=edges ).shape[ 0 ]

    def nChildren( self, node, edges=None ):
        return self.getChildren( node ).shape[ 0 ]

##########################################################################################################

class GraphMessagePasserFBS( GraphMessagePasser ):

    def toGraph( self, use_partial=False ):
        if( use_partial ):
            return self.partial_graph.toGraph()
        return self.full_graph.toGraph()

    def draw( self, use_partial=False, render=True, **kwargs ):
        if( use_partial ):
            return self.toGraph( use_partial=True ).draw( render=render, **kwargs )
        return self.toGraph().draw( render=render, **kwargs )

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

    def updateGraphs( self, graphs ):

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

        self.updateMasks( parent_masks, child_masks, feedback_sets=feedback_sets )

    def updateMasks( self, parent_masks, child_masks, feedback_sets=None ):

        # The main idea with this class is to have 2 different graphs.  The full
        # graph will cycles, and a graph without the feedback nodes which will
        # be acyclic.  Message passing will be done over the partial graph while
        # using computations with the full graph.  At the end of message passing,
        # the fbs nodes will be integrated out as needed.

        # This is the full directed graph.  It might have directed cycles
        self.full_graph = GraphMessagePasser()
        self.full_graph.updateMasks( parent_masks, child_masks )

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

        # # Remove edges that either don't have parents or don't have children
        # loose_edges = np.hstack( ( np.setdiff1d( _pmask_col, _cmask_col ), np.setdiff1d( _cmask_col, _pmask_col ) ) )
        # bad_cols = ~np.in1d( _pmask_col, loose_edges )
        # _pmask_row = _pmask_row[ bad_cols ]
        # _pmask_col = _pmask_col[ bad_cols ]
        # _pmask_data = _pmask_data[ bad_cols ]

        # bad_cols = ~np.in1d( _cmask_col, loose_edges )
        # _cmask_row = _cmask_row[ bad_cols ]
        # _cmask_col = _cmask_col[ bad_cols ]
        # _cmask_data = _cmask_data[ bad_cols ]

        # # Re-index the columns so they are consecutive
        # all_edges = np.unique( np.hstack( ( _pmask_col, _cmask_col ) ) )
        # edge_mapper = dict( [ ( e, i ) for i, e in enumerate( all_edges ) ] )
        # edge_mapped_reversed = dict( [ ( i, e ) for i, e in enumerate( all_edges ) ] )
        # self.fullGraphEdgeToPartialGraphEdge = np.vectorize( lambda x: edge_mapper[ x ], otypes=[np.float64] )
        # self.partialGraphEdgeToFullGraphEdge = np.vectorize( lambda x: edge_mapped_reversed[ x ], otypes=[np.float64] )

        # _pmask_col = np.array( [ edge_mapper[ e ] for e in _pmask_col ] )
        # _cmask_col = np.array( [ edge_mapper[ e ] for e in _cmask_col ] )

        # # The new shape will have fewer nodes
        # shape = ( self.full_graph.pmask.shape[ 0 ] - self.fbs.shape[ 0 ], all_edges.shape[ 0 ] )
        # parital_pmask = coo_matrix( ( _pmask_data, ( _pmask_row, _pmask_col ) ), shape=shape, dtype=int )

        # shape = ( self.full_graph.cmask.shape[ 0 ] - self.fbs.shape[ 0 ], all_edges.shape[ 0 ] )
        # parital_cmask = coo_matrix( ( _cmask_data, ( _cmask_row, _cmask_col ) ), shape=shape, dtype=int )

        # The new shape will have fewer nodes
        shape = ( self.full_graph.pmask.shape[ 0 ] - self.fbs.shape[ 0 ], self.full_graph.pmask.shape[ 1 ] )
        parital_pmask = coo_matrix( ( _pmask_data, ( _pmask_row, _pmask_col ) ), shape=shape, dtype=int )

        shape = ( self.full_graph.cmask.shape[ 0 ] - self.fbs.shape[ 0 ], self.full_graph.cmask.shape[ 1 ] )
        parital_cmask = coo_matrix( ( _cmask_data, ( _cmask_row, _cmask_col ) ), shape=shape, dtype=int )

        # This is the full graph without the feedback set nodes.  It will
        # be a directed acyclic graph
        self.partial_graph = GraphMessagePasser()
        # print( 'parital_pmask.shape', parital_pmask.shape )
        # print( 'parital_pmask', parital_pmask )
        # print( 'parital_cmask.shape', parital_cmask.shape )
        # print( 'parital_cmask', parital_cmask )
        # assert 0
        self.partial_graph.updateMasks( [ parital_pmask ], [ parital_cmask ] )

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

    @property
    def pmask( self ):
        return self.full_graph.pmask

    @property
    def cmask( self ):
        return self.full_graph.cmask

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

    def upDown( self, uWork, vWork, enable_loopy=False, loopyHasConverged=None, **kwargs ):
        # Message passing is from the partial graph!
        return self.partial_graph.upDown( uWork, vWork, enable_loopy=enable_loopy, loopyHasConverged=loopyHasConverged, **kwargs )

##########################################################################################################
