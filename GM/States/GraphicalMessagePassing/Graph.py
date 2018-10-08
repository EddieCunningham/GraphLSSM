import graphviz
from scipy.sparse import coo_matrix
import itertools
import autograd.numpy as np
from collections import deque, namedtuple
import networkx

__all__ = [ 'Graph', 'DataGraph', 'GroupGraph' ]

class Edges():
    def __init__( self ):
        self.up_edge = None
        self.down_edges = []

class Graph():
    # This class is how we make sparse matrices

    def __init__( self ):
        self.nodes = set()
        self.edge_children = list()
        self.edge_parents = list()

    ######################################################################

    def toNetworkX( self ):
        graph = networkx.DiGraph()

        for e, ( parents, children ) in enumerate( zip( self.edge_parents, self.edge_children ) ):
            for p in parents:
                graph.add_edge( p, -e - 1, family_type='parent' )
            for c in children:
                graph.add_edge( -e - 1, c, family_type='child' )

        return graph

    ######################################################################

    @property
    def roots( self ):
        if( hasattr( self, '_roots' ) == False ):
            self._roots = self.nodes - set( list( itertools.chain( *self.edge_children ) ) )
        return self._roots

    @property
    def leaves( self ):
        if( hasattr( self, '_leaves' ) == False ):
            self._leaves = self.nodes - set( list( itertools.chain( *self.edge_parents ) ) )
        return self._leaves

    @property
    def node_memberships( self ):
        if( hasattr( self, '_node_memberships' ) == False ):
            self._node_memberships = {}
            for e, ( parents, children ) in enumerate( zip( self.edge_parents, self.edge_children ) ):
                for parent in parents:
                    if( parent not in self._node_memberships ):
                        self._node_memberships[ parent ] = Edges()
                    if( e not in self._node_memberships[ parent ].down_edges ):
                        self._node_memberships[ parent ].down_edges.append( e )

                for child in children:
                    if( child not in self._node_memberships ):
                        self._node_memberships[ child ] = Edges()
                    if( self._node_memberships[ child ].up_edge is not None ):
                        assert self._node_memberships[ child ].up_edge == e
                    else:
                        self._node_memberships[ child ].up_edge = e

        return self._node_memberships

    ######################################################################

    def getParents( self, node ):
        up_edge = self.node_memberships[ node ].up_edge
        return self.edge_parents[ up_edge ] if up_edge is not None else []

    def getChildren( self, node ):
        ans = []
        for down_edge in self.node_memberships[ node ].down_edges:
            ans.append( self.edge_children[ down_edge ] )
        return ans

    ######################################################################

    def forwardPass( self ):
        edge_semaphores = np.array( [ len( e ) for e in self.edge_parents ] )

        # Get the first edges to start with
        for edge, parents in enumerate( self.edge_parents ):
            edge_semaphores[ edge ] -= len( set.intersection( self.roots, set( parents ) ) )

        for root in self.roots:
            yield root

        edges = np.arange( edge_semaphores.shape[ 0 ], dtype=int )

        done_edges = edge_semaphores == 0
        q = deque( edges[ done_edges ] )
        while( len( q ) > 0 ):

            edge = q.popleft()
            for child in self.edge_children[ edge ]:
                yield child
                for down_edge in self.node_memberships[ child ].down_edges:
                    edge_semaphores[ down_edge ] -= 1

            now_done = ( edge_semaphores == 0 ) & ( ~done_edges )
            q.extend( edges[ now_done ] )
            done_edges |= now_done

    def backwardPass( self ):
        edge_semaphores = np.array( [ len( e ) for e in self.edge_children ] )

        # Get the first edges to start with
        for edge, children in enumerate( self.edge_children ):
            edge_semaphores[ edge ] -= len( set.intersection( self.leaves, set( children ) ) )

        for leaf in self.leaves:
            yield leaf

        edges = np.arange( edge_semaphores.shape[ 0 ], dtype=int )

        done_edges = edge_semaphores == 0
        q = deque( edges[ done_edges ] )
        while( len( q ) > 0 ):

            edge = q.popleft()
            for parent in self.edge_parents[ edge ]:
                yield parent
                if( self.node_memberships[ parent ].up_edge is not None ):
                    edge_semaphores[ self.node_memberships[ parent ].up_edge ] -= 1

            now_done = ( edge_semaphores == 0 ) & ( ~done_edges )
            q.extend( edges[ now_done ] )
            done_edges |= now_done

    ######################################################################

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

    ######################################################################

    def addEdge( self, parents, children ):
        assert isinstance( parents, list ) or isinstance( parents, tuple )
        assert isinstance( children, list ) or isinstance( children, tuple )
        for node in parents + children:
            self.nodes.add( node )

        self.edge_children.append( children )
        self.edge_parents.append( parents )

    ######################################################################

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

    ######################################################################

    def toMatrix( self ):

        node_list = list( self.nodes )

        parent_mask = self.cooMatrixFromNodeEdge( node_list, self.edge_parents )
        child_mask = self.cooMatrixFromNodeEdge( node_list, self.edge_children )

        return parent_mask, child_mask

    ######################################################################

    @property
    def edge_style( self ):
        if( hasattr( self, '_edge_style' ) == False ):
            self.edge_style = dict( width='0.25',
                                    height='0.25',
                                    fontcolor='white',
                                    style='filled',
                                    fillcolor='black',
                                    fixedsize='true',
                                    fontsize='6' )
        return self._edge_style

    @edge_style.setter
    def edge_style( self, val ):
        self._edge_style = val

    ######################################################################

    @property
    def node_style( self ):
        if( hasattr( self, '_node_style' ) == False ):
            self.node_style = dict( fixedsize='true' )
        return self._node_style

    @node_style.setter
    def node_style( self, val ):
        self._node_style = val

    ######################################################################

    @property
    def highlight_node_style( self ):
        if( hasattr( self, '_highlight_edge_style' ) == False ):
            self.highlight_node_style = dict( fontcolor='white',
                                              style='filled',
                                              fillcolor='blue' )
        return self._highlight_edge_style

    @highlight_node_style.setter
    def highlight_node_style( self, val ):
        self._highlight_edge_style = val

    ######################################################################

    def draw( self, render=True, horizontal=False, labels=True, styles={}, node_to_style_key={}, edge_to_style_key={}, file_format='png', output_folder='.', output_name='graph' ):

        getEdgeStyle = lambda ne: styles[ edge_to_style_key[ ne ] ] if ne in edge_to_style_key else self.node_style

        d = graphviz.Digraph( format=file_format, filename=output_name, directory=output_folder )
        if( horizontal == True ):
            d.attr( rankdir='LR' )

        for e, ( parents, children ) in enumerate( zip( self.edge_parents, self.edge_children ) ):
            for p in parents:
                d.edge( '%d '%( p ), '%d'%( e ), **getEdgeStyle( ( p, e ) ) )
            for c in children:
                d.edge( '%d'%( e ), '%d '%( c ), **getEdgeStyle( ( e, c ) ) )

            if( labels == True ):
                d.node( '%d'%( e ), **self.edge_style )
            else:
                d.node( **self.edge_style )

        for n, style_key in node_to_style_key.items():
            if( labels == True ):
                d.node( '%d '%( n ), **styles[ style_key ] )
            else:
                d.node( **styles[ style_key ] )

        if( render ):
            d.render( cleanup=True )

        return d

##########################################################################

class DataGraph( Graph ):
    # Same as graph except each node will hold some data field

    def __init__( self ):
        super().__init__()
        self.data = {}
        self.possible_latent_states = {}

    ######################################################################

    def toNetworkX( self ):
        graph = super().toNetworkX()

        for node, datum in self.data.items():
            graph.nodes[ node ][ 'data' ] = datum
            if( node in self.possible_latent_states ):
                graph.nodes[ node ][ 'possible_latent_states' ] = self.possible_latent_states[ node ]
            else:
                graph.nodes[ node ][ 'possible_latent_states' ] = -1

        return graph

    ######################################################################

    @staticmethod
    def fromGraph( graph, nodes_data ):
        data_graph = DataGraph()
        data_graph.nodes = graph.nodes
        data_graph.edge_children = graph.edge_children
        data_graph.edge_parents = graph.edge_parents

        for node in graph.nodes:
            data_graph.data[ node ] = None

        for node, data in nodes_data:
            data_graph.data[ node ] = data

        return data_graph

    ######################################################################

    def addEdge( self, parents, children ):
        assert isinstance( parents, list ) or isinstance( parents, tuple )
        assert isinstance( children, list ) or isinstance( children, tuple )
        for node in parents + children:
            self.nodes.add( node )
            self.data[ node ] = None

        self.edge_children.append( children )
        self.edge_parents.append( parents )

    def setNodeData( self, nodes, datum ):
        if( isinstance( nodes, int ) ):
            self.data[ nodes ] = datum
        else:
            for node, data in zip( nodes, datum ):
                self.data[ node ] = data

    def setPossibleLatentStates( self, nodes, possible_states ):
        if( isinstance( nodes, int ) ):
            self.possible_latent_states[ nodes ] = possible_states
        else:
            for node, states in zip( nodes, possible_states ):
                self.possible_latent_states[ node ] = states

##########################################################################

class GroupGraph( DataGraph ):
    # Same as data graph, but can specify group for each node

    def __init__( self ):
        super().__init__()
        self.groups = {}

    def toNetworkX( self ):
        graph = super().toNetworkX()

        for node, group in self.groups.items():
            graph.nodes[ node ][ 'group' ] = group

        return graph

    def setGroups( self, nodes, groups ):
        if( isinstance( nodes, int ) ):
            self.groups[ nodes ] = groups
        else:
            for node, group in zip( nodes, groups ):
                self.groups[ node ] = group

    @staticmethod
    def fromGraph( graph, nodes_data, node_groups ):
        group_graph = GroupGraph()
        group_graph.nodes = graph.nodes
        group_graph.edge_children = graph.edge_children
        group_graph.edge_parents = graph.edge_parents

        for node in graph.nodes:
            group_graph.data[ node ] = None

        for node, data in nodes_data:
            group_graph.data[ node ] = data

        for node, group in node_groups:
            group_graph.groups[ node ] = group

        return group_graph
