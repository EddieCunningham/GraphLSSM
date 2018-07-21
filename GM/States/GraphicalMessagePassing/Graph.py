import graphviz
from scipy.sparse import coo_matrix
import os

__all__ = [ 'Graph', 'DataGraph' ]

class Graph():
    # This class is how we make sparse matrices

    def __init__( self ):
        self.nodes = set()
        self.edge_children = list()
        self.edge_parents = list()

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

    def draw( self, render=True, horizontal=False, styles={}, node_to_style_key={}, edge_to_style_key={}, file_format='png', output_folder='.', output_name='graph' ):

        getEdgeStyle = lambda ne: styles[ edge_to_style_key[ ne ] ] if ne in edge_to_style_key else self.node_style

        d = graphviz.Digraph( format=file_format, filename=output_name, directory=output_folder )
        if( horizontal == True ):
            d.attr( rankdir='LR' )

        for e, ( parents, children ) in enumerate( zip( self.edge_parents, self.edge_children ) ):
            for p in parents:
                d.edge( '%d '%( p ), '%d'%( e ), **getEdgeStyle( ( p, e ) ) )
            for c in children:
                d.edge( '%d'%( e ), '%d '%( c ), **getEdgeStyle( ( e, c ) ) )

            d.node( '%d'%( e ), **self.edge_style )

        for n, style_key in node_to_style_key.items():
            d.node( '%d '%( n ), **styles[ style_key ] )

        if( render ):
            d.render( cleanup=True )

        return d

##########################################################################

class DataGraph( Graph ):
    # Same as graph except each node will hold some data field

    def __init__( self ):
        super().__init__()
        self.data = {}

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

    def updateNodeData( self, nodes, datum ):
        if( isinstance( nodes, int ) ):
            self.data[ nodes ] = datum
        else:
            for node, data in zip( nodes, datum ):
                self.data[ node ] = data
