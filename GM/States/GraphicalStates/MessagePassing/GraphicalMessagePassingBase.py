import numpy as np
from scipy.sparse import dok_matrix, coo_matrix

class Graph():
    # This class is how we make sparse matrices

    def __init__():
        self.nodes = set()
        self.edges = set()
        self.edgeParents = set()

    def addEdge( self, parents, children ):
        assert isinstance( parents, list ) or isinstance( parents, tuple )
        assert isinstance( children, list ) or isinstance( children, tuple )
        for node in parents + children:
            self.nodes.add( node )

        self.edges.append( parents + children )
        self.edgeParents.append( parents )

    def toMatrix( self, returnParentMask=True ):
        nRows = len( self.node )
        nCols = len( self.edges )

        nodeList = list( self.nodes )
        edgeList = list( self.edges )
        edgeParentList = list( self.edgeParents )

        rows = []
        cols = []
        data = []

        for i, e in enumerate( self.edges ):
            colIndex = edgeList.index( e )
            for j, n in enumerate( n ):
                rowIndex = nodeList.index( n )

            rows.append( rowIndex )
            cols.append( colIndex )
            data.append( True )

        mat = coo_matrix( ( data, ( row, col ) ), shape=( nRows, nCols ), dtype=bool )

        if( returnParentMask == False ):
            return mat

        rows = []
        cols = []
        data = []

        for i, e in enumerate( self.edgeParents ):
            colIndex = edgeList.index( e )
            for j, n in enumerate( n ):
                rowIndex = nodeList.index( n )

            rows.append( rowIndex )
            cols.append( colIndex )
            data.append( True )

        parentMask = coo_matrix( ( data, ( row, col ) ), shape=( nRows, nCols ), dtype=bool )
        return mat, parentMask

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

    def updateParams( self, graphs, parentMasks, feedbackSets ):
        for graph, parentMask, feedbackSet in zip( graphs, parentMasks, feedbackSets ):
            assert isinstance( graph, coo_matrix )
            assert isinstance( parentMask, coo_matrix )
            assert isinstance( feedbackSet, coo_matrix )
            assert graph.shape == parentMask.shape and parentMask.shape == feedbackSet.shape
        self.pmask = self.combineGraphs( parentMasks )
        self.cmask = self.genChildMasks( graphs, parentMasks )
        self.fbs = self.combineGraphs( feedbackSets )

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

    def parents( self, nodes, split=True ):
        if( split ):
            return [ self.parents( n, split=False ) for n in nodes ]
        return np.unique( self.pmask.row[ np.in1d( self.pmask.col, self.cmask.col[ np.in1d( self.cmask.row, nodes ) ] ) ] )

    def children( self, nodes, split=True ):
        if( split ):
            return [ self.children( n, split=False ) for n in nodes ]
        return np.unique( self.cmask.row[ np.in1d( self.cmask.col, self.pmask.col[ np.in1d( self.pmask.row, nodes ) ] ) ] )

    def mates( self, nodes, split=True ):
        if( split ):
            return [ self.mates( n, split=False ) for n in nodes ]
        return np.setdiff1d( self.pmask.row[ np.in1d( self.pmask.col, self.pmask.col[ np.in1d( self.pmask.row, nodes ) ] ) ], nodes )

    def siblings( self, nodes, split=True ):
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

        # Parent of nodes in feedback set are nodes who
        fbsParents = self.parents( self.fbs )
        fbsChildren = self.children( self.fbs )

        # Generate the up and down base arrays
        uList = np.setdiff1d( np.hstack( ( rootIndices, fbsChildren ) ), self.fbs.row )
        vList = np.setdiff1d( np.hstack( ( leafIndices, fbsParents ) ), self.fbs.row )

        return ( uList, vList )

    ######################################################################

    def condition( self, nodes ):
        pass

    ######################################################################

    def uReady( self, nodes, U, V ):
        _, UVisited = U
        _, VVisited = V
        allParents = self.parents( nodes, split=True )
        allSiblings = self.siblings( nodes, split=True )

        uReady = []
        uNotReady = []
        for node, parents, siblings in zip( nodes, allParents, allSiblings ):

            if( not np.all( np.in1d( UVisited, parents ) ) or \
                not np.all( np.in1d( VVisited, parents ) ) or \
                not np.all( np.in1d( VVisited, siblings ) ) ):
                uNotReady.append( node )
            else:
                uReady.append( node )

        return np.array( uReady ), np.array( uNotReady )

    def vReady( self, nodes, U, V ):
        _, UVisited = U
        _, VVisited = V
        allMates = self.mates( nodes, split=True )
        allChildren = self.children( nodes, split=True )

        vReady = []
        vNotReady = []
        for node, mates, children in zip( nodes, allMates, allChildren ):

            if( not np.all( np.in1d( UVisited, mates ) ) or \
                not np.all( np.in1d( VVisited, mates ) ) or \
                not np.all( np.in1d( VVisited, children ) ) ):
                vNotReady.append( node )
            else:
                vReady.append( node )

        return np.array( vReady ), np.array( vNotReady )

    ######################################################################

    def uFilter( self, nodes, U, V, conditioning, workspace ):
        pass

    def vFilter( self, nodes, U, V, conditioning, workspace ):
        pass

    ######################################################################

    def filter( self ):

        workspace = self.genWorkspace()
        conditioning = self.condition( self.fbs )

        U, V = self.genFilterProbs()
        uList, vList = self.baseCaseNodes()

        # Filter over all of the graphs
        while( uList.size > 0 or vList.size > 0 ):

            # Find which nodes we can work on
            uReady, uNotReady = self.uReady( uList, U, V )
            vReady, vNotReady = self.vReady( vList, U, V )

            # Compute the next filter step
            self.uFilter( uReady, U, V, conditioning, workspace )
            self.vFilter( vReady, U, V, conditioning, workspace )

            # Put together the next nodes to work on
            uList = np.unique( np.concatenate( [ uNotReady, \
                                                 self.children( uReady ), \
                                                 self.siblings( vReady ) ] ) )
            vList = np.unique( np.concatenate( [ vNotReady, \
                                                 self.mates( uReady ), \
                                                 self.parents( vReady ) ] ) )

        # Integrate out the nodes that we cut
        self.integrateOutConditioning( U, V, conditioning, workspace )

        # Update the filter probs for the cut nodes
        self.filterConditionedNodes( U, V, conditioning, workspace )

        return alphas