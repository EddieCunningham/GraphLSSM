from GraphicalMessagePassingBase import Graph, GraphMessagePasser, dprint
from GraphFilterBase import GraphFilter
import numpy as np
from functools import reduce
from scipy.sparse import coo_matrix
import string
from collections import Iterable

import os
path = os.getcwd()

import sys
sys.path.append( '/Users/Eddie/GenModels' )
from GM.Distributions import Normal
sys.path.append( path )

def logEye( K ):
    mat = np.empty( ( K, K ) )
    mat[ : ] = np.NINF
    mat[ np.diag_indices( K ) ] = 0
    return mat

class GraphCategoricalForwardBackward( GraphFilter ):
    def __init__( self, K ):
        super( GraphCategoricalForwardBackward, self ).__init__()
        self.K = K

    def aFBS( self, U, V, node, downEdge, debug=True  ):
        # This should already be set when we generate the filter probs!
        return U[ node ]

    def genFilterProbs( self ):

        # Initialize U and V
        U = []
        for node in self.nodes:
            U.append( np.zeros( ( self.K, ) ) )

        V_row = self.pmask.row
        V_col = self.pmask.col
        V_data = []
        for node in self.pmask.row:
            V_data.append( np.zeros( ( self.K, ) ) )

        # Invalidate all data elements
        for node in self.nodes:
            U[ node ][ : ] = np.nan
            self.assignV( ( V_row, V_col, V_data ), node, np.nan, keepShape=True )

        # Initialize U and V for fbs nodes.  This is a special case because
        # we're basically cutting these nodes from the graph
        for i, fbs in enumerate( self.feedbackSets ):

            for node in fbs:
                fbsIndex = self.fbs.tolist().index( node )
                newShape = np.ones( fbsIndex )
                newShape[ 0 ] = self.K
                newShape[ -1 ] = self.K
                U[ node ] = np.eye( self.K ).reshape( newShape )
                self.assignV( ( V_row, V_col, V_data ), node, np.zeros( ( self.K, self.K ) ).reshape( newShape ) )

        return U, ( V_row, V_col, V_data )

    def genWorkspace( self ):
        return np.array( [] )

    ######################################################################

    def updateParamsFromGraphs( self, ys, initialDist, transDists, emissionDist, graphs ):
        super( GraphFilter, self ).updateParamsFromGraphs( graphs )
        assert initialDist.shape == ( self.K, )
        for transDist in transDists:
            assert np.allclose( np.ones( self.K ), transDist.sum( axis=-1 ) )
        assert emissionDist.shape[ 0 ] == self.K
        assert np.isclose( 1.0, initialDist.sum() )
        assert np.allclose( np.ones( self.K ), emissionDist.sum( axis=1 ) )
        self.pi0 = np.log( initialDist )
        self.pis = {}
        for dist in transDists:
            ndim = dist.ndim
            assert ndim not in self.pis
            self.pis[ ndim ] = np.log( dist )

        print( 'There are ', len( self.pis ), 'transition matrices with dims', list( self.pis.keys() ) )
        _L = np.log( emissionDist )

        if( not isinstance( ys, np.ndarray ) ):
            ys = np.array( ys )

        print('_L', _L)

        self.L = np.array( [ _L[ :, y ] for y in ys ] ).sum( axis=0 ).T

    ######################################################################

    def transitionProb( self, child, parents, parentOrder ):
        ndim = len( parents ) + 1
        pi = self.pis[ ndim ]
        # Reshape pi's axes to match parent order
        assert len( parents ) + 1 == pi.ndim
        assert parentOrder.shape[ 0 ] == parents.shape[ 0 ]
        pi = np.moveaxis( pi, np.arange( ndim ), np.hstack( ( parentOrder, ndim - 1 ) ) )
        return pi

    ######################################################################

    def emissionProb( self, node, forward=False ):
        prob = self.L[ node ].reshape( ( -1, ) )
        return prob

    ######################################################################

    @classmethod
    def multiplyTerms( cls, terms ):
        # Basically np.einsum but in log space

        assert isinstance( terms, Iterable )
        for t in terms:
            assert isinstance( t, Iterable )

        # Remove the empty terms
        terms = [ t for t in terms if np.prod( t.shape ) > 1 ]

        ndim = max( [ len( term.shape ) for term in terms ] )

        axes = [ [ i for i, s in enumerate( t.shape ) if s != 1 ] for t in terms ]

        # Get the shape of the output
        shape = np.ones( ndim, dtype=int )
        for ax, term in zip( axes, terms ):
            shape[ np.array( ax ) ] = term.squeeze().shape

        totalElts = shape.prod()
        if( totalElts > 1e8 ):
            assert 0, 'Don\'t do this on a cpu!  Too many terms: %d'%( int( totalElts ) )

        # Build a meshgrid out of each of the terms over the right axes
        # and sum.  Doing it this way because np.einsum doesn't work
        # for matrix multiplication in log space - we can't do np.einsum
        # but add instead of multiply over indices
        ans = np.zeros( shape )
        for ax, term in zip( axes, terms ):

            ax = [ i for i, s in enumerate( term.shape ) if s != 1 ]

            for _ in range( ndim - term.ndim ):
                term = term[ ..., None ]

            # Build a meshgrid to correspond to the final shape and repeat
            # over axes that aren't in ax
            reps = np.copy( shape )
            reps[ np.array( ax ) ] = 1

            ans += np.tile( term, reps )

        return ans

    ######################################################################

    @classmethod
    def integrate( cls, integrand, axes ):
        # Need adjusted axes because the relative axes in integrand change as we reduce
        # over each axis
        assert isinstance( axes, Iterable )
        if( len( axes ) == 0 ):
            return integrand

        assert max( axes ) < integrand.ndim
        axes = np.array( axes )
        axes[ axes < 0 ] = integrand.ndim + axes[ axes < 0 ]
        adjustedAxes = np.array( sorted( axes ) ) - np.arange( len( axes ) )
        for ax in adjustedAxes:
            integrand = np.logaddexp.reduce( integrand, axis=ax )
        return integrand

    ######################################################################

    def uBaseCase( self, node, debug=True ):

        initialDist = self.pi0
        dprint( 'initialDist:', initialDist, use=debug )

        emission = self.emissionProb( node )
        dprint( 'emission:', emission, use=debug )

        newU = self.multiplyTerms( terms=( emission, initialDist ) )
        dprint( 'newU:', newU, use=debug )
        return newU

    def vBaseCase( self, node, debug=True ):
        return np.zeros( self.K )

    ######################################################################

    def condition( self, nodeMask ):
        return np.arange( nodeMask.shape[ 0 ] )[ nodeMask ]

    ######################################################################

    def updateU( self, nodes, newU, U ):
        fbsSize = len( self.fbs )

        for u, node in zip( newU, nodes ):
            U[ node ] = u

    def updateV( self, nodes, edges, newV, V ):

        V_row, V_col, V_data = V

        print( 'UPDATING V' )
        print( 'nodes', nodes )
        print( 'edges', edges )
        print( 'newV', newV )
        print( 'V_data', V_data )

        for node, edge, v in zip( nodes, edges, newV ):
            if( edge is None ):
                continue

            dataIndices = np.in1d( V_row, node ) & np.in1d( V_col, edge )

            print( 'dataIndices', dataIndices )
            print( 'node', node )
            print( 'edge', edge )
            print( 'v', v )

            for i, maskValue in enumerate( dataIndices ):

                print( 'maskValue', maskValue )
                print( 'i', i )
                # Don't convert V_data to an np.array even though it makes this
                # step faster because it messes up when we add fbs nodes
                if( maskValue == True ):
                    V_data[ i ] = v

        print( 'finally, V_data is', V_data )


    ######################################################################
