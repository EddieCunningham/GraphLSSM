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
        fbsSize = self.fbs.shape[ 0 ]

        U = np.zeros( ( self.nodes.shape[ 0 ], ) + ( self.K, ) * ( fbsSize + 1 ) )

        V_row = self.pmask.row
        V_col = self.pmask.col
        V_data = np.zeros( ( self.pmask.row.shape[ 0 ], ) + ( self.K, ) * ( fbsSize + 1 ) )

        U[ : ] = np.nan
        V_data[ : ] = np.nan

        for node in self.fbs:

            U[ node ] = np.NINF

            # If node is in the fbs, the value of U for node should be zero everywhere except
            # where the index of node in the fbs equals the index of the latent state for node
            # (which should be the last dimension!)
            for i in range( self.K ):

                # ixArgs should look like [ range( k ), ..., [ i ], range( k ), ..., [ i ] ]
                # where the index of the first [ i ] is the same index that node is in the feedback set
                ixArgs = [ range( self.K ) if node != _node else [ i ] for _node in self.fbs ]
                ixArgs.append( [ i ] )

                U[ node ][ np.ix_( *ixArgs ) ] = 0

            # V values should be 0
            V_data[ node, : ] = np.NINF

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

    def transitionProb( self, children, parents ):
        ndim = len( parents ) + 1
        pi = self.pis[ ndim ]
        return pi

    ######################################################################

    def emissionProb( self, node, forward=False ):
        prob = self.L[ node ].reshape( ( -1, ) )
        return prob

    ######################################################################

    def multiplyTerms( self, terms, axes, ndim ):

        assert isinstance( terms, Iterable ), isinstance( axes, Iterable )
        assert len( axes ) == len( terms )
        for i, ( ax, term ) in enumerate( zip( axes, terms ) ):
            assert isinstance( ax, Iterable )
            assert len( ax ) == term.ndim, 'i: %d Ax: %s, term.shape: %s'%( i, str( ax ), str( term.shape ) )

        # Remove the empty terms
        axes, terms = list( zip( *[ ( a, t ) for a, t in zip( axes, terms ) if term.size > 0 ] ) )

        # Get the shape of the output
        shape = np.ones( ndim, dtype=int )
        for ax, term in zip( axes, terms ):
            print( 'shape', shape )
            print( 'ax', ax )
            shape[ np.array( ax ) ] = term.shape

        # Make sure that the output shape is consistent with all of the axes
        for ax, term in zip( axes, terms ):
            assert np.all( shape[ np.array( ax ) ] == term.shape ), 'shape: %s, term.shape: %s'%( shape, term.shape )

        # Build a meshgrid out of each of the terms over the right axes
        # and sum.  Doing it this way because np.einsum doesn't work
        # for matrix multiplication in log space - we can't do np.einsum
        # but add instead of multiply over indices
        ans = np.zeros( shape )
        for ax, term in zip( axes, terms ):

            # Reshape the term to correspond to the axes
            # If term.shape is (4,2) and ax is (0,3) -> (4,1,1,2)
            newShape = np.ones( ndim, dtype=int )
            newShape[ np.array( ax ) ] = term.shape
            term = term.reshape( newShape )

            # Build a meshgrid to correspond to the final shape and repeat
            # over axes that aren't in ax
            reps = np.copy( shape )
            reps[ np.array( ax ) ] = 1

            ans += np.tile( term, reps )

        # Squeeze the matrix and return the true axes that each axis spans
        ans = np.squeeze( ans )
        returnAxes = [ i for i, s in enumerate( shape ) if s != 1 ]

        return ans, returnAxes

    ######################################################################

    def integrate( self, integrand, integrandAxes, axes ):
        axes = np.array( axes )

        # Make sure that all of the axes we are integrating over are valid
        assert np.all( np.in1d( axes, np.array( integrandAxes ) ) ) == True

        if( axes.size == 0 ):
            return integrand, integrandAxes

        if( integrand.size == 0 ):
            return np.array( [] ), []

        finalAxes = np.setdiff1d( np.array( integrandAxes ), np.array( axes ) )

        # Make sure we integrate over the correct axes at each step
        adjustedAxes = np.array( sorted( axes ) ) - np.arange( len( axes ) )
        for ax in adjustedAxes:
            integrand = np.logaddexp.reduce( integrand, axis=ax )

        assert len( integrand.shape ) == len( finalAxes ), 'integrand.shape: %s finalAxes: %s integrandAxes: %s axes: %s'%( integrand.shape, finalAxes, integrandAxes, axes )

        return integrand, finalAxes

    ######################################################################

    def uBaseCase( self, node, debug=True ):

        initialDist = self.pi0
        initialDistAxes = ( 0, )
        dprint( 'initialDist:', initialDist, use=debug )
        dprint( 'initialDistAxes:', initialDistAxes, use=debug )

        emission = self.emissionProb( node )
        emissionAxes = ( 0, )
        dprint( 'emission:', emission, use=debug )
        dprint( 'emissionAxes:', emissionAxes, use=debug )

        newU, newUAxes = self.multiplyTerms( terms=( emission, initialDist ), \
                                             axes=( emissionAxes, initialDistAxes ), \
                                             ndim=1 )
        dprint( 'newU:', newU, use=debug )
        return newU

    def vBaseCase( self, leaves, V, workspace ):
        # Don't need to do anything because P( ∅ | ... ) = 1
        return

    ######################################################################

    def condition( self, nodeMask ):
        return np.arange( nodeMask.shape[ 0 ] )[ nodeMask ]

    ######################################################################

    def updateU( self, nodes, newU, U ):
        fbsSize = len( self.fbs )

        for u, node in zip( newU, nodes ):
            if( fbsSize == 0 ):
                U[ node, : ] = u
            else:
                U[ node, :, 0: fbsSize + 1 ] = u

    def updateV( self, nodes, edges, newV, V ):
        goodEdgeMask = ~np.in1d( np.array( edges ), None )

        _nodes = np.array( nodes )[ goodEdgeMask ]
        _edges = np.array( edges )[ goodEdgeMask ]
        _newV = np.array( newV )[ goodEdgeMask ]

        V_row, V_col, V_data = V

        dataIndices = np.in1d( V_row, _nodes ) & np.in1d( V_col, _edges )

        fbsSize = len( self.fbs )

        if( np.any( dataIndices ) ):
            if( fbsSize == 0 ):
                V_data[ dataIndices, : ] = _newV
            else:
                V_data[ dataIndices, :, 0: fbsSize + 1 ] = _newV

    ######################################################################
