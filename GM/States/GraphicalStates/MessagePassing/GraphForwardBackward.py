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

class GraphCategoricalForwardBackward( GraphFilter ):
    # Assumes that the number of parents per child is fixed to N
    def __init__( self, K, N ):
        super( GraphCategoricalForwardBackward, self ).__init__()
        self.K = K
        self.N = N

    def genFilterProbs( self ):
        U = np.zeros( ( self.nodes.shape[ 0 ], self.K ) )
        V_row = self.pmask.row
        V_col = self.pmask.col
        V_data = np.zeros( ( self.K, self.pmask.row.shape[ 0 ] ) )
        return U, ( V_row, V_col, V_data )

    def genWorkspace( self ):
        return np.array( [] )

    ######################################################################

    def updateParams( self, ys, initialDist, transDist, emissionDist, parentMasks, childMasks, feedbackSets=None ):
        super( GraphFilter, self ).updateParams( parentMasks, childMasks, feedbackSets=feedbackSets )
        assert initialDist.shape == ( self.K, )
        assert transDist.shape == tuple( [ self.K ] * ( self.N + 1 ) ), transDist.shape
        assert emissionDist.shape[ 0 ] == self.K
        assert np.isclose( 1.0, initialDist.sum() )
        assert np.allclose( np.ones( self.K ), transDist.sum( axis=-1 ) )
        assert np.allclose( np.ones( self.K ), emissionDist.sum( axis=1 ) )
        self.pi0 = np.log( initialDist )
        self.pi  = np.log( transDist )
        _L = np.log( emissionDist )

        if( not isinstance( ys, np.ndarray ) ):
            ys = np.array( ys )

        print('_L', _L)

        self.L = np.array( [ _L[ :, y ] for y in ys ] ).sum( axis=0 ).T

    ######################################################################

    def transitionProb( self, children, parents ):
        return self.pi

    ######################################################################

    def emissionProb( self, node, forward=False ):
        return self.L[ node ].reshape( ( -1, ) )

    ######################################################################

    def _multiply( self, axisTerms, overAll=None, axes=None ):
        # First term will be applied over all axes
        # Every term after that will be outer product-ed over a new axis.
        # Because we're working in log space, its like doing np.einsum
        # except adding instead of multiplying
        assert axes is not None
        print( axes )
        print( axisTerms )
        assert len( axes ) == len( axisTerms )
        if( len( axisTerms ) == 0 ):
            return np.array( [] )

        ndim = max( axes ) + 1
        print( axisTerms )
        print(ndim)

        # First sum over all terms that have the same axis
        uniqueAxisTerms = [ np.array( [] ) for _ in range( ndim ) ]
        for ax, term in zip( axes, axisTerms ):
            if( term.size == 0 ):
                continue
            if( uniqueAxisTerms[ ax ].size == 0 ):
                uniqueAxisTerms[ ax ] = term
            else:
                uniqueAxisTerms[ ax ] += term

        ans = np.array( sum( np.meshgrid( *uniqueAxisTerms, indexing='ij' ) ) )
        if( ans.size == 0 ):
            return ans
        # print( 'ans', ans )
        # print( 'overAll', overAll )
        return ans if overAll is None else np.sum( [ ans ] + [ o for o in overAll if o.size > 0 ] )

    def multiplyTerms( self, terms, axes=None ):

        # print( 'Axes:\n' )
        # for ax in axes:
        #     print( ax )
        #     print('---')
        # print(' Terms:\n')
        # for term in terms:
        #     print( term )
        #     print('---')

        assert axes is not None
        assert isinstance( terms, Iterable ) and isinstance( axes, Iterable )
        assert len( axes ) == len( terms )

        # Get the number of dimensions
        ndim = 0
        for ax in axes:
            for _ax in ax:
                ndim = max( _ax + 1, ndim )

        # print( 'ndim', ndim )

        # Reshape each of the inputs to have its elements over the
        # correct axes
        relevantAxes = []
        reshapedTerms = []
        for ax, term in zip( axes, terms ):
            assert isinstance( ax, tuple )

            if( term.size == 0 ):
                continue

            # The new shape moves the data of term
            # to the axes provided in ax
            newShape = np.ones( ndim, dtype=int )
            for _ax, s in zip( ax, term.shape ):
                newShape[ _ax ] = s

            # print( 'reshaping from', term.shape, 'to', newShape )

            relevantAxes.append( ax )
            reshapedTerms.append( term.reshape( newShape ) )

        # print( 'reshapedTerms', reshapedTerms)

        # Get the shape of the output
        shape = -1 * np.ones( ndim, dtype=int )
        for ax, term in zip( relevantAxes, reshapedTerms ):
            for _ax in ax:
                if( shape[ _ax ] == -1 ):
                    shape[ _ax ] = term.shape[ _ax ]
                else:
                    if( shape[ _ax ] != term.shape[ _ax ] ):
                        print( '_ax', _ax )
                        print( 'shape[ _ax ]', shape[ _ax ] )
                        print( 'term.shape[ _ax ]', term.shape[ _ax ] )
                        assert 0

        # print( 'shape', shape )

        # Build a meshgrid out of each of the terms over the right axes
        # and sum.  Doing it this way because np.einsum doesn't work
        # for matrix multiplication in log space - we can't do np.einsum
        # but add instead of multiply over indices
        ans = np.zeros( ( shape ) )
        for ax, term in zip( relevantAxes, reshapedTerms ):
            reps = np.copy( shape )
            for _ax in ax:
                reps[ _ax ] = 1
            ans += np.tile( term, reps )

        # print( 'ans', ans )

        return ans

    ######################################################################

    def integrate( self, integrand, axes=None ):
        assert axes is not None
        axes = np.array( axes )

        if( integrand.size == 0 ):
            ans.append( np.array( [] ) )
        else:
            # Make sure we integrate over the correct axes at each step
            axes[ axes < 0 ] += len( integrand.shape )
            axes = np.array( sorted( axes ) ) - np.arange( len( axes ) )
            for ax in axes:
                integrand = np.logaddexp.reduce( integrand, axis=ax )
        return integrand

    ######################################################################

    def uBaseCase( self, roots, U, conditioning, workspace, debug=False ):
        dprint( '\n\nComputing base case U for', roots, use=debug )

        initialDist = self.pi0
        dprint( 'initialDist:', initialDist, use=debug )

        emission = [ self.emissionProb( r ) for r in roots ]
        dprint( 'emission:', emission, use=debug )

        newU = [ self.multiplyTerms( ( e, initialDist ), axes=( ( 0, ), ( 0, ) ) ) for e in emission ]
        dprint( 'newU:', newU, use=debug )

        return newU

    def vBaseCase( self, leaves, V, conditioning, workspace ):
        return

    ######################################################################

    def condition( self, nodes ):
        pass

    ######################################################################

    def updateU( self, nodes, newU, U, conditioning ):
        for u, node in zip( newU, nodes ):
            U[ node ] = u

    def updateV( self, nodes, edges, newV, V, conditioning ):
        goodEdgeMask = ~np.in1d( np.array( edges ), None )

        _nodes = np.array( nodes )[ goodEdgeMask ]
        _edges = np.array( edges )[ goodEdgeMask ]
        _newV = np.array( newV )[ goodEdgeMask ]

        V_row, V_col, V_data = V

        dataIndices = np.in1d( V_row, _nodes ) & np.in1d( V_col, _edges )

        if( np.any( dataIndices ) ):
            V_data[ dataIndices ] = _newV

    ######################################################################

    def integrateOutConditioning( self, U, V, conditioning, workspace ):
        return

    def filterCutNodes( self, U, V, conditioning, workspace ):
        return

    ######################################################################
