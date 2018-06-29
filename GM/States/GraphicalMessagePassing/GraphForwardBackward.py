from GenModels.GM.States.GraphicalMessagePassing.GraphicalMessagePassingBase import *
from GenModels.GM.States.GraphicalMessagePassing.GraphFilterBase import *
import numpy as np
from functools import reduce
from scipy.sparse import coo_matrix
from collections import Iterable
from GenModels.GM.Utility import fbsData

from GenModels.GM.Distributions import Normal

__all__ = [ 'GraphCategoricalForwardBackward',
            'GraphCategoricalForwardBackwardFBS' ]

######################################################################

class _fowardBackwardMixin():

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

        return U, ( V_row, V_col, V_data )

    ######################################################################

    def updateParamsFromGraphs( self, ys, initialDist, transDists, emissionDist, graphs ):
        super( _fowardBackwardMixin, self ).updateParamsFromGraphs( graphs )
        self.K = initialDist.shape[ 0 ]
        assert initialDist.ndim == 1
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

        _L = np.log( emissionDist )

        if( not isinstance( ys, np.ndarray ) ):
            ys = np.array( ys )

        self.L = np.array( [ _L[ :, y ] for y in ys ] ).sum( axis=0 ).T

    ######################################################################

    def transitionProb( self, child ):
        parents, parentOrder = self.parents( child, getOrder=True )
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
        emission = self.emissionProb( node )
        newU = self.multiplyTerms( terms=( emission, initialDist ) )
        return newU

    def vBaseCase( self, node, debug=True ):
        return np.zeros( self.K )

    ######################################################################

    def updateU( self, nodes, newU, U ):

        for u, node in zip( newU, nodes ):
            U[ node ] = u

    def updateV( self, nodes, edges, newV, V ):

        V_row, V_col, V_data = V

        for node, edge, v in zip( nodes, edges, newV ):
            if( edge is None ):
                continue

            dataIndices = np.in1d( V_row, node ) & np.in1d( V_col, edge )

            for i, maskValue in enumerate( dataIndices ):

                # Don't convert V_data to an np.array even though it makes this
                # step faster because it messes up when we add fbs nodes
                if( maskValue == True ):
                    V_data[ i ] = v

######################################################################

class GraphCategoricalForwardBackward( _fowardBackwardMixin, GraphFilter ):
    pass

######################################################################

class GraphCategoricalForwardBackwardFBS( _fowardBackwardMixin, GraphFilterFBS ):

    def genFilterProbs( self ):

        # Initialize U and V
        U = []
        for node in self.nodes:
            U.append( fbsData( np.zeros( self.K ), -1 ) )

        V_row = self.pmask.row
        V_col = self.pmask.col
        V_data = []
        for node in self.pmask.row:
            V_data.append( fbsData( np.zeros( self.K ), -1 ) )

        # Invalidate all data elements
        for node in self.nodes:
            U[ node ][ : ] = np.nan
            self.assignV( ( V_row, V_col, V_data ), node, np.nan, keepShape=True )

        return U, ( V_row, V_col, V_data )

    ######################################################################

    def transitionProb( self, node ):
        parents, parentOrder = self.full_parents( node, getOrder=True )
        ndim = len( parents ) + 1
        pi = self.pis[ ndim ]
        # Reshape pi's axes to match parent order
        assert len( parents ) + 1 == pi.ndim
        assert parentOrder.shape[ 0 ] == parents.shape[ 0 ]
        pi = np.moveaxis( pi, np.arange( ndim ), np.hstack( ( parentOrder, ndim - 1 ) ) )

        fbsOffset = lambda x: self.fbsIndex( x, fromReduced=True, withinGraph=True ) + 1

        # Check if there are nodes in [ node, *parents ] that are in the fbs.
        # If there are, then move their axes
        fbsIndices = [ fbsOffset( parent ) for parent in parents if self.inFBS( parent, fromReduced=True ) ]
        if( self.inFBS( node, fromReduced=True ) ):
            fbsIndices.append( fbsOffset( node ) )

        if( len( fbsIndices ) > 0 ):
            expandBy = max( fbsIndices )
            for _ in range( expandBy ):
                pi = pi[ ..., None ]

            # If there are parents in the fbs, move them to the appropriate axes
            for i, parent in enumerate( parents ):
                if( self.inFBS( parent, fromReduced=True ) ):
                    pi = np.swapaxes( pi, i, fbsOffset( parent ) + ndim - 1 )

            if( self.inFBS( node, fromReduced=True ) ):
                # If the node is in the fbs, then move it to the appropriate axis
                pi = np.swapaxes( pi, ndim - 1, fbsOffset( node ) + ndim - 1 )

            return fbsData( pi, ndim )
        return fbsData( pi, -1 )

    ######################################################################

    def emissionProb( self, node, forward=False ):
        prob = self.L[ node ].reshape( ( -1, ) )
        if( self.inFBS( node, fromReduced=True ) ):
            return fbsData( prob, 0 )
        return fbsData( prob, -1 )

    ######################################################################

    @classmethod
    def multiplyTerms( cls, terms, useSuper=False ):
        if( useSuper ):
            return super().multiplyTerms( terms )
        # Basically np.einsum but in log space

        assert isinstance( terms, Iterable )
        for t in terms:
            assert isinstance( t, fbsData ), t

        # Remove the empty terms
        terms = [ t for t in terms if np.prod( t.shape ) > 1 ]

        if( len( terms ) == 0 ):
            return fbsData( np.array( [] ), 0 )

        # Separate out where the feedback set axes start and get the largest fbsAxis.
        # Need to handle case where ndim of term > all fbs axes
        # terms, fbsAxesStart = list( zip( *terms ) )
        fbsAxesStart = [ term.fbsAxis for term in terms ]
        terms = [ term.data for term in terms ]

        if( max( fbsAxesStart ) != -1 ):
            maxFBSAx = max( [ ax if ax != -1 else term.ndim for ax, term in zip( fbsAxesStart, terms ) ] )

            if( maxFBSAx > 0 ):
                # Pad extra dims at each term so that the fbs axes start the same way for every term
                for i, ax in enumerate( fbsAxesStart ):
                    if( ax == -1 ):
                        for _ in range( maxFBSAx - terms[ i ].ndim + 1 ):
                            terms[ i ] = terms[ i ][ ..., None ]
                    else:
                        for _ in range( maxFBSAx - ax ):
                            terms[ i ] = np.expand_dims( terms[ i ], axis=ax )
        else:
            maxFBSAx = -1

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

        return fbsData( ans, maxFBSAx )

    ######################################################################

    @classmethod
    def integrate( cls, integrand, axes, ignoreFBSAxis=False, useSuper=False ):
        if( useSuper == True ):
            return super().integrate( integrand, axes )
        # Need adjusted axes because the relative axes in integrand change as we reduce
        # over each axis
        assert isinstance( axes, Iterable )
        if( len( axes ) == 0 ):
            if( ignoreFBSAxis is True ):
                return integrand.data
            return integrand

        integrand, fbsAxisStart = ( integrand.data, integrand.fbsAxis )

        assert max( axes ) < integrand.ndim
        axes = np.array( axes )
        axes[ axes < 0 ] = integrand.ndim + axes[ axes < 0 ]
        adjustedAxes = np.array( sorted( axes ) ) - np.arange( len( axes ) )
        for ax in adjustedAxes:
            integrand = np.logaddexp.reduce( integrand, axis=ax )

        if( ignoreFBSAxis is True ):
            return integrand

        if( fbsAxisStart > -1 ):
            fbsAxisStart -= len( adjustedAxes )
            # assert fbsAxisStart > -1, adjustedAxes

        return fbsData( integrand, fbsAxisStart )

    ######################################################################

    def uBaseCase( self, node ):
        initialDist = fbsData( self.pi0, -1 )
        emission = self.emissionProb( node )
        return self.multiplyTerms( terms=( emission, initialDist ) )

    def vBaseCase( self, node ):
        return fbsData( np.zeros( self.K ), -1 )

