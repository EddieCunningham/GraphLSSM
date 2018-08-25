import numpy as np
from GenModels.GM.Utility import fbsData
from collections import namedtuple
from .GraphFilterBase import GraphFilterFBS
import numba

__all__ = [ 'GraphFilterFBSParallel' ]

class Node( namedtuple( 'CachedNodeInfo', [ 'partial_parents',
                                            'partial_parents_order',
                                            'partial_parents_in_fbs',
                                            'partial_parents_in_fbs_order',
                                            'partial_parents_in_fbs_fbs_index',
                                            'full_parents',
                                            'full_parents_order',
                                            'partial_parent_us',
                                            'partial_parent_vs',

                                            'partial_mates',
                                            'partial_mates_order',
                                            'partial_mate_us',
                                            'partial_mate_vs',


                                            'full_siblings',
                                            'full_sibling_vs',
                                            'siblings_shape_corrected_transition_distribution',
                                            'siblings_shape_corrected_emission_distribution',
                                            'full_siblings_in_fbs',

                                            'full_children',
                                            'full_children_vs',
                                            'children_shape_corrected_transition_distribution',
                                            'children_shape_corrected_emission_distribution',
                                            'full_children_in_fbs'

                                            'up_edge',
                                            'down_edges',

                                            'u',
                                            'v',
                                            'in_fbs',
                                            'fbs_index',
                                            'possible_latent_states',
                                            'shape_corrected_transition_distribution',
                                            'shape_corrected_emission_distribution' ] ) ):

    # The partial graph index will be used everywhere
    __slots__ = ()

######################################################################

def multiplyTerms( terms ):
    # Basically np.einsum but in log space

    assert isinstance( terms, Iterable )

    # Check if we should use the multiply for fbsData or for regular data
    fbs_data_count, non_fbs_data_count = ( 0, 0 )
    for t in terms:
        if( isinstance( t, fbsData ) ):
            fbs_data_count += 1
        else:
            non_fbs_data_count += 1

    # Can't mix types
    if( not ( fbs_data_count == 0 or non_fbs_data_count == 0 ) ):
        print( 'fbs_data_count', fbs_data_count )
        print( 'non_fbs_data_count', non_fbs_data_count )
        print( terms )
        for t in terms:
            if( isinstance( t, fbsData ) ):
                print( 'this ones good', t, type( t ) )
            else:
                print( 'this ones bad', t, type( t ) )
        assert 0

    # This parallel version expects there to always be fbs data
    if( fbs_data_count == 0 ):
        assert 0

    # Remove the empty terms
    terms = [ t for t in terms if np.prod( t.shape ) > 1 ]

    if( len( terms ) == 0 ):
        return fbsData( np.array( [] ), 0 )

    # Separate out where the feedback set axes start and get the largest fbs_axis.
    # Need to handle case where ndim of term > all fbs axes
    # terms, fbs_axes_start = list( zip( *terms ) )
    fbs_axes_start = [ term.fbs_axis for term in terms ]
    terms = [ term.data for term in terms ]

    if( max( fbs_axes_start ) != -1 ):
        max_fbs_axis = max( [ ax if ax != -1 else term.ndim for ax, term in zip( fbs_axes_start, terms ) ] )

        if( max_fbs_axis > 0 ):
            # Pad extra dims at each term so that the fbs axes start the same way for every term
            for i, ax in enumerate( fbs_axes_start ):
                if( ax == -1 ):
                    for _ in range( max_fbs_axis - terms[ i ].ndim + 1 ):
                        terms[ i ] = terms[ i ][ ..., None ]
                else:
                    for _ in range( max_fbs_axis - ax ):
                        terms[ i ] = np.expand_dims( terms[ i ], axis=ax )
    else:
        max_fbs_axis = -1

    ndim = max( [ len( term.shape ) for term in terms ] )

    axes = [ [ i for i, s in enumerate( t.shape ) if s != 1 ] for t in terms ]

    # Get the shape of the output
    shape = np.ones( ndim, dtype=int )
    for ax, term in zip( axes, terms ):
        shape[ np.array( ax ) ] = term.squeeze().shape

    total_elts = shape.prod()
    if( total_elts > 1e8 ):
        assert 0, 'Don\'t do this on a cpu!  Too many terms: %d'%( int( total_elts ) )

    # Build a meshgrid out of each of the terms over the right axes
    # and sum.  Doing it this way because np.einsum doesn't work
    # for matrix multiplication in log space - we can't do np.einsum
    # but add instead of multiply over indices
    ans = np.zeros( shape )
    for ax, term in zip( axes, terms ):

        for _ in range( ndim - term.ndim ):
            term = term[ ..., None ]

        ans += np.broadcast_to( term, ans.shape )

    return fbsData( ans, max_fbs_axis )

######################################################################

def integrate( integrand, axes ):

    if( not isinstance( integrand, fbsData ) ):
        assert 0

    # Need adjusted axes because the relative axes in integrand change as we reduce
    # over each axis
    assert isinstance( axes, Iterable )
    if( len( axes ) == 0 ):
        return integrand

    integrand, fbs_axis = ( integrand.data, integrand.fbs_axis )

    assert max( axes ) < integrand.ndim
    axes = np.array( axes )
    axes[ axes < 0 ] = integrand.ndim + axes[ axes < 0 ]
    adjusted_axes = np.array( sorted( axes ) ) - np.arange( len( axes ) )
    for ax in adjusted_axes:
        integrand = np.logaddexp.reduce( integrand, axis=ax )

    if( fbs_axis > -1 ):
        fbs_axis -= len( adjusted_axes )

    return fbsData( integrand, fbs_axis )

######################################################################

def extendAxes( term, target_axis, max_dim ):
    # Push the first axis out to target_axis, but don't change
    # the axes past max_dim
    term, fbs_axis = ( term.data, term.fbs_axis )

    # Add axes before the fbsAxes
    for _ in range( max_dim - target_axis - 1 ):
        term = np.expand_dims( term, 1 )
        if( fbs_axis != -1 ):
            fbs_axis += 1

    # Prepend axes
    for _ in range( target_axis ):
        term = np.expand_dims( term, 0 )
        if( fbs_axis != -1 ):
            fbs_axis += 1

    return fbsData( term, fbs_axis )

######################################################################

def uWork( node_obj ):

    n_parents = node_obj.full_parents.shape[ 0 ]

    # A over each parent (aligned according to ordering of parents)
    parent_as = [ None for _ in node_obj.partial_parents ]
    for i, o in enumerate( node_obj.partial_parents_order ):
        u = node_obj.partial_parent_us[ i ]
        vs = [ v for e, v in node_obj.partial_parent_vs[ i ] if e != node_obj.up_edge ]
        term = multiplyTerms( terms=( u, *vs ) )
        assert term.size > 0
        parent_as[ i ] = extendAxes( term, o, n_parents )

    # B over each sibling
    sibling_bs = [ None for _ in node_obj.full_siblings ]
    for i, s in enumerate( node_obj.full_siblings ):
        transition = node_obj.siblings_shape_corrected_transition_distribution[ i ]
        emission = node_obj.siblings_shape_corrected_emission_distribution[ i ]
        if( node_obj.full_siblings_in_fbs[ i ] == True ):
            term = multiplyTerms( terms=( transition, emission ) )
        else:
            vs = [ extendAxes( node_obj.full_sibling_vs[ i ], n_parents, n_parents + 1 ) ]
            integrand = multiplyTerms( terms=( transition, emission, *vs ) )
            term = integrate( integrand, axes=[ n_parents ] )
        sibling_bs[ i ] = term

    # Multiply all of the terms together
    # Integrate out the parent latent states
    integrand = multiplyTerms( terms=( node_obj.shape_corrected_transition_distribution, *parent_as, *sigling_bs ) )
    node_terms = integrate( integrand, axes=node_obj.partial_parents_order )

    # Squeeze all of the left most dims.  This is because if a parent is in
    # the fbs and we integrate out the other one, there will be an empty dim
    # remaining
    while( node_terms.shape[ 0 ] == 1 ):
        node_terms = node_terms.squeeze( axis=0 )

    # Emission for this node
    node_emission = node_obj.shape_corrected_emission_distribution

    # Combine this nodes emission with the rest of the calculation
    return multiplyTerms( terms=( node_terms, node_emission ) )

######################################################################

def vWork( node_obj, edge ):

    n_parents = node_obj.full_parents.shape[ 0 ]

    # A values over each mate (aligned according to ordering of mates)
    mate_as = [ None for _ in node_obj.partial_mates[ edge ] ]
    for i, o in enumerate( node_obj.partial_mates_order[ edge ] ):
        u = node_obj.partial_mate_us[ i ]
        vs = [ v for e, v in node_obj.partial_mate_vs[ edge ][ i ] if e != node_obj.up_edge ]
        term = multiplyTerms( terms=( u, *vs ) )
        assert term.size > 0
        mate_as[ i ] = extendAxes( term, o, n_parents )

    # B over each child
    child_bs = [ None for _ in node_obj.full_children ]
    for i, s in enumerate( node_obj.full_children ):
        transition = node_obj.children_shape_corrected_transition_distribution[ edge ][ i ]
        emission = node_obj.children_shape_corrected_emission_distribution[ edge ][ i ]
        if( node_obj.full_children_in_fbs[ edge ][ i ] == True ):
            term = multiplyTerms( terms=( transition, emission ) )
        else:
            vs = [ extendAxes( node_obj.full_children_vs[ edge ][ i ], n_parents, n_parents + 1 ) ]
            integrand = multiplyTerms( terms=( transition, emission, *vs ) )
            term = integrate( integrand, axes=[ n_parents ] )
        child_bs[ i ] = term

    # Multiply all of the terms together
    integrand = multiplyTerms( terms=( *child_bs, *mate_as ) )

    # Integrate out the mates latent states
    ans = integrate( integrand, axes=mate_order )

    # Squeeze all of the left most dims.  This is because if a parent is in
    # the fbs and we integrate out the other one, there will be an empty dim
    # remaining
    while( ans.shape[ 0 ] == 1 ):
        ans = ans.squeeze( axis=0 )

    return ans

######################################################################

class GraphFilterFBSParallel( GraphFilterFBS ):

    def gatherLocalInfo( self, node ):
        # Create the Node object for each node
        partial_parents

    def uFilter( self, is_base_case, nodes, U, V, parallel=True ):
        if( parallel == False ):
            return super().uFilter( is_base_case, nodes, U, V )

        new_u = []
        for node in nodes:
            if( is_base_case ):
                u = self.uBaseCase( node )
            else:
                u = self.u( U, V, node )
            new_u.append( u )

        self.updateU( nodes, new_u, U )

    def vFilter( self, is_base_case, nodes_and_edges, U, V, parallel=True ):
        if( parallel == False ):
            return super().vFilter( is_base_case, nodes_and_edges, U, V )

        nodes, edges = nodes_and_edges

        new_v = []
        for node, edge in zip( nodes, edges ):
            if( is_base_case ):
                assert edge == None
                v = self.vBaseCase( node )
            else:
                assert edge is not None
                v = self.v( U, V, node, edge )
            new_v.append( v )

        self.updateV( nodes, edges, new_v, V )
