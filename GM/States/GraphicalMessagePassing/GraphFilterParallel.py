import itertools
from functools import partial, lru_cache
import numpy as np
from GenModels.GM.Utility import fbsData
from collections import namedtuple, Iterable
from .GraphFilterBase import GraphFilterFBS
import joblib
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool
from threading import Lock
# NUMBA ISN'T WORKING!!!
# import numba

__all__ = [ 'GraphFilterFBSParallel' ]

######################################################################

UBaseNodeData = namedtuple( 'UBaseNodeData', [ 'shape_corrected_initial_distribution',
                                               'shape_corrected_emission_distribution' ] )

UNodeData = namedtuple( 'UNodeData', [ 'full_n_parents',
                                       'up_edge',
                                       'shape_corrected_transition_distribution',
                                       'shape_corrected_emission_distribution',
                                       'partial_parents',
                                       'partial_parents_order',
                                       'partial_parent_us',
                                       'partial_parent_vs',
                                       'full_siblings',
                                       'full_siblings_in_fbs',
                                       'full_sibling_vs',
                                       'siblings_shape_corrected_transition_distribution',
                                       'siblings_shape_corrected_emission_distribution' ] )

VNodeData = namedtuple( 'VNodeData', [ 'full_n_parents',
                                       'partial_mates',
                                       'partial_mates_order',
                                       'partial_mate_us',
                                       'partial_mate_vs',
                                       'full_children',
                                       'full_children_in_fbs',
                                       'full_children_vs',
                                       'children_shape_corrected_transition_distribution',
                                       'children_shape_corrected_emission_distribution' ] )

NodeJointInFBSDataIntegrate = namedtuple( 'NodeJointInFBSDataIntegrate', [ 'in_fbs', 'integrate_method', 'fbs_index', 'u', 'vs' ] )
NodeJointNotInFBSData = namedtuple( 'NodeJointNotInFBSData', [ 'in_fbs', 'u', 'vs' ] )
NodeJointInFBSData = namedtuple( 'NodeJointInFBSData', [ 'in_fbs',
                                                         'integrate_method',
                                                         'fbs_index',
                                                         'full_n_parents',
                                                         'up_edge',
                                                         'shape_corrected_transition_distribution',
                                                         'shape_corrected_emission_distribution',
                                                         'partial_parents',
                                                         'partial_parents_order',
                                                         'partial_parent_us',
                                                         'partial_parent_vs',
                                                         'full_siblings',
                                                         'full_siblings_in_fbs',
                                                         'full_sibling_vs',
                                                         'siblings_shape_corrected_transition_distribution',
                                                         'siblings_shape_corrected_emission_distribution' ] )

JointParentsData = namedtuple( 'JointParentsData', [ 'all_parents_in_fbs',
                                                     'full_siblings',
                                                     'full_siblings_in_fbs',
                                                     'full_sibling_vs',
                                                     'siblings_shape_corrected_transition_distribution',
                                                     'siblings_shape_corrected_emission_distribution',
                                                     'full_n_parents',
                                                     'shape_corrected_transition_distribution',
                                                     'shape_corrected_emission_distribution',
                                                     'vs',
                                                     'partial_parents',
                                                     'partial_parents_order',
                                                     'partial_parent_us',
                                                     'partial_parent_vs',
                                                     'parent_fbs_indices',
                                                     'parent_not_in_fbs_order',
                                                     'parent_in_fbs_order' ] )
JointParentsAllParentsInFBSData = namedtuple( 'JointParentsAllParentsInFBSData', [ 'all_parents_in_fbs',
                                                                'u',
                                                                'vs',
                                                                'parent_fbs_indices',
                                                                'parent_not_in_fbs_order',
                                                                'parent_in_fbs_order' ] )

JointParentChildData = namedtuple( 'JointParentChildData', [ 'in_fbs',
                                                     'fbs_index',
                                                     'all_parents_in_fbs',
                                                     'full_siblings',
                                                     'full_siblings_in_fbs',
                                                     'full_sibling_vs',
                                                     'siblings_shape_corrected_transition_distribution',
                                                     'siblings_shape_corrected_emission_distribution',
                                                     'full_n_parents',
                                                     'shape_corrected_transition_distribution',
                                                     'shape_corrected_emission_distribution',
                                                     'vs',
                                                     'partial_parents',
                                                     'partial_parents_order',
                                                     'partial_parent_us',
                                                     'partial_parent_vs',
                                                     'parent_fbs_indices',
                                                     'parent_not_in_fbs_order',
                                                     'parent_in_fbs_order' ] )
JointParentChildInFBSData = namedtuple( 'JointParentChildInFBSData', [ 'in_fbs',
                                                                       'fbs_index',
                                                                       'full_n_parents',
                                                                       'all_parents_in_fbs',
                                                                       'u',
                                                                       'vs',
                                                                       'parent_fbs_indices',
                                                                       'parent_not_in_fbs_order',
                                                                       'parent_in_fbs_order' ] )


######################################################################

def nonFBSMultiplyTerms( terms ):
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

    return ans

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

def nonFBSIntegrate( integrand, axes ):
    # Need adjusted axes because the relative axes in integrand change as we reduce
    # over each axis
    assert isinstance( axes, Iterable )
    if( len( axes ) == 0 ):
        return integrand

    assert max( axes ) < integrand.ndim
    axes = np.array( axes )
    axes[ axes < 0 ] = integrand.ndim + axes[ axes < 0 ]
    adjusted_axes = np.array( sorted( axes ) ) - np.arange( len( axes ) )
    for ax in adjusted_axes:
        integrand = np.logaddexp.reduce( integrand, axis=ax )

    return integrand

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

def aWork( full_n_parents, u, vs, order ):
    term = multiplyTerms( terms=( u, *vs ) )
    return extendAxes( term, order, full_n_parents )

def bWorkFBS( transition, emission ):
    return multiplyTerms( terms=( transition, emission ) )

def bWork( full_n_parents, transition, emission, vs ):
    emission = extendAxes( emission, full_n_parents, full_n_parents + 1 )
    vs = [ extendAxes( v, full_n_parents, full_n_parents + 1 ) for v in vs ]
    integrand = multiplyTerms( terms=( transition, emission, *vs ) )
    return integrate( integrand, axes=[ full_n_parents ] )

######################################################################

def uWork( node_data ):

    # A over each parent (aligned according to ordering of parents)
    parent_as = [ None for _ in node_data.partial_parents ]
    for i, order in enumerate( node_data.partial_parents_order ):
        parent_as[ i ] = aWork( node_data.full_n_parents,
                                node_data.partial_parent_us[ i ],
                                node_data.partial_parent_vs[ i ],
                                order )
    # B over each sibling
    sibling_bs = [ None for _ in node_data.full_siblings ]
    for i, s in enumerate( node_data.full_siblings ):
        if( node_data.full_siblings_in_fbs[ i ] == True ):
            term = bWorkFBS( node_data.siblings_shape_corrected_transition_distribution[ i ],
                             node_data.siblings_shape_corrected_emission_distribution[ i ] )
        else:
            term = bWork( node_data.full_n_parents,
                          node_data.siblings_shape_corrected_transition_distribution[ i ],
                          node_data.siblings_shape_corrected_emission_distribution[ i ],
                          node_data.full_sibling_vs[ i ] )
        sibling_bs[ i ] = term

    # Multiply all of the terms together
    # Integrate out the parent latent states
    integrand = multiplyTerms( terms=( node_data.shape_corrected_transition_distribution, *parent_as, *sibling_bs ) )
    node_terms = integrate( integrand, axes=node_data.partial_parents_order )

    # Squeeze all of the left most dims.  This is because if a parent is in
    # the fbs and we integrate out the other one, there will be an empty dim
    # remaining
    while( node_terms.shape[ 0 ] == 1 ):
        node_terms = node_terms.squeeze( axis=0 )

    # Emission for this node
    node_emission = node_data.shape_corrected_emission_distribution

    # Combine this nodes emission with the rest of the calculation
    return multiplyTerms( terms=( node_terms, node_emission ) )

######################################################################

def vWork( node_data ):

    # A values over each mate (aligned according to ordering of mates)
    mate_as = [ None for _ in node_data.partial_mates ]
    for i, order in enumerate( node_data.partial_mates_order ):
        mate_as[ i ] = aWork( node_data.full_n_parents,
                              node_data.partial_mate_us[ i ],
                              node_data.partial_mate_vs[ i ],
                              order )
    # B over each child
    child_bs = [ None for _ in node_data.full_children ]
    for i, s in enumerate( node_data.full_children ):

        if( node_data.full_children_in_fbs[ i ] == True ):
            term = bWorkFBS( node_data.children_shape_corrected_transition_distribution[ i ],
                             node_data.children_shape_corrected_emission_distribution[ i ] )
        else:
            term = bWork( node_data.full_n_parents,
                          node_data.children_shape_corrected_transition_distribution[ i ],
                          node_data.children_shape_corrected_emission_distribution[ i ],
                          node_data.full_children_vs[ i ] )
        child_bs[ i ] = term

    # Multiply all of the terms together
    # Integrate out the mates latent states
    integrand = multiplyTerms( terms=( *child_bs, *mate_as ) )
    ans = integrate( integrand, axes=node_data.partial_mates_order )

    # Squeeze all of the left most dims.  This is because if a parent is in
    # the fbs and we integrate out the other one, there will be an empty dim
    # remaining
    while( ans.shape[ 0 ] == 1 ):
        ans = ans.squeeze( axis=0 )

    return ans

######################################################################

def uBaseCaseWork( node_data ):
    return multiplyTerms( terms=( node_data.shape_corrected_initial_distribution, node_data.shape_corrected_emission_distribution ) )

######################################################################

def nodeJointSingleNodeComputation( node_data ):
    u = node_data.u
    vs = [ extendAxes( v, 0, 1 ) for v in node_data.vs ]
    return multiplyTerms( terms=( u, *vs ) )

def nodeJointSingleNodeNotInFBS( node_data ):
    # Compute the joint using the regular algorithm but on the partial_graph
    joint_fbs = nodeJointSingleNodeComputation( node_data )

    # Integrate the joint and return only the data portion
    int_axes = range( 1, joint_fbs.ndim )
    return integrate( joint_fbs, axes=int_axes ).data

def nodeJointSingleNodeInFBS( node_data ):
    # Compute the joint by integrating out the other fbs nodes
    joint_fbs = uWork( node_data )
    fbs_index = node_data.fbs_index
    keep_axis = fbs_index + joint_fbs.fbs_axis
    int_axes = np.setdiff1d( np.arange( joint_fbs.ndim ), keep_axis )
    return integrate( joint_fbs, axes=int_axes ).data

def nodeJointSingleNodeInFBSIntegrateMethod( node_data ):
    joint_fbs = nodeJointSingleNodeComputation( node_data )

    fbs_index = node_data.fbs_index
    keep_axis = fbs_index + joint_fbs.fbs_axis
    int_axes = np.setdiff1d( np.arange( joint_fbs.ndim ), keep_axis )
    ans = integrate( joint_fbs, axes=int_axes ).data
    return ans

def nodeJointSingleNode( node_data ):
    if( node_data.in_fbs ):
        if( node_data.integrate_method ):
            return nodeJointSingleNodeInFBSIntegrateMethod( node_data )
        return nodeJointSingleNodeInFBS( node_data )
    return nodeJointSingleNodeNotInFBS( node_data )

######################################################################

def integrateForJointParent( node_data, joint_fbs ):

    # Loop through the parents and find the axes that they are on so we can keep them
    keep_axes = [ joint_fbs.fbs_axis + i for i in node_data.parent_fbs_indices ] + node_data.parent_not_in_fbs_order

    # Integrate out all fbs nodes that aren't the parents
    int_axes = np.setdiff1d( np.arange( joint_fbs.ndim ), keep_axes )
    ans = integrate( joint_fbs, axes=int_axes ).data

    # Swap the order of the axes so that the answer has the correct parent order.
    fbs_parents = [ ( i, o ) for i, o in zip( node_data.parent_fbs_indices, node_data.parent_in_fbs_order ) ]
    fbs_order = sorted( fbs_parents, key=lambda x: x[ 0 ] )
    if( len( fbs_order ) > 0 ):
        fbs_order = list( list( zip( *fbs_order ) )[ 1 ] )

    true_order = node_data.parent_not_in_fbs_order + fbs_order
    transpose_axes = [ true_order.index( i ) for i in range( len( true_order ) ) ]

    return np.transpose( ans, transpose_axes )

def jointParentsSingleNode( node_data ):
    if( node_data.all_parents_in_fbs ):
        joint_fbs = nodeJointSingleNodeComputation( node_data )
    else:
        joint_with_child = jointParentChildSingleNodeComputation( node_data )
        joint_fbs = integrate( joint_with_child, axes=[ node_data.full_n_parents ] )
    return integrateForJointParent( node_data, joint_fbs )

######################################################################

def integrateForJointParentChild( node_data, joint_fbs ):

    # Loop through the parents and find the axes that they are on so we can keep them
    keep_axes = [ joint_fbs.fbs_axis + i for i in node_data.parent_fbs_indices ] + node_data.parent_not_in_fbs_order

    # Keep the axis that node is on
    if( node_data.in_fbs ):
        keep_axes.append( joint_fbs.fbs_axis + node_data.fbs_index )
    else:
        if( node_data.all_parents_in_fbs ):
            keep_axes.append( 0 )
        else:
            keep_axes.append( node_data.full_n_parents )

    # Integrate out all fbs nodes that aren't the parents
    int_axes = np.setdiff1d( np.arange( joint_fbs.ndim ), keep_axes )
    ans = integrate( joint_fbs, axes=int_axes ).data

    # Swap the order of the axes so that the answer has the correct parent order.
    fbs_parents = [ ( i, o ) for i, o in zip( node_data.parent_fbs_indices, node_data.parent_in_fbs_order ) ]
    fbs_order = sorted( fbs_parents, key=lambda x: x[ 0 ] )
    if( len( fbs_order ) > 0 ):
        fbs_order = list( list( zip( *fbs_order ) )[ 1 ] )

    # If node is in the fbs, then that means that its current axis is somewhere
    # after the fbs_axis.  So place the node's axis correctly
    if( node_data.in_fbs ):
        true_order = node_data.parent_not_in_fbs_order + \
                     fbs_order[ :node_data.fbs_index ] + \
                     [ node_data.full_n_parents ] + \
                     fbs_order[ node_data.fbs_index: ]
    else:
        true_order = node_data.parent_not_in_fbs_order + \
                     [ node_data.full_n_parents ] + \
                     fbs_order

    transpose_axes = [ true_order.index( i ) for i in range( len( true_order ) ) ]

    return np.transpose( ans, transpose_axes )

def jointParentChildSingleNodeComputation( node_data ):

    # Down each sibling
    sibling_bs = [ None for _ in node_data.full_siblings ]
    for i, s in enumerate( node_data.full_siblings ):
        if( node_data.full_siblings_in_fbs[ i ] == True ):
            term = bWorkFBS( node_data.siblings_shape_corrected_transition_distribution[ i ],
                             node_data.siblings_shape_corrected_emission_distribution[ i ] )
        else:
            term = bWork( node_data.full_n_parents,
                          node_data.siblings_shape_corrected_transition_distribution[ i ],
                          node_data.siblings_shape_corrected_emission_distribution[ i ],
                          node_data.full_sibling_vs[ i ] )
        sibling_bs[ i ] = term

    # Down this node, but don't integrate out this node
    emission = extendAxes( node_data.shape_corrected_emission_distribution,
                           node_data.full_n_parents,
                           node_data.full_n_parents + 1 )
    vs = [ extendAxes( v, node_data.full_n_parents, node_data.full_n_parents + 1 ) for v in node_data.vs ]
    node_term = multiplyTerms( terms=( node_data.shape_corrected_transition_distribution, emission, *vs ) )

    # Out from each parent
    parent_as = [ None for _ in node_data.partial_parents ]
    for i, order in enumerate( node_data.partial_parents_order ):
        parent_as[ i ] = aWork( node_data.full_n_parents,
                                node_data.partial_parent_us[ i ],
                                node_data.partial_parent_vs[ i ],
                                order )

    return multiplyTerms( terms=( node_term, *parent_as, *sibling_bs ) )

def jointParentChildSingleNode( node_data ):
    if( node_data.all_parents_in_fbs ):
        joint_fbs = nodeJointSingleNodeComputation( node_data )
    else:
        joint_fbs = jointParentChildSingleNodeComputation( node_data )
    return integrateForJointParentChild( node_data, joint_fbs )

######################################################################

def conditionalParentChildSingleNode( node_data ):
    if( node_data.all_parents_in_fbs ):
        parents_child_fbs_joint = nodeJointSingleNodeComputation( node_data )
        parents_fbs_joint = parents_child_fbs_joint
    else:
        parents_child_fbs_joint = jointParentChildSingleNodeComputation( node_data )
        parents_fbs_joint = integrate( parents_child_fbs_joint, axes=[ node_data.full_n_parents ] )

    parents_child_joint = integrateForJointParentChild( node_data, parents_child_fbs_joint )
    parents_joint = integrateForJointParent( node_data, parents_fbs_joint )

    return nonFBSMultiplyTerms( terms=( parents_child_joint, -parents_joint ) )

######################################################################
######################################################################
######################################################################

class GraphFilterFBSParallel( GraphFilterFBS ):

    def clearCache( self ):
        super().clearCache()
        self.cachedTransition.cache_clear()
        self.cachedEmission.cache_clear()

    ######################################################################

    def cleanup( self ):
        self.u_filter_process_pool.close()
        self.v_filter_process_pool.close()
        self.data_thread_pool.close()
        delattr( self, '_u_filter_process_pool' )
        delattr( self, '_v_filter_process_pool' )
        delattr( self, '_data_thread_pool' )

    def __del__( self ):
        self.cleanup()

    ######################################################################

    @property
    def u_filter_process_pool( self ):
        if( hasattr( self, '_u_filter_process_pool' ) == False ):
            self._u_filter_process_pool = Pool()
        return self._u_filter_process_pool

    @property
    def v_filter_process_pool( self ):
        if( hasattr( self, '_v_filter_process_pool' ) == False ):
            self._v_filter_process_pool = Pool()
        return self._v_filter_process_pool

    ######################################################################

    @property
    def data_thread_pool( self ):
        if( hasattr( self, '_data_thread_pool' ) == False ):
            self._data_thread_pool = ThreadPool()
        return self._data_thread_pool

    @property
    def u_filter_result( self ):
        if( hasattr( self, '_u_filter_result' ) == False ):
            self._u_filter_result = None
        return self._u_filter_result

    @u_filter_result.setter
    def u_filter_result( self, val ):
        self._u_filter_result = val

    ######################################################################

    @property
    def v_filter_result( self ):
        if( hasattr( self, '_v_filter_result' ) == False ):
            self._v_filter_result = None
        return self._v_filter_result

    @v_filter_result.setter
    def v_filter_result( self, val ):
        self._v_filter_result = val

    ######################################################################

    @property
    def last_u_nodes( self ):
        return self._last_u_nodes

    @last_u_nodes.setter
    def last_u_nodes( self, val ):
        self._last_u_nodes = val

    ######################################################################

    @property
    def last_v_nodes( self ):
        return self._last_v_nodes

    @last_v_nodes.setter
    def last_v_nodes( self, val ):
        self._last_v_nodes = val

    @property
    def last_v_edges( self ):
        return self._last_v_edges

    @last_v_edges.setter
    def last_v_edges( self, val ):
        self._last_v_edges = val

    ######################################################################

    @property
    def last_u( self ):
        return self._last_u

    @last_u.setter
    def last_u( self, val ):
        self._last_u = val

    @property
    def last_v( self ):
        return self._last_v

    @last_v.setter
    def last_v( self, val ):
        self._last_v = val

    ######################################################################

    @property
    def next_u_data( self ):
        if( hasattr( self, '_next_u_data' ) == False ):
            self._next_u_data = None
        return self._next_u_data

    @next_u_data.setter
    def next_u_data( self, val ):
        self._next_u_data = val

    @property
    def next_v_data( self ):
        if( hasattr( self, '_next_v_data' ) == False ):
            self._next_v_data = None
        return self._next_v_data

    @next_v_data.setter
    def next_v_data( self, val ):
        self._next_v_data = val

    ######################################################################

    def lock( self ):

        if( self.u_filter_result is not None ):
            new_u = self.u_filter_result.get()
            self.updateU( self.last_u_nodes, new_u, self.last_u )
            self.u_filter_result = None

        if( self.v_filter_result is not None ):
            new_v = self.v_filter_result.get()
            self.updateV( self.last_v_nodes, self.last_v_edges, new_v, self.last_v )
            self.v_filter_result = None

    ######################################################################

    @lru_cache()
    def cachedTransition( self, node ):
        return self.transitionProb( node, is_partial_graph_index=True )

    @lru_cache()
    def cachedEmission( self, node ):
        return self.emissionProb( node, is_partial_graph_index=True )

    ######################################################################

    def uBaseLocalInfo( self, node ):
        shape_corrected_initial_distribution = self.initialProb( node, is_partial_graph_index=True )
        shape_corrected_emission_distribution = self.cachedEmission( node )
        return UBaseNodeData( shape_corrected_initial_distribution, shape_corrected_emission_distribution )

    def uLocalInfo( self, node, U, V ):

        full_n_parents = self.nParents( node, is_partial_graph_index=True,
                                              use_partial_graph=False )
        up_edge = self.getUpEdges( node, is_partial_graph_index=True,
                                         use_partial_graph=False )
        shape_corrected_transition_distribution = self.cachedTransition( int( node ) )
        shape_corrected_emission_distribution = self.cachedEmission( int( node ) )

        # Parent information
        partial_parents, partial_parents_order = self.getPartialParents( node, get_order=True,
                                                                               is_partial_graph_index=True,
                                                                               return_partial_graph_index=True )
        partial_parent_us = [ self.uData( U, V, p ) for p in partial_parents ]
        partial_parent_vs = []
        for p in partial_parents:
            edges = self.getDownEdges( p, skip_edges=up_edge,
                                          is_partial_graph_index=True,
                                          use_partial_graph=True )
            vs = self.vData( U, V, p, edges=edges )
            partial_parent_vs.append( vs )

        # Sibling information
        full_siblings = self.getFullSiblings( node, is_partial_graph_index=True,
                                                    return_partial_graph_index=True )
        full_siblings_in_fbs = np.array( [ self.inFeedbackSet( s, is_partial_graph_index=True ) for s in full_siblings ], dtype=bool )
        full_sibling_vs = [ self.vData( U, V, s ) for s in full_siblings ]

        siblings_shape_corrected_transition_distribution = [ self.cachedTransition( s ) for s in full_siblings ]
        siblings_shape_corrected_emission_distribution = [ self.cachedEmission( s ) for s in full_siblings ]

        return UNodeData( full_n_parents,
                          up_edge,
                          shape_corrected_transition_distribution,
                          shape_corrected_emission_distribution,
                          partial_parents,
                          partial_parents_order,
                          partial_parent_us,
                          partial_parent_vs,
                          full_siblings,
                          full_siblings_in_fbs,
                          full_sibling_vs,
                          siblings_shape_corrected_transition_distribution,
                          siblings_shape_corrected_emission_distribution )

    ######################################################################

    def uFilter( self, is_base_case, nodes, U, V ):

        if( is_base_case ):
            # data = self.data_thread_pool.map( self.uBaseLocalInfo, nodes )
            data = [ self.uBaseLocalInfo( node ) for node in nodes ]
            work = uBaseCaseWork
        else:
            # data = self.data_thread_pool.starmap( self.uLocalInfo, zip( nodes, itertools.repeat( U ), itertools.repeat( V ) ) )
            data = [ self.uLocalInfo( node, U, V ) for node in nodes ]
            work = uWork

        self.u_filter_result = self.u_filter_process_pool.map_async( work, data )

        self.last_u_nodes = nodes
        self.last_u = U

    ######################################################################

    def vLocalInfo( self, node, edge, U, V ):

        # Mate information
        partial_mates, partial_mates_order = self.getPartialMates( node, get_order=True,
                                                                         edges=edge,
                                                                         is_partial_graph_index=True,
                                                                         return_partial_graph_index=True )
        partial_mate_us = [ self.uData( U, V, m ) for m in partial_mates ]
        partial_mate_vs = []
        for m in partial_mates:
            edges = self.getDownEdges( m, skip_edges=edge,
                                          is_partial_graph_index=True,
                                          use_partial_graph=True )
            vs = self.vData( U, V, m, edges=edges )
            partial_mate_vs.append( vs )

        # Children information
        full_children = self.getFullChildren( node, edges=edge,
                                                    is_partial_graph_index=True,
                                                    return_partial_graph_index=True )
        full_children_in_fbs = np.array( [ self.inFeedbackSet( c, is_partial_graph_index=True ) for c in full_children ], dtype=bool )
        full_children_vs = [ self.vData( U, V, c ) for c in full_children ]

        children_shape_corrected_transition_distribution = [ self.cachedTransition( c ) for c in full_children ]
        children_shape_corrected_emission_distribution = [ self.cachedEmission( c ) for c in full_children ]

        full_n_parents = self.nParents( full_children[ 0 ], is_partial_graph_index=True,
                                                            use_partial_graph=False )

        return VNodeData( full_n_parents,
                          partial_mates,
                          partial_mates_order,
                          partial_mate_us,
                          partial_mate_vs,
                          full_children,
                          full_children_in_fbs,
                          full_children_vs,
                          children_shape_corrected_transition_distribution,
                          children_shape_corrected_emission_distribution )

    ######################################################################

    def vFilter( self, is_base_case, nodes_and_edges, U, V ):

        if( is_base_case ):
            # Nothing actually happens here
            return

        # Processing can't pickle this class
        data = self.data_thread_pool.starmap( self.vLocalInfo, zip( *nodes_and_edges, itertools.repeat( U ), itertools.repeat( V ) ) )
        # data = [ self.vLocalInfo( node, edge, U, V ) for node, edge in zip( *nodes_and_edges ) ]
        self.v_filter_result = self.v_filter_process_pool.map_async( vWork, data )

        nodes, edges = nodes_and_edges
        self.last_v_nodes = nodes
        self.last_v_edges = edges
        self.last_v = V

    ######################################################################

    def filter( self ):

        self.partial_graph.lock = self.lock

        U, V = self.genFilterProbs()

        # Run message passing over the partial graph
        self.partial_graph.upDown( self.uFilter, self.vFilter, U=U, V=V )

        return U, V

    ######################################################################

    def nodeJointLocalInfo( self, node, U, V ):
        node_partial = self.fullGraphIndexToPartialGraphIndex( node )
        in_fbs = self.inFeedbackSet( node_partial, is_partial_graph_index=True )

        if( in_fbs ):
            fbs_index = self.fbsIndex( node_partial, is_partial_graph_index=True, within_graph=True )
            full_parents = self.getFullParents( node_partial, is_partial_graph_index=True, return_partial_graph_index=True )
            full_n_parents = full_parents.shape[ 0 ]
            n_parents_not_in_fbs = len( [ 1 for p in full_parents if self.inFeedbackSet( p, is_partial_graph_index=True ) == False ] )
            if( full_n_parents == 0 or n_parents_not_in_fbs == 0 ):
                another_node = self._anotherNode( node_partial, is_partial_graph_index=True, return_partial_graph_index=True )
                u = self.uData( U, V, another_node )
                vs = self.vData( U, V, another_node )
                return NodeJointInFBSDataIntegrate( in_fbs, True, fbs_index, u, vs )
            else:
                u_node_data = self.uLocalInfo( node_partial, U, V )
                return NodeJointInFBSData( in_fbs, False, fbs_index, *u_node_data )
        else:
            u = self.uData( U, V, node_partial )
            vs = self.vData( U, V, node_partial )
            return NodeJointNotInFBSData( in_fbs, u, vs )

    def nodeJoint( self, U, V, nodes ):
        # P( x, Y )
        data = self.data_thread_pool.starmap( self.nodeJointLocalInfo, zip( nodes, itertools.repeat( U ), itertools.repeat( V ) ) )
        # data = [ self.nodeJointLocalInfo( node, U, V ) for node in nodes ]
        joints = self.u_filter_process_pool.map( nodeJointSingleNode, data )
        return zip( nodes, joints )

    ######################################################################

    def jointParentsLocalInfo( self, node, U, V ):
        node_partial = self.fullGraphIndexToPartialGraphIndex( node )
        full_parents, full_parents_order = self.getFullParents( node_partial, get_order=True,
                                                          is_partial_graph_index=True,
                                                          return_partial_graph_index=True )
        full_n_parents = full_parents.shape[ 0 ]

        inFBS = lambda p: self.inFeedbackSet( p, is_partial_graph_index=True )

        all_parents_in_fbs = len( [ 1 for p in full_parents if inFBS( p ) == False ] ) == 0
        parent_fbs_indices = [ self.fbsIndex( p, is_partial_graph_index=True, within_graph=True ) for p in full_parents if inFBS( p ) ]

        parent_not_in_fbs_order = [ o for p, o in zip( full_parents, full_parents_order ) if inFBS( p ) == False ]
        parent_in_fbs_order = [ o for p, o in zip( full_parents, full_parents_order ) if inFBS( p ) ]

        if( all_parents_in_fbs ):
            if( inFBS( node_partial ) ):
                node_partial = self._anotherNode( node_partial, is_partial_graph_index=True, return_partial_graph_index=True )

            u = self.uData( U, V, node_partial )
            vs = self.vData( U, V, node_partial )
            node_data = JointParentsAllParentsInFBSData( all_parents_in_fbs,
                                                         u,
                                                         vs,
                                                         parent_fbs_indices,
                                                         parent_not_in_fbs_order,
                                                         parent_in_fbs_order )
        else:
            up_edge = self.getUpEdges( node_partial, is_partial_graph_index=True,
                                                     use_partial_graph=False )
            shape_corrected_transition_distribution = self.cachedTransition( int( node_partial ) )
            shape_corrected_emission_distribution = self.cachedEmission( int( node_partial ) )

            # Parent information
            partial_parents, partial_parents_order = self.getPartialParents( node_partial, get_order=True,
                                                                                   is_partial_graph_index=True,
                                                                                   return_partial_graph_index=True )
            partial_parent_us = [ self.uData( U, V, p ) for p in partial_parents ]
            partial_parent_vs = []
            for p in partial_parents:
                edges = self.getDownEdges( p, skip_edges=up_edge,
                                              is_partial_graph_index=True,
                                              use_partial_graph=True )
                vs = self.vData( U, V, p, edges=edges )
                partial_parent_vs.append( vs )

            # Sibling information
            full_siblings = self.getFullSiblings( node_partial, is_partial_graph_index=True,
                                                        return_partial_graph_index=True )
            full_siblings_in_fbs = np.array( [ inFBS( s ) for s in full_siblings ], dtype=bool )
            full_sibling_vs = [ self.vData( U, V, s ) for s in full_siblings ]

            siblings_shape_corrected_transition_distribution = [ self.cachedTransition( s ) for s in full_siblings ]
            siblings_shape_corrected_emission_distribution = [ self.cachedEmission( s ) for s in full_siblings ]
            vs = self.vData( U, V, node_partial )

            node_data = JointParentsData( all_parents_in_fbs,
                                          full_siblings,
                                          full_siblings_in_fbs,
                                          full_sibling_vs,
                                          siblings_shape_corrected_transition_distribution,
                                          siblings_shape_corrected_emission_distribution,
                                          full_n_parents,
                                          shape_corrected_transition_distribution,
                                          shape_corrected_emission_distribution,
                                          vs,
                                          partial_parents,
                                          partial_parents_order,
                                          partial_parent_us,
                                          partial_parent_vs,
                                          parent_fbs_indices,
                                          parent_not_in_fbs_order,
                                          parent_in_fbs_order )

        return node_data

    def jointParents( self, U, V, nodes ):
        # P( x_p1..pN, Y )
        non_roots = [ node for node in nodes if self.nParents( node, is_partial_graph_index=False ) > 0 ]

        data = self.data_thread_pool.starmap( self.jointParentsLocalInfo, zip( non_roots, itertools.repeat( U ), itertools.repeat( V ) ) )
        joints = self.u_filter_process_pool.map( jointParentsSingleNode, data )
        return zip( non_roots, joints )

    ######################################################################

    def jointParentChildLocalInfo( self, node, U, V ):

        node_partial = self.fullGraphIndexToPartialGraphIndex( node )
        in_fbs = self.inFeedbackSet( node_partial, is_partial_graph_index=True )
        if( in_fbs ):
            fbs_index = self.fbsIndex( node_partial, is_partial_graph_index=True, within_graph=True )
        else:
            fbs_index = -1

        other_args = self.jointParentsLocalInfo( node, U, V )
        if( other_args.all_parents_in_fbs ):
            full_n_parents = self.nParents( node_partial, is_partial_graph_index=True, use_partial_graph=False )
            return JointParentChildInFBSData( in_fbs, fbs_index, full_n_parents, *other_args )

        return JointParentChildData( in_fbs, fbs_index, *other_args )

    def jointParentChild( self, U, V, nodes ):
        # P( x_c, x_p1..pN, Y )
        non_roots = [ node for node in nodes if self.nParents( node, is_partial_graph_index=False ) > 0 ]
        roots = [ node for node in nodes if self.nParents( node, is_partial_graph_index=False ) == 0 ]

        data = self.data_thread_pool.starmap( self.jointParentChildLocalInfo, zip( non_roots, itertools.repeat( U ), itertools.repeat( V ) ) )
        joints = self.u_filter_process_pool.map( jointParentChildSingleNode, data )
        return itertools.chain( zip( non_roots, joints ), self.nodeJoint( U, V, roots ) )

    ######################################################################

    def marginalProb( self, U, V, node=None ):
        # P( Y )
        if( node is None ):
            marginal = 0.0
            for node in self.full_graph.parent_graph_assignments:
                node_data = self.nodeJointLocalInfo( node, U, V )
                joint = nodeJointSingleNode( node_data )
                marginal += nonFBSIntegrate( joint, axes=range( joint.ndim ) )
            return marginal

        node_data = self.nodeJointLocalInfo( node, U, V )
        joint = nodeJointSingleNode( node_data )
        return nonFBSIntegrate( joint, axes=range( joint.ndim ) )

    def nodeSmoothed( self, U, V, nodes, parent_child_smoothed=None ):
        # P( x | Y )
        # marginal = self.marginalProb( U, V )
        # return [ ( node, val - marginal ) for node, val in self.nodeJoint( U, V, nodes ) ]
        if( parent_child_smoothed is None ):
            return [ ( node, val - self.marginalProb( U, V, node=node ) ) for node, val in self.nodeJoint( U, V, nodes ) ]

        # Don't need to repeat computations.  In this case, just integrate out the child axis
        ans = []
        for node, with_parents in parent_child_smoothed:
            n_parents = self.nParents( node )
            if( n_parents == 0 ):
                ans.append( ( node, with_parents ) )
            else:
                ans.append( ( node, nonFBSIntegrate( with_parents, axes=np.arange( with_parents.ndim - 1 ) ) ) )

        return ans

    def parentsSmoothed( self, U, V, nodes, parent_child_smoothed=None ):
        # P( x_p1..pN | Y )
        # Mostly assuming that these will be called from within the graph
        if( parent_child_smoothed is None ):
            return [ ( node, val - self.marginalProb( U, V, node=node ) ) for node, val in self.jointParents( U, V, nodes ) ]

        # Don't need to repeat computations.  In this case, just integrate out the child axis
        return [ ( node, nonFBSIntegrate( with_child, axes=[ -1 ] ) ) for node, with_child in parent_child_smoothed ]

    def parentChildSmoothed( self, U, V, nodes ):
        # P( x_c, x_p1..pN | Y )
        # Mostly assuming that these will be called from within the graph
        return [ ( node, val - self.marginalProb( U, V, node=node ) ) for node, val in self.jointParentChild( U, V, nodes ) ]

    def conditionalParentChild( self, U, V, nodes ):
        # P( x_c | x_p1..pN, Y )

        non_roots = [ node for node in nodes if self.nParents( node, is_partial_graph_index=False ) > 0 ]
        roots = [ node for node in nodes if self.nParents( node, is_partial_graph_index=False ) == 0 ]

        data = self.data_thread_pool.starmap( self.jointParentChildLocalInfo, zip( non_roots, itertools.repeat( U ), itertools.repeat( V ) ) )
        # data = [ self.jointParentChildLocalInfo( node, U, V ) for node in non_roots ]
        joints = self.u_filter_process_pool.map( conditionalParentChildSingleNode, data )
        return itertools.chain( zip( non_roots, joints ), self.nodeSmoothed( U, V, roots ) )
