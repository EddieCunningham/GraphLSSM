import numpy as np
import copy
import itertools
import networkx as nx
from functools import partial
from matplotlib.pyplot import get_cmap
from matplotlib.colors import to_hex
print = partial( print, flush=True )

from bokeh.plotting import figure
from bokeh.layouts import layout, widgetbox, column, row
from bokeh.models import MultiLine, Circle, Square, Diamond, Range1d, Plot, Range1d, HoverTool, TapTool, BoxSelectTool, ColumnDataSource
from bokeh.models.widgets.tables import NumberFormatter
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, NodesOnly
from bokeh.palettes import Spectral4
from bokeh.io import show, output_notebook
from bokeh.models.widgets import CheckboxButtonGroup, TableColumn, DataTable, Button, RadioButtonGroup

from GenModels.research.Models import *
from GenModels.research.PedigreeWrappers import setGraphRootStates

__all__ = [ 'pedigreeRenderers', 'genHintonDiagram', 'bokehPlot' ]

######################################################################

def genHintonDiagram( weights, width_height=200 ):

    def hintonSizes( weights ):
        max_size = 2**np.ceil( np.log( np.abs( weights ).max() ) / np.log( 2 ) )
        return np.sqrt( np.abs( weights ) / max_size ), max_size
    sizes, max_size = hintonSizes( weights )

    last_axes = weights.shape[ -2: ]
    scale = width_height / ( 1 + last_axes[ -1 ] )

    width = int( width_height )
    height = int( last_axes[ 0 ] / last_axes[ 1 ] * width_height )

    if( weights.ndim > 2 ):
        all_diagrams = np.empty( weights.shape[ :-2 ], dtype=object )
    else:
        all_diagrams = np.empty( 1, dtype=object )

    for shape in itertools.product( *[ range( 1, s + 1 ) for s in weights.shape[ :-2 ] ] ):

        hinton = figure( width=width, height=height, x_range=Range1d( -1, last_axes[ -1 ] + 1 ), y_range=Range1d( -1,  last_axes[ -2 ] + 1 ), toolbar_location=None )
        hinton.grid.grid_line_color = None
        hinton.background_fill_color = 'black'
        hinton.axis.visible = False

        ys, xs = zip( *itertools.product( np.linspace( last_axes[ -2 ], 0, last_axes[ -2 ] ), np.linspace( 0,  last_axes[ -1 ], last_axes[ -1 ] ) ) )
        size_selector = tuple( [ ( s - 1, ) for s in shape ] )
        current_sizes = sizes[ size_selector ].ravel()
        current_weights = weights[ size_selector ].ravel()
        if( current_sizes.size == 0 ):
            current_sizes = np.copy( sizes.ravel() )

        ds = ColumnDataSource( data={ 'xs':xs, 'ys':ys, 'sizes': current_sizes*scale, 'weights': current_weights } )
        hinton.square( 'xs', 'ys', source=ds, color='white', size='sizes' )

        hover = HoverTool( tooltips=[ ( 'Weight', '@weights' ) ] )
        hinton.add_tools( hover )


        all_diagrams[ tuple( [ [ ( s-1, ) ] for s in shape ] ) ] = hinton

    return layout( all_diagrams.tolist() )

######################################################################

def genGraphRenderer( G, data, Shape ):
    graph_renderer = from_networkx( G, partial( nx.nx_agraph.pygraphviz_layout, prog='dot' ) )

    graph_renderer.node_renderer.data_source.data = data

    BaseShape = partial( Shape, line_width='line_widths', line_color='line_colors', fill_color='colors', size='shape_size', fill_alpha='visible', line_alpha='visible' )

    graph_renderer.node_renderer.glyph = BaseShape()
    graph_renderer.edge_renderer.glyph = MultiLine( line_color="#CCCCCC", line_alpha=0.2, line_width=5 )

    graph_renderer.inspection_policy = NodesOnly()

    return graph_renderer

######################################################################

def pedigreeRenderers( bokeh_state, data_only=False ):
    G = bokeh_state.pedigree.toNetworkX()

    node_index = np.array( G.nodes ).tolist()

    node_size = bokeh_state.node_size
    edge_size = bokeh_state.edge_size

    for n in node_index:
        if( n >= 0 ):
            assert n in bokeh_state.sex
            assert n in bokeh_state.probs
            assert n in bokeh_state.pls

    base_data = dict( index=node_index,
                      shape_size=[ node_size if n >= 0 else edge_size for n in node_index ],
                      sex=[ bokeh_state.sex[ n ] if n >= 0 else 'edge' for n in node_index ],
                      probs=[ np.round( bokeh_state.probs[ n ], decimals=2 ) if n >= 0 else np.array( [ -1 ] ) for n in node_index ],
                      possible_latent_states=[ bokeh_state.possibleLabels( n ) if n >= 0 else [ -1 ] for n in node_index ],
                      colors=[ bokeh_state.getColor( n ) if n >= 0 else 'edge' for n in node_index ],
                      line_colors=[ bokeh_state.getLineColor( n ) if n >= 0 else 'edge' for n in node_index ],
                      line_widths=[ bokeh_state.getLineWidth( n ) if n >= 0 else 'edge' for n in node_index ] )

    male_data = {}
    male_data[ 'visible' ] = [ 1 if n >= 0 and bokeh_state.sex[ n ] == 'male' else 0 for n in node_index ]
    male_data.update( base_data )

    female_data = {}
    female_data[ 'visible' ] = [ 1 if n >= 0 and bokeh_state.sex[ n ] == 'female' else 0 for n in node_index ]
    female_data.update( base_data )

    unknown_data = {}
    unknown_data[ 'visible' ] = [ 1 if n >= 0 and bokeh_state.sex[ n ] == 'unknown' else 0 for n in node_index ]
    unknown_data.update( base_data )

    if( data_only == True ):
        return male_data, female_data, unknown_data

    male_renderer = genGraphRenderer( G, male_data, Square )
    female_renderer = genGraphRenderer( G, female_data, Circle )
    unknown_renderer = genGraphRenderer( G, unknown_data, Diamond )

    return male_renderer, female_renderer, unknown_renderer

######################################################################

class BokehState():

    def __init__( self, pedigree, fbs, ad_priors, ar_priors, xl_priors ):
        pedigree.possible_latent_states = {}
        self.pedigree = pedigree
        self.fbs = fbs
        self.inheritance_pattern = 'XL'
        self.current_node = None
        self.sex = dict( [ ( node, attr[ 'sex' ] ) for node, attr in pedigree.attrs.items() ] )
        self.pls = dict( [ ( node, self._getStates( node, 'XL' ) ) for node in pedigree.nodes ] )
        for node, pls in pedigree.possible_latent_states.items():
            self.pls[ node ] = np.array( pls )

        self.ad_params = AutosomalParametersEM( *ad_priors )
        self.ar_params = AutosomalParametersEM( *ar_priors )
        self.xl_params = XLinkedParametersEM( *xl_priors )

        self.marginal = -1
        self.probs = self.filter()

        self.node_size = 30
        self.edge_size = 1

        self.cmap = get_cmap( 'Blues' )

    def resampleParameters( self ):
        if( self.inheritance_pattern == 'AD' ):
            self.ad_params.initial_dist.resample()
            self.ad_params.transition_dist.resample()
            self.ad_params.emission_dist.resample()
        elif( self.inheritance_pattern == 'AR' ):
            self.ar_params.initial_dist.resample()
            self.ar_params.transition_dist.resample()
            self.ar_params.emission_dist.resample()
        elif( self.inheritance_pattern == 'XL' ):
            for d in self.xl_params.initial_dists.values():
                d.resample()
            for d in self.xl_params.transition_dists.values():
                d.resample()
            for d in self.xl_params.emission_dists.values():
                d.resample()

        self.probs = self.filter()

    def changeInheritancePattern( self, ip ):
        self.inheritance_pattern = ip
        self.pls = dict( [ ( node, self._getStates( node, ip ) ) for node in self.pedigree.nodes ] )
        for node, pls in self.pedigree.possible_latent_states.items():
            self.pls[ node ] = np.array( pls )

        self.probs = self.filter()

    def probOfBeingCarrier( self, node, prob ):
        if( self.inheritance_pattern != 'XL' ):
            return prob[ :2 ].sum()

        if( self.sex[ node ] == 'female' ):
            return prob[ :2 ].sum()
        if( self.sex[ node ] == 'male' ):
            return prob[ 0 ]
        return prob[ [ 0, 1, 3 ] ].sum()

    def getColor( self, node ):
        probs = self.probs[ node ]
        carrier_prob = self.probOfBeingCarrier( node, probs )
        color = to_hex( self.cmap( carrier_prob ), keep_alpha=False )
        return color

    def getLineColor( self, node ):
        if( self.pedigree.data[ node ] == 1 ):
            return 'red'
        return 'black'

    def getLineWidth( self, node ):
        if( self.pedigree.data[ node ] == 1 ):
            return 5
        return 0.1

    def filter( self ):
        pedigree_copy = copy.deepcopy( self.pedigree )
        for n, possible_latent_states in self.pls.items():
            pedigree_copy.setPossibleLatentStates( int( n ), possible_latent_states )

        if( self.inheritance_pattern == 'AD' ):
            model = AutosomalDominant( [ ( pedigree_copy, self.fbs ) ], params=self.ad_params, method='EM' )
        elif( self.inheritance_pattern == 'AR' ):
            model = AutosomalRecessive( [ ( pedigree_copy, self.fbs ) ], params=self.ar_params, method='EM' )
        elif( self.inheritance_pattern == 'XL' ):
            model = XLinkedRecessive( [ ( pedigree_copy, self.fbs ) ], params=self.xl_params, method='EM' )
        node_smoothed, marginal = model.stateUpdate()
        self.marginal = marginal
        return node_smoothed

    def possibleLabels( self, node ):
        labels = self._getLabels( node, self.inheritance_pattern )
        return [ labels[ i ] for i in self.pls[ node ] ]

    def selectNodeFromTable( self, node ):
        self.current_node = int( node )

    def changePossibleLatentStates( self, node, new_pls ):
        self.pls[ node ] = np.array( new_pls, dtype=int )

    def updateCurrentPossibleLatentStates( self, new_pls ):

        if( np.setdiff1d( np.array( new_pls, dtype=int ), self.pls[ self.current_node ] ).size != 0 ):
            return False
        self.pls[ self.current_node ] = np.array( new_pls, dtype=int )
        self.probs = self.filter()
        return True

    def _getLabels( self, node, inheritance_pattern ):
        if( inheritance_pattern != 'XL' ):
            return [ 'AA', 'Aa', 'aa' ]

        sex = self.sex[ node ]
        if( sex == 'male' ):
            return [ 'XY', 'xY' ]
        if( sex == 'female' ):
            return [ 'XX', 'Xx', 'xx' ]
        if( sex == 'unknown' ):
            return [ 'XX', 'Xx', 'xx', 'XY', 'xY' ]

    def _getStates( self, node, inheritance_pattern ):
        return np.arange( len( self._getLabels( node, inheritance_pattern ) ), dtype=int )

    @property
    def current_labels( self ):
        return self._getLabels( self.current_node, self.inheritance_pattern )

    @property
    def current_active_labels( self ):
        return self.pls[ self.current_node ].tolist()

    @property
    def full_prob_matrix( self ):
        if( self.inheritance_pattern == 'XL' ):
            all_probs = np.zeros( ( len( self.probs ), 5 ) )
            for n, prob in self.probs.items():
                if( self.sex[ n ] == 'female' ):
                    all_probs[ n, 0:3 ] = prob
                elif( self.sex[ n ] == 'male' ):
                    all_probs[ n, 3:5 ] = prob
                else:
                    all_probs[ n, : ] = prob
        else:
            all_probs = np.zeros( ( len( self.probs ), 3 ) )
            for n, prob in self.probs.items():
                all_probs[ n, : ] = prob

        return all_probs

######################################################################

def bokehPlot( doc, pedigree, fbs ):

    # Define the priors
    ad_priors = autosomalDominantPriors( prior_strength=10.0 )
    ar_priors = autosomalRecessivePriors( prior_strength=10.0 )
    xl_priors = xLinkedRecessivePriors( prior_strength=10.0 )

    # Define the state
    bokeh_state = BokehState( pedigree, fbs, ad_priors, ar_priors, xl_priors )

    # Create the boundary for the plot
    layout = nx.nx_agraph.pygraphviz_layout( pedigree.toNetworkX(), prog='dot' )
    positions = np.array( list( layout.values() ) )
    x_min, x_max = positions[ :, 0 ].min() - 30, positions[ :, 0 ].max() + 30
    y_min, y_max = positions[ :, 1 ].min() - 30, positions[ :, 1 ].max() + 30

    # Create the pedigree figure that will graphically display the pedigree
    pedigree_fig = figure( title='log( Marginal ) = %5.3f'%( bokeh_state.marginal ), width=1000, height=600, tools='pan,wheel_zoom,reset,box_select', x_range=Range1d( x_min, x_max ), y_range=Range1d( y_min, y_max ) )
    pedigree_fig.grid.grid_line_color = None
    pedigree_fig.background_fill_color = 'white'
    pedigree_fig.axis.visible = False

    # Add a hover tool over the pedigree figure
    hover = HoverTool( tooltips=[ ( 'node', '@index' ), ( 'State probabilities', '[ @probs ]' ), ( 'Possible genotypes', '@possible_latent_states' ) ] )
    pedigree_fig.add_tools( hover )

    # Add the male, female and unknown shapes to the pedigree figure
    male_renderer, female_renderer, unknown_renderer = pedigreeRenderers( bokeh_state )
    pedigree_fig.renderers.extend( [ male_renderer, female_renderer, unknown_renderer ] )

    # Add the expected state probabilities
    state_col = TableColumn( field='latent_state', title='Latent State' )

    XL_cols = [ TableColumn( field='prob_XX', title='XX', formatter=NumberFormatter( format='0.00' ) ),
                TableColumn( field='prob_Xx', title='Xx', formatter=NumberFormatter( format='0.00' ) ),
                TableColumn( field='prob_xx', title='xx', formatter=NumberFormatter( format='0.00' ) ),
                TableColumn( field='prob_XY', title='XY', formatter=NumberFormatter( format='0.00' ) ),
                TableColumn( field='prob_xY', title='xY', formatter=NumberFormatter( format='0.00' ) ) ]
    XL_labels = [ 'XX', 'Xx', 'xx', 'XY', 'xY' ]

    A_cols = [ TableColumn( field='prob_AA', title='AA', formatter=NumberFormatter( format='0.00' ) ),
               TableColumn( field='prob_Aa', title='Aa', formatter=NumberFormatter( format='0.00' ) ),
               TableColumn( field='prob_aa', title='aa', formatter=NumberFormatter( format='0.00' ) ) ]
    A_labels = [ 'AA', 'Aa', 'aa' ]

    # Add a radio button menu for the inheritance pattern
    ip_radio = RadioButtonGroup( labels=[ 'AD', 'AR', 'XL' ], active=2 )

    # Add the possible states box
    button_title = Button( label='Possible latent states' )
    checkbox = CheckboxButtonGroup( labels=XL_labels )

    # Create the table that will display all of the nodes and their expected probabilities
    data_table = DataTable( columns=XL_cols, width=600 )

    # Update the possible latent states view when a node is selected from the data table
    def selectNewNode( attr, old, new ):
        nonlocal bokeh_state, checkbox, button_title
        current_node = new[ 0 ]

        bokeh_state.selectNodeFromTable( current_node )

        checkbox.labels = bokeh_state.current_labels
        checkbox.active = bokeh_state.current_active_labels

        button_title.label = 'Possible latent states for node %d'%( current_node )

    def updateTableData( state, table ):
        all_probs = state.full_prob_matrix
        if( state.inheritance_pattern == 'XL' ):
            table.source = ColumnDataSource( {
                'prob_XX' : all_probs[ :, 0 ],
                'prob_Xx' : all_probs[ :, 1 ],
                'prob_xx' : all_probs[ :, 2 ],
                'prob_XY' : all_probs[ :, 3 ],
                'prob_xY' : all_probs[ :, 4 ]
            } )
        else:
            table.source = ColumnDataSource( {
                'prob_AA' : all_probs[ :, 0 ],
                'prob_Aa' : all_probs[ :, 1 ],
                'prob_aa' : all_probs[ :, 2 ]
            } )
        table.source.selected.on_change( 'indices', selectNewNode )

    updateTableData( bokeh_state, data_table )

    def changeInheritancePattern( attr, old, new ):
        nonlocal bokeh_state, checkbox, button_title, data_table

        ip = ip_radio.active
        ip = [ 'AD', 'AR', 'XL' ][ ip ]

        # Update the state
        bokeh_state.changeInheritancePattern( ip )

        # Update the check box
        checkbox.labels = A_labels if ip != 'XL' else XL_labels

        # Update the data table
        data_table.columns = A_cols if ip != 'XL' else XL_cols
        updateTableData( bokeh_state, data_table )

        male_data, female_data, unknown_data = pedigreeRenderers( bokeh_state, data_only=True )

        male_renderer.node_renderer.data_source.data = male_data
        female_renderer.node_renderer.data_source.data = female_data
        unknown_renderer.node_renderer.data_source.data = unknown_data

        # Update the pedigree figure title
        pedigree_fig.title.text = 'log( Marginal ) = %5.3f'%( bokeh_state.marginal )

    ip_radio.on_change( 'active', changeInheritancePattern )

    # Update the possible latent states across all data sources
    def updatePossibleStates( attr, old, new ):
        nonlocal bokeh_state, data_table, male_renderer, female_renderer, unknown_renderer

        states = checkbox.active

        new_update = bokeh_state.updateCurrentPossibleLatentStates( states )

        if( new_update == False ):
            return

        male_data, female_data, unknown_data = pedigreeRenderers( bokeh_state, data_only=True )

        male_renderer.node_renderer.data_source.data = male_data
        female_renderer.node_renderer.data_source.data = female_data
        unknown_renderer.node_renderer.data_source.data = unknown_data

        updateTableData( bokeh_state, data_table )

        pedigree_fig.title.text = 'log( Marginal ) = %5.3f'%( bokeh_state.marginal )

    checkbox.on_change( 'active', updatePossibleStates )

    # Add the transition hinton diagram
    weights_a_trans, = bokeh_state.ad_params.transition_dist.params
    weights_a_emiss, = bokeh_state.ad_params.emission_dist.params
    hinton_a = [ genHintonDiagram( weights_a_trans ), genHintonDiagram( weights_a_emiss ) ]



    # Button to resample the parameters
    resample_button = Button( label='Resample parameters', button_type='success' )

    def resampleParameters():
        nonlocal bokeh_state

        bokeh_state.resampleParameters()

        male_data, female_data, unknown_data = pedigreeRenderers( bokeh_state, data_only=True )

        male_renderer.node_renderer.data_source.data = male_data
        female_renderer.node_renderer.data_source.data = female_data
        unknown_renderer.node_renderer.data_source.data = unknown_data

        updateTableData( bokeh_state, data_table )

        pedigree_fig.title.text = 'log( Marginal ) = %5.3f'%( bokeh_state.marginal )

    resample_button.on_click( resampleParameters )

    node_specific = column( button_title, checkbox, data_table, ip_radio, resample_button )

    doc.add_root( row( node_specific, pedigree_fig, *hinton_a ) )

    return doc