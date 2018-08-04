from GenModels.GM.States.GraphicalMessagePassing import DataGraph, GroupGraph, GraphHMMFBS, GraphHMMFBSMultiGroups
import numpy as np
from functools import reduce
from scipy.sparse import coo_matrix
from collections import Iterable
from GenModels.GM.Utility import fbsData
import copy

######################################################################

class _pedigreeMixin():

    def __init__( self ):
        super().__init__()
        self.attrs = {}
        self.studyID = None

    def setNodeAttrs( self, nodes, attrs ):
        if( isinstance( nodes, int ) ):
            if( nodes not in self.attrs ):
                self.attrs[ nodes ] = {}
            self.attrs[ nodes ].update( attrs )
        else:
            for node, attr in zip( nodes, attrs ):
                if( node not in self.attrs ):
                    self.attrs[ node ] = {}
                self.attrs[ node ].update( attr )

class Pedigree( _pedigreeMixin, DataGraph ):
    pass

class PedigreeSexMatters( _pedigreeMixin, GroupGraph ):

    @staticmethod
    def fromPedigree( pedigree, deep_copy=False ):
        deepcopy = copy.deepcopy if deep_copy else lambda x: x
        new_pedigree = PedigreeSexMatters()
        new_pedigree.nodes = deepcopy( pedigree.nodes )
        new_pedigree.edge_children = deepcopy( pedigree.edge_children )
        new_pedigree.edge_parents = deepcopy( pedigree.edge_parents )
        new_pedigree.data = deepcopy( pedigree.data )
        new_pedigree.possible_latent_states = deepcopy( pedigree.possible_latent_states )
        new_pedigree.attrs = deepcopy( pedigree.attrs )
        sex_to_index = lambda x: [ 'female', 'male', 'unknown' ].index( x )
        for node, attr in pedigree.attrs.items():
            new_pedigree.setGroups( nodes, sex_to_index( attr[ 'sex' ] ) )

        return new_pedigree

    def setNodeAttrs( self, nodes, attrs ):
        super().setNodeAttrs( nodes, attrs )

        sex_to_index = lambda x: [ 'female', 'male', 'unknown' ].index( x )
        self.setGroups( nodes, sex_to_index( attrs[ 'sex' ] ) )

######################################################################

class _pedigreeFilterMixin():

    def preprocessData( self, data_graphs ):
        super().preprocessData( data_graphs )
        self.node_attrs = []
        total_nodes = 0
        for graph, fbs in data_graphs:
            for node in graph.nodes:
                self.node_attrs.append( graph.attrs[ node ] )
            total_nodes += len( graph.nodes )

    def draw( self, render=True, **kwargs ):

        male_style = dict( shape='square' )
        female_style = dict( shape='circle' )
        unknown_style = dict( shape='diamond' )
        affected_male_style = dict( shape='square', fontcolor='black', style='bold', color='blue' )
        affected_female_style = dict( shape='circle', fontcolor='black', style='bold', color='blue' )
        affected_unknown_style = dict( shape='diamond', fontcolor='black', style='bold', color='blue' )
        styles = { 0: male_style, 1: female_style, 2: unknown_style, 3: affected_male_style, 4: affected_female_style, 5: affected_unknown_style }

        unaffected_males = []
        unaffected_females = []
        unaffected_unknowns = []
        affected_males = []
        affected_females = []
        affected_unknowns = []

        for n in self.nodes:
            attrs = self.node_attrs[ n ]
            if( attrs[ 'sex' ] == 'male' ):
                if( attrs[ 'affected' ] == True ):
                    affected_males.append( n )
                else:
                    unaffected_males.append( n )
            elif( attrs[ 'sex' ] == 'female' ):
                if( attrs[ 'affected' ] == True ):
                    affected_females.append( n )
                else:
                    unaffected_females.append( n )
            else:
                if( attrs[ 'affected' ] == True ):
                    affected_unknowns.append( n )
                else:
                    unaffected_unknowns.append( n )

        node_to_style_key =       dict( [ ( n, 0 ) for n in unaffected_males ] )
        node_to_style_key.update( dict( [ ( n, 1 ) for n in unaffected_females ] ) )
        node_to_style_key.update( dict( [ ( n, 2 ) for n in unaffected_unknowns ] ) )
        node_to_style_key.update( dict( [ ( n, 3 ) for n in affected_males ] ) )
        node_to_style_key.update( dict( [ ( n, 4 ) for n in affected_females ] ) )
        node_to_style_key.update( dict( [ ( n, 5 ) for n in affected_unknowns ] ) )

        kwargs.update( dict( styles=styles, node_to_style_key=node_to_style_key) )

        return self.toGraph().draw( render=render, **kwargs )

######################################################################

class PedigreeHMMFilter( _pedigreeFilterMixin, GraphHMMFBS ):
    pass

class PedigreeHMMFilterSexMatters( _pedigreeFilterMixin, GraphHMMFBSMultiGroups ):
    pass
