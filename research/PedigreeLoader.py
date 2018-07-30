import os
import subprocess
import sys
import numpy as np
import itertools
import json
from tqdm import tqdm

from GenModels.research.model import Pedigree as JSONPedigree
from GenModels.research.PedigreeWrappers import Pedigree
from GenModels.research.CycleDetector import computeFeedbackSet

class BiDirectionalDict( dict ):
    # https://stackoverflow.com/a/21894086
    def __init__(self, *args, **kwargs):
        super(BiDirectionalDict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value,[]).append(key)

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super(BiDirectionalDict, self).__setitem__(key, value)
        self.inverse.setdefault(value,[]).append(key)

    def __delitem__(self, key):
        self.inverse.setdefault(self[key],[]).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super(BiDirectionalDict, self).__delitem__(key)

######################################################################

def pedigreeToGraph( pedigree ):
    # Key is sorted mates
    # Value is [ parents, children ]
    edges = {}
    index_map = BiDirectionalDict( {} )

    for person in pedigree.family:
        if( person.Id not in index_map ):
            index_map[ person.Id ] = len( index_map )

        for mate, children in person.mateKids:
            parents = tuple( sorted( [ mate, person ], key=lambda p: p.sex ) )
            if( parents not in edges ):
                reindexed_parents = []
                for p in parents:
                    if( p.Id not in index_map ):
                        index_map[ p.Id ] = len( index_map )
                    reindexed_parents.append( index_map[ p.Id ] )

                reindexed_children = []
                for c in children:
                    if( c.Id not in index_map ):
                        index_map[ c.Id ] = len( index_map )
                    reindexed_children.append( index_map[ c.Id ] )

                edges[ parents ] = [ reindexed_parents, reindexed_children ]

    graph = Pedigree()
    for parents, children in edges.values():
        graph.addEdge( parents=parents, children=children )

    for person in pedigree.family:
        node = index_map[ person.Id ]
        graph.updateNodeAttrs( node, dict( sex=person.sex, affected=person.affected, carrier=person.carrier, age=person.age ) )

    return graph

######################################################################

def load( pedigree_folder_name='Pedigrees_JSON_Fixed_Label' ):

    graphs = []
    for dir_name, sub_dir_list, file_list in os.walk( pedigree_folder_name ):
        for file_name in file_list:
            full_file_name = os.path.join( pedigree_folder_name, file_name )

            with open( full_file_name ) as data_file:
                data = json.loads( json.load( data_file ) )
            pedigree = JSONPedigree( data )

            graph = pedigreeToGraph( pedigree )

            try:
                feedback_set = computeFeedbackSet( graph )
                graphs.append( ( graph, feedback_set ) )
            except Exception as Argument:
                print( 'Graph', file_name, 'is incorrect.', Argument )

    return graphs
