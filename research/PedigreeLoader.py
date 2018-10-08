import os
import subprocess
import sys
import autograd.numpy as np
import itertools
import json
from tqdm import tqdm

from GenModels.research.model import Pedigree as JSONPedigree
from GenModels.research.PedigreeWrappers import Pedigree, PedigreeSexMatters
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

        for mate, children in person.mateKids:
            # Order is female, male, unknown
            if( person.Id not in index_map ):
                index_map[ person.Id ] = len( index_map )
            if( mate.Id not in index_map ):
                index_map[ mate.Id ] = len( index_map )

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

    graph = PedigreeSexMatters() if pedigree.inheritancePattern == 'XL' else Pedigree()
    for parents, children in edges.values():
        graph.addEdge( parents=parents, children=children )

    graph.pedigree_obj = pedigree

    found_affected = False
    for person in pedigree.family:
        node = index_map[ person.Id ]
        graph.setNodeData( node, np.array( [ int( person.affected ) ] ) )
        graph.setNodeAttrs( node, dict( sex=person.sex, affected=person.affected, carrier=person.carrier, age=person.age ) )
        if( person.affected ):
            found_affected = True

    if( found_affected == False ):
        raise Exception( 'There isn\'t an affected node!' )
    graph.studyID = pedigree.studyID
    graph.ethnicity1 = pedigree.ethnicity1
    graph.ethnicity2 = pedigree.ethnicity2
    graph.inheritancePattern = pedigree.inheritancePattern

    return graph

######################################################################

def load( pedigree_folder_name='Pedigrees_JSON_FIXED_Label/' ):

    graphs = []
    for dir_name, sub_dir_list, file_list in os.walk( os.path.join( os.getcwd(), pedigree_folder_name ) ):

        for file_name in file_list:
            full_file_name = os.path.join( pedigree_folder_name, file_name )

            with open( full_file_name ) as data_file:
                data = json.loads( json.load( data_file ) )
            pedigree = JSONPedigree( data )

            try:
                graph = pedigreeToGraph( pedigree )
                feedback_set = computeFeedbackSet( graph )

                # computeFeedbackSet doesn't get this right
                if( graph.studyID == '3729MM' ):
                    feedback_set = np.array( [ 5 ] )

                graphs.append( ( graph, feedback_set ) )
            except Exception as Argument:
                print( 'Graph', file_name, 'is incorrect.', Argument )

    return graphs
