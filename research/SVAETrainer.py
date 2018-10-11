from GenModels.research.PedigreeLoader import load
from GenModels.research.PedigreeWrappers import Pedigree, PedigreeSexMatters
from GenModels.research.Models import *
import numpy as np
import matplotlib.pyplot as plt

######################################################################

def loadGraphs():
    graphs = load( pedigree_folder_name='GenModels/research/Pedigrees_JSON_FIXED_Label/')

    ad_graphs = np.array( [ ( graph, fbs ) for graph, fbs in graphs if graph.inheritancePattern == 'AD' ] )
    ar_graphs = np.array( [ ( graph, fbs ) for graph, fbs in graphs if graph.inheritancePattern == 'AR' ] )
    xl_graphs = np.array( [ ( graph, fbs ) for graph, fbs in graphs if graph.inheritancePattern == 'XL' ] )

    print( 'Number of graphs for AD: %d AR: %d XL: %d'%( len( ad_graphs ), len( ar_graphs ), len( xl_graphs ) ) )
    return ad_graphs, ar_graphs, xl_graphs

def pickleLoadedGraphs( ad_graphs, ar_graphs, xl_graphs ):
    import pickle
    all_graphs = ( ad_graphs, ar_graphs, xl_graphs )
    with open( 'python_graphs.p', 'wb' ) as file:
        pickle.dump( all_graphs, file )

def loadPickledGraphs():
    import pickle
    with open( 'python_graphs.p', 'rb' ) as file:
        ad_graphs, ar_graphs, xl_graphs = pickle.load( file )

    return ad_graphs, ar_graphs, xl_graphs

######################################################################

def stratifySample( ad_graphs, ar_graphs, xl_graphs, train_test_split=0.8 ):

    def trainTestIndices( graphs ):
        n_train = int( len( graphs ) * train_test_split )
        train_indices = np.random.choice( len( graphs ), n_train, replace=False )
        test_indices = np.setdiff1d( np.arange( len( graphs ) ), train_indices )
        return train_indices, test_indices

    ad_train_indices, ad_test_indices = trainTestIndices( ad_graphs )
    ar_train_indices, ar_test_indices = trainTestIndices( ar_graphs )
    xl_train_indices, xl_test_indices = trainTestIndices( xl_graphs )

    print( 'len( ad_train_indices )', len( ad_train_indices ) )
    print( 'len( ad_test_indices )', len( ad_test_indices ) )
    print( 'len( ar_train_indices )', len( ar_train_indices ) )
    print( 'len( ar_test_indices )', len( ar_test_indices ) )
    print( 'len( xl_train_indices )', len( xl_train_indices ) )
    print( 'len( xl_test_indices )', len( xl_test_indices ) )

    training_graphs = np.vstack( ( ad_graphs[ ad_train_indices ], ar_graphs[ ar_train_indices ], xl_graphs[ xl_train_indices ] ) )
    test_graphs = np.vstack( ( ad_graphs[ ad_test_indices ], ar_graphs[ ar_test_indices ], xl_graphs[ xl_test_indices ] ) )

    return np.random.permutation( training_graphs ), np.random.permutation( test_graphs )

# graphs = loadGraphs()
# pickleLoadedGraphs( *graphs )
# assert 0

ad_graphs, ar_graphs, xl_graphs = loadPickledGraphs()
training_graphs, test_graphs = stratifySample( ad_graphs, ar_graphs, xl_graphs )

trainer = InheritancePatternTrainer( training_graphs, test_graphs )
trainer.train( num_iters=30000 )