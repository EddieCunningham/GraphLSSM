from GenModels.research.PedigreeLoader import load
from GenModels.research.PedigreeWrappers import Pedigree, PedigreeSexMatters
from GenModels.research.Models import *
import numpy as np
import matplotlib.pyplot as plt
from autograd.misc.optimizers import adam, unrolledGrad
from autograd.misc import flatten
from autograd import grad, value_and_grad, jacobian

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

def upsample( current_indices, target_number ):
    # Randomly choose other indices to hit target_number
    extra_indices = []
    for _ in range( len( current_indices ), target_number ):
        j = np.random.choice( len( current_indices ) )
        extra_indices.append( current_indices[ j ] )
    current_indices = np.hstack( ( current_indices, extra_indices ) ).astype( int )
    return current_indices

def stratifySample( ad_graphs, ar_graphs, xl_graphs, train_test_split=0.8 ):

    def trainTestIndices( graphs ):
        n_train = int( len( graphs ) * train_test_split )
        train_indices = np.random.choice( len( graphs ), n_train, replace=False )
        test_indices = np.setdiff1d( np.arange( len( graphs ) ), train_indices )
        return train_indices, test_indices

    # Get the indices of the graphs that we want
    ad_train_indices, ad_test_indices = trainTestIndices( ad_graphs )
    ar_train_indices, ar_test_indices = trainTestIndices( ar_graphs )
    xl_train_indices, xl_test_indices = trainTestIndices( xl_graphs )

    print( 'len( ad_train_indices )', len( ad_train_indices ) )
    print( 'len( ad_test_indices )', len( ad_test_indices ) )
    print( 'len( ar_train_indices )', len( ar_train_indices ) )
    print( 'len( ar_test_indices )', len( ar_test_indices ) )
    print( 'len( xl_train_indices )', len( xl_train_indices ) )
    print( 'len( xl_test_indices )', len( xl_test_indices ) )

    # Up sample the indices so that all classes have the same number
    max_number = np.max( [ len( ad_train_indices ), len( ar_train_indices ), len( xl_train_indices ) ] )

    ad_train_indices = upsample( ad_train_indices, max_number )
    ar_train_indices = upsample( ar_train_indices, max_number )
    xl_train_indices = upsample( xl_train_indices, max_number )

    training_graphs = np.vstack( ( ad_graphs[ ad_train_indices ], ar_graphs[ ar_train_indices ], xl_graphs[ xl_train_indices ] ) )
    test_graphs = np.vstack( ( ad_graphs[ ad_test_indices ], ar_graphs[ ar_test_indices ], xl_graphs[ xl_test_indices ] ) )

    return np.random.permutation( training_graphs ), np.random.permutation( test_graphs )

######################################################################

def createCrossfoldIndices( n_graphs, n_folds=5 ):

    split_indices = [ int( x ) for x in np.linspace( 0, n_graphs - 1, n_folds + 1 ) ]
    all_test_indices = []
    all_train_indices = []
    for i in range( 1, n_folds + 1 ):
        test_indices = np.arange( split_indices[ i - 1 ], split_indices[ i ] )
        train_indices = np.setdiff1d( np.arange( n_graphs ), test_indices )

        all_test_indices.append( test_indices )
        all_train_indices.append( train_indices )

    return all_train_indices, all_test_indices

def generateCrossfoldData( train_indices, test_indices, graphs ):

    train_graphs = graphs[ train_indices ]

    ad_train_indices = np.array( [ i for i in train_indices if graphs[ i ][ 0 ].inheritancePattern == 'AD' ] )
    ar_train_indices = np.array( [ i for i in train_indices if graphs[ i ][ 0 ].inheritancePattern == 'AR' ] )
    xl_train_indices = np.array( [ i for i in train_indices if graphs[ i ][ 0 ].inheritancePattern == 'XL' ] )

    max_number = np.max( [ ad_train_indices.shape[ 0 ], ar_train_indices.shape[ 0 ], xl_train_indices.shape[ 0 ] ] )

    ad_train_indices = upsample( ad_train_indices, max_number )
    ar_train_indices = upsample( ar_train_indices, max_number )
    xl_train_indices = upsample( xl_train_indices, max_number )

    train_graphs = graphs[ np.hstack( [ ad_train_indices, ar_train_indices, xl_train_indices ] ) ]
    test_graphs = graphs[ test_indices ]

    return train_graphs, test_graphs

######################################################################

def loadTrainTestGraphs():

    ad_graphs, ar_graphs, xl_graphs = loadPickledGraphs()
    train_graphs, test_graphs = stratifySample( ad_graphs, ar_graphs, xl_graphs )

    return train_graphs, test_graphs

######################################################################

def loadCrossfoldGraphs( n_folds=5, fold_number=0 ):

    np.random.seed( 2 )

    ad_graphs, ar_graphs, xl_graphs = loadPickledGraphs()
    graphs = ad_graphs.tolist() + ar_graphs.tolist() + xl_graphs.tolist()
    graphs = np.random.permutation( graphs )

    train_indices, test_indices = createCrossfoldIndices( len( graphs ), n_folds=n_folds )
    train_graphs, test_graphs = generateCrossfoldData( train_indices[ fold_number ], test_indices[ fold_number ], graphs )

    return train_graphs, test_graphs

# graphs = loadGraphs()
# pickleLoadedGraphs( *graphs )

# assert 0

train_graphs, test_graphs = loadCrossfoldGraphs( n_folds=5, fold_number=1 )

trainer = InheritancePatternDES( train_graphs, test_graphs )
trainer.trainNonAutogradAdam( num_iters=100000, step_size=0.001 )
