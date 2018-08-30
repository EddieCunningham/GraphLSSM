from GenModels.GM.Distributions import Categorical, Dirichlet, TensorTransition, TensorTransitionDirichletPrior
from GenModels.research.PedigreeWrappers import PedigreeHMMFilter, PedigreeHMMFilterSexMatters
import numpy as np
from GenModels.GM.Models.DiscreteGraphModels import *

__all__ = [
    'autosomalDominantPriors',
    'autosomalRecessivePriors',
    'xLinkedRecessivePriors',
    'AutosomalDominant',
    'AutosomalRecessive',
    'XLinkedRecessive'
]

######################################################################

def autosomalTransitionPrior():
    # [ AA, Aa, aA, aa ] ( A is affected )
    # [ AA, Aa, aa ] ( A is affected )
    with_combo = np.array( [ [ [ 1.  , 0.  , 0.  , 0.   ],
                               [ 0.5 , 0.5 , 0.  , 0.   ],
                               [ 0.5 , 0.5 , 0.  , 0.   ],
                               [ 0.  , 1.  , 0.  , 0.   ] ],

                             [ [ 0.5 , 0.  , 0.5 , 0.   ],
                               [ 0.25, 0.25, 0.25, 0.25 ],
                               [ 0.25, 0.25, 0.25, 0.25 ],
                               [ 0.  , 0.5 , 0.  , 0.5  ] ],

                             [ [ 0.5 , 0.  , 0.5 , 0.   ],
                               [ 0.25, 0.25, 0.25, 0.25 ],
                               [ 0.25, 0.25, 0.25, 0.25 ],
                               [ 0.  , 0.5 , 0.  , 0.5  ] ],

                             [ [ 0.  , 0.  , 1.  , 0.   ],
                               [ 0.  , 0.  , 0.5 , 0.5  ],
                               [ 0.  , 0.  , 0.5 , 0.5  ],
                               [ 0.  , 0.  , 0.  , 1.   ] ] ] )

    without_combo = np.zeros_like( with_combo[ ..., [ 0, 1, 3 ] ] )
    without_combo[ ..., [ 0, 2 ] ] = with_combo[ ..., [ 0, 3 ] ]
    without_combo[ ..., [ 1 ] ] = with_combo[ ..., [ 1 ] ]  + with_combo[ ..., [ 2 ] ]

    without_combo = np.delete( without_combo, 1, axis=0 )
    without_combo = np.delete( without_combo, 1, axis=1 )

    return without_combo

def autosomalDominantEmissionPrior():
    # [ AA, Aa, aa ]
    # [ Not affected, affected ]
    return np.array( [ [ 0, 1 ],
                       [ 0, 1 ],
                       [ 1, 0 ] ] )

def autosomalRecessiveEmissionPrior():
    # [ AA, Aa, aa ]
    # [ Not affected, affected ]
    return np.array( [ [ 0, 1 ],
                       [ 1, 0 ],
                       [ 1, 0 ] ] )

######################################################################

def xLinkedFemaleTransitionPrior():
    # Female, male, female child
    # [ XX, Xx, xX, xx ] ( Index using males [ XY, xY ] )
    # [ XX, Xx, xx ] ( Index using males [ XY, xY ] )
    with_combo = np.array( [ [ [ 1. , 0. , 0. , 0.  ],
                               [ 0. , 0. , 1. , 0.  ] ],

                             [ [ 0.5, 0.5, 0. , 0.  ],
                               [ 0. , 0. , 0.5, 0.5 ] ],

                             [ [ 0.5, 0.5, 0. , 0.  ],
                               [ 0. , 0. , 0.5, 0.5 ] ],

                             [ [ 0. , 1. , 0. , 0.  ],
                               [ 0. , 0. , 0. , 1.  ] ] ] )

    without_combo = np.zeros_like( with_combo[ ..., [ 0, 1, 3 ] ] )
    without_combo[ ..., [ 0, 2 ] ] = with_combo[ ..., [ 0, 3 ] ]
    without_combo[ ..., [ 1 ] ] = with_combo[ ..., [ 1 ] ]  + with_combo[ ..., [ 2 ] ]

    without_combo = np.delete( without_combo, 1, axis=0 )

    return without_combo

def xLinkedMaleTransitionPrior():
    # Female, male, male child
    # [ XY, xY ] ( X is affected )
    return np.array( [ [ [ 1. , 0.  ],
                         [ 1. , 0.  ] ],

                       [ [ 0.5, 0.5 ],
                         [ 0.5, 0.5 ] ],

                       [ [ 0. , 1.  ],
                         [ 0. , 1.  ] ] ] )

def xLinkedUnknownTransitionPrior():
    # Female, male, unknown sex child
    # [ XX, Xx, xX, xx, XY, xY ]
    # [ XX, Xx, xx, XY, xY ]
    with_combo = np.array( [ [ [ 0.5 , 0.  , 0.  , 0.  , 0.5 , 0.   ] ,
                               [ 0.  , 0.  , 0.5 , 0.  , 0.5 , 0.   ] ] ,

                             [ [ 0.25, 0.25, 0.  , 0.  , 0.25, 0.25 ] ,
                               [ 0.  , 0.  , 0.25, 0.25, 0.25, 0.25 ] ] ,

                             [ [ 0.25, 0.25, 0.  , 0.  , 0.25, 0.25 ] ,
                               [ 0.  , 0.  , 0.25, 0.25, 0.25, 0.25 ] ] ,

                             [ [ 0.  , 0.5 , 0.  , 0.  , 0.  , 0.5  ] ,
                               [ 0.  , 0.  , 0.  , 0.5 , 0.  , 0.5  ] ] ] )

    without_combo = np.zeros_like( with_combo[ ..., [ 0, 1, 3, 4, 5 ] ] )
    without_combo[ ..., [ 0, 2, 3, 4 ] ] = with_combo[ ..., [ 0, 3, 4, 5 ] ]
    without_combo[ ..., [ 1 ] ] = with_combo[ ..., [ 1 ] ]  + with_combo[ ..., [ 2 ] ]

    without_combo = np.delete( without_combo, 1, axis=0 )


    return without_combo

# Going to ignore the ( unknown, unknown ) -> unknown case

def xLinkedFemaleEmissionPrior():
    # [ XX, Xx, xx ]
    # [ Not affected, affected ]
    return np.array( [ [ 0, 1 ],
                       [ 1, 0 ],
                       [ 1, 0 ] ] )

def xLinkedMaleEmissionPrior():
    # [ XY, xY ]
    # [ Not affected, affected ]
    return np.array( [ [ 0, 1 ],
                       [ 1, 0 ] ] )

def xLinkedUnknownEmissionPrior():
    # [ XX, Xx, xx, XY, xY ]
    # [ Not affected, affected ]
    return np.array( [ [ 0, 1 ],
                       [ 1, 0 ],
                       [ 1, 0 ],
                       [ 0, 1 ],
                       [ 1, 0 ] ] )

######################################################################

def autosomalDominantPriors( prior_strength=1.0 ):
    trans_prior = [ autosomalTransitionPrior() * prior_strength + 1 ]
    emiss_prior = autosomalDominantEmissionPrior() * prior_strength + 1
    return trans_prior, emiss_prior

def autosomalRecessivePriors( prior_strength=1.0 ):
    trans_prior = [ autosomalTransitionPrior() * prior_strength + 1 ]
    emiss_prior = autosomalRecessiveEmissionPrior() * prior_strength + 1
    return trans_prior, emiss_prior

######################################################################

class AutosomalDominant( GHMM ):
    def __init__( self, graphs, prior_strength=1.0, method='EM', priors=None, params=None, **kwargs ):

        # Generate the priors
        if( priors is None ):
            trans_prior, emiss_prior = autosomalDominantPriors( prior_strength )
        else:
            # Initialize using other priors
            trans_prior, emiss_prior = priors

        root_prior = np.ones( 3 )
        priors = ( root_prior, trans_prior, emiss_prior )

        super().__init__( graphs, prior_strength=prior_strength, method=method, priors=priors, params=params, **kwargs )

class AutosomalRecessive( GHMM ):
    def __init__( self, graphs, prior_strength=1.0, method='EM', priors=None, params=None, **kwargs ):

        # Generate the priors
        if( priors is None ):
            trans_prior, emiss_prior = autosomalRecessivePriors( prior_strength )
        else:
            # Initialize using other priors
            trans_prior, emiss_prior = priors

        root_prior = np.ones( 3 )
        priors = ( root_prior, trans_prior, emiss_prior )

        super().__init__( graphs, prior_strength=prior_strength, method=method, priors=priors, params=params, **kwargs )

######################################################################

def xLinkedRecessivePriors( prior_strength=1.0 ):
    female_trans_prior = xLinkedFemaleTransitionPrior() * prior_strength + 1
    male_trans_prior = xLinkedMaleTransitionPrior() * prior_strength + 1
    unknown_trans_prior = xLinkedUnknownTransitionPrior() * prior_strength + 1
    trans_priors = [ female_trans_prior, male_trans_prior, unknown_trans_prior ]

    female_emiss_prior = xLinkedFemaleEmissionPrior() * prior_strength + 1
    male_emiss_prior = xLinkedMaleEmissionPrior() * prior_strength + 1
    unknown_emiss_prior = xLinkedUnknownEmissionPrior() * prior_strength + 1
    emiss_priors = [ female_emiss_prior, male_emiss_prior, unknown_emiss_prior ]

    return trans_priors, emiss_priors

######################################################################

class XLinkedRecessive( GroupGHMM ):

    def __init__( self, graphs, prior_strength=1.0, method='SVI', priors=None, params=None, **kwargs ):
        assert method in [ 'EM', 'Gibbs', 'CAVI', 'SVI' ]
        self.graphs = graphs
        self.msg = PedigreeHMMFilterSexMatters()
        self.method = method

        # Generate the priors
        if( priors is None ):
            trans_priors, emiss_priors = xLinkedRecessivePriors( prior_strength )
        else:
            # Initialize using known priors
            trans_priors, emiss_priors = priors
            female_trans_prior, male_trans_prior, unknown_trans_prior = trans_priors
            female_emiss_prior, male_emiss_prior, unknown_emiss_prior = emiss_priors

        # Generate the parameters objects
        if( method == 'EM' ):
            self.params = XLinkedParametersEM( transition_priors=trans_priors, emission_priors=emiss_priors ) if params is None else params
        elif( method == 'Gibbs' ):
            self.params = XLinkedParametersGibbs( transition_priors=trans_priors, emission_priors=emiss_priors ) if params is None else params
        elif( method == 'CAVI' ):
            self.params = XLinkedParametersCAVI( transition_priors=trans_priors, emission_priors=emiss_priors ) if params is None else params
        else:
            self.params = XLinkedParametersSVI( transition_priors=trans_priors, emission_priors=emiss_priors ) if params is None else params

        # Generate the model objects
        if( method == 'EM' ):
            self.model = GroupEM( msg=self.msg, parameters=self.params )
        elif( method == 'Gibbs' ):
            self.model = GroupGibbs( msg=self.msg, parameters=self.params )
        elif( method == 'CAVI' ):
            self.model = GroupCAVI( msg=self.msg, parameters=self.params )
        else:
            step_size = kwargs[ 'step_size' ]
            self.minibatch_size = kwargs[ 'minibatch_size' ]
            minibatch_ratio = self.minibatch_size / len( self.graphs )
            self.model = GroupSVI( msg=self.msg, parameters=self.params, minibatch_ratio=minibatch_ratio, step_size=step_size )
            self.total_nodes = sum( [ len( graph.nodes ) for graph, fbs in self.graphs ] )

        if( method != 'SVI' ):
            self.msg.preprocessData( self.graphs )

    def stateUpdate( self, **kwargs ):

        if( self.method != 'SVI' ):
            return self.model.stateUpdate()

        minibatch_indices = np.random.randint( len( self.graphs ), size=self.minibatch_size )
        minibatch = [ self.graphs[ i ] for i in minibatch_indices ]
        self.msg.preprocessData( minibatch )

        # Compute minibatch ratio
        minibatch_nodes = sum( [ len( graph.nodes ) for graph, fbs in minibatch ] )
        self.params.setMinibatchRatio( self.total_nodes / minibatch_nodes )

        return self.model.stateUpdate()


    def fitStep( self, **kwargs ):

        if( self.method != 'SVI' ):
            return self.model.fitStep()

        minibatch_indices = np.random.randint( len( self.graphs ), size=self.minibatch_size )
        minibatch = [ self.graphs[ i ] for i in minibatch_indices ]
        self.msg.preprocessData( minibatch )

        # Compute minibatch ratio
        minibatch_nodes = sum( [ len( graph.nodes ) for graph, fbs in minibatch ] )
        self.params.setMinibatchRatio( self.total_nodes / minibatch_nodes )

        return self.model.fitStep()

    def sampleParams( self ):
        self.msg.updateParams( self.params.sampleInitialDist(), self.params.sampleTransitionDist(), self.params.sampleEmissionDist() )

    def marginal( self ):
        U, V = self.msg.filter()
        return self.msg.marginalProb( U, V )

    def generative( self ):
        assert self.method == 'Gibbs'
        return self.model.generativeProbability()

    def paramProb( self ):
        return self.params.paramProb()
