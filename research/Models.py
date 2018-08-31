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

class AutosomalDominant( GHMM ):
    def __init__( self, graphs=None, prior_strength=1.0, method='EM', priors=None, params=None, **kwargs ):

        # Generate the priors
        if( priors is None ):
            trans_prior, emiss_prior = autosomalDominantPriors( prior_strength )
        else:
            # Initialize using other priors
            trans_prior, emiss_prior = priors

        root_prior = np.ones( 3 ) * prior_strength
        priors = ( root_prior, trans_prior, emiss_prior )

        super().__init__( graphs, prior_strength=prior_strength, method=method, priors=priors, params=params, **kwargs )

class AutosomalRecessive( GHMM ):
    def __init__( self, graphs=None, prior_strength=1.0, method='EM', priors=None, params=None, **kwargs ):

        # Generate the priors
        if( priors is None ):
            trans_prior, emiss_prior = autosomalRecessivePriors( prior_strength )
        else:
            # Initialize using other priors
            trans_prior, emiss_prior = priors

        root_prior = np.ones( 3 ) * prior_strength
        priors = ( root_prior, trans_prior, emiss_prior )

        super().__init__( graphs, prior_strength=prior_strength, method=method, priors=priors, params=params, **kwargs )

######################################################################

class XLinkedRecessive( GroupGHMM ):
    def __init__( self, graphs=None, prior_strength=1.0, method='SVI', priors=None, params=None, **kwargs ):

        # Generate the priors
        if( priors is None ):
            trans_priors, emiss_priors = xLinkedRecessivePriors( prior_strength )
        else:
            # Initialize using known priors
            trans_priors, emiss_priors = priors
            female_trans_prior, male_trans_prior, unknown_trans_prior = trans_priors
            female_emiss_prior, male_emiss_prior, unknown_emiss_prior = emiss_priors

        groups = [ 0, 1, 2 ]
        root_priors = { 0: np.ones( 3 ) * prior_strength, 1: np.ones( 2 ) * prior_strength, 2: np.ones( 5 ) * prior_strength }
        trans_priors = dict( [ ( group, [ dist ] ) for group, dist in zip( groups, trans_priors ) ] )
        emiss_priors = dict( [ ( group, dist ) for group, dist in zip( groups, emiss_priors ) ] )

        priors = ( root_priors, trans_priors, emiss_priors )
        super().__init__( graphs, prior_strength=prior_strength, method=method, priors=priors, params=params, **kwargs )
