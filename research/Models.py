from GenModels.GM.Distributions import Categorical, Dirichlet, TensorTransition, TensorTransitionDirichletPrior
import autograd.numpy as np
from GenModels.GM.Models.DiscreteGraphModels import *
import copy
import matplotlib
import matplotlib.pyplot as plt

__all__ = [
    'autosomalDominantPriors',
    'autosomalRecessivePriors',
    'xLinkedRecessivePriors',
    'AutosomalDominant',
    'AutosomalRecessive',
    'XLinkedRecessive'
]

NO_DIAGNOSIS_PROB = 0.0

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
    # return np.array( [ [ 0, 1 ],
    #                    [ 0, 1 ],
    #                    [ 1, 0 ] ] )
    return np.array( [ [ NO_DIAGNOSIS_PROB, 1-NO_DIAGNOSIS_PROB ],
                       [ NO_DIAGNOSIS_PROB, 1-NO_DIAGNOSIS_PROB ],
                       [ 1, 0 ] ] )

def autosomalRecessiveEmissionPrior():
    # [ AA, Aa, aa ]
    # [ Not affected, affected ]
    # return np.array( [ [ 0, 1 ],
    #                    [ 1, 0 ],
    #                    [ 1, 0 ] ] )
    return np.array( [ [ NO_DIAGNOSIS_PROB, 1-NO_DIAGNOSIS_PROB ],
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
    # return np.array( [ [ 0, 1 ],
    #                    [ 1, 0 ],
    #                    [ 1, 0 ] ] )
    return np.array( [ [ NO_DIAGNOSIS_PROB, 1-NO_DIAGNOSIS_PROB ],
                       [ 1, 0 ],
                       [ 1, 0 ] ] )

def xLinkedMaleEmissionPrior():
    # [ XY, xY ]
    # [ Not affected, affected ]
    # return np.array( [ [ 0, 1 ],
    #                    [ 1, 0 ] ] )
    return np.array( [ [ NO_DIAGNOSIS_PROB, 1-NO_DIAGNOSIS_PROB ],
                       [ 1, 0 ] ] )

def xLinkedUnknownEmissionPrior():
    # [ XX, Xx, xx, XY, xY ]
    # [ Not affected, affected ]
    # return np.array( [ [ 0, 1 ],
    #                    [ 1, 0 ],
    #                    [ 1, 0 ],
    #                    [ 0, 1 ],
    #                    [ 1, 0 ] ] )
    return np.array( [ [ NO_DIAGNOSIS_PROB, 1-NO_DIAGNOSIS_PROB ],
                       [ 1, 0 ],
                       [ 1, 0 ],
                       [ NO_DIAGNOSIS_PROB, 1-NO_DIAGNOSIS_PROB ],
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

class _drawMixin():

    def _draw( self, graph_index=0, show_carrier_prob=True, probCarrierFunc=None ):
        graph = self.graphs[ 0 ][ 0 ]
        if( show_carrier_prob == False ):
            return graph.draw()

        assert probCarrierFunc is not None

        U, V = self.msg.filter()
        probs = dict( self.msg.nodeSmoothed( U, V, graph.nodes ) )

        # Have a unique style for each node
        node_to_style_key = dict( [ ( n, n ) for n in graph.nodes ] )
        styles = {}

        # The base outline styles
        male_style = dict( shape='square', style='filled' )
        female_style = dict( shape='circle', style='filled' )
        unknown_style = dict( shape='diamond', style='filled' )
        affected_male_style = dict( shape='square', fontcolor='black', style='filled', color='blue' )
        affected_female_style = dict( shape='circle', fontcolor='black', style='filled', color='blue' )
        affected_unknown_style = dict( shape='diamond', fontcolor='black', style='filled', color='blue' )

        # Get the style for each node
        for n in graph.nodes:

            # Get the base style
            attrs = graph.attrs[ n ]
            if( attrs[ 'sex' ] == 'male' ):
                if( attrs[ 'affected' ] == True ):
                    style = copy.deepcopy( affected_male_style )
                else:
                    style = copy.deepcopy( male_style )
            elif( attrs[ 'sex' ] == 'female' ):
                if( attrs[ 'affected' ] == True ):
                    style = copy.deepcopy( affected_female_style )
                else:
                    style = copy.deepcopy( female_style )
            else:
                if( attrs[ 'affected' ] == True ):
                    style = copy.deepcopy( affected_unknown_style )
                else:
                    style = copy.deepcopy( unknown_style )

            # Get the probability of being a carrier
            prob = np.exp( probs[ n ] )
            carrier_prob = probCarrierFunc( attrs[ 'sex' ], prob )
            style[ 'fillcolor' ] = matplotlib.colors.to_hex( plt.get_cmap( 'Reds' )( carrier_prob ) )

            styles[ n ] = style

        return graph.draw( _custom_args={ 'render': True, 'styles': styles, 'node_to_style_key': node_to_style_key } )


######################################################################

class AutosomalDominant( _drawMixin, GHMM ):
    def __init__( self, graphs=None, root_strength=1.0, prior_strength=1.0, method='EM', priors=None, params=None, **kwargs ):

        # Generate the priors
        if( priors is None ):
            trans_prior, emiss_prior = autosomalDominantPriors( prior_strength )
        else:
            # Initialize using other priors
            trans_prior, emiss_prior = priors

        root_prior = np.ones( 3 )
        root_prior[ 2 ] = root_strength
        priors = ( root_prior, trans_prior, emiss_prior )

        super().__init__( graphs, prior_strength=prior_strength, method=method, priors=priors, params=params, **kwargs )

    def draw( self, graph_index=0, show_carrier_prob=True ):
        def probCarrierFunc( sex, prob ):
            return prob[ 0 ] + prob[ 1 ]
        return self._draw( graph_index=graph_index, show_carrier_prob=show_carrier_prob, probCarrierFunc=probCarrierFunc )

class AutosomalRecessive( _drawMixin, GHMM ):
    def __init__( self, graphs=None, root_strength=1.0, prior_strength=1.0, method='EM', priors=None, params=None, **kwargs ):

        # Generate the priors
        if( priors is None ):
            trans_prior, emiss_prior = autosomalRecessivePriors( prior_strength )
        else:
            # Initialize using other priors
            trans_prior, emiss_prior = priors

        root_prior = np.ones( 3 )
        root_prior[ 2 ] = root_strength
        priors = ( root_prior, trans_prior, emiss_prior )

        super().__init__( graphs, prior_strength=prior_strength, method=method, priors=priors, params=params, **kwargs )

    def draw( self, graph_index=0, show_carrier_prob=True ):
        def probCarrierFunc( sex, prob ):
            return prob[ 0 ] + prob[ 0 ]
        return self._draw( graph_index=graph_index, show_carrier_prob=show_carrier_prob, probCarrierFunc=probCarrierFunc )

######################################################################

class XLinkedRecessive( _drawMixin, GroupGHMM ):
    def __init__( self, graphs=None, root_strength=1.0, prior_strength=1.0, method='SVI', priors=None, params=None, **kwargs ):

        # Generate the priors
        if( priors is None ):
            trans_priors, emiss_priors = xLinkedRecessivePriors( prior_strength )
        else:
            # Initialize using known priors
            trans_priors, emiss_priors = priors
            female_trans_prior, male_trans_prior, unknown_trans_prior = trans_priors
            female_emiss_prior, male_emiss_prior, unknown_emiss_prior = emiss_priors

        groups = [ 0, 1, 2 ]

        female_root_prior = np.ones( 3 )
        female_root_prior[ 2 ] = root_strength

        male_root_prior = np.ones( 2 )
        male_root_prior[ 1 ] = root_strength

        unknown_root_prior = np.ones( 5 )
        unknown_root_prior[ 2 ] = root_strength
        unknown_root_prior[ 4 ] = root_strength

        root_priors = { 0: female_root_prior, 1: male_root_prior, 2: unknown_root_prior }
        trans_priors = dict( [ ( group, [ dist ] ) for group, dist in zip( groups, trans_priors ) ] )
        emiss_priors = dict( [ ( group, dist ) for group, dist in zip( groups, emiss_priors ) ] )

        priors = ( root_priors, trans_priors, emiss_priors )
        super().__init__( graphs, prior_strength=prior_strength, method=method, priors=priors, params=params, **kwargs )

    def draw( self, graph_index=0, show_carrier_prob=True ):
        def probCarrierFunc( sex, prob ):
            if( sex == 'female' ):
                return prob[ 0 ] + prob[ 1 ]
            elif( sex == 'male' ):
                return prob[ 0 ]
            else:
                return prob[ 0 ] + prob[ 1 ] + prob[ 3 ]
        return self._draw( graph_index=graph_index, show_carrier_prob=show_carrier_prob, probCarrierFunc=probCarrierFunc )
