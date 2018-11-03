from GenModels.GM.Distributions import Categorical, Dirichlet, TensorTransition, TensorTransitionDirichletPrior
import autograd.numpy as np
from GenModels.GM.Models.DiscreteGraphModels import *
import copy
import matplotlib
import matplotlib.pyplot as plt
from .PedigreeWrappers import createDataset
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
from GenModels.GM.Utility import logsumexp, monitored_adam
from autograd import grad, value_and_grad
from autograd.misc.optimizers import adam
import os
from autograd.misc import flatten
import pickle

__all__ = [
    'autosomalDominantPriors',
    'autosomalRecessivePriors',
    'xLinkedRecessivePriors',
    'AutosomalDominant',
    'AutosomalRecessive',
    'XLinkedRecessive',
    'InheritancePatternSVAE',
    'InheritancePatternDES'
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
            style[ 'fillcolor' ] = matplotlib.colors.to_hex( plt.get_cmap( 'Blues' )( carrier_prob ) )
            style[ 'fontcolor' ] = 'white' if carrier_prob > 0.55 else 'black'
            # style[ 'label' ] = str( [ '%2.1f'%p for p in prob ] )
            # style[ 'label' ] = ''

            styles[ n ] = style

        print( 'Node [AA,Aa,aa]' )
        for n in graph.nodes:
            with np.printoptions( precision=3, suppress=True ):
                print( '%4d'%n, np.exp( probs[ n ] ) )
        # print( 'Male [XY,xY]' )
        # for n in graph.nodes:
        #     if( graph.attrs[ n ][ 'sex' ] == 'male' ):
        #         with np.printoptions( precision=3, suppress=True ):
        #             print( '%4d'%n, np.exp( probs[ n ] ) )
        # print( 'Female [XX,Xx,xx]' )
        # for n in graph.nodes:
        #     if( graph.attrs[ n ][ 'sex' ] == 'female' ):
        #         with np.printoptions( precision=3, suppress=True ):
        #             print( '%6d'%n, np.exp( probs[ n ] ) )
        # print( 'Unknown [XX,  Xx,  xx,  XY,  xY  ]' )
        # for n in graph.nodes:
        #     if( graph.attrs[ n ][ 'sex' ] == 'unknown' ):
        #         with np.printoptions( precision=2, suppress=True ):
        #             print( '%7d'%n, np.exp( probs[ n ] ) )


        return graph.draw( _custom_args={ 'render': True, 'styles': styles, 'node_to_style_key': node_to_style_key } )

######################################################################

class _autosomalDrawMixin( _drawMixin ):

    def draw( self, graph_index=0, show_carrier_prob=True ):
        def probCarrierFunc( sex, prob ):
            return ( 2*prob[ 0 ] + prob[ 1 ] ) / 2
        return self._draw( graph_index=graph_index, show_carrier_prob=show_carrier_prob, probCarrierFunc=probCarrierFunc )

class _xlDrawMixin( _drawMixin ):

    def draw( self, graph_index=0, show_carrier_prob=True ):
        def probCarrierFunc( sex, prob ):
            if( sex == 'female' ):
                return ( 2*prob[ 0 ] + prob[ 1 ] ) / 2
            elif( sex == 'male' ):
                return prob[ 0 ]
            else:
                return ( 2*prob[ 0 ] + prob[ 1 ] + prob[ 3 ] ) / 3
        return self._draw( graph_index=graph_index, show_carrier_prob=show_carrier_prob, probCarrierFunc=probCarrierFunc )

######################################################################

class AutosomalDominant( _autosomalDrawMixin, GHMM ):
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

class AutosomalRecessive( _autosomalDrawMixin, GHMM ):
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

######################################################################

class XLinkedRecessive( _xlDrawMixin, GroupGHMM ):
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

######################################################################

class AutosomalSVAE( _autosomalDrawMixin, GSVAE ):
    def __init__( self, graphs=None, root_strength=1.0, prior_strength=1.0, priors=None, **kwargs ):

        # Generate the priors
        if( priors is None ):
            trans_prior, _ = autosomalRecessivePriors( prior_strength )
        else:
            # Initialize using other priors
            trans_prior = priors

        root_prior = np.ones( 3 )
        root_prior[ 2 ] = root_strength
        priors = ( root_prior, trans_prior )

        super().__init__( graphs, prior_strength=prior_strength, priors=priors, d_obs=2, **kwargs )

class XLinkedSVAE( _xlDrawMixin, GroupGSVAE ):
    def __init__( self, graphs=None, root_strength=1.0, prior_strength=1.0, priors=None, **kwargs ):

        # Generate the priors
        if( priors is None ):
            trans_priors, _ = xLinkedRecessivePriors( prior_strength )
        else:
            # Initialize using known priors
            trans_priors = priors
            female_trans_prior, male_trans_prior, unknown_trans_prior = trans_priors

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

        priors = ( root_priors, trans_priors )
        super().__init__( graphs, prior_strength=prior_strength, priors=priors, d_obs=2, **kwargs )

######################################################################

class AutosomalDominantDES( _autosomalDrawMixin, DES ):
    def __init__( self, graphs=None, root_strength=1.0, prior_strength=1.0, priors=None, **kwargs ):

        # Generate the priors
        if( priors is None ):
            trans_prior, _ = autosomalRecessivePriors( prior_strength )
        else:
            # Initialize using other priors
            trans_prior = priors

        root_prior = np.ones( 3 )
        root_prior[ 2 ] = root_strength
        priors = ( root_prior, trans_prior )

        super().__init__( graphs, prior_strength=prior_strength, priors=priors, d_obs=2, inheritance_pattern='AD', **kwargs )

class AutosomalRecessiveDES( _autosomalDrawMixin, DES ):
    def __init__( self, graphs=None, root_strength=1.0, prior_strength=1.0, priors=None, **kwargs ):

        # Generate the priors
        if( priors is None ):
            trans_prior, _ = autosomalRecessivePriors( prior_strength )
        else:
            # Initialize using other priors
            trans_prior = priors

        root_prior = np.ones( 3 )
        root_prior[ 2 ] = root_strength
        priors = ( root_prior, trans_prior )

        super().__init__( graphs, prior_strength=prior_strength, priors=priors, d_obs=2, inheritance_pattern='AR', **kwargs )

class XLinkedDES( _xlDrawMixin, GroupDES ):
    def __init__( self, graphs=None, root_strength=1.0, prior_strength=1.0, priors=None, **kwargs ):

        # Generate the priors
        if( priors is None ):
            trans_priors, _ = xLinkedRecessivePriors( prior_strength )
        else:
            # Initialize using known priors
            trans_priors = priors
            female_trans_prior, male_trans_prior, unknown_trans_prior = trans_priors

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

        priors = ( root_priors, trans_priors )
        super().__init__( graphs, prior_strength=prior_strength, priors=priors, d_obs=2, inheritance_pattern='XL', **kwargs )

######################################################################

class InheritancePatternTrainer():

    def updateCurrentGraphAndLabel( self, training=True, index=None, graph=None ):

        if( graph is None ):

            # Choose the next graph
            if( training == True ):
                random_index = int( np.random.random() * len( self.training_set ) ) if index is None else index
                graph = self.training_set[ random_index ]
            else:
                random_index = int( np.random.random() * len( self.test_set ) ) if index is None else index
                graph = self.test_set[ random_index ]

        # Generate the label with label smoothing
        label = graph[ 0 ].inheritancePattern
        # label_one_hot = np.zeros( 3 ) + 0.1
        # label_one_hot[ [ 'AD', 'AR', 'XL' ].index( label ) ] = 0.8
        label_one_hot = np.zeros( 3 )
        label_one_hot[ [ 'AD', 'AR', 'XL' ].index( label ) ] = 1.0
        self.current_label_one_hot = label_one_hot

        # Update the models
        ad_graph, ar_graph, xl_graph = createDataset( [ graph ] )

        self.ad_model.msg.preprocessData( ad_graph )
        self.ar_model.msg.preprocessData( ar_graph )
        self.xl_model.msg.preprocessData( xl_graph )

    #####################################################################

    def inheritancePatternPrediction( self, model_parameters, graph=None, mc_samples=10 ):
        prediction_logits = self.inheritancePatternLogits( model_parameters, graph=graph, mc_samples=mc_samples )
        prediction_probs = prediction_logits - logsumexp( prediction_logits )
        return prediction_probs

    #####################################################################

    @property
    def labels( self ):
        return [ 'AD', 'AR', 'XL' ]

    @property
    def trainable_params( self ):

        ad_params = ( self.ad_model.params.emission_dist.recognizer_params, self.ad_model.params.emission_dist.generative_hyper_params )
        ar_params = ( self.ar_model.params.emission_dist.recognizer_params, self.ar_model.params.emission_dist.generative_hyper_params )

        xl_params = ( {}, {} )
        for group, dist in self.xl_model.params.emission_dists.items():
            xl_params[ 0 ][ group ] = dist.recognizer_params
            xl_params[ 1 ][ group ] = dist.generative_hyper_params

        return ( ad_params, ar_params, xl_params )

    #####################################################################

    def predict( self, params, graph_and_fbs, mc_samples ):
        probs = self.inheritancePatternPrediction( params, graph=graph_and_fbs, mc_samples=mc_samples )
        return self.labels[ np.argmax( probs ) ]

    #####################################################################

    def printMetrics( self, true_labels, predicted_labels, confident_mask=None, labels=None, i=None ):
        labels = labels if labels is not None else self.labels
        true_labels = np.array( true_labels )
        predicted_labels = np.array( predicted_labels )

        cm = confusion_matrix( y_true=true_labels, y_pred=predicted_labels, labels=labels )
        ck = cohen_kappa_score( y1=true_labels, y2=predicted_labels, labels=labels )
        accuracy = accuracy_score( y_true=true_labels, y_pred=predicted_labels )
        ad_accuracy = accuracy_score( y_true=true_labels[ true_labels == 0 ], y_pred=predicted_labels[ true_labels == 0 ] )
        ar_accuracy = accuracy_score( y_true=true_labels[ true_labels == 1 ], y_pred=predicted_labels[ true_labels == 1 ] )
        xl_accuracy = accuracy_score( y_true=true_labels[ true_labels == 2 ], y_pred=predicted_labels[ true_labels == 2 ] )

        if( i is not None ):
            print( '\n-----------------\ni', i )
        print( 'Confusion matrix:\n', cm )
        print( 'Cohen kappa:', ck )
        print( 'Accuracy:', accuracy )
        print( 'AD accuracy:', ad_accuracy )
        print( 'AR accuracy:', ar_accuracy )
        print( 'XL accuracy:', xl_accuracy )

        # if( confident_mask is not None ):

        #     cm = confusion_matrix( y_true=true_labels[ confident_mask ], y_pred=predicted_labels[ confident_mask ], labels=labels )
        #     ck = cohen_kappa_score( y1=true_labels[ confident_mask ], y2=predicted_labels[ confident_mask ], labels=labels )
        #     accuracy = accuracy_score( y_true=true_labels[ confident_mask ], y_pred=predicted_labels[ confident_mask ] )

        #     ad_accuracy = accuracy_score( y_true=true_labels[ confident_mask ][ true_labels[ confident_mask ] == 0 ], y_pred=predicted_labels[ confident_mask ][ true_labels[ confident_mask ] == 0 ] )
        #     ar_accuracy = accuracy_score( y_true=true_labels[ confident_mask ][ true_labels[ confident_mask ] == 1 ], y_pred=predicted_labels[ confident_mask ][ true_labels[ confident_mask ] == 1 ] )
        #     xl_accuracy = accuracy_score( y_true=true_labels[ confident_mask ][ true_labels[ confident_mask ] == 2 ], y_pred=predicted_labels[ confident_mask ][ true_labels[ confident_mask ] == 2 ] )

        #     print( '\nConfident confusion matrix:\n', cm )
        #     print( 'Confident cohen kappa:', ck )
        #     print( 'Confident accuracy:', accuracy )
        #     print( 'Confident AD accuracy:', ad_accuracy )
        #     print( 'Confident AR accuracy:', ar_accuracy )
        #     print( 'Confident XL accuracy:', xl_accuracy )

    #####################################################################

    def getTrainableParamsFromCheckpoint( self, filename='saved_params.p', use_backup=True ):
        if( use_backup and os.path.isfile( 'saved_params.p' ) ):
            with open( 'saved_params.p', 'rb' ) as file:
                trainable_params = pickle.load( file )
                print( 'Using last checkpoint' )
            return trainable_params
        else:
            return self.trainable_params

    def saveParamsToCheckpoint( self, params, filename='saved_params.p' ):
        with open( 'saved_params.p', 'wb' ) as file:
            pickle.dump( params, file )

######################################################################

class InheritancePatternSVAE( InheritancePatternTrainer ):

    def __init__( self, training_graphs, test_graphs, root_strength=100000000000, prior_strength=100000000000, **kwargs ):

        self.ad_model = AutosomalSVAE( graphs=None, root_strength=root_strength, prior_strength=prior_strength, **kwargs )
        self.ar_model = AutosomalSVAE( graphs=None, root_strength=root_strength, prior_strength=prior_strength, **kwargs )
        self.xl_model = XLinkedSVAE( graphs=None, root_strength=root_strength, prior_strength=prior_strength, **kwargs )

        self.training_set = training_graphs
        self.test_set = test_graphs

        self.k = 1.0

        # Set the first graph
        self.updateCurrentGraphAndLabel( training=True )

    #####################################################################

    def inheritancePatternLogits( self, model_parameters, graph=None, mc_samples=10 ):

        if( graph is not None ):
            self.updateCurrentGraphAndLabel( graph=graph )

        ad_emission_params, ar_emission_params, xl_emission_params = model_parameters

        all_logits = []
        for i in range( mc_samples ):
            ad_loss = self.ad_model.opt.computeLoss( ad_emission_params, 0 )
            ar_loss = self.ar_model.opt.computeLoss( ar_emission_params, 0 )
            xl_loss = self.xl_model.opt.computeLoss( xl_emission_params, 0 )

            prediction_logits = np.array( [ ad_loss, ar_loss, xl_loss ] )
            all_logits.append( prediction_logits )

        return logsumexp( np.array( all_logits ), axis=0 )

    #####################################################################

    def fullLoss( self, full_params, n_iter=0 ):

        # Each element in the set of logits is both the SVAE lower bound on P( Y ) and what we are
        # going to use to predict the inheritance pattern
        prediction_logits = self.inheritancePatternLogits( full_params, graph=None, mc_samples=1 )
        ad_loss, ar_loss, xl_loss = -prediction_logits

        # Compute the prediction loss.  We really want P( Y ) for each model,
        # but will settle for the lower bound given by the SVAE loss
        prediction_loss = -np.sum( self.current_label_one_hot * prediction_logits )

        return ad_loss + ar_loss + xl_loss + self.k * prediction_loss

    #####################################################################

    def train( self, num_iters=100 ):

        trainable_params = self.getTrainableParamsFromCheckpoint( 'saved_params.p' )

        # Callback to run the test set and update the next graph
        def callback( full_params, i, g ):

            if( i and i % 100 == 0 ):
                self.saveParamsToCheckpoint( full_params )

            # Every 500 steps, run the algorithm on the test set
            if( i % 500 == 0 ):
                true_labels, predicted_labels = [], []
                for graph_and_fbs in self.test_set:
                    if( np.random.random() < 0.3 ):
                        continue
                    true_labels.append( graph_and_fbs[ 0 ].inheritancePattern )
                    predicted_labels.append( self.predict( full_params, graph_and_fbs, 1 ) )

                # Print the confusion matrix and kappa score on the test set
                self.printMetrics( true_labels, predicted_labels )

            # Swap out the current graph and update the current label
            self.updateCurrentGraphAndLabel( training=True )

        # Optimize
        grads = grad( self.fullLoss )
        final_params = adam( grads, trainable_params, num_iters=num_iters, callback=callback )
        return final_params

######################################################################

class InheritancePatternDES( InheritancePatternTrainer ):

    def __init__( self, training_graphs, test_graphs, root_strength=100000000000, prior_strength=100000000000, **kwargs ):

        self.ad_model = AutosomalDominantDES( graphs=None, root_strength=root_strength, prior_strength=prior_strength, **kwargs )
        self.ar_model = AutosomalRecessiveDES( graphs=None, root_strength=root_strength, prior_strength=prior_strength, **kwargs )
        self.xl_model = XLinkedDES( graphs=None, root_strength=root_strength, prior_strength=prior_strength, **kwargs )

        self.training_set = training_graphs
        self.test_set = test_graphs

        self.k = 1.0

        # Set the first graph
        self.updateCurrentGraphAndLabel( training=True )

    #####################################################################

    @property
    def trainable_params( self ):

        ad_params = self.ad_model.params.emission_dist.recognizer_params
        ar_params = self.ar_model.params.emission_dist.recognizer_params

        xl_params = {}
        for group, dist in self.xl_model.params.emission_dists.items():
            xl_params[ group ] = dist.recognizer_params

        return ( ad_params, ar_params, xl_params )

    #####################################################################

    def inheritancePatternLogits( self, model_parameters, graph=None, mc_samples=10 ):

        if( graph is not None ):
            self.updateCurrentGraphAndLabel( graph=graph )

        ad_emission_params, ar_emission_params, xl_emission_params = model_parameters

        all_logits = []
        for i in range( mc_samples ):
            ad_loss = self.ad_model.opt.marginalLoss( ad_emission_params, 0 )
            ar_loss = self.ar_model.opt.marginalLoss( ar_emission_params, 0 )
            xl_loss = self.xl_model.opt.marginalLoss( xl_emission_params, 0 )

            prediction_logits = np.array( [ ad_loss, ar_loss, xl_loss ] )
            all_logits.append( prediction_logits )

        return logsumexp( np.array( all_logits ), axis=0 )

    #####################################################################

    def lossFunction( self, full_params, n_iter=0 ):

        assert 0, 'Not using this'

        prediction_logits = self.inheritancePatternLogits( full_params, graph=None, mc_samples=1 )
        ad_loss, ar_loss, xl_loss = -prediction_logits

        # Compute the prediction loss
        prediction_loss = -np.sum( self.current_label_one_hot * prediction_logits )

        return ad_loss + ar_loss + xl_loss + self.k * prediction_loss

    def categoricalLossGrad( self, prediction_logits ):

        # Normalize the logits
        norm_factor = logsumexp( prediction_logits )
        prediction_logits_norm = prediction_logits - norm_factor
        loss = -np.sum( self.current_label_one_hot * prediction_logits_norm )

        return -np.exp( prediction_logits_norm - norm_factor )

    def svmLossGrad( self, prediction_logits, margin=3 ):

        label = np.argmax( self.current_label_one_hot )

        # Compare the logits for the svm loss.  Going to use
        # a margin of 3 because we should consider something an
        # outlier if the model strongly thinks that a datapoint is
        # another class
        correct = prediction_logits[ label ]
        loss = 0.0
        svm_grad = np.zeros( 3 )
        for j in range( 3 ):
            if( j == label ):
                continue
            val = prediction_logits[ j ] - correct + margin
            if( val > 0 ):
                loss += val
                svm_grad[ j ] += 1
                svm_grad[ label ] -= 1

        return svm_grad

    def lossGrad( self, prediction_logits, full_params, return_flat=False, loss_type='hinge' ):

        if( loss_type == 'predictive' ):
            g_loss = self.categoricalLossGrad( prediction_logits )
        elif( loss_type == 'hinge' ):
            g_loss = self.svmLossGrad( prediction_logits )

        # d_loss_d_ad = self.current_label_one_hot[ 0 ] * g_loss[ 0 ]
        # d_loss_d_ar = self.current_label_one_hot[ 1 ] * g_loss[ 1 ]
        # d_loss_d_xl = self.current_label_one_hot[ 2 ] * g_loss[ 2 ]

        d_loss_d_ad = g_loss[ 0 ]
        d_loss_d_ar = g_loss[ 1 ]
        d_loss_d_xl = g_loss[ 2 ]

        ad_emission_params, ar_emission_params, xl_emission_params = full_params

        _ad_g, ad_unflatten = self.ad_model.opt.marginalLossGrad( ad_emission_params, 0, run_smoother=False, return_flat=True )
        _ar_g, ar_unflatten = self.ar_model.opt.marginalLossGrad( ar_emission_params, 0, run_smoother=False, return_flat=True )
        _xl_g, xl_unflatten = self.xl_model.opt.marginalLossGrad( xl_emission_params, 0, run_smoother=False, return_flat=True )

        if( return_flat ):
            ad_g = d_loss_d_ad * _ad_g, ad_unflatten
            ar_g = d_loss_d_ar * _ar_g, ar_unflatten
            xl_g = d_loss_d_xl * _xl_g, xl_unflatten
        else:
            ad_g = ad_unflatten( d_loss_d_ad * _ad_g )
            ar_g = ar_unflatten( d_loss_d_ar * _ar_g )
            xl_g = xl_unflatten( d_loss_d_xl * _xl_g )

        return ad_g, ar_g, xl_g

    #####################################################################

    def trainStep( self, params, g_shape, i, batch_size=10 ):

        total_g = np.zeros( g_shape )
        predicted = []
        trues = []
        confident_mask = []
        for b in range( batch_size ):

            logits = self.inheritancePatternLogits( params, graph=None, mc_samples=1 )
            ( ad_g, ad_uf ), ( ar_g, ar_uf ), ( xl_g, xl_uf ) = self.lossGrad( logits, params, return_flat=True )

            g = np.hstack( [ ad_g, ar_g, xl_g ] )
            total_g += g

            if( np.max( np.exp( logits ) / np.sum( np.exp( logits ) ) ) > 0.8 ):
                confident_mask.append( True )
            else:
                confident_mask.append( False )
            predicted.append( np.argmax( logits ) )
            trues.append( np.argmax( self.current_label_one_hot ) )

            self.updateCurrentGraphAndLabel( training=True )

        self.printMetrics( trues, predicted, confident_mask=confident_mask, labels=[ 0, 1, 2 ], i=i )

        return total_g

    def trainNonAutogradAdam( self, num_iters=100, step_size=0.001, b1=0.9, b2=0.999, eps=10**-8, lambd=0.0001 ):
        # Adam opt implementation taken from autograd.misc.optimizers

        # params = self.getTrainableParamsFromCheckpoint( 'saved_params.p', use_backup=True )
        params = self.getTrainableParamsFromCheckpoint( 'saved_params_use_in_cv.p', use_backup=True )

        flat_params, unflatten_params = flatten( params )

        m = np.zeros( len( flat_params ) )
        v = np.zeros( len( flat_params ) )
        for i in range( num_iters ):

            # Save the current parameteres
            if( i and i % 50 == 0 ):
                self.saveParamsToCheckpoint( params, filename='saved_params.p' )

            # Run the algorithm on the test set
            if( i and i % 50 == 0 ):
                true_labels, predicted_labels = [], []
                for graph_and_fbs in self.test_set:
                    true_labels.append( self.labels.index( graph_and_fbs[ 0 ].inheritancePattern ) )
                    predicted_labels.append( self.labels.index( self.predict( params, graph_and_fbs, 1 ) ) )

                # Print the confusion matrix and kappa score on the test set
                self.printMetrics( true_labels, predicted_labels, labels=[ 0, 1, 2 ], i=0 )

            # Training step
            g = self.trainStep( params, flat_params.shape, i=i, batch_size=1 )

            # L2 Regularization.  Don't bother only applying this to the weight parameters
            g += lambd * 2 * flat_params

            # Adam algorithm (from autograd.misc.optimizers.py)
            m = ( 1 - b1 ) * g        + b1 * m
            v = ( 1 - b2 ) * ( g**2 ) + b2 * v
            mhat = m / ( 1 - b1**( i + 1 ) )
            vhat = v / ( 1 - b2**( i + 1 ) )

            d_params = step_size * mhat / ( np.sqrt( vhat ) + eps )

            print()
            print( np.min( flat_params ), np.max( flat_params ) )
            print( 'Grad max', np.max( np.abs( d_params ) ), '(', np.max( np.abs( g ) ), ')' )

            flat_params = flat_params - d_params
            params = unflatten_params( flat_params )
            flat_params, _ = flatten( params )


        return params

    #####################################################################

    def train( self, num_iters=100 ):

        trainable_params = self.getTrainableParamsFromCheckpoint( 'saved_params.p' )

        # Callback to run the test set and update the next graph
        def callback( full_params, i, g ):

            if( i and i % 100 == 0 ):
                self.saveParamsToCheckpoint( full_params )

            # Every 500 steps, run the algorithm on the test set
            if( i % 500 == 0 ):
                true_labels, predicted_labels = [], []
                for graph_and_fbs in self.test_set:
                    if( np.random.random() < 0.3 ):
                        continue
                    true_labels.append( graph_and_fbs[ 0 ].inheritancePattern )
                    predicted_labels.append( self.predict( full_params, graph_and_fbs, 1 ) )

                # Print the confusion matrix and kappa score on the test set
                self.printMetrics( true_labels, predicted_labels )

            # Swap out the current graph and update the current label
            self.updateCurrentGraphAndLabel( training=True )

        # Optimize
        grads = grad( self.lossFunction )
        final_params = adam( grads, svae_params, num_iters=num_iters, callback=callback )
        return final_params
