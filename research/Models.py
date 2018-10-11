from GenModels.GM.Distributions import Categorical, Dirichlet, TensorTransition, TensorTransitionDirichletPrior
import autograd.numpy as np
from GenModels.GM.Models.DiscreteGraphModels import *
import copy
import matplotlib
import matplotlib.pyplot as plt
from .PedigreeWrappers import createDataset
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from GenModels.GM.Utility import logsumexp, monitored_adam
from autograd import grad, value_and_grad
from autograd.misc.optimizers import adam
import os
import pickle

__all__ = [
    'autosomalDominantPriors',
    'autosomalRecessivePriors',
    'xLinkedRecessivePriors',
    'AutosomalDominant',
    'AutosomalRecessive',
    'XLinkedRecessive',
    'InheritancePatternTrainer'
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

######################################################################

class AutosomalSVAE( _drawMixin, GSVAE ):
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

    def draw( self, graph_index=0, show_carrier_prob=True ):
        def probCarrierFunc( sex, prob ):
            return prob[ 0 ] + prob[ 0 ]
        return self._draw( graph_index=graph_index, show_carrier_prob=show_carrier_prob, probCarrierFunc=probCarrierFunc )

class XLinkedSVAE( _drawMixin, GroupGSVAE ):
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

    def draw( self, graph_index=0, show_carrier_prob=True ):
        def probCarrierFunc( sex, prob ):
            if( sex == 'female' ):
                return prob[ 0 ] + prob[ 1 ]
            elif( sex == 'male' ):
                return prob[ 0 ]
            else:
                return prob[ 0 ] + prob[ 1 ] + prob[ 3 ]
        return self._draw( graph_index=graph_index, show_carrier_prob=show_carrier_prob, probCarrierFunc=probCarrierFunc )

######################################################################

class InheritancePatternTrainer():

    def __init__( self, training_graphs, test_graphs, root_strength=100000000000, prior_strength=100000000000, **kwargs ):

        self.ad_model = AutosomalSVAE( graphs=None, root_strength=root_strength, prior_strength=prior_strength, **kwargs )
        self.ar_model = AutosomalSVAE( graphs=None, root_strength=root_strength, prior_strength=prior_strength, **kwargs )
        self.xl_model = XLinkedSVAE( graphs=None, root_strength=root_strength, prior_strength=prior_strength, **kwargs )

        self.training_set = training_graphs
        self.test_set = test_graphs

        self.k = 1.0

        # Set the first graph
        self.updateCurrentGraphAndLabel( training=True )

    def updateCurrentGraphAndLabel( self, training=True, index=None, graph=None ):

        if( graph is None ):
            # Choose the next graph
            if( training == True ):
                random_index = int( np.random.random() * len( self.training_set ) ) if index is None else index
                graph = self.training_set[ random_index ]
            else:
                random_index = int( np.random.random() * len( self.test_set ) ) if index is None else index
                graph = self.test_set[ random_index ]

        # Generate the label
        label = graph[ 0 ].inheritancePattern
        label_one_hot = np.zeros( 3 )
        label_one_hot[ [ 'AD', 'AR', 'XL' ].index( label ) ] = 1.0
        self.current_label_one_hot = label_one_hot

        # Update the models
        ad_graph, ar_graph, xl_graph = createDataset( [ graph ] )

        self.ad_model.msg.preprocessData( ad_graph )
        self.ar_model.msg.preprocessData( ar_graph )
        self.xl_model.msg.preprocessData( xl_graph )

    def inheritancePatternLogits( self, model_parameters, graph=None, mc_samples=10 ):

        if( graph is not None ):
            self.updateCurrentGraphAndLabel( graph=graph )

        ad_emission_params, ar_emission_params, xl_emission_params = model_parameters

        all_logits = []
        for i in range( mc_samples ):
            ad_loss = self.ad_model.opt.computeLoss( ad_emission_params, 0 )
            ar_loss = self.ar_model.opt.computeLoss( ar_emission_params, 0 )
            xl_loss = self.xl_model.opt.computeLoss( xl_emission_params, 0 )

            prediction_logits = np.array( [ -ad_loss, -ar_loss, -xl_loss ] )
            all_logits.append( prediction_logits )

        return logsumexp( np.array( all_logits ), axis=0 )

    def inheritancePatternPrediction( self, model_parameters, graph=None, mc_samples=10 ):
        prediction_logits = self.inheritancePatternLogits( model_parameters, graph=graph, mc_samples=mc_samples )
        prediction_probs = prediction_logits - logsumexp( prediction_logits )
        return prediction_probs

    def fullLoss( self, full_params, n_iter=0 ):

        ad_emission_params, ar_emission_params, xl_emission_params = full_params

        # Each element in the set of logits is both the SVAE lower bound on P( Y ) and what we are
        # going to use to predict the inheritance pattern
        prediction_logits = self.inheritancePatternLogits( full_params, graph=None, mc_samples=1 )
        ad_loss, ar_loss, xl_loss = -prediction_logits

        # Compute the prediction loss.  We really want P( Y ) for each model,
        # but will settle for the lower bound given by the SVAE loss
        prediction_loss = -np.sum( self.current_label_one_hot * prediction_logits )

        return ad_loss + ar_loss + xl_loss + self.k * prediction_loss

    def train( self, num_iters=100 ):

        ad_params = ( self.ad_model.params.emission_dist.recognizer_params, self.ad_model.params.emission_dist.generative_hyper_params )
        ar_params = ( self.ar_model.params.emission_dist.recognizer_params, self.ar_model.params.emission_dist.generative_hyper_params )

        xl_params = ( {}, {} )
        for group, dist in self.xl_model.params.emission_dists.items():
            xl_params[ 0 ][ group ] = dist.recognizer_params
            xl_params[ 1 ][ group ] = dist.generative_hyper_params

        svae_params = ( ad_params, ar_params, xl_params )

        if( os.path.isfile( 'saved_params.p' ) ):
            with open( 'saved_params.p', 'rb' ) as file:
                svae_params = pickle.load( file )
                print( 'Using last checkpoint' )

        # Callback to run the test set and update the next graph
        def callback( full_params, i, g ):

            if( i and i % 100 == 0 ):
                with open( 'saved_params.p', 'wb' ) as file:
                    pickle.dump( full_params, file )

            # Every 500 steps, run the algorithm on the test set
            if( i % 500 == 0 ):
                labels = [ 'AD', 'AR', 'XL' ]
                # true_labels = [ g.inheritancePattern for g, fbs in self.test_set ]
                predicted_labels = []
                true_labels = []
                for graph_and_fbs in self.test_set:
                    if( np.random.random() < 0.3 ):
                        continue
                    true_labels.append( graph_and_fbs[ 0 ].inheritancePattern )
                    probs = self.inheritancePatternPrediction( full_params, graph=graph_and_fbs, mc_samples=1 )
                    predicted_labels.append( labels[ np.argmax( probs ) ] )

                # Print the confusion matrix and kappa score on the test set
                cm = confusion_matrix( y_true=np.array( true_labels ), y_pred=np.array( predicted_labels ), labels=labels )
                ck = cohen_kappa_score( y1=np.array( true_labels ), y2=np.array( predicted_labels ), labels=labels )
                print( 'Confusion matrix:\n', cm )
                print( 'Cohen kappa:', ck )

            # Swap out the current graph and update the current label
            self.updateCurrentGraphAndLabel( training=True )

        # Optimize
        grads = grad( self.fullLoss )
        final_params = adam( grads, svae_params, num_iters=num_iters, callback=callback )
        return final_params
