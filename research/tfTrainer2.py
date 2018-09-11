import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

import os
import sys
import numpy as np
import pandas as pd
import pickle
from functools import partial
from sklearn.metrics import confusion_matrix

def loadData( test_train_split=0.8 ):

    df = pd.read_csv( 'GenModels/research/pedigreeAnswers.csv' )
    df_labels = pd.read_csv( 'GenModels/research/Complete_pedigree_data.csv' )

    # Set the labels for the regular df
    for studyID, IP in df_labels[ [ 'Patient ID', 'Inheritance Pattern' ] ].values:
        if( studyID in df[ 'Patient ID' ].values ):
            index = df.index[ df[ 'Patient ID' ] == studyID ][ 0 ]
            df.loc[ index, 'IP' ] = IP

    # Drop nan inheritance patterns
    bad_mask = df[ 'IP' ].isna()
    df = df[ ~bad_mask ]

    # Load the ghmm results
    results = pickle.load( open( 'GenModels/research/full_results.p', 'rb' ) )

    # Add the extra features from the ghmm
    for ( graph, _ ), _, _, p in sorted( results, key=lambda x: x[ 0 ][ 0 ].studyID ):
        if( graph.studyID in df[ 'Patient ID' ].values ):
            index = df.index[ df[ 'Patient ID' ] == graph.studyID ][ 0 ]
            df.loc[ index, 'AD_Prob' ] = p[ 0 ]
            df.loc[ index, 'AR_Prob' ] = p[ 1 ]
            df.loc[ index, 'XL_Prob' ] = p[ 2 ]
            df.loc[ index, 'nAffectedMales' ] = len( [ 1 for n in graph.nodes if graph.attrs[ n ][ 'sex' ] == 'male' and graph.attrs[ n ][ 'affected' ] == True ] )
            df.loc[ index, 'nAffectedFemales' ] = len( [ 1 for n in graph.nodes if graph.attrs[ n ][ 'sex' ] == 'female' and graph.attrs[ n ][ 'affected' ] == True ] )
            df.loc[ index, 'nMales' ] = len( [ 1 for n in graph.nodes if graph.attrs[ n ][ 'sex' ] == 'male' ] )
            df.loc[ index, 'nFemales' ] = len( [ 1 for n in graph.nodes if graph.attrs[ n ][ 'sex' ] == 'female' ] )
            df.loc[ index, 'nPeople' ] = len( graph.nodes )
            if( not graph.inheritancePattern == df.loc[ index, 'IP' ] ):
                print( 'graph.inheritancePattern', graph.inheritancePattern )
                print( 'df.loc[ index, \'IP\' ]', df.loc[ index, 'IP' ] )
                assert 0

    # Use these features
    keep_cols = [ 'Patient ID',
                  'numAff',
                  'multGenAff',
                  'MAffSon',
                  'MAffDau',
                  'multDx',
                  'consang',
                  'nAffectedMales',
                  'nAffectedFemales',
                  'nMales',
                  'nFemales',
                  'nPeople',
                  'AD_Prob',
                  'AR_Prob',
                  'XL_Prob',
                  'IP' ]

    # Drop the rows that we don't have ghmm results for
    df = df[ keep_cols ].dropna()
    df = df.set_index( 'Patient ID' )

    # Drop inheritance patterns that aren't AD, AR or XL
    mask = df[ 'IP' ] == 'AD'
    mask |= df[ 'IP' ] == 'AR'
    mask |= df[ 'IP' ] == 'XL'
    df = df[ mask ]

    # Turn True, False to 1, 0
    def func( col ):
        return np.array( col, dtype=int )
    df[ [ 'multGenAff', 'MAffSon', 'MAffDau', 'multDx', 'consang', 'nPeople', 'nAffectedFemales', 'nAffectedMales', 'nFemales', 'nMales' ] ] = df[ [ 'multGenAff', 'MAffSon', 'MAffDau', 'multDx', 'consang', 'nPeople', 'nAffectedFemales', 'nAffectedMales', 'nFemales', 'nMales' ] ].apply( func )

    # Turn AD, AR, XL into 0, 1, 2
    def func( col ):
        col = np.array( col )
        col[ col == 'AD' ] = 0
        col[ col == 'AR' ] = 1
        col[ col == 'XL' ] = 2
        return col
    df[ 'IP' ] = df[ 'IP' ].apply( func )

    def split( df ):
        # Finally, split the dataframe into 3 different categories: Full data, partial data and labels
        full_data = df[ [ 'numAff',
                          'multGenAff',
                          'MAffSon',
                          'MAffDau',
                          'multDx',
                          'consang',
                          'nAffectedMales',
                          'nAffectedFemales',
                          'nMales',
                          'nFemales',
                          'nPeople',
                          'AD_Prob',
                          'AR_Prob',
                          'XL_Prob' ] ]

        labels = df[ [ 'IP' ] ]
        return full_data, labels

    # Shuffle the data
    df = df.sample( frac=1 )

    n_rows = int( test_train_split * df.shape[ 0 ] )

    train_data = split( df.iloc[ :n_rows, : ] )
    test_data = split( df.iloc[ n_rows:, : ] )

    return train_data, test_data

#####################################################################################

def oneHot( array, n_classes ):
    array_one_hot = np.zeros( ( array.shape[ 0 ], n_classes ) )
    array_one_hot[ np.arange( array.shape[ 0 ], dtype=int ), array ] = 1
    return array_one_hot

def toTFInputs( full_data, labels ):

    num_aff = full_data[ 'numAff' ].values.ravel()[ :, None ] #* 0
    mult_gen_aff = oneHot( full_data[ 'multGenAff' ].values.ravel(), 2 ) #* 0
    m_aff_son = oneHot( full_data[ 'MAffSon' ].values.ravel(), 2 ) #* 0
    m_aff_dau = oneHot( full_data[ 'MAffDau' ].values.ravel(), 2 ) #* 0
    mult_dx = oneHot( full_data[ 'multDx' ].values.ravel(), 2 ) * 0
    consag = oneHot( full_data[ 'consang' ].values.ravel(), 2 ) #* 0
    n_aff_males = full_data[ 'nAffectedMales' ].values.ravel()[ :, None ] #* 0 + 1
    n_aff_females = full_data[ 'nAffectedFemales' ].values.ravel()[ :, None ] #* 0 + 1
    n_males = full_data[ 'nMales' ].values.ravel()[ :, None ] #* 0 + 1
    n_females = full_data[ 'nFemales' ].values.ravel()[ :, None ] #* 0 + 1
    n_people = full_data[ 'nPeople' ].values.ravel()[ :, None ] #* 0 + 1
    ip_probs = full_data[ [ 'AD_Prob', 'AR_Prob', 'XL_Prob' ] ].values #* 0
    true_ip = oneHot( labels.values.ravel().astype( int ), 3 )

    return num_aff, mult_gen_aff, m_aff_son, m_aff_dau, mult_dx, consag, n_aff_males, n_aff_females, n_males, n_females, n_people, ip_probs, true_ip

def loadTFDataset( batch_size=1024 ):
    train_data, test_data = loadData()
    train_dataset = tf.data.Dataset.from_tensor_slices( toTFInputs( *train_data ) ).batch( batch_size )
    test_dataset = tf.data.Dataset.from_tensor_slices( toTFInputs( *test_data ) ).batch( batch_size )

    return train_dataset, test_dataset

#####################################################################################

class CategoricalModel( tf.keras.Model ):
    def __init__( self, n_labels, name ):
        super().__init__( name=name )
        self.dense1 = tf.keras.layers.Dense( 16, activation=tf.nn.leaky_relu, kernel_initializer=tf.keras.initializers.lecun_normal() )
        self.logits = tf.keras.layers.Dense( n_labels )

    def wrappedCall( self, *args, training=True ):
        args = [ tf.cast( a, tf.float64 ) for a in args ]
        p = tf.concat( args, axis=-1 )
        p = self.dense1( p )
        p = self.logits( p )
        return p

    @staticmethod
    def logProb( x, logits ):
        ans = -tf.nn.softmax_cross_entropy_with_logits_v2( labels=x, logits=logits )
        return ans

    @staticmethod
    def reparamSample( gumbel, logits, tau=0.1 ):
        # Use the Gumbel-Softmax reparametrization trick
        one_hot = tf.exp( ( gumbel + logits ) / tau )
        return one_hot / tf.reduce_sum( one_hot, axis=-1 )

#########################

class NormalModel( tf.keras.Model ):
    def __init__( self, state_size, name ):
        super().__init__( name=name )
        self.dense1 = tf.keras.layers.Dense( 16, activation=tf.nn.leaky_relu, kernel_initializer=tf.keras.initializers.lecun_normal() )
        self.dense2 = tf.keras.layers.Dense( 16, activation=tf.nn.leaky_relu, kernel_initializer=tf.keras.initializers.lecun_normal() )
        self.mu = tf.keras.layers.Dense( state_size, activation=None, use_bias=True )
        self.log_sigma = tf.keras.layers.Dense( state_size, activation=tf.nn.sigmoid, use_bias=False )

    def wrappedCall( self, *args, training=True ):
        args = [ tf.cast( a, tf.float64 ) for a in args ]
        y = tf.concat( args, axis=-1 )
        y = self.dense1( y )
        y = self.dense2( y )
        mu, log_sigma = self.mu( y ), self.log_sigma( y )
        return mu, log_sigma

    @staticmethod
    def logProb( x, mu, log_sigma, dim=2 ):
        if( dim == 2 ):
            contract = 'bx,bx->b'
        elif( dim == 3 ):
            contract = 'ibx,ibx->ib'
        log_prob = mu - x
        log_prob = -0.5 * tf.einsum( contract, log_prob, log_prob / np.exp( log_sigma ) ) # -0.5*( x - mu ).T * sigma^-1 * ( x - mu )
        log_prob -= 0.5 * tf.reduce_sum( log_sigma, axis=-1 )                               # -0.5 * log( det( sigma ) )
        log_prob -= x.shape[ -1 ].value / 2 * np.log( 2 * np.pi )                           # -k / 2 * log( 2*pi )
        return log_prob

    @staticmethod
    def reparamSample( noise, mu, log_sigma, dim=2 ):
        if( dim == 2 ):
            contract = 'bi,bi->bi'
        elif( dim == 3 ):
            contract = 'bij,bij->bij'
        sigma = tf.exp( log_sigma )
        return tf.einsum( contract, noise, tf.sqrt( sigma ) ) + mu

#########################

class DirichletModel( tf.keras.Model ):
    def __init__( self, state_size, name ):
        super().__init__( name=name )
        self.dense1 = tf.keras.layers.Dense( 16, activation=tf.nn.leaky_relu, kernel_initializer=tf.keras.initializers.lecun_normal() )
        self.alphas = tf.keras.layers.Dense( state_size, activation=tf.nn.relu, use_bias=False, kernel_initializer=tf.keras.initializers.lecun_normal() )

    def wrappedCall( self, *args, training=True ):
        args = [ tf.cast( a, tf.float64 ) for a in args ]
        alpha = tf.concat( args, axis=-1 )
        alpha = self.dense1( alpha )
        alpha = self.alphas( alpha ) + 1.0
        return alpha

    @staticmethod
    def logProb( x, alpha ):
        ans = tf.reduce_sum( x * ( alpha - 1 ), axis=-1 ) \
             - tf.reduce_sum( tf.lgamma( alpha ), axis=-1 ) \
             + tf.lgamma( tf.reduce_sum( alpha, axis=-1 ) )
        return ans

#########################

class BinomialModel( tf.keras.Model ):
    def __init__( self, name='Binom' ):
        super().__init__( name=name )
        self.dense1 = tf.keras.layers.Dense( 16, activation=tf.nn.leaky_relu, kernel_initializer=tf.keras.initializers.lecun_normal() )
        self.p = tf.keras.layers.Dense( 1, activation=tf.nn.sigmoid, use_bias=False )

    def wrappedCall( self, *args, training=True ):
        args = [ tf.cast( a, tf.float64 ) for a in args ]
        p = tf.concat( args, axis=-1 )
        p = self.dense1( p )
        p = self.p( p )
        return p

    @staticmethod
    def logProb( k, p, N ):
        binomial_coefficient = tf.lgamma( 1 + tf.cast( N, tf.float64 ) )
        binomial_coefficient -= tf.lgamma( 1 + tf.cast( k, tf.float64 ) )
        binomial_coefficient -= tf.lgamma( 1 + tf.cast( N, tf.float64 ) - tf.cast( k, tf.float64 ) )
        ans = tf.log( p ) * tf.cast( k, tf.float64 ) + tf.cast( N - k, tf.float64 ) * tf.log( 1 - p ) + binomial_coefficient
        ans = tf.reduce_sum( ans, axis=-1 )
        return ans

#####################################################################################

class NumAff( BinomialModel ):
    def __init__( self ):
        super().__init__( name='NumAff' )

    def call( self, z, true_ip, training=True ):
        return self.wrappedCall( z, true_ip, training=training )

class NumAffMales( BinomialModel ):
    def __init__( self ):
        super().__init__( name='NumAffMales' )

    def call( self, z, true_ip, training=True ):
        return self.wrappedCall( z, true_ip, training=training )

class NumAffFemales( BinomialModel ):
    def __init__( self ):
        super().__init__( name='NumAffFemales' )

    def call( self, z, true_ip, training=True ):
        return self.wrappedCall( z, true_ip, training=training )

class NAffMalesTotal( BinomialModel ):
    def __init__( self ):
        super().__init__( name='NAffMalesTotal' )

    def call( self, z, true_ip, training=True ):
        return self.wrappedCall( z, true_ip, training=training )

###########

class MultGenAff( CategoricalModel ):
    def __init__( self ):
        super().__init__( n_labels=2, name='MultGenAff' )

    def call( self, z, true_ip, training=True ):
        return self.wrappedCall( z, true_ip, training=training )

###########

class MAffSon( CategoricalModel ):
    def __init__( self ):
        super().__init__( n_labels=2, name='MAffSon' )

    def call( self, z, true_ip, training=True ):
        return self.wrappedCall( z, true_ip, training=training )

###########

class MAffDau( CategoricalModel ):
    def __init__( self ):
        super().__init__( n_labels=2, name='MAffDau' )

    def call( self, z, true_ip, training=True ):
        return self.wrappedCall( z, true_ip, training=training )

###########

class MultDx( CategoricalModel ):
    def __init__( self ):
        super().__init__( n_labels=2, name='MultDx' )

    def call( self, z, true_ip, training=True ):
        return self.wrappedCall( z, true_ip, training=training )

###########

class Consang( CategoricalModel ):
    def __init__( self ):
        super().__init__( n_labels=2, name='Consang' )

    def call( self, z, true_ip, training=True ):
        return self.wrappedCall( z, true_ip, training=training )

###########

class GHMMProbs( DirichletModel ):
    def __init__( self ):
        super().__init__( state_size=3, name='GHMMProbs' )

    def call( self, z, true_ip, training=True ):
        return self.wrappedCall( z, true_ip, training=training )

###########

class AuxulliaryGenerative( NormalModel ):
    def __init__( self, auxiliary_state_size ):
        super().__init__( auxiliary_state_size, name='AuxulliaryGenerative' )

    def call( self, z, true_ip, num_aff, n_aff_males, n_aff_females, mult_gen_aff, m_aff_son, m_aff_dau, mult_dx, consag, ip_probs, training=True ):
        return self.wrappedCall( z, true_ip, num_aff, n_aff_males, n_aff_females, mult_gen_aff, m_aff_son, m_aff_dau, mult_dx, consag, ip_probs, training=training )

###########

class Labeler( CategoricalModel ):
    def __init__( self ):
        super().__init__( n_labels=3, name='Labeler' )

    def call( self, a, num_aff, n_aff_males, n_aff_females, mult_gen_aff, m_aff_son, m_aff_dau, mult_dx, consag, ip_probs, training=True ):
        return self.wrappedCall( a, num_aff, n_aff_males, n_aff_females, mult_gen_aff, m_aff_son, m_aff_dau, mult_dx, consag, ip_probs, training=training )

###########

class LatentInference( NormalModel ):
    def __init__( self, latent_state_size ):
        super().__init__( latent_state_size, name='LatentInference' )

    def call( self, a, true_ip, num_aff, n_aff_males, n_aff_females, mult_gen_aff, m_aff_son, m_aff_dau, mult_dx, consag, ip_probs, training=True ):
        return self.wrappedCall( a, true_ip, num_aff, n_aff_males, n_aff_females, mult_gen_aff, m_aff_son, m_aff_dau, mult_dx, consag, ip_probs, training=training )

###########

class AuxiliaryInference( NormalModel ):
    def __init__( self, auxiliary_state_size ):
        super().__init__( auxiliary_state_size, name='AuxiliaryInference' )

    def call( self, num_aff, n_aff_males, n_aff_females, mult_gen_aff, m_aff_son, m_aff_dau, mult_dx, consag, ip_probs, training=True ):
        return self.wrappedCall( num_aff, n_aff_males, n_aff_females, mult_gen_aff, m_aff_son, m_aff_dau, mult_dx, consag, ip_probs, training=training )

#####################################################################################

def train_epoch( data, eval_data, models, optimizers, constants ):

    n_importance_samples = constants[ 'n_importance_samples' ]
    n_a = constants[ 'auxilliary_state_size' ]
    n_z = constants[ 'latent_state_size' ]

    p_num_aff = models[ 'p_num_aff' ]
    # p_num_aff_males = models[ 'p_num_aff_males' ]
    # p_num_aff_females = models[ 'p_num_aff_females' ]
    # p_num_aff_males_total = models[ 'p_num_aff_males_total' ]
    p_mult_gen_aff = models[ 'p_mult_gen_aff' ]
    p_m_aff_son = models[ 'p_m_aff_son' ]
    p_m_aff_dau = models[ 'p_m_aff_dau' ]
    p_mult_dx = models[ 'p_mult_dx' ]
    p_consang = models[ 'p_consang' ]
    p_ghmm_prob = models[ 'p_ghmm_prob' ]
    p_a = models[ 'p_a' ]
    labeler = models[ 'labeler' ]
    q_z = models[ 'q_z' ]
    q_a = models[ 'q_a' ]

    adam = optimizers[ 'adam' ]

    for i, ( num_aff, mult_gen_aff, m_aff_son, m_aff_dau, mult_dx, consag, n_aff_males, n_aff_females, n_males, n_females, n_people, ip_probs, true_ip ) in enumerate( tfe.Iterator( data ) ):

        batch_size = num_aff.shape[ 0 ]

        with tf.device( 'CPU' ):
            tf.assign_add( constants[ 'step_counter' ], 1 )

        with tf.contrib.summary.record_summaries_every_n_global_steps( constants[ 'log_interval' ], global_step=constants[ 'step_counter' ] ):

            importance_weights = []

            p_num_aff_grads = []
            # p_num_aff_males_grads = []
            # p_num_aff_females_grads = []
            # p_num_aff_males_total_grads = []
            p_mult_gen_aff_grads = []
            p_m_aff_son_grads = []
            p_m_aff_dau_grads = []
            p_mult_dx_grads = []
            p_consang_grads = []
            p_ghmm_prob_grads = []
            labeler_grads = []
            q_z_grads = []
            q_a_grads = []

            for i in range( n_importance_samples ):

                epsilon_a = tf.random_normal( shape=( batch_size, n_a ), dtype=tf.float64 )
                epsilon_z = tf.random_normal( shape=( batch_size, n_z ), dtype=tf.float64 )

                with tfe.GradientTape( persistent=True ) as g:

                    # Sample a ~ q( a | ... )
                    q_a_mu, q_a_log_sigma = q_a( num_aff, n_aff_males, n_aff_females, mult_gen_aff, m_aff_son, m_aff_dau, mult_dx, consag, ip_probs )
                    a = q_a.reparamSample( epsilon_a, q_a_mu, q_a_log_sigma )

                    # Sample z ~ q( z | ... )
                    q_z_mu, q_z_log_sigma = q_z( a, true_ip, num_aff, n_aff_males, n_aff_females, mult_gen_aff, m_aff_son, m_aff_dau, mult_dx, consag, ip_probs )
                    z = q_z.reparamSample( epsilon_z, q_z_mu, q_z_log_sigma )

                    # Generate p( z )
                    p_z_mu, p_z_log_sigma = tf.zeros_like( q_a_mu ), tf.zeros_like( q_a_mu )

                    # Generate p( x_i | z, y )
                    num_aff_p = p_num_aff( z, true_ip )
                    # num_aff_males_p = p_num_aff_males( z, true_ip )
                    # num_aff_females_p = p_num_aff_females( z, true_ip )
                    # p_num_aff_males_total_p = p_num_aff_males_total( z, true_ip )
                    mult_gen_aff_logit = p_mult_gen_aff( z, true_ip )
                    m_aff_son_logit = p_m_aff_son( z, true_ip )
                    m_aff_dau_logit = p_m_aff_dau( z, true_ip )
                    mult_dx_logit = p_mult_dx( z, true_ip )
                    consang_logit = p_consang( z, true_ip )
                    ghmm_prob_alpha = p_ghmm_prob( z, true_ip )

                    # Generate p( a | ... )
                    p_a_mu, p_a_log_sigma = p_a( z, true_ip, num_aff, n_aff_males, n_aff_females, mult_gen_aff, m_aff_son, m_aff_dau, mult_dx, consag, ip_probs )

                    # Compute p( x_i | z, y )
                    imporatance_elbo = BinomialModel.logProb( num_aff, num_aff_p, n_people )
                    # imporatance_elbo = BinomialModel.logProb( n_aff_males, num_aff_males_p, n_males )
                    # imporatance_elbo += BinomialModel.logProb( n_aff_females, num_aff_females_p, n_females )
                    # imporatance_elbo = BinomialModel.logProb( n_aff_males, p_num_aff_males_total_p, num_aff )
                    imporatance_elbo1 = CategoricalModel.logProb( mult_gen_aff, mult_gen_aff_logit )
                    imporatance_elbo2 = CategoricalModel.logProb( m_aff_son, m_aff_son_logit )
                    imporatance_elbo3 = CategoricalModel.logProb( m_aff_dau, m_aff_dau_logit )
                    imporatance_elbo4 = CategoricalModel.logProb( mult_dx, mult_dx_logit )
                    imporatance_elbo5 = CategoricalModel.logProb( consag, consang_logit )
                    imporatance_elbo6 = DirichletModel.logProb( ip_probs, ghmm_prob_alpha )

                    # if( i == 0 ):
                    #     print( '\nimporatance_elbo', imporatance_elbo )
                    #     print( 'imporatance_elbo1', imporatance_elbo1 )
                    #     print( 'imporatance_elbo2', imporatance_elbo2 )
                    #     print( 'imporatance_elbo3', imporatance_elbo3 )
                    #     print( 'imporatance_elbo4', imporatance_elbo4 )
                    #     print( 'imporatance_elbo5', imporatance_elbo5 )
                    #     print( 'imporatance_elbo6', imporatance_elbo6 )

                    imporatance_elbo += imporatance_elbo1 + imporatance_elbo2 + imporatance_elbo3 + imporatance_elbo4 + imporatance_elbo5 + imporatance_elbo6
                    # Compute p( z ) and p( a | ... )
                    imporatance_elbo += NormalModel.logProb( a, p_a_mu, p_a_log_sigma )
                    imporatance_elbo += NormalModel.logProb( z, p_z_mu, p_z_log_sigma )

                    # Compute q( a | ... ) and q( z | ... )
                    imporatance_elbo -= NormalModel.logProb( a, q_a_mu, q_a_log_sigma )
                    imporatance_elbo -= NormalModel.logProb( z, q_z_mu, q_z_log_sigma )

                    # Generate the labeler loss
                    ip_prediction_logits = labeler( a, num_aff, n_aff_males, n_aff_females, mult_gen_aff, m_aff_son, m_aff_dau, mult_dx, consag, ip_probs )
                    labeler_loss = CategoricalModel.logProb( true_ip, ip_prediction_logits )

                    imporatance_elbo = -imporatance_elbo - labeler_loss

                # Update the importance weights and gradients
                importance_weights.append( tf.reduce_sum( imporatance_elbo, axis=0 ) )

                p_num_aff_grads.append( g.gradient( imporatance_elbo, p_num_aff.variables ) )
                # p_num_aff_males_grads.append( g.gradient( imporatance_elbo, p_num_aff_males.variables ) )
                # p_num_aff_females_grads.append( g.gradient( imporatance_elbo, p_num_aff_females.variables ) )
                # p_num_aff_males_total_grads.append( g.gradient( imporatance_elbo, p_num_aff_males_total.variables ) )
                p_mult_gen_aff_grads.append( g.gradient( imporatance_elbo, p_mult_gen_aff.variables ) )
                p_m_aff_son_grads.append( g.gradient( imporatance_elbo, p_m_aff_son.variables ) )
                p_m_aff_dau_grads.append( g.gradient( imporatance_elbo, p_m_aff_dau.variables ) )
                p_mult_dx_grads.append( g.gradient( imporatance_elbo, p_mult_dx.variables ) )
                p_consang_grads.append( g.gradient( imporatance_elbo, p_consang.variables ) )
                p_ghmm_prob_grads.append( g.gradient( imporatance_elbo, p_ghmm_prob.variables ) )
                labeler_grads.append( g.gradient( imporatance_elbo, labeler.variables ) )
                q_z_grads.append( g.gradient( imporatance_elbo, q_z.variables ) )
                q_a_grads.append( g.gradient( imporatance_elbo, q_a.variables ) )

            # Compute the gradients
            total_weight = tf.reduce_logsumexp( importance_weights, axis=-1 )

            print( '\ntotal_weight', total_weight )

            p_num_aff_grad = [ tf.zeros_like( v ) for v in p_num_aff.variables ]
            # p_num_aff_males_grad = [ tf.zeros_like( v ) for v in p_num_aff_males.variables ]
            # p_num_aff_females_grad = [ tf.zeros_like( v ) for v in p_num_aff_females.variables ]
            # p_num_aff_males_total_grad = [ tf.zeros_like( v ) for v in p_num_aff_males_total.variables ]
            p_mult_gen_aff_grad = [ tf.zeros_like( v ) for v in p_mult_gen_aff.variables ]
            p_m_aff_son_grad = [ tf.zeros_like( v ) for v in p_m_aff_son.variables ]
            p_m_aff_dau_grad = [ tf.zeros_like( v ) for v in p_m_aff_dau.variables ]
            p_mult_dx_grad = [ tf.zeros_like( v ) for v in p_mult_dx.variables ]
            p_consang_grad = [ tf.zeros_like( v ) for v in p_consang.variables ]
            p_ghmm_prob_grad = [ tf.zeros_like( v ) for v in p_ghmm_prob.variables ]
            labeler_grad = [ tf.zeros_like( v ) for v in labeler.variables ]
            q_z_grad = [ tf.zeros_like( v ) for v in q_z.variables ]
            q_a_grad = [ tf.zeros_like( v ) for v in q_a.variables ]

            for i in range( n_importance_samples ):
                w = tf.exp( importance_weights[ i ] - total_weight )

                for j, g in enumerate( p_num_aff_grads[ i ] ):
                    p_num_aff_grad[ j ] += w * g

                # for j, g in enumerate( p_num_aff_males_grads[ i ] ):
                #     p_num_aff_males_grad[ j ] += w * g

                # for j, g in enumerate( p_num_aff_females_grads[ i ] ):
                #     p_num_aff_females_grad[ j ] += w * g

                # for j, g in enumerate( p_num_aff_males_total_grads[ i ] ):
                #     p_num_aff_males_total_grad[ j ] += w * g


                for j, g in enumerate( p_mult_gen_aff_grads[ i ] ):
                    p_mult_gen_aff_grad[ j ] += w * g

                for j, g in enumerate( p_m_aff_son_grads[ i ] ):
                    p_m_aff_son_grad[ j ] += w * g

                for j, g in enumerate( p_m_aff_dau_grads[ i ] ):
                    p_m_aff_dau_grad[ j ] += w * g

                for j, g in enumerate( p_mult_dx_grads[ i ] ):
                    p_mult_dx_grad[ j ] += w * g

                for j, g in enumerate( p_consang_grads[ i ] ):
                    p_consang_grad[ j ] += w * g

                for j, g in enumerate( p_ghmm_prob_grads[ i ] ):
                    p_ghmm_prob_grad[ j ] += w * g

                for j, g in enumerate( labeler_grads[ i ] ):
                    # labeler_grad[ j ] += w * g
                    labeler_grad[ j ] += g / n_importance_samples

                for j, g in enumerate( q_z_grads[ i ] ):
                    q_z_grad[ j ] += w * g

                for j, g in enumerate( q_a_grads[ i ] ):
                    q_a_grad[ j ] += w * g

            # Take an optimization step
            adam.apply_gradients( zip( p_num_aff_grad, p_num_aff.variables ) )
            # adam.apply_gradients( zip( p_num_aff_males_grad, p_num_aff_males.variables ) )
            # adam.apply_gradients( zip( p_num_aff_females_grad, p_num_aff_females.variables ) )
            # adam.apply_gradients( zip( p_num_aff_males_total_grad, p_num_aff_males_total.variables ) )
            adam.apply_gradients( zip( p_mult_gen_aff_grad, p_mult_gen_aff.variables ) )
            adam.apply_gradients( zip( p_m_aff_son_grad, p_m_aff_son.variables ) )
            adam.apply_gradients( zip( p_m_aff_dau_grad, p_m_aff_dau.variables ) )
            adam.apply_gradients( zip( p_mult_dx_grad, p_mult_dx.variables ) )
            adam.apply_gradients( zip( p_consang_grad, p_consang.variables ) )
            adam.apply_gradients( zip( p_ghmm_prob_grad, p_ghmm_prob.variables ) )
            adam.apply_gradients( zip( labeler_grad, labeler.variables ) )
            adam.apply_gradients( zip( q_z_grad, q_z.variables ) )
            adam.apply_gradients( zip( q_a_grad, q_a.variables ) )

    K = 5000

    for i, ( num_aff, mult_gen_aff, m_aff_son, m_aff_dau, mult_dx, consag, n_aff_males, n_aff_females, n_males, n_females, n_people, ip_probs, true_ip ) in enumerate( tfe.Iterator( data ) ):

        batch_size = num_aff.shape[ 0 ]

        epsilon_a = tf.random_normal( shape=( K, batch_size, n_a ), dtype=tf.float64 )

        # Sample a ~ q( a | ... )
        q_a_mu, q_a_log_sigma = q_a( num_aff, n_aff_males, n_aff_females, mult_gen_aff, m_aff_son, m_aff_dau, mult_dx, consag, ip_probs )
        q_a_mu = tf.reshape( q_a_mu, shape=( 1, batch_size, q_a_mu.shape[ -1 ] ) )
        q_a_log_sigma = tf.reshape( q_a_log_sigma, shape=( 1, batch_size, q_a_log_sigma.shape[ -1 ] ) )
        a = q_a.reparamSample( epsilon_a, q_a_mu, q_a_log_sigma, dim=3 )

        reshapeHelp = lambda x: tf.tile( tf.reshape( x, shape=( 1, batch_size, x.shape[ -1 ] ) ), multiples=( K, 1, 1 ) )

        # Generate the labeler loss
        ip_prediction_logits = labeler( a, reshapeHelp( num_aff ),
                                            reshapeHelp( n_aff_males ),
                                            reshapeHelp( n_aff_females ),
                                              reshapeHelp( mult_gen_aff ),
                                              reshapeHelp( m_aff_son ),
                                              reshapeHelp( m_aff_dau ),
                                              reshapeHelp( mult_dx ),
                                              reshapeHelp( consag ),
                                              reshapeHelp( ip_probs ) )
        ip_prediction_logits = tf.reduce_sum( ip_prediction_logits, axis=0 )

        true = np.argmax( true_ip.numpy(), axis=1 ).ravel()
        predicted = np.argmax( ip_prediction_logits.numpy(), axis=1 ).ravel()
        accuracy = ( true == predicted ).sum() / true.shape[ 0 ]

        print( 'true', true )
        print( 'predicted', predicted )
        print( 'train accuracy', accuracy )

    for i, ( num_aff, mult_gen_aff, m_aff_son, m_aff_dau, mult_dx, consag, n_aff_males, n_aff_females, n_males, n_females, n_people, ip_probs, true_ip ) in enumerate( tfe.Iterator( eval_data ) ):

        batch_size = num_aff.shape[ 0 ]

        epsilon_a = tf.random_normal( shape=( K, batch_size, n_a ), dtype=tf.float64 )

        # Sample a ~ q( a | ... )
        q_a_mu, q_a_log_sigma = q_a( num_aff, n_aff_males, n_aff_females, mult_gen_aff, m_aff_son, m_aff_dau, mult_dx, consag, ip_probs )
        q_a_mu = tf.reshape( q_a_mu, shape=( 1, batch_size, q_a_mu.shape[ -1 ] ) )
        q_a_log_sigma = tf.reshape( q_a_log_sigma, shape=( 1, batch_size, q_a_log_sigma.shape[ -1 ] ) )
        a = q_a.reparamSample( epsilon_a, q_a_mu, q_a_log_sigma, dim=3 )

        reshapeHelp = lambda x: tf.tile( tf.reshape( x, shape=( 1, batch_size, x.shape[ -1 ] ) ), multiples=( K, 1, 1 ) )

        # Generate the labeler loss
        ip_prediction_logits = labeler( a, reshapeHelp( num_aff ),
                                           reshapeHelp( n_aff_males ),
                                            reshapeHelp( n_aff_females ),
                                              reshapeHelp( mult_gen_aff ),
                                              reshapeHelp( m_aff_son ),
                                              reshapeHelp( m_aff_dau ),
                                              reshapeHelp( mult_dx ),
                                              reshapeHelp( consag ),
                                              reshapeHelp( ip_probs ) )
        ip_prediction_logits = tf.reduce_sum( ip_prediction_logits, axis=0 )

        true = np.argmax( true_ip.numpy(), axis=1 ).ravel()
        predicted = np.argmax( ip_prediction_logits.numpy(), axis=1 ).ravel()
        accuracy = ( true == predicted ).sum() / true.shape[ 0 ]

        mat = confusion_matrix( true, predicted, labels=[0,1,2] )
        mat = mat / mat.sum( axis=-1 )[ :,  None ]

        print( 'true', true )
        print( 'predicted', predicted )
        print( 'test accuracy', accuracy )
        print( 'mat', mat )

#####################################################################################

def trainProbModel():

    # Hyper parameters
    latent_state_size = 12
    auxilliary_state_size = 12

    learning_rate = 0.001

    # Save Directories
    output_dir = '/model_outputs'
    checkpoint_dir = '/model_checkpoints'

    # Models and optimizers
    models = {
        'p_num_aff': NumAff(),
        'p_num_aff_males': NumAffMales(),
        'p_num_aff_females': NumAffFemales(),
        'p_num_aff_males_total': NAffMalesTotal(),
        'p_mult_gen_aff': MultGenAff(),
        'p_m_aff_son': MAffSon(),
        'p_m_aff_dau': MAffDau(),
        'p_mult_dx': MultDx(),
        'p_consang': Consang(),
        'p_ghmm_prob': GHMMProbs(),
        'p_a': AuxulliaryGenerative( auxilliary_state_size ),
        'labeler': Labeler(),
        'q_z': LatentInference( latent_state_size ),
        'q_a': AuxiliaryInference( auxilliary_state_size )
    }
    optimizers = {
        'adam': tf.train.AdamOptimizer( learning_rate )
    }
    constants = {
        'step_counter': tf.train.get_or_create_global_step(),
        'log_interval': 10,
        'n_importance_samples': 1,
        'latent_state_size': latent_state_size,
        'auxilliary_state_size': auxilliary_state_size,
        'y_prior': np.ones( 3 )
    }

    # Summary and checkpoint handlers
    summary_writer = tf.contrib.summary.create_file_writer( output_dir, flush_millis=1000 )
    checkpoint_prefix = os.path.join( checkpoint_dir, 'ckpt' )
    latest_cpkt = tf.train.latest_checkpoint( checkpoint_dir )

    if( latest_cpkt ):
        print( 'Using latest checkpoint at', latest_cpkt )

    checkpoint = tfe.Checkpoint( **models )
    checkpoint.restore( latest_cpkt )

    # Data
    train_dataset, test_dataset = loadTFDataset()

    # Training loop
    for epoch in range( 1000 ):

        with summary_writer.as_default():
            train_epoch( data=train_dataset,
                         eval_data=test_dataset,
                         models=models,
                         optimizers=optimizers,
                         constants=constants )

        if( epoch % 5 == 0 ):
            checkpoint.save( checkpoint_prefix )


trainProbModel()
