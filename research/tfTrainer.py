import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

import os
import sys
import autograd.numpy as np
import pandas as pd
import pickle
from functools import partial

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
            df.loc[ index, 'nMales' ] = len( [ 1 for n in graph.nodes if graph.attrs[ n ][ 'sex' ] == 'male' ] )
            df.loc[ index, 'nFemales' ] = len( [ 1 for n in graph.nodes if graph.attrs[ n ][ 'sex' ] == 'female' ] )
            df.loc[ index, 'nUnknowns' ] = len( [ 1 for n in graph.nodes if graph.attrs[ n ][ 'sex' ] == 'unknown' ] )
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
                  'skipping',
                  'sibAff',
                  'halfSibAff',
                  'cousAff',
                  'MFAff',
                  'ons20M',
                  'multDx',
                  'consang',
                  'nMales',
                  'nFemales',
                  'nUnknowns',
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
    df[ [ 'multGenAff', 'MAffSon', 'MAffDau', 'multDx', 'consang', 'nMales', 'nFemales', 'nUnknowns' ] ] = df[ [ 'multGenAff', 'MAffSon', 'MAffDau', 'multDx', 'consang', 'nMales', 'nFemales', 'nUnknowns' ] ].apply( func )

    # Turn True, False, Unknown to 1, 0, -1
    def func( col ):
        col = np.array( col )
        col[ col == 'True' ] = 1
        col[ col == 'False' ] = 0
        col[ col == 'Unknown' ] = -1
        return col
    df[ 'ons20M' ] = df[ 'ons20M' ].apply( func )

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
        # full_data = df[ [ 'numAff', 'multGenAff', 'MAffSon', 'MAffDau', 'skipping', 'sibAff', 'halfSibAff', 'cousAff', 'MFAff', 'multDx', 'consang' ] ]
        full_data = df[ [ 'numAff',
                          'multGenAff',
                          'MAffSon',
                          'MAffDau',
                          'skipping',
                          'sibAff',
                          'halfSibAff',
                          'cousAff',
                          'MFAff',
                          'multDx',
                          'consang',
                          'nMales',
                          'nFemales',
                          'nUnknowns',
                          'AD_Prob',
                          'AR_Prob',
                          'XL_Prob' ] ]

        partial_data = df[ [ 'ons20M' ] ]
        labels = df[ [ 'IP' ] ]
        return full_data, partial_data, labels

    ##############################################################################
    # I HAVE NO IDEA IF THIS IS CORRECT.  VERIFY LATER
    ##############################################################################


    # Shuffle the data
    df = df.sample( frac=1 )

    n_rows = int( test_train_split * df.shape[ 0 ] )

    train_data = split( df.iloc[ :n_rows, : ] )
    test_data = split( df.iloc[ n_rows:, : ] )

    return train_data, test_data

#####################################################################################

def sklearnFit():
    from sklearn.ensemble import GradientBoostingClassifier

    ( train_full_data, train_partial_data, train_labels ), ( test_full_data, test_partial_data, test_labels ) = loadData()

    clf = GradientBoostingClassifier( n_estimators=1000 )
    clf.fit( train_full_data.values, train_labels.values.ravel() )

    print( clf.score( test_full_data, test_labels ) )

#####################################################################################

def oneHot( array, n_classes ):
    array_one_hot = np.zeros( ( array.shape[ 0 ], n_classes ) )
    array_one_hot[ np.arange( array.shape[ 0 ], dtype=int ), array ] = 1
    return array_one_hot

def toTFInputs( full_data, partial_data, labels ):
    full_data_int = full_data.values[ :, :-3 ].astype( int )
    full_data_prob = full_data.values[ :, -3: ].astype( float )
    partial_data = partial_data.values.astype( int ).ravel()[ :, None ]
    labels = labels.values.ravel().astype( int )
    labels_one_hot = oneHot( labels, 3 )

    return full_data_int, full_data_prob, partial_data, labels_one_hot

def loadTFDataset():
    train_data, test_data = loadData()
    print( train_data )
    assert 0
    train_dataset = tf.data.Dataset.from_tensor_slices( toTFInputs( *train_data ) ).batch( 1024 )
    test_dataset = tf.data.Dataset.from_tensor_slices( toTFInputs( *test_data ) ).batch( 1024 )

    return train_dataset, test_dataset

#####################################################################################

class Q_z( tf.keras.Model ):
    # Inference model over latent state (pedigree)
    # Normal distribution

    def __init__( self, latent_state_size ):
        super().__init__( name='Q_z' )
        self.dense1 = tf.keras.layers.Dense( 10 )
        self.mu = tf.keras.layers.Dense( latent_state_size, activation=None, use_bias=True )
        self.log_sigma = tf.keras.layers.Dense( latent_state_size, activation=tf.nn.sigmoid, use_bias=False )

    def call( self, a, y, m, d1, d2, training=True ):
        z = tf.concat( ( a, tf.cast( y, d2.dtype ), tf.cast( m, d2.dtype ), tf.cast( d1, d2.dtype ), d2 ), axis=-1 )
        z = self.dense1( z )
        mu, log_sigma = self.mu( z ), self.log_sigma( z )
        return mu, log_sigma

class Q_y( tf.keras.Model ):
    # Inference model over labels (inheritance pattern).
    # This is the prediction network
    # Categorical distribution

    def __init__( self, n_labels ):
        super().__init__( name='Q_y' )
        self.dense1 = tf.keras.layers.Dense( 10 )
        self.logits = tf.keras.layers.Dense( n_labels )

    def call( self, a, m, d1, d2, training=True ):
        y = tf.concat( ( a, m, tf.cast( d1, d2.dtype ), d2 ), axis=-1 )
        y = self.dense1( y )
        y = self.logits( y )
        return y

class Q_m( tf.keras.Model ):
    # Inference model over missing data (ons20M)
    # Categorical distribution

    def __init__( self, missing_data_classes ):
        super().__init__( name='Q_m' )
        self.dense1 = tf.keras.layers.Dense( 10 )
        self.logits = tf.keras.layers.Dense( missing_data_classes )

    def call( self, a, d1, d2, training=True ):
        m = tf.concat( ( a, tf.cast( d1, d2.dtype ), d2 ), axis=-1 )
        m = self.dense1( m )
        m = self.logits( m )
        return m

class Q_a( tf.keras.Model ):
    # Inference model over auxilliary variable (shading?)
    # Normal distribution

    def __init__( self, auxilliary_state_size ):
        super().__init__( name='Q_a' )
        self.dense1 = tf.keras.layers.Dense( 10 )
        self.mu = tf.keras.layers.Dense( auxilliary_state_size, activation=None, use_bias=True )
        self.log_sigma = tf.keras.layers.Dense( auxilliary_state_size, activation=tf.nn.sigmoid, use_bias=False )

    def call( self, d1, d2, training=True ):
        a = tf.concat( ( tf.cast( d1, d2.dtype ), d2 ), axis=-1 )
        a = self.dense1( a )
        mu, log_sigma = self.mu( a ), self.log_sigma( a )
        return mu, log_sigma

#####################################################################################

class P_m( tf.keras.Model ):
    # Generative model over missing data (ons20M)
    # Categorical distribution

    def __init__( self, missing_data_classes ):
        super().__init__( name='P_m' )
        self.dense1 = tf.keras.layers.Dense( 10 )
        self.logits = tf.keras.layers.Dense( missing_data_classes )

    def call( self, y, a, z, training=True ):
        m = tf.concat( ( y, a, z ), axis=-1 )
        m = self.dense1( m )
        m = self.logits( m )
        return m

class P_d1( tf.keras.Model ):
    # Generative model over known data (count features)
    # Multinomial distribution

    def __init__( self, known_data_1_size ):
        super().__init__( name='P_d1' )
        self.dense1 = tf.keras.layers.Dense( 10 )
        self.logits = tf.keras.layers.Dense( known_data_1_size )

    def call( self, y, a, z, training=True ):
        d1 = tf.concat( ( y, a, z ), axis=-1 )
        d1 = self.dense1( d1 )
        d1 = self.logits( d1 )
        return d1

class P_d2( tf.keras.Model ):
    # Generative model over known data (ghmm probs)
    # Dirichlet distribution

    def __init__( self, known_data_2_size ):
        super().__init__( name='P_d2' )
        self.dense1 = tf.keras.layers.Dense( 10 )
        self.alphas = tf.keras.layers.Dense( known_data_2_size, activation=partial( tf.nn.leaky_relu, alpha=1.0 ) )

    def call( self, d1, y, a, z, training=True ):
        d2 = tf.concat( ( tf.cast( d1, z.dtype ), tf.cast( y, z.dtype ), a, z ), axis=-1 )
        d2 = self.dense1( d2 )
        d2 = self.alphas( d2 )
        return d2

class P_a( tf.keras.Model ):
    # Generative model over auxilliary variable
    # Normal distribution

    def __init__( self, auxilliary_state_size ):
        super().__init__( name='P_a' )
        self.dense1 = tf.keras.layers.Dense( 10 )
        self.mu = tf.keras.layers.Dense( auxilliary_state_size, use_bias=True )
        self.log_sigma = tf.keras.layers.Dense( auxilliary_state_size, activation=tf.nn.sigmoid, use_bias=False )

    def call( self, y, z, training=True ):
        a = tf.concat( ( y, z ), axis=-1 )
        a = self.dense1( a )
        mu, log_sigma = self.mu( a ), self.log_sigma( a )
        return mu, log_sigma

#####################################################################################

def reshapeHelper( tensor, batch_size, n_imp, dim ):
    tensor_reshaped = tf.reshape( tensor, shape=( batch_size, 1, dim ) )
    tensor_reshaped = tf.tile( tensor_reshaped, multiples=( 1, n_imp, 1 ) )
    return tensor_reshaped

def logNormalPDF( x, mu, log_sigma ):
    log_prob = mu - x
    log_prob = -0.5 * tf.einsum( 'bx,bx->b', log_prob, log_prob / np.exp( log_sigma ) ) # -0.5*( x - mu ).T * sigma^-1 * ( x - mu )
    log_prob -= 0.5 * tf.reduce_sum( log_sigma, axis=-1 )                               # -0.5 * log( det( sigma ) )
    log_prob -= x.shape[ -1 ].value / 2 * np.log( 2 * np.pi )                           # -k / 2 * log( 2*pi )
    return log_prob

def logDirichletPDF( x, alpha ):
    return tf.reduce_sum( x * ( alpha - 1 ), axis=-1 ) \
         - tf.reduce_sum( tf.lgamma( alpha ), axis=-1 ) \
         + tf.lgamma( tf.reduce_sum( alpha, axis=-1 ) )

def train_epoch( data, models, optimizers, constants ):

    n_imp = constants[ 'n_importance_samples' ]
    n_a = constants[ 'auxilliary_state_size' ]
    n_d1 = constants[ 'known_data_1_size' ]
    n_d2 = constants[ 'known_data_2_size' ]
    n_m = constants[ 'missing_data_classes' ]
    n_classes = constants[ 'n_classes' ]
    q_a = models[ 'q_a' ]
    q_m = models[ 'q_m' ]
    q_y = models[ 'q_y' ]
    q_z = models[ 'q_z' ]
    p_a = models[ 'p_a' ]
    p_m = models[ 'p_m' ]
    p_d1 = models[ 'p_d1' ]
    p_d2 = models[ 'p_d2' ]

    for i, ( d1, d2, m, y ) in enumerate( tfe.Iterator( data ) ):

        batch_size = d1.shape[ 0 ]
        m = m.numpy()

        with tf.device( 'CPU' ):
            tf.assign_add( constants[ 'step_counter' ], 1 )

        with tf.contrib.summary.record_summaries_every_n_global_steps( constants[ 'log_interval' ], global_step=constants[ 'step_counter' ] ):

            importance_weights = []
            q_a_grads = []
            q_m_grads = []
            q_y_grads = []
            q_z_grads = []
            p_a_grads = []
            p_m_grads = []
            p_d1_grads = []
            p_d2_grads = []

            for i in range( n_imp ):

                epsilon = tf.random_normal( shape=( batch_size, n_a ) )
                epsilon = tf.cast( epsilon, tf.float64 )

                with tfe.GradientTape( persistent=True ) as g:

                    # Sample a ~ q( a | d1, d2 )
                    q_a_mu, q_a_log_sigma = q_a( d1, d2 )
                    a_sigma = tf.exp( q_a_log_sigma )
                    a = tf.einsum( 'ba,ba->ba', epsilon, tf.sqrt( a_sigma ) ) + q_a_mu

                    # Sample m ~ q( m | a, d1, d2 ) if m is missing
                    missing_mask = m.ravel() == -1
                    q_m_logits = q_m( a, d1, d2 )
                    m_samples = tf.distributions.Categorical( logits=q_m_logits ).sample( 1 )
                    m_samples = tf.transpose( m_samples, ( 1, 0 ) ).numpy()

                    # Replace the missing values in m
                    m[ missing_mask ] = m_samples[ missing_mask ]

                    # One hot m
                    m_one_hot = oneHot( m.ravel(), n_m )

                    # Sample z ~ q( z | a, y, m, d1, d2 )
                    q_z_mu, q_z_log_sigma = q_z( a, y, m, d1, d2 )
                    z_sigma = tf.exp( q_z_log_sigma )
                    z = tf.einsum( 'bz,bz->bz', epsilon, tf.sqrt( z_sigma ) ) + q_z_mu

                    # Generate q( y | a, d1, d2, m )
                    q_y_logits = q_y( a, m, d1, d2 )

                    # Generate p( m | y, a, z )
                    p_m_logits = p_m( y, a, z )

                    # Generate p( a | y, z )
                    p_a_mu, p_a_log_sigma = p_a( y, z )

                    # Generate p( d1 | y, a, z )
                    p_d1_logits = p_d1( y, a, z )

                    # Generate p( d2 | y, a, z )
                    p_d2_alpha = p_d2( d1, y, a, z )

                    # Generate p( y )
                    p_y_logits = constants[ 'y_prior' ]
                    p_y_logits = tf.reshape( p_y_logits, shape=( 1, n_classes ) )
                    p_y_logits = tf.tile( p_y_logits, multiples=( batch_size, 1 ) )

                    # Generate p( z )
                    p_z_mu, p_z_log_sigma = tf.zeros_like( q_a_mu ), tf.zeros_like( q_a_mu )

                    # Compute p( m | y, a, z ), p( a | y, z ), p( d1 | y, a, z ), p( d2 | y, a, z ), p( y ), p( z )
                    imporatance_elbo = tf.nn.softmax_cross_entropy_with_logits_v2( labels=m_one_hot, logits=p_m_logits )
                    imporatance_elbo += logNormalPDF( a, p_a_mu, p_a_log_sigma )
                    print( d1 )
                    assert 0
                    imporatance_elbo += tf.nn.softmax_cross_entropy_with_logits_v2( labels=d1, logits=p_d1_logits )
                    imporatance_elbo += logDirichletPDF( d2, p_d2_alpha )
                    print( y )
                    assert 0
                    imporatance_elbo += tf.nn.softmax_cross_entropy_with_logits_v2( labels=y, logits=p_y_logits )
                    imporatance_elbo += logNormalPDF( z, p_z_mu, p_z_log_sigma )

                    # Compute q( z | a, y, d1, d2, m ), q( y | a, d1, d2, m ), q( m | a, d1, d2 ), q( a | d1, d2 )
                    imporatance_elbo -= logNormalPDF( z, q_z_mu, q_z_log_sigma )
                    imporatance_elbo -= tf.nn.softmax_cross_entropy_with_logits_v2( labels=y, logits=q_y_logits )
                    imporatance_elbo -= tf.nn.softmax_cross_entropy_with_logits_v2( labels=m, logits=q_m_logits )
                    imporatance_elbo -= logNormalPDF( a, q_a_mu, q_a_log_sigma )

                    imporatance_elbo = -imporatance_elbo

                # Update the importance weights and gradients
                importance_weights.append( tf.reduce_sum( imporatance_elbo, axis=0 ) )

                q_a_grads.append( g.gradient( imporatance_elbo, q_a.variables ) )
                q_m_grads.append( g.gradient( imporatance_elbo, q_m.variables ) )
                q_y_grads.append( g.gradient( imporatance_elbo, q_y.variables ) )
                q_z_grads.append( g.gradient( imporatance_elbo, q_z.variables ) )
                p_a_grads.append( g.gradient( imporatance_elbo, p_a.variables ) )
                p_m_grads.append( g.gradient( imporatance_elbo, p_m.variables ) )
                p_d1_grads.append( g.gradient( imporatance_elbo, p_d1.variables ) )
                p_d2_grads.append( g.gradient( imporatance_elbo, p_d2.variables ) )

            # Compute the gradients
            total = tf.reduce_sum( importance_weights )

            print( 'total', total )

            q_a_grad = [ tf.zeros_like( v ) for v in q_a.variables ]
            q_m_grad = [ tf.zeros_like( v ) for v in q_m.variables ]
            q_y_grad = [ tf.zeros_like( v ) for v in q_y.variables ]
            q_z_grad = [ tf.zeros_like( v ) for v in q_z.variables ]
            p_a_grad = [ tf.zeros_like( v ) for v in p_a.variables ]
            p_m_grad = [ tf.zeros_like( v ) for v in p_m.variables ]
            p_d1_grad = [ tf.zeros_like( v ) for v in p_d1.variables ]
            p_d2_grad = [ tf.zeros_like( v ) for v in p_d2.variables ]

            for i in range( n_imp ):
                w = importance_weights[ i ] / total

                for j, g in enumerate( q_a_grads[ i ] ):
                    q_a_grad[ j ] += w * g

                for j, g in enumerate( q_m_grads[ i ] ):
                    q_m_grad[ j ] += w * g

                for j, g in enumerate( q_y_grads[ i ] ):
                    q_y_grad[ j ] += w * g

                for j, g in enumerate( q_z_grads[ i ] ):
                    q_z_grad[ j ] += w * g

                for j, g in enumerate( p_a_grads[ i ] ):
                    p_a_grad[ j ] += w * g

                for j, g in enumerate( p_m_grads[ i ] ):
                    p_m_grad[ j ] += w * g

                for j, g in enumerate( p_d1_grads[ i ] ):
                    p_d1_grad[ j ] += w * g

                for j, g in enumerate( p_d2_grads[ i ] ):
                    p_d2_grad[ j ] += w * g

            # Take an optimization step
            optimizers[ 'q_a_opt' ].apply_gradients( zip( q_a_grad, q_a.variables ) )
            optimizers[ 'q_m_opt' ].apply_gradients( zip( q_m_grad, q_m.variables ) )
            optimizers[ 'q_y_opt' ].apply_gradients( zip( q_y_grad, q_y.variables ) )
            optimizers[ 'q_z_opt' ].apply_gradients( zip( q_z_grad, q_z.variables ) )
            optimizers[ 'p_a_opt' ].apply_gradients( zip( p_a_grad, p_a.variables ) )
            optimizers[ 'p_m_opt' ].apply_gradients( zip( p_m_grad, p_m.variables ) )
            optimizers[ 'p_d1_opt' ].apply_gradients( zip( p_d1_grad, p_d1.variables ) )
            optimizers[ 'p_d2_opt' ].apply_gradients( zip( p_d2_grad, p_d2.variables ) )

#####################################################################################

def trainProbModel():

    # Hyper parameters
    latent_state_size = 12
    auxilliary_state_size = 12
    missing_data_classes = 2
    known_data_1_size = 11
    known_data_2_size = 3
    n_classes = 3

    learning_rate = 0.01

    # Save Directories
    output_dir = '/model_outputs'
    checkpoint_dir = '/model_checkpoints'

    # Models and optimizers
    models = {
        'q_z': Q_z( latent_state_size ),
        'q_y': Q_y( n_classes ),
        'q_m': Q_m( missing_data_classes ),
        'q_a': Q_a( auxilliary_state_size ),
        'p_m': P_m( missing_data_classes ),
        'p_d1': P_d1( known_data_1_size ),
        'p_d2': P_d2( known_data_2_size ),
        'p_a': P_a( auxilliary_state_size )
    }
    optimizers = {
        'q_z_opt': tf.train.AdamOptimizer( learning_rate ),
        'q_y_opt': tf.train.AdamOptimizer( learning_rate ),
        'q_m_opt': tf.train.AdamOptimizer( learning_rate ),
        'q_a_opt': tf.train.AdamOptimizer( learning_rate ),
        'p_m_opt': tf.train.AdamOptimizer( learning_rate ),
        'p_d1_opt': tf.train.AdamOptimizer( learning_rate ),
        'p_d2_opt': tf.train.AdamOptimizer( learning_rate ),
        'p_a_opt': tf.train.AdamOptimizer( learning_rate )
    }
    constants = {
        'step_counter': tf.train.get_or_create_global_step(),
        'log_interval': 10,
        'n_importance_samples': 50,
        'latent_state_size': latent_state_size,
        'auxilliary_state_size': auxilliary_state_size,
        'missing_data_classes': missing_data_classes,
        'known_data_1_size': known_data_1_size,
        'known_data_2_size': known_data_2_size,
        'n_classes': n_classes,
        'y_prior': np.ones( 3 )
    }

    # Summary and checkpoint handlers
    summary_writer = tf.contrib.summary.create_file_writer( output_dir, flush_millis=1000 )
    checkpoint_prefix = os.path.join( checkpoint_dir, 'ckpt' )
    latest_cpkt = tf.train.latest_checkpoint( checkpoint_dir )

    if( latest_cpkt ):
        print( 'Using latest checkpoint at', latest_checkpoint )

    checkpoint = tfe.Checkpoint( **models )
    checkpoint.restore( latest_cpkt )

    # Data
    train_dataset, test_dataset = loadTFDataset()

    # Training loop
    for epoch in range( 40 ):

        with summary_writer.as_default():
            train_epoch( data=train_dataset,
                         models=models,
                         optimizers=optimizers,
                         constants=constants )



trainProbModel()
# sklearnFit()