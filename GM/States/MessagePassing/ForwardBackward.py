from GenModels.GM.States.MessagePassing.FilterBase import MessagePasser
import numpy as np
from functools import reduce
from GenModels.GM.Distributions import Normal, Categorical, Transition, Regression

__all__ = [ 'CategoricalForwardBackward',
            'GaussianForwardBackward',
            'SLDSForwardBackward' ]

#########################################################################################

class CategoricalForwardBackward( MessagePasser ):
    # Categorical emissions.  Everything is done in log space

    @property
    def K( self ):
        return self._K

    @property
    def D_latent( self ):
        return self.K

    @property
    def D_obs( self ):
        return self._emissionSize

    def genFilterProbs( self ):
        return np.empty( ( self.T, self.K ) )

    ######################################################################

    def parameterCheck( self, initialDist, transDist, emissionDist, ys=None ):
        assert initialDist.shape[ 0 ] == transDist.shape[ 0 ] and transDist.shape[ 0 ] == transDist.shape[ 1 ]
        assert emissionDist.shape[ 0 ] == transDist.shape[ 0 ]

        assert np.isclose( 1.0, initialDist.sum() )
        assert np.allclose( np.ones( transDist.shape[ 0 ] ), transDist.sum( axis=1 ) )
        assert np.allclose( np.ones( transDist.shape[ 0 ] ), emissionDist.sum( axis=1 ) )

    def preprocessData( self, ys, computeMarginal=True ):
        ys = np.array( ys )
        self._T = ys.shape[ 1 ]

        # L refers to P( y | x ) where y is the passed in data point.
        # This is different to _L which is a matrix over all possible
        # ys and all possible xs.
        # Here compute prod_{ m in measurements }P( y_m | x )
        self.L = self._L.T[ ys ].sum( axis=0 )

    def updateParams( self, initialDist, transDist, emissionDist, ys=None, computeMarginal=True ):

        if( not( hasattr( self, 'paramCheck' ) and self.paramCheck == False ) ):
            self.parameterCheck( initialDist, transDist, emissionDist, ys=ys )

        nInit, = Categorical.standardToNat( initialDist )
        nTrans, = Transition.standardToNat( transDist )
        nEmiss, = Transition.standardToNat( emissionDist )

        self.updateNatParams( nInit, nTrans, nEmiss, ys=ys )

    def updateNatParams( self, log_initialDist, log_transDist, log_emissionDist, ys=None, computeMarginal=True ):

        self._K = log_transDist.shape[ 0 ]

        self.pi0 = log_initialDist
        self.pi  = log_transDist
        self._L  = log_emissionDist
        self._emissionSize = self._L.shape[ 1 ]

        if( ys is not None ):
            self.preprocessData( ys )
        else:
            self._T = None

    ######################################################################

    def transitionProb( self, t, t1, forward=False ):

        ans = self.pi

        if( self.chainCuts is not None ):
            t1Index = self.chainCuts[ :, 0 ] == t1

            if( np.any( t1Index ) ):

                # Here, only 1 col of transitions is possible
                assert self.chainCuts[ t1Index ].size == 2
                ans = np.empty_like( self.pi )
                ans[ : ] = np.NINF
                x1 = self.chainCuts[ t1Index, 1 ]
                ans[ :, x1 ] = 0.0

        return ans if forward == True else ans.T

    ######################################################################

    def emissionProb( self, t, forward=False, ys=None ):

        if( ys is None ):
            emiss = self.L[ t ]
        else:
            emiss = self._L.T[ ys[ :, t ] ].sum( axis=0 )

        return emiss if forward == True else np.broadcast_to( emiss, ( self.K, self.K ) )

    ######################################################################

    def forwardStep( self, t, alpha ):
        # Write P( y_1:t-1, x_t-1 ) in terms of [ x_t, x_t-1 ]
        _alpha = np.broadcast_to( alpha, ( self.K, self.K ) )
        return super( CategoricalForwardBackward, self ).forwardStep( t, _alpha )

    def backwardStep( self, t, beta ):
        # Write P( y_t+2:T | x_t+1 ) in terms of [ x_t+1, x_t ]
        _beta = np.broadcast_to( beta, ( self.K, self.K ) )
        return super( CategoricalForwardBackward, self ).backwardStep( t, _beta )

    ######################################################################

    def multiplyTerms( self, terms ):
        return np.add.reduce( terms  )

    ######################################################################

    def integrate( self, integrand, forward=True ):
        # Add the values in log space
        return np.logaddexp.reduce( integrand, axis=1 )

    ######################################################################

    def forwardBaseCase( self ):

        ans = self.pi0

        if( self.chainCuts is not None ):
            if( self.chainCuts[ 0, 0 ] == 0 ):
                ans = np.empty_like( self.pi0 )
                ans[ : ] = np.NINF
                x = self.chainCuts[ 0, 1 ]
                ans[ x ] = 0.0

        return ans + self.emissionProb( 0, forward=True )

    def backwardBaseCase( self ):
        return np.zeros( self.K )

    ######################################################################

    def forwardFilter( self, knownLatentStates=None ):

        if( knownLatentStates is not None ):

            assert np.abs( knownLatentStates - knownLatentStates.astype( int ) ).sum() == 0.0
            knownLatentStates = knownLatentStates.astype( int )

            if( knownLatentStates.size == 0 ):
                self.chainCuts = None
            else:

                # Assert that knownLatentStates is sorted
                assert np.any( np.diff( knownLatentStates[ :, 0 ] ) <= 0 ) == False

                # Mark that we are cutting the markov chain at these indices
                self.chainCuts = knownLatentStates
        else:
            self.chainCuts = None

        return super( CategoricalForwardBackward, self ).forwardFilter()

    ######################################################################

    def backwardFilter( self, knownLatentStates=None ):
        if( knownLatentStates is not None ):

            assert np.abs( knownLatentStates - knownLatentStates.astype( int ) ).sum() == 0.0
            knownLatentStates = knownLatentStates.astype( int )

            # Assert that knownLatentStates is sorted
            assert np.any( np.diff( knownLatentStates[ :, 0 ] ) <= 0 ) == False

            # Mark that we are cutting the markov chain at these indices
            self.chainCuts = knownLatentStates
        else:
            self.chainCuts = None

        return super( CategoricalForwardBackward, self ).backwardFilter()

    ######################################################################

    def childParentJoint( self, t, alphas, betas, ys=None ):
        # P( x_t+1, x_t, Y ) = P( y_t+1 | x_t+1 ) * P( y_t+2:T | x_t+1 ) * P( x_t+1 | x_t ) * P( x_t, y_1:t )

        alpha = np.broadcast_to( alphas[ t ], ( self.K, self.K ) ).T
        transition = self.transitionProb( t, t + 1, forward=False )
        beta = np.broadcast_to( betas[ t + 1 ], ( self.K, self.K ) )
        emission = self.emissionProb( t + 1, forward=False, ys=ys )

        return self.multiplyTerms( ( alpha, transition, beta, emission ) )

    ######################################################################

    @classmethod
    def log_marginalFromAlphaBeta( cls, alpha, beta ):
        return np.logaddexp.reduce( alpha + beta )

#########################################################################################

class GaussianForwardBackward( CategoricalForwardBackward ):
    # Gaussian emissions

    def parameterCheck( self, initialDist, transDist, mus, sigmas, ys=None ):

        assert initialDist.shape[ 0 ] == transDist.shape[ 0 ] and transDist.shape[ 0 ] == transDist.shape[ 1 ]
        assert len( mus ) == transDist.shape[ 0 ]
        assert len( sigmas ) == transDist.shape[ 0 ]
        assert np.isclose( 1.0, initialDist.sum() )
        assert np.allclose( np.ones( transDist.shape[ 0 ] ), transDist.sum( axis=1 ) )

    def preprocessData( self, ys, computeMarginal=True ):
        ys = np.array( ys )
        self._T = ys.shape[ 1 ]

        # Compute all of the emission probs here.  This just makes the code cleaner
        self.L = np.zeros( ( self.T, self.K ) )

        for k in range( self.K ):

            n1 = self.n1Emiss[ k ]
            n2 = self.n2Emiss[ k ]

            self.L[ :, k ] = Normal.log_likelihood( ys, natParams=( n1, n2 ) ).sum( axis=0 )

    def updateParams( self, initialDist, transDist, mus, sigmas, ys=None, computeMarginal=True ):

        self.parameterCheck( initialDist, transDist, mus, sigmas, ys )

        nInit, = Categorical.standardToNat( initialDist )
        nTrans, = Transition.standardToNat( transDist )
        n1Emiss, n2Emiss = zip( *[ Normal.standardToNat( mu, sigma ) for mu, sigma in zip( mus, sigmas ) ] )

        self.updateNatParams( nInit, nTrans, n1Emiss, n2Emiss, ys=ys )

    def updateNatParams( self, log_initialDist, log_transDist, n1Emiss, n2Emiss, ys=None, computeMarginal=True ):

        self._K = log_transDist.shape[ 0 ]

        self.pi0 = log_initialDist
        self.pi  = log_transDist
        self.n1Emiss = n1Emiss
        self.n2Emiss = n2Emiss

        if( ys is not None ):
            self.preprocessData( ys )
        else:
            self._T = None

    ######################################################################

    def emissionProb( self, t, forward=False, ys=None ):

        if( ys is None ):
            emiss = self.L[ t ]
        else:
            emiss = np.zeros( self.K )

            for k in range( self.K ):

                n1 = self.n1Emiss[ k ]
                n2 = self.n2Emiss[ k ]

                emiss += Normal.log_likelihood( ys[ :, t ], natParams=( n1, n2 ) ).sum( axis=0 )

        return emiss if forward == True else np.broadcast_to( emiss, ( self.K, self.K ) )

#########################################################################################

class SLDSForwardBackward( CategoricalForwardBackward ):

    def parameterCheck( self, initialDist, transDist, mu0, sigma0, u, As, sigmas, xs=None ):

        assert initialDist.shape[ 0 ] == transDist.shape[ 0 ] and transDist.shape[ 0 ] == transDist.shape[ 1 ]
        assert np.isclose( 1.0, initialDist.sum() )
        assert np.allclose( np.ones( transDist.shape[ 0 ] ), transDist.sum( axis=1 ) )

    def preprocessData( self, xs, u=None, computeMarginal=True ):
        xs = np.array( xs )

        # Not going to use multiple measurements here
        assert xs.ndim == 2

        self._T = xs.shape[ 0 ]

        # Compute P( x_t | x_t-1, z ) for all of the observations over each z

        self.L0 = Normal.log_likelihood( xs[ 0 ], natParams=( self.n1_0, self.n2_0 ) )

        self.L = np.empty( ( self.T - 1, self.K ) )

        for i, ( n1, n2, n3 ) in enumerate( zip( self.n1Trans, self.n2Trans, self.n3Trans ) ):

            def ll( _x ):
                x, x1 = np.split( _x, 2 )
                return Regression.log_likelihood( ( x, x1 ), natParams=( n1, n2, n3 ) )

            self.L[ :, i ] = np.apply_along_axis( ll, -1, np.hstack( ( xs[ :-1 ], xs[ 1: ] ) ) )

    def updateParams( self, initialDist, transDist, mu0, sigma0, u, As, sigmas, xs=None, computeMarginal=True ):

        self.parameterCheck( initialDist, transDist, mu0, sigma0, u, As, sigmas, xs=xs )

        nInit, = Categorical.standardToNat( initialDist )
        nTrans, = Transition.standardToNat( transDist )
        nat1_0, nat2_0 = Normal.standardToNat( mu0, sigma0 )
        nat1Trans, nat2Trans, nat3Trans = zip( *[ Regression.standardToNat( A, sigma ) for A, sigma in zip( As, sigmas ) ] )

        self.updateNatParams( nInit, nTrans, nat1_0, nat2_0, nat1Trans, nat2Trans, nat3Trans, u=u, xs=xs )

    def updateNatParams( self, log_initialDist, log_transDist, nat1_0, nat2_0, nat1Trans, nat2Trans, nat3Trans, u=None, xs=None, computeMarginal=True ):

        self._K = log_initialDist.shape[ 0 ]

        self.pi0 = log_initialDist
        self.pi  = log_transDist

        self.n1_0 = nat1_0
        self.n2_0 = nat2_0

        self.n1Trans = nat1Trans
        self.n2Trans = nat2Trans
        self.n3Trans = nat3Trans

        if( xs is not None ):
            self.preprocessData( xs, u=u )
        else:
            self._T = None

    ######################################################################

    def childParentJoint( self, t, alphas, betas, xs=None ):
        alpha = np.broadcast_to( alphas[ t ], ( self.K, self.K ) ).T
        transition = self.transitionProb( t, t + 1, forward=False )
        beta = np.broadcast_to( betas[ t + 1 ], ( self.K, self.K ) )
        emission = self.emissionProb( t + 1, forward=False, xs=xs )

        return self.multiplyTerms( ( alpha, transition, beta, emission ) )

    ######################################################################

    def emissionProb( self, t, forward=False, xs=None ):
        if( xs is None ):
            emiss = self.L[ t - 1 ]
        else:

            emiss = np.zeros( self.K )

            for i, ( n1, n2, n3 ) in enumerate( zip( self.n1Trans, self.n2Trans, self.n3Trans ) ):

                def ll( _x ):
                    x, x1 = np.split( _x, 2 )
                    return Regression.log_likelihood( ( x, x1 ), natParams=( n1, n2, n3 ) )

                emiss[ i ] = np.apply_along_axis( ll, -1, np.hstack( ( xs[ :-1, t ], xs[ 1: , t ] ) ) )

        return emiss if forward == True else np.broadcast_to( emiss, ( self.K, self.K ) )

    ######################################################################

    def forwardBaseCase( self ):
        return self.pi0 + self.L0

    def marginalForward( self, lastAlpha ):
        return np.logaddexp.reduce( lastAlpha )

    def marginalBackward( self, firstBeta ):
        return np.logaddexp.reduce( firstBeta + self.forwardBaseCase() )
