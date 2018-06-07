from GenModels.GM.States.MessagePassing.FilterBase import MessagePasser
import numpy as np
from functools import reduce
from GenModels.GM.Distributions import Normal, Categorical, Transition

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

    def genWorkspace( self ):
        return np.empty( ( self.K, self.K ) )

    ######################################################################

    def parameterCheck( self, initialDist, transDist, emissionDist, ys=None ):
        assert initialDist.shape[ 0 ] == transDist.shape[ 0 ] and transDist.shape[ 0 ] == transDist.shape[ 1 ]
        assert emissionDist.shape[ 0 ] == transDist.shape[ 0 ]

        assert np.isclose( 1.0, initialDist.sum() )
        assert np.allclose( np.ones( transDist.shape[ 0 ] ), transDist.sum( axis=1 ) )
        assert np.allclose( np.ones( transDist.shape[ 0 ] ), emissionDist.sum( axis=1 ) )

    def preprocessData( self, ys ):
        ys = np.array( ys )
        self._T = ys.shape[ 1 ]

        # L refers to P( y | x ) where y is the passed in data point.
        # This is different to _L which is a matrix over all possible
        # ys and all possible xs.
        # Here compute prod_{ m in measurements }P( y_m | x )
        self.L = self._L.T[ ys ].sum( axis=0 )

    def updateParams( self, initialDist, transDist, emissionDist, ys=None ):

        self.parameterCheck( initialDist, transDist, emissionDist, ys=ys )

        nInit, = Categorical.standardToNat( initialDist )
        nTrans, = Transition.standardToNat( transDist )
        nEmiss, = Transition.standardToNat( emissionDist )

        self.updateNatParams( nInit, nTrans, nEmiss, ys=ys )

    def updateNatParams( self, log_initialDist, log_transDist, log_emissionDist, ys=None ):

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

    def emissionProb( self, t, forward=False ):
        return self.L[ t ] if forward == True else np.broadcast_to( self.L[ t ], ( self.K, self.K ) )

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

    def childParentJoint( self, t, alphas, betas ):
        # P( x_t+1, x_t, Y ) = P( y_t+1 | x_t+1 ) * P( y_t+2:T | x_t+1 ) * P( x_t+1 | x_t ) * P( x_t, y_1:t )

        emission = self.emissionProb( t + 1, forward=False )
        transition = self.transitionProb( t, t + 1, forward=False )
        alpha = np.broadcast_to( alphas[ t ], ( self.K, self.K ) ).T
        beta = np.broadcast_to( betas[ t + 1 ], ( self.K, self.K ) )

        return self.multiplyTerms( ( emission, transition, alpha, beta ) )

#########################################################################################

class GaussianForwardBackward( CategoricalForwardBackward ):
    # Gaussian emissions

    def parameterCheck( self, initialDist, transDist, mus, sigmas, ys=None ):

        assert initialDist.shape[ 0 ] == transDist.shape[ 0 ] and transDist.shape[ 0 ] == transDist.shape[ 1 ]
        assert len( mus ) == transDist.shape[ 0 ]
        assert len( sigmas ) == transDist.shape[ 0 ]
        assert np.isclose( 1.0, initialDist.sum() )
        assert np.allclose( np.ones( transDist.shape[ 0 ] ), transDist.sum( axis=1 ) )

    def preprocessData( self, ys ):
        ys = np.array( ys )
        self._T = ys.shape[ 1 ]

        # Compute all of the emission probs here.  This just makes the code cleaner
        self.L = np.zeros( ( self.T, self.K ) )

        for k in range( self.K ):

            n1, n2 = self.natMuSigmas[ k ]

            self.L[ :, k ] = Normal.log_likelihood( ys, natParams=( n1, n2 ) ).sum( axis=0 )

    def updateParams( self, initialDist, transDist, mus, sigmas, ys=None ):

        self.parameterCheck( initialDist, transDist, mus, sigmas, ys )

        nInit, = Categorical.standardToNat( initialDist )
        nTrans, = Transition.standardToNat( transDist )
        nEmiss = [ Normal.standardToNat( mu, sigma ) for mu, sigma in zip( mus, sigmas ) ]

        self.updateNatParams( nInit, nTrans, nEmiss, ys=ys )

    def updateNatParams( self, log_initialDist, log_transDist, natMuSigmas, ys=None ):

        self._K = log_transDist.shape[ 0 ]

        self.pi0 = log_initialDist
        self.pi  = log_transDist
        self.natMuSigmas = natMuSigmas

        if( ys is not None ):
            self.preprocessData( ys )
        else:
            self._T = None

#########################################################################################

class SLDSForwardBackward( CategoricalForwardBackward ):

    def parameterCheck( self, initialDist, transDist, mu0, sigma0, u, As, sigmas, ys=None ):

        assert initialDist.shape[ 0 ] == transDist.shape[ 0 ] and transDist.shape[ 0 ] == transDist.shape[ 1 ]
        assert np.isclose( 1.0, initialDist.sum() )
        assert np.allclose( np.ones( transDist.shape[ 0 ] ), transDist.sum( axis=1 ) )

    def preprocessData( self, ys ):
        ys = np.array( ys )
        self._T = ys.shape[ 1 ]

        # This is ugly, but all it does is calculate the emission likelihood over all of
        # the data sets and over all of the modes.  Not sure why I did it like this.  Probably change
        # at some point in the future

        sig0Inv = np.linalg.inv( self.sigma0 )
        self.L0 = -0.5 * np.einsum( 'ni,ij,nj', ys[ :, 0 ] - self.mu0, sig0Inv, ys[ :, 0 ] - self.mu0 ) - \
                   0.5 * np.linalg.slogdet( self.sigma0 )[ 1 ] - \
                   self.K / 2 * np.log( 2 * np.pi )

        sigInvs = np.linalg.inv( self.sigmas )
        mus = ys[ :, 1: ] - np.einsum( 'kij,ntj->knti', self.As, ys[ :, :-1 ] ) - self.u[ :-1 ]
        self.L = -0.5 * np.einsum( 'knti,kij,kntj->tk', mus, sigInvs, mus ) - \
                  0.5 * np.linalg.slogdet( self.sigmas )[ 1 ] - \
                  self.K / 2 * np.log( 2 * np.pi )

    def updateParams( self, initialDist, transDist, mu0, sigma0, u, As, sigmas, ys=None ):

        self.parameterCheck( initialDist, transDist, mu0, sigma0, u, As, sigmas, ys=ys )
        self._K = transDist.shape[ 0 ]

        self.pi0 = np.log( initialDist )
        self.pi  = np.log( transDist )

        self.mu0 = mu0
        self.sigma0 = sigma0

        self.u = u

        self.As = As
        self.sigmas = sigmas

        if( ys is not None ):
            self.preprocessData( ys )
        else:
            self._T = None

    def updateNatParams( self, initialDist, transDist, mu0, sigma0, u, As, sigmas, ys=None ):
        assert 0, 'Redo this class'

    ######################################################################

    def emissionProb( self, t, forward=False ):
        return self.L[ t - 1 ]

    ######################################################################

    def forwardBaseCase( self ):
        return self.pi0 + self.L0

    def marginalForward( self, lastAlpha ):
        return np.logaddexp.reduce( lastAlpha )

    def marginalBackward( self, firstBeta ):
        return np.logaddexp.reduce( firstBeta + self.forwardBaseCase() )
