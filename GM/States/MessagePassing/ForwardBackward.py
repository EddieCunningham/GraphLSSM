from GenModels.GM.States.MessagePassing.FilterBase import MessagePasser
import numpy as np
from functools import reduce
from GenModels.GM.Distributions import Normal

__all__ = [ 'CategoricalForwardBackward',
            'GaussianForwardBackward',
            'SLDSForwardBackward' ]

#########################################################################################

class CategoricalForwardBackward( MessagePasser ):
    # Categorical emissions.  Everything is done in log space

    def __init__( self ):
        pass

    @property
    def T(self):
        return self._T

    @property
    def K(self):
        return self._K

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
        # ys and all possible xs
        self.L = self._L.T[ ys ].sum( axis=0 )

    def updateParams( self, initialDist, transDist, emissionDist, ys=None ):

        self.parameterCheck( initialDist, transDist, emissionDist, ys=ys )
        self._K = transDist.shape[ 0 ]

        self.pi0 = np.log( initialDist )
        self.pi  = np.log( transDist )
        self._L  = np.log( emissionDist )

        if( ys is not None ):
            self.preprocessData( ys )
        else:
            self._T = None

    ######################################################################

    def transitionProb( self, t, t1, forward=False ):
        return self.pi

    ######################################################################

    def emissionProb( self, t, forward=False ):
        return self.L[ t ]

    ######################################################################

    def multiplyTerms( self, terms, out=None ):
        # Using functools.reduce instead of np.add.reduce so that numpy
        # can broadcast and add emission vector to transition matrix
        if( out is not None ):
            reduce( lambda x, y: np.add( x, y, out=out ), terms )
        else:
            return reduce( lambda x, y: np.add( x, y ), terms )

    ######################################################################

    def integrate( self, integrand, forward=True, out=None ):
        # Add the values in log space
        if( out is not None ):
            np.logaddexp.reduce( integrand, axis=0, out=out )
        else:
            return np.logaddexp.reduce( integrand, axis=0 )

    ######################################################################

    def forwardBaseCase( self ):
        return self.pi0 + self.emissionProb( 0 )

    def backwardBaseCase( self ):
        return np.zeros( self.K )

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
            mu = self.mus[ k ]
            sigma = self.sigmas[ k ]

            # Holy shit this got slow
            for _ys in ys:
                for __ys in _ys:
                    self.L[ :, k ] += Normal.log_likelihood( __ys, params=( mu, sigma ) )

    def updateParams( self, initialDist, transDist, mus, sigmas, ys=None ):

        self.parameterCheck( initialDist, transDist, mus, sigmas, ys )
        self._K = transDist.shape[ 0 ]

        self.mus = mus
        self.sigmas = sigmas

        self.pi0 = np.log( initialDist )
        self.pi  = np.log( transDist )

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
