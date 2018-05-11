from GenModels.GM.States.StandardStates.MessagePassing.MessagePassingBase import MessagePasser
import numpy as np
from functools import reduce

import os
path = os.getcwd()

import sys
sys.path.append( '/Users/Eddie/GenModels' )
from GM.Distributions import Normal
sys.path.append( path )

#########################################################################################

class CategoricalForwardBackward( MessagePasser ):
    # Categorical emissions.  Everything is done in log space

    def __init__( self, T, K ):
        self.K = K
        super( CategoricalForwardBackward, self ).__init__( T )

    def genFilterProbs( self ):
        return np.empty( ( self.T, self.K ) )

    def genWorkspace( self ):
        return np.empty( ( self.K, self.K ) )

    ######################################################################

    def updateParams( self, ys, initialDist, transDist, emissionDist ):
        assert initialDist.shape == ( self.K, )
        assert transDist.shape == ( self.K, self.K )
        assert emissionDist.shape[ 0 ] == self.K
        assert np.isclose( 1.0, initialDist.sum() )
        assert np.allclose( np.ones( self.K ), transDist.sum( axis=1 ) )
        assert np.allclose( np.ones( self.K ), emissionDist.sum( axis=1 ) )
        self.pi0 = np.log( initialDist )
        self.pi  = np.log( transDist )
        _L   = np.log( emissionDist )

        if( not isinstance( ys, np.ndarray ) ):
            ys = np.array( ys )

        self.L = _L.T[ ys ].sum( axis=0 )

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

    def updateParams( self, ys, initialDist, transDist, mus, sigmas ):
        assert initialDist.shape == ( self.K, )
        assert transDist.shape == ( self.K, self.K )
        assert len( mus ) == self.K
        assert len( sigmas ) == self.K
        assert np.isclose( 1.0, initialDist.sum() )
        assert np.allclose( np.ones( self.K ), transDist.sum( axis=1 ) )
        self.pi0 = np.log( initialDist )
        self.pi  = np.log( transDist )

        # Compute all of the emission probs before hand
        self.L = np.zeros( ( self.T, self.K ) )

        for k in range( self.K ):
            mu = mus[ k ]
            sigma = sigmas[ k ]

            self.L[ :, k ] = Normal.log_likelihood( ys, params=( mu, sigma ) ).sum( axis=0 )

#########################################################################################

class SLDSForwardBackward( CategoricalForwardBackward ):

    def updateParams( self, ys, initialDist, transDist, mu0, sigma0, u, As, sigmas ):
        assert initialDist.shape == ( self.K, )
        assert transDist.shape == ( self.K, self.K )
        assert np.isclose( 1.0, initialDist.sum() )
        assert np.allclose( np.ones( self.K ), transDist.sum( axis=1 ) )
        self.pi0 = np.log( initialDist )
        self.pi  = np.log( transDist )

        # This is ugly, but all it does is calculate the emission likelihood over all of
        # the data sets and over all of the modes.  Definitely should rewrite using
        # tensor_dot

        sig0Inv = np.linalg.inv( sigma0 )
        self.L0 = -0.5 * np.einsum( 'ni,ij,nj', ys[ :, 0 ] - mu0, sig0Inv, ys[ :, 0 ] - mu0 ) - \
                   0.5 * np.linalg.slogdet( sigma0 )[ 1 ] - \
                   self.K / 2 * np.log( 2 * np.pi )

        sigInvs = np.linalg.inv( sigmas )
        mus = ys[ :, 1: ] - np.einsum( 'kij,ntj->knti', As, ys[ :, :-1 ] ) - u[ :-1 ]
        self.L = -0.5 * np.einsum( 'knti,kij,kntj->tk', mus, sigInvs, mus ) - \
                  0.5 * np.linalg.slogdet( sigmas )[ 1 ] - \
                  self.K / 2 * np.log( 2 * np.pi )

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
