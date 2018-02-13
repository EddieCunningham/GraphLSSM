from MessagePassingBase import MessagePasser
import numpy as np
from functools import reduce

import os
path = os.getcwd()

import sys
sys.path.append( '/Users/Eddie/GenModels' )
from GM.Distributions import Normal
sys.path.append( path )

notNone = lambda _x: _x is not None

class KalmanFilter( MessagePasser ):
    # Kalman filter with only 1 set of parameters

    def __init__( self, T, D_latent, D_obs ):
        self.D_latent = D_latent
        self.D_obs = D_obs
        super( KalmanFilter, self ).__init__( T )

    def genFilterProbs( self ):
        return np.array( [ [ np.empty( ( self.D_latent, self.D_latent ) ), \
                             np.empty( self.D_latent ), \
                             np.array( 0. ) ] for _ in range( self.T ) ] )

    def genWorkspace( self ):
        return [ np.empty( ( self.D_latent, self.D_latent ) ), \
                 np.empty( ( self.D_latent, self.D_latent ) ), \
                 np.empty( ( self.D_latent, self.D_latent ) ), \
                 np.empty( self.D_latent ), \
                 np.empty( self.D_latent ), \
                 np.array( 0. ) ]

    ######################################################################

    def updateParams( self, y, u, A, sigma, C, R, mu0, sigma0 ):
        self.u = u

        self.A = A
        self.sigma = sigma
        self.sigInv = np.linalg.inv( sigma )
        self.J11 = self.sigInv
        self.J12 = -self.sigInv @ self.A
        self.J22 = self.A.T @ self.sigInv @ self.A
        self.log_Z = 0.5 * np.linalg.slogdet( sigma )[ 1 ]

        self.C = C
        self.R = R
        self.RInv = np.linalg.inv( R )
        self.Jy = C.T @ self.RInv @ C

        self.mu0 = mu0
        self.sigma0 = sigma0

        super( KalmanFilter, self ).updateParams( y )

    ######################################################################

    def transitionProb( self, t, t1, forward=False ):
        u = self.u[ t ]

        # These are constant because we're using one set of parameters
        J11 = self.J11
        J12 = self.J12
        J22 = self.J22

        h1 = self.sigInv.dot( u )
        h2 = self.J12.T.dot( u )

        log_Z = 0.5 * u.dot( self.sigInv ).dot( u ) + self.log_Z

        return J11, J12, J22, h1, h2, log_Z

    ######################################################################

    def emissionProb( self, t, forward=False ):

        # P( y_t | x_t ) as a function of x_t
        J = self.Jy
        h = ( self.C.T @ self.RInv ).dot( self.y[ t ] )
        log_Z = Normal.log_partition( natParams=( -0.5 * J, h ) )

        if( forward ):
            return J, h, np.array( log_Z )
        else:
            # Because this is before the integration step
            return J, None, None, h, None, log_Z

    ######################################################################

    def forwardStep( self, t, alpha, workspace=None, out=None ):
        # Write P( y_1:t-1, x_t-1 ) in terms of [ x_t, x_t-1 ]
        J, h, log_Z = alpha
        _alpha = ( None, None, J, None, h, log_Z )
        super( KalmanFilter, self ).forwardStep( t, _alpha, workspace=workspace, out=out )

    def backwardStep( self, t, beta, workspace=None, out=None ):
        # Write P( y_t+2:T | x_t+1 ) in terms of [ x_t+1, x_t ]
        J, h, log_Z = beta
        _beta = ( J, None, None, h, None, log_Z )
        super( KalmanFilter, self ).backwardStep( t, _beta, workspace=workspace, out=out )

    ######################################################################

    def multiplyTerms( self, terms, out=None ):

        if( out is not None ):
            for i, x in enumerate( zip( *terms ) ):
                filteredX = [ _x for _x in x if _x is not None ]
                np.add.reduce( filteredX, out=out[ i ] )
        else:
            return [ np.add.reduce( [_x for _x in x if _x is not None ] ) for x in zip( *terms ) ]

    ######################################################################

    def integrate( self, integrand, forward=True, out=None ):

        if( forward ):
            # Integrate x_t-1
            J, h, log_Z = Normal.marginalizeX2( *integrand )
        else:
            # Integrate x_t+1
            J, h, log_Z = Normal.marginalizeX1( *integrand )

        if( out is not None ):
            out[ 0 ] = J
            out[ 1 ] = h
            out[ 2 ] = np.array( log_Z )
        else:
            return J, h, log_Z

    ######################################################################

    def forwardBaseCase( self ):

        # P( y_0 | x_0 )
        Jy, hy, log_Zy = self.emissionProb( 0, forward=True )

        # P( x_0 )
        J, h = Normal.standardToNat( self.mu0, self.sigma0 )
        J /= -0.5
        log_Z = Normal.log_partition( params=( self.mu0, self.sigma0 ) )

        # P( y_0, x_0 )
        return self.multiplyTerms( ( ( J, h, log_Z ), ( Jy, hy, log_Zy ) ) )

    def backwardBaseCase( self ):
        return [ np.zeros( ( self.D_latent, self.D_latent ) ), \
                 np.zeros( self.D_latent ), \
                 0. ]
