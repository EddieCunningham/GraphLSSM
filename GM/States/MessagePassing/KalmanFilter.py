from GenModels.GM.States.MessagePassing.FilterBase import MessagePasser
import numpy as np
from functools import reduce
from GenModels.GM.Distributions import Normal
from GenModels.GM.Utility import *
from toolz import curry

__all__ = [ 'KalmanFilter',
            'SwitchingKalmanFilter' ]

class KalmanFilter( MessagePasser ):
    # Kalman filter with only 1 set of parameters

    @property
    def T( self ):
        return self._T

    @T.setter
    def T( self, val ):
        self._T = val

    @property
    def stateSize( self ):
        return self.D_latent

    @property
    def D_latent( self ):
        return self._D_latent

    @property
    def D_obs( self ):
        return self._D_obs

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

    def parameterCheck( self, A, sigma, C, R, mu0, sigma0, u=None, ys=None ):

        if( ys is not None ):
            ys = np.array( ys )
            if( ys.ndim == 2 ):
                ys = ys[ None ]
            else:
                assert ys.ndim == 3
            if( u is not None ):
                assert ys.shape[ 1 ] == u.shape[ 0 ]
            assert C.shape[ 0 ] == ys.shape[ 2 ]

        assert A.shape[ 0 ] == A.shape[ 1 ] and mu0.shape[ 0 ] == sigma0.shape[ 0 ] and sigma0.shape == A.shape
        if( u is not None ):
            assert A.shape[ 0 ] == u.shape[ 1 ]
        assert A.shape == sigma.shape

        assert C.shape[ 0 ] == R.shape[ 0 ] and R.shape[ 0 ] == R.shape[ 1 ]
        assert C.shape[ 1 ] == A.shape[ 0 ]

    def preprocessData( self, u, ys ):
        assert u is not None and ys is not None
        self.u = u

        ys = np.array( ys )
        if( ys.ndim == 2 ):
            ys = ys[ None ]
        else:
            assert ys.ndim == 3

        assert self.C.shape[ 0 ] == ys.shape[ 2 ]
        if( u is not None ):
            assert ys.shape[ 1 ] == u.shape[ 0 ]
            assert u.shape[ 1 ] == self.D_latent

        self._T = ys.shape[ 1 ]

        RInv = invPsd( self.R )

        # P( y | x ) ~ N( -0.5 * Jy, hy )
        self.hy = ys.dot( RInv @ self.C ).sum( 0 )
        self.Jy = self.C.T @ RInv @ self.C
        partition = np.vectorize( lambda J, h: Normal.log_partition( natParams=( -0.5 * J, h ) ), signature='(n,n),(n)->()' )
        self.log_Zy = partition( self.Jy, self.hy )

    def updateParams( self, A, sigma, C, R, mu0, sigma0, u=None, ys=None ):

        self.parameterCheck( A, sigma, C, R, mu0, sigma0, u=u, ys=ys )

        self._D_latent = A.shape[ 0 ]
        self._D_obs = C.shape[ 0 ]

        self.A = A
        self.sigma = sigma

        sigInv = invPsd( sigma )
        self.J11 = sigInv
        self.J12 = -sigInv @ A
        self.J22 = A.T @ sigInv @ A
        self.log_Z = 0.5 * np.linalg.slogdet( sigma )[ 1 ]

        self.C = C
        self.R = R

        self.mu0 = mu0
        self.sigma0 = sigma0

        if( ys is not None ):
            self.preprocessData( u, ys )
        else:
            self._T = None
            self.u = None

    ######################################################################

    def transitionProb( self, t, t1, forward=False ):

        J11 = self.J11
        J12 = self.J12
        J22 = self.J22

        if( self.u is not None ):
            u = self.u[ t ]
            h1 = J11.dot( u )
            h2 = J12.T.dot( u )
            log_Z = 0.5 * u.dot( h1 ) + self.log_Z
        else:
            h1 = np.zeros( J11.shape[ 0 ] )
            h2 = np.zeros_like( h1 )
            log_Z = self.log_Z

        return J11, J12, J22, h1, h2, log_Z

    ######################################################################

    def emissionProb( self, t, forward=False ):

        # P( y_t | x_t ) as a function of x_t
        J = self.Jy
        h = self.hy[ t ]
        log_Z = self.log_Zy[ t ]

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

#########################################################################################

class SwitchingKalmanFilter( KalmanFilter ):
    # Kalman filter with multiple dynamics parameters and modes

    def parameterCheck( self, z, As, sigmas, C, R, mu0, sigma0, u=None, ys=None ):

        if( ys is not None ):
            ys = np.array( ys )
            if( ys.ndim == 2 ):
                ys = ys[ None ]
            else:
                assert ys.ndim == 3
            if( u is not None ):
                assert ys.shape[ 1 ] == u.shape[ 0 ]
            assert ys.shape[ 1 ] == z.shape[ 0 ]
            assert C.shape[ 0 ] == ys.shape[ 2 ]

        if( u is not None ):
            u.shape[ 1 ] == mu0.shape[ 0 ]
        assert mu0.shape[ 0 ] == sigma0.shape[ 0 ]

        assert len( As ) == len( sigmas )
        for A, sigma in zip( As, sigmas ):
            assert A.shape[ 0 ] == A.shape[ 1 ] and sigma0.shape == A.shape
            if( u is not None ):
                assert A.shape[ 0 ] == u.shape[ 1 ]
            assert A.shape == sigma.shape

        assert C.shape[ 0 ] == R.shape[ 0 ] and R.shape[ 0 ] == R.shape[ 1 ]
        assert C.shape[ 1 ] == A.shape[ 1 ]

    def updateParams( self, z, As, sigmas, C, R, mu0, sigma0, u=None, ys=None ):

        self.parameterCheck( z, As, sigmas, C, R, mu0, sigma0, u=u, ys=ys )
        self._D_latent = mu0.shape[ 0 ]
        self._D_obs = C.shape[ 0 ]

        self.z = z

        # Save everything because memory probably isn't a big issue
        self.As = As
        self.J11s = [ invPsd( sigma ) for sigma in sigmas ]
        self.J12s = [ -sigInv @ A for A, sigInv in zip( self.As, self.J11s ) ]
        self.J22s = [ A.T @ sigInv @ A for A, sigInv in zip( self.As, self.J11s ) ]
        self.log_Zs = np.array( [ 0.5 * np.linalg.slogdet( sigma )[ 1 ] for sigma in sigmas ] )

        self.C = C
        self.R = R

        self.mu0 = mu0
        self.sigma0 = sigma0

        if( ys is not None ):
            self.preprocessData( u, ys )
        else:
            self._T = None
            self.u = None

    ######################################################################

    def transitionProb( self, t, t1, forward=False ):
        u = self.u[ t ]
        k = self.z[ t1 ]

        J11 = self.J11s[ k ]
        J12 = self.J12s[ k ]
        J22 = self.J22s[ k ]

        h1 = J11.dot( u )
        h2 = J12.T.dot( u )

        log_Z = 0.5 * u.dot( h1 ) + self.log_Zs[ k ]

        return J11, J12, J22, h1, h2, log_Z