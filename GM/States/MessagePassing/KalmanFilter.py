from GenModels.GM.States.MessagePassing.FilterBase import MessagePasser
import numpy as np
from functools import reduce
from collections import namedtuple
from GenModels.GM.Distributions import Normal, Regression
from toolz import curry

__all__ = [ 'KalmanFilter',
            'SwitchingKalmanFilter' ]

class MaskedData():

    def __init__( self, data, mask, shape=None ):
        assert isinstance( data, np.ndarray )
        assert isinstance( mask, np.ndarray )
        assert mask.dtype == bool
        self.mask = mask
        self.data = data
        self.shape = shape if shape is not None else self.data.shape[ -1 ]

    def __getitem__( self, key ):
        if( np.any( self.mask[ key ] ) ):
            return np.zeros( self.shape )
        return self.data[ key ]

class KalmanFilter( MessagePasser ):
    # Kalman filter with only 1 set of parameters

    ######################################################################

    @property
    def A( self ):
        return self._A

    @property
    def sigma( self ):
        return self._sigma

    @property
    def C( self ):
        return self._C

    @property
    def R( self ):
        return self._R

    @property
    def mu0( self ):
        return self._mu0

    @property
    def sigma0( self ):
        return self._sigma0

    ######################################################################

    @property
    def u( self ):
        return self._u

    @u.setter
    def u( self, val ):
        self._u = MaskedData( *val )

    ######################################################################

    @property
    def T( self ):
        return self._T

    @T.setter
    def T( self, val ):
        self._T = val

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

    def preprocessData( self, ys, u=None ):
        ys is not None

        ys = np.array( ys )
        if( ys.ndim == 2 ):
            ys = ys[ None ]
        else:
            assert ys.ndim == 3

        assert self.J1Emiss.shape[ 0 ] == ys.shape[ 2 ]

        self._T = ys.shape[ 1 ]

        self.hy = ys.dot( self._hy ).sum( 0 )

        # P( y | x ) ~ N( -0.5 * Jy, hy )
        partition = np.vectorize( lambda J, h: Normal.log_partition( natParams=( -0.5 * J, h ) ), signature='(n,n),(n)->()' )
        self.log_Zy = partition( self.Jy, self.hy )

    def updateParams( self, A, sigma, C, R, mu0, sigma0, u=None, ys=None ):

        self.parameterCheck( A, sigma, C, R, mu0, sigma0, u=u, ys=ys )

        n1Trans, n2Trans, n3Trans = Regression.standardToNat( A, sigma )
        n1Emiss, n2Emiss, n3Emiss = Regression.standardToNat( C, R )
        n1Init, n2Init = Normal.standardToNat( mu0, sigma0 )

        self.updateNatParams( n1Trans, n2Trans, n3Trans, n1Emiss, n2Emiss, n3Emiss, n1Init, n2Init, u=u, ys=ys )

    def updateNatParams( self, n1Trans, n2Trans, n3Trans, n1Emiss, n2Emiss, n3Emiss, n1Init, n2Init, u=None, ys=None ):
        # This doesn't exactly use natural parameters, but uses J = -2 * n1 and h = n2

        self._D_latent = n1Trans.shape[ 0 ]
        self._D_obs = n1Emiss.shape[ 0 ]

        self.J11 = -2 * n1Trans
        self.J12 = -n3Trans.T
        self.J22 = -2 * n2Trans
        self.log_Z = 0.5 * np.linalg.slogdet( np.linalg.inv( self.J11 ) )[ 1 ]

        self.J1Emiss = -2 * n1Emiss
        self.Jy = -2 * n2Emiss
        self._hy = n3Emiss.T

        self.J0 = -2 * n1Init
        self.h0 = n2Init
        self.log_Z0 = Normal.log_partition( natParams=( -2 * self.J0, self.h0 ) )

        if( ys is not None ):
            self.preprocessData( ys, u=u )
        else:
            self._T = None

        if( u is not None ):
            assert u.shape == ( self.T, self.D_latent )
            uMask = np.isnan( u )
            self.u = ( u, uMask, None )
        else:
            uMask = np.zeros( self.T, dtype=bool )
            self.u = ( None, uMask, self.D_latent )

    ######################################################################

    def transitionProb( self, t, t1, forward=False ):
        # Generate P( x_t | x_t-1 ) as a function of [ x_t, x_t-1 ]

        J11 = self.J11
        J12 = self.J12
        J22 = self.J22

        u = self.u[ t ]
        h1 = J11.dot( u )
        h2 = J12.T.dot( u )
        log_Z = 0.5 * u.dot( h1 ) + self.log_Z

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

    @classmethod
    def alignOnUpper( cls, J, h, log_Z ):
        return ( J, None, None, h, None, log_Z )

    @classmethod
    def alignOnLower( cls, J, h, log_Z ):
        return ( None, None, J, None, h, log_Z )

    ######################################################################

    def forwardStep( self, t, alpha, workspace=None, out=None ):
        # Write P( y_1:t-1, x_t-1 ) in terms of [ x_t, x_t-1 ]
        _alpha = self.alignOnLower( *alpha )
        super( KalmanFilter, self ).forwardStep( t, _alpha, workspace=workspace, out=out )

    def backwardStep( self, t, beta, workspace=None, out=None ):
        # Write P( y_t+2:T | x_t+1 ) in terms of [ x_t+1, x_t ]
        _beta = self.alignOnUpper( *beta )
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

        # P( y_0, x_0 )
        return self.multiplyTerms( ( ( self.J0, self.h0, self.log_Z0 ), ( Jy, hy, log_Zy ) ) )

    def backwardBaseCase( self ):
        return [ np.zeros( ( self.D_latent, self.D_latent ) ), \
                 np.zeros( self.D_latent ), \
                 0. ]

#########################################################################################

class SwitchingKalmanFilter( KalmanFilter ):
    # Kalman filter with multiple dynamics parameters and modes

    @property
    def As( self ):
        return self._As

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

        n1Trans, n2Trans, n3Trans = zip( *[ Regression.standardToNat( A, sigma ) for A, sigma in zip( As, sigmas ) ] )
        n1Emiss, n2Emiss, n3Emiss = Regression.standardToNat( C, R )
        n1Init, n2Init = Normal.standardToNat( mu0, sigma0 )

        self.updateNatParams( z, n1Trans, n2Trans, n3Trans, n1Emiss, n2Emiss, n3Emiss, n1Init, n2Init, u=u, ys=ys )

    def updateNatParams( self, z, n1Trans, n2Trans, n3Trans, n1Emiss, n2Emiss, n3Emiss, n1Init, n2Init, u=None, ys=None ):

        self._D_latent = n2Init.shape[ 0 ]
        self._D_obs = n1Emiss.shape[ 0 ]

        self.z = z

        self.J11s = [ -2 * n for n in n1Trans ]
        self.J12s = [ -n.T for n in n3Trans ]
        self.J22s = [ -2 * n for n in n2Trans ]
        self.log_Zs = [ 0.5 * np.linalg.slogdet( np.linalg.inv( J11 ) )[ 1 ] for J11 in self.J11s ]

        self.J1Emiss = -2 * n1Emiss
        self.Jy = -2 * n2Emiss
        self._hy = n3Emiss.T

        self.J0 = -2 * n1Init
        self.h0 = n2Init
        self.log_Z0 = Normal.log_partition( natParams=( -2 * self.J0, self.h0 ) )

        if( ys is not None ):
            self.preprocessData( ys, u=u )
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