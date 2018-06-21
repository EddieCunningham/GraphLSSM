from GenModels.GM.States.MessagePassing.FilterBase import MessagePasser
import numpy as np
from functools import reduce
from collections import namedtuple
from GenModels.GM.Distributions import Normal, Regression
from toolz import curry
from scipy.linalg import cho_factor, cho_solve

__all__ = [ 'KalmanFilter',
            'SwitchingKalmanFilter',
            'StableKalmanFilter' ]

_HALF_LOG_2_PI = 0.5 * np.log( 2 * np.pi )

class MaskedData():

    def __init__( self, data=None, mask=None, shape=None ):
        if( data is None ):
            assert shape is not None
            self.data = None
            self.mask = None
            self.shape = shape
        else:
            assert isinstance( data, np.ndarray )
            assert isinstance( mask, np.ndarray )
            assert mask.dtype == bool
            self.mask = mask
            self.data = data
            self.shape = shape if shape is not None else self.data.shape[ -1 ]

        # So that we don't have to alocate a new numpy array every time
        self._zero = np.zeros( self.shape )

    @property
    def zero( self ):
        return self._zero

    def __getitem__( self, key ):
        if( self.mask is None or np.any( self.mask[ key ] ) ):
            return self.zero
        return self.data[ key ]

##########################################################################

class KalmanFilter( MessagePasser ):
    # Kalman filter with only 1 set of parameters

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

        if( u is not None ):
            assert u.shape == ( self.T, self.D_latent )
            uMask = np.isnan( u )
            self.u = ( u, uMask, None )

    def updateParams( self, A, sigma, C, R, mu0, sigma0, u=None, ys=None ):

        self.parameterCheck( A, sigma, C, R, mu0, sigma0, u=u, ys=ys )

        n1Trans, n2Trans, n3Trans = Regression.standardToNat( A, sigma )
        n1Emiss, n2Emiss, n3Emiss = Regression.standardToNat( C, R )
        n1Init, n2Init = Normal.standardToNat( mu0, sigma0 )

        self.updateNatParams( n1Trans, n2Trans, n3Trans, n1Emiss, n2Emiss, n3Emiss, n1Init, n2Init, u=u, ys=ys )

        self.fromNatural = False
        self._A = A
        self._sigma = sigma
        self._C = C
        self._R = R
        self._mu0 = mu0
        self._sigma0 = sigma0

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
            self.preprocessData( ys )
        else:
            self._T = None

        if( u is not None ):
            assert u.shape == ( self.T, self.D_latent )
            uMask = np.isnan( u )
            self.u = ( u, uMask, None )
        else:
            uMask = np.zeros( self.T, dtype=bool )
            self.u = ( None, uMask, self.D_latent )

        self.fromNatural = True

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

        # Because this is before the integration step
        return self.alignOnUpper( J, h, log_Z )

    ######################################################################

    @classmethod
    def alignOnUpper( cls, J, h, log_Z ):
        return ( J, None, None, h, None, log_Z )

    @classmethod
    def alignOnLower( cls, J, h, log_Z ):
        return ( None, None, J, None, h, log_Z )

    ######################################################################

    def forwardStep( self, t, alpha ):
        # Write P( y_1:t-1, x_t-1 ) in terms of [ x_t, x_t-1 ]
        _alpha = self.alignOnLower( *alpha )
        return super( KalmanFilter, self ).forwardStep( t, _alpha )

    def backwardStep( self, t, beta ):
        # Write P( y_t+2:T | x_t+1 ) in terms of [ x_t+1, x_t ]
        _beta = self.alignOnUpper( *beta )
        return super( KalmanFilter, self ).backwardStep( t, _beta )

    ######################################################################

    def multiplyTerms( self, terms ):
        return [ np.add.reduce( [_x for _x in x if _x is not None ] ) for x in zip( *terms ) ]

    ######################################################################

    def integrate( self, integrand, forward=True ):

        if( forward ):
            # Integrate x_t-1
            J, h, log_Z = Normal.marginalizeX2( *integrand )
        else:
            # Integrate x_t+1
            J, h, log_Z = Normal.marginalizeX1( *integrand )

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

    ######################################################################

    def childParentJoint( self, t, alphas, betas ):
        # P( x_t+1, x_t, Y ) = P( y_t+1 | x_t+1 ) * P( y_t+2:T | x_t+1 ) * P( x_t+1 | x_t ) * P( x_t, y_1:t )

        Jhy = self.emissionProb( t + 1, forward=False )
        Jht = self.transitionProb( t, t + 1 )
        Jhf = self.alignOnLower( *alphas[ t ] )
        Jhb = self.alignOnUpper( *betas[ t + 1 ] )

        J11, J12, J22, h1, h2, logZ = self.multiplyTerms( [ Jhy, Jht, Jhf, Jhb ] )
        return J11, J12, J22, h1, h2, logZ

    ######################################################################

    @classmethod
    def log_marginalFromAlphaBeta( cls, alpha, beta ):
        Ja, ha, log_Za = alpha
        Jb, hb, log_Zb = beta
        return Normal.log_partition( natParams=( -0.5*( Ja + Jb ), ( ha + hb ) ) ) - ( log_Za + log_Zb )

#########################################################################################

class SwitchingKalmanFilter( KalmanFilter ):
    # Kalman filter with multiple dynamics parameters and modes

    @property
    def As( self ):
        return self._As

    ######################################################################

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

    ######################################################################

    def updateParams( self, z, As, sigmas, C, R, mu0, sigma0, u=None, ys=None ):

        self.parameterCheck( z, As, sigmas, C, R, mu0, sigma0, u=u, ys=ys )

        n1Trans, n2Trans, n3Trans = zip( *[ Regression.standardToNat( A, sigma ) for A, sigma in zip( As, sigmas ) ] )
        n1Emiss, n2Emiss, n3Emiss = Regression.standardToNat( C, R )
        n1Init, n2Init = Normal.standardToNat( mu0, sigma0 )

        self.updateNatParams( z, n1Trans, n2Trans, n3Trans, n1Emiss, n2Emiss, n3Emiss, n1Init, n2Init, u=u, ys=ys )

    ######################################################################

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
            self.preprocessData( ys )
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
        u = self.u[ t ]
        k = self.z[ t1 ]

        J11 = self.J11s[ k ]
        J12 = self.J12s[ k ]
        J22 = self.J22s[ k ]

        h1 = J11.dot( u )
        h2 = J12.T.dot( u )

        log_Z = 0.5 * u.dot( h1 ) + self.log_Zs[ k ]

        return J11, J12, J22, h1, h2, log_Z

#########################################################################################

class StableKalmanFilter( KalmanFilter ):

    # https://homes.cs.washington.edu/~ebfox/publication-files/PhDthesis_ebfox.pdf
    # Algorithms 19 and 20
    # Doesn't work with variational message passing!!!!!

    @property
    def AInv( self ):
        return self._AInv

    def updateParams( self, A, sigma, C, R, mu0, sigma0, u=None, ys=None ):
        super( StableKalmanFilter, self ).updateParams( A, sigma, C, R, mu0, sigma0, u=u, ys=ys )
        self._AInv = np.linalg.inv( A )

    ######################################################################

    def forwardStep( self, t, alpha ):

        J, h, logZ = alpha
        u = self.u[ t - 1 ]

        M = self.AInv.T @ J @ self.AInv
        H = np.linalg.solve( M.T + self.J11.T, M.T ).T
        L = -H
        L[ np.diag_indices( L.shape[ 0 ] ) ] += 1

        _J = L @ M @ L.T
        _J += H @ self.J11 @ H.T
        _J += self.Jy

        _h = h + J @ self.AInv.dot( u )
        _h = L @ self.AInv.T @ _h
        _h += self.hy[ t ]

        # Transition and last
        _logZ = 0.5 * u.dot( self.J11.dot( u ) ) + self.log_Z + logZ

        JInt = J + self.J22
        hInt = self.J12.T.dot( u ) + h
        JChol = cho_factor( JInt, lower=True )
        JInvh = cho_solve( JChol, hInt )

        # Marginalization
        _logZ += -0.5 * hInt.dot( JInvh ) + \
                 np.log( np.diag( JChol[ 0 ] ) ).sum() - \
                 self.D_latent * _HALF_LOG_2_PI

        # Emission
        _logZ += self.log_Zy[ t ]

        return _J, _h, _logZ

    ######################################################################

    def backwardStep( self, t, beta ):

        J, h, logZ = beta
        u = self.u[ t ]

        J = J + self.Jy
        h = h + self.hy[ t + 1 ]

        H = np.linalg.solve( J.T + self.J11.T, J.T ).T
        L = -H
        L[ np.diag_indices( L.shape[ 0 ] ) ] += 1

        _J = L @ J @ L.T
        _J += H @ self.J11 @ H.T
        _J = self.A.T @ _J @ self.A

        _h = h - J @ u
        _h = self.A.T @ L @ _h

        # Transition, emission and last
        _logZ = 0.5 * u.dot( self.J11.dot( u ) ) + self.log_Z + self.log_Zy[ t + 1 ] + logZ

        JInt = J + self.J11
        hInt = self.J11.dot( u ) + h
        JChol = cho_factor( JInt, lower=True )
        JInvh = cho_solve( JChol, hInt )

        # Marginalization
        _logZ += -0.5 * hInt.dot( JInvh ) + \
                 np.log( np.diag( JChol[ 0 ] ) ).sum() - \
                 self.D_latent * _HALF_LOG_2_PI

        return _J, _h, _logZ

    ######################################################################

    def forwardFilter( self ):

        alphas = self.genFilterProbs()
        alphas[ 0 ] = self.forwardBaseCase()

        # Only use this version for non-variational kalman filtering
        fStep = self.forwardStep if self.fromNatural == False else super( StableKalmanFilter, self ).forwardStep

        for t in range( 1, self.T ):
            alphas[ t ] = fStep( t, alphas[ t - 1 ] )

        return alphas

    ######################################################################

    def backwardFilter( self ):

        betas = self.genFilterProbs()
        betas[ -1 ] = self.backwardBaseCase()

        # Only use this version for non-variational kalman filtering
        bStep = self.backwardStep if self.fromNatural == False else super( StableKalmanFilter, self ).backwardStep

        for t in reversed( range( self.T - 1 ) ):
            betas[ t ] = bStep( t, betas[ t + 1 ] )

        return betas
