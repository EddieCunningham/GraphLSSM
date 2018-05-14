from GenModels.GM.States.StandardStates.StateBase import StateBase
from GenModels.GM.States.MessagePassing.KalmanFilter import *
from GenModels.GM.Distributions import Normal
from GenModels.GM.Utility import *

__all__ = [ 'LDSState' ]

class LDSState( StateBase, KalmanFilter ):

    # priorClass = LDSModel

    @params.setter
    def params( self, val ):
        A, sigma, C, R, mu0, sigma0 = val
        self.updateParams( A, sigma, C, R, mu0, sigma0 )
        self._params = val

    @property
    def constParams( self ):
        return None

    @property
    def dataN( self, x ):
        return x.shape[ -2 ]

    ######################################################################

    def preprocessData( self, u=None, ys=None ):
        assert not ( u is None and ys is None )

        if( ys is not None ):
            super( LDSState, self ).preprocessData( u, ys )
        elif( u is not None ):
            self.u = u

    ######################################################################

    def genState( self ):
        return np.empty( ( self.T, self.D_latent ) )

    ######################################################################

    def sampleStep( self, J, h ):
        return Normal.sample( natParams=( -0.5 * J, h ) )

    def likelihoodStep( self, x, J, h ):
        return Normal.logLikelihood( x, natParams=( -0.5 * J, h ) )

    ######################################################################

    def forwardArgs( self, t, beta, prevX ):
        # P( x_t+1 | x_t, y_t+1:T ) = P( y_t+1 | x_t+1 ) * P( x_t+1 | x_t ) * P( y_t+2:T | x_t+1 ) / P( y_t+1:T | x_t )
        #                           ∝ P( y_t+1 | x_t+1 ) * P( x_t+1 | x_t ) * P( y_t+2:T | x_t+1 )

        if( beta is None ):
            _J = -0.5 * self.J11        if t > 0 else invPsd( self.sigma0 )
            _h = -self.J12.dot( prevX ) if t > 0 else invPsd( self.sigma0 ).dot( self.mu0 )
            return _J, _h

        J, h, _ = beta

        if( t == 0 ):
            J0, h0 = Normal.standardToNat( self.mu0, self.sigma0, returnPrecision=True )

            _J = J + self.Jy + J0
            _h = h + self.hy[ 0 ] + h0

        else:
            _J = J + self.Jy + self.J11
            _h = h + self.hy[ t ] + self.J11.dot( self.A.dot( prevX ) + self.u[ t - 1 ] )

        return _J, _h

    def backwardArgs( self, t, alpha, prevX ):
        # P( x_t | x_t+1, y_1:t ) = P( x_t+1 | x_t ) * P( x_t, y_1:t ) / sum_{ z_t }[ P( x_t+1 | z_t ) * P( z_t, y_1:t ) ]
        #                         ∝ P( x_t+1 | x_t ) * P( x_t, y_1:t )

        J, h, _ = alpha

        if( t == self.T - 1 ):
            _J = J
            _h = h
        else:
            _J = J + self.J22
            _h = h - self.J12.T.dot( prevX - self.u[ t ] )

        return _J, _h

    ######################################################################

    @classmethod
    def sample( cls, params, u=None, ys=None, T=None, forwardFilter=True ):
        dummy = StateBase()
        dummy.params = params
        return dummy.isample( u=u, ys=ys, T=T, forwardFilter=forwardFilter )

    def isample( self, u=None, ys=None, T=None, forwardFilter=True )
        if( u is not None or ys is not None ):
            self.preprocessData( u, ys )
            T = self.T
        super( LDSState, self ).isample( ys=None, T=T, forwardFilter=forwardFilter )

    ######################################################################

    @classmethod
    def log_likelihood( cls, x, params, u=None, ys=None, T=None, forwardFilter=True ):
        dummy = StateBase()
        dummy.params = params
        return dummy.ilog_likelihood( x, u=u, ys=ys, T=T, forwardFilter=forwardFilter )

    def ilog_likelihood( self, x, u=None, ys=None, T=None, forwardFilter=True )
        if( u is not None or ys is not None ):
            self.preprocessData( u, ys )
            T = self.T
        super( LDSState, self ).ilog_likelihood( x, ys=None, T=T, forwardFilter=forwardFilter )
