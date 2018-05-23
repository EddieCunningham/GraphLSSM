from GenModels.GM.States.StandardStates.StateBase import StateBase
from GenModels.GM.States.MessagePassing.KalmanFilter import *
from GenModels.GM.Distributions import Normal
from GenModels.GM.Utility import *
import numpy as np

__all__ = [ 'LDSState' ]

class LDSState( KalmanFilter, StateBase ):

    # priorClass = LDSModel

    @property
    def params( self ):
        return self._params

    @params.setter
    def params( self, val ):
        A, sigma, C, R, mu0, sigma0 = val
        self.updateParams( A, sigma, C, R, mu0, sigma0 )
        self._params = val

    ######################################################################

    @property
    def constParams( self ):
        return self.u

    @classmethod
    def dataN( cls, x ):
        if( x.ndim == 2 ):
            return 1
        return x.shape[ 0 ]

    ######################################################################

    @classmethod
    def standardToNat( self, A, sigma, C, R, mu0, sigma0 ):
        n1, n2, n3 = Regression.standardToNat( A, sigma )
        n4, n5, n6 = Regression.standardToNat( C, R )
        n7, n8 = Normal.standardToNat( mu0, sigma0 )
        return n1, n2, n3, n4, n5, n6, n7, n8

    @classmethod
    def natToStandard( n1, n2, n3, n4, n5, n6, n7, n8 ):
        A, sigma = Regression.natToStandard( n1, n2, n3 )
        C, R = Regression.natToStandard( n4, n5, n6 )
        mu0, sigma0 = Normal.natToStandard( n7, n8 )
        return A, sigma, C, R, mu0, sigma0

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, constParams=None, forPost=False ):
        # Compute T( x )
        ( x, ys ) = x
        u = constParams
        t1 = Regression.sufficientStats( ( x[ :-1 ], x[ 1: ] - u ), constParams=constParams )
        t2 = Regression.sufficientStats( ( x, ys ), constParams=constParams )
        t3 = Normal.sufficientStats( x[ 0 ], constParams=constParams )
        return t1, t2, t3

    @classmethod
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( natParams is None )
        A, sigma, C, R, mu0, sigma0 = params if params is not None else cls.natToStandard( *natParams )

        A1 = Regression.log_partition( params=( A, sigma ) )
        A2 = Regression.log_partition( params=( C, R ) )
        A3 = Normal.log_partition( params=( mu0, sigma0 ) )
        return A1 + A2 + A3

    ##########################################################################

    def preprocessData( self, u=None, ys=None ):
        assert u is not None and ys is not None

        if( ys is not None ):
            super( LDSState, self ).preprocessData( u, ys )
        elif( u is not None ):
            self.u = u

    ######################################################################

    def genStates( self ):
        return np.empty( ( self.T, self.D_latent ) )

    ######################################################################

    def sampleEmissions( self, x ):
        # Sample from P( y | x, ϴ )
        assert self.dataN( x=x ) == 1

        def sampleStep( _x ):
            return Normal.sample( params=( self.C.dot( _x ), self.R ) )

        return np.apply_along_axis( sampleStep, -1, x )[ None ]

    def emissionLikelihood( self, x, ys ):
        # Compute P( y | x, ϴ )
        def likelihoodStep( _x, _y ):
            return Normal.log_likelihood( _y, params=( self.C.dot( _x ), self.R ) )
        log_likelihood = np.vectorize( likelihoodStep, signature='(n),(m)->()' )
        return log_likelihood( x, ys[ 0 ] ).sum()

    ######################################################################

    def sampleStep( self, J, h ):
        return Normal.sample( natParams=( -0.5 * J, h ) )

    def likelihoodStep( self, x, J, h ):
        return Normal.log_likelihood( x, natParams=( -0.5 * J, h ) )

    ######################################################################

    def forwardArgs( self, t, beta, prevX ):
        # P( x_t+1 | x_t, y_t+1:T ) = P( y_t+1 | x_t+1 ) * P( x_t+1 | x_t ) * P( y_t+2:T | x_t+1 ) / P( y_t+1:T | x_t )
        #                           ∝ P( y_t+1 | x_t+1 ) * P( x_t+1 | x_t ) * P( y_t+2:T | x_t+1 )

        if( beta is None ):
            _J =  self.J11              if t > 0 else invPsd( self.sigma0 )
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

    ######################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, u=None, forwardFilter=True, conditionOnY=False ):
        assert params is not None
        dummy = StateBase()
        dummy.params = params
        return dummy.ilog_likelihood( x, u=u, forwardFilter=forwardFilter, conditionOnY=conditionOnY )
