from GenModels.GM.States.StandardStates.StateBase import StateBase
from GenModels.GM.States.MessagePassing.KalmanFilter import *
from GenModels.GM.Distributions import Normal, Regression, InverseWishart
from GenModels.GM.Utility import *
import numpy as np

# IMPORTANT NOTE
# There is a really weird heisenbug somewhere in either here or Regression.py.
# Sometimes (seemingly randomly) the A, sigma, C or R parameters change
# for different function calls and mess things up.  I'm not sure if I fixed
# it, but I haven't been able to find the root cause.

__all__ = [ 'LDSState' ]

def definePrior():
    from GenModels.GM.ModelPriors.LDSMNIWPrior import LDSMNIWPrior
    LDSState.priorClass = LDSMNIWPrior

class LDSState( KalmanFilter, StateBase ):

    priorClass = None

    def __init__( self, A=None, sigma=None, C=None, R=None, mu0=None, sigma0=None, prior=None, hypers=None, _stabilize=False ):

        # This flag will force A to have eigenvalues between 0 and 1.  This is so
        # that the sequences that are sampled don't quickly go off to infinity
        if( A is not None and _stabilize ):
            A = stabilize( A )

        definePrior()
        super( LDSState, self ).__init__( A, sigma, C, R, mu0, sigma0, prior=prior, hypers=hypers )

    @property
    def params( self ):
        return self._params

    @params.setter
    def params( self, val ):
        self.standardChanged = True
        A, sigma, C, R, mu0, sigma0 = val
        self.updateParams( A, sigma, C, R, mu0, sigma0 )
        self._params = val

    ######################################################################

    @property
    def constParams( self ):
        return self.u

    @classmethod
    def dataN( cls, x ):
        ( x, ys ) = x
        if( x.ndim == 2 ):
            return 1
        return x.shape[ 0 ]

    @classmethod
    def sequenceLength( cls, x ):
        assert cls.dataN( x ) == 1
        ( x, ys ) = x
        return Regression.dataN( ( x, ys ) )

    ######################################################################

    @classmethod
    def standardToNat( cls, A, sigma, C, R, mu0, sigma0 ):

        n1, n2, n3 = Regression.standardToNat( A, sigma )
        n4, n5, n6 = Regression.standardToNat( C, R )
        n7, n8 = Normal.standardToNat( mu0, sigma0 )
        return n1, n2, n3, n4, n5, n6, n7, n8

    @classmethod
    def natToStandard( cls, n1, n2, n3, n4, n5, n6, n7, n8 ):
        A, sigma = Regression.natToStandard( n1, n2, n3 )
        C, R = Regression.natToStandard( n4, n5, n6 )
        mu0, sigma0 = Normal.natToStandard( n7, n8 )
        return A, sigma, C, R, mu0, sigma0

    ##########################################################################

    # Need to take take into account sequence length when computing the posterior parameters.
    # TODO: GENERALIZE THE EXPONENTIAL DISTRIBUTION BASE CLASS TO DISTRIBUTIONS THAT ARE
    # MADE WITH COMPONENTS, LIKE THIS CLASS
    @classmethod
    def posteriorPriorNatParams( cls, x, constParams=None, priorParams=None, priorNatParams=None ):
        assert ( priorParams is None ) ^ ( priorNatParams is None )

        t1, t2, t3, t4, t5, t6, t7, t8 = cls.sufficientStats( x, constParams=constParams )
        priorNatParams = priorNatParams if priorNatParams is not None else cls.priorClass.standardToNat( *priorParams )

        assert cls.dataN( x ) == 1 # For the moment
        N = cls.sequenceLength( x )
        stats = [ t1, t2, t3, t4, t5, t6, t7, t8, N - 1, N - 1, N, N, 1, 1, 1 ]

        return [ np.add( s, p ) for s, p in zip( stats, priorNatParams ) ]

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, constParams=None ):
        # Compute T( x )
        assert cls.dataN( x ) == 1 # For now

        ( x, ys ) = x
        u = constParams
        xIn  = x[ :-1 ]
        xOut = x[ 1: ] - u if u is not None else x[ 1: ]
        t1, t2, t3 = Regression.sufficientStats( x=( xIn, xOut ), constParams=constParams )
        t4, t5, t6 = Regression.sufficientStats( x=( x, ys[ 0 ] ), constParams=constParams )
        t7, t8 = Normal.sufficientStats( x=x[ 0 ], constParams=constParams )
        return t1, t2, t3, t4, t5, t6, t7, t8

    @classmethod
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( natParams is None )

        N = cls.sequenceLength( x )
        ( x, ys ) = x

        # Need to multiply each partition by the length of each sequence!!!!
        A, sigma, C, R, mu0, sigma0 = params if params is not None else cls.natToStandard( *natParams )
        A1, A2 = Regression.log_partition( x=( x[ :-1 ], x[ 1: ] ), params=( A, sigma ), split=True )
        A1 *= N - 1
        A2 *= N - 1

        A3, A4 = Regression.log_partition( x=( x, ys ), params=( C, R ), split=True )
        A3 *= N
        A4 *= N

        A5, A6, A7 = Normal.log_partition( x=x[ 0 ], params=( mu0, sigma0 ), split=True )

        if( split == True ):
            return A1, A2, A3, A4, A5, A6, A7
        return A1 + A2 + A3 + A4 + A5 + A6 + A7

    ##########################################################################

    def preprocessData( self, u=None, ys=None ):
        ys is not None

        if( ys is not None ):
            super( LDSState, self ).preprocessData( ys, u=u )
        elif( u is not None ):
            self.u = u

    ######################################################################

    def genStates( self ):
        return np.empty( ( self.T, self.D_latent ) )

    ######################################################################

    def sampleEmissions( self, x ):
        # Sample from P( y | x, ϴ )
        assert x.ndim == 2
        def sampleStep( _x ):
            return Normal.sample( params=( self.C.dot( _x ), self.R ) )
        return np.apply_along_axis( sampleStep, -1, x )[ None ]

    def emissionLikelihood( self, x, ys ):
        # Compute P( y | x, ϴ )
        assert ys.size == ys.squeeze().size
        ans = 0.0
        for t, ( _x, _y ) in enumerate( zip( x, ys[ 0 ] ) ):
            term = Normal.log_likelihood( _y, params=( self.C.dot( _x ), self.R ) )
            ans += term
        return ans

    ######################################################################

    def sampleStep( self, J, h ):
        return Normal.sample( params=Normal.natToStandard( J, h, fromPrecision=True ) )

    def likelihoodStep( self, x, J, h ):
        return Normal.log_likelihood( x, params=Normal.natToStandard( J, h, fromPrecision=True ) )

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
    def sample( cls, params=None, natParams=None, u=None, ys=None, T=None, forwardFilter=True ):
        assert ( params is None ) ^ ( natParams is None )
        params = params if params is not None else cls.natToStandard( *natParams )
        dummy = cls( *params )
        return dummy.isample( u=u, ys=ys, T=T, forwardFilter=forwardFilter )

    ######################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, natParams=None, u=None, forwardFilter=True, conditionOnY=False ):
        assert ( params is None ) ^ ( natParams is None )
        params = params if params is not None else cls.natToStandard( *natParams )
        dummy = cls( *params )
        return dummy.ilog_likelihood( x, u=u, forwardFilter=forwardFilter, conditionOnY=conditionOnY )
