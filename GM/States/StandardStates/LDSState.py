from GenModels.GM.States.StandardStates.StateBase import StateBase
from GenModels.GM.States.MessagePassing.KalmanFilter import *
from GenModels.GM.Distributions import Normal, Regression, InverseWishart, MatrixNormalInverseWishart, NormalInverseWishart
import numpy as np
from GenModels.GM.Utility import stabilize as stab

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

    def __init__( self, A=None, sigma=None, C=None, R=None, mu0=None, sigma0=None, prior=None, hypers=None ):
        definePrior()
        super( LDSState, self ).__init__( A, sigma, C, R, mu0, sigma0, prior=prior, hypers=hypers )

    @property
    def params( self ):
        if( self.naturalChanged ):
            self._params = self.natToStandard( *self.natParams )
            self.naturalChanged = False
        return self._params

    @property
    def natParams( self ):
        if( self.standardChanged ):
            self._natParams = self.standardToNat( *self.params )
            self.standardChanged = False
        return self._natParams

    @params.setter
    def params( self, val ):
        self.standardChanged = True
        A, sigma, C, R, mu0, sigma0 = val
        self.updateParams( A, sigma, C, R, mu0, sigma0 )
        self._params = val

    @natParams.setter
    def natParams( self, val ):
        self.naturalChanged = True
        n1, n2, n3, n4, n5, n6, n7, n8 = val
        self.updateNatParams( n1, n2, n3, n4, n5, n6, n7, n8 )
        self._natParams = val

    ######################################################################

    @property
    def constParams( self ):
        return self.u

    @classmethod
    def dataN( cls, x, conditionOnY=False, checkY=False ):
        cls.checkShape( x, conditionOnY=conditionOnY, checkY=checkY )
        if( conditionOnY == False ):
            x, y = x
        if( isinstance( x, tuple ) ):
            return len( x )
        return 1

    @classmethod
    def sequenceLength( cls, x, conditionOnY=False, checkY=False ):
        cls.checkShape( x, conditionOnY=conditionOnY, checkY=checkY )

        if( cls.dataN( x, conditionOnY=conditionOnY, checkY=checkY ) == 1 ):
            if( conditionOnY == False ):
                xs, ys = x
                return xs.shape[ 0 ]

            if( checkY == False ):
                return x.shape[ 0 ]
            else:
                return x.shape[ 1 ]
        else:
            assert 0, 'Only pass in a single example'

    @classmethod
    def unpackSingleSample( cls, x, conditionOnY=False, checkY=False ):
        if( conditionOnY == False ):
            xs, ys = x
            return xs[ 0 ], ys[ 0 ]
        return x[ 0 ]

    @classmethod
    def sampleShapes( cls, conditionOnY=False ):
        # We can have multiple measurements for the same latent state
        # ( ( Sample #, time, dim1 ), ( Sample #, measurement #, time, dim2 ) )
        if( conditionOnY == False ):
            return ( ( None, None, None ), ( None, None, None, None ) )
        return ( None, None, None )

    def isampleShapes( cls, conditionOnY=False ):
        if( conditionOnY == False ):
            return ( ( None, self.T, self.D_latent ), ( None, self.T, None, self.D_obs ) )
        return ( None, self.T, self.D_latent )

    @classmethod
    def checkShape( cls, x, conditionOnY=False, checkY=False ):
        if( conditionOnY == False ):
            xs, ys = x
            if( isinstance( xs, tuple ) ):
                assert isinstance( ys, tuple )
                assert len( xs ) == len( ys )
                for x, y in zip( xs, ys ):
                    assert x.shape[ 0 ] == y.shape[ 1 ]
            else:
                assert isinstance( xs, np.ndarray )
                assert isinstance( ys, np.ndarray )
                assert xs.ndim == 2
                assert ys.ndim == 2
                assert xs[ 0 ].shape == ys[ 1 ].shape
        else:
            if( isinstance( x, tuple ) ):
                for _x in x:
                    if( checkY == True ):
                        assert _x.ndim == 3 or _x.ndim == 2
                    else:
                        assert _x.ndim == 2
            else:
                if( checkY == True ):
                    assert x.ndim == 3 or x.ndim == 2
                else:
                    assert x.ndim == 2

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
        # Compute T( x ).  This is for when we're treating this class as P( x, y | Ѳ )
        ( x, ys ) = x
        u = constParams
        xIn  = x[ :-1 ]
        xOut = x[ 1: ] - u if u is not None else x[ 1: ]
        t1, t2, t3 = Regression.sufficientStats( x=( xIn, xOut ), constParams=constParams )
        t4, t5, t6 = Regression.sufficientStats( x=( x, ys ), constParams=constParams )
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
            return Normal.sample( natParams=( -0.5 * self.J1Emiss, self._hy.dot( _x ) ) )
        return np.apply_along_axis( sampleStep, -1, x )[ None ]

    def emissionLikelihood( self, x, ys ):
        # Compute P( y | x, ϴ )
        assert ys.size == ys.squeeze().size
        ans = 0.0
        for t, ( _x, _y ) in enumerate( zip( x, ys[ 0 ] ) ):
            ans += Normal.log_likelihood( _y, natParams=( -0.5 * self.J1Emiss, self._hy.dot( _x ) ) )
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
            _J =  self.J11                                                if t > 0 else self.J0
            _h = -self.J12.dot( prevX ) + self.J11.dot( self.u[ t - 1 ] ) if t > 0 else self.h0
            return _J, _h

        J, h, _ = beta

        if( t == 0 ):
            _J = J + self.Jy + self.J0
            _h = h + self.hy[ 0 ] + self.h0

        else:
            _J = J + self.Jy + self.J11
            _h = h + self.hy[ t ] - self.J12.dot( prevX ) + self.J11.dot( self.u[ t - 1 ] )

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

    ######################################################################

    def expectedStatsBlock( self, t, alphas, betas ):
        # E[ x_t * x_t^T ], E[ x_t+1 * x_t^T ] and E[ x_t+1 * x_t+1^T ]

        # Find the natural parameters for P( x_t+1, x_t | Y )
        # P( x_t+1, x_t | Y ) = P( y_t+1 | x_t+1 ) * P( y_t+2:T | x_t+1 ) * P( x_t+1 | x_t ) * P( x_t, y_1:t )

        Jhy = self.emissionProb( t + 1 )
        Jht = self.transitionProb( t, t + 1 )
        Jhf = self.alignOnLower( *alphas[ t ] )
        Jhb = self.alignOnUpper( *betas[ t + 1 ] )

        J11, J12, J22, h1, h2, _ = self.multiplyTerms( [ Jhy, Jht, Jhf, Jhb ] )

        J = np.block( [ J11, J12 ], [ J12.T, J22 ] )
        h = np.hstack( ( h1, h2 ) )

        # The first expected sufficient statistic for N( x_t+1, x_t | Y ) will
        # be a block matrix with blocks E[ x_t+1 * x_t+1^T ], E[ x_t+1 * x_t^T ]
        # and E[ x_t * x_t^T ]
        E = Normal.expectedSufficientStats( natParams=( -0.5 * J, h ) )[ 0 ]

        D = h1.shape[ 0 ]
        Ext1_xt1 = E[ np.ix( [ 0, D ], [ 0, D ] ) ]
        Ext1_xt = E[ np.ix( [ D, 2 * D ], [ 0, D ] ) ]
        Ext_xt = E[ np.ix( [ D, 2 * D ], [ D, 2 * D ] ) ]
        return Ext1_xt1, Ext1_xt, Ext_xt

    def conditionedExpectedSufficientStats( self, alphas, betas ):

        t1, t2, t3 = self.expectedStatsBlock( 0, alphas, betas )

        for t in range( 1, self.T - 1 ):

            Ext1_xt1, Ext1_xt, Ext_xt = self.expectedStatsBlock( t, alphas, betas )
            t1 += Ext1_xt1
            t2 += Ext1_xt
            t3 += Ext_xt

        return t1, t2, t3

    ######################################################################

    def fullSample( self, measurements=1, T=None, size=1, stabilize=False ):

        if( stabilize == True ):

            J11 = np.copy( self.J11 )
            J12 = np.copy( self.J12 )
            J22 = np.copy( self.J22 )

            A, sigma = Regression.natToStandard( -0.5 * self.J11, -0.5 * self.J22, -self.J12.T )
            A = stab( A )

            n1, n2, n3 = Regression.standardToNat( A, sigma )

            self.J11 = -2 * n1
            self.J12 = -n3.T
            self.J22 = -2 * n2
            self.log_Z = 0.5 * np.linalg.slogdet( np.linalg.inv( self.J11 ) )[ 1 ]

        ans = super( LDSState, self ).fullSample( measurements=measurements, T=T, size=size )

        if( stabilize == True ):

            self.J11 = J11
            self.J12 = J12
            self.J22 = J22
            self.log_Z = 0.5 * np.linalg.slogdet( np.linalg.inv( self.J11 ) )[ 1 ]

        return ans

    ######################################################################

    @classmethod
    def generate( cls, measurements=4, T=5, D_latent=3, D_obs=2, size=1, stabilize=False ):

        A, sigma = MatrixNormalInverseWishart.generate( D_in=D_latent, D_out=D_latent )
        C, R = MatrixNormalInverseWishart.generate( D_in=D_latent, D_out=D_obs )
        mu0, sigma0 = NormalInverseWishart.generate( D=D_latent )

        dummy = LDSState( A=A, sigma=sigma, C=C, R=R, mu0=mu0, sigma0=sigma0 )
        return dummy.isample( measurements=measurements, T=T, size=size, stabilize=stabilize )