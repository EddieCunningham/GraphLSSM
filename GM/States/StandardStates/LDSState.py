from GenModels.GM.States.StandardStates.StateBase import StateBase
from GenModels.GM.States.MessagePassing.KalmanFilter import *
from GenModels.GM.Distributions import Normal, Regression, InverseWishart, MatrixNormalInverseWishart, NormalInverseWishart
import numpy as np
from GenModels.GM.Utility import stabilize as stab

__all__ = [ 'LDSState' ]

def definePrior():
    from GenModels.GM.ModelPriors.LDSMNIWPrior import LDSMNIWPrior
    LDSState.priorClass = LDSMNIWPrior

class LDSState( KalmanFilter, StateBase ):

    priorClass = None

    def __init__( self, A=None, sigma=None, C=None, R=None, mu0=None, sigma0=None, prior=None, hypers=None ):
        definePrior()
        super( LDSState, self ).__init__( A, sigma, C, R, mu0, sigma0, prior=prior, hypers=hypers )

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
    def nMeasurements( cls, x ):
        cls.checkShape( x )

        if( cls.dataN( x ) == 1 ):
            xs, ys = x
            if( ys.ndim == xs.ndim ):
                return 1
            return ys.shape[ 0 ]
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
                assert ys.ndim == 3
                assert xs.shape[ 0 ] == ys.shape[ 1 ]
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

    @classmethod
    def sufficientStats( cls, x, constParams=None ):
        # Compute T( x ).  This is for when we're treating this class as P( x, y | Ѳ )

        if( cls.dataN( x ) > 1 ):
            t = [ 0, 0, 0, 0, 0, 0, 0, 0 ]
            for _x, _ys in zip( *x ):
                s = cls.sufficientStats( ( _x, _ys ), constParams=constParams )
                for i in range( 8 ):
                    t[ i ] += s[ i ]
            return tuple( t )

        ( x, ys ) = x
        u = constParams

        xIn  = x[ :-1 ]
        xOut = x[ 1: ] - u[ :-1 ]

        t1, t2, t3 = Regression.sufficientStats( x=( xIn, xOut ), constParams=constParams )
        t4, t5, t6 = Regression.sufficientStats( x=( x, ys ), constParams=constParams )
        t7, t8 = Normal.sufficientStats( x=x[ 0 ], constParams=constParams )
        return t1, t2, t3, t4, t5, t6, t7, t8

    ##########################################################################

    @classmethod
    def partitionFactors( cls, x ):
        N = cls.dataN( x )
        if( N > 1 ):
            transFactor = 0
            emissionFactor = 0
            initFactor = 0
            for _x, _y in zip( *x ):
                M = cls.nMeasurements( ( _x, _y ) )
                T = cls.sequenceLength( ( _x, _y ) )
                initFactor += 1
                transFactor += ( T - 1 )
                emissionFactor += T * M
        else:
            T = cls.sequenceLength( x )
            M = cls.nMeasurements( x )
            transFactor = T - 1
            emissionFactor = T * M
            initFactor = 1

        return transFactor, emissionFactor, initFactor

    ##########################################################################

    @classmethod
    def posteriorPriorNatParams( cls, x, constParams=None, priorParams=None, priorNatParams=None ):
        assert ( priorParams is None ) ^ ( priorNatParams is None )

        t1, t2, t3, t4, t5, t6, t7, t8 = cls.sufficientStats( x, constParams=constParams )
        priorNatParams = priorNatParams if priorNatParams is not None else cls.priorClass.standardToNat( *priorParams )

        transFactor, emissionFactor, initFactor = cls.partitionFactors( x )
        stats = [ t1, t2, t3, t4, t5, t6, t7, t8, transFactor, transFactor, emissionFactor, emissionFactor, initFactor, initFactor, initFactor ]

        for s, p in zip( stats, priorNatParams ):
            if( isinstance( s, np.ndarray ) ):
                assert isinstance( p, np.ndarray )
                assert s.shape == p.shape
            else:
                assert not isinstance( s, np.ndarray ) and not isinstance( p, np.ndarray )
        return [ np.add( s, p ) for s, p in zip( stats, priorNatParams ) ]

    ##########################################################################

    @classmethod
    def log_posteriorExpFam( cls, x, params=None, natParams=None, constParams=None, priorParams=None, priorNatParams=None ):
        assert ( params is None ) ^ ( natParams is None ) and ( priorParams is None ) ^ ( priorNatParams is None )

        cls.checkShape( x )
        postNatParams = cls.posteriorPriorNatParams( x, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams )

        params = params if params is not None else cls.natToStandard( *natParams )
        cls.priorClass.checkShape( params )
        stat = cls.priorClass.sufficientStats( params, constParams=constParams )
        part = cls.priorClass.log_partition( params, natParams=postNatParams, split=True )

        return cls.priorClass.log_pdf( postNatParams, stat, part )

    ##########################################################################

    # The difference between this and the base implementation is that we don't
    # multiply the partition by dataN here.  This saves us from having to use
    # the average sequence length in the parition function.  Not sure
    # which way is more correct, but this seems to make more sense
    @classmethod
    def log_likelihoodExpFam( cls, x, constParams=None, params=None, natParams=None ):
        assert ( params is None ) ^ ( natParams is None )
        natParams = natParams if natParams is not None else cls.standardToNat( *params )
        cls.checkShape( x )
        stats = cls.sufficientStats( x, constParams=constParams )
        A1, A2, A3, A4, A5, A6, A7 = cls.log_partition( x, natParams=natParams, split=True )
        transFactor, emissionFactor, initFactor = cls.partitionFactors( x )
        A1 *= transFactor
        A2 *= transFactor
        A3 *= emissionFactor
        A4 *= emissionFactor
        A5 *= initFactor
        A6 *= initFactor
        A7 *= initFactor
        part = A1 + A2 + A3 + A4 + A5 + A6 + A7
        return cls.log_pdf( natParams, stats, part )

    ##########################################################################

    @classmethod
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( natParams is None )

        # Need to multiply each partition by the length of each sequence!!!!
        A, sigma, C, R, mu0, sigma0 = params if params is not None else cls.natToStandard( *natParams )
        A1, A2 = Regression.log_partition( params=( A, sigma ), split=True )
        A3, A4 = Regression.log_partition( params=( C, R ), split=True )
        A5, A6, A7 = Normal.log_partition( params=( mu0, sigma0 ), split=True )

        if( split == True ):
            return A1, A2, A3, A4, A5, A6, A7
        return A1 + A2 + A3 + A4 + A5 + A6 + A7

    ##########################################################################

    @classmethod
    def log_partitionGradient( cls, params=None, natParams=None ):
        # ?? Not sure what to do considering one of the natural parameters in Regression is redundant
        assert 0, 'Just don\'t call this.  Not sure what to do at the moment'

    def _testLogPartitionGradient( self ):
        pass

    ##########################################################################

    def preprocessData( self, u=None, ys=None ):
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
            return Normal.sample( natParams=( -0.5 * self.J1Emiss, self._hy.dot( _x ) ) )[ 0 ]
        return np.apply_along_axis( sampleStep, -1, x )[ None ]

    def emissionLikelihood( self, x, ys ):
        # Compute P( y | x, ϴ )
        if( x.ndim == 2 ):
            # Multiple time steps
            if( ys.ndim == 2 ):
                assert x.shape[ 0 ] == ys.shape[ 0 ]
            else:
                # There are multiple measurements per latent state
                assert ys.ndim == 3
                assert x.shape[ 0 ] == ys.shape[ 1 ]

                # Put the time index in front
                ys = np.swapaxes( ys, 0, 1 )

            assert x.shape[ 0 ] == ys.shape[ 0 ]

            ans = 0.0
            for t, ( _x, _ys ) in enumerate( zip( x, ys ) ):
                ans += Normal.log_likelihood( _ys, natParams=( -0.5 * self.J1Emiss, self._hy.dot( _x ) ) )
            return ans

        else:
            # Only 1 example.  I don't think this code will ever be called
            assert x.ndim == 1
            if( ys.ndim == 1 ):
                pass
            else:
                assert ys.ndim == 2

            return Normal.log_likelihood( _ys, natParams=( -0.5 * self.J1Emiss, self._hy.dot( _x ) ) )

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

    def isample( self, u=None, ys=None, measurements=1, T=None, forwardFilter=True, size=1, stabilize=False ):
        if( ys is not None ):
            preprocessKwargs = { 'u': u }
            filterKwargs = {}
            return self.conditionedSample( ys=ys, forwardFilter=forwardFilter, preprocessKwargs=preprocessKwargs, filterKwargs=filterKwargs )
        return self.fullSample( measurements=measurements, T=T, size=size, stabilize=stabilize )

    ######################################################################

    def ilog_likelihood( self, x, forwardFilter=True, conditionOnY=False, expFam=False, preprocessKwargs={}, filterKwargs={}, u=None, seperateLikelihoods=False ):
        preprocessKwargs.update( { 'u': u } )
        return super( LDSState, self ).ilog_likelihood( x=x, forwardFilter=forwardFilter, conditionOnY=conditionOnY, expFam=expFam, preprocessKwargs=preprocessKwargs, filterKwargs=filterKwargs, seperateLikelihoods=seperateLikelihoods )

    ######################################################################

    def EStep( self, ys=None, u=None, preprocessKwargs={}, filterKwargs={} ):
        preprocessKwargs.update( { 'u': u } )
        return super( LDSState, self ).EStep( ys=ys, preprocessKwargs=preprocessKwargs, filterKwargs=filterKwargs )

    ######################################################################

    def expectedTransitionStatsBlock( self, t, alphas, betas ):
        # E[ x_t * x_t^T ], E[ x_t+1 * x_t^T ] and E[ x_t+1 * x_t+1^T ]

        # Find the natural parameters for P( x_t+1, x_t | Y )
        J11, J12, J22, h1, h2, _ = self.childParentJoint( t, alphas, betas )

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

    def conditionedExpectedSufficientStats( self, ys, alphas, betas ):

        assert 0, 'Add in other stats'
        t1, t2, t3 = self.expectedTransitionStatsBlock( 0, alphas, betas )

        for t in range( 1, self.T - 1 ):

            Ext1_xt1, Ext1_xt, Ext_xt = self.expectedTransitionStatsBlock( t, alphas, betas )
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

    ######################################################################

    def MStep( self, stats ):
        return Regression.maxLikelihoodFromStats( stats )