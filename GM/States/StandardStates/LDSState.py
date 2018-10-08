from GenModels.GM.States.StandardStates.StateBase import StateBase
from GenModels.GM.States.MessagePassing.KalmanFilter import *
from GenModels.GM.Distributions import Normal, Regression, InverseWishart, MatrixNormalInverseWishart, NormalInverseWishart
import autograd.numpy as np
from GenModels.GM.Utility import stabilize as stab
from GenModels.GM.Utility import rightSolve, MaskedData, toBlocks
import itertools

__all__ = [ 'LDSState' ]

def definePrior():
    from GenModels.GM.ModelPriors.LDSMNIWPrior import LDSMNIWPrior
    LDSState.priorClass = LDSMNIWPrior

class LDSState( StableKalmanFilter, StateBase ):

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
    def initialStats( cls, x, constParams=None ):
        # Assumes that only a single element is passed in
        assert x.ndim == 1
        return Normal.sufficientStats( x=x, constParams=constParams )

    @classmethod
    def transitionStats( cls, x, constParams=None ):
        u, t = constParams
        lastX, x = x
        assert lastX.ndim == 1 and lastX.shape == x.shape
        xIn = lastX
        xOut = x - u[ t ]
        return Regression.sufficientStats( x=( xIn, xOut ) )

    @classmethod
    def emissionStats( cls, x, constParams=None ):
        _x, y = x
        assert _x.ndim == 1
        return Regression.sufficientStats( x=( _x, y ) )

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
    def posteriorPriorNatParams( cls, x=None, constParams=None, prior_params=None, prior_nat_params=None, stats=None ):
        assert ( prior_params is None ) ^ ( prior_nat_params is None )
        assert ( x is None ) ^ ( stats is None )

        prior_nat_params = prior_nat_params if prior_nat_params is not None else cls.priorClass.standardToNat( *prior_params )

        if( x is not None ):
            t1, t2, t3, t4, t5, t6, t7, t8 = cls.sufficientStats( x, constParams=constParams )
            transFactor, emissionFactor, initFactor = cls.partitionFactors( x )
            stats = [ t1, t2, t3, t4, t5, t6, t7, t8, transFactor, transFactor, emissionFactor, emissionFactor, initFactor, initFactor, initFactor ]

            for s, p in zip( stats, prior_nat_params ):
                if( isinstance( s, np.ndarray ) ):
                    assert isinstance( p, np.ndarray )
                    assert s.shape == p.shape
                else:
                    assert not isinstance( s, np.ndarray ) and not isinstance( p, np.ndarray )

        # for s in stats:
        #     print( s )
        #     print()
        #     print()

        pnp = [ np.add( s, p ) for s, p in zip( stats, prior_nat_params ) ]

        # pp = cls.priorClass.natToStandard( *pnp )
        # for p in pp:
        #     print()
        #     print( p )
        # assert 0
        return pnp

    ##########################################################################

    @classmethod
    def log_posteriorExpFam( cls, x, params=None, nat_params=None, constParams=None, prior_params=None, prior_nat_params=None, stats=None ):
        assert ( params is None ) ^ ( nat_params is None ) and ( prior_params is None ) ^ ( prior_nat_params is None )
        assert ( x is None ) ^ ( stats is None )

        if( x is not None ):
            cls.checkShape( x )
        postNatParams = cls.posteriorPriorNatParams( x=x, constParams=constParams, prior_params=prior_params, prior_nat_params=prior_nat_params, stats=stats )

        params = params if params is not None else cls.natToStandard( *nat_params )
        cls.priorClass.checkShape( params )
        stat = cls.priorClass.sufficientStats( params, constParams=constParams )
        part = cls.priorClass.log_partition( params, nat_params=postNatParams, split=True )

        return cls.priorClass.log_pdf( postNatParams, stat, part )

    ##########################################################################

    # The difference between this and the base implementation is that we don't
    # multiply the partition by dataN here.  This saves us from having to use
    # the average sequence length in the parition function.  Not sure
    # which way is more correct, but this seems to make more sense
    @classmethod
    def log_likelihoodExpFam( cls, x, constParams=None, params=None, nat_params=None ):
        assert ( params is None ) ^ ( nat_params is None )
        nat_params = nat_params if nat_params is not None else cls.standardToNat( *params )
        cls.checkShape( x )
        stats = cls.sufficientStats( x, constParams=constParams )
        A1, A2, A3, A4, A5, A6, A7 = cls.log_partition( x, nat_params=nat_params, split=True )
        transFactor, emissionFactor, initFactor = cls.partitionFactors( x )
        A1 *= transFactor
        A2 *= transFactor
        A3 *= emissionFactor
        A4 *= emissionFactor
        A5 *= initFactor
        A6 *= initFactor
        A7 *= initFactor
        part = A1 + A2 + A3 + A4 + A5 + A6 + A7
        return cls.log_pdf( nat_params, stats, part )

    ##########################################################################

    @classmethod
    def log_partition( cls, x=None, params=None, nat_params=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( nat_params is None )

        # Need to multiply each partition by the length of each sequence!!!!
        A, sigma, C, R, mu0, sigma0 = params if params is not None else cls.natToStandard( *nat_params )
        A1, A2 = Regression.log_partition( params=( A, sigma ), split=True )
        A3, A4 = Regression.log_partition( params=( C, R ), split=True )
        A5, A6, A7 = Normal.log_partition( params=( mu0, sigma0 ), split=True )

        if( split == True ):
            return A1, A2, A3, A4, A5, A6, A7
        return A1 + A2 + A3 + A4 + A5 + A6 + A7

    ##########################################################################

    @classmethod
    def log_partitionGradient( cls, params=None, nat_params=None ):
        # ?? Not sure what to do considering one of the natural parameters in Regression is redundant
        assert 0, 'Just don\'t call this.  Not sure what to do at the moment'

    def _testLogPartitionGradient( self ):
        pass

    ##########################################################################

    def preprocessData( self, u=None, ys=None, computeMarginal=True ):
        if( ys is not None ):
            super( LDSState, self ).preprocessData( ys, u=u, computeMarginal=computeMarginal )
        elif( u is not None ):
            self.u = u

    ######################################################################

    def genStates( self ):
        return np.empty( ( self.T, self.D_latent ) )

    def genEmissions( self, measurements ):
        return np.empty( ( measurements, self.T, self.D_obs ) )

    def genStats( self ):
        return ( ( np.zeros( ( self.D_latent, self.D_latent ) ),
                   np.zeros( self.D_latent ) ),

                 ( np.zeros( ( self.D_latent, self.D_latent ) ),
                   np.zeros( ( self.D_latent, self.D_latent ) ),
                   np.zeros( ( self.D_latent, self.D_latent ) ) ),

                 ( np.zeros( ( self.D_obs, self.D_obs ) ),
                   np.zeros( ( self.D_latent, self.D_latent ) ),
                   np.zeros( ( self.D_latent, self.D_obs ) ) ) )

    ######################################################################

    def sampleSingleEmission( self, x, measurements=1 ):
        assert x.size == x.squeeze().shape[ 0 ]
        return Normal.sample( nat_params=( -0.5 * self.J1Emiss, self._hy.dot( x.squeeze() ) ), size=measurements )

    def sampleEmissions( self, x ):
        # Sample from P( y | x, ϴ )
        assert x.ndim == 2
        return np.apply_along_axis( self.sampleSingleEmission, -1, x )[ None ]

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
                ans += Normal.log_likelihood( _ys, nat_params=( -0.5 * self.J1Emiss, self._hy.dot( _x ) ) )
            return ans

        else:
            # Only 1 example.  I don't think this code will ever be called
            assert x.ndim == 1
            if( ys.ndim == 1 ):
                pass
            else:
                assert ys.ndim == 2

            return Normal.log_likelihood( _ys, nat_params=( -0.5 * self.J1Emiss, self._hy.dot( _x ) ) )

    ######################################################################

    def sampleStep( self, J, h ):
        return Normal.unpackSingleSample( Normal.sample( params=Normal.natToStandard( J, h, fromPrecision=True ) ) )

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

    def conditionedSample( self, ys=None, u=None, forwardFilter=True, preprocessKwargs={}, filterKwargs={}, returnStats=False ):
        # Sample x given y

        if( u is not None and u.ndim == 3 ):
            # Multiple u's
            assert len( ys ) == len( u )
            it = zip( ys, u )
        else:
            assert u is None or u.ndim == 2
            it = zip( ys, itertools.repeat( u, len( ys ) ) )

        ans = [] if returnStats == False else [ [], [] ]
        for y, u in it:

            self.preprocessData( ys=y, u=u, computeMarginal=False, **preprocessKwargs )

            x = self.genStates() if returnStats == False else self.genStats()

            def workFuncForStateSample( lastX, t, *args ):
                nonlocal x
                x[ t ] = self.sampleStep( *args )
                return x[ t ]

            def workFuncForStats( lastX, t, *args ):
                nonlocal x
                _x = self.sampleStep( *args )
                _y = y[ :, t ]

                # Compute the initial distribution, transition and emission sufficient statistics
                if( lastX is None ):
                    stats = self.initialStats( _x, constParams=self.constParams )
                    for i, stat in enumerate( stats ):
                        x[ 0 ][ i ][ : ] = stat
                else:
                    # Change this way of passing t to constParams eventually
                    stats = self.transitionStats( ( lastX, _x ), constParams=( self.constParams, t ) )
                    for i, stat in enumerate( stats ):
                        x[ 1 ][ i ][ : ] += stat

                stats = self.emissionStats( ( _x, _y ), constParams=self.constParams )
                for i, stat in enumerate( stats ):
                    x[ 2 ][ i ][ : ] += stat

                return _x

            # Use the correct work function and filter function for the given arguments
            workFunc = workFuncForStateSample if returnStats == False else workFuncForStats
            filterFunc = self.forwardFilterBackwardRecurse if forwardFilter else self.backwardFilterForwardRecurse
            filterFunc( workFunc, **filterKwargs )

            if( returnStats == False ):
                ans.append( ( x, y ) )
            else:
                # Accumulate the statistics
                M, T = y.shape[ 0:2 ]
                ans = self.accumulateStats( ans, x, M, T )

        # Make sure that the sampled states are in the expected form
        if( returnStats == False ):
            ans = tuple( list( zip( *ans ) ) )
            self.checkShape( ans )

        return ans

    def isample( self, u=None, ys=None, measurements=1, T=None, forwardFilter=True, size=1, stabilize=False, returnStats=False, computeMarginal=True ):
        if( ys is not None ):
            preprocessKwargs = { 'computeMarginal': computeMarginal }
            filterKwargs = {}
            ans =  self.conditionedSample( ys=ys, u=u, forwardFilter=forwardFilter, preprocessKwargs=preprocessKwargs, filterKwargs=filterKwargs, returnStats=returnStats )
        else:
            ans =  self.fullSample( measurements=measurements, T=T, size=size, stabilize=stabilize, returnStats=returnStats )

        if( returnStats == True ):
            # Re-order the stats so that they fit with the prior nat params
            t7, t8, t1, t2, t3, t4, t5, t6 = ans[ 0 ]
            N, M, T, TM = ans[ 1 ]
            transFactor = T - N
            emissionFactor = TM
            initFactor = N
            ans = [ t1, t2, t3, t4, t5, t6, t7, t8, transFactor, transFactor, emissionFactor, emissionFactor, initFactor, initFactor, initFactor ]
        return ans

    ######################################################################

    def ilog_likelihood( self, x, u=None, forwardFilter=True, conditionOnY=False, expFam=False, preprocessKwargs={}, filterKwargs={}, seperateLikelihoods=False ):

        if( expFam ):
            return self.log_likelihoodExpFam( x, constParams=self.constParams, nat_params=self.nat_params )

        size = self.dataN( x )

        x, ys = x

        if( u is not None and u.ndim == 3 ):
            assert len( u ) == len( x )
            it = zip( x, ys, u )
        else:
            assert u is None or u.ndim == 2
            it = zip( x, ys, itertools.repeat( u, len( x ) ) )

        ans = np.zeros( size )

        for i, ( x, ys, u ) in enumerate( it ):

            self.preprocessData( ys=ys, u=u, **preprocessKwargs )

            def workFunc( lastX, t, *args ):
                nonlocal ans, x
                term = self.likelihoodStep( x[ t ], *args )
                ans[ i ] += term
                return x[ t ]

            if( conditionOnY == False ):
                # This is if we want to compute P( x, y | ϴ )
                self.noFilterForwardRecurse( workFunc )
                ans[ i ] += self.emissionLikelihood( x, ys )
            else:
                if( forwardFilter ):
                    # Otherwise compute P( x | y, ϴ )
                    assert conditionOnY == True
                    self.forwardFilterBackwardRecurse( workFunc, **filterKwargs )
                else:
                    assert conditionOnY == True
                    self.backwardFilterForwardRecurse( workFunc, **filterKwargs )

        if( seperateLikelihoods == True ):
            return ans

        return ans.sum()

    ######################################################################

    def EStep( self, ys=None, u=None, preprocessKwargs={}, filterKwargs={} ):
        preprocessKwargs.update( { 'u': u } )
        return super( LDSState, self ).EStep( ys=ys, preprocessKwargs=preprocessKwargs, filterKwargs=filterKwargs )

    ######################################################################

    def expectedTransitionStatsBlock( self, t, alphas, betas, ys=None, u=None ):
        # E[ x_t * x_t^T ], E[ x_t+1 * x_t^T ] and E[ x_t+1 * x_t+1^T ]

        # Find the natural parameters for P( x_t+1, x_t | Y )
        J11, J12, J22, h1, h2, _ = self.childParentJoint( t, alphas, betas, ys=ys, u=u )

        J = np.block( [ [ J11, J12 ], [ J12.T, J22 ] ] )
        h = np.hstack( ( h1, h2 ) )

        # The first expected sufficient statistic for N( x_t+1, x_t | Y ) will
        # be a block matrix with blocks E[ x_t+1 * x_t+1^T ], E[ x_t+1 * x_t^T ]
        # and E[ x_t * x_t^T ]
        E, _ = Normal.expectedSufficientStats( nat_params=( -0.5 * J, h ) )

        D = h1.shape[ 0 ]
        Ext1_xt1, Ext1_xt, Ext_xt = toBlocks( E, D )

        return Ext1_xt1, Ext1_xt, Ext_xt

    def expectedEmissionStats( self, t, ys, alphas, betas, conditionOnY=True ):
        # E[ x_t * x_t^T ], E[ y_t * x_t^T ] and E[ y_t * y_t^T ]

        # Find the expected sufficient statistic for N( y_t, x_t | Y )
        J_x, h_x, _ = np.add( alphas[ t ], betas[ t ] )

        if( conditionOnY == False ):
            J11 = self.J1Emiss
            J12 = -self._hy
            J22 = self.Jy + J_x
            D = J11.shape[ 0 ]
            J = np.block( [ [ J11, J12 ], [ J12.T, J22 ] ] )
            h = np.hstack( ( np.zeros( D ), h_x ) )

            # This is a block matrix with block E[ y_t * y_t^T ], E[ y_t * x_t^T ]
            # and E[ x_t * x_t^T ]
            E, _ = Normal.expectedSufficientStats( nat_params=( -0.5 * J, h ) )

            Eyt_yt, Eyt_xt, Ext_xt = toBlocks( E, D )
        else:
            Ext_xt, E_xt = Normal.expectedSufficientStats( nat_params=( -0.5 * J_x, h_x ) )
            Eyt_yt = np.einsum( 'mi,mj->ij', ys[ :, t ], ys[ :, t ] )
            Eyt_xt = np.einsum( 'mi,j->ij', ys[ :, t ], E_xt )

        return Eyt_yt, Eyt_xt, Ext_xt

    def expectedInitialStats( self, alphas, betas ):
        # E[ x_0 * x_0 ], E[ x_0 ]
        J, h, _ = np.add( alphas[ 0 ], betas[ 0 ] )
        return Normal.expectedSufficientStats( nat_params=( -0.5 * J, h ) )

    def conditionedExpectedSufficientStats( self, ys, u, alphas, betas, forMStep=False ):

        Ext1_xt1 = np.zeros( ( self.D_latent, self.D_latent ) )
        Ext1_xt = np.zeros( ( self.D_latent, self.D_latent ) )
        Ext_xt = np.zeros( ( self.D_latent, self.D_latent ) )
        if( forMStep ):
            Eut_ut = np.zeros( ( self.D_latent, self.D_latent ) )
            Ext_ut = np.zeros( ( self.D_latent, self.D_latent ) )
            Ext1_ut = np.zeros( ( self.D_latent, self.D_latent ) )
            allT = 0
            allM = 0

        Eyt_yt = np.zeros( ( self.D_obs, self.D_obs ) )
        Eyt_xt = np.zeros( ( self.D_obs, self.D_latent ) )
        Ext_xt_y = np.zeros( ( self.D_latent, self.D_latent ) )

        Ex0_x0 = np.zeros( ( self.D_latent, self.D_latent ) )
        Ex0 = np.zeros( self.D_latent )

        if( u is not None and u.ndim == 3 ):
            # Multiple u's
            assert len( ys ) == len( u )
            it = zip( ys, alphas, betas, u )
        else:
            assert u is None or u.ndim == 2
            if( u is None ):
                J, _, _ = alphas[ 0 ][ 0 ]
                u = np.zeros( ( len( alphas[ 0 ] ), J.shape[ 0 ] ) )
            it = zip( ys, alphas, betas, itertools.repeat( u, len( ys ) ) )

        for i, ( _ys, _alphas, _betas, _u ) in enumerate( it ):

            uMask = np.isnan( _u )
            _u = MaskedData( _u, uMask, None )

            M, T, _ = _ys.shape

            if( forMStep ):
                allT += T - 1
                allM += T * M

            for t in range( 1, T ):

                _Ext1_xt1, _Ext1_xt, _Ext_xt = self.expectedTransitionStatsBlock( t - 1, _alphas, _betas, ys=_ys, u=_u )
                Ext1_xt1 += _Ext1_xt1
                Ext1_xt += _Ext1_xt
                Ext_xt += _Ext_xt

                if( forMStep ):
                    J, h, _ = np.add( _alphas[ t - 1 ], _betas[ t - 1 ] )
                    J1, h1, _ = np.add( _alphas[ t ], _betas[ t ] )
                    Ext = Normal.natToStandard( J, h, fromPrecision=True )[ 0 ]
                    Ex1t = Normal.natToStandard( J1, h1, fromPrecision=True )[ 0 ]
                    Eut_ut += np.outer( _u[ t - 1 ], _u[ t - 1 ] )
                    Ext_ut += np.outer( Ext, _u[ t - 1 ] )
                    Ext1_ut += np.outer( Ex1t, _u[ t - 1 ] )

            for t in range( T ):
                _Eyt_yt, _Eyt_xt, _Ext_xt = self.expectedEmissionStats( t, _ys, _alphas, _betas, conditionOnY=True )
                Eyt_yt += _Eyt_yt
                Eyt_xt += _Eyt_xt
                Ext_xt_y += _Ext_xt

            _Ex0_x0, _Ex0 = self.expectedInitialStats( _alphas, _betas )
            Ex0_x0 += _Ex0_x0
            Ex0 += _Ex0

        if( forMStep ):
            return Ext1_xt1, Ext1_xt, Ext_xt, Eyt_yt, Eyt_xt, Ext_xt_y, Ex0_x0, Ex0, Eut_ut, Ext_ut, Ext1_ut, allT, allM

        return Ext1_xt1, Ext1_xt, Ext_xt, Eyt_yt, Eyt_xt, Ext_xt_y, Ex0_x0, Ex0

    ######################################################################

    def fullSample( self, stabilize=False, **kwargs ):

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

        ans = super( LDSState, self ).fullSample( **kwargs )

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
        return dummy.isample( measurements=measurements, T=T, size=size, stabilize=stabilize, computeMarginal=False )

    ######################################################################

    def EStep( self, ys=None, u=None, preprocessKwargs={}, filterKwargs={} ):

        if( u is not None and u.ndim == 3 ):
            # Multiple u's
            assert len( ys ) == len( u )
            it = zip( ys, u )
        else:
            assert u is None or u.ndim == 2
            it = zip( ys, itertools.repeat( u, len( ys ) ) )

        def work( _ys, _u ):
            self.preprocessData( ys=_ys, u=_u, **preprocessKwargs )
            a = self.forwardFilter( **filterKwargs )
            b = self.backwardFilter( **filterKwargs )

            return a, b

        alphas, betas = zip( *[ work( _ys, _u ) for _ys, _u in it ] )

        self.last_normalizer = self.ilog_marginal( ys, alphas=alphas, betas=betas )
        return alphas, betas

    ######################################################################

    def MStep( self, ys, u, alphas, betas ):

        Ext1_xt1, Ext1_xt, Ext_xt, Eyt_yt, Eyt_xt, Ext_xt_y, Ex0_x0, Ex0, Eut_ut, Ext_ut, Ext1_ut, allT, allM = self.conditionedExpectedSufficientStats( ys, u, alphas, betas, forMStep=True )

        D = len( ys )

        mu0 = Ex0 / D
        sigma0 = np.outer( mu0, mu0 ) + ( Ex0_x0 - 2 * np.outer( Ex0, mu0 ) ) / D

        ###############

        # from autograd import jacobian
        # import autograd.numpy as anp
        # sigInv = anp.linalg.inv( self.sigma )

        # def ACheck( A ):
        #     return -0.5 * anp.trace( sigInv @ Ext1_xt1 ) + \
        #            -0.5 * anp.trace( A.T @ sigInv @ A @ Ext_xt ) + \
        #            anp.trace( sigInv @ A @ Ext1_xt.T ) + \
        #            -0.5 * anp.linalg.slogdet( self.sigma )[ 1 ] + \
        #            anp.trace( sigInv @ Ext1_ut ) + \
        #            -anp.trace( A.T @ sigInv @ Ext_ut )

        # def sigmaCheck( sigInv ):
        #     return -0.5 * anp.trace( sigInv @ Ext1_xt1 ) + \
        #            -0.5 * anp.trace( self.A.T @ sigInv @ self.A @ Ext_xt ) + \
        #            -0.5 * anp.trace( sigInv @ Eut_ut ) + \
        #            anp.trace( sigInv @ self.A @ Ext1_xt.T ) + \
        #            anp.trace( sigInv @ Ext1_ut.T ) + \
        #            -anp.trace( self.A.T @ sigInv @ Ext_ut.T ) + \
        #            -0.5 * anp.linalg.slogdet( anp.linalg.inv( sigInv ) )[ 1 ]

        # def testA():
        #     # Correct
        #     return -sigInv @ ( self.A @ Ext_xt - Ext1_xt + Ext_ut )

        # def testSigma():
        #     # Correct
        #     return 0.5 * ( -Ext1_xt1 - self.A@Ext_xt@self.A.T - Eut_ut + 2*Ext1_xt@self.A.T + 2*Ext1_ut - 2*self.A@Ext_ut + self.sigma )

        # print( jacobian( ACheck )( self.A ) )
        # print( testA() )
        # print()
        # print( jacobian( sigmaCheck )( np.linalg.inv( self.sigma ) ) )
        # print( testSigma() )
        # assert 0

        ###############

        A = rightSolve( Ext_xt, Ext1_xt - Ext_ut )
        sigma = ( Ext1_xt1 + Eut_ut - Ext1_xt @ A.T - 2 * Ext1_ut + A @ Ext_ut ) / allT
        # sigma_2 = ( Ext1_xt1 + A @ Ext_xt @ A.T + Eut_ut - 2 * Ext1_xt @ A.T - 2 * Ext1_ut + 2 * A @ Ext_ut ) / allT

        C = rightSolve( Ext_xt_y , Eyt_xt )
        R = ( Eyt_yt - Eyt_xt @ C.T ) / allM
        # R_2 = ( Eyt_yt - 2 * Eyt_xt @ C.T + C @ Ext_xt_y @ C.T ) / allM

        print()
        print( 'mu0', mu0 )
        print( 'sigma0', sigma0 )
        print( np.linalg.eigvals( sigma0 ) )

        print()
        print( 'A', A )
        print( 'sigma', sigma )
        print( np.linalg.eigvals( sigma ) )

        print()
        print( 'C', C )
        print( 'R', R )
        print( np.linalg.eigvals( R ) )

        return A, sigma, C, R, mu0, sigma0

    ######################################################################

    @classmethod
    def variationalPosteriorPriorNatParams( cls, ys=None, u=None, constParams=None, params=None, nat_params=None, prior_params=None, prior_nat_params=None, return_normalizer=False ):
        assert ( params is None ) ^ ( nat_params is None )
        assert ( prior_params is None ) ^ ( prior_nat_params is None )

        # Because this will only be used to do variational inference,
        # make sure that the observed data is passed in
        assert ys is not None

        expectedStats, normalizer = cls.expectedSufficientStats( ys=ys, u=u, params=params, nat_params=nat_params, return_normalizer=True )

        # Assume that these are variational parameters
        prior_nat_params = prior_nat_params if prior_nat_params is not None else cls.priorClass.standardToNat( *prior_params )

        dataN = cls.dataN( ys, conditionOnY=True, checkY=True )
        expectedStats = expectedStats + tuple( [ dataN for _ in range( len( prior_nat_params ) - len( expectedStats ) ) ] )

        ans = [ np.add( s, p ) for s, p in zip( expectedStats, prior_nat_params ) ]
        return ans if return_normalizer == False else ( ans, normalizer )

    @classmethod
    def expectedSufficientStats( cls, ys=None, u=None, params=None, nat_params=None, return_normalizer=False, **kwargs ):
        assert ( params is None ) ^ ( nat_params is None )
        params = params if params is not None else cls.natToStandard( *nat_params )
        for p in params:
            print( p )
            print()
        dummy = cls( *params )
        return dummy.iexpectedSufficientStats( ys=ys, u=u, return_normalizer=return_normalizer, **kwargs )

    def iexpectedSufficientStats( self, ys=None, u=None, preprocessKwargs={}, filterKwargs={}, return_normalizer=False ):

        if( ys is None ):
            return super( StateBase, self ).iexpectedSufficientStats()

        alphas, betas = self.EStep( ys=ys, u=u, preprocessKwargs=preprocessKwargs, filterKwargs=filterKwargs )
        stats = self.conditionedExpectedSufficientStats( ys, u, alphas, betas )

        if( return_normalizer ):
            return stats, self.last_normalizer
        return stats
