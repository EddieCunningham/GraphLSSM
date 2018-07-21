from GenModels.GM.States.StandardStates.StateBase import StateBase
from GenModels.GM.States.MessagePassing.ForwardBackward import *
from GenModels.GM.Distributions import Categorical, Transition, Dirichlet, TransitionDirichletPrior
import numpy as np

__all__ = [ 'HMMState' ]

def definePrior():
    from GenModels.GM.ModelPriors.HMMDirichletPrior import HMMDirichletPrior
    HMMState.priorClass = HMMDirichletPrior

class HMMState( CategoricalForwardBackward, StateBase ):

    priorClass = None

    def __init__( self, initialDist=None, transDist=None, emissionDist=None, prior=None, hypers=None, paramCheck=True ):
        definePrior()
        # If we're doing variational inference, turn the parameter check off
        self.paramCheck = paramCheck
        super( HMMState, self ).__init__( initialDist, transDist, emissionDist, prior=prior, hypers=hypers )

    ######################################################################

    @property
    def initialDist( self ):
        return self._params[ 0 ]

    @property
    def transDist( self ):
        return self._params[ 1 ]

    @property
    def emissionDist( self ):
        return self._params[ 2 ]

    ######################################################################

    @property
    def constParams( self ):
        return self.D_latent, self.D_obs

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
        # ( ( Sample #, time ), ( Sample #, measurement #, time ) )
        if( conditionOnY == False ):
            return ( ( None, None ), ( None, None, None ) )
        return ( None, None )

    def isampleShapes( cls, conditionOnY=False ):
        if( conditionOnY == False ):
            return ( ( None, self.T ), ( None, None, self.T ) )
        return ( None, self.T )

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
                assert xs.ndim == 1
                assert ys.ndim == 2
                assert xs.shape[ 0 ] == ys.shape[ 1 ]
        else:
            if( isinstance( x, tuple ) ):
                for _x in x:
                    if( checkY == True ):
                        assert _x.ndim == 2 or _x.ndim == 1
                    else:
                        assert _x.ndim == 1
            else:
                if( checkY == True ):
                    assert x.ndim == 2 or x.ndim == 1
                else:
                    assert x.ndim == 1

    ######################################################################

    @classmethod
    def standardToNat( cls, initialDist, transDist, emissionDist ):
        n1, = Categorical.standardToNat( initialDist )
        n2, = Transition.standardToNat( transDist )
        n3, = Transition.standardToNat( emissionDist )
        return n1, n2, n3

    @classmethod
    def natToStandard( cls, n1, n2, n3 ):
        initialDist, = Categorical.natToStandard( n1 )
        transDist, = Transition.natToStandard( n2 )
        emissionDist, = Transition.natToStandard( n3 )
        return initialDist, transDist, emissionDist

    ##########################################################################

    @classmethod
    def initialStats( cls, x, constParams=None ):
        # Assumes that only a single element is passed in
        D_latent, D_obs = constParams
        return Categorical.sufficientStats( x=[ x ], constParams=D_latent )

    @classmethod
    def transitionStats( cls, x, constParams=None ):
        ( D_latent, D_obs ), t = constParams
        lastX, x = x
        return Transition.sufficientStats( ( lastX, x ), constParams=( D_latent, D_latent ) )

    @classmethod
    def emissionStats( cls, x, constParams=None ):
        D_latent, D_obs = constParams
        x, y = x
        return Transition.sufficientStats( ( x, y ), constParams=( D_latent, D_obs ) )

    @classmethod
    def sufficientStats( cls, x, constParams=None ):
        # Compute T( x )

        if( cls.dataN( x ) > 1 ):
            t = [ 0, 0, 0 ]
            for j, ( _x, _ys ) in enumerate( zip( *x ) ):
                s = cls.sufficientStats( ( _x, _ys ), constParams=constParams )
                for i in range( 3 ):
                    t[ i ] += s[ i ]
            return tuple( t )

        assert cls.dataN( x ) == 1
        ( x, ys ) = x
        assert constParams is not None
        D_latent, D_obs = constParams
        t1, = Categorical.sufficientStats( [ x[ 0 ] ] , constParams=D_latent )
        t2, = Transition.sufficientStats( ( x[ :-1 ], x[ 1: ] ), constParams=( D_latent, D_latent ) )
        t3, = Transition.sufficientStats( ( x, ys ), constParams=( D_latent, D_obs ) )
        return t1, t2, t3

    @classmethod
    def log_partition( cls, x=None, params=None, nat_params=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( nat_params is None )
        if( split ):
            return ( 0, 0, 0 )
        return 0

    @classmethod
    def log_partitionGradient( cls, params=None, nat_params=None , split=False):
        return ( 0, 0, 0 ) if split == False else ( ( 0, 0, 0 ), ( 0, ) )

    def _testLogPartitionGradient( self ):
        # Don't need to test this
        pass

    ######################################################################

    def genStates( self ):
        return np.empty( self.T, dtype=int )

    def genEmissions( self, measurements=1 ):
        return np.empty( ( measurements, self.T ), dtype=int )

    def genStats( self ):
        return ( ( np.zeros( self.D_latent ) , ),
                 ( np.zeros( ( self.D_latent, self.D_latent ) ) , ),
                 ( np.zeros( ( self.D_latent, self.D_obs ) ) , ) )

    ######################################################################

    def sampleSingleEmission( self, x, measurements=1 ):
        p = self.emissionDist[ x ]
        return Categorical.sample( params=( p, ), size=measurements )

    def sampleEmissions( self, x ):
        # Sample from P( y | x, ϴ )
        assert x.ndim == 1
        return np.apply_along_axis( self.sampleSingleEmission, -1, x.reshape( ( -1, 1 ) ) ).ravel()[ None ]

    def emissionLikelihood( self, x, ys ):
        # Compute P( y | x, ϴ )
        return self._L[ x, ys ].sum()

    ######################################################################

    def sampleStep( self, p ):
        return int( Categorical.sample( nat_params=( p, ) ) )

    def likelihoodStep( self, x, p ):
        return Categorical.log_likelihood( np.array( [ x ] ), nat_params=( p, ) )

    ######################################################################

    def forwardArgs( self, t, beta, prevX ):
        # P( x_t+1 | x_t, y_t+1:T ) = P( y_t+1 | x_t+1 ) * P( x_t+1 | x_t ) * P( y_t+2:T | x_t+1 ) / P( y_t+1:T | x_t )
        #                           ∝ P( y_t+1 | x_t+1 ) * P( x_t+1 | x_t ) * P( y_t+2:T | x_t+1 )
        if( beta is None ):
            return ( self.pi[ prevX ], ) if t > 0 else ( self.pi0, )

        if( t == 0 ):
            unNormalized = self.pi0 + self.L[ t ] + beta
        else:
            unNormalized = self.pi[ prevX ] + self.L[ t ] + beta

        return ( unNormalized - np.logaddexp.reduce( unNormalized ), )

    def backwardArgs( self, t, alpha, prevX ):
        # P( x_t | x_t+1, y_1:t ) = P( x_t+1 | x_t ) * P( x_t, y_1:t ) / sum_{ z_t }[ P( x_t+1 | z_t ) * P( z_t, y_1:t ) ]
        #                         ∝ P( x_t+1 | x_t ) * P( x_t, y_1:t )
        unNormalized = alpha + self.pi[ prevX ] if t < self.T - 1 else alpha
        normalized = unNormalized - np.logaddexp.reduce( unNormalized )
        return ( normalized, )

    ######################################################################

    def isample( self, ys=None, knownLatentStates=None, measurements=1, T=None, forwardFilter=True, size=1, returnStats=False ):
        if( ys is not None ):
            preprocessKwargs = {}
            filterKwargs = { 'knownLatentStates': knownLatentStates }
            ans = self.conditionedSample( ys=ys, forwardFilter=forwardFilter, preprocessKwargs=preprocessKwargs, filterKwargs=filterKwargs, returnStats=returnStats )
        else:
            ans =  self.fullSample( measurements=measurements, T=T, size=size, returnStats=returnStats )

        if( returnStats ):
            # Re-order the stats
            t1, t2, t3 = ans[ 0 ]
            N, M, T, TM = ans[ 1 ]
            ans = [ t1, t2, t3 ]
        return ans

    def ilog_likelihood( self, x, forwardFilter=True, conditionOnY=False, expFam=False,  preprocessKwargs={}, filterKwargs={}, knownLatentStates=None, seperateLikelihoods=False ):
        filterKwargs.update( { 'knownLatentStates': knownLatentStates } )
        return super( HMMState, self ).ilog_likelihood( x=x, forwardFilter=forwardFilter, conditionOnY=conditionOnY, expFam=expFam, preprocessKwargs=preprocessKwargs, filterKwargs=filterKwargs, seperateLikelihoods=seperateLikelihoods )

    ######################################################################

    def conditionedExpectedSufficientStats( self, ys, alphas, betas, forMStep=False ):

        totalMarginal = 0

        smoothedSumsForY = np.zeros( ( self.D_latent, self.D_obs ) )
        smoothedSumsForY[ : ] = np.NINF

        jointSum = np.zeros( ( self.D_latent, self.D_latent ) )
        jointSum[ : ] = np.NINF

        smoothSum = np.zeros( self.D_latent )
        smoothSum[ : ] = np.NINF

        smoothSumNotLast = np.zeros( self.D_latent )
        smoothSumNotLast[ : ] = np.NINF

        smoothSum0 = np.zeros( self.D_latent )
        smoothSum0[ : ] = np.NINF

        totalObs = 0

        for i, ( _ys, _alphas, _betas ) in enumerate( zip( ys, alphas, betas ) ):

            # Compute P( Y )
            marginal = np.logaddexp.reduce( _alphas[ 0 ] + _betas[ 0 ] )
            totalMarginal += marginal

            # P( X | Y )
            smoothed = _alphas + _betas - marginal

            # sum_{ t in T }E[ P( x_t | Y ) ]
            smoothSum0 = np.logaddexp( smoothSum0, smoothed[ 0 ] )
            smoothSumNotLast = np.logaddexp( smoothSumNotLast, np.logaddexp.reduce( smoothed[ :-1 ], axis=0 ) )
            smoothSum = np.logaddexp( smoothSum, np.logaddexp.reduce( smoothed, axis=0 ) )

            # sum_{ t=1:T-1 }E[ P( x_t, x_t+1 | Y ) ]
            T = smoothed.shape[ 0 ]
            joint = np.logaddexp.reduce( [ self.childParentJoint( t, _alphas, _betas, ys=_ys ) - marginal for t in range( T - 1 ) ] )
            jointSum = np.logaddexp( jointSum, joint )

            # sum_{ t=1:T }E[ P( x_t, y_t | Y ) ]
            for _ys_ in _ys:
                for i in range( self.D_latent ):
                    for j in range( self.D_obs ):
                        relevantVals = smoothed[ :, i ][ _ys_ == j ]
                        if( relevantVals.size > 0 ):
                            smoothedSumsForY[ i, j ] = np.logaddexp( smoothedSumsForY[ i, j ], np.logaddexp.reduce( relevantVals, axis=0 ) )

            totalObs += len( _ys )

        if( forMStep ):
            return smoothSum0, jointSum, smoothSumNotLast, smoothedSumsForY, smoothSum, totalObs

        return np.exp( smoothSum0 ), np.exp( jointSum ), np.exp( smoothedSumsForY )

    ######################################################################

    @classmethod
    def generate( cls, measurements=4, T=5, D_latent=3, D_obs=2, size=1 ):
        initialDist = Dirichlet.generate( D=D_latent )
        transDist = TransitionDirichletPrior.generate( D_in=D_latent, D_out=D_latent )
        emissionDist = TransitionDirichletPrior.generate( D_in=D_latent, D_out=D_obs )

        dummy = cls( initialDist, transDist, emissionDist )
        return dummy.isample( measurements=measurements, T=T, size=size )

    ######################################################################

    def MStep( self, ys, alphas, betas ):

        N = self.dataN( ys, conditionOnY=True, checkY=True )

        smoothSum0, jointSum, smoothSumNotLast, smoothedSumsForY, smoothSum, totalObs = self.conditionedExpectedSufficientStats( ys, alphas, betas, forMStep=True )

        pi0 = smoothSum0 - np.log( N )
        pi = jointSum - smoothSumNotLast[ :, None ]
        L =  smoothedSumsForY - smoothSum[ :, None ] - np.log( totalObs ) + np.log( len( ys ) )

        return pi0, pi, L
