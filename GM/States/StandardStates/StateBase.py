import numpy as np
from GenModels.GM.Distributions import ExponentialFam
from abc import ABC, abstractmethod

class StateBase( ExponentialFam ):

    # This is a distribution over P( x, y | ϴ ).
    # Will still be able to do inference over P( x | y, ϴ )

    ######################################################################

    @property
    @abstractmethod
    def T( self ):
        pass

    @abstractmethod
    def forwardFilter( self ):
        pass

    @abstractmethod
    def backwardFilter( self ):
        pass

    @abstractmethod
    def preprocessData( self, ys ):
        pass

    @abstractmethod
    def parameterCheck( self, *args ):
        pass

    @abstractmethod
    def updateParams( self, *args ):
        pass

    @abstractmethod
    def genStates( self ):
        pass

    @abstractmethod
    def sampleStep( self, *args ):
        pass

    @abstractmethod
    def likelihoodStep( self, *args ):
        pass

    @abstractmethod
    def forwardArgs( self, t, beta, prevX ):
        pass

    @abstractmethod
    def backwardArgs( self, t, alpha, prevX ):
        pass

    @abstractmethod
    def sequenceLength( cls, x ):
        pass

    ######################################################################

    def noFilterForwardRecurse( self, workFunc ):
        lastVal = None
        for t in range( self.T ):
            args = self.forwardArgs( t, None, lastVal )
            lastVal = workFunc( t, *args )

    def forwardFilterBackwardRecurse( self, workFunc ):
        # P( x_1:T | y_1:T ) = prod_{ x_t=T:1 }[ P( x_t | x_t+1, y_1:t ) ] * P( x_T | y_1:T )
        alphas = self.forwardFilter()

        lastVal = None
        for t in reversed( range( self.T ) ):
            args = self.backwardArgs( t, alphas[ t ], lastVal )
            lastVal = workFunc( t, *args )

    def backwardFilterForwardRecurse( self, workFunc ):
        # P( x_1:T | y_1:T ) = prod_{ x_t=1:T }[ P( x_t+1 | x_t, y_t+1:T ) ] * P( x_1 | y_1:T )
        betas = self.backwardFilter()

        lastVal = None
        for t in range( self.T ):
            args = self.forwardArgs( t, betas[ t ], lastVal )
            lastVal = workFunc( t, *args )

    ######################################################################

    @abstractmethod
    def sampleEmissions( self, x ):
        # Sample from P( y | x, ϴ )
        pass

    @abstractmethod
    def emissionLikelihood( self, x, ys ):
        # Compute P( y | x, ϴ )
        pass

    ######################################################################

    @abstractmethod
    def conditionedExpectedSufficientStats( self, alphas, betas ):
        pass

    @classmethod
    def expectedSufficientStats( cls, ys=None, alphas=None, betas=None, params=None, natParams=None, **kwargs ):
        assert ( params is None ) ^ ( natParams is None )
        params = params if params is not None else cls.natToStandard( *natParams )
        dummy = cls( *params )
        return dummy.iexpectedSufficientStats( ys=ys, alphas=alphas, betas=betas, **kwargs )

    def iexpectedSufficientStats( self, ys=None, alphas=None, betas=None, **kwargs ):
        if( ys is None ):
            return super( StateBase, self ).iexpectedSufficientStats()

        self.preprocessData( ys=ys, **kwargs )

        if( alphas is None ):
            alphas = self.forwardFilter()

        if( betas is None ):
            betas = self.backwardFilter()

        return self.conditionedExpectedSufficientStats( alphas, betas )

    ######################################################################

    @classmethod
    def sample( cls, ys=None, params=None, natParams=None, measurements=1, T=None, forwardFilter=True, size=1 ):
        assert ( params is None ) ^ ( natParams is None )
        params = params if params is not None else cls.natToStandard( *natParams )

        dummy = cls( *params )
        return dummy.isample( ys=ys, measurements=measurements, T=T, forwardFilter=forwardFilter, size=size )

    ######################################################################

    def conditionedSample( self, ys=None, forwardFilter=True, **kwargs ):
        # Sample x given y

        size = self.dataN( ys, conditionOnY=True, checkY=True )

        if( size > 1 ):
            it = iter( ys )
        else:
            it = iter( [ ys ] )

        ans = []
        for y in it:

            self.preprocessData( ys=y, **kwargs )

            x = self.genStates()

            def workFunc( t, *args ):
                nonlocal x
                x[ t ] = self.sampleStep( *args )
                return x[ t ]

            if( forwardFilter ):
                self.forwardFilterBackwardRecurse( workFunc )
            else:
                self.backwardFilterForwardRecurse( workFunc )

            ans.append( ( x, y ) )

        ans = tuple( list( zip( *ans ) ) )

        self.checkShape( ans )
        return ans

    ######################################################################

    def fullSample( self, measurements=1, T=None, size=1 ):
        # Sample x and y

        assert T is not None
        self.T = T

        ans = []

        for _ in range( size ):
            x = self.genStates()

            def workFunc( t, *args ):
                nonlocal x
                x[ t ] = self.sampleStep( *args )
                return x[ t ]

            # This is if we want to sample from P( x, y | ϴ )
            self.noFilterForwardRecurse( workFunc )

            # We can have multiple measurements from the same latent state
            ys = np.array( [ self.sampleEmissions( x ) for _ in range( measurements ) ] )
            ys = np.swapaxes( ys, 0, 1 )

            ans.append( ( x, ys[ 0 ] ) )

        ans = tuple( list( zip( *ans ) ) )

        self.checkShape( ans )
        return ans

    ######################################################################

    def isample( self, ys=None, measurements=1, T=None, forwardFilter=True, size=1, **kwargs ):

        if( ys is not None ):
            return self.conditionedSample( ys=ys, forwardFilter=forwardFilter, **kwargs )
        return self.fullSample( measurements=measurements, T=T, size=size )

    ######################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, natParams=None, forwardFilter=True, conditionOnY=False ):
        assert ( params is None ) ^ ( natParams is None )
        params = params if params is not None else cls.natToStandard( *natParams )

        dummy = cls( *params )
        return dummy.ilog_likelihood( x, forwardFilter=forwardFilter, conditionOnY=conditionOnY )

    def ilog_likelihood( self, x, forwardFilter=True, conditionOnY=False, expFam=False, **kwargs ):

        if( expFam ):
            return self.log_likelihoodExpFam( x, constParams=self.constParams, natParams=self.natParams )

        size = self.dataN( x )

        x, ys = x

        if( size > 1 ):
            it = zip( x, ys )
        else:
            it = iter( [ x, ys ] )

        ans = np.zeros( size )

        for i, ( x, ys ) in enumerate( it ):

            self.preprocessData( ys=ys, **kwargs )

            def workFunc( t, *args ):
                nonlocal ans, x
                term = self.likelihoodStep( x[ t ], *args )
                ans[ i ] += term
                return x[ t ]

            if( conditionOnY == False ):
                # This is if we want to compute P( x, y | ϴ )
                self.noFilterForwardRecurse( workFunc )
                ans += self.emissionLikelihood( x, ys )
            else:
                if( forwardFilter ):
                    # Otherwise compute P( x | y, ϴ )
                    assert conditionOnY == True
                    self.forwardFilterBackwardRecurse( workFunc )
                else:
                    assert conditionOnY == True
                    self.backwardFilterForwardRecurse( workFunc )

        return ans
