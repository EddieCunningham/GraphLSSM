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

    @classmethod
    def sample( cls, ys=None, params=None, natParams=None, T=None, forwardFilter=True, size=1 ):
        assert ( params is None ) ^ ( natParams is None )
        params = params if params is not None else cls.natToStandard( *natParams )

        dummy = cls( *params )
        return dummy.isample( ys=ys, T=T, forwardFilter=forwardFilter, size=size )

    def isample( self, ys=None, T=None, forwardFilter=True, size=1, **kwargs ):

        if( ys is not None ):
            self.preprocessData( ys=ys, **kwargs )
        else:
            assert T is not None
            self.T = T

        x = self.genStates()

        def workFunc( t, *args ):
            nonlocal x
            x[ t ] = self.sampleStep( *args )
            return x[ t ]

        if( ys is None ):
            # This is if we want to sample from P( x, y | ϴ )
            self.noFilterForwardRecurse( workFunc )
            ys = self.sampleEmissions( x )
        elif( forwardFilter ):
            # Otherwise sample from P( x | y, ϴ )
            self.forwardFilterBackwardRecurse( workFunc )
        else:
            self.backwardFilterForwardRecurse( workFunc )

        return x, ys

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

        ( x, ys ) = x
        assert ys is not None

        self.preprocessData( ys=ys, **kwargs )

        ans = 0.0

        def workFunc( t, *args ):
            nonlocal ans, x
            term = self.likelihoodStep( x[ t ], *args )
            ans += term
            if( t == 0 ):
                print( '\nP( x_0 ) = %4.2f'%( term ) )
            else:
                print( 'P( x_%d | x_%d ) = %4.2f'%( t, t - 1, term ) )
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
