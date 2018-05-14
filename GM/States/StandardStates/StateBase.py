import numpy as np
from GenModels.GM.Distributions import Distribution

class StateBase( Distribution ):

    # This is a class for the distribution over the latent state.
    # Inputs and data will be kept as constParams

    ######################################################################

    @property
    @abstractmethod
    def T(self):
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

    @classmethod
    def sample( cls, params, ys=None, T=None, forwardFilter=True ):
        dummy = StateBase()
        dummy.updateParams( params )
        return dummy.isample( ys=ys, T=T, forwardFilter=forwardFilter )

    def isample( self, ys=None, T=None, forwardFilter=True ):

        if( ys is not None ):
            self.preprocessData( ys )
        else:
            assert T is not None
            self.T = T

        x = self.genStates()

        def workFunc( t, *args ):
            nonlocal x
            x[ t ] = self.sampleStep( *args )
            return x[ t ]

        if( ys is None ):
            self.noFilterForwardRecurse( workFunc )
        elif( forwardFilter ):
            self.forwardFilterBackwardRecurse( workFunc )
        else:
            self.backwardFilterForwardRecurse( workFunc )

        return x

    ######################################################################

    @classmethod
    def log_likelihood( self, x, params, ys=None, T=None, forwardFilter=True ):
        dummy = StateBase()
        dummy.updateParams( params )
        return dummy.ilog_likelihood( x, ys=ys, T=T, forwardFilter=forwardFilter )

    def ilog_likelihood( self, x, ys=None, forwardFilter=True ):

        if( ys is not None ):
            self.preprocessData( ys )
        else:
            assert T is not None
            self.T = T

        ans = 0.0

        def workFunc( t, *args ):
            nonlocal ans, x
            ans += self.likelihoodStep( x[ t ], *args )
            return x[ t ]

        if( ys is None ):
            self.noFilterForwardRecurse( workFunc )
        elif( forwardFilter ):
            self.forwardFilterBackwardRecurse( workFunc )
        else:
            self.backwardFilterForwardRecurse( workFunc )

        return ans
