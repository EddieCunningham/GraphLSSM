from abc import ABC, abstractmethod
import numpy as np

class MessagePasser( ABC ):
    # Base message passing class for forward backward
    # and kalman filter type algorithms

    @property
    def T( self ):
        return self._T

    @T.setter
    def T( self, val ):
        self._T = val

    @property
    @abstractmethod
    def stateSize( self ):
        pass

    @abstractmethod
    def preprocessData( self, ys ):
        pass

    @abstractmethod
    def parameterCheck( self, *args ):
        pass

    @abstractmethod
    def genFilterProbs( self ):
        pass

    @abstractmethod
    def genWorkspace( self ):
        pass

    @abstractmethod
    def updateParams( self, ys ):
        pass

    @abstractmethod
    def transitionProb( self, t, t1 ):
        pass

    @abstractmethod
    def emissionProb( self, t ):
        pass

    @abstractmethod
    def integrate( self, integrand, outMem ):
        pass

    @abstractmethod
    def forwardBaseCase( self ):
        pass

    @abstractmethod
    def backwardBaseCase( self ):
        pass

    ######################################################################

    def forwardStep( self, t, alpha, workspace=None, out=None ):

        # Generate P( x_t | x_t-1 ) as a function of [ x_t, x_t-1 ]
        transition = self.transitionProb( t - 1, t, forward=True )

        # Compute P( x_t | x_t-1 ) * P( y_1:t-1, x_t-1 ) = P( y_1:t-1, x_t, x_t-1 )
        self.multiplyTerms( ( transition, alpha ), out=workspace )

        # Integrate out x_t-1 = P( y_1:t-1, x_t )
        self.integrate( workspace, forward=True, out=out )

        # Generate P( y_t | x_t ) as a function of x_t
        emission = self.emissionProb( t, forward=True )

        # Compute P( y_t | x_t ) * P( y_1:t-1, x_t ) = P( y_1:t, x_t )
        self.multiplyTerms( ( emission, out ), out=out )

    ######################################################################

    def backwardStep( self, t, beta, workspace=None, out=None ):

        # Generate P( x_t+1 | x_t ) as a function of [ x_t+1, x_t ]
        transition = self.transitionProb( t, t + 1, forward=False )

        # Generate P( y_t+1 | x_t+1 ) as a function of [ x_t+1, x_t ]
        emission = self.emissionProb( t + 1, forward=False )

        # Compute P( y_t+1 | x_t+1 ) * P( x_t+1 | x_t ) * P( y_t+2:T | x_t+1 ) = P( y_t+1:T, x_t+1 | x_t )
        self.multiplyTerms( ( emission, transition, beta ), out=workspace )

        # Integrate out x_t+1 = P( y_t+1:T | x_t )
        self.integrate( workspace, forward=False, out=out )

    ######################################################################

    def forwardFilter( self ):

        workspace = self.genWorkspace()

        alphas = self.genFilterProbs()
        alphas[ 0 ] = self.forwardBaseCase()

        for t in range( 1, self.T ):
            self.forwardStep( t, alphas[ t - 1 ], workspace=workspace, out=alphas[ t ] )

        return alphas

    ######################################################################

    def backwardFilter( self ):

        workspace = self.genWorkspace()

        betas = self.genFilterProbs()
        betas[ -1 ] = self.backwardBaseCase()

        for t in reversed( range( self.T - 1 ) ):
            self.backwardStep( t, betas[ t + 1 ], workspace=workspace, out=betas[ t ] )

        return betas

    ######################################################################

    def smooth( self ):
        alphas = self.forwardFilter()
        betas = self.backwardFilter()
        return alphas + betas
