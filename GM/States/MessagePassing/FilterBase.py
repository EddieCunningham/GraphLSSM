from abc import ABC, abstractmethod
import autograd.numpy as np

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
    def D_latent( self ):
        pass

    @property
    @abstractmethod
    def D_obs( self ):
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
    def updateParams( self, ys ):
        pass

    @abstractmethod
    def updateNatParams( self, ys ):
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

    @classmethod
    @abstractmethod
    def log_marginalFromAlphaBeta( cls, alpha, beta ):
        pass

    ######################################################################

    def forwardStep( self, t, alpha ):

        # Generate P( x_t | x_t-1 ) as a function of [ x_t, x_t-1 ]
        transition = self.transitionProb( t - 1, t, forward=True )

        # Compute P( x_t | x_t-1 ) * P( y_1:t-1, x_t-1 ) = P( y_1:t-1, x_t, x_t-1 )
        workspace = self.multiplyTerms( ( transition, alpha ) )

        # Integrate out x_t-1 = P( y_1:t-1, x_t )
        out = self.integrate( workspace, forward=True )

        # Generate P( y_t | x_t ) as a function of x_t
        emission = self.emissionProb( t, forward=True )

        # Compute P( y_t | x_t ) * P( y_1:t-1, x_t ) = P( y_1:t, x_t )
        return self.multiplyTerms( ( emission, out ) )

    ######################################################################

    def backwardStep( self, t, beta ):

        # Generate P( x_t+1 | x_t ) as a function of [ x_t+1, x_t ]
        transition = self.transitionProb( t, t + 1, forward=False )

        # Generate P( y_t+1 | x_t+1 ) as a function of x_t+1
        emission = self.emissionProb( t + 1, forward=False )

        # Compute P( y_t+1 | x_t+1 ) * P( x_t+1 | x_t ) * P( y_t+2:T | x_t+1 ) = P( y_t+1:T, x_t+1 | x_t )
        workspace = self.multiplyTerms( ( emission, transition, beta ) )

        # Integrate out x_t+1 = P( y_t+1:T | x_t )
        return self.integrate( workspace, forward=False )

    ######################################################################

    def forwardFilter( self ):

        alphas = self.genFilterProbs()
        alphas[ 0 ] = self.forwardBaseCase()

        for t in range( 1, self.T ):
            alphas[ t ] = self.forwardStep( t, alphas[ t - 1 ] )

        return alphas

    ######################################################################

    def backwardFilter( self ):

        betas = self.genFilterProbs()
        betas[ -1 ] = self.backwardBaseCase()

        for t in reversed( range( self.T - 1 ) ):
            betas[ t ] = self.backwardStep( t, betas[ t + 1 ] )

        return betas

    ######################################################################

    def childParentJoint( self, t, alphas, betas ):
        # P( x_t+1, x_t, Y ) = P( y_t+1 | x_t+1 ) * P( y_t+2:T | x_t+1 ) * P( x_t+1 | x_t ) * P( x_t, y_1:t )

        emission = self.emissionProb( t + 1, forward=False )
        transition = self.transitionProb( t, t + 1, forward=False )
        alpha = alphas[ t ]
        beta = betas[ t + 1 ]

        return self.multiplyTerms( ( emission, transition, alpha, beta ) )

    ######################################################################

    def smooth( self ):
        alphas = self.forwardFilter()
        betas = self.backwardFilter()
        return alphas + betas
