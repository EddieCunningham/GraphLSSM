import numpy as np

class MessagePasser():
    # Base message passing class for forward backward
    # and kalman filter type algorithms

    def __init__( self, T ):
        self.T = T

    def genFilterProbs( self ):
        assert 0

    def genWorkspace( self ):
        assert 0

    def updateParams( self, ys ):
        self.ys = ys

    def transitionProb( self, t, t1 ):
        assert 0

    def emissionProb( self, t ):
        assert 0

    def combineTerms( self, *terms ):
        assert 0

    def integrate( self, integrand, outMem ):
        assert 0

    def forwardBaseCase( self ):
        assert 0

    def backwardBaseCase( self ):
        assert 0

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
