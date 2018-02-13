from MessagePassingBase import MessagePasser
import numpy as np
from functools import reduce

class CategoricalForwardBackward( MessagePasser ):
    # Categorical emissions.  Everything is done in log space

    def __init__( self, T, K ):
        self.K = K
        super( CategoricalForwardBackward, self ).__init__( T )

    def genFilterProbs( self ):
        return np.empty( ( self.T, self.K ) )

    def genWorkspace( self ):
        return np.empty( ( self.K, self.K ) )

    ######################################################################

    def updateParams( self, y, initialDist, transDist, emissionDist ):
        assert initialDist.shape == ( self.K, )
        assert transDist.shape == ( self.K, self.K )
        assert emissionDist.shape[ 0 ] == self.K
        assert np.isclose( 1.0, initialDist.sum() )
        assert np.allclose( np.ones( self.K ), transDist.sum( axis=1 ) )
        assert np.allclose( np.ones( self.K ), emissionDist.sum( axis=1 ) )
        self.pi0 = np.log( initialDist )
        self.pi  = np.log( transDist )
        self.L   = np.log( emissionDist )
        super( CategoricalForwardBackward, self ).updateParams( y )

    ######################################################################

    def transitionProb( self, t, t1, forward=False ):
        return self.pi

    ######################################################################

    def emissionProb( self, t, forward=False ):
        return self.L[ :, self.y[ t ] ]

    ######################################################################

    def multiplyTerms( self, terms, out=None ):
        # Numpy should broadcast the terms correctly so that
        # a matrix is returned if we're before the integration,
        # otherwise a vector
        if( out is not None ):
            reduce( lambda x, y: np.add( x, y, out=out ), terms )
        else:
            return reduce( lambda x, y: np.add( x, y ), terms )

    ######################################################################

    def integrate( self, integrand, forward=True, out=None ):
        # Add the values in log space
        if( out is not None ):
            np.logaddexp.reduce( integrand, axis=0, out=out )
        else:
            return np.logaddexp.reduce( integrand, axis=0 )

    ######################################################################

    def forwardBaseCase( self ):
        return self.pi0 + self.L[ :, self.y[ 0 ] ]

    def backwardBaseCase( self ):
        return np.zeros( self.K )


