import numpy as np
from GenModels.GM.Distributions.Base import ExponentialFam
from GenModels.GM.Distributions.Normal import Normal
from GenModels.GM.Utility import *

__all__ = [ 'Regression' ]

def definePrior():
    # Doing this to get around circular dependency
    from GenModels.GM.Distributions.MatrixNormalInverseWishart import MatrixNormalInverseWishart
    Regression.priorClass = MatrixNormalInverseWishart

class Regression( ExponentialFam ):

    priorClass = None

    def __init__( self, A=None, sigma=None, prior=None, hypers=None ):
        definePrior()
        super( Regression, self ).__init__( A, sigma, prior=prior, hypers=hypers )

    ##########################################################################

    @property
    def A( self ):
        return self._params[ 0 ]

    @property
    def sigma( self ):
        return self._params[ 1 ]

    ##########################################################################

    @classmethod
    def paramShapes( cls, D_out=None, D_in=None ):
        assert D_out is not None and D_in is not None
        return [ ( D_out, D_in ), ( D_out, D_out ) ]

    @classmethod
    def inferDims( cls, params=None ):
        assert params is not None
        A, sigma = params
        return { 'D_in': A.shape[ 1 ], 'D_out': A.shape[ 0 ] }

    @classmethod
    def outputShapes( cls, D_in=None, D_out=None ):
        assert D_in is not None and D_out is not None
        return [ ( D_in, ), ( D_out, ) ]

    ##########################################################################

    @classmethod
    def standardToNat( cls, A, sigma ):

        sigInv = np.linalg.inv( sigma )

        n1 = -0.5 * sigInv
        n2 = -0.5 * A.T @ sigInv @ A
        n3 = A.T @ sigInv

        return n1, n2, n3

    @classmethod
    def natToStandard( cls, n1, n2, n3 ):
        sigma = -0.5 * np.linalg.inv( n1 )
        A = sigma @ n3.T
        return A, sigma

    ##########################################################################

    @property
    def constParams( self ):
        return None

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, constParams=None, forPost=False ):
        # Compute T( x )

        if( isinstance( x, list ) ):
            x, y = zip( *x )
            x = np.vstack( x )
            y = np.vstack( y )
        else:
            x, y = x

        if( x.ndim == 1 ):
            x = x.reshape( ( 1, -1 ) )
        if( y.ndim == 1 ):
            y = y.reshape( ( 1, -1 ) )

        t1 = y.T.dot( y )
        t2 = x.T.dot( x )
        t3 = x.T.dot( y )

        if( forPost ):
            t4 = x.shape[ 0 ]
            t5 = x.shape[ 0 ]
            return t1, t2, t3, t4, t5
        return t1, t2, t3

    @classmethod
    @checkExpFamArgs
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        A, sigma = params if params is not None else cls.natToStandard( *natParams )

        p = sigma.shape[ 0 ]

        A1 = 0.5 * np.linalg.slogdet( sigma )[ 1 ]
        A2 = p / 2 * np.log( 2 * np.pi )

        if( split ):
            return A1, A2
        return A1 + A2

    ##########################################################################

    @classmethod
    @fullSampleSupport
    @checkExpFamArgs( allowNone=True )
    def sample( cls, x=None, params=None, natParams=None ):
        # Sample from P( x | Ѳ; α )

        A, sigma = params if params is not None else cls.natToStandard( *natParams )
        D_out, D_in = A.shape
        if( x is None ):
            x = Normal.sample( params=( np.zeros( D_in ), np.eye( D_in ) ) )
        return x, Normal.sample( params=( A.dot( x ), sigma ) )

    ##########################################################################

    @classmethod
    @fullLikelihoodSupport
    @checkExpFamArgs
    def log_likelihood( cls, x, conditionOnX=True, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        A, sigma = params if params is not None else cls.natToStandard( *natParams )
        D_out, D_in = A.shape

        x, y = x
        log_y = Normal.log_likelihood( y, params=( A.dot( x ), sigma ) )

        if( conditionOnX == True ):
            return log_y

        log_x = Normal.log_likelihood( x, params=( np.zeros( D_in ), np.eye( D_in ) ) )
        return log_x + log_y
