import numpy as np
from GenModels.GM.Distributions.Base import ExponentialFam, checkExpFamArgs, multiSampleLikelihood
from GenModels.GM.Distributions.Normal import Normal

__all__ = [ 'Regression' ]

def definePrior():
    # Doing this to get around circular dependency
    from GenModels.GM.Distributions.MatrixNormalInverseWishart import MatrixNormalInverseWishart
    Regression.priorClass =MatrixNormalInverseWishart

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
    def dataN( cls, x, ravel=False ):

        if( ravel == False ):
            xs, ys = x
            return ys.shape[ 0 ]
        else:
            return x.shape[ 0 ]

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
    @checkExpFamArgs
    def sample( cls, x=None, params=None, natParams=None, D_in=None, D_out=None, size=1, ravel=False ):
        # Sample from P( x | Ѳ; α )
        if( params is None and natParams is None ):
            assert D_in is not None and D_out is not None
            params = ( np.zeros( ( D_out, D_in ) ), np.eye( D_in ) )

        A, sigma = params if params is not None else cls.natToStandard( *natParams )
        D = A.shape[ 1 ]
        if( x is None ):
            x = np.array( [ Normal.sample( params=( np.zeros( D ), np.eye( D ) ), size=1 ) for _ in range( size ) ] )
            y = np.array( [ Normal.sample( params=( A.dot( _x ), sigma ), size=1 ) for _x in x ] )
            return ( x, y ) if ravel == False else np.hstack( ( x, y ) )
        return Normal.sample( params=( A.dot( x ), sigma ), size=size )

    ##########################################################################

    @classmethod
    @checkExpFamArgs
    @multiSampleLikelihood
    def log_likelihood( cls, x, params=None, natParams=None, ravel=False ):
        # Compute P( x | Ѳ; α )
        A, sigma = params if params is not None else cls.natToStandard( *natParams )

        if( ravel == False ):
            x, y = x
        else:
            D_in, D_out = A.shape
            x, y = np.split( x, [ D_in ] )
            assert x.shape[ 1 ] == D_in and y.shape[ 1 ] == D_out

        assert x.shape == y.shape

        return Normal.log_likelihood( y, ( A.dot( x ), sigma ) )
