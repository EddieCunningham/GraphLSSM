import numpy as np
from GenModels.GM.Distributions.Base import ExponentialFam
from GenModels.GM.Distributions.Normal import Normal
from GenModels.GM.Utility import invPsd

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
    def dataN( cls, x ):
        xs, ys = x
        assert xs.ndim == 2
        return xs.shape[ 0 ]

    ##########################################################################

    @classmethod
    def standardToNat( cls, A, sigma ):
        sigInv = invPsd( sigma )

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
    def sufficientStats( cls, x, constParams=None ):
        # Compute T( x )

        cls.dataN( x )

        x, y = x

        if( x.ndim == 1 ):
            x = x.reshape( ( 1, -1 ) )
        if( y.ndim == 1 ):
            y = y.reshape( ( 1, -1 ) )

        t1 = y.T.dot( y )
        t2 = x.T.dot( x )
        t3 = x.T.dot( y )

        return t1, t2, t3

    @classmethod
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( natParams is None )
        A, sigma = params if params is not None else cls.natToStandard( *natParams )
        if( natParams is not None ):
            n1, n2, n3 = cls.standardToNat( A, sigma )
            assert np.allclose( n1, natParams[ 0 ] )
            assert np.allclose( n2, natParams[ 1 ] )
            assert np.allclose( n3, natParams[ 2 ] )

        n = sigma.shape[ 0 ]

        A1 = 0.5 * np.linalg.slogdet( sigma )[ 1 ]
        A2 = n / 2 * np.log( 2 * np.pi )

        print( 'In log partition' )
        print( 'A', A )
        print( 'sigma', sigma )
        print( 'A1', A1 )
        print( 'A2', A2 )

        if( split ):
            return A1, A2
        return A1 + A2

    ##########################################################################

    @classmethod
    def sample( cls, x=None, params=None, natParams=None, size=1 ):
        # Sample from P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )
        A, sigma = params if params is not None else cls.natToStandard( *natParams )
        D = A.shape[ 1 ]
        if( x is None ):
            x = np.array( [ Normal.sample( params=( np.zeros( D ), np.eye( D ) ), size=1 ) for _ in range( size ) ] )
            y = np.array( [ Normal.sample( params=( A.dot( _x ), sigma ), size=1 ) for _x in x ] )
            return ( x, y )
        return Normal.sample( params=( A.dot( x ), sigma ), size=size )

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )
        A, sigma = params if params is not None else cls.natToStandard( *natParams )

        x, y = x
        assert x.shape[ 0 ] == y.shape[ 0 ]
        if( x.ndim != 1 ):
            return sum( [ Normal.log_likelihood( _y, params=( A.dot( _x ), sigma ) ) for _x, _y in zip( x, y ) ] )
        return Normal.log_likelihood( y, params=( A.dot( x ), sigma ) )
