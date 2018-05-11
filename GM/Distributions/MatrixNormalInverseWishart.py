import numpy as np
from GenModels.GM.Distributions.Base import ExponentialFam
from scipy.special import multigammaln
from GenModels.GM.Distributions.TensorNormal import TensorNormal
from GenModels.GM.Distributions.InverseWishart import InverseWishart
from GenModels.GM.Distributions.Regression import Regression

class MatrixNormalInverseWishart( ExponentialFam ):
    # This class is written with the intention of making it a prior for
    # a normal distribution with an unknown mean and covariance

    def __init__( self, M, V, psi, nu, Q=0, prior=None, hypers=None ):
        super( MatrixNormalInverseWishart, self ).__init__( M, V, psi, nu, Q, prior=prior, hypers=hypers )

    ##########################################################################

    @property
    def M( self ):
        return self._params[ 0 ]

    @property
    def V( self ):
        return self._params[ 1 ]

    @property
    def psi( self ):
        return self._params[ 2 ]

    @property
    def nu( self ):
        return self._params[ 3 ]

    ##########################################################################

    @classmethod
    def dataN( cls, x ):
        if( not isinstance( x[ 0 ], np.ndarray ) ):
            return len( x )
        assert len( x ) == 2
        assert isinstance( x[ 0 ], np.ndarray ) and isinstance( x[ 1 ], np.ndarray )
        return 1

    ##########################################################################

    @classmethod
    def standardToNat( cls, M, V, psi, nu, Q ):

        p = M.shape[ 0 ]
        VInv = np.linalg.inv( V )
        n1 = M @ VInv @ M.T + psi
        n2 = VInv
        n3 = VInv @ M.T
        n4 = nu + 2 * p + 1
        n5 = p + Q
        return n1, n2, n3, n4, n5

    @classmethod
    def natToStandard( cls, n1, n2, n3, n4, n5 ):

        p = n2.shape[ 0 ]

        V = np.linalg.inv( n2 )
        M = n3.T @ V
        psi = n1 - M @ n2 @ M.T
        nu = n4 - 1 - 2 * p

        # The roll of Q is to offset excess normal base measures!
        Q = n5 - p
        return M, V, psi, nu, Q

    ##########################################################################

    @property
    def constParams( self ):
        return None

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, constParams=None, forPost=False ):
        # Compute T( x )
        if( cls.dataN( x ) > 1 ):
            t = ( 0, 0, 0, 0, 0 )
            for _x in x:
                t = np.add( t, cls.sufficientStats( _x, forPost=forPost ) )
            return t

        t1, t2, t3 = Regression.standardToNat( *x )
        t4, t5 = Regression.log_partition( params=x, split=True )
        return t1, t2, t3, -t4, -t5

    @classmethod
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( natParams is None )

        M, V, psi, nu, Q = params if params is not None else cls.natToStandard( *natParams )

        p = psi.shape[ 0 ]

        A1, A2, A3 = InverseWishart.log_partition( params=( psi, nu ), split=True )
        A4 = p / 2 * np.linalg.slogdet( V )[ 1 ]
        A5 = -Q * ( p / 2 * np.log( 2 * np.pi ) )

        if( split ):
            return A1, A2, A3, A4, A5
        return A1 + A2 + A3 + A4 + A5

    ##########################################################################

    @classmethod
    def sample( cls, params=None, natParams=None, D_in=None, D_out=None, size=1 ):
        # Sample from P( x | Ѳ; α )
        if( params is None and natParams is None ):
            assert D_in is not None and D_out is not None
            params = ( np.zeros( D_out, D_in ), np.eye( D_in ), np.eye( D_out ), D_out )

        assert ( params is None ) ^ ( natParams is None )
        M, V, psi, nu, _ = params if params is not None else cls.natToStandard( *natParams )

        if( size > 1 ):
            return [ cls.sample( params=params, natParams=natParams, size=1 ) for _ in range( size ) ]

        sigma = InverseWishart.sample( params=( psi, nu ) )
        A = TensorNormal.sample( params=( M, ( sigma, V ) ), size=1 )[ 0 ]
        return A, sigma

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )
        M, V, psi, nu, _ = params if params is not None else cls.natToStandard( *natParams )
        if( cls.dataN( x ) > 1 ):
            return sum( [ cls.log_likelihood( _x, params=params, natParams=natParams ) for _x in x ] )
        A, sigma = x
        return InverseWishart.log_likelihood( sigma, params=( psi, nu ) ) + \
               TensorNormal.log_likelihood( A[ None ], params=( M, ( sigma, V ) ) )
