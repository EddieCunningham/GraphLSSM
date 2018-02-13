import numpy as np
from Base import Exponential
from scipy.special import multigammaln
from scipy.stats import matrix_normal
from InverseWishart import InverseWishart
import Regression
import Normal

class MatrixNormalInverseWishart( Exponential ):
    # This class is written with the intention of making it a prior for
    # a normal distribution with an unknown mean and covariance

    def __init__( self, M, V, psi, nu, Q=0, prior=None, hypers=None ):
        super( MatrixNormalInverseWishart, self ).__init__( M, V, psi, nu, Q, prior=prior, hypers=hypers )

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

    @classmethod
    def sufficientStats( cls, x, forPost=False ):
        # Compute T( x )
        t1, t2, t3 = Regression.Regression.standardToNat( *x )
        t4, t5 = Regression.Regression.log_partition( params=x, split=True )
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
    def sample( cls, params=None, natParams=None ):
        # Sample from P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )
        M, V, psi, nu, _ = params if params is not None else cls.natToStandard( *natParams )

        sigma = InverseWishart.sample( params=( psi, nu ) )
        A = matrix_normal.rvs( mean=M, rowcov=sigma, colcov=V, size=1 )
        return A, sigma

    @classmethod
    def basicSample( cls, D_out, D_in ):
        # Sample from P( x | Ѳ; α )

        psi = np.eye( D_out )
        nu = D_out
        M = np.ones( ( D_out, D_in ) )
        V = np.eye( D_in )

        sigma = InverseWishart.sample( params=( psi, nu ) )
        A = matrix_normal.rvs( mean=M, rowcov=sigma, colcov=V, size=1 )
        return A, sigma

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )
        M, V, psi, nu, _ = params if params is not None else cls.natToStandard( *natParams )
        A, sigma = x
        return InverseWishart.log_likelihood( sigma, params=( psi, nu ) ) + \
               matrix_normal.logpdf( X=A, mean=M, rowcov=sigma, colcov=V )