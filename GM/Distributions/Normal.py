import numpy as np
from Base import Exponential
from scipy.stats import multivariate_normal
from NormalInverseWishart import NormalInverseWishart

class Normal( Exponential ):

    priorClass = NormalInverseWishart

    def __init__( self, mu=None, sigma=None, prior=None, hypers=None ):
        super( Normal, self ).__init__( mu, sigma, prior=prior, hypers=hypers )

    ##########################################################################

    @classmethod
    def standardToNat( cls, mu, sigma ):
        n1 = np.linalg.inv( sigma )
        n2 = n1.dot( mu )
        n1 *= -0.5
        return n1, n2

    @classmethod
    def natToStandard( cls, n1, n2 ):
        sigma = -0.5 * np.linalg.inv( n1 )
        mu = sigma.dot( n2 )
        return mu, sigma

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, forPost=False ):
        # Compute T( x )
        if( x.ndim == 1 ):
            x = x.reshape( ( 1, -1 ) )
        t1 = x.T.dot( x )
        t2 = x.sum( axis=0 )
        if( forPost ):
            # This for when we add to the NIW natural params
            t3 = x.shape[ 0 ]
            t4 = x.shape[ 0 ]
            t5 = x.shape[ 0 ]
            return t1, t2, t3, t4, t5
        return t1, t2

    @classmethod
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( natParams is None )

        mu, sigma = params if params is not None else cls.natToStandard( *natParams )

        k = sigma.shape[ 0 ]
        sigInv = np.linalg.inv( sigma )
        A1 = 0.5 * mu.dot( sigInv ).dot( mu )
        A2 = 0.5 * np.linalg.slogdet( sigma )[ 1 ]

        log_h = k / 2 * np.log( 2 * np.pi )

        if( split ):
            return ( A1, A2, log_h )
        return A1 + A2 + log_h

    ##########################################################################

    @classmethod
    def sample( cls, params=None, natParams=None, size=1 ):
        # Sample from P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )
        mu, sigma = params if params is not None else cls.natToStandard( *natParams )
        return multivariate_normal.rvs( mean=mu, cov=sigma, size=size )

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )
        mu, sigma = params if params is not None else cls.natToStandard( *natParams )
        return multivariate_normal.logpdf( x, mean=mu, cov=sigma )

    ##########################################################################

    @classmethod
    def marginalizeX1( cls, J11, J12, J22, h1, h2, log_Z ):
        K = h1.shape[ 0 ]
        J11Inv = np.linalg.inv( J11 )
        J = J22 - J12.T @ J11Inv @ J12
        h = h2 - ( J12.T @ J11Inv ).dot( h1 )
        log_Z = log_Z - \
                0.5 * h1.dot( J11Inv ).dot( h1 ) + \
                0.5 * np.linalg.slogdet( J11 )[ 1 ] - \
                K / 2 * np.log( 2 * np.pi )
        return J, h, log_Z

    @classmethod
    def marginalizeX2( cls, J11, J12, J22, h1, h2, log_Z ):
        return cls.marginalizeX1( J22, J12.T, J11, h2, h1, log_Z )

