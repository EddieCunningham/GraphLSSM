import numpy as np
from GenModels.GM.Distributions.Base import ExponentialFam, checkExpFamArgs, multiSampleLikelihood
from scipy.stats import multivariate_normal
from scipy.linalg import cho_factor, cho_solve
from GenModels.GM.Utility import *

_HALF_LOG_2_PI = 0.5 * np.log( 2 * np.pi )

__all__ = [ 'Normal' ]

def definePrior():
    from GenModels.GM.Distributions.NormalInverseWishart import NormalInverseWishart
    Normal.priorClass = NormalInverseWishart

class Normal( ExponentialFam ):

    priorClass = None

    def __init__( self, mu=None, sigma=None, prior=None, hypers=None ):
        definePrior()
        super( Normal, self ).__init__( mu, sigma, prior=prior, hypers=hypers )

    ##########################################################################

    @property
    def mu( self ):
        return self._params[ 0 ]

    @property
    def sigma( self ):
        return self._params[ 1 ]

    ##########################################################################

    @property
    def constParams( self ):
        return None

    @classmethod
    def dataN( cls, x ):
        if( x.ndim == 2 ):
            return x.shape[ 0 ]
        return 1

    ##########################################################################

    @classmethod
    def standardToNat( cls, mu, sigma, returnPrecision=False ):
        n1 = invPsd( sigma )
        n2 = n1.dot( mu )
        if( returnPrecision == False ):
            n1 *= -0.5
        return n1, n2

    @classmethod
    def natToStandard( cls, n1, n2 ):
        sigma = -0.5 * np.linalg.inv( n1 )
        mu = sigma.dot( n2 )
        return mu, sigma

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, constParams=None, forPost=False ):
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
    @checkExpFamArgs
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )

        if( natParams is not None ):
            n1, n2 = natParams
            k = n1.shape[ 0 ]
            A1 = -0.25 * n2.dot( np.linalg.solve( n1, n2 ) )
            A2 = -0.5 * np.linalg.slogdet( -2 * n1 )[ 1 ]
        else:
            mu, sigma = params
            k = sigma.shape[ 0 ]
            A1 = 0.5 * mu.dot( np.linalg.solve( sigma, mu ) )
            A2 = 0.5 * np.linalg.slogdet( sigma )[ 1 ]

        log_h = k * _HALF_LOG_2_PI

        if( split ):
            return ( A1, A2, log_h )
        return A1 + A2 + log_h

    ##########################################################################

    @classmethod
    @checkExpFamArgs( allowNone=True )
    def sample( cls, params=None, natParams=None, D=None, size=1, ravel=False ):
        # Sample from P( x | Ѳ; α )
        if( params is None and natParams is None ):
            assert D is not None
            params = ( np.zeros( D ), np.eye( D ) )
        mu, sigma = params if params is not None else cls.natToStandard( *natParams )
        return multivariate_normal.rvs( mean=mu, cov=sigma, size=size )

    ##########################################################################

    @classmethod
    @checkExpFamArgs
    @multiSampleLikelihood
    def log_likelihood( cls, x, params=None, natParams=None, ravel=False ):
        # Compute P( x | Ѳ; α )
        mu, sigma = params if params is not None else cls.natToStandard( *natParams )
        return multivariate_normal.logpdf( x, mean=mu, cov=sigma )

    ##########################################################################

    @classmethod
    def marginalizeX1( cls, J11, J12, J22, h1, h2, log_Z ):
        K = h1.shape[ 0 ]

        J11Chol = cho_factor( J11, lower=True )
        J11Invh1 = cho_solve( J11Chol, h1 )

        J = J22 - J12.T @ cho_solve( J11Chol, J12 )
        h = h2 - J12.T.dot( J11Invh1 )

        log_Z = log_Z - \
                0.5 * h1.dot( J11Invh1 ) + \
                np.log( np.diag( J11Chol[ 0 ] ) ).sum() - \
                K * _HALF_LOG_2_PI
        return J, h, log_Z

    @classmethod
    def marginalizeX2( cls, J11, J12, J22, h1, h2, log_Z ):
        return cls.marginalizeX1( J22, J12.T, J11, h2, h1, log_Z )

