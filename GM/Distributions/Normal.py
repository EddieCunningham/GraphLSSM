import numpy as np
from GenModels.GM.Distributions.Base import ExponentialFam
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
    def dataN( cls, x, constParams=None ):
        cls.checkShape( x )
        if( x.ndim == 2 ):
            return x.shape[ 0 ]
        return 1

    @classmethod
    def unpackSingleSample( cls, x ):
        return x[ 0 ]

    @classmethod
    def sampleShapes( cls ):
        # ( Sample #, dim )
        return ( None, None )

    def isampleShapes( cls ):
        return ( None, self.mus.shape[ 0 ] )

    @classmethod
    def checkShape( cls, x ):
        assert isinstance( x, np.ndarray )
        assert x.ndim == 2 or x.ndim == 1

    ##########################################################################

    @classmethod
    def standardToNat( cls, mu, sigma, returnPrecision=False ):
        n1 = invPsd( sigma )
        n2 = n1.dot( mu )
        if( returnPrecision == False ):
            n1 *= -0.5
        return n1, n2

    @classmethod
    def natToStandard( cls, n1, n2, fromPrecision=False ):
        sigma = np.linalg.inv( n1 )
        if( fromPrecision == False ):
            sigma *= -0.5
        mu = sigma.dot( n2 )
        return mu, sigma

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, constParams=None ):
        # Compute T( x )
        if( x.ndim == 1 ):
            x = x.reshape( ( 1, -1 ) )
        t1 = x.T.dot( x )
        t2 = x.sum( axis=0 )
        return t1, t2

    @classmethod
    def log_partition( cls, x=None, params=None, nat_params=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( nat_params is None )

        if( nat_params is not None ):
            n1, n2 = nat_params
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

    @classmethod
    def log_partitionGradient( cls, params=None, nat_params=None, split=False ):
        # Derivative w.r.t. natural params. Also the expected sufficient stat
        assert ( params is None ) ^ ( nat_params is None )

        mu, sigma = params if params is not None else cls.natToStandard( *nat_params )
        d1 = np.outer( mu, mu ) + sigma
        d2 = mu

        return ( d1, d2 ) if split == False else ( ( d1, d2 ), ( 0, ) )

    def _testLogPartitionGradient( self ):

        import autograd.numpy as anp
        from autograd import jacobian

        n1, n2 = self.nat_params

        def part( _n1 ):
            _n1Inv = anp.linalg.inv( _n1 )
            A = -0.25 * anp.dot( anp.dot( n2, _n1Inv ), n2 ) - 0.5 * anp.linalg.slogdet( -2 * _n1 )[ 1 ]
            return A

        d1 = jacobian( part )( n1 )

        def part( _n2 ):
            _n1Inv = anp.linalg.inv( n1 )
            A = -0.25 * anp.dot( anp.dot( _n2, _n1Inv ), _n2 ) - 0.5 * anp.linalg.slogdet( -2 * n1 )[ 1 ]
            return A

        d2 = jacobian( part )( n2 )

        dAdn1, dAdn2 = self.log_partitionGradient( nat_params=self.nat_params )

        assert np.allclose( d1, dAdn1 )
        assert np.allclose( d2, dAdn2 )

    ##########################################################################

    @classmethod
    def generate( cls, D=2, size=1 ):
        params = ( np.zeros( D ), np.eye( D ) )
        samples = cls.sample( params=params, size=size )
        return samples if size > 1 else cls.unpackSingleSample( samples )

    @classmethod
    def sample( cls, params=None, nat_params=None, size=1 ):
        # Sample from P( x | Ѳ; α )
        assert ( params is None ) ^ ( nat_params is None )
        mu, sigma = params if params is not None else cls.natToStandard( *nat_params )
        ans = multivariate_normal.rvs( mean=mu, cov=sigma, size=size )
        if( mu.size == 1 ):
            ans = ans.reshape( ( -1, 1 ) )
        if( size == 1 ):
            ans = ans[ None ]
        cls.checkShape( ans )
        return ans

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, nat_params=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( nat_params is None )
        mu, sigma = params if params is not None else cls.natToStandard( *nat_params )

        if( x.ndim == 2 ):
            return multivariate_normal.logpdf( x, mean=mu, cov=sigma ).sum()
        return multivariate_normal.logpdf( x, mean=mu, cov=sigma )

    ##########################################################################

    @classmethod
    def marginalizeX1( cls, J11, J12, J22, h1, h2, log_Z, computeMarginal=True ):
        K = h1.shape[ 0 ]

        J11Chol = cho_factor( J11, lower=True )
        J11Invh1 = cho_solve( J11Chol, h1 )

        J = J22 - J12.T @ cho_solve( J11Chol, J12 )
        h = h2 - J12.T.dot( J11Invh1 )

        if( computeMarginal ):
            log_Z = log_Z - \
                    0.5 * h1.dot( J11Invh1 ) + \
                    np.log( np.diag( J11Chol[ 0 ] ) ).sum() - \
                    K * _HALF_LOG_2_PI
        else:
            log_Z = 0

        return J, h, log_Z

    @classmethod
    def marginalizeX2( cls, J11, J12, J22, h1, h2, log_Z, computeMarginal=True ):
        return cls.marginalizeX1( J22, J12.T, J11, h2, h1, log_Z, computeMarginal=computeMarginal )

    ##########################################################################

    @classmethod
    def mode( cls, params=None, nat_params=None ):
        mu, sigma = params if params is not None else cls.natToStandard( *nat_params )
        return ( mu, )

    ##########################################################################

    @classmethod
    def maxLikelihood( cls, x ):
        N = cls.dataN( x )
        t1, t2 = cls.sufficientStats( x )
        mu = t2 / n
        sigma = t1 / n
        return mu, sigma