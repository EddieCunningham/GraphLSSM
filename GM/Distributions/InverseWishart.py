import numpy as np
from GenModels.GM.Distributions.Base import ExponentialFam
from scipy.stats import invwishart
from scipy.special import multigammaln, digamma
import autograd
from GenModels.GM.Utility import multigammalnDerivative

_LOG_2 = np.log( 2 )

class InverseWishart( ExponentialFam ):

    def __init__( self, psi, nu, prior=None, hypers=None ):
        super( InverseWishart, self ).__init__( psi, nu, prior=prior, hypers=hypers )

    ##########################################################################

    @property
    def psi( self ):
        return self._params[ 0 ]

    @property
    def nu( self ):
        return self._params[ 1 ]

    ##########################################################################

    @classmethod
    def dataN( cls, x ):
        cls.checkShape( x )
        if( x.ndim == 3 ):
            return x.shape[ 0 ]
        return 1

    @classmethod
    def unpackSingleSample( cls, x ):
        return x[ 0 ]

    @classmethod
    def sampleShapes( cls ):
        # ( Sample #, dim1, dim2 )
        return ( None, None, None )

    def isampleShapes( cls ):
        return ( None, self.psi.shape[ 0 ], self.psi.shape[ 1 ] )

    @classmethod
    def checkShape( cls, x ):
        assert isinstance( x, np.ndarray )
        if( x.ndim == 3 ):
            assert x.shape[ 1 ] == x.shape[ 2 ]
        else:
            assert x.ndim == 2
            assert x.shape[ 0 ] == x.shape[ 1 ]

    ##########################################################################

    @classmethod
    def standardToNat( cls, psi, nu ):
        p = psi.shape[ 0 ]
        n1 = -0.5 * psi
        n2 = -0.5 * ( nu + p + 1 )
        return n1, n2

    @classmethod
    def natToStandard( cls, n1, n2 ):
        p = n1.shape[ 0 ]
        psi = -2 * n1
        nu = -2 * n2 - p - 1
        return psi, nu

    ##########################################################################

    @property
    def constParams( self ):
        return None

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, constParams=None ):
        # Compute T( x )

        if( cls.dataN( x ) > 1 ):
            t1 = np.linalg.inv( x ).sum( axis=0 )
            t2 = np.linalg.slogdet( x )[ 1 ].sum()
        else:
            t1 = np.linalg.inv( x )
            t2 = np.linalg.slogdet( x )[ 1 ]
        return t1, t2

    @classmethod
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( natParams is None )

        # its just easier to use the standard params
        psi, nu = params if params is not None else cls.natToStandard( *natParams )
        p = psi.shape[ 0 ]

        A1 = -nu / 2 * np.linalg.slogdet( psi )[ 1 ]
        A2 = multigammaln( nu / 2, p )
        A3 = nu * p / 2 * _LOG_2

        if( split ):
            return A1, A2, A3
        return A1 + A2 + A3

    @classmethod
    def log_partitionGradient( cls, params=None, natParams=None ):
        # Derivative w.r.t. natural params
        assert ( params is None ) ^ ( natParams is None )
        n1, n2 = natParams if natParams is not None else cls.standardToNat( *params )
        n1Inv = np.linalg.inv( n1 )
        p = n1.shape[ 0 ]

        k = -( n2 + ( p + 1 ) / 2 )
        d1 = -k * n1Inv
        d2 = np.linalg.slogdet( -2 * n1 )[ 1 ] - multigammalnDerivative( d=p, x=k ) - p * _LOG_2

        return d1, d2

    def _testLogPartitionGradient( self ):

        import autograd.numpy as anp
        import autograd.scipy as asp
        from autograd import jacobian

        n1, n2 = self.natParams
        p = n1.shape[ 0 ]

        def part( _n1 ):
            A = -( -2 * n2 - p - 1 ) / 2 * anp.linalg.slogdet( -2 * _n1 )[ 1 ] + asp.special.multigammaln( ( -2 * n2 - p - 1 ) / 2, p ) + ( -2 * n2 - p - 1 ) * p / 2 * _LOG_2
            return A

        d1 = jacobian( part )( n1 )

        def part( _n2 ):
            A = -( -2 * _n2 - p - 1 ) / 2 * anp.linalg.slogdet( -2 * n1 )[ 1 ] + asp.special.multigammaln( ( -2 * _n2 - p - 1 ) / 2, p ) + ( -2 * _n2 - p - 1 ) * p / 2 * _LOG_2
            return A

        d2 = jacobian( part )( n2 )

        dAdn1, dAdn2 = self.log_partitionGradient( natParams=self.natParams )

        assert np.allclose( d1, dAdn1 )
        assert np.allclose( d2, dAdn2 )

    ##########################################################################

    @classmethod
    def generate( cls, D=2, size=1 ):
        params = ( np.eye( D ), D )
        samples = cls.sample( params=params, size=size )
        return samples if size > 1 else cls.unpackSingleSample( samples )

    @classmethod
    def sample( cls, params=None, natParams=None, size=1 ):
        # Sample from P( x | Ѳ; α )

        if( params is None and natParams is None ):
            assert D is not None
            params = ( np.eye( D ), D )

        assert ( params is None ) ^ ( natParams is None )
        psi, nu = params if params is not None else cls.natToStandard( *natParams )
        ans = invwishart.rvs( df=nu, scale=psi, size=size )
        if( size == 1 ):
            ans = ans[ None ]
        cls.checkShape( ans )
        return ans

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )
        # There is a bug in scipy's invwishart.logpdf! Don't use it!
        return cls.log_likelihoodExpFam( x, params=params, natParams=natParams )
