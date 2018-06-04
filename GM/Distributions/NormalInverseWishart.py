import numpy as np
from GenModels.GM.Distributions.Base import ExponentialFam
from scipy.special import multigammaln
from GenModels.GM.Distributions.InverseWishart import InverseWishart
from GenModels.GM.Distributions.Normal import Normal
from GenModels.GM.Utility import multigammalnDerivative

class NormalInverseWishart( ExponentialFam ):
    # This class is written with the intention of making it a prior for
    # a normal distribution with an unknown mean and covariance

    def __init__( self, mu_0, kappa, psi, nu, Q=0, prior=None, hypers=None ):
        super( NormalInverseWishart, self ).__init__( mu_0, kappa, psi, nu, Q, prior=prior, hypers=hypers )

    ##########################################################################

    @property
    def mu_0( self ):
        return self._params[ 0 ]

    @property
    def kappa( self ):
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
        cls.checkShape( x )
        mu, sigma = x
        if( mu.ndim == 2 ):
            return mu.shape[ 0 ]
        return 1

    @classmethod
    def unpackSingleSample( cls, x ):
        mu, sigma = x
        return mu[ 0 ], sigma[ 0 ]

    @classmethod
    def sampleShapes( cls ):
        # ( ( Sample #, dim ), ( Sample #, dim1, dim2 ) )
        return ( ( None, None ), ( None, None, None ) )

    def isampleShapes( cls ):
        return ( ( None, self.mu_0.shape[ 0 ] ), ( None, self.psi.shape[ 0 ], self.psi.shape[ 1 ] ) )

    @classmethod
    def checkShape( cls, x ):
        assert isinstance( x, tuple )
        mu, sigma = x
        assert isinstance( mu, np.ndarray ) and isinstance( sigma, np.ndarray )
        if( mu.ndim == 2 ):
            assert sigma.ndim == 3
            assert mu.shape[ 0 ] == sigma.shape[ 0 ]
            assert mu.shape[ 1 ] == sigma.shape[ 1 ]
            assert sigma.shape[ 1 ] == sigma.shape[ 1 ]
        else:
            assert mu.ndim == 1 and sigma.ndim == 2
            assert mu.shape[ 0 ] == sigma.shape[ 0 ]
            assert sigma.shape[ 0 ] == sigma.shape[ 1 ]

    ##########################################################################

    @classmethod
    def standardToNat( cls, mu_0, kappa, psi, nu, Q ):
        n1 = kappa * np.outer( mu_0, mu_0 ) + psi
        n2 = kappa * mu_0
        n3 = kappa
        n4 = nu + psi.shape[ 0 ] + 2
        n5 = 1 + Q
        return n1, n2, n3, n4, n5

    @classmethod
    def natToStandard( cls, n1, n2, n3, n4, n5 ):
        kappa = n3
        mu_0 = 1 / kappa * n2
        psi = n1 - kappa * np.outer( mu_0, mu_0 )
        p = mu_0.shape[ 0 ]
        nu = n4 - p - 2

        # The roll of Q is to offset excess normal base measures!
        Q = n5 - 1
        return mu_0, kappa, psi, nu, Q

    ##########################################################################

    @property
    def constParams( self ):
        return None

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, constParams=None ):
        # Compute T( x )
        if( cls.dataN( x ) > 1 ):
            t = ( 0, 0, 0, 0, 0 )
            for mu, sigma in zip( *x ):
                t = np.add( t, cls.sufficientStats( ( mu, sigma ) ) )
            return t

        t1, t2 = Normal.standardToNat( *x )
        t3, t4, t5 = Normal.log_partition( params=x, split=True )
        return t1, t2, -t3, -t4, -t5

    @classmethod
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( natParams is None )

        mu_0, kappa, psi, nu, Q = params if params is not None else cls.natToStandard( *natParams )

        p = psi.shape[ 0 ]

        A4 = -p / 2 * np.log( kappa )
        A5 = -Q * ( p / 2 * np.log( 2 * np.pi ) )

        if( split == True ):
            A1, A2, A3 = InverseWishart.log_partition( x, params=( psi, nu ), split=True )
            return A1, A2, A3, A4, A5

        A = InverseWishart.log_partition( x, params=( psi, nu ), split=False )
        return A + A4 + A5

    @classmethod
    def log_partitionGradient( cls, params=None, natParams=None ):
        # Derivative w.r.t. natural params
        assert ( params is None ) ^ ( natParams is None )
        n1, n2, n3, n4, n5 = natParams if natParams is not None else cls.standardToNat( *params )
        p = n2.shape[ 0 ]

        k = -( n4 - p - 2 ) / 2
        P = n1 - np.outer( n2, n2 ) / n3
        Q = np.linalg.inv( P )

        d1 = k * Q
        d2 = -2 * k / n3 * Q.dot( n2 )
        d3 = k * Q.T.dot( n2 ).dot( n2 ) / n3**2 - p / ( 2 * n3 )
        d4 = -0.5 * np.linalg.slogdet( P )[ 1 ] + 0.5 * multigammalnDerivative( d=p, x=-k ) + p / 2 * np.log( 2 )
        d5 = -p / 2 * np.log( 2 * np.pi )

        return d1, d2, d3, d4, d5

    def _testLogPartitionGradient( self ):

        import autograd.numpy as anp
        import autograd.scipy as asp
        from autograd import jacobian

        n1, n2, n3, n4, n5 = self.natParams
        p = n2.shape[ 0 ]

        def _part( mu_0, kappa, psi, nu, Q ):
            p = mu_0.shape[ 0 ]
            A1 = -nu / 2 * anp.linalg.slogdet( psi )[ 1 ]
            A2 = asp.special.multigammaln( nu / 2, p )
            A3 = nu * p / 2 * anp.log( 2 )
            A4 = -p / 2 * anp.log( kappa )
            A5 = -Q * ( p / 2 * anp.log( 2 * anp.pi ) )
            return A1 + A2 + A3 + A4 + A5

        def aNat2Std( _n1, _n2, _n3, _n4, _n5 ):
            p = n2.shape[ 0 ]
            kappa = _n3
            mu_0 = 1 / kappa * _n2
            psi = _n1 - kappa * anp.outer( mu_0, mu_0 )
            nu = _n4 - p - 2
            Q = _n5 - 1
            return mu_0, kappa, psi, nu, Q

        def part( _n1, _n2, _n3, _n4, _n5 ):
            mu_0, kappa, psi, nu, Q = aNat2Std( _n1, _n2, _n3, _n4, _n5 )
            return _part( mu_0, kappa, psi, nu, Q )

        def p1( _n1 ):
            return part( _n1, n2, n3, n4, n5 )

        def p2( _n2 ):
            return part( n1, _n2, n3, n4, n5 )

        def p3( _n3 ):
            return part( n1, n2, _n3, n4, n5 )

        def p4( _n4 ):
            return part( n1, n2, n3, _n4, n5 )

        def p5( _n5 ):
            return part( n1, n2, n3, n4, _n5 )

        _d1 = jacobian( p1 )( n1 )
        _d2 = jacobian( p2 )( n2 )
        _d3 = jacobian( p3 )( n3 )
        _d4 = jacobian( p4 )( float( n4 ) )
        _d5 = jacobian( p5 )( float( n5 ) )

        d1, d2, d3, d4, d5 = self.log_partitionGradient( natParams=self.natParams )

        assert np.allclose( _d1, d1 )
        assert np.allclose( _d2, d2 )
        assert np.allclose( _d3, d3 )
        assert np.allclose( _d4, d4 )
        assert np.allclose( _d5, d5 )

    ##########################################################################

    @classmethod
    def generate( cls, D=2, size=1 ):
        params = ( np.zeros( D ), D, np.eye( D ), D, 0 )
        samples = cls.sample( params=params, size=size )
        return samples if size > 1 else cls.unpackSingleSample( samples )

    @classmethod
    def sample( cls, params=None, natParams=None, size=1 ):
        # Sample from P( x | Ѳ; α )

        assert ( params is None ) ^ ( natParams is None )
        mu_0, kappa, psi, nu, _ = params if params is not None else cls.natToStandard( *natParams )

        if( size > 1 ):
            ans = tuple( list( zip( *[ cls.unpackSingleSample( cls.sample( params=params, natParams=natParams, size=1 ) ) for _ in range( size ) ] ) ) )
            mu, sigma = ans
            ans = ( np.array( mu ), np.array( sigma ) )
        else:
            sigma = InverseWishart.sample( params=( psi, nu ) )
            mu = Normal.sample( params=( mu_0, InverseWishart.unpackSingleSample( sigma ) / kappa ) )
            ans = ( mu, sigma )

        cls.checkShape( ans )
        return ans

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )
        mu_0, kappa, psi, nu, _ = params if params is not None else cls.natToStandard( *natParams )

        if( cls.dataN( x ) > 1 ):
            return sum( [ cls.log_likelihood( ( mu, sigma ), params=params, natParams=natParams ) for mu, sigma in zip( *x ) ] )
        mu, sigma = x
        return InverseWishart.log_likelihood( sigma, params=( psi, nu ) ) + \
               Normal.log_likelihood( mu, params=( mu_0, sigma / kappa ) )
