from GenModels.GM.Distributions import ExponentialFam
from GenModels.GM.Distributions import NormalInverseWishart, MatrixNormalInverseWishart
from GenModels.GM.States.StandardStates.LDSState import LDSState
from GenModels.GM.Distributions import Normal, Regression
import numpy as np

__all__ = [ 'LDSMNIWPrior' ]

class LDSMNIWPrior( ExponentialFam ):

    # This class is a distribution over P( Ѳ | α )

    priorClass = None

    def __init__( self, M_trans, V_trans, psi_trans, nu_trans,
                        M_emiss, V_emiss, psi_emiss, nu_emiss,
                        mu_0   , kappa_0, psi_0    , nu_0    , Q0=0, Q1=0, Q2=0, prior=None, hypers=None ):
        super( LDSMNIWPrior, self ).__init__( M_trans, V_trans, psi_trans, nu_trans, Q1,
                                              M_emiss, V_emiss, psi_emiss, nu_emiss, Q2,
                                              mu_0   , kappa_0, psi_0    , nu_0    , Q0, prior=prior, hypers=hypers )

    ##########################################################################

    @property
    def M_trans( self ):
        return self._params[ 0 ]

    @property
    def V_trans( self ):
        return self._params[ 1 ]

    @property
    def psi_trans( self ):
        return self._params[ 2 ]

    @property
    def nu_trans( self ):
        return self._params[ 3 ]

    @property
    def M_emiss( self ):
        return self._params[ 5 ]

    @property
    def V_emiss( self ):
        return self._params[ 6 ]

    @property
    def psi_emiss( self ):
        return self._params[ 7 ]

    @property
    def nu_emiss( self ):
        return self._params[ 8 ]

    @property
    def mu_0( self ):
        return self._params[ 10 ]

    @property
    def kappa_0( self ):
        return self._params[ 11 ]

    @property
    def psi_0( self ):
        return self._params[ 12 ]

    @property
    def nu_0( self ):
        return self._params[ 13 ]

    ##########################################################################

    @classmethod
    def dataN( cls, x ):
        A, sigma, C, R, mu0, sigma0 = x
        if( isinstance( mu0, np.ndarray ) and mu0.ndim == 2 ):
            assert A.ndim == 3
            assert sigma.ndim == 3
            assert C.ndim == 3
            assert R.ndim == 3
            assert sigma0.ndim == 3
            return mu0.shape[ 0 ]
        elif( isinstance( mu0, tuple ) ):
            assert len( A ) == len( mu0 )
            assert len( sigma ) == len( mu0 )
            assert len( C ) == len( mu0 )
            assert len( R ) == len( mu0 )
            assert len( sigma0 ) == len( mu0 )
            return len( mu0 )
        return 1

    @classmethod
    def unpackSingleSample( cls, x ):
        A, sigma, C, R, mu0, sigma0 = x
        return A[ 0 ], sigma[ 0 ], C[ 0 ], R[ 0 ], mu0[ 0 ], sigma0[ 0 ]

    @classmethod
    def sampleShapes( cls ):
        # ( ( Sample #, d_latent, d_latent ), ( Sample #, d_latent, d_latent ),
        #   ( Sample #, d_obs, d_latent ), ( Sample #, d_obs, d_obs ),
        #   ( Sample #, d_latent ), ( Sample #, d_latent, d_latent ) )
        return ( ( None, None, None ), ( None, None, None ),
                 ( None, None, None ), ( None, None, None ),
                 ( None, None ), ( None, None, None ) )

    def isampleShapes( cls ):
        D_latent = self.A.shape[ 0 ]
        D_obs = self.R.shape[ 0 ]
        return ( ( None, D_latent, D_latent ), ( None, D_latent, D_latent ),
                 ( None, D_obs, D_latent ), ( None, D_obs, D_obs ),
                 ( None, D_latent ), ( None, D_latent, D_latent ) )

    @classmethod
    def checkShape( cls, x ):
        assert isinstance( x, tuple )
        A, sigma, C, R, mu0, sigma0 = x
        assert isinstance( A, np.ndarray ) and isinstance( sigma, np.ndarray )
        assert isinstance( C, np.ndarray ) and isinstance( R, np.ndarray )
        assert isinstance( mu0, np.ndarray ) and isinstance( sigma0, np.ndarray )
        if( mu0.ndim == 2 ):
            assert A.ndim == 3 and sigma.ndim == 3
            assert C.ndim == 3 and R.ndim == 3
            assert sigma0.ndim == 3
            assert A.shape[ 0 ] == sigma.shape[ 0 ] and C.shape[ 0 ] == R.shape[ 0 ]
            assert A.shape[ 0 ] == C.shape[ 0 ] and mu0.shape[ 0 ] == sigma.shape[ 0 ]
            assert A.shape[ 0 ] == mu0.shape[ 0 ]
            D_latent = A.shape[ 1 ]
            D_obs = R.shape[ 1 ]
        else:
            assert A.ndim == 2 and sigma.ndim == 2
            assert C.ndim == 2 and R.ndim == 2
            assert mu0.ndim == 1 and sigma0.ndim == 2
            D_latent = A.shape[ 0 ]
            D_obs = R.shape[ 0 ]

        assert A.shape[ -2 ]      == D_latent and A.shape[ -1 ]      == D_latent
        assert sigma.shape[ -2 ]  == D_latent and sigma.shape[ -1 ]  == D_latent
        assert C.shape[ -2 ]      == D_obs    and C.shape[ -1 ]      == D_latent
        assert R.shape[ -2 ]      == D_obs    and R.shape[ -1 ]      == D_obs
        assert mu0.shape[ -1 ]    == D_latent
        assert sigma0.shape[ -2 ] == D_latent and sigma0.shape[ -1 ] == D_latent

    ##########################################################################

    @classmethod
    def standardToNat( cls, M_trans, V_trans, psi_trans, nu_trans, Q1,
                            M_emiss, V_emiss, psi_emiss, nu_emiss, Q2,
                            mu_0   , kappa_0, psi_0    , nu_0    , Q0 ):
        n1 , n2 , n3 , n4 , n5  = MatrixNormalInverseWishart.standardToNat( M_trans, V_trans, psi_trans, nu_trans, Q1 )
        n6 , n7 , n8 , n9 , n10 = MatrixNormalInverseWishart.standardToNat( M_emiss, V_emiss, psi_emiss, nu_emiss, Q2 )
        n11, n12, n13, n14, n15 = NormalInverseWishart.standardToNat( mu_0, kappa_0, psi_0, nu_0, Q0 )
        return n1, n2, n3, n6, n7, n8, n11, n12, n4, n5, n9, n10, n13, n14, n15

    @classmethod
    def natToStandard( cls, n1, n2, n3, n6, n7, n8, n11, n12, n4, n5, n9, n10, n13, n14, n15 ):
        # The order of the inputs to this function is important because it has the partition values at the end
        M_trans, V_trans, psi_trans, nu_trans, Q1 = MatrixNormalInverseWishart.natToStandard( n1, n2, n3, n4, n5 )
        M_emiss, V_emiss, psi_emiss, nu_emiss, Q2 = MatrixNormalInverseWishart.natToStandard( n6, n7 , n8 , n9 , n10 )
        mu_0   , kappa_0, psi_0    , nu_0    , Q0 = NormalInverseWishart.natToStandard( n11, n12, n13, n14, n15 )
        return M_trans, V_trans, psi_trans, nu_trans, Q1, M_emiss, V_emiss, psi_emiss, nu_emiss, Q2, mu_0, kappa_0, psi_0, nu_0, Q0

    ##########################################################################

    @property
    def constParams( self ):
        return None

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, constParams=None ):
        # Compute T( x )
        if( cls.dataN( x ) > 1 ):
            t = ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
            for A, sigma, C, R, mu0, sigma0 in zip( *x ):
                t = np.add( t, cls.sufficientStats( ( A, sigma, C, R, mu0, sigma0 ) ) )
            return t

        t1, t2, t3, t4, t5, t6, t7, t8 = LDSState.standardToNat( *x )
        t9, t10, t11, t12, t13, t14, t15 = LDSState.log_partition( params=x, split=True )
        return t1, t2, t3, t4, t5, t6, t7, t8, -t9, -t10, -t11, -t12, -t13, -t14, -t15

    @classmethod
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( natParams is None )
        M_trans, V_trans, psi_trans, nu_trans, Q1, M_emiss, V_emiss, psi_emiss, nu_emiss, Q2, mu_0, kappa_0, psi_0, nu_0, Q0 = params if params is not None else cls.natToStandard( *natParams )

        A1 = MatrixNormalInverseWishart.log_partition( x=x, params=( M_trans, V_trans, psi_trans, nu_trans, Q1 ), split=split )
        A2 = MatrixNormalInverseWishart.log_partition( x=x, params=( M_emiss, V_emiss, psi_emiss, nu_emiss, Q2 ), split=split )
        A3 = NormalInverseWishart.log_partition( x=x, params=( mu_0, kappa_0, psi_0, nu_0, Q0 ), split=split )
        return A1 + A2 + A3

    @classmethod
    def log_partitionGradient( cls, params=None, natParams=None, split=False ):
        # Derivative w.r.t. natural params. Also the expected sufficient stat
        assert ( params is None ) ^ ( natParams is None )
        n1, n2, n3, n6, n7, n8, n11, n12, n4, n5, n9, n10, n13, n14, n15 = natParams if natParams is not None else cls.standardToNat( *params )
        d1, d2, d3, d4, d5 = MatrixNormalInverseWishart.log_partitionGradient( natParams=( n1, n2, n3, n4, n5 ) )
        d6, d7, d8, d9, d10 = MatrixNormalInverseWishart.log_partitionGradient( natParams=( n6, n7, n8, n9, n10 ) )
        d11, d12, d13, d14, d15 = NormalInverseWishart.log_partitionGradient( natParams=( n11, n12, n13, n14, n15 ) )

        if( split == False ):
            return d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15
        return ( ( d1, d2, d3, d6, d7, d8, d11, d12 ), ( d4, d5, d9, d10, d13, d14, d15 ) )

    def _testLogPartitionGradient( self ):

        import autograd.numpy as anp
        import autograd.scipy as asp
        from autograd import jacobian
        from functools import partial

        def _MNIWpart( M, V, psi, nu, Q ):
            n, p = M.shape
            A1 = -nu / 2 * anp.linalg.slogdet( psi )[ 1 ]
            A2 = asp.special.multigammaln( nu / 2, n )
            A3 = nu * n / 2 * anp.log( 2 )
            A4 = n / 2 * anp.linalg.slogdet( V )[ 1 ]
            A5 = -Q * ( n / 2 * anp.log( 2 * anp.pi ) )
            return A1 + A2 + A3 + A4 + A5

        def aMNIWNat2Std( _n1, _n2, _n3, _n4, _n5 ):
            p, n = _n3.shape
            V = anp.linalg.inv( _n2 )
            M = _n3.T @ V
            psi = _n1 - M @ _n2 @ M.T
            nu = _n4 - 1 - n - p
            Q = _n5 - p
            return M, V, psi, nu, Q

        def _NIWpart( mu_0, kappa, psi, nu, Q ):
            p = mu_0.shape[ 0 ]
            A1 = -nu / 2 * anp.linalg.slogdet( psi )[ 1 ]
            A2 = asp.special.multigammaln( nu / 2, p )
            A3 = nu * p / 2 * anp.log( 2 )
            A4 = -p / 2 * anp.log( kappa )
            A5 = -Q * ( p / 2 * anp.log( 2 * anp.pi ) )
            return A1 + A2 + A3 + A4 + A5

        def aNIWNat2Std( _n1, _n2, _n3, _n4, _n5 ):
            p = n2.shape[ 0 ]
            kappa = _n3
            mu_0 = 1 / kappa * _n2
            psi = _n1 - kappa * anp.outer( mu_0, mu_0 )
            nu = _n4 - p - 2
            Q = _n5 - 1
            return mu_0, kappa, psi, nu, Q

        def part( _n1, _n2, _n3, _n6, _n7, _n8, _n11, _n12, _n4, _n5, _n9, _n10, _n13, _n14, _n15 ):
            M_trans, V_trans, psi_trans, nu_trans, Q_trans = aMNIWNat2Std( _n1, _n2, _n3, _n4, _n5 )
            M_emiss, V_emiss, psi_emiss, nu_emiss, Q_emiss = aMNIWNat2Std( _n6, _n7, _n8, _n9, _n10 )
            mu_0, kappa_0, psi_0, nu_0, Q0 = aNIWNat2Std( _n11, _n12, _n13, _n14, _n15 )
            d1 = _MNIWpart( M_trans, V_trans, psi_trans, nu_trans, Q_trans )
            d2 = _MNIWpart( M_emiss, V_emiss, psi_emiss, nu_emiss, Q_emiss )
            d3 = _NIWpart( mu_0, kappa_0, psi_0, nu_0, Q0 )
            return d1 + d2 + d3

        def p( n, i=None ):
            ns = [ _n if j != i else n for j, _n in enumerate( self.natParams ) ]
            return part( *ns )

        #0   1   2   3   4   5    6   7    8   9  10   11   12   13   14
        n1, n2, n3, n6, n7, n8, n11, n12, n4, n5, n9, n10, n13, n14, n15 = self.natParams

        _d1 = jacobian( partial( p, i=0 ) )( n1 )
        _d2 = jacobian( partial( p, i=1 ) )( n2 )
        _d3 = jacobian( partial( p, i=2 ) )( n3 )
        _d4 = jacobian( partial( p, i=8 ) )( float( n4 ) )
        _d5 = jacobian( partial( p, i=9 ) )( float( n5 ) )
        _d6 = jacobian( partial( p, i=3 ) )( n6 )
        _d7 = jacobian( partial( p, i=4 ) )( n7 )
        _d8 = jacobian( partial( p, i=5 ) )( n8 )
        _d9 = jacobian( partial( p, i=10 ) )( float( n9 ) )
        _d10 = jacobian( partial( p, i=11 ) )( float( n10 ) )
        _d11 = jacobian( partial( p, i=6 ) )( n11 )
        _d12 = jacobian( partial( p, i=7 ) )( n12 )
        _d13 = jacobian( partial( p, i=12 ) )( float( n13 ) )
        _d14 = jacobian( partial( p, i=13 ) )( float( n14 ) )
        _d15 = jacobian( partial( p, i=14 ) )( float( n15 ) )

        d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15 = self.log_partitionGradient( natParams=self.natParams )

        assert np.allclose( _d1, d1 )
        assert np.allclose( _d2, d2 )
        assert np.allclose( _d3, d3 )
        assert np.allclose( _d4, d4 )
        assert np.allclose( _d5, d5 )
        assert np.allclose( _d6, d6 )
        assert np.allclose( _d7, d7 )
        assert np.allclose( _d8, d8 )
        assert np.allclose( _d9, d9 )
        assert np.allclose( _d10, d10 )
        assert np.allclose( _d11, d11 )
        assert np.allclose( _d12, d12 )
        assert np.allclose( _d13, d13 )
        assert np.allclose( _d14, d14 )
        assert np.allclose( _d15, d15 )

    ##########################################################################

    @classmethod
    def generate( cls, D_latent=3, D_obs=2, size=1 ):

        # Not sure why I used a tuple here
        params = (
            ( 'M_trans', np.zeros( ( D_latent, D_latent ) ) ),
            ( 'V_trans', np.eye( D_latent ) ),
            ( 'psi_trans', np.eye( D_latent ) ),
            ( 'nu_trans', D_latent ),
            ( 'Q1', 0 ),

            ( 'M_emiss', np.zeros( ( D_obs, D_latent ) ) ),
            ( 'V_emiss', np.eye( D_latent ) ),
            ( 'psi_emiss', np.eye( D_obs ) ),
            ( 'nu_emiss', D_obs ),
            ( 'Q2', 0 ),

            ( 'mu_0', np.zeros( D_latent ) ),
            ( 'kappa_0', D_latent ),
            ( 'psi_0', np.eye( D_latent ) ),
            ( 'nu_0', D_latent )
            ( 'Q0', 0 )
        )

        _, params = zip( *params )

        samples = cls.sample( params=params, size=size )
        return samples if size > 1 else cls.unpackSingleSample( samples )

    @classmethod
    def sample( cls, params=None, natParams=None, size=1 ):
        # Sample from P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )

        M_trans, V_trans, psi_trans, nu_trans, Q1, M_emiss, V_emiss, psi_emiss, nu_emiss, Q2, mu_0, kappa_0, psi_0, nu_0, Q0 = params if params is not None else cls.natToStandard( *natParams )

        A, sigma = MatrixNormalInverseWishart.sample( params=( M_trans, V_trans, psi_trans, nu_trans, Q1 ), size=size )
        C, R = MatrixNormalInverseWishart.sample( params=( M_emiss, V_emiss, psi_emiss, nu_emiss, Q2 ), size=size )
        mu0sigma0 = NormalInverseWishart.sample( params=( mu_0, kappa_0, psi_0, nu_0, Q0 ), size=size )
        mu0, sigma0 = mu0sigma0

        ans = ( A, sigma, C, R, mu0, sigma0 )
        cls.checkShape( ans )
        return ans

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )

        A, sigma, C, R, mu0, sigma0 = x
        M_trans, V_trans, psi_trans, nu_trans, Q1, M_emiss, V_emiss, psi_emiss, nu_emiss, Q2, mu_0, kappa_0, psi_0, nu_0, Q0 = params if params is not None else cls.natToStandard( *natParams )

        ans1 = MatrixNormalInverseWishart.log_likelihood( ( A, sigma ), params=( M_trans, V_trans, psi_trans, nu_trans, Q1 ) )
        ans2 = MatrixNormalInverseWishart.log_likelihood( ( C, R ), params=( M_emiss, V_emiss, psi_emiss, nu_emiss, Q2 ) )
        ans3 = NormalInverseWishart.log_likelihood( ( mu0, sigma0 ), params=( mu_0, kappa_0, psi_0, nu_0, Q0 ) )

        # print( '\nans1', ans1 )
        # print( 'ans2', ans2 )
        # print( 'ans3', ans3 )

        ans = ans1 + ans2 + ans3
        # print( 'ans', ans )

        return ans

    ##########################################################################

    @classmethod
    def log_pdf( cls, natParams, sufficientStats, log_partition=None ):

        from collections import Iterable

        ans1 = 0.0
        for i, ( natParam, stat ) in enumerate( zip( natParams[ :3 ], sufficientStats[ :3 ] ) ):
            ans1 += ( natParam * stat ).sum()
        for i, ( natParam, stat ) in enumerate( zip( natParams[ 8:10 ], sufficientStats[ 8:10 ] ) ):
            ans1 += ( natParam * stat ).sum()
        ans1 -= sum( log_partition[ :5 ] )

        ans2 = 0.0
        for i, ( natParam, stat ) in enumerate( zip( natParams[ 3:6 ], sufficientStats[ 3:6 ] ) ):
            ans2 += ( natParam * stat ).sum()
        for i, ( natParam, stat ) in enumerate( zip( natParams[ 10:12 ], sufficientStats[ 10:12 ] ) ):
            ans2 += ( natParam * stat ).sum()
        ans2 -= sum( log_partition[ 5:10 ] )

        ans3 = 0.0
        for i, ( natParam, stat ) in enumerate( zip( natParams[ 6:8 ], sufficientStats[ 6:8 ] ) ):
            ans3 += ( natParam * stat ).sum()
        for i, ( natParam, stat ) in enumerate( zip( natParams[ 12:15 ], sufficientStats[ 12:15 ] ) ):
            ans3 += ( natParam * stat ).sum()
        ans3 -= sum( log_partition[ 10: ] )

        # print( '\nIN LOGPDF' )
        # print( 'ans1', ans1 )
        # print( 'ans2', ans2 )
        # print( 'ans3', ans3 )

        ans = ans1 + ans2 + ans3

        # print( 'ans', ans )

        # if( log_partition is not None ):
        #     if( isinstance( log_partition, tuple ) ):
        #         ans -= sum( log_partition )
        #     else:
        #         ans -= log_partition

        assert isinstance( ans, Iterable ) == False, log_partition

        return ans
