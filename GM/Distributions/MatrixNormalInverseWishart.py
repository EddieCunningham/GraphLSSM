import autograd.numpy as np
from GenModels.GM.Distributions.Base import ExponentialFam
from scipy.special import multigammaln
from GenModels.GM.Distributions.TensorNormal import TensorNormal
from GenModels.GM.Distributions.InverseWishart import InverseWishart
from GenModels.GM.Distributions.Regression import Regression
from GenModels.GM.Utility import multigammalnDerivative, cheatPrecisionHelper

from scipy.stats import matrix_normal

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
    def dataN( cls, x, constParams=None ):
        cls.checkShape( x )
        A, sigma = x
        if( A.ndim == 3 ):
            return A.shape[ 0 ]
        return 1

    @classmethod
    def unpackSingleSample( cls, x ):
        A, sigma = x
        return A[ 0 ], sigma[ 0 ]

    @classmethod
    def sampleShapes( cls ):
        # ( ( Sample #, dim1, dim2 ), ( Sample #, dim1, dim2 ) )
        return ( ( None, None, None ), ( None, None, None ) )

    def isampleShapes( cls ):
        return ( ( None, self.M.shape[ 0 ], self.M.shape[ 1 ] ), ( None, self.psi.shape[ 0 ], self.psi.shape[ 1 ] ) )

    @classmethod
    def checkShape( cls, x ):
        assert isinstance( x, tuple )
        A, sigma = x
        assert isinstance( A, np.ndarray ) and isinstance( sigma, np.ndarray )
        if( A.ndim == 3 ):
            assert sigma.ndim == 3
            assert A.shape[ 0 ] == sigma.shape[ 0 ]
            assert A.shape[ 1 ] == sigma.shape[ 1 ]
            assert sigma.shape[ 1 ] == sigma.shape[ 2 ]
        else:
            assert A.ndim == 2
            assert sigma.ndim == 2
            assert A.shape[ 0 ] == sigma.shape[ 0 ]
            assert sigma.shape[ 0 ] == sigma.shape[ 1 ]

    ##########################################################################

    @classmethod
    def standardToNat( cls, M, V, psi, nu, Q ):

        n, p = M.shape

        VInv = np.linalg.inv( V )
        n1 = M @ VInv @ M.T + psi
        n2 = VInv
        n3 = VInv @ M.T
        n4 = nu + n + p + 1
        n5 = p + Q
        return n1, n2, n3, n4, n5

    @classmethod
    def natToStandard( cls, n1, n2, n3, n4, n5 ):

        p, n = n3.shape

        V = np.linalg.inv( n2 )
        M = n3.T @ V
        psi = n1 - M @ n2 @ M.T
        nu = n4 - 1 - n - p
        Q = n5 - p

        # # Numerical padding ( https://github.com/mattjj/pybasicbayes/blob/master/pybasicbayes/distributions/regression.py#L96 )
        # # Having precision issues right now so going to keep this in for the moment and hope it helps
        # V = cheatPrecisionHelper( V, p )
        # psi = cheatPrecisionHelper( psi, n )
        return M, V, psi, nu, Q

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
            for A, sigma in zip( *x ):
                t = np.add( t, cls.sufficientStats( ( A, sigma ) ) )
            return t

        t1, t2, t3 = Regression.standardToNat( *x )
        t4, t5 = Regression.log_partition( params=x, split=True )
        return t1, t2, t3, -t4, -t5

    @classmethod
    def log_partition( cls, x=None, params=None, nat_params=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( nat_params is None )

        M, V, psi, nu, Q = params if params is not None else cls.natToStandard( *nat_params )

        n, p = M.shape

        A1, A2, A3 = InverseWishart.log_partition( params=( psi, nu ), split=True )
        A4 = n / 2 * np.linalg.slogdet( V )[ 1 ]
        A5 = -Q * ( n / 2 * np.log( 2 * np.pi ) )

        if( split ):
            return A1, A2, A3, A4, A5
        return A1 + A2 + A3 + A4 + A5

    @classmethod
    def log_partitionGradient( cls, params=None, nat_params=None, split=False ):
        # Derivative w.r.t. natural params. Also the expected sufficient stat
        assert ( params is None ) ^ ( nat_params is None )
        n1, n2, n3, n4, n5 = nat_params if nat_params is not None else cls.standardToNat( *params )

        p, n = n3.shape

        k = ( n4 - 1 - n - p ) / 2
        n2Inv = np.linalg.inv( n2 )
        P = n1 - n3.T @ n2Inv @ n3
        Q = np.linalg.inv( P )

        d1 = -k * Q
        d2 = -k * ( n2Inv @ n3 @ Q @ n3.T @ n2Inv ).T - n / 2 * n2Inv.T
        d3 = 2 * k * n2Inv @ n3 @ Q
        d4 = -0.5 * np.linalg.slogdet( P )[ 1 ] + 0.5 * multigammalnDerivative( d=n, x=k ) + n / 2 * np.log( 2 )
        d5 = -n / 2 * np.log( 2 * np.pi )

        return ( d1, d2, d3, d4, d5 ) if split == False else ( ( d1, d2, d3 ), ( d4, d5 ) )

    def _testLogPartitionGradient( self ):

        import autograd.numpy as anp
        import autograd.scipy as asp
        from autograd import jacobian

        n1, n2, n3, n4, n5 = self.nat_params
        p, n = n3.shape

        def _part( M, V, psi, nu, Q ):
            A1 = -nu / 2 * anp.linalg.slogdet( psi )[ 1 ]
            A2 = asp.special.multigammaln( nu / 2, n )
            A3 = nu * n / 2 * anp.log( 2 )
            A4 = n / 2 * anp.linalg.slogdet( V )[ 1 ]
            A5 = -Q * ( n / 2 * anp.log( 2 * anp.pi ) )
            return A1 + A2 + A3 + A4 + A5

        def aNat2Std( _n1, _n2, _n3, _n4, _n5 ):
            p, n = _n3.shape
            V = anp.linalg.inv( _n2 )
            M = _n3.T @ V
            psi = _n1 - M @ _n2 @ M.T
            nu = _n4 - 1 - n - p
            Q = _n5 - p
            return M, V, psi, nu, Q

        def part( _n1, _n2, _n3, _n4, _n5 ):
            M, V, psi, nu, Q = aNat2Std( _n1, _n2, _n3, _n4, _n5 )
            return _part( M, V, psi, nu, Q )

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

        d1, d2, d3, d4, d5 = self.log_partitionGradient( nat_params=self.nat_params )

        assert np.allclose( _d1, d1 )
        assert np.allclose( _d2, d2 )
        assert np.allclose( _d3, d3 )
        assert np.allclose( _d4, d4 )
        assert np.allclose( _d5, d5 )

    ##########################################################################

    @classmethod
    def generate( cls, D_in=3, D_out=2, size=1 ):
        params = ( np.zeros( ( D_out, D_in ) ), np.eye( D_in ), np.eye( D_out ), D_out, 0 )
        samples = cls.sample( params=params, size=size )
        return samples if size > 1 else cls.unpackSingleSample( samples )

    @classmethod
    def sample( cls, params=None, nat_params=None, size=1 ):
        # Sample from P( x | Ѳ; α )
        assert ( params is None ) ^ ( nat_params is None )
        M, V, psi, nu, _ = params if params is not None else cls.natToStandard( *nat_params )

        if( size > 1 ):
            ans = list( zip( *[ cls.unpackSingleSample( cls.sample( params=params, nat_params=nat_params, size=1 ) ) for _ in range( size ) ] ) )
            A, sigma = ans
            ans = ( np.array( A ), np.array( sigma ) )
        else:
            sigma = InverseWishart.sample( params=( psi, nu ) )
            A = matrix_normal.rvs( mean=M, rowcov=InverseWishart.unpackSingleSample( sigma ), colcov=V )[ None ]
            # A = TensorNormal.sample( params=( M, ( InverseWishart.unpackSingleSample( sigma ), V ) ), size=1 )
            # print( A )
            # assert 0
            ans = ( A, sigma )

        cls.checkShape( ans )
        return ans

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, nat_params=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( nat_params is None )
        M, V, psi, nu, _ = params if params is not None else cls.natToStandard( *nat_params )
        if( cls.dataN( x ) > 1 ):
            return sum( [ cls.log_likelihood( ( A, sigma ), params=params, nat_params=nat_params ) for A, sigma in zip( *x ) ] )
        A, sigma = x
        return InverseWishart.log_likelihood( sigma, params=( psi, nu ) ) + \
               TensorNormal.log_likelihood( A[ None ], params=( M, ( sigma, V ) ) )
