from GenModels.GM.Distributions import ExponentialFam
from GenModels.GM.Distributions import Dirichlet, TransitionDirichletPrior
from GenModels.GM.States.StandardStates.HMMState import HMMState
import autograd.numpy as np

__all__ = [ 'HMMDirichletPrior' ]

class HMMDirichletPrior( ExponentialFam ):

    # This class just puts dirichlet priors over each row in the initial distribution,
    # transition distribution and emission distribution.  So this class is a distribution
    # over P( Ѳ | α )

    priorClass = None

    def __init__( self, alpha_0, alpha_pi, alpha_L, prior=None, hypers=None ):
        super( HMMDirichletPrior, self ).__init__( alpha_0, alpha_pi, alpha_L, prior=prior, hypers=hypers )

    ##########################################################################

    @property
    def alpha_0( self ):
        return self._params[ 0 ]

    @property
    def alpha_pi( self ):
        return self._params[ 1 ]

    @property
    def alpha_L( self ):
        return self._params[ 2 ]

    ##########################################################################

    @classmethod
    def dataN( cls, x, constParams=None ):
        pi_0, pi, L = x
        if( pi_0.ndim == 1 ):
            return 1
        return pi_0.shape[ 0 ]

    @classmethod
    def unpackSingleSample( cls, x ):
        pi_0, pi, L = x
        return pi_0[ 0 ], pi[ 0 ], L[ 0 ]

    @classmethod
    def sampleShapes( cls ):
        # ( ( Sample #, d_latent ), ( Sample #, d_latent, d_latent ), ( Sample #, d_latent, d_obs ) )
        return ( ( None, None ), ( None, None, None ), ( None, None, None ) )

    def isampleShapes( cls ):
        return ( ( None, self.alpha_0.shape[ 0 ] ), ( None, self.alpha_pi.shape[ 0 ], self.alpha_pi.shape[ 1 ] ), ( None, self.alpha_L.shape[ 0 ], self.alpha_L.shape[ 1 ] ) )

    @classmethod
    def checkShape( cls, x ):
        assert isinstance( x, tuple )
        pi_0, pi, L = x
        assert isinstance( pi_0, np.ndarray ) and isinstance( pi, np.ndarray ) and isinstance( L, np.ndarray )
        if( pi_0.ndim == 2 ):
            assert pi.ndim == 3
            assert L.ndim == 3
            assert pi_0.shape[ 0 ] == pi.shape[ 0 ] and pi.shape[ 0 ] == L.shape[ 0 ]
            assert pi_0.shape[ 1 ] == pi.shape[ 1 ] and pi.shape[ 1 ] == pi.shape[ 2 ] and pi.shape[ 2 ] == L.shape[ 1 ]
        else:
            assert pi_0.ndim == 1
            assert pi.ndim == 2
            assert L.ndim == 2
            assert pi_0.shape[ 0 ] == pi.shape[ 0 ] and pi.shape[ 0 ] == pi.shape[ 1 ] and pi.shape[ 1 ] == L.shape[ 0 ]

    ##########################################################################

    @classmethod
    def standardToNat( cls, alpha_0, alpha_pi, alpha_L ):
        n1, = Dirichlet.standardToNat( alpha_0 )
        n2, = TransitionDirichletPrior.standardToNat( alpha_pi )
        n3, = TransitionDirichletPrior.standardToNat( alpha_L )
        return n1, n2, n3

    @classmethod
    def natToStandard( cls, n1, n2, n3 ):
        alpha_0, = Dirichlet.natToStandard( n1 )
        alpha_pi, = TransitionDirichletPrior.natToStandard( n2 )
        alpha_L, = TransitionDirichletPrior.natToStandard( n3 )
        return alpha_0, alpha_pi, alpha_L

    ##########################################################################

    @property
    def constParams( self ):
        return None

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, constParams=None ):
        # Compute T( x )
        if( cls.dataN( x ) > 1 ):
            t = ( 0, 0, 0, 0, 0, 0 )
            for pi_0, pi, L in zip( *x ):
                t = np.add( t, cls.sufficientStats( ( pi_0, pi, L ) ) )
            return t

        t1, t2, t3 = HMMState.standardToNat( *x )
        t4, t5, t6 = HMMState.log_partition( params=x, split=True )
        return t1, t2, t3, -t4, -t5, -t6

    @classmethod
    def log_partition( cls, x=None, params=None, nat_params=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( nat_params is None )
        alpha_0, alpha_pi, alpha_L = params if params is not None else cls.natToStandard( *nat_params )
        A1 = Dirichlet.log_partition( params=( alpha_0, ), split=split )
        A2 = TransitionDirichletPrior.log_partition( params=( alpha_pi, ), split=split )
        A3 = TransitionDirichletPrior.log_partition( params=( alpha_L, ), split=split )
        return A1 + A2 + A3

    @classmethod
    def log_partitionGradient( cls, params=None, nat_params=None, split=False ):
        # Derivative w.r.t. natural params. Also the expected sufficient stat
        assert ( params is None ) ^ ( nat_params is None )
        n1, n2, n3 = nat_params if nat_params is not None else cls.standardToNat( *params )

        d1 = Dirichlet.log_partitionGradient( nat_params=( n1, ) )[ 0 ]
        d2 = TransitionDirichletPrior.log_partitionGradient( nat_params=( n2, ) )[ 0 ]
        d3 = TransitionDirichletPrior.log_partitionGradient( nat_params=( n3, ) )[ 0 ]
        return ( d1, d2, d3 ) if split == False else ( ( d1, d2, d3 ), ( 0, ) )

    def _testLogPartitionGradient( self ):

        import autograd.numpy as anp
        import autograd.scipy as asp
        from autograd import jacobian

        n1, n2, n3 = self.nat_params

        def dirPart( _n ):
            d = anp.sum( asp.special.gammaln( ( _n + 1 ) ) ) - asp.special.gammaln( anp.sum( _n + 1 ) )
            return d

        def transDirPart( _n ):
            d = 0.0
            for __n in _n:
                d = d + anp.sum( asp.special.gammaln( ( __n + 1 ) ) ) - asp.special.gammaln( anp.sum( __n + 1 ) )
            return d

        def part( _n1, _n2, _n3 ):
            d1 = dirPart( _n1 )
            d2 = transDirPart( _n2 )
            d3 = transDirPart( _n3 )
            return d1 + d2 + d3

        def p1( _n1 ):
            return part( _n1, n2, n3 )

        def p2( _n2 ):
            return part( n1, _n2, n3 )

        def p3( _n3 ):
            return part( n1, n2, _n3 )

        d1, d2, d3 = self.log_partitionGradient( nat_params=self.nat_params )
        _d1 = jacobian( p1 )( n1 )
        _d2 = jacobian( p2 )( n2 )
        _d3 = jacobian( p3 )( n3 )

        assert np.allclose( _d1, d1 )
        assert np.allclose( _d2, d2 )
        assert np.allclose( _d3, d3 )

    ##########################################################################

    @classmethod
    def generate( cls, D_latent=3, D_obs=2, size=1 ):
        params = ( np.ones( D_latent ), np.ones( ( D_latent, D_latent ) ), np.ones( D_latent, D_obs ) )
        samples = cls.sample( params=params, size=size )
        return samples if size > 1 else cls.unpackSingleSample( samples )

    @classmethod
    def sample( cls, params=None, nat_params=None, size=1 ):
        # Sample from P( x | Ѳ; α )
        assert ( params is None ) ^ ( nat_params is None )

        alpha_0, alpha_pi, alpha_L = params if params is not None else cls.natToStandard( *nat_params )

        pi_0 = Dirichlet.sample( params=( alpha_0, ), size=size )
        pi = TransitionDirichletPrior.sample( params=( alpha_pi, ), size=size )
        L = TransitionDirichletPrior.sample( params=( alpha_L, ), size=size )

        ans = ( pi_0, pi, L )
        cls.checkShape( ans )
        return ans

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, nat_params=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( nat_params is None )

        alpha_0, alpha_pi, alpha_L = params if params is not None else cls.natToStandard( *nat_params )

        pi_0, pi, L = x

        ans = Dirichlet.log_likelihood( pi_0, params=( alpha_0, ) )
        ans += TransitionDirichletPrior.log_likelihood( pi, params=( alpha_pi, ) )
        ans += TransitionDirichletPrior.log_likelihood( L, params=( alpha_L, ) )

        return ans
