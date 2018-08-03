import numpy as np
from GenModels.GM.Distributions.Base import ExponentialFam
from scipy.special import gammaln
from GenModels.GM.Distributions import Dirichlet, TensorTransition
import itertools

__all__ = [ 'TensorTransitionDirichletPrior' ]

class TensorTransitionDirichletPrior( ExponentialFam ):

    def __init__( self, alpha=None, prior=None, hypers=None ):
        super( TensorTransitionDirichletPrior, self ).__init__( alpha, prior=prior, hypers=hypers )
        self.Ds = self.alpha.shape

    ##########################################################################

    @property
    def alpha( self ):
        return self._params[ 0 ]

    ##########################################################################

    @classmethod
    def dataN( cls, x, constParams=None ):
        assert constParams is not None
        Ds = constParams
        cls.checkShape( x )
        if( isinstance( x, tuple ) ):
            assert len( x ) == 1
            if( x[ 0 ].ndim > len( Ds ) ):
                return x[ 0 ].shape[ 0 ]
            return 1
        if( x.ndim > len( Ds ) ):
            return x.shape[ 0 ]
        return 1

    @classmethod
    def unpackSingleSample( cls, x ):
        return x[ 0 ]

    @classmethod
    def sampleShapes( cls, Ds ):
        # ( Sample #, dim )
        return tuple( [ None for _ in len( Ds ) + 1 ] )

    def isampleShapes( cls, Ds ):
        return ( None, *Ds )

    @classmethod
    def checkShape( cls, x ):
        if( isinstance( x, tuple ) ):
            assert len( x ) == 1
            assert isinstance( x[ 0 ], np.ndarray )
        else:
            assert isinstance( x, np.ndarray )

    ##########################################################################

    @classmethod
    def standardToNat( cls, alpha ):
        return ( alpha - 1, )

    @classmethod
    def natToStandard( cls, n ):
        return ( n + 1, )

    ##########################################################################

    @property
    def constParams( self ):
        return self.Ds

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, constParams=None ):
        # Compute T( x )
        Ds = constParams
        if( isinstance( x, tuple ) or x.ndim > len( Ds ) ):
            t = ( 0, 0 )
            for _x in x:
                t = np.add( t, cls.sufficientStats( _x, constParams=constParams ) )
            return t

        t1, = TensorTransition.standardToNat( x )
        t2, = TensorTransition.log_partition( params=( x, ), split=True )
        return t1, -t2

    @classmethod
    def log_partition( cls, x=None, params=None, nat_params=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( nat_params is None )
        alpha, = params if params is not None else cls.natToStandard( *nat_params )
        last_dim = alpha.shape[ -1 ]
        return sum( [ Dirichlet.log_partition( params=( a, ) ) for a in alpha.reshape( ( -1, last_dim ) ) ] )

    @classmethod
    def log_partitionGradient( cls, params=None, nat_params=None, split=False ):
        # Derivative w.r.t. natural params. Also the expected sufficient stat
        assert ( params is None ) ^ ( nat_params is None )
        alpha, = nat_params if nat_params is not None else cls.standardToNat( *params )
        last_dim = alpha.shape[ -1 ]

        d = np.vstack( [ Dirichlet.log_partitionGradient( nat_params=( a, ) ) for a in alpha.reshape( ( -1, last_dim ) ) ] ).reshape( alpha.shape )
        return ( d, ) if split == False else ( ( d, ), ( 0, ) )

    def _testLogPartitionGradient( self ):

        import autograd.numpy as anp
        import autograd.scipy as asp
        from autograd import jacobian

        n, = self.nat_params
        def part( _n ):
            ans = 0.0
            last_dim = _n.shape[ -1 ]
            for __n in _n.reshape( ( -1, last_dim ) ):
                ans = ans + anp.sum( asp.special.gammaln( ( __n + 1 ) ) ) - asp.special.gammaln( anp.sum( __n + 1 ) )
            return ans

        d = self.log_partitionGradient( nat_params=self.nat_params )
        _d = jacobian( part )( n )

        assert np.allclose( d, _d )

    ##########################################################################

    @classmethod
    def generate( cls, Ds=[ 2, 3, 4 ], size=1, unpack=True ):
        params = np.ones( Ds )
        samples = cls.sample( params=( params, ), size=size )
        return samples if size > 1 or unpack == False else cls.unpackSingleSample( samples )

    @classmethod
    def sample( cls, params=None, nat_params=None, size=1 ):
        # Sample from P( x | Ѳ; α )
        assert ( params is None ) ^ ( nat_params is None )

        ( alpha, ) = params if params is not None else cls.natToStandard( *nat_params )

        ans = np.empty( alpha.shape + ( size, ) )
        for indices in itertools.product( *[ range( s ) for s in alpha.shape[ :-1 ] ] ):
            ans[ indices ] = Dirichlet.generate( D=alpha.shape[ -1 ], size=size, unpack=False ).T

        ans = np.rollaxis( ans, -1 )

        cls.checkShape( ans )
        return ans

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, nat_params=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( nat_params is None )
        ( alpha, ) = params if params is not None else cls.natToStandard( *nat_params )
        if( isinstance( x, tuple ) ):
            assert len( x ) == 1
            x, = x
        assert isinstance( x, np.ndarray )

        if( x.ndim > alpha.ndim ):
            assert x.ndim == alpha.ndim + 1
            return sum( [ TensorTransitionDirichletPrior.log_likelihood( _x, params=( alpha, ) ) for _x in x ] )

        last_dim = alpha.shape[ -1 ]
        return sum( [ Dirichlet.log_likelihood( _x, params=( a, ) ) for _x, a in zip( x.reshape( ( -1, last_dim ) ), alpha.reshape( ( -1, last_dim ) ) ) ] )
