import numpy as np
from GenModels.GM.Distributions.Base import ExponentialFam
from scipy.stats import dirichlet
from scipy.special import gammaln, digamma
from GenModels.GM.Distributions.Categorical import Categorical

class Dirichlet( ExponentialFam ):

    def __init__( self, alpha=None, prior=None, hypers=None ):
        super( Dirichlet, self ).__init__( alpha, prior=prior, hypers=hypers )

    ##########################################################################

    @property
    def alpha( self ):
        return self._params[ 0 ]

    ##########################################################################

    @classmethod
    def dataN( cls, x ):
        if( isinstance( x, tuple ) ):
            assert len( x ) == 1
            x = x[ 0 ]
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
        return ( None, self.D )

    @classmethod
    def checkShape( cls, x ):
        if( isinstance( x, tuple ) ):
            assert len( x ) == 1
            x = x[ 0 ]
        assert isinstance( x, np.ndarray ), x
        assert x.ndim == 2 or x.ndim == 1

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
        return None

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, constParams=None ):
        # Compute T( x )
        if( cls.dataN( x ) > 1 ):
            t = ( 0, 0 )
            for _x in x:
                t = np.add( t, cls.sufficientStats( _x ) )
            return t

        ( t1, ) = Categorical.standardToNat( x )
        ( t2, ) = Categorical.log_partition( params=( x, ), split=True )
        return t1, t2

    @classmethod
    def log_partition( cls, x=None, params=None, nat_params=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( nat_params is None )
        ( alpha, ) = params if params is not None else cls.natToStandard( *nat_params )
        A1 = gammaln( alpha ).sum()
        A2 = -gammaln( alpha.sum() )
        if( split ):
            return A1, A2
        return A1 + A2

    @classmethod
    def log_partitionGradient( cls, params=None, nat_params=None, split=False ):
        # Derivative w.r.t. natural params. Also the expected sufficient stat
        assert ( params is None ) ^ ( nat_params is None )
        n, = nat_params if nat_params is not None else cls.standardToNat( *params )
        assert np.all( n > 0 )
        d = digamma( ( n + 1 ) ) - digamma( ( n + 1 ).sum() )
        return ( d, ) if split == False else ( ( d, ), ( 0, ) )

    def _testLogPartitionGradient( self ):

        import autograd.numpy as anp
        import autograd.scipy as asp
        from autograd import jacobian

        n, = self.nat_params

        def part( _n ):
            d = anp.sum( asp.special.gammaln( ( _n + 1 ) ) ) - asp.special.gammaln( anp.sum( _n + 1 ) )
            return d

        d = jacobian( part )( n )
        dAdn = self.log_partitionGradient( nat_params=self.nat_params )

        assert np.allclose( d, dAdn )

    ##########################################################################

    @classmethod
    def generate( cls, D=2, size=1 ):
        params = ( np.ones( D ), )
        samples = cls.sample( params=params, size=size )
        return samples if size > 1 else cls.unpackSingleSample( samples )

    @classmethod
    def sample( cls, params=None, nat_params=None, size=1 ):
        # Sample from P( x | Ѳ; α )
        assert ( params is None ) ^ ( nat_params is None )

        ( alpha, ) = params if params is not None else cls.natToStandard( *nat_params )
        ans = dirichlet.rvs( alpha=alpha, size=size )
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
        if( x.ndim == 2 ):
            return sum( [ dirichlet.logpdf( _x, alpha=alpha ) for _x in x ] )
        assert isinstance( x, np.ndarray ) and x.ndim == 1
        return dirichlet.logpdf( x, alpha=alpha )

    ##########################################################################

    @classmethod
    def mode( cls, params=None, nat_params=None ):
        assert ( params is None ) ^ ( nat_params is None )
        ( n, ) = nat_params if nat_params is not None else cls.standardToNat( *params )
        return ( n / n.sum(), )