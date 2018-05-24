import numpy as np
from GenModels.GM.Distributions.Base import ExponentialFam
from scipy.stats import dirichlet
from scipy.special import gammaln
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
        if( x.ndim == 2 ):
            return x.shape[ 0 ]
        return 1

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
    def sufficientStats( cls, x, constParams=None, forPost=False ):
        # Compute T( x )
        if( cls.dataN( x ) > 1 ):
            t = ( 0, 0 )
            for _x in x:
                t = np.add( t, cls.sufficientStats( _x, forPost=forPost ) )
            return t

        ( t1, ) = Categorical.standardToNat( x )
        ( t2, ) = Categorical.log_partition( params=( x, ), split=True )
        return t1, t2

    @classmethod
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( natParams is None )
        ( alpha, ) = params if params is not None else cls.natToStandard( *natParams )
        A1 = gammaln( alpha ).sum()
        A2 = -gammaln( alpha.sum() )
        if( split ):
            return A1, A2
        return A1 + A2

    ##########################################################################

    @classmethod
    def sample( cls, params=None, natParams=None, size=1 ):
        # Sample from P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )

        if( params is not None ):
            if( not isinstance( params, tuple ) and \
                not isinstance( params, list ) ):
                params = ( params, )

        ( alpha, ) = params if params is not None else cls.natToStandard( *natParams )
        ans = dirichlet.rvs( alpha=alpha, size=size )
        return ans

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )
        ( alpha, ) = params if params is not None else cls.natToStandard( *natParams )
        if( isinstance( x, tuple ) ):
            assert len( x ) == 1
            x, = x
        assert isinstance( x, np.ndarray )
        if( x.ndim == 2 ):
            return sum( [ dirichlet.logpdf( _x, alpha=alpha ) for _x in x ] )
        assert isinstance( x, np.ndarray ) and x.ndim == 1
        return dirichlet.logpdf( x, alpha=alpha )
