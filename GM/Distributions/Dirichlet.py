import numpy as np
from Base import Exponential
from scipy.stats import dirichlet
from scipy.special import gammaln
import Categorical

class Dirichlet( Exponential ):

    def __init__( self, alpha=None, prior=None, hypers=None ):
        super( Dirichlet, self ).__init__( alpha, prior=prior, hypers=hypers )

    ##########################################################################

    @classmethod
    def standardToNat( cls, alpha ):
        return ( alpha - 1, )

    @classmethod
    def natToStandard( cls, n ):
        return ( n + 1, )

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, forPost=False ):
        # Compute T( x )
        ( t1, ) = Categorical.Categorical.standardToNat( *x )
        return ( t1, )

    @classmethod
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( natParams is None )
        ( alpha, ) = params if params is not None else cls.natToStandard( *natParams )
        ( x, ) = x
        D = x.shape[ 0 ]
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
            if( not isinstance( params, tuple ) or \
                not isinstance( params, list ) ):
                params = ( params, )

        ( alpha, ) = params if params is not None else cls.natToStandard( *natParams )
        return dirichlet.rvs( alpha=alpha, size=size )

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )
        ( alpha, ) = params if params is not None else cls.natToStandard( *natParams )
        assert isinstance( x, np.ndarray )
        if( x.ndim > 1 ):
            x = x.reshape( -1 )
        return dirichlet.logpdf( x, alpha=alpha )
