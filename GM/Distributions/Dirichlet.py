import numpy as np
from GenModels.GM.Distributions.Base import ExponentialFam, checkExpFamArgs, multiSampleLikelihood
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
    def dataN( cls, x, ravel=False ):
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
        if( isinstance( x, tuple ) ):
            assert len( x ) == 1
            x, = x
        if( x.ndim == 2 ):
            t = ( 0, )
            for _x in x:
                t = np.add( t, cls.sufficientStats( _x, forPost=forPost ) )
            return t
        assert isinstance( x, np.ndarray ) and x.ndim == 1
        ( t1, ) = Categorical.standardToNat( x )
        return ( t1, )

    @classmethod
    @checkExpFamArgs
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        ( alpha, ) = params if params is not None else cls.natToStandard( *natParams )
        A1 = gammaln( alpha ).sum()
        A2 = -gammaln( alpha.sum() )
        if( split ):
            return A1, A2
        return A1 + A2

    ##########################################################################

    @classmethod
    @checkExpFamArgs
    def sample( cls, params=None, natParams=None, D=None, size=1, ravel=False ):
        # Sample from P( x | Ѳ; α )
        if( params is not None ):
            if( not isinstance( params, tuple ) or \
                not isinstance( params, list ) ):
                params = ( params, )
        elif( natParams is None ):
            assert D is not None
            params = ( np.ones( D ), )

        ( alpha, ) = params if params is not None else cls.natToStandard( *natParams )
        ans = dirichlet.rvs( alpha=alpha, size=size )
        return ans

    ##########################################################################

    @classmethod
    @checkExpFamArgs
    @multiSampleLikelihood
    def log_likelihood( cls, x, params=None, natParams=None, ravel=False ):
        # Compute P( x | Ѳ; α )
        ( alpha, ) = params if params is not None else cls.natToStandard( *natParams )
        if( isinstance( x, tuple ) ):
            assert len( x ) == 1
            x, = x
        assert isinstance( x, np.ndarray ) and x.ndim == 1
        return dirichlet.logpdf( x, alpha=alpha )
