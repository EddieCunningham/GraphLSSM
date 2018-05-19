import numpy as np
from GenModels.GM.Distributions.Base import ExponentialFam
from scipy.stats import dirichlet
from scipy.special import gammaln
from GenModels.GM.Distributions.Categorical import Categorical
from GenModels.GM.Utility import *

class Dirichlet( ExponentialFam ):

    def __init__( self, alpha=None, prior=None, hypers=None ):
        super( Dirichlet, self ).__init__( alpha, prior=prior, hypers=hypers )

    ##########################################################################

    @property
    def alpha( self ):
        return self._params[ 0 ]

    ##########################################################################

    @classmethod
    def paramShapes( cls, D=None ):
        assert D is not None
        return [ ( D, ) ]

    @classmethod
    def inferDims( cls, params=None ):
        assert params is not None
        alpha, = params
        return { 'D': alpha.shape[ 0 ] }

    @classmethod
    def outputShapes( cls, D=None ):
        assert D is not None
        return [ ( D, ) ]

    ##########################################################################

    @classmethod
    def easyParamSample( cls, D=None ):
        assert D is not None
        return ( np.ones( D ), )

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
    @fullSampleSupport
    @checkExpFamArgs( allowNone=True )
    def sample( cls, params=None, natParams=None ):
        # Sample from P( x | Ѳ; α )
        ( alpha, ) = params if params is not None else cls.natToStandard( *natParams )
        ans = dirichlet.rvs( alpha=alpha )[ 0 ]
        return ans

    ##########################################################################

    @classmethod
    @fullLikelihoodSupport
    @checkExpFamArgs
    def log_likelihood( cls, x, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        ( alpha, ) = params if params is not None else cls.natToStandard( *natParams )
        if( isinstance( x, tuple ) ):
            assert len( x ) == 1
            x, = x
        assert isinstance( x, np.ndarray ) and x.ndim == 1
        return dirichlet.logpdf( x, alpha=alpha )
