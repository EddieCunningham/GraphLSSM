import numpy as np
from GenModels.GM.Distributions.Base import ExponentialFam
from GenModels.GM.Utility import *

__all__ = [ 'Categorical' ]

def definePrior():
    # Because of circular dependency
    from GenModels.GM.Distributions.Dirichlet import Dirichlet
    Categorical.priorClass = Dirichlet

class Categorical( ExponentialFam ):

    priorClass = None

    def __init__( self, p=None, prior=None, hypers=None ):
        definePrior()
        super( Categorical, self ).__init__( p, prior=prior, hypers=hypers )
        self.D = self.p.shape[ 0 ]

    ##########################################################################

    @property
    def p( self ):
        return self._params[ 0 ]

    ##########################################################################

    @classmethod
    def paramShapes( cls, D=None ):
        assert D is not None
        return [ ( D, ) ]

    @classmethod
    def inferDims( cls, params=None ):
        assert params is not None
        p, = params
        return { 'D': p.shape[ 0 ] }

    @classmethod
    def outputShapes( cls, D=None ):
        assert D is not None
        return [ ( 1, ) ]

    ##########################################################################

    @classmethod
    def easyParamSample( cls, D=None ):
        assert D is not None
        return ( np.ones( D ) / D, )

    @classmethod
    @fullSampleSupport
    def paramSample( cls, priorParams=None, **D ):
        # Sample from P( Ѳ; α )
        if( cls.priorClass == None ):
            return cls.easyParamSample( **D )
        return ( cls.priorClass.sample( priorParams, **D ), )

    @fullSampleSupport
    def iparamSample( self ):
        return ( self.prior.isample(), )

    ##########################################################################

    @classmethod
    def standardToNat( cls, p ):
        n = np.log( p )
        return ( n, )

    @classmethod
    def natToStandard( cls, n ):
        p = np.exp( n )
        return ( p, )

    ##########################################################################

    @property
    def constParams( self ):
        return self.D

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, constParams=None, forPost=False ):
    # def sufficientStats( cls, x, D=None, constParams=None, forPost=False ):
        # Compute T( x )
        assert isinstance( x, np.ndarray ) and x.ndim == 1, x
        D = constParams
        assert D is not None
        t1 = np.bincount( x, minlength=D )
        return ( t1, )

    @classmethod
    @checkExpFamArgs
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        if( split ):
            return ( 0, )
        return 0

    ##########################################################################

    @classmethod
    @fullSampleSupport
    @checkExpFamArgs( allowNone=True )
    def sample( cls, params=None, natParams=None ):
        # Sample from P( x | Ѳ; α )
        ( p, ) = params if params is not None else cls.natToStandard( *natParams )
        assert p.ndim == 1, p
        return np.array( np.random.choice( p.shape[ 0 ], p=p ) )

    ##########################################################################

    @classmethod
    @fullLikelihoodSupport
    @checkExpFamArgs
    def log_likelihood( cls, x, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        ( p, ) = params if params is not None else cls.natToStandard( *natParams )
        return np.log( p[ x ] )
