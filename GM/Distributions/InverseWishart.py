import numpy as np
from GenModels.GM.Distributions.Base import ExponentialFam
from scipy.stats import invwishart
from scipy.special import multigammaln
from GenModels.GM.Utility import *

class InverseWishart( ExponentialFam ):

    def __init__( self, psi, nu, prior=None, hypers=None ):
        super( InverseWishart, self ).__init__( psi, nu, prior=prior, hypers=hypers )

    ##########################################################################

    @property
    def psi( self ):
        return self._params[ 0 ]

    @property
    def nu( self ):
        return self._params[ 1 ]

    ##########################################################################

    @classmethod
    def paramShapes( cls, D=None ):
        assert D is not None
        return [ ( D, D ), D ]

    @classmethod
    def inferDims( cls, params=None ):
        assert params is not None
        psi, nu = params
        return { 'D': psi.shape[ 0 ] }

    @classmethod
    def outputShapes( cls, D=None ):
        assert D is not None
        return [ ( D, D ) ]

    ##########################################################################

    @classmethod
    def standardToNat( cls, psi, nu ):
        p = psi.shape[ 0 ]
        n1 = -0.5 * psi
        n2 = -0.5 * ( nu + p + 1 )
        return n1, n2

    @classmethod
    def natToStandard( cls, n1, n2 ):
        p = n1.shape[ 0 ]
        psi = -2 * n1
        nu = -2 * n2 - p - 1
        return psi, nu

    ##########################################################################

    @property
    def constParams( self ):
        return None

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, constParams=None, forPost=False ):
        # Compute T( x )

        if( cls.dataN( x ) > 1 ):
            t1 = np.linalg.inv( x ).sum( axis=0 )
            t2 = np.linalg.slogdet( x )[ 1 ].sum()
        else:
            t1 = np.linalg.inv( x )
            t2 = np.linalg.slogdet( x )[ 1 ]
        return t1, t2

    @classmethod
    @checkExpFamArgs
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )

        # its just easier to use the standard params
        psi, nu = params if params is not None else cls.natToStandard( *natParams )
        p = psi.shape[ 0 ]

        A1 = -nu / 2 * np.linalg.slogdet( psi )[ 1 ]
        A2 = multigammaln( nu / 2, p )
        A3 = nu * p / 2 * np.log( 2 )

        if( split ):
            return A1, A2, A3
        return A1 + A2 + A3

    ##########################################################################

    @classmethod
    @fullSampleSupport
    @checkExpFamArgs( allowNone=True )
    def sample( cls, params=None, natParams=None ):
        # Sample from P( x | Ѳ; α )
        psi, nu = params if params is not None else cls.natToStandard( *natParams )
        samples = invwishart.rvs( df=nu, scale=psi )
        return samples

    ##########################################################################

    @classmethod
    @fullLikelihoodSupport
    @checkExpFamArgs
    def log_likelihood( cls, x, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        # There is a bug in scipy's invwishart.logpdf! Don't use it!
        return cls.log_likelihoodExpFam( x, params=params, natParams=natParams )
