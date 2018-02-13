import numpy as np
from Base import Exponential
from scipy.stats import invwishart
from scipy.special import multigammaln

class InverseWishart( Exponential ):

    def __init__( self, psi, nu, prior=None, hypers=None ):
        super( InverseWishart, self ).__init__( psi, nu, prior=prior, hypers=hypers )

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

    @classmethod
    def sufficientStats( cls, x, forPost=False ):
        # Compute T( x )

        t1 = np.linalg.inv( x )
        _, t2 = np.linalg.slogdet( x )
        return t1, t2

    @classmethod
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( natParams is None )

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
    def sample( cls, params=None, natParams=None ):
        # Sample from P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )
        psi, nu = params if params is not None else cls.natToStandard( *natParams )
        return invwishart.rvs( df=nu, scale=psi, size=1 )

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )
        # There is a bug in scipy's invwishart.logpdf! Don't use it!
        nat = natParams if natParams is not None else cls.standardToNat( *params )
        stat = cls.sufficientStats( x )
        part = cls.log_partition( x, natParams=nat, split=True )

        return Exponential.log_pdf( nat, stat, part )