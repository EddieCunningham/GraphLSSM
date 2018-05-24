from GenModels.GM.Distributions import ExponentialFam
from GenModels.GM.Distributions import NormalInverseWishart, MatrixNormalInverseWishart
import numpy as np

__all__ = [ 'LDSModel' ]

class LDSModel( ExponentialFam ):

    # This class is a distribution over P( Ѳ | α )

    priorClass = None

    def __init__( self, M_trans, V_trans, psi_trans, nu_trans,
                        M_emiss, V_emiss, psi_emiss, nu_emiss,
                        mu_0   , kappa_0, psi_0    , nu_0    , Q0=0, Q1=0, Q2=0, prior=None, hypers=None ):
        super( LDSModel, self ).__init__( M_trans, V_trans, psi_trans, nu_trans, Q1,
                                          M_emiss, V_emiss, psi_emiss, nu_emiss, Q2,
                                          mu_0   , kappa_0, psi_0    , nu_0    , Q0, prior=prior, hypers=hypers )

    ##########################################################################

    @property
    def M_trans( self ):
        return self._params[ 0 ]

    @property
    def V_trans( self ):
        return self._params[ 1 ]

    @property
    def psi_trans( self ):
        return self._params[ 2 ]

    @property
    def nu_trans( self ):
        return self._params[ 3 ]

    @property
    def M_emiss( self ):
        return self._params[ 5 ]

    @property
    def V_emiss( self ):
        return self._params[ 6 ]

    @property
    def psi_emiss( self ):
        return self._params[ 7 ]

    @property
    def nu_emiss( self ):
        return self._params[ 8 ]

    @property
    def mu_0( self ):
        return self._params[ 10 ]

    @property
    def kappa_0( self ):
        return self._params[ 11 ]

    @property
    def psi_0( self ):
        return self._params[ 12 ]

    @property
    def nu_0( self ):
        return self._params[ 13 ]

    ##########################################################################

    @classmethod
    def dataN( cls, x ):
        A, sigma, C, R, mu0, sigma0 = x
        if( mu0.ndim == 1 ):
            return 1
        return mu0.shape[ 0 ]

    ##########################################################################

    @classmethod
    def standardToNat( cls, M_trans, V_trans, psi_trans, nu_trans, Q1,
                            M_emiss, V_emiss, psi_emiss, nu_emiss, Q2,
                            mu_0   , kappa_0, psi_0    , nu_0    , Q0 ):
        n1 , n2 , n3 , n4 , n5  = MatrixNormalInverseWishart.standardToNat( M_trans, V_trans, psi_trans, nu_trans, Q1 )
        n6 , n7 , n8 , n9 , n10 = MatrixNormalInverseWishart.standardToNat( M_emiss, V_emiss, psi_emiss, nu_emiss, Q2 )
        n11, n12, n13, n14, n15 = NormalInverseWishart.standardToNat( mu_0, kappa_0, psi_0, nu_0, Q0 )
        return n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15

    @classmethod
    def natToStandard( cls, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15 ):
        M_trans, V_trans, psi_trans, nu_trans, Q1 = MatrixNormalInverseWishart.natToStandard( n1, n2, n3, n4, n5 )
        M_emiss, V_emiss, psi_emiss, nu_emiss, Q2 = MatrixNormalInverseWishart.natToStandard( n6 , n7 , n8 , n9 , n10 )
        mu_0   , kappa_0, psi_0    , nu_0    , Q0 = NormalInverseWishart.natToStandard( n11, n12, n13, n14, n15 )
        return M_trans, V_trans, psi_trans, nu_trans, Q1, M_emiss, V_emiss, psi_emiss, nu_emiss, Q2, mu_0, kappa_0, psi_0, nu_0, Q0

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

        t1, t2, t3 = LDSState.standardToNat( *x )
        t4, t5, t6 = LDSState.log_partition( params=x, split=True )
        return t1, t2, t3, -t4, -t5, -t6

    @classmethod
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( natParams is None )
        alpha_0, alpha_pi, alpha_L = params if params is not None else cls.natToStandard( *natParams )
        A1 = NormalInverseWishart.log_partition( params=alpha_0, split=split )
        A2 = MatrixNormalInverseWishart.log_partition( params=alpha_pi, split=split )
        A3 = MatrixNormalInverseWishart.log_partition( params=alpha_L, split=split )
        return A1 + A2 + A3

    ##########################################################################

    @classmethod
    def sample( cls, params=None, natParams=None, size=1 ):
        # Sample from P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )

        M_trans, V_trans, psi_trans, nu_trans, Q1, M_emiss, V_emiss, psi_emiss, nu_emiss, Q2, mu_0, kappa_0, psi_0, nu_0, Q0 = params if params is not None else cls.natToStandard( *natParams )

        A, sigma = MatrixNormalInverseWishart.sample( params=( M_trans, V_trans, psi_trans, nu_trans, Q1 ), size=size )
        C, R = MatrixNormalInverseWishart.sample( params=( M_emiss, V_emiss, psi_emiss, nu_emiss, Q2 ), size=size )
        mu0, sigma0 = NormalInverseWishart.sample( params=( mu_0, kappa_0, psi_0, nu_0, Q0 ), size=size )

        return A, sigma, C, R, mu0, sigma0

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )

        A, sigma, C, R, mu0, sigma0 = x
        M_trans, V_trans, psi_trans, nu_trans, Q1, M_emiss, V_emiss, psi_emiss, nu_emiss, Q2, mu_0, kappa_0, psi_0, nu_0, Q0 = params if params is not None else cls.natToStandard( *natParams )

        ans = MatrixNormalInverseWishart.log_likelihood( ( A, sigma ), params=( M_trans, V_trans, psi_trans, nu_trans, Q1 ) )
        ans += MatrixNormalInverseWishart.log_likelihood( ( C, R ), params=( M_emiss, V_emiss, psi_emiss, nu_emiss, Q2 ) )
        ans += NormalInverseWishart.log_likelihood( ( mu0, sigma0 ), params=( mu_0, kappa_0, psi_0, nu_0, Q0 ) )

        return ans
