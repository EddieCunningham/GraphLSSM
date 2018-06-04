from GenModels.GM.Distributions import ExponentialFam
from GenModels.GM.Distributions import NormalInverseWishart, MatrixNormalInverseWishart
from GenModels.GM.States.StandardStates.LDSState import LDSState
from GenModels.GM.Distributions import Normal, Regression
import numpy as np

__all__ = [ 'LDSMNIWPrior' ]

class LDSMNIWPrior( ExponentialFam ):

    # This class is a distribution over P( Ѳ | α )

    priorClass = None

    def __init__( self, M_trans, V_trans, psi_trans, nu_trans,
                        M_emiss, V_emiss, psi_emiss, nu_emiss,
                        mu_0   , kappa_0, psi_0    , nu_0    , Q0=0, Q1=0, Q2=0, prior=None, hypers=None ):
        super( LDSMNIWPrior, self ).__init__( M_trans, V_trans, psi_trans, nu_trans, Q1,
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
        if( isinstance( mu0, np.ndarray ) and mu0.ndim == 2 ):
            assert A.ndim == 3
            assert sigma.ndim == 3
            assert C.ndim == 3
            assert R.ndim == 3
            assert sigma0.ndim == 3
            return mu0.shape[ 0 ]
        elif( isinstance( mu0, tuple ) ):
            assert len( A ) == len( mu0 )
            assert len( sigma ) == len( mu0 )
            assert len( C ) == len( mu0 )
            assert len( R ) == len( mu0 )
            assert len( sigma0 ) == len( mu0 )
            return len( mu0 )
        return 1

    ##########################################################################

    @classmethod
    def standardToNat( cls, M_trans, V_trans, psi_trans, nu_trans, Q1,
                            M_emiss, V_emiss, psi_emiss, nu_emiss, Q2,
                            mu_0   , kappa_0, psi_0    , nu_0    , Q0 ):
        n1 , n2 , n3 , n4 , n5  = MatrixNormalInverseWishart.standardToNat( M_trans, V_trans, psi_trans, nu_trans, Q1 )
        n6 , n7 , n8 , n9 , n10 = MatrixNormalInverseWishart.standardToNat( M_emiss, V_emiss, psi_emiss, nu_emiss, Q2 )
        n11, n12, n13, n14, n15 = NormalInverseWishart.standardToNat( mu_0, kappa_0, psi_0, nu_0, Q0 )
        return n1, n2, n3, n6, n7, n8, n11, n12, n4, n5, n9, n10, n13, n14, n15
        # return n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15

    @classmethod
    def natToStandard( cls, n1, n2, n3, n6, n7, n8, n11, n12, n4, n5, n9, n10, n13, n14, n15 ):
    # def natToStandard( cls, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15 ):
        M_trans, V_trans, psi_trans, nu_trans, Q1 = MatrixNormalInverseWishart.natToStandard( n1, n2, n3, n4, n5 )
        M_emiss, V_emiss, psi_emiss, nu_emiss, Q2 = MatrixNormalInverseWishart.natToStandard( n6, n7 , n8 , n9 , n10 )
        mu_0   , kappa_0, psi_0    , nu_0    , Q0 = NormalInverseWishart.natToStandard( n11, n12, n13, n14, n15 )
        return M_trans, V_trans, psi_trans, nu_trans, Q1, M_emiss, V_emiss, psi_emiss, nu_emiss, Q2, mu_0, kappa_0, psi_0, nu_0, Q0

    ##########################################################################

    @property
    def constParams( self ):
        return None

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, constParams=None ):
        # Compute T( x )
        if( cls.dataN( x ) > 1 ):
            t = ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
            for A, sigma, C, R, mu0, sigma0 in zip( *x ):
                t = np.add( t, cls.sufficientStats( ( A, sigma, C, R, mu0, sigma0 ) ) )
            return t

        t1, t2, t3, t4, t5, t6, t7, t8 = LDSState.standardToNat( *x )

        A, sigma, C, R, mu0, sigma0 = x
        # t9, t10, t11, t12, t13, t14, t15 = LDSState.log_partition( params=x, split=True )
        # Using these instead because LDSState.log_partition requires data ( should probably find way around that )
        # This is a global factor!!! Should definitely make a seperate class for this
        t9, t10 = Regression.log_partition( params=( A, sigma ), split=True )
        t11, t12 = Regression.log_partition( params=( C, R ), split=True )
        t13, t14, t15 = Normal.log_partition( params=( mu0, sigma0 ), split=True )

        return t1, t2, t3, t4, t5, t6, t7, t8, -t9, -t10, -t11, -t12, -t13, -t14, -t15
        # return t1, t2, t3, -t9, -t10, t4, t5, t6, -t11, -t12, t7, t8, -t13, -t14, -t15

    @classmethod
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( natParams is None )
        M_trans, V_trans, psi_trans, nu_trans, Q1, M_emiss, V_emiss, psi_emiss, nu_emiss, Q2, mu_0, kappa_0, psi_0, nu_0, Q0 = params if params is not None else cls.natToStandard( *natParams )

        A1 = MatrixNormalInverseWishart.log_partition( x=x, params=( M_trans, V_trans, psi_trans, nu_trans, Q1 ), split=split )
        A2 = MatrixNormalInverseWishart.log_partition( x=x, params=( M_emiss, V_emiss, psi_emiss, nu_emiss, Q2 ), split=split )
        A3 = NormalInverseWishart.log_partition( x=x, params=( mu_0, kappa_0, psi_0, nu_0, Q0 ), split=split )
        return A1 + A2 + A3

    ##########################################################################

    @classmethod
    def sample( cls, params=None, natParams=None, size=1 ):
        # Sample from P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )

        M_trans, V_trans, psi_trans, nu_trans, Q1, M_emiss, V_emiss, psi_emiss, nu_emiss, Q2, mu_0, kappa_0, psi_0, nu_0, Q0 = params if params is not None else cls.natToStandard( *natParams )

        A, sigma = MatrixNormalInverseWishart.sample( params=( M_trans, V_trans, psi_trans, nu_trans, Q1 ), size=size )
        C, R = MatrixNormalInverseWishart.sample( params=( M_emiss, V_emiss, psi_emiss, nu_emiss, Q2 ), size=size )
        mu0sigma0 = NormalInverseWishart.sample( params=( mu_0, kappa_0, psi_0, nu_0, Q0 ), size=size )
        mu0, sigma0 = mu0sigma0

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
