from GenModels.GM.Distributions import ExponentialFam
from GenModels.GM.Distributions import Dirichlet, TransitionDirichletPrior
from GenModels.GM.States.StandardStates.HMMState import HMMState
import numpy as np

__all__ = [ 'HMMDirichletPrior' ]

class HMMDirichletPrior( ExponentialFam ):

    # This class just puts dirichlet priors over each row in the initial distribution,
    # transition distribution and emission distribution.  So this class is a distribution
    # over P( Ѳ | α )

    priorClass = None

    def __init__( self, alpha_0, alpha_pi, alpha_L, prior=None, hypers=None ):
        super( HMMDirichletPrior, self ).__init__( alpha_0, alpha_pi, alpha_L, prior=prior, hypers=hypers )

    ##########################################################################

    @property
    def alpha_0( self ):
        return self._params[ 0 ]

    @property
    def alpha_pi( self ):
        return self._params[ 1 ]

    @property
    def alpha_L( self ):
        return self._params[ 2 ]

    ##########################################################################

    @classmethod
    def dataN( cls, x ):
        pi_0, pi, L = x
        if( pi_0.ndim == 1 ):
            return 1
        return pi_0.shape[ 0 ]

    ##########################################################################

    @classmethod
    def standardToNat( cls, alpha_0, alpha_pi, alpha_L ):
        n1, = Dirichlet.standardToNat( alpha_0 )
        n2, = TransitionDirichletPrior.standardToNat( alpha_pi )
        n3, = TransitionDirichletPrior.standardToNat( alpha_L )
        return n1, n2, n3

    @classmethod
    def natToStandard( cls, n1, n2, n3 ):
        alpha_0, = Dirichlet.natToStandard( n1 )
        alpha_pi, = TransitionDirichletPrior.natToStandard( n2 )
        alpha_L, = TransitionDirichletPrior.natToStandard( n3 )
        return alpha_0, alpha_pi, alpha_L

    ##########################################################################

    @property
    def constParams( self ):
        return None

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, constParams=None ):
        # Compute T( x )
        if( cls.dataN( x ) > 1 ):
            t = ( 0, 0, 0, 0, 0, 0 )
            for pi_0, pi, L in zip( *x ):
                t = np.add( t, cls.sufficientStats( ( pi_0, pi, L ) ) )
            return t

        t1, t2, t3 = HMMState.standardToNat( *x )
        t4, t5, t6 = HMMState.log_partition( params=x, split=True )
        return t1, t2, t3, -t4, -t5, -t6

    @classmethod
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( natParams is None )
        alpha_0, alpha_pi, alpha_L = params if params is not None else cls.natToStandard( *natParams )
        A1 = Dirichlet.log_partition( params=( alpha_0, ), split=split )
        A2 = TransitionDirichletPrior.log_partition( params=( alpha_pi, ), split=split )
        A3 = TransitionDirichletPrior.log_partition( params=( alpha_L, ), split=split )
        return A1 + A2 + A3

    ##########################################################################

    @classmethod
    def sample( cls, params=None, natParams=None, size=1 ):
        # Sample from P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )

        alpha_0, alpha_pi, alpha_L = params if params is not None else cls.natToStandard( *natParams )

        pi_0 = Dirichlet.sample( params=( alpha_0, ), size=size )
        pi = TransitionDirichletPrior.sample( params=( alpha_pi, ), size=size )
        L = TransitionDirichletPrior.sample( params=( alpha_L, ), size=size )

        return pi_0.squeeze(), pi.squeeze(), L.squeeze()

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )

        pi_0, pi, L = x
        alpha_0, alpha_pi, alpha_L = params if params is not None else cls.natToStandard( *natParams )

        ans = Dirichlet.log_likelihood( pi_0, params=( alpha_0, ) )
        ans += TransitionDirichletPrior.log_likelihood( pi, params=( alpha_pi, ) )
        ans += TransitionDirichletPrior.log_likelihood( L, params=( alpha_L, ) )

        return ans
