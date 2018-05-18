import numpy as np
from GenModels.GM.Distributions.Base import TensorExponentialFam, checkExpFamArgs
from GenModels.GM.Distributions.Dirichlet import Dirichlet

class TensorCategorical( TensorExponentialFam ):

    priorClass = None

    def __init__( self, p=None, prior=None, hypers=None ):
        super( TensorCategorical, self ).__init__( p, prior=prior, hypers=hypers )
        self.Ds = self.p.shape

    ##########################################################################

    @property
    def p( self ):
        return self._params[ 0 ]

    ##########################################################################

    @classmethod
    def dataN( cls, x ):
        return x.shape[ 0 ]

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
        return self.Ds

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, Ds=None, constParams=None, forPost=False ):
        # Compute T( x )
        # Return the indices of chosen values.  Will deal with everything else
        # in combine
        return ( x, )

    @classmethod
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( natParams is None )
        if( split ):
            return ( 0, )
        return 0

    ##########################################################################

    @classmethod
    @checkExpFamArgs( allowNone=True )
    def sample( cls, Ds=None, params=None, natParams=None, size=1, ravel=False ):
        # Sample from P( x | Ѳ; α )
        if( params is None and natParams is None ):
            assert Ds is not None
            assert isinstance( Ds, tuple ) or isinstance( Ds, list )
            p = np.random.random( Ds )
            p /= p.sum()
            ( p, ) = params

        ( p, ) = params if params is not None else cls.natToStandard( *natParams )

        Ds = p.shape
        totalD = np.prod( Ds )

        def placement( val ):
            total = totalD
            ans = np.empty_like( Ds )
            for i, d in enumerate( Ds ):
                total /= d
                ans[ i ] = val // total
                val -= ( val // total ) * total
            return ans

        choice = np.random.choice( totalD, size, p=p.ravel() )
        return np.array( list( map( placement, choice ) ) )

    ##########################################################################

    @classmethod
    @checkExpFamArgs
    def log_likelihood( cls, x, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        ( p, ) = params if params is not None else cls.natToStandard( *natParams )
        assert isinstance( x, np.ndarray ) and x.ndim == 2
        return np.log( p[ [ x[ :, i ] for i in range( len( p.shape ) ) ] ] ).sum()

    ##########################################################################

    @classmethod
    def combine( cls, stat, nat ):
        log_p = nat
        indices = stat
        return log_p[ [ indices[ :, i ] for i in range( len( log_p.shape ) ) ] ].sum()