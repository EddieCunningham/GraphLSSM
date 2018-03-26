import numpy as np
from Base import TensorExponentialFam
from TensorNormal import TensorNormal
from Normal import Normal
import string
from functools import reduce
from Regression import Regression

_HALF_LOG_2_PI = 0.5 * np.log( 2 * np.pi )

class TensorRegression( TensorExponentialFam ):

    # Just for the moment
    priorClass = None

    def __init__( self, A=None, sigma=None, prior=None, hypers=None ):
        super( TensorRegression, self ).__init__( A, sigma, prior=prior, hypers=hypers )

    ##########################################################################

    @property
    def A( self ):
        return self._params[ 0 ]

    @property
    def sigma( self ):
        return self._params[ 1 ]

    ##########################################################################

    @classmethod
    def dataN( cls, x ):
        xs, ys = x
        return ys.shape[ 0 ]

    ##########################################################################

    @classmethod
    def standardToNat( cls, A, sigma ):

        sigInv = np.linalg.inv( sigma )

        n1 = ( -0.5 * sigInv, A, A )
        n2 = ( sigInv, A )
        n3 = ( -0.5 * sigInv, )

        return n1, n2, n3

    @classmethod
    def natToStandard( cls, n1, n2, n3 ):

        sigInv, A = n2
        sigma = np.linalg.inv( sigInv )

        return A, sigma

    ##########################################################################

    @property
    def constParams( self ):
        return None

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, constParams=None, forPost=False ):
        # Compute T( x )

        xs, y = x

        assert ( isinstance( xs, tuple ) or isinstance( xs, list ) ) and isinstance( y, np.ndarray )
        assert y.ndim == 2
        for _x in xs:
            assert _x.shape == y.shape

        t1 = ( *xs, *xs )
        t2 = ( *xs, y )
        t3 = ( y, y )

        if( forPost ):
            # This for when we add to the MNIW natural params
            t4 = ( x.shape[ 0 ], )
            t5 = ( x.shape[ 0 ], )
            return t1, t2, t3, t4, t5
        return t1, t2, t3


    @classmethod
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( natParams is None )
        A, sigma = params if params is not None else cls.natToStandard( *natParams )

        p = sigma.shape[ 0 ]

        A1 = 0.5 * np.linalg.slogdet( sigma )[ 1 ]
        A2 = p / 2 * np.log( 2 * np.pi )

        if( split ):
            return A1, A2
        return A1 + A2

    ##########################################################################

    @classmethod
    def sample( cls, xs=None, params=None, natParams=None, size=1 ):
        # Sample from P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )
        A, sigma = params if params is not None else cls.natToStandard( *natParams )

        D = sigma.shape[ 0 ]
        N = len( A.shape )
        if( xs is None ):
            xs = [ np.random.random( ( size, D ) ) for _ in range( N - 1 ) ]
            returnBoth = True
        else:
            returnBoth = False

        ind = string.ascii_letters[ :N ]
        tInd = string.ascii_letters[ N ]

        contract = ind + ',' + tInd + ( ',' + tInd ).join( [ l for l in ind[ 1: ] ] ) + '->' + tInd + ind[ 0 ]
        mus = np.einsum( contract, A, *xs, optimize=( N > 2 ) )

        ys = np.array( [ Normal.sample( params=( mu, sigma ) ) for mu in mus ] )

        return xs, ys if returnBoth else ys

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )

        A, sigma = params if params is not None else cls.natToStandard( *natParams )

        xs, ys = x
        assert ( isinstance( xs, tuple ) or isinstance( xs, list ) ) and isinstance( ys, np.ndarray )
        assert len( xs ) == len( A.shape ) - 1 and ys.ndim == 2
        for x in xs:
            assert isinstance( x, np.ndarray ) and x.shape == ys.shape

        N = len( A.shape )
        ind = string.ascii_letters[ :N ]
        tInd = string.ascii_letters[ N ]

        contract = ind + ',' + tInd + ( ',' + tInd ).join( [ l for l in ind[ 1: ] ] ) + '->' + tInd + ind[ 0 ]
        mus = np.einsum( contract, A, *xs, optimize=( N > 2 ) )

        return sum( [ Normal.log_likelihood( y, params=( mu, sigma ) ) for mu, y in zip( mus, ys ) ] )

    ##########################################################################

    @classmethod
    def combine( cls, stat, nat ):
        # Just have special cases

        D = 2
        ind1 = string.ascii_letters[ :D ]

        if( len( nat ) == 1 ):
            tInd = string.ascii_letters[ D ]
            xInd = tInd + ( ',' + tInd + '' ).join( ind1 )
            contract = ind1 + ',' + xInd + '->'
        elif( len( nat ) == 2 ):
            N = len( stat ) - 1
            ind2 = ind1[ 1 ] + string.ascii_letters[ D: D + N ]
            tInd = string.ascii_letters[ D + N ]
            xInd = tInd + ( ',' + tInd + '' ).join( [ l for l in ind2[ 1: ] + ind1[ 0 ] ] )
            contract = ind1 + ',' + ind2 + ',' + xInd + '->'
        elif( len( nat ) == 3 ):
            N = len( stat ) // 2
            assert len( stat ) / 2 == N
            ind2 = ind1[ 0 ] + string.ascii_letters[ D: D + N ]
            ind3 = ind1[ 1 ] + string.ascii_letters[ D + N: D + 2 * N ]
            tInd = string.ascii_letters[ D + 2 * N ]
            xInd = tInd + ( ',' + tInd + '' ).join( [ l for l in ind2[ 1: ] + ind3[ 1: ] ] )
            contract = ind1 + ',' + ind2 + ',' + ind3 + ',' + xInd + '->'

        return np.einsum( contract, *nat, *stat )
