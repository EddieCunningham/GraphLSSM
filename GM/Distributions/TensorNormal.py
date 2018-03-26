import numpy as np
from Base import TensorExponentialFam
from Normal import Normal
from InverseWishart import InverseWishart
import string
from functools import reduce

_HALF_LOG_2_PI = 0.5 * np.log( 2 * np.pi )

class TensorNormal( TensorExponentialFam ):

    # Just for the moment
    priorClass = None

    def __init__( self, M=None, covs=None, prior=None, hypers=None ):
        super( TensorNormal, self ).__init__( M, covs, prior=prior, hypers=hypers )

    ##########################################################################

    @property
    def M( self ):
        return self._params[ 0 ]

    @property
    def covs( self ):
        return self._params[ 1 ]

    ##########################################################################

    @classmethod
    def dataN( cls, x ):
        return x.shape[ 0 ]

    ##########################################################################

    @classmethod
    def invs( cls, x, k=None ):
        ans = [ np.linalg.inv( _x ) for _x in x ]
        if( k is not None ):
            ans[ 0 ] *= -0.5
        return ans

    @classmethod
    def standardToNat( cls, M, covs ):
        n1 = cls.invs( covs, -0.5 )
        n2 = ( M, *cls.invs( covs ) )
        return n1, n2

    @classmethod
    def natToStandard( cls, n1, n2 ):
        M = n2[ 0 ]
        covs = cls.invs( n1, -0.5 )
        return M, covs

    ##########################################################################

    @property
    def constParams( self ):
        return None

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, constParams=None, forPost=False ):
        # Compute T( x )
        if( x.ndim == 1 ):
            x = x.reshape( ( 1, -1 ) )
        t1 = ( x, x )
        t2 = ( x, )
        if( forPost ):
            # This for when we add to the TNIW natural params
            t3 = ( x.shape[ 0 ], )
            t4 = ( x.shape[ 0 ], )
            t5 = ( x.shape[ 0 ], )
            return t1, t2, t3, t4, t5
        return t1, t2

    @classmethod
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( natParams is None )

        M, covs = params if params is not None else cls.natToStandard( *natParams )

        totalDim = np.prod( [ cov.shape[ 0 ] for cov in covs ] )

        A1 = 0.5 * sum( [ totalDim / cov.shape[ 0 ] * np.linalg.slogdet( cov )[ 1 ] for cov in covs ] )
        A2 = 0.5 * cls.combine( ( M[ None ], M[ None ] ), cls.invs( covs ) )
        log_h = totalDim * _HALF_LOG_2_PI

        if( split ):
            return ( A1, A2, log_h )
        return A1 + A2 + log_h

    ##########################################################################

    @classmethod
    def sample( cls, params=None, natParams=None, Ds=None, size=1 ):
        if( params is None and natParams is None ):
            assert Ds is not None
            assert isinstance( Ds, tuple )
            params = ( np.zeros( Ds ), [ InverseWishart.sample( D=D ) for D in Ds ] )

        # Sample from P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )
        M, covs = params if params is not None else cls.natToStandard( *natParams )

        covChols = [ np.linalg.cholesky( cov ) for cov in covs ]
        shapes = [ cov.shape[ 0 ] for cov in covs ]
        N = len( shapes )
        totalDim = np.prod( [ size ] + shapes )

        X = Normal.sample( D=1, size=totalDim ).reshape( [ size ] + shapes )

        ind1 = string.ascii_letters[ : N ]
        ind2 = string.ascii_letters[ N : N * 2 ]
        t = string.ascii_letters[ N * 2 ]
        contract = t + ind1 + ',' + ','.join( [ a + b for a, b in zip( ind1, ind2 ) ] ) + '->' + t + ind2

        return M + np.einsum( contract, X, *covChols )

    ##########################################################################

    @classmethod
    def log_likelihoodRavel( cls, x, params=None, natParams=None ):
        assert ( params is None ) ^ ( natParams is None )
        M, covs = params if params is not None else cls.natToStandard( *natParams )

        assert x.shape[ 1: ] == M.shape

        fullMu = M.ravel()
        ans = 0.0
        for _x in x:
            fullCov = reduce( lambda x, y: np.kron( x, y ), covs )
            ans += Normal.log_likelihood( _x.ravel(), params=( fullMu, fullCov ) )
        return ans

    @classmethod
    def log_likelihood( cls, x, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )
        M, covs = params if params is not None else cls.natToStandard( *natParams )

        totalDim = np.prod( [ cov.shape[ 0 ] for cov in covs ] )
        covInvs = cls.invs( covs )

        assert x.shape[ 1: ] == M.shape
        dataN = cls.dataN( x )

        statNat = -0.5 * cls.combine( ( x - M, x - M ), covInvs )
        part = -0.5 * sum( [ totalDim / cov.shape[ 0 ] * np.linalg.slogdet( cov )[ 1 ] for cov in covs ] ) + \
               - totalDim * _HALF_LOG_2_PI
        return statNat + part * dataN

    def ilog_likelihood( self, x, expFam=False, ravel=False ):
        if( ravel ):
            return self.log_likelihoodRavel( x, params=self.params )
        return super( TensorNormal, self ).ilog_likelihood( x, expFam=expFam )

    ##########################################################################

    @classmethod
    def marginalizeX1( J11, J12, J22, h1, h2, logZ, D, N, intX1=True ):

        # Integrate exp{ -0.5 * < J11, x1:N, x1:N >
        #                -0.5 * < J22, y1:N, y1:N >
        #                     + < J12, y1:N, x1:N >
        #                     + < h1, x1:N >
        #                     + < h2, y1:N >
        #                     - logZ }
        # over x1:N

        if( intX1 ):
            _J11 = J11.reshape( ( D**N, D**N ) )
            _J12 = J12.reshape( ( D**N, D**N ) )
            _J22 = J22.reshape( ( D**N, D**N ) )
            _h1 = h1.ravel()
            _h2 = h2.ravel()
        else:
            _J11 = J22.reshape( ( D**N, D**N ) )
            _J12 = J12.reshape( ( D**N, D**N ) ).T
            _J22 = J11.reshape( ( D**N, D**N ) )
            _h1 = h2.ravel()
            _h2 = h1.ravel()

        np.linalg.inv( _J11 )
        J11Chol = cho_factor( _J11, lower=True )
        J11Invh1 = cho_solve( J11Chol, _h1 )

        J = _J22 - _J12.T @ cho_solve( J11Chol, _J12 )
        h = _h2 - _J12.T.dot( J11Invh1 )

        logZ = logZ - \
                0.5 * _h1.dot( J11Invh1 ) + \
                np.log( np.diag( J11Chol[ 0 ] ) ).sum() - \
                D**N * _HALF_LOG_2_PI
        return J.reshape( [ D for _ in range( 2 * N ) ] ), \
               h.reshape( [ D for _ in range( N ) ] ), \
               logZ

    @classmethod
    def marginalizeX2( cls, J11, J12, J22, h1, h2, logZ, D, N ):
        return cls.marginalizeX1( J11, J12, J22, h1, h2, logZ, D, N, intX1=False )

    ##########################################################################

    @classmethod
    def combine( cls, stat, nat ):

        N = len( stat ) + len( nat ) - 2

        ind1 = string.ascii_letters[ : N ]
        ind2 = string.ascii_letters[ N : N * 2 ]
        t = string.ascii_letters[ N * 2 ]
        if( len( stat ) == 1 ):
            contract = t + ind1 + ',' + ind2 + ',' + ','.join( [ a + b for a, b in zip( ind1, ind2 ) ] ) + '->'
        else:
            assert len( stat ) == 2
            contract = t + ind1 + ',' + t + ind2 + ',' + ','.join( [ a + b for a, b in zip( ind1, ind2 ) ] ) + '->'

        return np.einsum( contract, *stat, *nat, optimize=( N > 2 ) )
