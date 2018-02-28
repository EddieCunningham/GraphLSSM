import numpy as np
from Base import TensorExponentialFam
from Normal import Normal
import string
from functools import reduce

_HALF_LOG_2_PI = 0.5 * np.log( 2 * np.pi )

class TensorNormal( TensorExponentialFam ):


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

    @classmethod
    def sufficientStats( cls, x, forPost=False ):
        # Compute T( x )
        if( x.ndim == 1 ):
            x = x.reshape( ( 1, -1 ) )
        t1 = ( x, x )
        t2 = ( x, )
        if( forPost ):
            # This for when we add to the NIW natural params
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
        A2 = 0.5 * TensorExponentialFam.combine( ( M[ None ], M[ None ] ), cls.invs( covs ), size=1 )
        log_h = totalDim * _HALF_LOG_2_PI

        if( split ):
            return ( A1, A2, log_h )
        return A1 + A2 + log_h

    ##########################################################################

    @classmethod
    def sample( cls, params=None, natParams=None, size=1 ):
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
        t = string.ascii_letters[ N*2 ]
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

        statNat = -0.5 * TensorExponentialFam.combine( ( x - M, x - M ), covInvs, size=x.shape[ 0 ] )
        part = -0.5 * sum( [ totalDim / cov.shape[ 0 ] * np.linalg.slogdet( cov )[ 1 ] for cov in covs ] ) + \
               -0.5 * totalDim * np.log( 2 * np.pi )
        return statNat + part * dataN


    def ilog_likelihood( self, x, expFam=False, ravel=False ):
        if( ravel ):
            return self.log_likelihoodRavel( x, params=self.params )
        return super( TensorNormal, self ).ilog_likelihood( x, expFam=expFam )

    ##########################################################################

    def likelihoodDefinitionTestNoPartition( self ):
        x = self.isample( size=10 )
        ans1 = self.ilog_likelihood( x )
        trueAns1 = self.ilog_likelihood( x, ravel=True )

        x = self.isample( size=10 )
        ans2 = self.ilog_likelihood( x )
        trueAns2 = self.ilog_likelihood( x, ravel=True )
        assert np.isclose( ans1 - ans2, trueAns1 - trueAns2 ), ( ans1 - ans2 ) - ( trueAns1 - trueAns2 )

    def likelihoodNoPartitionTest( self ):
        self.likelihoodDefinitionTestNoPartition()
        super( TensorNormal, self ).likelihoodNoPartitionTest()

    def likelihoodDefinitionTest( self ):
        x = self.isample( size=10 )
        ans1 = self.ilog_likelihood( x )
        ans2 = self.ilog_likelihood( x, ravel=True )
        assert np.isclose( ans1, ans2 ), ans1 - ans2

    def likelihoodTest( self ):
        self.likelihoodDefinitionTest()
        super( TensorNormal, self ).likelihoodTest()