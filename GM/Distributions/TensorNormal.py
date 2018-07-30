import numpy as np
from GenModels.GM.Distributions.Base import TensorExponentialFam
from GenModels.GM.Distributions.Normal import Normal
from GenModels.GM.Distributions.InverseWishart import InverseWishart
import string
from functools import reduce

_HALF_LOG_2_PI = 0.5 * np.log( 2 * np.pi )

class TensorNormal( TensorExponentialFam ):

    ## THIS CLASS IS NOT THE GENERAL CASE.  ASSUMES COVARIANCE TENSOR IS THE KRONECKER PRODUCT
    # OF MULTIPLE COVARIANCE MATRICES
    # Originally I thought that you could use different covariance matrices as the parameters
    # like you can in the matrix normal distibution.  But I don't think you can in the general
    # case.  The covariance tensor in the general case I don't think has to be the product of
    # 2d covariance matrices.  In that case, everything gets very computationally expensive
    # and probably isn't worth analyzing at the moment.

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

    @property
    def constParams( self ):
        return None

    @classmethod
    def dataN( cls, x, constParams=None ):
        cls.checkShape( x )
        return x.shape[ 0 ]

    @classmethod
    def unpackSingleSample( cls, x ):
        # Not going to unpack samples
        return x

    @classmethod
    def sampleShapes( cls, Ds ):
        # ( Sample #, dim )
        return tuple( [ None for _ in len( Ds ) + 1 ] )

    def isampleShapes( cls, Ds ):
        return ( None, *Ds )

    @classmethod
    def checkShape( cls, x ):
        assert isinstance( x, np.ndarray )
        assert np.any( np.array( x.shape[ 1: ] ) == 1 ) == False

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
    def realStandardToNat( cls, M, covs ):
        # Avoid this because it is unnecessarily expensive
        cov_invs = [ np.linalg.inv( cov ) for cov in covs ]
        n1 = reduce( lambda x, y: np.kron( x, y ), cov_invs ).reshape( M.shape + M.shape )
        N = len( M.shape )
        ind1 = string.ascii_letters[ : N ]
        ind2 = string.ascii_letters[ N : N * 2 ]
        contract = ind2 + ',' + ind1 + ind2 + '->' + ind1
        n2 = np.einsum( contract, M, n1 )
        return -0.5 * n1, n2

    @classmethod
    def natToStandard( cls, n1, n2 ):
        M = n2[ 0 ]
        covs = cls.invs( n1, -0.5 )
        return M, covs

    @classmethod
    def realNatToStandard( cls, n1, n2 ):
        assert 0, 'Can\'t get back the original matrices!  Pretty sure that there are multiple solutions'

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, constParams=None ):
        # Compute T( x )
        if( x.ndim == 1 ):
            x = x.reshape( ( 1, -1 ) )
        t1 = ( x, x )
        t2 = ( x, )
        return t1, t2

    @classmethod
    def log_partition( cls, x=None, params=None, nat_params=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( nat_params is None )

        M, covs = params if params is not None else cls.natToStandard( *nat_params )

        total_dim = np.prod( [ cov.shape[ 0 ] for cov in covs ] )

        A1 = 0.5 * sum( [ total_dim / cov.shape[ 0 ] * np.linalg.slogdet( cov )[ 1 ] for cov in covs ] )
        A2 = 0.5 * cls.combine( ( M[ None ], M[ None ] ), cls.invs( covs ) )
        log_h = total_dim * _HALF_LOG_2_PI

        if( split ):
            return ( A1, A2, log_h )
        return A1 + A2 + log_h

    @classmethod
    def log_partitionGradient( cls, params=None, nat_params=None, split=False ):
        # Derivative w.r.t. natural params. Also the expected sufficient stat
        assert ( params is None ) ^ ( nat_params is None )

        # Luckily this ends up being easy in the general case
        M, covs = params if params is not None else cls.natToStandard( *nat_params )
        assert 0, 'Wait until prior to figure out how to add in the covs'
        d1 = ( M, M, *covs )
        d2 = M

        return d1, d2

    def _testLogPartitionGradient( self ):
        assert 0

    ##########################################################################

    @classmethod
    def generate( cls, Ds=[ 2, 3 ], size=1 ):
        if( np.any( np.array( Ds ) == 1 ) ):
            assert 0, 'Can\'t have an empty dim'
        params = ( np.zeros( Ds ),[ np.eye( D ) for D in Ds ] )
        samples = cls.sample( params=params, size=size )
        return samples if size > 1 else cls.unpackSingleSample( samples )

    @classmethod
    def sample( cls, params=None, nat_params=None, size=1 ):
        # Sample from P( x | Ѳ; α )
        assert ( params is None ) ^ ( nat_params is None )
        M, covs = params if params is not None else cls.natToStandard( *nat_params )

        cov_chols = [ np.linalg.cholesky( cov ) for cov in covs ]
        shapes = [ cov.shape[ 0 ] for cov in covs ]
        N = len( shapes )
        total_dim = np.prod( [ size ] + shapes )

        X = Normal.generate( D=1, size=total_dim ).reshape( [ size ] + shapes )

        ind1 = string.ascii_letters[ : N ]
        ind2 = string.ascii_letters[ N : N * 2 ]
        t = string.ascii_letters[ N * 2 ]
        contract = t + ind1 + ',' + ','.join( [ a + b for a, b in zip( ind1, ind2 ) ] ) + '->' + t + ind2

        ans = M + np.einsum( contract, X, *cov_chols )
        cls.checkShape( ans )
        return ans

    ##########################################################################

    @classmethod
    def log_likelihoodRavel( cls, x, params=None, nat_params=None ):
        assert ( params is None ) ^ ( nat_params is None )
        M, covs = params if params is not None else cls.natToStandard( *nat_params )

        assert x.shape[ 1: ] == M.shape

        fullMu = M.ravel()
        ans = 0.0
        for _x in x:
            fullCov = reduce( lambda x, y: np.kron( x, y ), covs )
            ans += Normal.log_likelihood( _x.ravel(), params=( fullMu, fullCov ) )
        return ans

    @classmethod
    def log_likelihood( cls, x, params=None, nat_params=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( nat_params is None )
        M, covs = params if params is not None else cls.natToStandard( *nat_params )

        total_dim = np.prod( [ cov.shape[ 0 ] for cov in covs ] )
        cov_invs = cls.invs( covs )

        assert x.shape[ 1: ] == M.shape
        dataN = cls.dataN( x )

        stat_nat = -0.5 * cls.combine( ( x - M, x - M ), cov_invs )
        part = -0.5 * sum( [ total_dim / cov.shape[ 0 ] * np.linalg.slogdet( cov )[ 1 ] for cov in covs ] ) + \
               - total_dim * _HALF_LOG_2_PI
        ans = stat_nat + part * dataN
        return ans

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

        # THERE IS A BUG IN NP.EINSUM AT THE MOMENT!!! KEEP OPTIMIZE OFF
        return np.einsum( contract, *stat, *nat, optimize=False )
