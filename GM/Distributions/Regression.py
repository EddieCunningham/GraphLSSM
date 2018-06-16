import numpy as np
from GenModels.GM.Distributions.Base import ExponentialFam
from GenModels.GM.Distributions.Normal import Normal
from GenModels.GM.Utility import invPsd

__all__ = [ 'Regression' ]

def definePrior():
    # Doing this to get around circular dependency
    from GenModels.GM.Distributions.MatrixNormalInverseWishart import MatrixNormalInverseWishart
    Regression.priorClass = MatrixNormalInverseWishart

class Regression( ExponentialFam ):

    priorClass = None

    def __init__( self, A=None, sigma=None, prior=None, hypers=None ):
        definePrior()
        super( Regression, self ).__init__( A, sigma, prior=prior, hypers=hypers )

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
        cls.checkShape( x )
        xs, ys = x
        if( xs.ndim == 2 ):
            return xs.shape[ 0 ]
        return 1

    @classmethod
    def unpackSingleSample( cls, x ):
        xs, ys = x
        return xs[ 0 ], ys[ 0 ]

    @classmethod
    def sampleShapes( cls ):
        # ( ( Sample #, dim1 ), ( Sample #, dim2 ) )
        return ( ( None, None ), ( None, None ) )

    def isampleShapes( cls ):
        return ( ( None, self.A.shape[ 0 ] ), ( None, self.A.shape[ 1 ] ) )

    @classmethod
    def checkShape( cls, x ):
        assert isinstance( x, tuple )
        x, y = x
        assert isinstance( x, np.ndarray ) and isinstance( y, np.ndarray )
        if( x.ndim == 2 ):
            assert y.ndim == 2
            assert x.shape[ 0 ] == y.shape[ 0 ]
        else:
            assert x.ndim == 1 and y.ndim == 1

    ##########################################################################

    @classmethod
    def standardToNat( cls, A, sigma ):
        sigInv = invPsd( sigma )

        n1 = -0.5 * sigInv
        n2 = -0.5 * A.T @ sigInv @ A
        n3 = A.T @ sigInv

        return n1, n2, n3

    @classmethod
    def natToStandard( cls, n1, n2, n3 ):
        sigma = -0.5 * np.linalg.inv( n1 )
        A = sigma @ n3.T
        return A, sigma

    ##########################################################################

    @property
    def constParams( self ):
        return None

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, constParams=None ):
        # Compute T( x )
        x, y = x
        assert isinstance( x, np.ndarray )
        assert isinstance( y, np.ndarray )

        if( x.ndim == 1 ):
            # Only 1 point was passed in
            x = x.reshape( ( 1, -1 ) )
            assert y.ndim == 1 or y.ndim == 2
            t2 = x.T.dot( x )

            if( y.ndim == 1 ):
                # 1 measurement for x
                y = y.reshape( ( 1, -1 ) )
                t1 = y.T.dot( y )
                t3 = x.T.dot( y )
            else:
                # Multiple measurements for x
                t2 *= y.shape[ 0 ]
                t1 = np.einsum( 'i,mj->ij', x, y )
                t3 = np.einsum( 'mi,mj->ij', y, y )
        else:
            # Multiple data points were passed in
            t2 = x.T.dot( x )

            if( y.ndim == 3 ):
                # Multiple measurements of y per x
                assert x.shape[ 0 ] == y.shape[ 1 ]
                t2 *= y.shape[ 0 ]
                t1 = np.einsum( 'mti,mtj->ij', y, y )
                t3 = np.einsum( 'ti,mtj->ij', x, y )
            elif( y.ndim == 2 ):
                # One measurement of y per x
                assert x.shape[ 0 ] == y.shape[ 0 ]
                t1 = y.T.dot( y )
                t3 = x.T.dot( y )
            else:
                assert 0, 'Invalid dim'

        return t1, t2, t3

    @classmethod
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( natParams is None )
        A, sigma = params if params is not None else cls.natToStandard( *natParams )
        if( natParams is not None ):
            n1, n2, n3 = cls.standardToNat( A, sigma )
            assert np.allclose( n1, natParams[ 0 ] )
            assert np.allclose( n2, natParams[ 1 ] )
            assert np.allclose( n3, natParams[ 2 ] )

        n = sigma.shape[ 0 ]

        A1 = 0.5 * np.linalg.slogdet( sigma )[ 1 ]
        A2 = n / 2 * np.log( 2 * np.pi )

        if( split ):
            return A1, A2
        return A1 + A2

    @classmethod
    def log_partitionGradient( cls, params=None, natParams=None ):
        # ?? Not sure what to do considering one of the natural parameters is redundant
        assert 0, 'Just don\'t call this.  Not sure what to do at the moment'

    def _testLogPartitionGradient( self ):
        pass

    ##########################################################################

    @classmethod
    def generate( cls, D_in=3, D_out=2, size=1 ):
        params = ( np.zeros( ( D_out, D_in ) ), np.eye( D_out ) )
        samples = cls.sample( params=params, size=size )
        return samples if size > 1 else cls.unpackSingleSample( samples )

    @classmethod
    def sample( cls, x=None, params=None, natParams=None, size=1 ):
        # Sample from P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )
        A, sigma = params if params is not None else cls.natToStandard( *natParams )
        D = A.shape[ 1 ]
        if( x is None ):
            x = np.array( [ Normal.unpackSingleSample( Normal.sample( params=( np.zeros( D ), np.eye( D ) ), size=1 ) ) for _ in range( size ) ] )
            y = np.array( [ Normal.unpackSingleSample( Normal.sample( params=( A.dot( _x ), sigma ), size=1 ) ) for _x in x ] )
            ans = ( x, y )
            cls.checkShape( ans )
            return ans

        ans = Normal.sample( params=( A.dot( x ), sigma ), size=size )
        Normal.checkShape( ans )
        return ans

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )
        A, sigma = params if params is not None else cls.natToStandard( *natParams )

        x, y = x
        assert x.shape[ 0 ] == y.shape[ 0 ]
        if( x.ndim != 1 ):
            return sum( [ Normal.log_likelihood( _y, params=( A.dot( _x ), sigma ) ) for _x, _y in zip( x, y ) ] )
        return Normal.log_likelihood( y, params=( A.dot( x ), sigma ) )

    ##########################################################################

    @classmethod
    def toJoint( cls, u=None, params=None, natParams=None ):
        # Given the parameters for P( y | A, sigma, x, u ),
        # return the natural parameters for P( [ y, x ] | A, sigma, u )
        assert ( params is None ) ^ ( natParams is None )
        n1, n2, n3 = natParams if natParams is not None else cls.standardToNat( *params )

        _n1 = np.block( [ n1, 0.5 * n3.T ], [ 0.5 * n3, n2 ] )
        if( u is None ):
            _n2 = np.zeros( n1.shape[ 0 ] )
        else:
            _n2 = np.hstack( ( -2 * n1.dot( u ), -n3.dot( u ) ) )

        return _n1, _n2

    @classmethod
    def toInverse( cls, y, u=None, params=None, natParams=None ):
        # Given the parameters for P( y | A, sigma, x, u ),
        # return the natural parameters for P( x | A, sigma, y, u )
        assert ( params is None ) ^ ( natParams is None )
        n1, n2, n3 = natParams if natParams is not None else cls.standardToNat( *params )

        _n1 = n2
        if( u is None ):
            _n2 = n3.dot( y )
        else:
            _n2 = n3.dot( y - u )

        return _n1, _n2

    ##########################################################################

    @classmethod
    def maxLikelihoodFromStats( cls, t1, t2, t3 ):
        yyT, xyT, xxT = t1, t2, t3
        A = np.linalg.solve( xxT, xyT.T )
        sigma = t1 + A @ xxT @ A.T - 2 * xyT.T @ A.T
        return A, sigma

    @classmethod
    def maxLikelihood( cls, x ):
        t1, t2, t3 = cls.sufficientStats( x )
        return cls.maxLikelihoodFromStats( t1, t2, t3 )
