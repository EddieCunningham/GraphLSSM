import numpy as np
from GenModels.GM.Distributions.Base import Conjugate
from GenModels.GM.Distributions.TensorCategorical import TensorCategorical
from GenModels.GM.Distributions.Dirichlet import Dirichlet

class Transition( TensorCategorical ):
    # This class is basically just a tensor categorial, except we
    # normalize over the last index of p

    def __init__( self, p=None, prior=None, hypers=None ):
        super( Transition ).__init__( self, p=p, prior=prior, hypers=hypers )

    ##########################################################################

    @classmethod
    def sample( cls, x=None, params=None, natParams=None, size=1, ravel=False ):
        # Sample from P( x | Ѳ; α )

        if( x is None ):
            return super( Transition, cls ).sample( params=params, natParams=natParams, size=size )

        assert ( params is None ) ^ ( natParams is None )
        ( p, ) = params if params is not None else cls.natToStandard( *natParams )

        assert isinstance( x, np.ndarray )
        assert x.ndim == 1 and x.size[ 0 ] == p.ndim - 1

        D = p.shape[ 0 ]
        return np.random.choice( D, size, p=p[ x ] ), x

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, y=None, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        if( y is None ):
            return super( Transition, cls ).log_likelihood( x, params=params, natParams=natParams, size=size )

        return np.log( p[ x ][ y ] ).sum()

    ##########################################################################
