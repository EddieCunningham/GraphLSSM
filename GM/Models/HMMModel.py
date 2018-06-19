import numpy as np
from GenModels.GM.ModelPriors import HMMDirichletPrior
from GenModels.GM.States.StandardStates import HMMState
from GenModels.GM.Models.ModelBase import _InferenceModel

__all__ = [ 'HMMModel' ]

class HMMModel( _InferenceModel ):

    # This is a wrapper around a state class along with its prior class

    def __init__( self, initialDist=None, transDist=None, emissionDist=None,
                        alpha_0=None, alpha=None, L=None ):
        if( alpha_0 is None or alpha is None or L is None ):
            assert alpha_0 is None and alpha is None and L is None
            assert initialDist is not None
        else:
            assert alpha_0 is not None and alpha is not None and L is not None

        if( initialDist is None or transDist is None or emissionDist is None ):
            assert initialDist is None and transDist is None and emissionDist is None
        else:
            assert initialDist is not None and transDist is not None and emissionDist is not None

        if( alpha_0 is not None ):
            # Use a prior
            self.prior = HMMDirichletPrior( alpha_0, alpha, L )
        else:
            # Use a weak prior in this case
            self.prior = HMMDirichletPrior( np.ones_like( initialDist ), np.ones_like( transDist ), np.ones_like( emissionDist ) )

        self.state = HMMState( prior=self.prior )

    def predict( self, T=3, measurements=1, knownLatentStates=None, size=1 ):
        return self.state.isample( T=T, measurements=measurements, knownLatentStates=knownLatentStates, size=size )

    @classmethod
    def generate( self, T=10, latentSize=3, obsSize=2, measurements=1, knownLatentStates=None, size=10 ):
        # Generate fake data
        params = {
            'alpha_0': np.ones( latentSize ),
            'alpha': np.ones( ( latentSize, latentSize ) ),
            'L': np.ones( ( latentSize, obsSize ) )
        }
        dummy = HMMModel( **params )
        return dummy.predict( T=T, measurements=measurements, knownLatentStates=knownLatentStates, size=size )
