import numpy as np
from GenModels.GM.ModelPriors import HMMDirichletPrior
from GenModels.GM.States.StandardStates import HMMState

class HMMModel():

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
            self.state = HMMState( prior=prior )
        else:
            self.prior = None
            self.state = HMMState( initialDist, transDist, emissionDist )

    def sample( cls, ys=None, T=None, forwardFilter=True ):
        # Sample from P( x | Ѳ; α )
        return self.state.isample( ys=ys, T=T, forwardFilter=forwardFilter )

    def resample( cls, x ):
        # Sample from P( Ѳ | x; α )
        self.state.resample( x=x )
