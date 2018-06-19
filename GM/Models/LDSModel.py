import numpy as np
from GenModels.GM.ModelPriors import LDSMNIWPrior
from GenModels.GM.States.StandardStates import LDSState
from GenModels.GM.Models.ModelBase import _InferenceModel

__all__ = [ 'LDSModel' ]

class LDSModel( _InferenceModel ):

    # This is a wrapper around a state class along with its prior class

    def __init__( self, A=None, sigma=None, C=None, R=None, mu0=None, sigma0=None,
                        M_trans=None, V_trans=None, psi_trans=None, nu_trans=None,
                        M_emiss=None, V_emiss=None, psi_emiss=None, nu_emiss=None,
                        mu_0=None, kappa_0=None, psi_0=None, nu_0=None ):

        if( M_trans is None or \
            V_trans is None or \
            psi_trans is None or \
            nu_trans is None or \
            M_emiss is None or \
            V_emiss is None or \
            psi_emiss is None or \
            nu_emiss is None or \
            mu_0 is None or \
            kappa_0 is None or \
            psi_0 is None or \
            nu_0 is None ):

            assert M_trans is None and V_trans is None and psi_trans is None and nu_trans is None and M_emiss is None and V_emiss is None and psi_emiss is None and nu_emiss is None and mu_0 is None and kappa_0 is None and psi_0 is None and nu_0 is None
            assert A is not None
        else:
            assert M_trans is not None and V_trans is not None and psi_trans is not None and nu_trans is not None and M_emiss is not None and V_emiss is not None and psi_emiss is not None and nu_emiss is not None and mu_0 is not None and kappa_0 is not None and psi_0 is not None and nu_0 is not None

        if( A is None or sigma is None or R is None or mu0 is None or sigma0 is None ):
            assert A is None and sigma is None and R is None and mu0 is None and sigma0 is None
        else:
            assert A is not None and sigma is not None and R is not None and mu0 is not None and sigma0 is not None

        if( M_trans is not None ):
            # Use a prior
            self.prior = LDSMNIWPrior( M_trans, V_trans, psi_trans, nu_trans,
                                       M_emiss, V_emiss, psi_emiss, nu_emiss,
                                       mu_0, kappa_0, psi_0, nu_0 )
        else:
            # Use a weak prior in this case
            D_latent = A.shape[ 0 ]
            D_obs = C.shape[ 0 ]
            self.prior = LDSMNIWPrior( **LDSModel._genericParams( D_latent, D_obs ) )

        self.state = LDSState( prior=self.prior )

    @staticmethod
    def _genericParams( D_latent, D_obs ):
        params = {
            'mu_0': np.zeros( D_latent ),
            'kappa_0': D_latent,
            'psi_0': np.eye( D_latent ),
            'nu_0': D_latent,

            'M_trans': np.zeros( ( D_latent, D_latent ) ),
            'V_trans': np.eye( D_latent ),
            'psi_trans': np.eye( D_latent ),
            'nu_trans': D_latent,

            'M_emiss': np.zeros( ( D_obs, D_latent ) ),
            'V_emiss': np.eye( D_latent ),
            'psi_emiss': np.eye( D_obs ),
            'nu_emiss': D_obs
        }
        return params

    def predict( self, T=3, measurements=1, size=1, stabilize=False ):
        return self.state.isample( T=T, measurements=measurements, size=size, stabilize=stabilize )

    @classmethod
    def generate( self, T=10, latentSize=3, obsSize=2, measurements=1, size=10, stabilize=False ):
        # Generate fake data
        dummy = LDSModel( **LDSModel._genericParams( latentSize, obsSize ) )
        return dummy.predict( T=T, measurements=measurements, size=size, stabilize=stabilize )
