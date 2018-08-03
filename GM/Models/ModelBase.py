import numpy as np
from abc import ABC, abstractmethod
from GenModels.GM.Utility import deepCopy, verboseRange

__all__ = [ '_InferenceModel' ]

##########################################################################

class _ModelBase( ABC ):

    stateClass = None

    def __init__( self, *args, **kwargs ):
        pass

    @abstractmethod
    def fit( self, ys ):
        pass

    @abstractmethod
    def predict( self ):
        pass

    # @abstractmethod
    # def predictFromMeanField( self ):
    #     pass

##########################################################################

class _GibbsMixin():

    def resample_step( self, ys, **kwargs ):

        # This way is way faster.  Probably going to use the stats way when I re-implement in cython though
        x = self.state.isample( ys=ys, size=1, **kwargs )
        x = self.state.unpackSingleSample( x )
        self.state.params = self.state.prior.unpackSingleSample( self.state.iposteriorSample( x ) )

        for p in self.state.params:
            print( p )
            print()
            print()
            print()
        print( '======================' )

        # stats = self.state.isample( ys=ys, size=1, returnStats=True, **kwargs )
        # self.state.params = self.state.prior.unpackSingleSample( self.state.iposteriorSample( stats=stats ) )

    def gibbs( self, ys, nIters=1000, burnIn=1000, skip=10, verbose=False, **kwargs ):
        for i in verboseRange( skip * ( nIters ) + burnIn, verbose ):
            self.resample_step( ys, **kwargs )

##########################################################################

class _EMMixin():

    def expectationMaximization( self, ys, nIters=100, monitorMarginal=10, verbose=False, preprocessKwargs={}, filterKwargs={} ):

        lastMarginal = 999

        for i in verboseRange( nIters, verbose ):
            alphas, betas = self.state.EStep( ys=ys, preprocessKwargs=preprocessKwargs, filterKwargs=filterKwargs )
            self.state.nat_params = self.state.MStep( ys, alphas, betas )

            marginal = self.state.last_normalizer
            if( np.isclose( marginal, lastMarginal ) ):
                break

            lastMarginal = marginal

##########################################################################

class _CoordinateAscentVIMixin():

    def cavi( self, ys, maxIters=1000, verbose=False ):

        last_elbo = 9999

        for i in verboseRange( maxIters, verbose ):

            if( i > 0 ):
                self.state.prior.mf_nat_params = prior_mf_nat_params

            self.state.mf_nat_params = self.state.iexpectedNatParams( use_mean_field=True )
            prior_mf_nat_params, normalizer = self.state.variationalPosteriorPriorNatParams( ys=ys,
                                                                                             nat_params=self.state.mf_nat_params,
                                                                                             prior_nat_params=self.state.prior.nat_params,
                                                                                             return_normalizer=True )

            # The ELBO computation is only valid right after the variational E step
            elbo = self.state.ELBO( normalizer=normalizer,
                                    prior_mf_nat_params=self.state.prior.mf_nat_params,
                                    prior_nat_params=self.state.prior.nat_params )

            if( np.isclose( last_elbo, elbo ) ):
                break

            last_elbo = elbo

##########################################################################

class _InferenceModel( _ModelBase, _EMMixin, _GibbsMixin, _CoordinateAscentVIMixin ):

    def fit( self, ys, method='gibbs', **kwargs ):
        if( method == 'gibbs' ):
            return self.gibbs( ys, **kwargs )
        elif( method == 'cavi' ):
            return self.cavi( ys, **kwargs )
        elif( method == 'EM' ):
            return self.expectationMaximization( ys, **kwargs )
        else:
            assert 0, 'Invalid method type'
