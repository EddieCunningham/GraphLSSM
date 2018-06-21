import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from GenModels.GM.Utility import deepCopy


__all__ = [ '_InferenceModel' ]

def verboseRange( numbIters, verbose ):
    if( verbose ):
        return tqdm( range( numbIters ) )
    return range( numbIters )

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

        x = self.state.isample( ys=ys, size=1, **kwargs )
        x = self.state.unpackSingleSample( x )
        print( x )
        print()
        print()
        print( '------------------------------' )
        print()
        for p in self.state.params:
            print( p )
            print()
        print()
        print( '===========================================' )
        print()
        self.state.params = self.state.prior.unpackSingleSample( self.state.iposteriorSample( x ) )

    def gibbs( self, ys, nIters=1000, burnIn=1000, skip=10, verbose=False, **kwargs ):
        for i in verboseRange( skip * ( nIters ) + burnIn, verbose ):
            self.resample_step( ys, **kwargs )

##########################################################################

class _EMMixin():

    def expectationMaximization( self, ys, nIters=100, monitorMarginal=10, verbose=False, preprocessKwargs={}, filterKwargs={} ):

        lastMarginal = 999

        for i in verboseRange( nIters, verbose ):
            alphas, betas = self.state.EStep( ys=ys, preprocessKwargs=preprocessKwargs, filterKwargs=filterKwargs )
            self.state.natParams = self.state.MStep( ys, alphas, betas )

            marginal = self.state.lastNormalizer
            if( np.isclose( marginal, lastMarginal ) ):
                break

            lastMarginal = marginal

##########################################################################

class _CoordinateAscentVIMixin():

    def cavi( self, ys, maxIters=1000, verbose=False ):

        lastElbo = 9999

        for i in verboseRange( maxIters, verbose ):

            if( i > 0 ):
                self.state.prior.mfNatParams = priorMFNatParams

            self.state.mfNatParams = self.state.iexpectedNatParams( useMeanField=True )
            priorMFNatParams, normalizer = self.state.variationalPosteriorPriorNatParams( ys=ys,
                                                                                          natParams=self.state.mfNatParams,
                                                                                          priorNatParams=self.state.prior.natParams,
                                                                                          returnNormalizer=True )

            # The ELBO computation is only valid right after the variational E step
            elbo = self.state.ELBO( normalizer=normalizer,
                                    priorMFNatParams=self.state.prior.mfNatParams,
                                    priorNatParams=self.state.prior.natParams )

            if( np.isclose( lastElbo, elbo ) ):
                break

            lastElbo = elbo

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
