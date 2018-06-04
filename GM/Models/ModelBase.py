import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm

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

##########################################################################

class _GibbsMixin():

    def resample_step( self, ys ):
        x = self.state.isample( ys=ys )
        self.state.params = self.state.iposteriorSample( x )

        _y = self.predict( T=10 )

        ll = self.state.ilog_joint( x )
        print( x )
        print( _y )
        print()
        print()
        print()
        print()

    def gibbs( self, ys, nIters=1000, burnIn=1000, skip=10, verbose=False ):
        for i in verboseRange( skip * ( nIters ) + burnIn, verbose ):
            self.resample_step( ys )

##########################################################################

class _CoordinateAscentVIMixin():

    def ELBO( self, ys ):

        expectedStats = self.state.iexpectedSufficientStats( ys=ys )
        expectedNatParams = self.state.iexpectedNatParams()

        p_z = self.state.log_joint( ( expectedStats, ys ), natParams=expectedNatParams )

        q_z1 = self.state.log_params( priorNatParams=expectedNatParams )
        q_z2 = self.state.ilog_likelihood( x=expectedNatParams )

        return p_z - q_z1 - q_z2

    ######################################################################

    def cavi( self, ys, maxIters=1000, verbose=False ):

        lastElbo = 0

        for i in verboseRange( maxIters, verbose ):
            expectedNatParams = self.state.iexpectedNatParams()
            self.state.natParams = self.state.variationalPosteriorPriorNatParams( ys=ys, priorNatParams=expectedNatParams )

            elbo = self.ELBO( ys=ys )
            print( 'ELBO', elbo )

            if( np.isclose( lastElbo, elbo ) ):
                break

##########################################################################

class _InferenceModel( _ModelBase, _GibbsMixin, _CoordinateAscentVIMixin ):

    def fit( self, ys, method='gibbs', **kwargs ):
        if( method == 'gibbs' ):
            return self.gibbs( ys, **kwargs )
        elif( method == 'cavi' ):
            return self.cavi( ys, **kwargs )
        else:
            assert 0, 'Invalid method type'
