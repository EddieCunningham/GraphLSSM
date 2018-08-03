import numpy as np
from GenModels.GM.Distributions import ExponentialFam
from abc import ABC, abstractmethod
import itertools

class StateBase( ExponentialFam ):

    # This is a distribution over P( x, y | ϴ ).
    # Will still be able to do inference over P( x | y, ϴ )

    def __init__( self, *args, **kwargs ):
        self._normalizerValid = False
        super( StateBase, self ).__init__( *args, **kwargs )

    ######################################################################

    @property
    def params( self ):
        if( self.naturalChanged ):
            self._params = self.natToStandard( *self.nat_params )
            self.naturalChanged = False
        return self._params

    @property
    def nat_params( self ):
        if( self.standard_changed ):
            self._nat_params = self.standardToNat( *self.params )
            self.standard_changed = False
        return self._nat_params

    @params.setter
    def params( self, val ):
        self.standard_changed = True
        self.naturalChanged = False
        self.updateParams( *val )
        self._params = val

    @nat_params.setter
    def nat_params( self, val ):
        self.naturalChanged = True
        self.standard_changed = False
        self.updateNatParams( *val )
        self._nat_params = val

    ##########################################################################
    ## Mean field parameters for variational inference.  Only update from ##
    ## natrual mean field params ##

    @property
    def mf_params( self ):
        if( self.mfNaturalChanged ):
            self._mf_params = self.natToStandard( *self.mf_nat_params )
            self.mfNaturalChanged = False
        return self._mf_params

    @mf_params.setter
    def mf_params( self, val ):
        assert 0, 'Don\'t update this way!  All of the message passing algorithms (should) only work with natural params!'

    @property
    def mf_nat_params( self ):
        return self._mf_nat_params

    @mf_nat_params.setter
    def mf_nat_params( self, val ):
        print( '-----------------' )
        print( self )
        for v in val:
            print( v )
            print()
        print( '-----------------' )
        self.mfNaturalChanged = True
        self.updateNatParams( *val )
        self._mf_nat_params = val

    ######################################################################

    @property
    def last_normalizer( self ):
        if( self._normalizerValid ):
            return self._last_normalizer
        return None

    @last_normalizer.setter
    def last_normalizer( self, val ):
        self._normalizerValid = True
        self._last_normalizer = val

    ######################################################################

    @property
    @abstractmethod
    def T( self ):
        pass

    @abstractmethod
    def forwardFilter( self ):
        pass

    @abstractmethod
    def backwardFilter( self ):
        pass

    @abstractmethod
    def preprocessData( self, ys ):
        pass

    @abstractmethod
    def parameterCheck( self, *args ):
        pass

    @abstractmethod
    def updateParams( self, *args ):
        pass

    @abstractmethod
    def genStates( self ):
        pass

    @abstractmethod
    def genEmissions( self, measurements ):
        pass

    @abstractmethod
    def sampleStep( self, *args ):
        pass

    @abstractmethod
    def likelihoodStep( self, *args ):
        pass

    @abstractmethod
    def forwardArgs( self, t, beta, prevX ):
        pass

    @abstractmethod
    def backwardArgs( self, t, alpha, prevX ):
        pass

    @abstractmethod
    def sequenceLength( cls, x ):
        pass

    @abstractmethod
    def nMeasurements( cls, x ):
        pass

    ######################################################################

    def noFilterForwardRecurse( self, workFunc ):
        lastVal = None
        for t in range( self.T ):
            args = self.forwardArgs( t, None, lastVal )
            lastVal = workFunc( lastVal, t, *args )

        self._normalizerValid = False

    def forwardFilterBackwardRecurse( self, workFunc, **kwargs ):
        # P( x_1:T | y_1:T ) = prod_{ x_t=T:1 }[ P( x_t | x_t+1, y_1:t ) ] * P( x_T | y_1:T )
        alphas = self.forwardFilter( **kwargs )

        lastVal = None
        for t in reversed( range( self.T ) ):
            args = self.backwardArgs( t, alphas[ t ], lastVal )
            lastVal = workFunc( lastVal, t, *args )

        # Reset the normalizer flag
        self._normalizerValid = False

    def backwardFilterForwardRecurse( self, workFunc, **kwargs ):
        # P( x_1:T | y_1:T ) = prod_{ x_t=1:T }[ P( x_t+1 | x_t, y_t+1:T ) ] * P( x_1 | y_1:T )
        betas = self.backwardFilter( **kwargs )

        lastVal = None
        for t in range( self.T ):
            args = self.forwardArgs( t, betas[ t ], lastVal )
            lastVal = workFunc( lastVal, t, *args )

        # Reset the normalizer flag
        self._normalizerValid = False

    ######################################################################

    @abstractmethod
    def sampleSingleEmission( self, x ):
        pass

    @abstractmethod
    def sampleEmissions( self, x, measurements=1 ):
        # Sample from P( y | x, ϴ )
        pass

    @abstractmethod
    def emissionLikelihood( self, x, ys ):
        # Compute P( y | x, ϴ )
        pass

    ######################################################################
    # These methods are for incrementally computing stats instead of all
    # at the end

    @classmethod
    @abstractmethod
    def genStats( cls ):
        pass

    @classmethod
    @abstractmethod
    def initialStats( cls, x, constParams=None ):
        pass

    @classmethod
    @abstractmethod
    def transitionStats( cls, x, constParams=None ):
        pass

    @classmethod
    @abstractmethod
    def emissionStats( cls, x, constParams=None ):
        pass

    ######################################################################

    @abstractmethod
    def conditionedExpectedSufficientStats( self, alphas, betas ):
        pass

    @classmethod
    def expectedSufficientStats( cls, ys=None, params=None, nat_params=None, return_normalizer=False, **kwargs ):
        assert ( params is None ) ^ ( nat_params is None )
        params = params if params is not None else cls.natToStandard( *nat_params )
        dummy = cls( *params, paramCheck=False )
        return dummy.iexpectedSufficientStats( ys=ys, return_normalizer=return_normalizer, **kwargs )

    def iexpectedSufficientStats( self, ys=None, preprocessKwargs={}, filterKwargs={}, return_normalizer=False ):

        if( ys is None ):
            return super( StateBase, self ).iexpectedSufficientStats()

        alphas, betas = self.EStep( ys=ys, preprocessKwargs=preprocessKwargs, filterKwargs=filterKwargs )
        stats = self.conditionedExpectedSufficientStats( ys, alphas, betas )

        if( return_normalizer ):
            return stats, self.last_normalizer
        return stats

    ######################################################################

    @classmethod
    def sample( cls, ys=None, params=None, nat_params=None, measurements=1, T=None, forwardFilter=True, size=1, returnStats=False, **kwargs ):
        assert ( params is None ) ^ ( nat_params is None )
        params = params if params is not None else cls.natToStandard( *nat_params )
        dummy = cls( *params )
        return dummy.isample( ys=ys, measurements=measurements, T=T, forwardFilter=forwardFilter, size=size, returnStats=returnStats, **kwargs )

    ######################################################################

    def accumulateStats( self, accumulated, stats, M, T ):
        # Accumulated will have the form [ [ all stats ], [ size, measurements, time, measurements*time ] ]
        # stats will have the shape ( [ initialStats ], [ transitionStats ], [ emissionStats ] )
        chainedStats = list( itertools.chain( *stats ) )

        if( len( accumulated[ 0 ] ) == 0 ):
            accumulated[ 0 ] = chainedStats
            accumulated[ 1 ] = [ 1, M, T, T * M ]
        else:
            for i, stat in enumerate( chainedStats ):
                accumulated[ 0 ][ i ][ : ] += stat
            accumulated[ 1 ][ 0 ] += 1
            accumulated[ 1 ][ 1 ] += M
            accumulated[ 1 ][ 2 ] += T
            accumulated[ 1 ][ 3 ] += T * M
        return accumulated

    def conditionedSample( self, ys=None, forwardFilter=True, preprocessKwargs={}, filterKwargs={}, returnStats=False ):
        # Sample x given y

        size = self.dataN( ys, conditionOnY=True, checkY=True )

        if( size > 1 ):
            it = iter( ys )
        else:
            it = iter( ys )
            # it = iter( [ ys ] )

        ans = [] if returnStats == False else [ [], [] ]
        for y in it:

            self.preprocessData( ys=y, computeMarginal=False, **preprocessKwargs )

            x = self.genStates() if returnStats == False else self.genStats()

            def workFuncForStateSample( lastX, t, *args ):
                nonlocal x
                x[ t ] = self.sampleStep( *args )
                return x[ t ]

            def workFuncForStats( lastX, t, *args ):
                nonlocal x
                _x = self.sampleStep( *args )
                _y = y[ :, t ]

                # Compute the initial distribution, transition and emission sufficient statistics
                if( lastX is None ):
                    stats = self.initialStats( _x, constParams=self.constParams )
                    for i, stat in enumerate( stats ):
                        x[ 0 ][ i ][ : ] = stat
                else:
                    # Change this way of passing t to constParams eventually
                    stats = self.transitionStats( ( lastX, _x ), constParams=( self.constParams, t ) )
                    for i, stat in enumerate( stats ):
                        x[ 1 ][ i ][ : ] += stat

                stats = self.emissionStats( ( _x, _y ), constParams=self.constParams )
                for i, stat in enumerate( stats ):
                    x[ 2 ][ i ][ : ] += stat

                return _x

            # Use the correct work function and filter function for the given arguments
            workFunc = workFuncForStateSample if returnStats == False else workFuncForStats
            filterFunc = self.forwardFilterBackwardRecurse if forwardFilter else self.backwardFilterForwardRecurse
            filterFunc( workFunc, **filterKwargs )

            if( returnStats == False ):
                ans.append( ( x, y ) )
            else:
                # Accumulate the statistics
                M, T = y.shape[ 0:2 ]
                ans = self.accumulateStats( ans, x, M, T )

        # Make sure that the sampled states are in the expected form
        if( returnStats == False ):
            ans = tuple( list( zip( *ans ) ) )
            self.checkShape( ans )

        return ans

    ######################################################################

    def fullSample( self, measurements=2, T=None, size=1, returnStats=False ):
        # Sample x and y

        assert T is not None
        self.T = T

        ans = [] if returnStats == False else [ [], [] ]
        for _ in range( size ):

            if( returnStats == False ):
                x = self.genStates()
                y = self.genEmissions( measurements )
            else:
                x = self.genStats()

            def workFuncForStateSample( lastX, t, *args ):
                nonlocal x
                x[ t ] = self.sampleStep( *args )
                # The first two axes should be ( measurements, time )
                y[ :, t, ... ] = self.sampleSingleEmission( x[ t ], measurements=measurements )
                return x[ t ]

            def workFuncForStats( lastX, t, *args ):
                nonlocal x
                _x = self.sampleStep( *args )
                _y = self.sampleSingleEmission( _x, measurements=measurements )

                # Compute the initial distribution, transition and emission sufficient statistics
                if( lastX is None ):
                    stats = self.initialStats( _x, constParams=self.constParams )
                    for i, stat in enumerate( stats ):
                        x[ 0 ][ i ][ : ] = stat
                else:
                    # Change this way of passing t to constParams eventually
                    stats = self.transitionStats( ( lastX, _x ), constParams=( self.constParams, t ) )
                    for i, stat in enumerate( stats ):
                        x[ 1 ][ i ][ : ] += stat

                stats = self.emissionStats( ( _x, _y ), constParams=self.constParams )
                for i, stat in enumerate( stats ):
                    x[ 2 ][ i ][ : ] += stat

                return _x

            workFunc = workFuncForStateSample if returnStats == False else workFuncForStats

            # This is if we want to sample from P( x, y | ϴ )
            self.noFilterForwardRecurse( workFunc )

            if( returnStats == False ):
                ans.append( ( x, y ) )
            else:
                # Accumulate the statistics.
                # Each x has the shape of ( [ initialStats, ], [ transitionStats, ], [ emissionStats, ] )
                M, T = ( measurements, T )
                ans = self.accumulateStats( ans, x, M, T )

        if( returnStats == False ):
            ans = tuple( list( zip( *ans ) ) )
            self.checkShape( ans )

        return ans

    ######################################################################

    def isample( self, ys=None, measurements=1, T=None, forwardFilter=True, size=1, retutnStats=False ):
        # If returnStats is true, then return t( x, y ) instead of ( x, y )
        if( ys is not None ):
            return self.conditionedSample( ys=ys, forwardFilter=forwardFilter, returnStats=returnStats )
        return self.fullSample( measurements=measurements, T=T, size=size, returnStats=returnStats )

    ######################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, nat_params=None, forwardFilter=True, conditionOnY=False, seperateLikelihoods=False, preprocessKwargs={}, filterKwargs={} ):
        assert ( params is None ) ^ ( nat_params is None )
        params = params if params is not None else cls.natToStandard( *nat_params )

        dummy = cls( *params )
        return dummy.ilog_likelihood( x, forwardFilter=forwardFilter, conditionOnY=conditionOnY, seperateLikelihoods=seperateLikelihoods, preprocessKwargs=preprocessKwargs, filterKwargs=filterKwargs )

    def ilog_likelihood( self, x, forwardFilter=True, conditionOnY=False, expFam=False, preprocessKwargs={}, filterKwargs={}, seperateLikelihoods=False ):

        if( expFam ):
            return self.log_likelihoodExpFam( x, constParams=self.constParams, nat_params=self.nat_params )

        size = self.dataN( x )

        x, ys = x

        if( size > 1 ):
            it = zip( x, ys )
        else:
            # Need to add for case where size is 1 and unpacked vs size is 1 and packed
            it = zip( x, ys )
            # it = iter( [ [ x, ys ] ] )

        ans = np.zeros( size )

        for i, ( x, ys ) in enumerate( it ):

            self.preprocessData( ys=ys, **preprocessKwargs )

            def workFunc( lastX, t, *args ):
                nonlocal ans, x
                term = self.likelihoodStep( x[ t ], *args )
                ans[ i ] += term
                return x[ t ]

            if( conditionOnY == False ):
                # This is if we want to compute P( x, y | ϴ )
                self.noFilterForwardRecurse( workFunc )
                ans[ i ] += self.emissionLikelihood( x, ys )
            else:
                if( forwardFilter ):
                    # Otherwise compute P( x | y, ϴ )
                    assert conditionOnY == True
                    self.forwardFilterBackwardRecurse( workFunc, **filterKwargs )
                else:
                    assert conditionOnY == True
                    self.backwardFilterForwardRecurse( workFunc, **filterKwargs )

        if( seperateLikelihoods == True ):
            return ans

        return ans.sum()

    ######################################################################

    @classmethod
    def log_marginal( cls, x, params=None, nat_params=None, seperateMarginals=False, preprocessKwargs={}, filterKwargs={}, alphas=None, betas=None ):
        assert ( params is None ) ^ ( nat_params is None )
        params = params if params is not None else cls.natToStandard( *nat_params )

        dummy = cls( *params )
        return dummy.ilog_marginal( x, seperateMarginals=seperateMarginals, preprocessKwargs=preprocessKwargs, filterKwargs=filterKwargs, alphas=alphas, betas=betas )

    def ilog_marginal( self, ys, seperateMarginals=False, preprocessKwargs={}, filterKwargs={}, alphas=None, betas=None ):

        size = self.dataN( ys, conditionOnY=True, checkY=True )

        def work( _ys ):
            self.preprocessData( ys=_ys, **preprocessKwargs )
            alpha = self.forwardFilter( **filterKwargs )
            beta = self.backwardFilter( **filterKwargs )
            return self.log_marginalFromAlphaBeta( alpha[ 0 ], beta[ 0 ] )

        # if( size == 1 ):
        #     return work( ys )

        ans = np.empty( size )

        if( alphas is not None or betas is not None ):
            assert alphas is not None and betas is not None
            for i, ( _ys, _alpha, _beta ) in enumerate( zip( ys, alphas, betas ) ):
                ans[ i ] = self.log_marginalFromAlphaBeta( _alpha[ 0 ], _beta[ 0 ] )
        else:
            for i, _ys in enumerate( ys ):
                ans[ i ] = work( _ys )

        if( seperateMarginals == False ):
            ans = ans.sum()

        return ans

    ######################################################################

    def EStep( self, ys=None, preprocessKwargs={}, filterKwargs={} ):

        def work( _ys ):
            self.preprocessData( ys=_ys, **preprocessKwargs )
            a = self.forwardFilter( **filterKwargs )
            b = self.backwardFilter( **filterKwargs )

            return a, b

        if( self.dataN( ys, conditionOnY=True, checkY=True ) > 1 ):
            alphas, betas = zip( *[ work( _ys ) for _ys in ys ] )
        else:
            alphas, betas = zip( *[ work( _ys ) for _ys in ys ] )
            # alphas, betas = work( ys )

        self.last_normalizer = self.ilog_marginal( ys, alphas=alphas, betas=betas )
        return alphas, betas

    @abstractmethod
    def MStep( self, ys, alphas, betas ):
        pass

    ######################################################################

    @classmethod
    def ELBO( cls, ys=None,
                   mf_params=None,
                   mf_nat_params=None,
                   prior_mf_params=None,
                   prior_mf_nat_params=None,
                   prior_params=None,
                   prior_nat_params=None,
                   normalizer=None,
                   **kwargs ):

        if( ys is None ):
            assert normalizer is not None
        else:
            mf_params = mf_params if mf_params is not None else cls.natToStandard( *mf_nat_params )
            dummy = cls( *mf_params, paramCheck=False )
            dummy.EStep( ys=ys, **kwargs )
            normalizer = dummy.last_normalizer

        klDiv = cls.priorClass.KLDivergence( params1=prior_params, nat_params1=prior_nat_params, params2=prior_mf_params, nat_params2=prior_mf_nat_params )

        return normalizer + klDiv

    def iELBO( self, ys, **kwargs ):

        # E_{ q( x, Ѳ ) }[ log_P( y, x | Ѳ ) - log_q( x ) ] = normalization term after message passing
        # E_{ q( x, Ѳ ) }[ log_p( Ѳ ) - log_q( Ѳ ) ] = KL divergence between p( Ѳ ) and q( Ѳ )

        # Probably want a better way to do this than just creating a dummy state instance
        dummy = type( self )( *self.mf_params, paramCheck=False )
        dummy.EStep( ys=ys, **kwargs )
        normalizer = dummy.last_normalizer

        klDiv = self.prior.iKLDivergence( otherNatParams=self.prior.mf_nat_params )

        return normalizer + klDiv

