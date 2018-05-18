from abc import ABC, abstractmethod
import numpy as np
from GenModels.GM.Utility import *
import string
import tqdm
from functools import wraps
from collections import Iterable

__all__ = [ 'Distribution', \
            'Conjugate', \
            'ExponentialFam', \
            'checkExpFamArgs' ]

class Distribution( ABC ):

    priorClass = None

    def __init__( self, *params, prior=None, hypers=None ):
        if( prior is not None ):
            assert isinstance( prior, self.priorClass )
            self.prior = prior
        elif( hypers is not None ):
            self.prior = self.priorClass( *hypers )

        # Set the parameters
        if( prior is None and hypers is None ):
            self.params = params
        else:
            self.resample()

    ##########################################################################

    def paramChoices( self, includeSelf=True, includePrior=False, **kwargs ):

        params = {}
        if( includeSelf ):
            params.update( { 'params': self.params } )

        if( includePrior ):
            params.update( { 'priorParams': self.prior.params } )

        params.update( kwargs )

        return params

    ##########################################################################

    @property
    def params( self ):
        return self._params

    @params.setter
    @abstractmethod
    def params( self, val ):
        pass

    @property
    @abstractmethod
    def constParams( self ):
        # These are things that are constant for every instance of the class
        pass

    @classmethod
    @abstractmethod
    def dataN( cls, x, ravel=False ):
        # This is necessary to know how to tell the size of an output
        pass

    @classmethod
    def outputShapes( cls, params ):
        # Return the shapes of the outputs based on params
        x = cls.sample( params, size=1, ravel=False )
        if( isinstance( x, tuple ) or isinstance( x, list ) ):
            return [ _x.shape for _x in x ]
        return [ x.shape ]

    @classmethod
    def unravelSample( cls, x, params ):
        shapes = cls.outputShapes( params )

        split = np.split( x, [ prod( s ) for s in shapes[ :-1 ] ] )
        ans = []
        for item, s in zip( split, shapes ):
            assert item.shape == s
            ans.append( item )
        return ans

    ##########################################################################

    @classmethod
    @abstractmethod
    def sample( cls, params, size=1, ravel=False ):
        # Sample from P( x | Ѳ; α )
        pass

    def isample( self, size=1, ravel=False ):
        return self.sample( **self.paramChoices( includeSelf=True, includePrior=False, size=size, ravel=ravel ) )

    @classmethod
    @abstractmethod
    def log_likelihood( cls, x, params, ravel=False ):
        # Compute P( x | Ѳ; α )
        pass

    def ilog_likelihood( self, x, ravel=False ):
        return self.log_likelihood( x, **self.paramChoices( includeSelf=True, includePrior=False, ravel=ravel ) )

    ##########################################################################

    @classmethod
    @multiParamSample
    def jointSample( cls, priorParams, size=1, ravel=False ):
        # Sample from P( x, Ѳ; α )
        theta = cls.paramSample( priorParams=priorParams, ravel=ravel )
        x = cls.sample( params=theta, ravel=ravel )
        return x, theta

    def ijointSample( self, size=1, ravel=False ):
        return self.jointSample( **self.paramChoices( includeSelf=False, includePrior=True, size=size, ravel=ravel ) )

    @classmethod
    def log_joint( cls, x, params, priorParams, ravel=False ):
        # Compute P( x, Ѳ; α )
        return cls.log_likelihood( x, params, ravel=ravel ) + cls.log_params( params, priorParams, ravel=ravel )

    def ilog_joint( self, x, ravel=False ):
        return self.log_joint( x, **self.paramChoices( includeSelf=True, includePrior=True, ravel=ravel ) )

    ##########################################################################

    @classmethod
    @abstractmethod
    def posteriorSample( cls, x, priorParams, size=1, ravel=False ):
        # Sample from P( Ѳ | x; α )
        pass

    def iposteriorSample( self, x, size=1, ravel=False ):
        return self.posteriorSample( x, **self.paramChoices( includeSelf=False, includePrior=True, size=size ) )

    @classmethod
    @abstractmethod
    def log_posterior( cls, x, params, priorParams, ravel=False ):
        # Compute P( Ѳ | x; α )
        pass

    def ilog_posterior( self, x, ravel=False ):
        return self.log_posterior( x, self.params, self.prior.params, ravel=ravel )

    ##########################################################################

    @classmethod
    def paramSample( cls, priorParams, size=1, ravel=False ):
        # Sample from P( Ѳ; α )
        return cls.priorClass.sample( priorParams, size=size, ravel=ravel )

    def iparamSample( self, size=1, ravel=False ):
        return self.prior.isample( size=size, ravel=ravel )

    @classmethod
    def log_params( cls, params, priorParams, ravel=False ):
        # Compute P( Ѳ; α )
        return cls.priorClass.log_likelihood( params, priorParams, ravel=ravel )

    def ilog_params( self, ravel=False ):
        return self.log_params( self.params, self.prior.params, ravel=ravel )

    ##########################################################################

    @classmethod
    def log_marginal( cls, x, params, priorParams, ravel=False ):
        likelihood = cls.log_likelihood( x, params, ravel=ravel )
        posterior = cls.log_posterior( x, params, priorParams, ravel=ravel )
        params = cls.log_params( params, priorParams, ravel=ravel )
        return likelihood + params - posterior

    def ilog_marginal( self, x, ravel=False ):
        return self.log_marginal( x, **self.paramChoices( includeSelf=True, includePrior=True, constParams=self.constParams ) )

    ##########################################################################

    def resample( self, x=None ):
        if( x is None ):
            self.params = self.iparamSample()
        else:
            self.params = self.iposteriorSample( x )

    ##########################################################################

    def metropolisHastings( self, maxN=10000, burnIn=3000, size=1000, skip=50, verbose=True, concatX=True ):

        x = self.isample( ravel=True )
        p = self.ilog_likelihood( x )

        maxN = max( maxN, burnIn + size * skip )
        it = range( maxN )
        if( verbose ):
            it = tqdm.tqdm( it, desc='Metropolis Hastings (once every %d)'%( skip ) )

        samples = []

        for i in it:
            candidate = randomStep( x )
            pCandidate = self.ilog_likelihood( candidate )
            if( pCandidate >= p ):
                x, p = ( candidate, pCandidate )
            else:
                u = np.random.rand()
                if( u < np.exp( pCandidate - p ) ):
                    x, p = ( candidate, pCandidate )

            if( i > burnIn and i % skip == 0 ):
                if( concatX ):
                    samples.append( deepCopy( fullyRavel( x ) ) )
                else:
                    samples.append( deepCopy( x ) )
                if( len( samples ) >= size ):
                    break

        samples = np.vstack( samples )

        return samples

    ##########################################################################

    def functionalityTest( self, N=7, **D ):

        x = self.sample( **D, size=N, ravel=False )
        y = self.sample( **D, size=N, ravel=True )

        assert self.dataN( x, ravel=False ) == N
        assert self.dataN( y, ravel=True ) == N

        self.log_likelihood( x, params=self.paramSample(), ravel=False )
        self.log_likelihood( y, params=self.paramSample(), ravel=True )

    def marginalTest( self, N=1 ):
        # P( x ) should stay the same for different settings of params
        x = self.isample( size=10 )

        self.resample()
        marginal = self.ilog_marginal( x )

        for _ in range( N ):
            self.resample()
            marginal2 = self.ilog_marginal( x )
            assert np.isclose( marginal, marginal2 ), marginal2 - marginal

    def sampleTest( self, regAxes, mhAxes, plotFn, nRegPoints=1000, nMHPoints=1000, burnIn=3000, nMHForget=50 ):
        # Compare the MH sampler to the implemented sampler

        # Generate the joint samples
        regSamples = self.isample( size=nRegPoints )
        projections, axisChoices = plotFn( regSamples )
        for ( xs, ys ), ax, ( a, b ) in zip( projections, regAxes, axisChoices ):
            mask = ~is_outlier( xs ) & ~is_outlier( ys )
            ax.scatter( xs[ mask ], ys[ mask ], s=0.5, alpha=0.08, color='red' )
            ax.set_title( 'Ax %d vs %d'%( a, b ) )

        # Generate the metropolis hastings points
        mhSamples = self.metropolisHastings( burnIn=burnIn, skip=nMHForget, size=nMHPoints )
        projections, _ = plotFn( mhSamples, axisChoices=axisChoices )
        for ( xs, ys ), ax in zip( projections, mhAxes ):
            mask = ~is_outlier( xs ) & ~is_outlier( ys )
            ax.scatter( xs[ mask ], ys[ mask ], s=0.5, alpha=0.08, color='blue' )

    def gewekeTest( self, jointAxes, gibbsAxes, plotFn, nJointPoints=1000, nGibbsPoints=1000, burnIn=3000, nGibbsForget=50 ):
        # Sample from P( x, Ѳ ) using forward sampling and gibbs sampling and comparing their plots

        self.resample()

        # Generate the joint samples
        jointSamples = self.ijointSample( size=nJointPoints )
        projections, axisChoices = plotFn( *jointSamples )
        for ( xs, ys ), ax, ( a, b ) in zip( projections, jointAxes, axisChoices ):
            mask = ~is_outlier( xs ) & ~is_outlier( ys )
            ax.scatter( xs[ mask ], ys[ mask ], s=0.5, alpha=0.08, color='red' )
            ax.set_title( 'Ax %d vs %d'%( a, b ) )

        # Generate the gibbs points
        gibbsSamples = self.igibbsJointSample( burnIn=burnIn, skip=nGibbsForget, size=nGibbsPoints )
        projections, _ = plotFn( *gibbsSamples, axisChoices=axisChoices )
        for ( xs, ys ), ax in zip( projections, gibbsAxes ):
            mask = ~is_outlier( xs ) & ~is_outlier( ys )
            ax.scatter( xs[ mask ], ys[ mask ], s=0.5, alpha=0.08, color='blue' )

####################################################################################################################################

class Conjugate( Distribution ):
    # Fill this in at some point
    pass

####################################################################################################################################

@doublewrap
def checkExpFamArgs( func, allowNone=False ):

    @wraps( func )
    def wrapper( *args, **kwargs ):

        if( 'params' in kwargs and 'natParams' in kwargs ):
            params = kwargs[ 'params' ]
            natParams = kwargs[ 'natParams' ]
            if( allowNone ):
                if( not( params is None and natParams is None ) ):
                    assert ( params is None ) ^ ( natParams is None ), kwargs
            else:
                assert ( params is None ) ^ ( natParams is None ), kwargs

        if( 'priorParams' in kwargs and 'priorNatParams' in kwargs ):
            priorParams = kwargs[ 'priorParams' ]
            priorNatParams = kwargs[ 'priorNatParams' ]
            assert ( priorParams is None ) ^ ( priorNatParams is None ), kwargs

        return func( *args, **kwargs )

    return wrapper

####################################################################################################################################

class ExponentialFam( Conjugate ):

    def __init__( self, *params, prior=None, hypers=None ):

        self.standardChanged = False
        self.naturalChanged = False

        super( ExponentialFam, self ).__init__( prior=prior, hypers=hypers )

        # Set the parameters
        if( prior is None and hypers is None ):
            self.params = params
        else:
            self.params = self.iparamSample()

        # Set the natural parameters
        self.natParams

    ##########################################################################

    def paramChoices( self, includeSelf=True, includePrior=False, **kwargs ):

        if( includeSelf ):
            if( self.standardChanged ):
                kwargs.update( { 'params': self.params } )
            else:
                kwargs.update( { 'natParams': self.natParams } )

        if( includePrior ):
            if( self.prior.standardChanged ):
                kwargs.update( { 'priorParams': self.prior.params } )
            else:
                kwargs.update( { 'priorNatParams': self.prior.natParams } )

        return kwargs

    ##########################################################################

    @property
    def params( self ):
        if( self.naturalChanged ):
            self._params = self.natToStandard( *self.natParams )
            self.naturalChanged = False
        return self._params

    @params.setter
    def params( self, val ):
        self.standardChanged = True
        self._params = val

    @property
    def natParams( self ):
        if( self.standardChanged ):
            self._natParams = self.standardToNat( *self.params )
            self.standardChanged = False
        return self._natParams

    @natParams.setter
    def natParams( self, val ):
        self.naturalChanged = True
        self._natParams = val

    ##########################################################################

    @classmethod
    @abstractmethod
    def standardToNat( cls, *params ):
        pass

    @classmethod
    @abstractmethod
    def natToStandard( cls, *natParams ):
        pass

    ##########################################################################

    @classmethod
    @checkExpFamArgs
    @abstractmethod
    def sufficientStats( cls, x, constParams=None, forPost=False ):
        # Compute T( x ).  forPost is True if this is being
        # used for something related to the posterior.
        pass

    ##########################################################################

    @classmethod
    @checkExpFamArgs
    @abstractmethod
    def log_partition( cls, x, params=None, natParams=None, split=False ):
        # The terms that make up the log partition
        pass

    def ilog_partition( self, x, split=False ):
        return self.log_partition( x, **self.paramChoices( includeSelf=True, includePrior=False, split=split ) )

    ##########################################################################

    @classmethod
    @checkExpFamArgs
    def paramSample( cls, priorParams=None, priorNatParams=None, size=1 ):
        # Sample from P( Ѳ; α )
        return cls.priorClass.sample( params=priorParams, natParams=priorNatParams, size=size )

    ##########################################################################

    @classmethod
    @checkExpFamArgs
    @multiParamSample
    def jointSample( cls, priorParams=None, priorNatParams=None, size=1 ):
        # Sample from P( x, Ѳ; α )
        theta = cls.paramSample( priorParams=priorParams, priorNatParams=priorNatParams )
        x = cls.sample( params=theta )
        return x, theta

    ##########################################################################

    @classmethod
    @checkExpFamArgs
    def posteriorPriorNatParams( cls, x, constParams=None, priorParams=None, priorNatParams=None ):
        stats = cls.sufficientStats( x, constParams=constParams, forPost=True )
        priorNatParams = priorNatParams if priorNatParams is not None else cls.priorClass.standardToNat( *priorParams )
        return np.add( stats, priorNatParams )

    ##########################################################################

    @classmethod
    @checkExpFamArgs
    def posteriorSample( cls, x, constParams=None, priorParams=None, priorNatParams=None, size=1 ):
        # Sample from P( Ѳ | x; α )
        postNatParams = cls.posteriorPriorNatParams( x, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams )
        return cls.priorClass.sample( natParams=postNatParams, size=size )

    def iposteriorSample( self, x, size=1 ):
        return self.posteriorSample( x, **self.paramChoices( includeSelf=False, includePrior=True, constParams=self.constParams, size=size ) )

    ####################################################################################################################################################

    @classmethod
    @checkExpFamArgs
    def log_likelihoodExpFam( cls, x, constParams=None, params=None, natParams=None ):
        natParams = natParams if natParams is not None else cls.standardToNat( *params )
        stats = cls.sufficientStats( x, constParams=constParams )
        dataN = cls.dataN( x )
        part = cls.log_partition( x, natParams=natParams ) * dataN
        return cls.log_pdf( natParams, stats, part )

    def ilog_likelihood( self, x, expFam=False ):
        if( expFam ):
            return self.log_likelihoodExpFam( x, constParams=self.constParams, natParams=self.natParams )
        return super( ExponentialFam, self ).ilog_likelihood( x )

    ##########################################################################

    @classmethod
    @checkExpFamArgs
    def log_params( cls, params=None, natParams=None, priorParams=None, priorNatParams=None ):
        # Compute P( Ѳ; α )
        params = params if params is not None else cls.natToStandard( *natParams )
        return cls.priorClass.log_likelihood( params, params=priorParams, natParams=priorNatParams )

    def ilog_params( self, expFam=False ):
        return self.prior.ilog_likelihood( self.params, expFam=expFam )

    ##########################################################################

    @classmethod
    @checkExpFamArgs
    def log_jointExpFam( cls, x, params=None, natParams=None, constParams=None, priorParams=None, priorNatParams=None ):
        postNatParams = cls.posteriorPriorNatParams( x, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams )

        params = params if params is not None else cls.natToStandard( *natParams )
        stat = cls.priorClass.sufficientStats( params, constParams=constParams )
        part = cls.priorClass.log_partition( params, params=priorParams, natParams=priorNatParams, split=True )

        return cls.log_pdf( postNatParams, stat, part )

    @classmethod
    @checkExpFamArgs
    def log_joint( cls, x, params=None, natParams=None, priorParams=None, priorNatParams=None ):
        # Compute P( x, Ѳ; α )
        return cls.log_params( params=params, natParams=natParams, priorParams=priorParams, priorNatParams=priorNatParams ) + \
               cls.log_likelihood( x, params=params, natParams=natParams )

    def ilog_joint( self, x, expFam=False ):
        if( expFam ):
            return self.log_jointExpFam( x, params=self.params, constParams=self.constParams, priorNatParams=self.prior.natParams )
        return super( ExponentialFam, self ).ilog_joint( x )

    ##########################################################################

    @classmethod
    def log_posteriorExpFam( cls, x, params=None, natParams=None, constParams=None, priorParams=None, priorNatParams=None ):
        assert ( params is None ) ^ ( natParams is None ) and ( priorParams is None ) ^ ( priorNatParams is None )

        postNatParams = cls.posteriorPriorNatParams( x, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams )

        params = params if params is not None else cls.natToStandard( *natParams )
        stat = cls.priorClass.sufficientStats( params, constParams=constParams )
        part = cls.priorClass.log_partition( params, natParams=postNatParams, split=True )

        return cls.log_pdf( postNatParams, stat, part )

    @classmethod
    @checkExpFamArgs
    def log_posterior( cls, x, params=None, natParams=None, constParams=None, priorParams=None, priorNatParams=None, ravel=False ):
        # Compute P( Ѳ | x; α )
        params = params if params is not None else cls.natToStandard( *natParams )
        postNatParams = cls.posteriorPriorNatParams( x, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams )
        return cls.priorClass.log_likelihood( params, natParams=postNatParams, ravel=ravel )

    def ilog_posterior( self, x, expFam=False ):
        if( expFam ):
            return self.log_posteriorExpFam( x, params=self.params, constParams=self.constParams, priorNatParams=self.prior.natParams )
        return self.log_posterior( x, **self.paramChoices( includeSelf=True, includePrior=True, constParams=self.constParams ) )

    ##########################################################################

    @classmethod
    @checkExpFamArgs
    def log_marginal( cls, x, params=None, natParams=None, constParams=None, priorParams=None, priorNatParams=None, ravel=False ):
        # Compute P( x; α )
        params = params if params is not None else cls.natToStandard( *natParams )

        likelihood = cls.log_likelihood( x, params=params, natParams=natParams, ravel=ravel )
        posterior = cls.log_posterior( x, params=params, natParams=natParams, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams, ravel=ravel )
        params = cls.log_params( params=params, natParams=natParams, priorParams=priorParams, priorNatParams=priorNatParams, ravel=ravel )
        return likelihood + params - posterior

    ##########################################################################

    @classmethod
    def log_pdf( cls, natParams, sufficientStats, log_partition=None ):

        ans = 0.0
        for natParam, stat in zip( natParams, sufficientStats ):
            ans += ( natParam * stat ).sum()

        if( log_partition is not None ):
            if( isinstance( log_partition, tuple ) ):
                ans -= sum( log_partition )
            else:
                ans -= log_partition

        return ans

    ##########################################################################

    def paramNaturalTest( self ):
        params = self.params
        params2 = self.natToStandard( *self.standardToNat( *params ) )
        for p1, p2 in zip( params, params2 ):
            assert np.allclose( p1, p2 )

    def likelihoodNoPartitionTestExpFam( self ):
        x = self.isample( size=10 )
        ans1 = self.ilog_likelihood( x, expFam=True )
        trueAns1 = self.ilog_likelihood( x )

        x = self.isample( size=10 )
        ans2 = self.ilog_likelihood( x, expFam=True )
        trueAns2 = self.ilog_likelihood( x )
        assert np.isclose( ans1 - ans2, trueAns1 - trueAns2 ), ( ans1 - ans2 ) - ( trueAns1 - trueAns2 )

    def likelihoodTestExpFam( self ):
        x = self.isample( size=10 )
        ans1 = self.ilog_likelihood( x, expFam=True )
        ans2 = self.ilog_likelihood( x )
        assert np.isclose( ans1, ans2 ), ans1 - ans2

    def paramTestExpFam( self ):
        self.prior.likelihoodTestExpFam()

    def jointTestExpFam( self ):
        x = self.isample( size=10 )
        ans1 = self.ilog_joint( x, expFam=True )
        ans2 = self.ilog_joint( x )
        assert np.isclose( ans1, ans2 ), ans1 - ans2

    def posteriorTestExpFam( self ):
        x = self.isample( size=10 )
        ans1 = self.ilog_posterior( x, expFam=True )
        ans2 = self.ilog_posterior( x )
        assert np.isclose( ans1, ans2 ), ans1 - ans2

####################################################################################################################################

class TensorExponentialFam( ExponentialFam ):

    def __init__( self, *params, prior=None, hypers=None ):
        super( TensorExponentialFam, self ).__init__( *params, prior=prior, hypers=hypers )

    @classmethod
    @abstractmethod
    def combine( cls, stat, nat, size=None ):
        pass

    @classmethod
    def log_pdf( cls, natParams, sufficientStats, log_partition=None ):

        ans = 0.0
        for natParam, stat in zip( natParams, sufficientStats ):
            ans += cls.combine( stat, natParam )

        if( log_partition is not None ):
            if( isinstance( log_partition, tuple ) ):
                ans -= sum( log_partition )
            else:
                ans -= log_partition

        return ans

    def paramNaturalTest( self ):
        params = self.params
        params2 = self.natToStandard( *self.standardToNat( *params ) )
        for p1, p2 in zip( params, params2 ):
            if( isinstance( p1, tuple ) or isinstance( p1, list ) ):
                for _p1, _p2 in zip( p1, p2 ):
                    assert np.allclose( _p1, _p2 )
            else:
                assert np.allclose( p1, p2 )
