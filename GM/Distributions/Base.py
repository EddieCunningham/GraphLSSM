from abc import ABC, abstractmethod
import numpy as np
from GenModels.GM.Utility import *
import string
import tqdm
from functools import wraps
from collections import Iterable
import inspect

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
    def dataN( cls, x, ravel=False ):
        if( isinstance( x, list ) ):
            return len( x )
        return 1

    @classmethod
    @abstractmethod
    def paramShapes( cls, **D ):
        pass

    @classmethod
    @abstractmethod
    def inferDims( cls, params=None ):
        pass

    @classmethod
    @abstractmethod
    def outputShapes( cls, **D ):
        pass

    @classmethod
    def unravelSample( cls, x, params ):
        Ds = cls.inferDims( params=params )
        shapes = cls.outputShapes( **Ds )
        split = np.split( x, [ np.prod( s ) for s in shapes[ :-1 ] ] )
        ans = []
        for item, s in zip( split, shapes ):
            assert item.size == np.prod( s )
            ans.append( item.reshape( s ) )
        return ans if len( ans ) > 1 else ans[ 0 ]

    @classmethod
    def unravelSampleAndParams( cls, xTheta ):
        pass

    ##########################################################################

    @classmethod
    @fullSampleSupport
    @abstractmethod
    def sample( cls, params ):
        # Sample a single element from P( x | Ѳ; α )
        pass

    @fullSampleSupport
    def isample( self ):
        return self.sample( **self.paramChoices( includeSelf=True, includePrior=False ) )

    @classmethod
    @fullLikelihoodSupport
    @abstractmethod
    def log_likelihood( cls, x, params ):
        # Compute P( x | Ѳ; α )
        pass

    def ilog_likelihood( self, x, **kwargs ):
        return self.log_likelihood( x, **self.paramChoices( includeSelf=True, includePrior=False, **kwargs ) )

    ##########################################################################

    @classmethod
    def jointSample( cls, priorParams=None, **D ):
        # Sample from P( x, Ѳ; α ).  This won't support ravelling or multi sampling
        params = cls.paramSample( priorParams=priorParams, **D )
        x = cls.sample( params=params )
        return x, params

    def ijointSample( self ):
        return self.jointSample( **self.paramChoices( includeSelf=False, includePrior=True ) )

    @classmethod
    def log_joint( cls, x, params, priorParams=None ):
        # Compute P( x, Ѳ; α )
        return cls.log_likelihood( x, params ) + cls.log_params( params, priorParams=priorParams )

    def ilog_joint( self, x, params ):
        return self.log_joint( x, params, **self.paramChoices( includeSelf=False, includePrior=True ) )
        # return self.log_joint( x, params, **self.paramChoices( includeSelf=True, includePrior=True ) )

    ##########################################################################

    @classmethod
    @abstractmethod
    def posteriorSample( cls, x, priorParams, constParams=None ):
        # Sample from P( Ѳ | x; α )
        pass

    def iposteriorSample( self, x ):
        return self.posteriorSample( x, **self.paramChoices( includeSelf=False, includePrior=True, constParams=self.constParams ) )

    @classmethod
    @abstractmethod
    def log_posterior( cls, x, params, priorParams ):
        # Compute P( Ѳ | x; α )
        pass

    def ilog_posterior( self, x ):
        return self.log_posterior( x, self.params, self.prior.params )

    ##########################################################################

    @classmethod
    def easyParamSample( cls, **D ):
        paramShapes = cls.paramShapes( **D )
        assert isinstance( paramShapes, list ) or isinstance( paramSample, tuple )
        params = []
        for s in paramShapes:
            if( not isinstance( s, list ) and not isinstance( s, tuple ) ):
                sample = s
            elif( len( s ) == 2 and s[ 0 ] == s[ 1 ] ):
                sample = np.eye( s[ 0 ] )
            else:
                sample = np.zeros( s )
            params.append( sample )
        return tuple( params )

    @classmethod
    def paramSample( cls, priorParams=None, **D ):
        # Sample from P( Ѳ; α )
        if( cls.priorClass == None ):
            return cls.easyParamSample( **D )
        return cls.priorClass.sample( priorParams, **D )

    def iparamSample( self ):
        if( self.priorClass == None ):
            D = self.inferDims( params=self.params )
            return self.easyParamSample( **D )
        return self.prior.isample()

    @classmethod
    def log_params( cls, params, priorParams=None ):
        # Compute P( Ѳ; α )
        if( cls.priorClass == None ):
            return 0.0
        return cls.priorClass.log_likelihood( params, priorParams )

    def ilog_params( self ):
        return self.log_params( self.params, self.prior.params )

    ##########################################################################

    @classmethod
    def log_marginal( cls, x, params, priorParams ):
        likelihood = cls.log_likelihood( x, params )
        posterior = cls.log_posterior( x, params, priorParams )
        params = cls.log_params( params, priorParams )
        return likelihood + params - posterior

    def ilog_marginal( self, x ):
        if( self.priorClass == None ):
            return 0.0
        return self.log_marginal( x, **self.paramChoices( includeSelf=True, includePrior=True, constParams=self.constParams ) )

    ##########################################################################

    def resample( self, x=None ):
        if( x is None ):
            self.params = self.iparamSample()
        else:
            self.params = self.iposteriorSample( x )

    ##########################################################################

    def metropolisHastings( self, maxN=10000, burnIn=1000, size=1, skip=50, verbose=True, concatX=True ):

        x = self.isample( ravel=True )
        p = self.ilog_likelihood( x, ravel=True )

        maxN = max( maxN, burnIn + size * skip )
        it = range( maxN )
        if( verbose ):
            print( 'On class ', self )
            it = tqdm.tqdm( it, desc='Metropolis Hastings (once every %d)'%( skip ) )

        samples = []

        for i in it:
            candidate = randomStep( x )
            pCandidate = self.ilog_likelihood( candidate, ravel=True )
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

####################################################################################################################################

class Conjugate( Distribution ):
    # Fill this in at some point
    pass

####################################################################################################################################

@doublewrap
def addNatParams( func, includeSelf=True, includePrior=False, allowNone=False ):

    @wraps( func )
    def wrapper( *args, natParams=None, priorNatParams=None, **kwargs ):

        if( includeSelf ):
            kwargs[ 'natParams' ] = natParams
        if( includePrior ):
            kwargs[ 'includePrior' ] = priorNatParams

        bound_arguments = sig.bind( *args, **kwargs )
        bound_arguments.apply_defaults()
        if( includeSelf and allowNone == False ):
            assert ( bound_arguments[ 'params' ] is None ) ^ ( bound_arguments[ 'natParams' ] is None )
        if( includePrior and allowNone == False ):
            assert ( bound_arguments[ 'params' ] is None ) ^ ( bound_arguments[ 'natParams' ] is None )

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
    def paramSample( cls, priorParams=None, priorNatParams=None, **D ):
        # Sample from P( Ѳ; α )
        if( cls.priorClass == None ):
            return cls.easyParamSample( **D )
        return cls.priorClass.sample( params=priorParams, natParams=priorNatParams, **D )

    ##########################################################################

    @classmethod
    @checkExpFamArgs
    def jointSample( cls, priorParams=None, priorNatParams=None, **D ):
        # Sample from P( x, Ѳ; α )
        theta = cls.paramSample( priorParams=priorParams, priorNatParams=priorNatParams, **D )
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
    def posteriorSample( cls, x, constParams=None, priorParams=None, priorNatParams=None ):
        # Sample from P( Ѳ | x; α )
        postNatParams = cls.posteriorPriorNatParams( x, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams )
        return cls.priorClass.sample( natParams=postNatParams )

    ####################################################################################################################################################

    @classmethod
    @checkExpFamArgs
    def log_likelihoodExpFam( cls, x, constParams=None, params=None, natParams=None ):
        natParams = natParams if natParams is not None else cls.standardToNat( *params )
        stats = cls.sufficientStats( x, constParams=constParams )
        dataN = cls.dataN( x )
        part = cls.log_partition( x, natParams=natParams ) * dataN
        return cls.log_pdf( natParams, stats, part )

    @classmethod
    @checkExpFamArgs
    def log_jointExpFam( cls, x, params=None, natParams=None, constParams=None, priorParams=None, priorNatParams=None ):
        postNatParams = cls.posteriorPriorNatParams( x, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams )

        params = params if params is not None else cls.natToStandard( *natParams )
        stat = cls.priorClass.sufficientStats( params, constParams=constParams )
        part = cls.priorClass.log_partition( params, params=priorParams, natParams=priorNatParams, split=True )

        return cls.log_pdf( postNatParams, stat, part )

    @classmethod
    def log_posteriorExpFam( cls, x, params=None, natParams=None, constParams=None, priorParams=None, priorNatParams=None ):
        assert ( params is None ) ^ ( natParams is None ) and ( priorParams is None ) ^ ( priorNatParams is None )

        postNatParams = cls.posteriorPriorNatParams( x, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams )

        params = params if params is not None else cls.natToStandard( *natParams )
        stat = cls.priorClass.sufficientStats( params, constParams=constParams )
        part = cls.priorClass.log_partition( params, natParams=postNatParams, split=True )

        return cls.log_pdf( postNatParams, stat, part )

    ##########################################################################

    def ilog_likelihood( self, x, expFam=False, **kwargs ):
        if( expFam ):
            return self.log_likelihoodExpFam( x, constParams=self.constParams, natParams=self.natParams )
        return super( ExponentialFam, self ).ilog_likelihood( x, **kwargs )

    ##########################################################################

    @classmethod
    @checkExpFamArgs
    def log_params( cls, params=None, natParams=None, priorParams=None, priorNatParams=None ):
        # Compute P( Ѳ; α )
        if( cls.priorClass == None ):
            return 0.0
        params = params if params is not None else cls.natToStandard( *natParams )
        return cls.priorClass.log_likelihood( params, params=priorParams, natParams=priorNatParams )

    def ilog_params( self, expFam=False ):
        return self.prior.ilog_likelihood( self.params, expFam=expFam )

    ##########################################################################

    @classmethod
    @checkExpFamArgs
    def log_joint( cls, x, params=None, natParams=None, priorParams=None, priorNatParams=None ):
        # Compute P( x, Ѳ; α )
        return cls.log_params( params=params, natParams=natParams, priorParams=priorParams, priorNatParams=priorNatParams ) + \
               cls.log_likelihood( x, params=params, natParams=natParams )

    def ilog_joint( self, x, expFam=False ):
        if( expFam ):
            return self.log_jointExpFam( x, params=self.params, constParams=self.constParams, priorNatParams=self.prior.natParams )
        return super( ExponentialFam, self ).ilog_joint( x, params=self.params )

    ##########################################################################

    @classmethod
    @checkExpFamArgs
    def log_posterior( cls, x, params=None, natParams=None, constParams=None, priorParams=None, priorNatParams=None ):
        # Compute P( Ѳ | x; α )
        params = params if params is not None else cls.natToStandard( *natParams )
        postNatParams = cls.posteriorPriorNatParams( x, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams )
        return cls.priorClass.log_likelihood( params, natParams=postNatParams )

    def ilog_posterior( self, x, expFam=False ):
        if( expFam ):
            return self.log_posteriorExpFam( x, params=self.params, constParams=self.constParams, priorNatParams=self.prior.natParams )
        return self.log_posterior( x, **self.paramChoices( includeSelf=True, includePrior=True, constParams=self.constParams ) )

    ##########################################################################

    @classmethod
    @checkExpFamArgs
    def log_marginal( cls, x, params=None, natParams=None, constParams=None, priorParams=None, priorNatParams=None ):
        # Compute P( x; α )
        params = params if params is not None else cls.natToStandard( *natParams )

        likelihood = cls.log_likelihood( x, params=params, natParams=natParams )
        posterior = cls.log_posterior( x, params=params, natParams=natParams, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams )
        params = cls.log_params( params=params, natParams=natParams, priorParams=priorParams, priorNatParams=priorNatParams )
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
    def combine( cls, stat, nat ):
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
