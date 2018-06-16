from abc import ABC, abstractmethod
import numpy as np
from GenModels.GM.Utility import *
import string
import tqdm
from collections import Iterable
from GenModels.GM.Distributions.Mixins.Sampling import _GibbsMixin, _MetropolisHastingMixin
from autograd import jacobian
from functools import partial

__all__ = [ 'Distribution', \
            'Conjugate', \
            'ExponentialFam' ]

class Distribution( ABC, _GibbsMixin, _MetropolisHastingMixin ):

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

    ##########################################################################

    @classmethod
    @abstractmethod
    def dataN( cls, x ):
        # This is necessary to know how to tell the size of an output
        pass

    @classmethod
    @abstractmethod
    def unpackSingleSample( cls, x ):
        # This is for convenience when we know size == 1
        pass

    @classmethod
    @abstractmethod
    def sampleShapes( cls ):
        # Specify the shape of a sample
        pass

    @abstractmethod
    def isampleShapes( cls ):
        # Specify the shape of a sample when the parameters are known
        pass

    @abstractmethod
    def checkShape( cls, x ):
        # Check that the shape of a sample is correct
        pass

    ##########################################################################

    @classmethod
    @abstractmethod
    def generate( cls, *dims, size=1, **kwargs ):
        # Generate samples easily without having to specify parameters
        pass

    @classmethod
    @abstractmethod
    def sample( cls, params, size=1 ):
        # Sample from P( x | Ѳ; α )
        pass

    def isample( self, size=1 ):
        return self.sample( self.params, size=size )

    @classmethod
    @abstractmethod
    def log_likelihood( cls, x, params ):
        # Compute P( x | Ѳ; α )
        pass

    def ilog_likelihood( self, x ):
        return self.log_likelihood( x, self.params )

    ##########################################################################

    @classmethod
    def jointSample( cls, priorParams, size=1 ):
        # Sample from P( x, Ѳ; α )
        xs = [ None for _ in range( size ) ]
        thetas = [ None for _ in range( size ) ]
        for i in range( size ):
            theta = cls.priorClass.unpackSingleSample( cls.paramSample( priorParams=priorParams ) )
            x = cls.unpackSingleSample( cls.sample( params=theta ) )
            xs[ i ] = x
            thetas[ i ] = theta
        return xs, thetas

    def ijointSample( self, size=1 ):
        return self.jointSample( self.prior.params, size=size )

    @classmethod
    def log_joint( cls, x, params, priorParams ):
        # Compute P( x, Ѳ; α )
        return cls.log_likelihood( x, params ) + cls.log_params( params, priorParams )

    def ilog_joint( self, x ):
        return self.log_joint( x, self.params )

    ##########################################################################

    @classmethod
    @abstractmethod
    def posteriorSample( cls, x, priorParams, size=1 ):
        # Sample from P( Ѳ | x; α )
        pass

    def iposteriorSample( self, x, size=1 ):
        return self.posteriorSample( x, self.prior.params, size=size )

    @classmethod
    @abstractmethod
    def log_posterior( cls, x, params, priorParams ):
        # Compute P( Ѳ | x; α )
        pass

    def ilog_posterior( self, x ):
        return self.log_posterior( x, self.params, self.prior.params )

    ##########################################################################

    @classmethod
    def paramSample( cls, priorParams, size=1 ):
        # Sample from P( Ѳ; α )
        return cls.priorClass.sample( priorParams, size=size )

    def iparamSample( self, size=1 ):
        return self.prior.isample( size=size )

    @classmethod
    def log_params( cls, params, priorParams ):
        # Compute P( Ѳ; α )
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
        return self.log_marginal( x, self.params, self.prior.params )

    ##########################################################################

    def resample( self, x=None ):
        if( x is None ):
            newParams = self.priorClass.unpackSingleSample( self.iparamSample() )
        else:
            newParams = self.priorClass.unpackSingleSample( self.iposteriorSample( x ) )

        if( isinstance( newParams, np.ndarray ) ):
            self.params = ( newParams, )
        else:
            self.params = newParams

    ##########################################################################

    @classmethod
    def mode( cls, params=None ):
        raise NotImplementedError

    @classmethod
    def maxLikelihood( cls, x ):
        raise NotImplementedError

    @classmethod
    def MAP( cls, x ):
        raise NotImplementedError

####################################################################################################################################

class Conjugate( Distribution ):
    # Fill this in at some point
    pass

####################################################################################################################################

class ExponentialFam( Conjugate ):

    def __init__( self, *params, prior=None, hypers=None ):

        self.standardChanged = False
        self.naturalChanged = False
        self.mfStandardChanged = False
        self.mfNaturalChanged = False

        super( ExponentialFam, self ).__init__( *params, prior=prior, hypers=hypers )

        # Set the natural parameters
        self.natParams

        # Set the mean field params and natural params
        self.mfNatParams = self.natParams
        self.mfParams

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
        self.naturalChanged = False
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
        self.standardChanged = False
        self._natParams = val

    ##########################################################################
    ## Mean field parameters for variational inference ##

    @property
    def mfParams( self ):
        if( self.mfNaturalChanged ):
            self._mfParams = self.natToStandard( *self.mfNatParams )
            self.mfNaturalChanged = False
        return self._mfParams

    @mfParams.setter
    def mfParams( self, val ):
        self.mfStandardChanged = True
        self._mfParams = val

    @property
    def mfNatParams( self ):
        if( self.mfStandardChanged ):
            self._mfNatParams = self.standardToNat( *self.mfParams )
            self.mfStandardChanged = False
        return self._mfNatParams

    @mfNatParams.setter
    def mfNatParams( self, val ):
        self.mfNaturalChanged = True
        self._mfNatParams = val

    ##########################################################################

    @classmethod
    @abstractmethod
    def standardToNat( cls, *params ):
        pass

    @classmethod
    @abstractmethod
    def natToStandard( cls, *natParams ):
        pass

    ####################################################################################################################################################

    @classmethod
    @abstractmethod
    def sufficientStats( cls, x, constParams=None ):
        # Compute T( x )
        pass

    ##########################################################################

    @classmethod
    @abstractmethod
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # The terms that make up the log partition.  x is in case the base measure
        # needs x
        pass

    def ilog_partition( self, x=None, split=False ):
        if( self.standardChanged ):
            return self.log_partition( x=x, params=self.params, split=split )
        return self.log_partition( x=x, natParams=self.natParams, split=split )

    ##########################################################################

    @classmethod
    def expectedNatParams( cls, priorParams=None, priorNatParams=None ):
        # This is for when we want to do variational inference.
        # Use the fact that in conjugate models, the conjugate prior
        # t( x ) = ( n, -logZ ) for the child
        expectedNatParams, exptectedPartition = cls.priorClass.log_partitionGradient( params=priorParams, natParams=priorNatParams, split=True )
        return expectedNatParams

    def iexpectedNatParams( self, useMeanField=False ):
        if( useMeanField == False ):
            if( self.standardChanged ):
                return self.expectedNatParams( priorParams=self.prior.params )
            return self.expectedNatParams( priorNatParams=self.prior.natParams )
        else:
            if( self.mfStandardChanged ):
                return self.expectedNatParams( priorParams=self.prior.mfParams )
            return self.expectedNatParams( priorNatParams=self.prior.mfNatParams )

    ##########################################################################

    @classmethod
    @abstractmethod
    def log_partitionGradient( cls, params=None, natParams=None, split=False ):
        # This is the expected sufficient statistic E_{ p( x | n ) }[ T( x ) ]
        pass

    @abstractmethod
    def _testLogPartitionGradient( self ):
        # Use autograd here to check
        pass

    def ilog_partitionGradient( self ):
        return self.log_partitionGradient( natParams=self.natParams )

    @classmethod
    def expectedSufficientStats( cls, params=None, natParams=None ):
        return cls.log_partitionGradient( params=params, natParams=natParams )

    def iexpectedSufficientStats( self ):
        return self.ilog_partitionGradient()

    ##########################################################################

    @classmethod
    def score( cls, x, params=None, natParams=None, constParams=None ):
        # v( n, x ) = d/dn{ logP( x | n ) }
        cls.checkShape( x )
        stats = cls.sufficientStats( x, constParams=constParams )
        part = cls.log_partitionGradient( params=params, natParams=natParams )
        assert len( stats ) == len( part )
        return [ s - p for s, p in zip( stats, part ) ]

    def iscore( self, x ):
        if( self.standardChanged ):
            return self.score( x, params=self.params, constParams=self.constParams )
        return self.score( x, natParams=self.natParams, constParams=self.constParams )

    ##########################################################################

    @classmethod
    # @abstractmethod
    def fisherInfo( cls, x=None, params=None, natParams=None ):
        # This is the fisher information matrix
        # TODO (probably not worth it unless I figure out how to use autograd for it)
        pass

    def ifisherInfo( self, x=None ):
        return self.fisherInfo( x=x, natParams=self.natParams )

    ##########################################################################

    @classmethod
    def log_statMGF( cls, s, x=None, params=None, natParams=None ):
        assert ( params is None ) ^ ( natParams is None )
        natParams = natParams if natParams is not None else cls.standardToNat( *params )
        nS = [ _s + n for _s, n in zip( s, natParams ) ]
        return cls.log_partition( x=x, natParams=nS ) - cls.log_partition( x=x, natParams=natParams )

    def ilog_statMGF( self, s, x=None ):
        # x is only required when the base measure is a function of x
        return self.log_MGF( s, x=x, natParams=self.natParams )

    ##########################################################################

    @classmethod
    def KLDivergence( cls, params1=None, natParams1=None, params2=None, natParams2=None ):
        assert ( params1 is None ) ^ ( natParams1 is None )
        assert ( params2 is None ) ^ ( natParams2 is None )
        natParams1 = natParams1 if natParams1 is not None else cls.standardToNat( *params1 )
        natParams2 = natParams2 if natParams2 is not None else cls.standardToNat( *params2 )
        assert len( natParams1 ) == len( natParams2 )

        natDiff = []
        for n1, n2 in zip( natParams1, natParams2 ):
            assert n1.shape == n2.shape
            natDiff.append( n1 - n2 )

        ans = 0.0
        for n, p in zip( natDiff, cls.log_partitionGradient( natParams=natParams1 ) ):
            ans += ( n * p ).sum()

        ans -= cls.log_partition( natParams=natParams1 )
        ans += cls.log_partition( natParams=natParams2 )
        return ans

    def iKLDivergence( self, otherParams=None, otherNatParams=None, other=None ):
        if( other is not None ):
            assert isinstance( other, ExponentialFam )
            return self.KLDivergence( natParams1=self.natParams, natParams2=other.natParams )

        assert ( otherParams is None ) ^ ( otherNatParams is None )
        return self.KLDivergence( natParams1=self.natParams, params2=otherParams, natParams2=otherNatParams )

    ####################################################################################################################################################

    @classmethod
    def sample( cls, params=None, natParams=None, size=1 ):
        # Sample from P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )

    def isample( self, size=1 ):
        if( self.standardChanged ):
            return self.sample( params=self.params, size=size )
        return self.sample( natParams=self.natParams, size=size )

    ##########################################################################

    @classmethod
    def paramSample( cls, priorParams=None, priorNatParams=None, size=1 ):
        # Sample from P( Ѳ; α )
        assert ( priorParams is None ) ^ ( priorNatParams is None )
        return cls.priorClass.sample( params=priorParams, natParams=priorNatParams, size=size )

    def iparamSample( self, size=1 ):
        return self.prior.isample( size=size )

    ##########################################################################

    @classmethod
    def jointSample( cls, priorParams=None, priorNatParams=None, size=1 ):
        # Sample from P( x, Ѳ; α )
        assert ( priorParams is None ) ^ ( priorNatParams is None )
        xs = [ None for _ in range( size ) ]
        thetas = [ None for _ in range( size ) ]
        for i in range( size ):
            theta = cls.priorClass.unpackSingleSample( cls.paramSample( priorParams=priorParams, priorNatParams=priorNatParams ) )
            x = cls.unpackSingleSample( cls.sample( params=theta ) )
            xs[ i ] = x
            thetas[ i ] = theta
        return xs, thetas

    def ijointSample( self, size=1 ):
        if( self.prior.standardChanged ):
            return self.jointSample( priorParams=self.prior.params, size=size )
        return self.jointSample( priorNatParams=self.prior.natParams, size=size )

    ##########################################################################

    @classmethod
    def posteriorPriorNatParams( cls, x, constParams=None, priorParams=None, priorNatParams=None ):
        assert ( priorParams is None ) ^ ( priorNatParams is None )

        cls.checkShape( x )
        stats = cls.sufficientStats( x, constParams=constParams )
        priorNatParams = priorNatParams if priorNatParams is not None else cls.priorClass.standardToNat( *priorParams )

        dataN = cls.dataN( x )
        stats = stats + tuple( [ dataN for _ in range( len( priorNatParams ) - len( stats ) ) ] )

        return [ np.add( s, p ) for s, p in zip( stats, priorNatParams ) ]

    @classmethod
    def variationalPosteriorPriorNatParams( cls, ys=None, constParams=None, params=None, natParams=None, priorParams=None, priorNatParams=None, returnNormalizer=False ):
        assert ( params is None ) ^ ( natParams is None )
        assert ( priorParams is None ) ^ ( priorNatParams is None )

        # Because this will only be used to do variational inference,
        # make sure that the observed data is passed in
        assert ys is not None

        expectedStats, normalizer = cls.expectedSufficientStats( ys=ys, params=params, natParams=natParams, returnNormalizer=True )
        # for t in expectedStats:
        #     print( t )

        # Assume that these are variational parameters
        priorNatParams = priorNatParams if priorNatParams is not None else cls.priorClass.standardToNat( *priorParams )

        # for p in priorNatParams:
        #     print( np.exp( p ) )

        dataN = cls.dataN( ys, conditionOnY=True, checkY=True )
        expectedStats = expectedStats + tuple( [ dataN for _ in range( len( priorNatParams ) - len( expectedStats ) ) ] )

        ans = [ np.add( s, p ) for s, p in zip( expectedStats, priorNatParams ) ]
        # for a in ans:
        #     print( np.exp( a ) )
        # assert 0
        return ans if returnNormalizer == False else ( ans, normalizer )

    ##########################################################################

    @classmethod
    def posteriorSample( cls, x, constParams=None, priorParams=None, priorNatParams=None, size=1 ):
        # Sample from P( Ѳ | x; α )
        assert ( priorParams is None ) ^ ( priorNatParams is None )

        cls.checkShape( x )
        postNatParams = cls.posteriorPriorNatParams( x, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams )
        return cls.priorClass.sample( natParams=postNatParams, size=size )

    def iposteriorSample( self, x, size=1 ):
        if( self.prior.standardChanged ):
            return self.posteriorSample( x, constParams=self.constParams, priorParams=self.prior.params, size=size )
        return self.posteriorSample( x, constParams=self.constParams, priorNatParams=self.prior.natParams, size=size )

    ####################################################################################################################################################

    @classmethod
    def log_likelihoodExpFam( cls, x, constParams=None, params=None, natParams=None ):
        assert ( params is None ) ^ ( natParams is None )
        natParams = natParams if natParams is not None else cls.standardToNat( *params )
        cls.checkShape( x )
        stats = cls.sufficientStats( x, constParams=constParams )
        dataN = cls.dataN( x )
        part = cls.log_partition( x, natParams=natParams ) * dataN
        assert isinstance( part, Iterable ) == False

        return cls.log_pdf( natParams, stats, part )

    @classmethod
    def log_likelihood( cls, x, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )

    def ilog_likelihood( self, x, expFam=False ):
        if( expFam ):
            return self.log_likelihoodExpFam( x, constParams=self.constParams, natParams=self.natParams )
        if( self.standardChanged ):
            return self.log_likelihood( x, params=self.params )
        return self.log_likelihood( x, natParams=self.natParams )

    ##########################################################################

    @classmethod
    def log_params( cls, params=None, natParams=None, priorParams=None, priorNatParams=None ):
        # Compute P( Ѳ; α )
        assert ( params is None ) ^ ( natParams is None ) and ( priorParams is None ) ^ ( priorNatParams is None )
        params = params if params is not None else cls.natToStandard( *natParams )
        cls.priorClass.checkShape( params )
        return cls.priorClass.log_likelihood( params, params=priorParams, natParams=priorNatParams )

    def ilog_params( self, expFam=False ):
        return self.prior.ilog_likelihood( self.params, expFam=expFam )

    ##########################################################################

    @classmethod
    def log_jointExpFam( cls, x, params=None, natParams=None, constParams=None, priorParams=None, priorNatParams=None ):
        assert ( params is None ) ^ ( natParams is None ) and ( priorParams is None ) ^ ( priorNatParams is None )

        cls.checkShape( x )
        postNatParams = cls.posteriorPriorNatParams( x, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams )

        params = params if params is not None else cls.natToStandard( *natParams )
        cls.priorClass.checkShape( params )
        stat = cls.priorClass.sufficientStats( params, constParams=constParams )
        part = cls.priorClass.log_partition( params, params=priorParams, natParams=priorNatParams )

        return cls.log_pdf( postNatParams, stat, part )

    @classmethod
    def log_joint( cls, x, params=None, natParams=None, priorParams=None, priorNatParams=None ):
        # Compute P( x, Ѳ; α )
        assert ( params is None ) ^ ( natParams is None ) and ( priorParams is None ) ^ ( priorNatParams is None )

        return cls.log_params( params=params, natParams=natParams, priorParams=priorParams, priorNatParams=priorNatParams ) + \
               cls.log_likelihood( x, params=params, natParams=natParams )

    def ilog_joint( self, x, expFam=False ):
        if( expFam ):
            return self.log_jointExpFam( x, params=self.params, constParams=self.constParams, priorNatParams=self.prior.natParams )
        if( self.standardChanged ):
            if( self.prior.standardChanged ):
                return self.log_joint( x, params=self.params, priorParams=self.prior.params )
            return self.log_joint( x, params=self.params, priorNatParams=self.prior.natParams )
        if( self.prior.standardChanged ):
            return self.log_joint( x, natParams=self.natParams, priorParams=self.prior.params )
        return self.log_joint( x, natParams=self.natParams, priorNatParams=self.prior.natParams )

    ##########################################################################

    @classmethod
    def log_posteriorExpFam( cls, x, params=None, natParams=None, constParams=None, priorParams=None, priorNatParams=None ):
        assert ( params is None ) ^ ( natParams is None ) and ( priorParams is None ) ^ ( priorNatParams is None )

        cls.checkShape( x )
        postNatParams = cls.posteriorPriorNatParams( x, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams )

        params = params if params is not None else cls.natToStandard( *natParams )
        cls.priorClass.checkShape( params )
        stat = cls.priorClass.sufficientStats( params, constParams=constParams )
        part = cls.priorClass.log_partition( params, natParams=postNatParams )

        return cls.log_pdf( postNatParams, stat, part )

    @classmethod
    def log_posterior( cls, x, params=None, natParams=None, constParams=None, priorParams=None, priorNatParams=None ):
        # Compute P( Ѳ | x; α )
        assert ( params is None ) ^ ( natParams is None ) and ( priorParams is None ) ^ ( priorNatParams is None )
        params = params if params is not None else cls.natToStandard( *natParams )
        cls.checkShape( x )
        postNatParams = cls.posteriorPriorNatParams( x, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams )

        assert cls.priorClass.dataN( params ) == 1
        return cls.priorClass.log_likelihood( params, natParams=postNatParams )

    def ilog_posterior( self, x, expFam=False ):
        if( expFam ):
            return self.log_posteriorExpFam( x, params=self.params, constParams=self.constParams, priorNatParams=self.prior.natParams )
        if( self.standardChanged ):
            if( self.prior.standardChanged ):
                return self.log_posterior( x, params=self.params, constParams=self.constParams, priorParams=self.prior.params )
            return self.log_posterior( x, params=self.params, constParams=self.constParams, priorNatParams=self.prior.natParams )
        if( self.prior.standardChanged ):
            return self.log_posterior( x, natParams=self.natParams, constParams=self.constParams, priorParams=self.prior.params )
        return self.log_posterior( x, natParams=self.natParams, constParams=self.constParams, priorNatParams=self.prior.natParams )

    ##########################################################################

    @classmethod
    def log_marginal( cls, x, params=None, natParams=None, constParams=None, priorParams=None, priorNatParams=None ):
        # Compute P( x; α )
        assert ( params is None ) ^ ( natParams is None ) and ( priorParams is None ) ^ ( priorNatParams is None )
        params = params if params is not None else cls.natToStandard( *natParams )

        cls.checkShape( x )

        likelihood = cls.log_likelihood( x, params=params, natParams=natParams )
        posterior = cls.log_posterior( x, params=params, natParams=natParams, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams )
        params = cls.log_params( params=params, natParams=natParams, priorParams=priorParams, priorNatParams=priorNatParams )
        return likelihood + params - posterior

    def ilog_marginal( self, x ):
        if( self.standardChanged ):
            if( self.prior.standardChanged ):
                return self.log_marginal( x, params=self.params, constParams=self.constParams, priorParams=self.prior.params )
            return self.log_marginal( x, params=self.params, constParams=self.constParams, priorNatParams=self.prior.natParams )
        if( self.prior.standardChanged ):
            return self.log_marginal( x, natParams=self.natParams, constParams=self.constParams, priorParams=self.prior.params )
        return self.log_marginal( x, natParams=self.natParams, constParams=self.constParams, priorNatParams=self.prior.natParams )

    ##########################################################################

    @classmethod
    def log_pdf( cls, natParams, sufficientStats, log_partition=None ):

        ans = 0.0
        for i, ( natParam, stat ) in enumerate( zip( natParams, sufficientStats ) ):
            ans += ( natParam * stat ).sum()

        if( log_partition is not None ):
            if( isinstance( log_partition, tuple ) ):
                ans -= sum( log_partition )
            else:
                ans -= log_partition

        assert isinstance( ans, Iterable ) == False, log_partition

        return ans

    ##########################################################################

    @classmethod
    def mode( cls, params=None, natParams=None ):
        raise NotImplementedError

    @classmethod
    def MAP( cls, x=None, priorParams=None, priorNatParams=None ):
        assert ( priorParams is None ) ^ ( priorNatParams is None )
        cls.checkShape( x )
        postNatParams = cls.posteriorPriorNatParams( x, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams )
        return cls.priorClass.mode( natParams=postNatParams )

    ##########################################################################

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
