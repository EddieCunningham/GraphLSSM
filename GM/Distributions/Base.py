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
            self.prior = self.priorClass( **hypers )

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
    def dataN( cls, x, constParams=None ):
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
        self.nat_params

        # Set the mean field params and natural params
        self.mfNatParams = self.nat_params
        self.mfParams

    ##########################################################################

    @property
    def params( self ):
        if( self.naturalChanged ):
            self._params = self.natToStandard( *self.nat_params )
            self.naturalChanged = False
        return self._params

    @params.setter
    def params( self, val ):
        self.standardChanged = True
        self.naturalChanged = False
        self._params = val

    @property
    def nat_params( self ):
        if( self.standardChanged ):
            self._nat_params = self.standardToNat( *self.params )
            self.standardChanged = False
        return self._nat_params

    @nat_params.setter
    def nat_params( self, val ):
        self.naturalChanged = True
        self.standardChanged = False
        self._nat_params = val

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
    def natToStandard( cls, *nat_params ):
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
    def log_partition( cls, x=None, params=None, nat_params=None, split=False ):
        # The terms that make up the log partition.  x is in case the base measure
        # needs x
        pass

    def ilog_partition( self, x=None, split=False ):
        if( self.standardChanged ):
            return self.log_partition( x=x, params=self.params, split=split )
        return self.log_partition( x=x, nat_params=self.nat_params, split=split )

    ##########################################################################

    @classmethod
    def expectedNatParams( cls, priorParams=None, priorNatParams=None ):
        # This is for when we want to do variational inference.
        # Use the fact that in conjugate models, the conjugate prior
        # t( x ) = ( n, -logZ ) for the child
        expectedNatParams, exptectedPartition = cls.priorClass.log_partitionGradient( params=priorParams, nat_params=priorNatParams, split=True )
        return expectedNatParams

    def iexpectedNatParams( self, useMeanField=False ):
        if( useMeanField == False ):
            if( self.standardChanged ):
                return self.expectedNatParams( priorParams=self.prior.params )
            return self.expectedNatParams( priorNatParams=self.prior.nat_params )
        else:
            if( self.mfStandardChanged ):
                return self.expectedNatParams( priorParams=self.prior.mfParams )
            return self.expectedNatParams( priorNatParams=self.prior.mfNatParams )

    ##########################################################################

    @classmethod
    @abstractmethod
    def log_partitionGradient( cls, params=None, nat_params=None, split=False ):
        # This is the expected sufficient statistic E_{ p( x | n ) }[ T( x ) ]
        pass

    @abstractmethod
    def _testLogPartitionGradient( self ):
        # Use autograd here to check
        pass

    def ilog_partitionGradient( self ):
        return self.log_partitionGradient( nat_params=self.nat_params )

    @classmethod
    def expectedSufficientStats( cls, params=None, nat_params=None ):
        return cls.log_partitionGradient( params=params, nat_params=nat_params )

    def iexpectedSufficientStats( self ):
        return self.ilog_partitionGradient()

    ##########################################################################

    @classmethod
    def entropy( cls, params=None, nat_params=None ):
        # H[ p ] = -E_{ p( x | n ) }[ logP( x | n ) ]
        assert ( params is None ) ^ ( nat_params is None )
        nat_params = nat_params if nat_params is not None else cls.standardToNat( *params )

        expectedStats = cls.expectedSufficientStats( nat_params=nat_params )
        log_partition = cls.log_partition( nat_params=nat_params )

        return -sum( [ ( s * n ).sum() for s, n in zip( expectedStats, nat_params ) ] ) + log_partition

    def ientropy( self ):
        if( self.standardChanged ):
            return self.entropy( params=self.params )
        return self.entropy( nat_params=self.nat_params )

    ##########################################################################

    @classmethod
    def score( cls, x, params=None, nat_params=None, constParams=None ):
        # v( n, x ) = d/dn{ logP( x | n ) }
        cls.checkShape( x )
        stats = cls.sufficientStats( x, constParams=constParams )
        part = cls.log_partitionGradient( params=params, nat_params=nat_params )
        assert len( stats ) == len( part )
        return [ s - p for s, p in zip( stats, part ) ]

    def iscore( self, x ):
        if( self.standardChanged ):
            return self.score( x, params=self.params, constParams=self.constParams )
        return self.score( x, nat_params=self.nat_params, constParams=self.constParams )

    ##########################################################################

    @classmethod
    # @abstractmethod
    def fisherInfo( cls, x=None, params=None, nat_params=None ):
        # This is the fisher information matrix
        # TODO (probably not worth it unless I figure out how to use autograd for it)
        pass

    def ifisherInfo( self, x=None ):
        return self.fisherInfo( x=x, nat_params=self.nat_params )

    ##########################################################################

    @classmethod
    def log_statMGF( cls, s, x=None, params=None, nat_params=None ):
        assert ( params is None ) ^ ( nat_params is None )
        nat_params = nat_params if nat_params is not None else cls.standardToNat( *params )
        nS = [ _s + n for _s, n in zip( s, nat_params ) ]
        return cls.log_partition( x=x, nat_params=nS ) - cls.log_partition( x=x, nat_params=nat_params )

    def ilog_statMGF( self, s, x=None ):
        # x is only required when the base measure is a function of x
        return self.log_MGF( s, x=x, nat_params=self.nat_params )

    ##########################################################################

    @classmethod
    def KLDivergence( cls, params1=None, nat_params1=None, params2=None, nat_params2=None ):
        assert ( params1 is None ) ^ ( nat_params1 is None )
        assert ( params2 is None ) ^ ( nat_params2 is None )
        nat_params1 = nat_params1 if nat_params1 is not None else cls.standardToNat( *params1 )
        nat_params2 = nat_params2 if nat_params2 is not None else cls.standardToNat( *params2 )
        assert len( nat_params1 ) == len( nat_params2 )

        natDiff = []
        for n1, n2 in zip( nat_params1, nat_params2 ):
            assert n1.shape == n2.shape
            natDiff.append( n1 - n2 )

        ans = 0.0
        for n, p in zip( natDiff, cls.log_partitionGradient( nat_params=nat_params1 ) ):
            ans += ( n * p ).sum()

        ans -= cls.log_partition( nat_params=nat_params1 )
        ans += cls.log_partition( nat_params=nat_params2 )
        return ans

    def iKLDivergence( self, otherParams=None, otherNatParams=None, other=None ):
        if( other is not None ):
            assert isinstance( other, ExponentialFam )
            return self.KLDivergence( nat_params1=self.nat_params, nat_params2=other.nat_params )

        assert ( otherParams is None ) ^ ( otherNatParams is None )
        return self.KLDivergence( nat_params1=self.nat_params, params2=otherParams, nat_params2=otherNatParams )

    ####################################################################################################################################################

    @classmethod
    def sample( cls, params=None, nat_params=None, size=1 ):
        # Sample from P( x | Ѳ; α )
        assert ( params is None ) ^ ( nat_params is None )

    def isample( self, size=1 ):
        if( self.standardChanged ):
            return self.sample( params=self.params, size=size )
        return self.sample( nat_params=self.nat_params, size=size )

    ##########################################################################

    @classmethod
    def paramSample( cls, priorParams=None, priorNatParams=None, size=1 ):
        # Sample from P( Ѳ; α )
        assert ( priorParams is None ) ^ ( priorNatParams is None )
        return cls.priorClass.sample( params=priorParams, nat_params=priorNatParams, size=size )

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
        return self.jointSample( priorNatParams=self.prior.nat_params, size=size )

    ##########################################################################

    @classmethod
    def posteriorPriorNatParams( cls, x=None, constParams=None, priorParams=None, priorNatParams=None, stats=None ):
        assert ( priorParams is None ) ^ ( priorNatParams is None )
        assert ( x is None ) ^ ( stats is None )

        priorNatParams = priorNatParams if priorNatParams is not None else cls.priorClass.standardToNat( *priorParams )

        if( x is not None ):
            cls.checkShape( x )
            stats = cls.sufficientStats( x, constParams=constParams )
            dataN = cls.dataN( x, constParams=constParams )
            stats = stats + tuple( [ dataN for _ in range( len( priorNatParams ) - len( stats ) ) ] )

        return [ np.add( s, p ) for s, p in zip( stats, priorNatParams ) ]

    @classmethod
    def variationalPosteriorPriorNatParams( cls, ys=None, constParams=None, params=None, nat_params=None, priorParams=None, priorNatParams=None, returnNormalizer=False ):
        assert ( params is None ) ^ ( nat_params is None )
        assert ( priorParams is None ) ^ ( priorNatParams is None )

        # Because this will only be used to do variational inference,
        # make sure that the observed data is passed in
        assert ys is not None

        expectedStats, normalizer = cls.expectedSufficientStats( ys=ys, params=params, nat_params=nat_params, returnNormalizer=True )

        # Assume that these are variational parameters
        priorNatParams = priorNatParams if priorNatParams is not None else cls.priorClass.standardToNat( *priorParams )

        dataN = cls.dataN( ys, conditionOnY=True, checkY=True, constParams=constParams )
        expectedStats = expectedStats + tuple( [ dataN for _ in range( len( priorNatParams ) - len( expectedStats ) ) ] )

        ans = [ np.add( s, p ) for s, p in zip( expectedStats, priorNatParams ) ]
        return ans if returnNormalizer == False else ( ans, normalizer )

    ##########################################################################

    @classmethod
    def posteriorSample( cls, x=None, constParams=None, priorParams=None, priorNatParams=None, size=1, stats=None ):
        # Sample from P( Ѳ | x; α )
        assert ( priorParams is None ) ^ ( priorNatParams is None )
        assert ( x is None ) ^ ( stats is None )

        if( x is not None ):
            cls.checkShape( x )
        postNatParams = cls.posteriorPriorNatParams( x=x, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams, stats=stats )
        return cls.priorClass.sample( nat_params=postNatParams, size=size )

    def iposteriorSample( self, x=None, stats=None, size=1 ):
        assert ( x is None ) ^ ( stats is None )
        if( self.prior.standardChanged ):
            return self.posteriorSample( x=x, constParams=self.constParams, priorParams=self.prior.params, size=size, stats=stats )
        return self.posteriorSample( x=x, constParams=self.constParams, priorNatParams=self.prior.nat_params, size=size, stats=stats )

    ####################################################################################################################################################

    @classmethod
    def log_likelihoodExpFam( cls, x, constParams=None, params=None, nat_params=None ):
        assert ( params is None ) ^ ( nat_params is None )
        nat_params = nat_params if nat_params is not None else cls.standardToNat( *params )
        cls.checkShape( x )
        stats = cls.sufficientStats( x, constParams=constParams )
        dataN = cls.dataN( x, constParams=constParams )
        part = cls.log_partition( x, nat_params=nat_params ) * dataN
        assert isinstance( part, Iterable ) == False

        return cls.log_pdf( nat_params, stats, part )

    @classmethod
    def log_likelihood( cls, x, params=None, nat_params=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( nat_params is None )

    def ilog_likelihood( self, x, expFam=False ):
        if( expFam ):
            return self.log_likelihoodExpFam( x, constParams=self.constParams, nat_params=self.nat_params )
        if( self.standardChanged ):
            return self.log_likelihood( x, params=self.params )
        return self.log_likelihood( x, nat_params=self.nat_params )

    ##########################################################################

    @classmethod
    def log_params( cls, params=None, nat_params=None, priorParams=None, priorNatParams=None ):
        # Compute P( Ѳ; α )
        assert ( params is None ) ^ ( nat_params is None ) and ( priorParams is None ) ^ ( priorNatParams is None )
        params = params if params is not None else cls.natToStandard( *nat_params )
        cls.priorClass.checkShape( params )
        return cls.priorClass.log_likelihood( params, params=priorParams, nat_params=priorNatParams )

    def ilog_params( self, expFam=False ):
        return self.prior.ilog_likelihood( self.params, expFam=expFam )

    ##########################################################################

    @classmethod
    def log_jointExpFam( cls, x, params=None, nat_params=None, constParams=None, priorParams=None, priorNatParams=None ):
        assert ( params is None ) ^ ( nat_params is None ) and ( priorParams is None ) ^ ( priorNatParams is None )

        cls.checkShape( x )
        postNatParams = cls.posteriorPriorNatParams( x, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams )

        params = params if params is not None else cls.natToStandard( *nat_params )
        cls.priorClass.checkShape( params )
        stat = cls.priorClass.sufficientStats( params, constParams=constParams )
        part = cls.priorClass.log_partition( params, params=priorParams, nat_params=priorNatParams )

        return cls.log_pdf( postNatParams, stat, part )

    @classmethod
    def log_joint( cls, x, params=None, nat_params=None, priorParams=None, priorNatParams=None ):
        # Compute P( x, Ѳ; α )
        assert ( params is None ) ^ ( nat_params is None ) and ( priorParams is None ) ^ ( priorNatParams is None )

        return cls.log_params( params=params, nat_params=nat_params, priorParams=priorParams, priorNatParams=priorNatParams ) + \
               cls.log_likelihood( x, params=params, nat_params=nat_params )

    def ilog_joint( self, x, expFam=False ):
        if( expFam ):
            return self.log_jointExpFam( x, params=self.params, constParams=self.constParams, priorNatParams=self.prior.nat_params )
        if( self.standardChanged ):
            if( self.prior.standardChanged ):
                return self.log_joint( x, params=self.params, priorParams=self.prior.params )
            return self.log_joint( x, params=self.params, priorNatParams=self.prior.nat_params )
        if( self.prior.standardChanged ):
            return self.log_joint( x, nat_params=self.nat_params, priorParams=self.prior.params )
        return self.log_joint( x, nat_params=self.nat_params, priorNatParams=self.prior.nat_params )

    ##########################################################################

    @classmethod
    def log_posteriorExpFam( cls, x, params=None, nat_params=None, constParams=None, priorParams=None, priorNatParams=None, stats=None ):
        assert ( params is None ) ^ ( nat_params is None ) and ( priorParams is None ) ^ ( priorNatParams is None )
        assert ( x is None ) ^ ( stats is None )

        if( x is not None ):
            cls.checkShape( x )
        postNatParams = cls.posteriorPriorNatParams( x=x, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams, stats=stats )

        params = params if params is not None else cls.natToStandard( *nat_params )
        cls.priorClass.checkShape( params )
        stat = cls.priorClass.sufficientStats( params, constParams=constParams )
        part = cls.priorClass.log_partition( params, nat_params=postNatParams, split=False )
        return cls.log_pdf( postNatParams, stat, part )

        return cls.priorClass.log_pdf( postNatParams, stat, part )

    @classmethod
    def log_posterior( cls, x=None, params=None, nat_params=None, constParams=None, priorParams=None, priorNatParams=None, stats=None ):
        # Compute P( Ѳ | x; α )
        assert ( params is None ) ^ ( nat_params is None ) and ( priorParams is None ) ^ ( priorNatParams is None )
        assert ( x is None ) ^ ( stats is None )
        params = params if params is not None else cls.natToStandard( *nat_params )

        if( x is not None ):
            cls.checkShape( x )
        postNatParams = cls.posteriorPriorNatParams( x=x, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams, stats=stats )

        assert cls.priorClass.dataN( params, constParams=constParams ) == 1
        return cls.priorClass.log_likelihood( params, nat_params=postNatParams )

    def ilog_posterior( self, x=None, expFam=False, stats=None ):
        assert ( x is None ) ^ ( stats is None )
        if( expFam ):
            return self.log_posteriorExpFam( x=x, params=self.params, constParams=self.constParams, priorNatParams=self.prior.nat_params, stats=stats )
        if( self.standardChanged ):
            if( self.prior.standardChanged ):
                return self.log_posterior( x=x, params=self.params, constParams=self.constParams, priorParams=self.prior.params, stats=stats )
            return self.log_posterior( x=x, params=self.params, constParams=self.constParams, priorNatParams=self.prior.nat_params, stats=stats )
        if( self.prior.standardChanged ):
            return self.log_posterior( x=x, nat_params=self.nat_params, constParams=self.constParams, priorParams=self.prior.params, stats=stats )
        return self.log_posterior( x=x, nat_params=self.nat_params, constParams=self.constParams, priorNatParams=self.prior.nat_params, stats=stats )

    ##########################################################################

    @classmethod
    def log_marginal( cls, x, params=None, nat_params=None, constParams=None, priorParams=None, priorNatParams=None ):
        # Compute P( x; α )
        assert ( params is None ) ^ ( nat_params is None ) and ( priorParams is None ) ^ ( priorNatParams is None )
        params = params if params is not None else cls.natToStandard( *nat_params )

        cls.checkShape( x )

        likelihood = cls.log_likelihood( x, params=params, nat_params=nat_params )
        posterior = cls.log_posterior( x, params=params, nat_params=nat_params, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams )
        params = cls.log_params( params=params, nat_params=nat_params, priorParams=priorParams, priorNatParams=priorNatParams )
        return likelihood + params - posterior

    def ilog_marginal( self, x ):
        if( self.standardChanged ):
            if( self.prior.standardChanged ):
                return self.log_marginal( x, params=self.params, constParams=self.constParams, priorParams=self.prior.params )
            return self.log_marginal( x, params=self.params, constParams=self.constParams, priorNatParams=self.prior.nat_params )
        if( self.prior.standardChanged ):
            return self.log_marginal( x, nat_params=self.nat_params, constParams=self.constParams, priorParams=self.prior.params )
        return self.log_marginal( x, nat_params=self.nat_params, constParams=self.constParams, priorNatParams=self.prior.nat_params )

    ##########################################################################

    @classmethod
    def log_pdf( cls, nat_params, sufficientStats, log_partition=None ):

        ans = 0.0
        for i, ( natParam, stat ) in enumerate( zip( nat_params, sufficientStats ) ):
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
    def mode( cls, params=None, nat_params=None ):
        raise NotImplementedError

    @classmethod
    def MAP( cls, x=None, priorParams=None, priorNatParams=None ):
        assert ( priorParams is None ) ^ ( priorNatParams is None )
        cls.checkShape( x )
        postNatParams = cls.posteriorPriorNatParams( x, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams )
        return cls.priorClass.mode( nat_params=postNatParams )

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
    def log_pdf( cls, nat_params, sufficientStats, log_partition=None ):

        ans = 0.0
        for natParam, stat in zip( nat_params, sufficientStats ):
            ans += cls.combine( stat, natParam )

        if( log_partition is not None ):
            if( isinstance( log_partition, tuple ) ):
                ans -= sum( log_partition )
            else:
                ans -= log_partition

        return ans
