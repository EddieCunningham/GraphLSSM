from abc import ABC, abstractmethod
import numpy as np
from GenModels.GM.Utility import *
import string
import tqdm

__all__ = [ 'Distribution', \
            'Conjugate', \
            'ExponentialFam' ]

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
    def dataN( cls, x ):
        # This is necessary to know how to tell the size of an output
        pass

    ##########################################################################

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
            theta = cls.paramSample( priorParams=priorParams )
            x = cls.sample( params=theta )
            xs[ i ] = x
            thetas[ i ] = theta
        return xs, thetas

    def ijointSample( self, size=1 ):
        return self.jointSample( self.prior.params, size=size )

    @classmethod
    def gibbsJointSample( cls, priorParams, burnIn=5000, skip=100, size=1, verbose=True ):
        params = cls.paramSample( priorParams )
        it = range( burnIn )
        if( verbose == True ):
            it = tqdm.tqdm( it, desc='Burn in' )
        for _ in it:
            x = cls.sample( params )
            params = cls.posteriorSample( x, priorParams )

        xResult = [ None for _ in range( size ) ]
        pResult = [ None for _ in range( size ) ]

        it = range( size )
        if( verbose == True ):
            it = tqdm.tqdm( it, desc='Gibbs (once every %d)'%( skip ) )

        for i in it:
            x = cls.sample( params )
            params = cls.posteriorSample( x, priorParams )
            xResult[ i ] = x
            pResult[ i ] = params
            for _ in range( skip ):
                x = cls.sample( params )
                params = cls.posteriorSample( x, priorParams )

        return np.array( xResult ), np.array( pResult )

    def igibbsJointSample( self, burnIn=100, skip=10, size=1 ):
        return self.gibbsJointSample( self.prior.params, burnIn=burnIn, skip=skip, size=size )

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
            self.params = self.iparamSample()
        else:
            self.params = self.iposteriorSample( x )

    ##########################################################################

    def metropolisHastings( self, x=None, maxN=10000, burnIn=3000, size=1000, skip=50, verbose=True, concatX=True ):

        x = self.isample()

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

####################################################################################################################################

class Conjugate( Distribution ):
    # Fill this in at some point
    pass

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
    @abstractmethod
    def sufficientStats( cls, x, constParams=None, forPost=False ):
        # Compute T( x ).  forPost is True if this is being
        # used for something related to the posterior.
        pass

    ##########################################################################

    @classmethod
    @abstractmethod
    def log_partition( cls, x, params=None, natParams=None, split=False ):
        # The terms that make up the log partition
        pass

    def ilog_partition( self, x, split=False ):
        if( self.standardChanged ):
            return self.log_partition( x, params=self.params, split=split )
        return self.log_partition( x, natParams=self.natParams, split=split )

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
            theta = cls.paramSample( priorParams=priorParams, priorNatParams=priorNatParams )
            x = cls.sample( params=theta )
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

        stats = cls.sufficientStats( x, constParams=constParams, forPost=True )
        priorNatParams = priorNatParams if priorNatParams is not None else cls.priorClass.standardToNat( *priorParams )
        return np.add( stats, priorNatParams )

    ##########################################################################

    @classmethod
    def posteriorSample( cls, x, constParams=None, priorParams=None, priorNatParams=None, size=1 ):
        # Sample from P( Ѳ | x; α )
        assert ( priorParams is None ) ^ ( priorNatParams is None )

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
        stats = cls.sufficientStats( x, constParams=constParams )
        dataN = cls.dataN( x )
        part = cls.log_partition( x, natParams=natParams ) * dataN
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
        return cls.priorClass.log_likelihood( params, params=priorParams, natParams=priorNatParams )

    def ilog_params( self, expFam=False ):
        return self.prior.ilog_likelihood( self.params, expFam=expFam )

    ##########################################################################

    @classmethod
    def log_jointExpFam( cls, x, params=None, natParams=None, constParams=None, priorParams=None, priorNatParams=None ):
        assert ( params is None ) ^ ( natParams is None ) and ( priorParams is None ) ^ ( priorNatParams is None )

        postNatParams = cls.posteriorPriorNatParams( x, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams )

        params = params if params is not None else cls.natToStandard( *natParams )
        stat = cls.priorClass.sufficientStats( params, constParams=constParams )
        part = cls.priorClass.log_partition( params, params=priorParams, natParams=priorNatParams, split=True )

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
    def gibbsJointSample( cls, constParams=None, priorParams=None, priorNatParams=None, burnIn=5000, skip=100, size=1, verbose=True ):
        assert ( priorParams is None ) ^ ( priorNatParams is None )

        params = cls.paramSample( priorParams=priorParams, priorNatParams=priorNatParams )
        it = range( burnIn )
        if( verbose == True ):
            it = tqdm.tqdm( it, desc='Burn in' )
        for _ in it:
            x = cls.sample( params=params )
            params = cls.posteriorSample( x, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams )

        xResult = [ None for _ in range( size ) ]
        pResult = [ None for _ in range( size ) ]

        it = range( size )
        if( verbose == True ):
            it = tqdm.tqdm( it, desc='Gibbs (once every %d)'%( skip ) )

        for i in it:
            x = cls.sample( params=params )
            params = cls.posteriorSample( x, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams )
            xResult[ i ] = x
            pResult[ i ] = params
            for _ in range( skip ):
                x = cls.sample( params=params )
                params = cls.posteriorSample( x, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams )

        return xResult, pResult

    def igibbsJointSample( self, burnIn=100, skip=10, size=1 ):
        if( self.prior.standardChanged ):
            return self.gibbsJointSample( constParams=self.constParams, priorParams=self.prior.params, burnIn=burnIn, skip=skip, size=size )
        return self.gibbsJointSample( constParams=self.constParams, priorNatParams=self.prior.natParams, burnIn=burnIn, skip=skip, size=size )

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
    def log_posterior( cls, x, params=None, natParams=None, constParams=None, priorParams=None, priorNatParams=None ):
        # Compute P( Ѳ | x; α )
        assert ( params is None ) ^ ( natParams is None ) and ( priorParams is None ) ^ ( priorNatParams is None )
        params = params if params is not None else cls.natToStandard( *natParams )
        postNatParams = cls.posteriorPriorNatParams( x, constParams=constParams, priorParams=priorParams, priorNatParams=priorNatParams )
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
        for natParam, stat in zip( natParams, sufficientStats ):
            ans += ( natParam * stat ).sum()

        if( log_partition is not None ):
            if( isinstance( log_partition, tuple ) ):
                ans -= sum( log_partition )
            else:
                ans -= log_partition

        return ans

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
