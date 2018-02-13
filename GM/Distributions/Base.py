import numpy as np
__all__ = [ 'Distribution', \
            'Conjugate', \
            'Exponential' ]

class Distribution():

    def __init__( self, *args, **kwargs ):
        pass

    @property
    def params( self ):
        pass

    @params.setter
    def params( self, val ):
        pass

    ##########################################################################

    @classmethod
    def sample( cls, params ):
        # Sample from P( x | Ѳ )
        pass

    def isample( self ):
        return self.sample( self.params )

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, params ):
        # Compute P( x | Ѳ )
        pass

    def ilog_likelihood( self, x ):
        return self.log_likelihood( x, self.params )

####################################################################################################################################

class Conjugate( Distribution ):

    priorClass = None

    def __init__( self, prior=None, hypers=None ):
        if( prior is not None ):
            self.prior = prior
        elif( hypers is not None ):
            self.prior = self.priorClass( *hypers )

    ##########################################################################

    @classmethod
    def postPriorParams( cls, x, params ):
        pass

    def ipostPriorParams( x ):
        self.postPriorParams( x, self.params )

    ##########################################################################

    @classmethod
    def posteriorSample( cls, x, params ):
        # Sample from P( Ѳ | x; α )

        posteriorParams = cls.postPriorParams( x, *params )
        return cls.priorClass.sample( posteriorParams )

    def iposteriorSample( x ):
        return self.posteriorSample( x, self.params )

####################################################################################################################################

class Exponential( Conjugate ):

    def __init__( self, *params, prior=None, hypers=None ):
        super( Exponential, self ).__init__( prior=prior, hypers=hypers )

        self.standardChanged = False
        self.naturalChanged = False

        # Set the parameters
        if( prior is None and hypers is None ):
            self.params = params
        else:
            self.params = self.iparamSample()

        # Set the natural parameters implicitely
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
    def standardToNat( cls, *params ):
        pass

    @classmethod
    def natToStandard( cls, *natParams ):
        pass

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, forPost=False ):
        # Compute T( x ).  forPost is True if this is being
        # used for something related to the posterior.
        assert ( params is None ) ^ ( natParams is None )

    ##########################################################################

    @classmethod
    def log_partition( cls, x, params=None, natParams=None, split=False ):
        # The terms that make up the log partition
        assert ( params is None ) ^ ( natParams is None )

    def ilog_partition( self, x, split=False ):
        if( self.standardChanged ):
            return self.log_partition( x, params=self.params, split=split )
        return self.log_partition( x, natParams=self.natParams, split=split )

    ####################################################################################################################################################

    @classmethod
    def sample( cls, params=None, natParams=None ):
        # Sample from P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )

    def isample( self ):
        if( self.standardChanged ):
            return self.sample( params=self.params )
        return self.sample( natParams=self.natParams )

    ##########################################################################

    @classmethod
    def paramSample( cls, priorParams=None, priorNatParams=None ):
        # Sample from P( Ѳ; α )
        assert ( priorParams is None ) ^ ( priorNatParams is None )
        return cls.priorClass.sample( priorParams=priorParams, priorNatParams=priorNatParams )

    def iparamSample( self ):
        return self.prior.isample()

    ##########################################################################

    @classmethod
    def jointSample( cls, priorParams=None, priorNatParams=None ):
        # Sample from P( x, Ѳ; α )
        assert ( priorParams is None ) ^ ( priorNatParams is None )
        theta = cls.paramSample( priorParams=priorParams, priorNatParams=priorNatParams )
        x = cls.log_likelihood( x, params=theta )
        return x, theta

    def ijointSample( self ):
        if( self.prior.standardChanged ):
            return self.jointSample( priorParams=self.prior.params )
        return self.jointSample( priorNatParams=self.prior.natParams )

    ##########################################################################
    @classmethod
    def posteriorPriorNatParams( cls, x, priorParams=None, priorNatParams=None ):
        assert ( priorParams is None ) ^ ( priorNatParams is None )

        stats = cls.sufficientStats( x, forPost=True )
        priorNatParams = priorNatParams if priorNatParams is not None else cls.priorClass.standardToNat( *priorParams )
        return np.add( stats, priorNatParams )

    ##########################################################################

    @classmethod
    def posteriorSample( cls, x, priorParams=None, priorNatParams=None ):
        # Sample from P( Ѳ | x; α )
        assert ( priorParams is None ) ^ ( priorNatParams is None )

        postNatParams = cls.posteriorPriorNatParams( x, priorParams=priorParams, priorNatParams=priorNatParams )
        return cls.priorClass.sample( natParams=postNatParams )

    def iposteriorSample( self, x ):
        if( self.prior.standardChanged ):
            return self.posteriorSample( x, priorParams=self.prior.params )
        return self.posteriorSample( x, priorNatParams=self.prior.natParams )

    ####################################################################################################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )

    def ilog_likelihood( self, x ):
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

    def ilog_params( self ):
        return self.prior.ilog_likelihood( self.params )

    ##########################################################################

    @classmethod
    def log_joint( cls, x, params=None, natParams=None, priorParams=None, priorNatParams=None ):
        # Compute P( x, Ѳ; α )
        assert ( params is None ) ^ ( natParams is None ) and ( priorParams is None ) ^ ( priorNatParams is None )

        return cls.log_params( params=params, natParams=natParams, priorParams=priorParams, priorNatParams=priorNatParams ) + \
               cls.log_likelihood( x, params=params, natParams=natParams )

    def ilog_joint( self, x ):
        if( self.standardChanged ):
            if( self.prior.standardChanged ):
                return self.log_joint( x, params=self.params, priorParams=self.prior.params )
            return self.log_joint( x, params=self.params, priorNatParams=self.prior.natParams )
        if( self.prior.standardChanged ):
            return self.log_joint( x, natParams=self.natParams, priorParams=self.prior.params )
        return self.log_joint( x, natParams=self.natParams, priorNatParams=self.prior.natParams )

    ##########################################################################

    @classmethod
    def log_posterior( cls, x, params=None, natParams=None, priorParams=None, priorNatParams=None ):
        # Compute P( Ѳ | x; α )
        assert ( params is None ) ^ ( natParams is None ) and ( priorParams is None ) ^ ( priorNatParams is None )

        params = params if params is not None else cls.natToStandard( *natParams )
        postNatParams = cls.posteriorPriorNatParams( x, priorParams=priorParams, priorNatParams=priorNatParams )
        return cls.priorClass.log_likelihood( params, natParams=postNatParams )

    def ilog_posterior( self, x ):
        if( self.standardChanged ):
            if( self.prior.standardChanged ):
                return self.log_posterior( x, params=self.params, priorParams=self.prior.params )
            return self.log_posterior( x, params=self.params, priorNatParams=self.prior.natParams )
        if( self.prior.standardChanged ):
            return self.log_posterior( x, natParams=self.natParams, priorParams=self.prior.params )
        return self.log_posterior( x, natParams=self.natParams, priorNatParams=self.prior.natParams )

    ##########################################################################

    @staticmethod
    def log_pdf( natParams, sufficientStats, log_partition=None ):

        ans = 0.0
        for natParam, stat in zip( natParams, sufficientStats ):
            ans += ( natParam * stat ).sum()

        if( log_partition is not None ):
            ans -= sum( log_partition )

        return ans