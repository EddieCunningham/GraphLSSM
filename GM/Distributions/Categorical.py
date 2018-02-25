import numpy as np
from Base import ExponentialFam
from Dirichlet import Dirichlet


class Categorical( ExponentialFam ):

    priorClass = Dirichlet

    def __init__( self, p=None, prior=None, hypers=None ):
        super( Categorical, self ).__init__( p, prior=prior, hypers=hypers )

    ##########################################################################

    @property
    def p( self ):
        return self._params[ 0 ]

    ##########################################################################

    @classmethod
    def standardToNat( cls, p ):
        n = np.log( p )
        return ( n, )

    @classmethod
    def natToStandard( cls, n ):
        p = np.exp( n )
        return ( p, )

    ##########################################################################

    @classmethod
    def sufficientStats( cls, x, D=None, forPost=False ):
        # Compute T( x )
        assert isinstance( x, np.ndarray ) and x.ndim == 1
        assert D is not None
        t1 = np.bincount( x, minlength=D )
        return ( t1, )

    @classmethod
    def log_partition( cls, x=None, params=None, natParams=None, split=False ):
        # Compute A( Ѳ ) - log( h( x ) )
        assert ( params is None ) ^ ( natParams is None )
        if( split ):
            return ( 0, )
        return 0

    ##########################################################################

    @classmethod
    def sample( cls, params=None, natParams=None, size=1 ):
        # Sample from P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )
        if( params is not None ):
            if( not isinstance( params, tuple ) or \
                not isinstance( params, list ) ):
                params = ( params, )

        ( p, ) = params if params is not None else cls.natToStandard( *natParams )
        return np.random.choice( p.shape[ 0 ], size, p=p )

    ##########################################################################

    @classmethod
    def log_likelihood( cls, x, params=None, natParams=None ):
        # Compute P( x | Ѳ; α )
        assert ( params is None ) ^ ( natParams is None )
        ( p, ) = params if params is not None else cls.natToStandard( *natParams )
        if( isinstance( x, np.ndarray ) ):
            assert x.size == 1
            x = x[ 0 ]
        return np.log( p[ x ] )

    ##########################################################################

    @classmethod
    def posteriorPriorNatParams( cls, x, D=None, priorParams=None, priorNatParams=None ):
        assert ( priorParams is None ) ^ ( priorNatParams is None )
        assert D is not None

        stats = cls.sufficientStats( x, D=D, forPost=True )
        priorNatParams = priorNatParams if priorNatParams is not None else cls.priorClass.standardToNat( *priorParams )
        return np.add( stats, priorNatParams )


    @classmethod
    def log_likelihoodExpFam( cls, x, D=None, params=None, natParams=None ):
        assert ( params is None ) ^ ( natParams is None )
        assert D is not None
        natParams = natParams if natParams is not None else cls.standardToNat( *params )
        stats = cls.sufficientStats( x, D=D )
        part = cls.log_partition( x, natParams=natParams )
        return ExponentialFam.log_pdf( natParams, stats, part )

    @classmethod
    def log_jointExpFam( cls, x, D=None, params=None, natParams=None, priorParams=None, priorNatParams=None ):
        assert ( params is None ) ^ ( natParams is None ) and ( priorParams is None ) ^ ( priorNatParams is None )
        assert D is not None

        postNatParams = cls.posteriorPriorNatParams( x, D=D, priorParams=priorParams, priorNatParams=priorNatParams )

        params = params if params is not None else cls.natToStandard( *natParams )
        stat = cls.priorClass.sufficientStats( params )
        part = cls.priorClass.log_partition( params, params=priorParams, natParams=priorNatParams, split=True )

        return ExponentialFam.log_pdf( postNatParams, stat, part )

    @classmethod
    def log_posteriorExpFam( cls, x, D=None, params=None, natParams=None, priorParams=None, priorNatParams=None ):
        assert ( params is None ) ^ ( natParams is None ) and ( priorParams is None ) ^ ( priorNatParams is None )
        assert D is not None

        postNatParams = cls.posteriorPriorNatParams( x, D=D, priorParams=priorParams, priorNatParams=priorNatParams )

        params = params if params is not None else cls.natToStandard( *natParams )
        stat = cls.priorClass.sufficientStats( params )
        part = cls.priorClass.log_partition( params, natParams=postNatParams, split=True )

        return ExponentialFam.log_pdf( postNatParams, stat, part )

    @classmethod
    def log_posterior( cls, x, D=None, params=None, natParams=None, priorParams=None, priorNatParams=None ):
        # Compute P( Ѳ | x; α )
        assert ( params is None ) ^ ( natParams is None ) and ( priorParams is None ) ^ ( priorNatParams is None )
        assert D is not None

        params = params if params is not None else cls.natToStandard( *natParams )
        postNatParams = cls.posteriorPriorNatParams( x, D=D, priorParams=priorParams, priorNatParams=priorNatParams )
        return cls.priorClass.log_likelihood( params, natParams=postNatParams )

    ##########################################################################

    def ilog_likelihood( self, x, D=None, expFam=False ):
        if( expFam ):
            return self.log_likelihoodExpFam( x, D=D, natParams=self.natParams )
        return super( Categorical, self ).ilog_likelihood( x )

    def ilog_joint( self, x, D=None, expFam=False ):
        if( expFam ):
            return self.log_jointExpFam( x, D=D, params=self.params, priorNatParams=self.prior.natParams )
        return super( Categorical, self ).ilog_joint( x )

    def ilog_posterior( self, x, D=None, expFam=False ):
        if( expFam ):
            return self.log_posteriorExpFam( x, D=D, params=self.params, priorNatParams=self.prior.natParams )
        if( self.standardChanged ):
            if( self.prior.standardChanged ):
                return self.log_posterior( x, D=D, params=self.params, priorParams=self.prior.params )
            return self.log_posterior( x, D=D, params=self.params, priorNatParams=self.prior.natParams )
        if( self.prior.standardChanged ):
            return self.log_posterior( x, D=D, natParams=self.natParams, priorParams=self.prior.params )
        return self.log_posterior( x, D=D, natParams=self.natParams, priorNatParams=self.prior.natParams )

    ##########################################################################

    def likelihoodNoPartitionTest( self ):
        x = self.isample()
        D = self.p.shape[ 0 ]
        ans1 = self.ilog_likelihood( x, D=D, expFam=True )
        trueAns1 = self.ilog_likelihood( x )

        x = self.isample()
        D = self.p.shape[ 0 ]
        ans2 = self.ilog_likelihood( x, D=D, expFam=True )
        trueAns2 = self.ilog_likelihood( x )
        assert np.isclose( ans1 - ans2, trueAns1 - trueAns2 ), ( ans1 - ans2 ) - ( trueAns1 - trueAns2 )

    def likelihoodTest( self ):
        x = self.isample()
        D = self.p.shape[ 0 ]
        ans1 = self.ilog_likelihood( x, D=D, expFam=True )
        ans2 = self.ilog_likelihood( x )
        assert np.isclose( ans1, ans2 ), ans1 - ans2

    def jointTest( self ):
        x = self.isample()
        D = self.p.shape[ 0 ]
        ans1 = self.ilog_joint( x, D=D, expFam=True )
        ans2 = self.ilog_joint( x )
        assert np.isclose( ans1, ans2 ), ans1 - ans2

    def posteriorTest( self ):
        x = self.isample()
        D = self.p.shape[ 0 ]
        ans1 = self.ilog_posterior( x, D=D, expFam=True )
        ans2 = self.ilog_posterior( x, D=D )
        assert np.isclose( ans1, ans2 ), ans1 - ans2
