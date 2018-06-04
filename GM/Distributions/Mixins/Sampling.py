import numpy as np

__all__ = [ '_GibbsMixin', '_MetropolisHastingMixin' ]

class _GibbsMixin():

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

####################################################################################################################################

class _MetropolisHastingMixin():

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
