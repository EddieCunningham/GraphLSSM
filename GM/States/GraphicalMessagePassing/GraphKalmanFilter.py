from GenModels.GM.States.GraphicalMessagePassing.GraphicalMessagePassingBase import *
from GenModels.GM.States.GraphicalMessagePassing.GraphFilterBase import *
import numpy as np
from functools import reduce
from scipy.sparse import coo_matrix
from collections import Iterable
from GenModels.GM.Utility import fbsData

from GenModels.GM.Distributions.Regression import TensorRegression, TensorNormal


class _kalmanFilterMixin():

    def genFilterProbs( self ):
        pass

    def updateParamsFromGraphs( self, ys, initial_dist, transition_dist, emission_dist, graphs ):
        pass

    def transitionProb( self, child ):
        pass

    def emissionProb( self, node, forward=False ):
        pass

    @classmethod
    def multiplyTerms( cls, terms ):
        pass

    @classmethod
    def integrate( cls, integrand, axes ):
        pass

    def uBaseCase( self, node, debug=True ):
        pass

    def vBaseCase( self, node, debug=True ):
        pass

    def filter( self ):
        pass

    def updateU( self, nodes, newU, U ):
        pass

    def updateV( self, nodes, edges, newV, V ):
        pass
