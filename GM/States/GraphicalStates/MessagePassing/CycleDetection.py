from GraphicalMessagePassingBase import Graph, GraphMessagePasser
import numpy as np
from scipy.sparse import coo_matrix


# Strategies:
# Transision matrix approach
# Ant colony approach

class CycleDetector( GraphMessagePasser ):

    def __init__( self ):
        super( CycleDetector, self ).__init__()

    def transitionMatrix( self ):
        pass