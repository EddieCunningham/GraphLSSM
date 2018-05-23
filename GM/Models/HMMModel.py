from GenModels.GM.Models import ModelBase
from GenModels.GM.States.StandardStates.HMMState import HMMState
import numpy as np

__all__ = [ 'HMMModel' ]

class HMMModel( ModelBase ):

    priorClass = None

    def __init__( self, )