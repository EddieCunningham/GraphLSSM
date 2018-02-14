import sys
sys.path.append( '/Users/Eddie/GenModels/GM/States/StandardStates/MessagePassing/' )

from .MessagePassingBase import MessagePasser
from .ForwardBackward import CategoricalForwardBackward, GaussianForwardBackward, SLDSForwardBackward
from .KalmanFilter import KalmanFilter, SwitchingKalmanFilter