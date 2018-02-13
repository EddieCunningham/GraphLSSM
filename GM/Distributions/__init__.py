import sys
sys.path.append( '/Users/Eddie/GenModels/GM/Distributions/' )

from .Base import Distribution, Conjugate, Exponential
from .MatrixNormalInverseWishart import MatrixNormalInverseWishart
from .Regression import Regression
from .InverseWishart import InverseWishart
from .NormalInverseWishart import NormalInverseWishart
from .Normal import Normal
from .Dirichlet import Dirichlet
from .Categorical import Categorical
