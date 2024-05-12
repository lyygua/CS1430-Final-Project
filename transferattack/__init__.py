from .gradient.fgsm import FGSM
from .gradient.mifgsm import MIFGSM

from .input_transformation.ssm import SSM
from .advanced_objective.fia import FIA

attack_zoo = {
    'fgsm': FGSM,
    'mifgsm': MIFGSM,

    'ssm': SSM,

    'fia': FIA,
}

__version__ = '1.0.0'