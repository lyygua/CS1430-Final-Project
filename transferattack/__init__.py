from .gradient.fgsm import FGSM
from .gradient.mifgsm import MIFGSM
from .gradient.vmifgsm import VMIFGSM
from .gradient.vnifgsm import VNIFGSM

from .input_transformation.ssm import SSM
from .advanced_objective.fia import FIA
from .input_transformation.test import TEST

attack_zoo = {
    'fgsm': FGSM,
    'mifgsm': MIFGSM,
    'vmifgsm': VMIFGSM,
    'vnifgsm': VNIFGSM,

    'ssm': SSM,
    'test': TEST,

    'fia': FIA,
}

__version__ = '1.0.0'