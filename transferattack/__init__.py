from .gradient.fgsm import FGSM
from .gradient.mifgsm import MIFGSM

from .input_transformation.ssm import SSM

attack_zoo = {
    'fgsm': FGSM,
    'mifgsm': MIFGSM,

    'ssm': SSM,
}

__version__ = '1.0.0'