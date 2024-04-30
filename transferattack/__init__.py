from .gradient.fgsm import FGSM

attack_zoo = {
    'fgsm': FGSM,
}

__version__ = '1.0.0'