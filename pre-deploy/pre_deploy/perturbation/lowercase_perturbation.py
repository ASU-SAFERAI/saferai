# Reference:
# https://github.com/stanford-crfm/helm/blob/fc2baa6b3be42f5e46e9ef70b2cc7f2e25f0a977/src/helm/benchmark/augmentations/

from random import Random

from .perturbation_base import TextPerturbation
from .perturbation_description import PerturbationDescription


class LowerCasePerturbation(TextPerturbation):
    """
    Simple perturbation turning input and references into lowercase.
    """

    name: str = "lowercase"

    @property
    def description(self) -> PerturbationDescription:
        return PerturbationDescription(name=self.name, robustness=True)

    def perturb(self, text: str, rng: Random) -> str:
        return text.lower()
