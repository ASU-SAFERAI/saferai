# Reference:
# https://github.com/stanford-crfm/helm/blob/fc2baa6b3be42f5e46e9ef70b2cc7f2e25f0a977/src/helm/benchmark/augmentations/

from dataclasses import dataclass
from random import Random

from .perturbation_base import TextPerturbation
from .perturbation_description import PerturbationDescription


class ExtraSpacePerturbation(TextPerturbation):
    """
    A toy perturbation that replaces existing spaces in the text with
    `num_spaces` number of spaces.
    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        num_spaces: int = 0

    name: str = "extra_space"

    def __init__(self, num_spaces: int):
        self.num_spaces = num_spaces

    @property
    def description(self) -> PerturbationDescription:
        return ExtraSpacePerturbation.Description(name=self.name, robustness=True, num_spaces=self.num_spaces)

    def perturb(self, text: str, rng: Random) -> str:
        return text.replace(" ", " " * self.num_spaces)
