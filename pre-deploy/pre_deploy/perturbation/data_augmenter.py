# Reference:
# https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/augmentations/data_augmenter.py

from dataclasses import dataclass
from typing import List

from .perturbation_base import Perturbation
from .scenario import Instance


@dataclass(frozen=True)
class Processor:
    include_original: bool
    skip_unchanged: bool
    perturbations: List[Perturbation]
    seeds_per_instance: int

    def process(self, instance: Instance) -> List[Instance]:
        result: List[Instance] = []
        if self.include_original:
            #  we want to include the original even when the perturbation does not change the input
            result.append(instance)

        for perturbation in self.perturbations:
            for i in range(self.seeds_per_instance):
                perturbed_instance: Instance = perturbation.apply(instance, seed=None if i == 0 else i)
                if self.skip_unchanged and perturbed_instance.input == instance.input:
                    continue
                result.append(perturbed_instance)
        return result


@dataclass(frozen=True)
class DataAugmenter:

    # Perturbations to apply to generate new instances
    perturbations: List[Perturbation]

    def generate(
        self,
        instances: List[Instance],
        include_original: bool = True,
        skip_unchanged: bool = False,
        seeds_per_instance: int = 1,
    ) -> List[Instance]:
        """
        Given a list of Instances, generate a new list of perturbed Instances.
        include_original controls whether to include the original Instance in the new list of Instances.
        skip_unchanged controls whether we include instances for which the perturbation did not change the input.
        """
        processor = Processor(
            include_original=include_original,
            skip_unchanged=skip_unchanged,
            perturbations=self.perturbations,
            seeds_per_instance=seeds_per_instance,
        )
        results: List[List[Instance]] = [processor.process(instance) for instance in instances]
        output_instances = [instance for result in results for instance in result]

        return output_instances
