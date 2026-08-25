import logging
from typing import List

from .data_augmenter import DataAugmenter
from .perturbation_base import Perturbation
from .scenario import Instance, Input, Reference, Output

from ..input import GoldenTestSet

logger = logging.getLogger(__name__)


def gender_perturbation(golden_test_set: GoldenTestSet) -> GoldenTestSet:
    from .gender_perturbation import GenderPerturbation
    perturbations = [GenderPerturbation(prob=1.0, mode="pronouns", source_class="male", target_class="female"),
                     GenderPerturbation(prob=1.0, mode="terms", source_class="male", target_class="female")]
    perturbed_data = generate_perturbed_data(golden_test_set=golden_test_set, perturbations=perturbations)
    return perturbed_data


def ave_perturbation(golden_test_set: GoldenTestSet, prob: float=1.0) -> GoldenTestSet:
    from .dialect_perturbation import DialectPerturbation
    ave_perturbations = [DialectPerturbation(prob=prob, source_class="SAE", target_class="AAVE")]
    perturbed_data = generate_perturbed_data(golden_test_set=golden_test_set,
                                             perturbations=ave_perturbations)
    return perturbed_data


def typos_perturbation(golden_test_set: GoldenTestSet, prob: float=0.1) -> GoldenTestSet:
    from .typos_perturbation import TyposPerturbation
    perturbations = [TyposPerturbation(prob=prob)]
    perturbed_data = generate_perturbed_data(golden_test_set=golden_test_set,
                                             perturbations=perturbations)
    return perturbed_data


def synonym_perturbation(golden_test_set: GoldenTestSet, prob: float=1.0) -> GoldenTestSet:
    from .synonym_perturbation import SynonymPerturbation
    perturbations = [SynonymPerturbation(prob=prob)]
    perturbed_data = generate_perturbed_data(golden_test_set=golden_test_set,
                                             perturbations=perturbations)
    return perturbed_data


def spelling_perturbation(golden_test_set: GoldenTestSet) -> GoldenTestSet:
    from perturbation.contraction_expansion_perturbation import ContractionPerturbation
    from perturbation.lowercase_perturbation import LowerCasePerturbation
    from perturbation.misspelling_perturbation import MisspellingPerturbation
    from perturbation.space_perturbation import SpacePerturbation
    spelling_perturbations = [
        ContractionPerturbation(),
        LowerCasePerturbation(),
        MisspellingPerturbation(prob=0.1),
        SpacePerturbation(max_spaces=3),
    ]
    perturbed_data = generate_perturbed_data(golden_test_set=golden_test_set,
                                             perturbations=spelling_perturbations)
    return perturbed_data


def generate_perturbed_data(golden_test_set: GoldenTestSet,
                            perturbations: List[Perturbation]) -> GoldenTestSet:
    perturbed_instances = _compute_perturbed_instances(golden_test_set, perturbations)

    perturbed_data = GoldenTestSet.from_dict({
        "id": golden_test_set.id + "_perturbed",
        "data": [
            {
                "id": instance.id,
                "input": instance.input.text,
                "expected_output": instance.references[0].output.text,
                "metadata": golden_test_set[instance.id].metadata
            } for instance in perturbed_instances
        ]
    })

    return perturbed_data


def _compute_perturbed_instances(golden_test_set: GoldenTestSet,
                                 perturbations: List[Perturbation]) -> List[Instance]:
    instances = [
        Instance(id=pair.id,
                 input=Input(text=str(pair.input)),
                 references=[Reference(Output(text=str(pair.expected_output)), tags=[])], )
        for pair in golden_test_set.golden_pairs
    ]
    data_augmenter = DataAugmenter(perturbations=perturbations)
    perturbed_instances = data_augmenter.generate(instances, include_original=False, skip_unchanged=True)
    return perturbed_instances
