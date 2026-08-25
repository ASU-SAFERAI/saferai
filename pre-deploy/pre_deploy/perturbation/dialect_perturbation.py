# Reference:
# https://github.com/stanford-crfm/helm/blob/fc2baa6b3be42f5e46e9ef70b2cc7f2e25f0a977/src/helm/benchmark/augmentations/dialect_perturbation.py

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Dict, Optional, List
from random import Random
import re

from .utils import match_case
from .perturbation_description import PerturbationDescription
from .perturbation_base import TextPerturbation

_THIS_DIR = Path(__file__).parent
SAE_TO_AAVE_MAPPING_FILE_PATH = f"{_THIS_DIR}/data/SAE_to_AAVE_mapping.json"


class DialectPerturbation(TextPerturbation):
    """Individual fairness perturbation for dialect."""

    """ Short unique identifier of the perturbation (e.g., extra_space) """
    name: str = "dialect"

    should_perturb_references: bool = True

    """ Output path to store external files and folders """
    OUTPUT_PATH = os.path.join("/tmp/perturbation", "data", name)

    """ Dictionary mapping dialects to one another """
    SAE = "SAE"
    AAVE = "AAVE"

    """ Dictionary containing the URIs for the dialect mapping dictionaries

    Keys are tuples of the form (source_class, target_class), such as
    ("SAE", "AAVE"). Mapping dictionaries are from the sources listed below,
    converted to JSON and stored in Google Cloud Storage.

        (1) SAE to AAVE dictionary is from Ziems et al. (2022)

                Paper: https://arxiv.org/abs/2204.03031
                GitHub: https://github.com/GT-SALT/value/

    """
    MAPPING_DICT_URIS = {
        (SAE, AAVE): (
            "https://storage.googleapis.com/crfm-helm-public/source_datasets/"
            "augmentations/dialect_perturbation/SAE_to_AAVE_mapping.json"
        )
    }

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        """Description for the DialectPerturbation class."""

        prob: float = 0.0
        source_class: str = ""
        target_class: str = ""
        mapping_file_path: Optional[str] = None

    def __init__(self, prob: float, source_class: str, target_class: str,
                 mapping_file_path: str = SAE_TO_AAVE_MAPPING_FILE_PATH):
        """Initialize the dialect perturbation.

        If mapping_file_path is not provided, (source_class, target_class)
        should be ("SAE", "AAVE").

        Args:
            prob: Probability of substituting a word in the original class with
                a word in the target class given that a substitution is
                available.
            source_class: The source dialect that will be substituted with
                the target dialect. Case-insensitive.
            target_class: The target dialect.
            mapping_file_path: The absolute path to a file containing the
                word mappings from the source dialect to the target dialect in
                a json format. The json dictionary must be of type
                Dict[str, List[str]]. Otherwise, the default dictionary in
                self.MAPPING_DICTS for the provided source and target classes
                will be used, if available.
        """
        # TODO: Update path so it is not hard-coded to benchmark_output
        # https://github.com/stanford-crfm/benchmarking/issues/493
        self.output_path: str = self.OUTPUT_PATH
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        # Assign parameters to instance variables
        assert 0 <= prob <= 1
        self.prob = prob
        self.source_class: str = source_class.upper()
        self.target_class: str = target_class.upper()

        self.mapping_file_path: str = mapping_file_path
        self.mapping_dict: Dict[str, List[str]] = self.load_mapping_dict()

        # Pattern capturing any occurence of the given words in the text, surrounded by characters other than
        # alphanumeric characters and '-'. We use re.escape since the words in our dictionary may
        # contain special RegEx characters.
        words = [re.escape(w) for w in self.mapping_dict.keys()]
        words_string = "|".join(words)
        self.pattern = f"[^\\w-]({words_string})[^\\w-]"

    @property
    def description(self) -> PerturbationDescription:
        """Return a perturbation description for this class."""
        return DialectPerturbation.Description(
            name=self.name,
            fairness=True,
            prob=self.prob,
            source_class=self.source_class,
            target_class=self.target_class,
            mapping_file_path=self.mapping_file_path,
        )

    def load_mapping_dict(self) -> Dict[str, List[str]]:
        """Load the mapping dict."""
        with open(self.mapping_file_path, 'r', encoding="utf-8") as openfile:
            mapping = json.load(openfile)
        return mapping

    def perturb(self, text: str, rng: Random) -> str:
        """Substitute the source dialect in text with the target dialect with probability self.prob."""

        # Substitution function
        def sub_func(m: re.Match):
            match_str = m.group(0)  # The full match (e.g. " With ", " With,", " With.")
            word = m.group(1)  # Captured group (e.g. "With")
            if rng.uniform(0, 1) < self.prob:
                synonyms = self.mapping_dict[word.lower()]
                synonym = rng.choice(synonyms)  # Synonym (e.g. "wit")
                synonym = match_case(word, synonym)  # Synoynm with matching case (e.g. "Wit")
                match_str = match_str.replace(
                    word, synonym
                )  # Synonym placed in the matching group (e.g. " Wit ", " Wit,", " Wit.")
            return match_str

        # Execute the RegEx
        return re.sub(self.pattern, sub_func, text, flags=re.IGNORECASE)
