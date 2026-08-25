# Reference:
# https://github.com/stanford-crfm/helm/blob/fc2baa6b3be42f5e46e9ef70b2cc7f2e25f0a977/src/helm/benchmark/augmentations/

from dataclasses import dataclass
import json
import os
from random import Random
from pathlib import Path
from typing import Dict, List

import nltk
from nltk.corpus import wordnet
import spacy

from .perturbation_description import PerturbationDescription
from .perturbation_base import TextPerturbation
from .utils import match_case


class SynonymPerturbation(TextPerturbation):
    """
    Synonyms. For implementation details, see
    https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/synonym_substitution/transformation.py

    This perturbation adds noise to a text source by randomly inserting synonyms of randomly selected
    words excluding punctuations and stopwords.
    The space of synonyms depends on WordNet and could be limited. The transformation might introduce
    non-grammatical segments.

    Perturbation example:

    **Input:**
        This was a good movie, would watch again.

    **Output:**
        This was a dependable movie, would determine again.
    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        prob: float = 0.0

    name: str = "synonym"

    # For downloading wordnet_synonyms.json
    FILE_NAME: str = "wordnet_synonyms.json"
    SOURCE_URI: str = (
        "https://storage.googleapis.com/crfm-helm-public/source_datasets/"
        "augmentations/synonym_perturbation/wordnet_synonyms.json"
    )

    def __init__(self, prob: float):
        # Assign parameters to instance variables
        self.prob: float = prob

        # Initialize the model with spaCy: https://spacy.io/models/en
        try:
            self.spacy_model = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")  # type: ignore
            self.spacy_model = spacy.load("en_core_web_sm")

        _THIS_DIR = Path(__file__).parent
        SAE_TO_AAVE_MAPPING_FILE_PATH = f"{_THIS_DIR}/data/SAE_to_AAVE_mapping.json"

        output_dir = os.path.join(str(_THIS_DIR), "data", self.name)
        output_dir_for_wordnet = os.path.join(str(_THIS_DIR), "data", self.name)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        nltk.data.path.append(output_dir)
        try:
            # We cannot use wordnet.synsets directly since it's not thread-safe. So we copy the synsets to
            # wordnet_synonyms.json and use that in combination with _morphy (as done in the original wordnet.synsets).
            wordnet.ensure_loaded()
        except LookupError:
            # Fix NLTK SSL error
            import ssl

            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context

            if not os.path.exists(os.path.join(output_dir, "corpora/wordnet")):
                nltk.download("wordnet", download_dir="/tmp/nltk_data")
            if not os.path.exists(os.path.join(output_dir, "corpora/omw-1.4")):
                nltk.download("omw-1.4", download_dir="/tmp/nltk_data")
        nltk.data.path.insert(0, "/tmp/nltk_data")
        wordnet.ensure_loaded()
        target_path = os.path.join(output_dir_for_wordnet, self.FILE_NAME)
        with open(target_path) as f:
            self.wordnet_synonyms: Dict[str, List[str]] = json.load(f)

    @property
    def description(self) -> PerturbationDescription:
        return SynonymPerturbation.Description(name=self.name, robustness=True, prob=self.prob)

    def perturb(self, text: str, rng: Random) -> str:
        spacy_to_wordnet_pos = {
            "VERB": "v",
            "NOUN": "n",
            "ADV": "r",
            "ADJ": "s",
        }

        doc = self.spacy_model(text)

        perturbed_text = ""

        for token in doc:
            word = token.text
            wordnet_pos = spacy_to_wordnet_pos.get(token.pos_)
            synonyms = []
            if wordnet_pos:
                for base in wordnet._morphy(word.lower(), wordnet_pos):  # _morphy returns the base form of a word
                    synonyms.extend(self.wordnet_synonyms.get(f"{base}:{wordnet_pos}", []))
            synonyms = [s for s in synonyms if s != word.lower()]
            synonyms = list(dict.fromkeys(synonyms))  # Make the list unique while preserving the order
            if synonyms and rng.uniform(0, 1) < self.prob:
                synonym = rng.choice(synonyms)
                word = match_case(word, synonym)
            perturbed_text += word + token.whitespace_

        return perturbed_text
