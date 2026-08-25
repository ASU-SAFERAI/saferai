from dataclasses import dataclass, field
from typing import List, Dict, Hashable
import random

from .validation import ValidationUtilities


@dataclass
class GoldenPair:
    """
    Represents a golden pair of input and expected output.
    """
    id: str
    input: str
    expected_output: str
    metadata: Dict[str, Hashable] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization validation for dataset attributes."""
        self.id = ValidationUtilities._validate_id(self.id, "id")
        self.input = ValidationUtilities._validate_type(self.input, str, "input")
        self.expected_output = ValidationUtilities._validate_type(self.expected_output, str, "expected_output")
        self.metadata = ValidationUtilities._validate_type(self.metadata, dict, "metadata")

    @classmethod
    def from_dict(cls, data: Dict) -> 'GoldenPair':
        """Create a GoldenPair instance from a dictionary."""
        ValidationUtilities._validate_dict(data)
        required_keys = ['id', 'input', 'expected_output']
        ValidationUtilities._validate_keys_exist(data, required_keys)

        return cls(
            id=data['id'],
            input=data['input'],
            expected_output=data['expected_output'],
            metadata=data.get('metadata', {})
        )

    def to_dict(self) -> Dict:
        """Convert the GoldenPair instance to a dictionary."""
        return {
            'id': self.id,
            'input': self.input,
            'expected_output': self.expected_output,
            'metadata': self.metadata
        }

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"GoldenPair(id='{self.id}', input='{self.input}', expected_output='{self.expected_output}')"

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging."""
        return (f"GoldenPair(id='{self.id}', "
                f"input='{self.input}', "
                f"expected_output='{self.expected_output}', "
                f"metadata_keys={list(self.metadata.keys())})")


@dataclass
class GoldenTestSet:
    """
    Represents a golden test set containing multiple golden pairs.
    Attributes:
        id: Unique identifier for the test set
        golden_pairs: List of golden pairs in the test set
        metadata: Additional metadata for the test set
    """
    id: str
    golden_pairs: List[GoldenPair]
    metadata: Dict[str, Hashable] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization validation for test set attributes."""
        self.id = ValidationUtilities._validate_id(self.id, "id")
        self.golden_pairs = ValidationUtilities._validate_type(self.golden_pairs, list, "golden_pairs")
        self.metadata = ValidationUtilities._validate_type(self.metadata, dict, "metadata")

        self._id_index = {pair.id: pair for pair in self.golden_pairs}
        if len(self._id_index) != len(self.golden_pairs):
            raise ValueError("Duplicate GoldenPair ids detected.")

    def __getitem__(self, key):
        return self._id_index[key]

    def __contains__(self, key):
        return key in self._id_index

    def get(self, key, default=None):
        return self._id_index.get(key, default)

    def __iter__(self):
        return iter(self.golden_pairs)

    def __hash__(self):
        # Hashable by stable id
        return hash(self.id)

    def __eq__(self, other):
        raise NotImplementedError("Equality comparison is not implemented for GoldenPair.")

    @classmethod
    def from_dict(cls, data: Dict) -> 'GoldenTestSet':
        """
        Create a GoldenTestSet instance from a dictionary.
        Expected dictionary format:
        {
            'id': str,
            'data': List[Dict],  # Each dict corresponds to a GoldenPair
            'metadata': Dict (optional)
        }
        """
        ValidationUtilities._validate_dict(data)
        required_keys = ['id', 'data']
        ValidationUtilities._validate_keys_exist(data, required_keys)

        golden_pairs = [GoldenPair.from_dict(pair) for pair in data['data']]
        return cls(
            id=data['id'],
            golden_pairs=golden_pairs,
            metadata=data.get('metadata', {})
        )

    def to_dict(self) -> Dict:
        """Convert the Conversation instance to a dictionary."""
        return {
            'id': self.id,
            'data': {pair.id: pair.to_dict() for pair in self.golden_pairs},
            'metadata': self.metadata
        }

    def sample(self, n: int) -> "GoldenTestSet":
        sampled_pairs = random.sample(self.golden_pairs, min(n, len(self.golden_pairs)))
        return GoldenTestSet(
            id=self.id,
            golden_pairs=sampled_pairs,
            metadata=self.metadata
        )

    def __len__(self) -> int:
        """Return the number of golden pairs in the test set."""
        return len(self.golden_pairs)

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"GoldenTestSet(id='{self.id}', golden_pairs={len(self.golden_pairs)})"

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging."""
        return (f"GoldenTestSet(id='{self.id}', "
                f"golden_pairs_count={len(self.golden_pairs)}, "
                f"metadata_keys={list(self.metadata.keys())})")
