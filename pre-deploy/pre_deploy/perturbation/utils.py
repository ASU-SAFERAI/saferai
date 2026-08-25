# Reference:
# https://github.com/stanford-crfm/helm/blob/main/src/helm/common/general.py

import json
from typing import List


def format_tags(tags: List[str]) -> str:
    """Takes a list of tags and outputs a string: tag_1,tag_2,...,tag_n"""
    return f"[{','.join(tags)}]"


def format_text(text: str) -> str:
    return json.dumps(text)


def match_case(source_word: str, target_word: str) -> str:
    """Return a version of the target_word where the case matches the source_word."""
    # Check for all lower case source_word
    if all(letter.islower() for letter in source_word):
        return target_word.lower()
    # Check for all caps source_word
    if all(letter.isupper() for letter in source_word):
        return target_word.upper()
    # Check for capital source_word
    if source_word and source_word[0].isupper():
        return target_word.capitalize()
    return target_word
