ACCURACY_WITH_REFERENCE_EVALUATION_STEPS = [
    "Identify the key facts or details in the 'expected output'.",
    "Compare the 'actual output' to see if it includes those key facts.",
    "If the 'actual output' omits any key fact, subtract a significant portion of the score.",
    "If the 'actual output' contradicts 'expected output' facts, reduce score drastically (or fail).",
    "If the 'actual output' contains more details but does not contradict the 'expected output', award a perfect score.",
    "DO NOT PENALIZE DIFFERENCES IN WORDING, STRUCTURE, AND FORMATTING AS LONG AS THE KEY FACTS ARE PRESENT AND ACCURATE.",
    "Vague language is acceptable as long as the key facts are present.",
    "Assign a final score based on completeness and correctness."
]


ACCURACY_WITH_REFERENCE_RUBRICS = [
    {
        "score_range": (0, 2),
        "expected_outcome": "Response is incorrect or directly contradicts the expected output; key facts are missing or wrong and the overall meaning is not preserved."
    },
    {
        "score_range": (3, 5),
        "expected_outcome": "Partial match: some relevant facts are present but multiple important details are omitted or inaccurate; response shows misunderstandings that reduce usefulness."
    },
    {
        "score_range": (6, 8),
        "expected_outcome": "Mostly correct: most key facts from the expected output are present and accurate, with only minor omissions or slight inaccuracies that do not change the main conclusions."
    },
    {
        "score_range": (9, 10),
        "expected_outcome": "Complete and accurate: all key facts and reasoning from the expected output are preserved, no contradictions, clear structure, and any extra details are correct and relevant."
    }
]
