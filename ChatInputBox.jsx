BLIMP_TASKS = {
    "distractor_agreement_relative_clause": {
        "hf_name": "distractor_agreement_relative_clause",
        "description": "Subject-verb agreement across relative clause",
        "relevance": "Core phenomenon — tests relative clause syntax directly",
        "priority": 1,
    },
    "anaphor_number_agreement": {
        "hf_name": "anaphor_number_agreement",
        "description": "Reflexive pronoun number agreement",
        "relevance": "Pure syntactic agreement — zero semantic content",
        "priority": 2,
    },
    "anaphor_gender_agreement": {
        "hf_name": "anaphor_gender_agreement",
        "description": "Reflexive pronoun gender agreement",
        "relevance": "Pure syntactic agreement — complements number agreement",
        "priority": 3,
    },
    "passive_1": {
        "hf_name": "passive_1",
        "description": "Passive voice constructions",
        "relevance": "Active/passive alternation — directly relevant to our training pairs",
        "priority": 4,
    },
    "principle_A_c_command": {
        "hf_name": "principle_A_c_command",
        "description": "Binding Principle A — c-command domain",
        "relevance": "Long-distance syntactic dependency, structural constraint",
        "priority": 5,
    },
}
