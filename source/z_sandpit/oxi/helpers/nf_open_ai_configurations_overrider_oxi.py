from source.code.configurations.boro_configurations.nf_open_ai_configurations import NfOpenAiConfigurations


def override_nf_open_ai_configurations_oxi() \
    -> None:
    NfOpenAiConfigurations.OPEN_AI_API_KEY = \
        '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'
