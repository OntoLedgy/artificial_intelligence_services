from source.b_code.configurations.boro_configurations.nf_open_ai_configurations import NfOpenAiConfigurations


def override_nf_open_ai_configurations_oxi() \
    -> None:
    NfOpenAiConfigurations.OPEN_AI_TEMPERATURE = \
        0.7
    
    NfOpenAiConfigurations.DEFAULT_GRAPH_RAG_ORCHESTRATOR_OPEN_AI_TEMPERATURE = \
        0.5
    
    