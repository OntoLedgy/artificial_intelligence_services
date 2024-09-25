class NfGeneralConfigurations:
    default_string_empty = \
        str()
    
    #TODO: Review these to see if needed
    RECURSIVE_CHARACTER_TEXTSPLITTER_CHUNK_SIZE = \
        1000
    
    RECURSIVE_CHARACTER_TEXTSPLITTER_CHUNK_OVERLAP = \
        50
    
    DEFAULT_DATA_CHUNK_SIZE_FOR_TRAINING = \
        512
    
    DEFAULT_TRUNCATE_CONTEXT_MAX_TOKENS = \
        4000
    
    HUGGING_FACE_MODEL_NAME = \
        'accounting_fine_tuned'
    
    # def test_text_generation(self):
    #
    #     model_path = r'data/outputs/models/'
    #     model_name = "accounting_fine_tuned"
    
    # def truncate_context(
    #         context,
    #         max_tokens = 4000)
    
    # def prepare_data_for_training(
    #         texts,
    #         chunk_size = 512

# model.generate(
#         input_ids,
#         max_length=200,
#         num_return_sequences=1,
#         no_repeat_ngram_size=2,
#         top_p=0.95,
#         temperature=0.7,
#         do_sample=True,

# def get_response(
#         query,
#         client,
#         model_name="gpt-3.5-turbo",
#         input_file='retrieved_articles.txt',
#         max_context_tokens=12000):

