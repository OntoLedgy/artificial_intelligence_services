class NfGeneralConfigurations:
    default_string_empty = \
        str()
    
    #TODO: Review these to see if needed
    RECURSIVE_CHARACTER_TEXTSPLITTER_CHUNK_SIZE = \
        1000
    
    RECURSIVE_CHARACTER_TEXTSPLITTER_CHUNK_OVERLAP = \
        50
    
    NUMBER_OF_ROWS = \
        10
    
    MAXIMUM_WORKERS = \
        10
    
    # number_of_rows = 10,
    # maximum_workers = 10

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

