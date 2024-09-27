from nf_common_source.code.services.reporting_service.wrappers.run_and_log_function_wrapper import run_and_log_function
from services.data_preparation.objects.texts import Texts
from services.orchestrators.list_of_dictionaries_to_json_file_writer import write_list_of_dictionaries_to_json_file


class ChunkedTexts:
    def __init__(
            self,
            texts: Texts,
            chunk_size: int):
        self.texts = \
            texts
        
        self.chunk_size = \
            chunk_size
        
        self.chunked_texts = \
            self.chunk_texts()
    
    
    @run_and_log_function
    def chunk_texts(
            self):
        chunked_texts = \
            list()
        
        for text in \
                self.texts.source_texts:
            for i \
                    in range(0,len(text),self.chunk_size):
                chunk = \
                    text[i: i + self.chunk_size]
                
                chunked_texts.append(
                        {'text': chunk})
        
        return \
            chunked_texts
    
    # def create_training_data(
    #         pdf_texts):
    #     # Create training data
    #     training_data = prepare_data_for_training(
    #         pdf_texts)
    #
    #     # TODO: MKh - should 'training_data.jsonl' and 'w' and '\n' be a common literal?
    #     # Save the dataset in JSONL format
    #     with open(
    #             "training_data.jsonl",
    #             "w") as f:
    #         for entry in training_data:
    #             json.dump(
    #                 entry,
    #                 f)
    #             f.write(
    #                 "\n")
    
    
    def export_to_jsonl(
            self,
            output_folder_path: str) \
            -> None:
        write_list_of_dictionaries_to_json_file(
                output_file_path=output_folder_path,
                list_of_dictionaries=self.chunked_texts)
