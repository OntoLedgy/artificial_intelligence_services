from bclearer_orchestration_services.reporting_service.wrappers.run_and_log_function_wrapper_latest import run_and_log_function
from chunking.objects.texts import Texts
from data_export.list_of_dictionaries_to_json_file_writer import write_list_of_dictionaries_to_json_file


class ChunkedTexts:
    def __init__(
            self,
            texts: Texts,
            chunk_size: int,
            output_file_path: str):  # TODO: this should be the output folder, not file path
        self.texts = \
            texts
        
        self.chunk_size = \
            chunk_size
        
        self.output_file_path = \
            output_file_path
        
        self.chunked_texts = \
            self.chunk_texts()
    
    
    @run_and_log_function()
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
    
    def export_to_jsonl(
            self) \
            -> None:
        write_list_of_dictionaries_to_json_file(
                output_file_path=self.output_file_path,
                list_of_dictionaries=self.chunked_texts)
