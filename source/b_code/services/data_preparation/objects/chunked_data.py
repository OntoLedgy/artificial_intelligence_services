from services.orchestrators.list_of_dictionaries_to_json_file_writer import write_list_of_dictionaries_to_json_file


class ChunkedData:
    def __init__(
            self,
            chunked_data: list,
            chunk_size: int):
        self.chunked_data = \
            chunked_data
        
        self.chunk_size = \
            chunk_size
    
    # TODO: would it make sense to move the method for chunking data to here??
    
    def export_to_jsonl(
            self,
            output_folder_path: str) \
            -> None:
        write_list_of_dictionaries_to_json_file(
                output_file_path=output_folder_path,
                list_of_dictionaries=self.chunked_data)
