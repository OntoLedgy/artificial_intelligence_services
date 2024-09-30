import json


#TODO: wrap the format of texts (currently a list of strings)
def save_texts_to_jsonl(
        texts,
        output_jsonl_file="pdf_texts.jsonl"):
    """
    Saves a list of texts to a JSONL file.

    Args:
        texts (List[str]): A list of text strings to be saved.
        output_jsonl_file (str): Path to the output JSONL file.
    """
    with open(output_jsonl_file, 'w', encoding='utf-8') as f:
        for text in texts:
            json_line = json.dumps(text)
            f.write(json_line + '\n')
