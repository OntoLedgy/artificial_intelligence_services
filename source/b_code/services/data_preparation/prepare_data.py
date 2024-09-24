import json


# TODO: MKh - should these be separate files? Helpers?
# TODO: MKh - should these be clean coded?
# --------Prep
def prepare_data_for_training(
        texts,
        chunk_size = 512):
    data = \
        []
    
    for text \
            in texts:
        # Split text into smaller chunks
        for i \
                in range(
                0,
                len(
                    text),
                chunk_size):
            chunk = text[i:i + chunk_size]
            
            data.append(
                {
                    "text": chunk
                    })
    
    return \
        data


def create_training_data(
        pdf_texts):
    # Create training data
    training_data = prepare_data_for_training(
        pdf_texts)
    
    # TODO: MKh - should 'training_data.jsonl' and 'w' and '\n' be a common literal?
    # Save the dataset in JSONL format
    with open(
            'training_data.jsonl',
            'w') as f:
        for entry \
                in training_data:
            json.dump(
                entry,
                f)
            f.write(
                '\n')
