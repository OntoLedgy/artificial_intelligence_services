import json


def prepare_data_for_training(
        texts,
        chunk_size=512):
    data = []
    for text in texts:
        add_chunks(chunk_size, data, text)

    return data


def add_chunks(chunk_size, data, text):
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        data.append({"text": chunk})


def create_training_data(
        text):
    training_data = prepare_data_for_training(text)

    with open('training_data.jsonl', 'w') as f:
        #use interop services
        for entry in training_data:
            json.dump(entry, f)
            f.write('\n')
