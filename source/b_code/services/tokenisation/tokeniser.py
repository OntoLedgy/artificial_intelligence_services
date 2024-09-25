import json

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
import tiktoken

from configurations.boro_configurations.nf_open_ai_configurations import NfOpenAiConfigurations


def num_tokens_from_string(
        string: str,
        model: str = "gpt-4o") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


class Tokeniser:
    def __init__(
            self,
            model_name=NfOpenAiConfigurations.OPEN_AI_MODEL_NAME_GPT2):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenized_dataset = {}

    def tokenize(self,
                 data_files):

        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Load the dataset from the JSONL file
        dataset = load_dataset(
            'json',
            data_files=data_files)

        # Tokenize the dataset
        def tokenize_function(examples):
            tokens = self.tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
            tokens['labels'] = tokens['input_ids'].copy()  # Use input_ids as labels
            return tokens

        self.tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True)

    def print_tokenized_data(
            self,
            split='train',
            num_samples=5):
        """
        Prints the tokenized data for analysis.

        Args:
            tokenized_dataset: The tokenized dataset (Hugging Face Dataset or DatasetDict object).
            tokenizer: The tokenizer used to tokenize the data.
            num_samples: Number of samples to print for analysis.
        """

        # Use the 'train' split of the tokenized data
        tokenized_data_train = self.tokenized_dataset[split]

        for i in range(min(num_samples, len(tokenized_data_train))):
            print(f"Sample {i + 1}:")
            # Get the token IDs for this sample
            token_ids = tokenized_data_train[i]['input_ids']

            # Convert token IDs back to text
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)

            # Print the token IDs, tokens, and the reconstructed text
            print(f"Token IDs: {token_ids}")
            print(f"Tokens: {tokens}")
            print(f"Reconstructed Text: {text}")
            print('-' * 50)

    def save_tokenized_data_to_file(
            self,
            split='train',
            output_file="tokenized_dataset.json"):
        """
        Saves the tokenized data to a JSONL file.

        Args:
            tokenized_dataset: The tokenized dataset (Hugging Face Dataset object).
            output_file: Path to the output file where tokenized data will be saved.
        """
        # Convert the entire dataset to a dictionary
        tokenized_data_train = self.tokenized_dataset[split]
        dataset_dict = tokenized_data_train.to_dict()

        with open(output_file, 'w') as f:
            # Iterate through the dataset using a for loop
            for i in range(len(dataset_dict)):
                tokenized_entry = {
                    "input_ids": dataset_dict['input_ids'][i],
                    "attention_mask": dataset_dict['attention_mask'][i]
                }
                # Write the dictionary as a JSON line
                f.write(json.dumps(tokenized_entry) + '\n')

    def read_tokenized_data_from_file(self, input_file):
        """
        Reads tokenized data from a JSONL file and returns it as a Hugging Face Dataset.

        Args:
            input_file: Path to the JSONL file containing tokenized data.

        Returns:
            A Hugging Face Dataset object containing the tokenized data.
        """
        data = []

        # Read the JSONL file
        with open(input_file, 'r') as f:
            for line in f:
                # Load each line as a dictionary
                tokenized_entry = json.loads(line.strip())
                # Add 'labels' to the tokenized_entry by copying 'input_ids'
                tokenized_entry['labels'] = tokenized_entry['input_ids'].copy()
                data.append(tokenized_entry)

        # Convert the list of dictionaries into a Hugging Face Dataset
        dataset = Dataset.from_list(data)

        return dataset
