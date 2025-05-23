import csv

from configurations.constants import UTF_8_ENCODING
from configurations.constants import WRITE_ACRONYM


def export_dictionary_of_strings_to_csv(
        output_file_path: str,
        dictionary_of_strings: dict,
        keys_column_name: str = 'Column A',
        values_column_name: str = 'Column B',
        encoding: str = UTF_8_ENCODING ) \
        -> None:
    with open(
            output_file_path,
            mode=WRITE_ACRONYM,
            newline='',
            encoding=encoding) as file:
        writer = csv.DictWriter(
                file,
                fieldnames=[keys_column_name, values_column_name])
        
        writer.writeheader()
        
        for key, value in dictionary_of_strings.items():
            writer.writerow(
                    {
                        keys_column_name  : key,
                        values_column_name: value
                        })
