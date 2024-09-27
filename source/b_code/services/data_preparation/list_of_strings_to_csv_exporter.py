import csv


def export_list_of_strings_to_csv(
        output_file_path: str,
        list_of_strings: list,
        encoding: str = 'utf-8') \
        -> None:
    with open(
            output_file_path,
            mode='w',
            newline='',
            encoding=encoding) as file:
        writer = \
            csv.writer(
                    file)
        
        for text \
                in list_of_strings:
            writer.writerow(
                    [text])
