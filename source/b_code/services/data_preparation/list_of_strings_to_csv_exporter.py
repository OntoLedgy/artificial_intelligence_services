import csv


def export_list_of_strings_to_csv(
        output_file_path: str,
        list_of_strings: list) \
        -> None:
    with open(
            output_file_path,
            mode='w',
            newline='') as file:
        writer = \
            csv.writer(
                    file)
        
        for text \
                in list_of_strings:
            writer.writerow(
                    [text])
