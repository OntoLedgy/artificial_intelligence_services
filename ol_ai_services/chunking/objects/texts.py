import os
from data_export.list_of_strings_to_csv_exporter import export_list_of_strings_to_csv


# TODO: All that the class does is getting the texts and add it to a list, and output to disk
class Texts:
    def __init__(
            self,
            source_texts: list,
            output_folder_path: str = None):
        
        self.source_texts = \
            source_texts
        
        self.output_folder_path = \
            output_folder_path
        
    # TODO: Create an export to csv method for inspection?? - DONE
    def export_to_csv(
            self) \
            -> None:
        if not os.path.exists(self.output_folder_path):
            os.makedirs(
                    self.output_folder_path)
            
        output_file_path = \
            os.path.join(
                self.output_folder_path,
                'texts.csv')
        
        export_list_of_strings_to_csv(
                output_file_path=output_file_path,
                list_of_strings=self.source_texts)

    #TODO: remove this code, after testing
    # def load_pdf_source_texts(
    #         self) \
    #         -> list:
    #     source_texts = \
    #         list()
    #
    #     for filename \
    #             in os.listdir(
    #                 self.source_texts_folder_path):
    #         if filename.endswith(PDF_FILE_EXTENSION):
    #             self.__extract_texts_from_pdf(
    #                 filename=filename,
    #                 source_texts=source_texts)
    #
    #     return \
    #         source_texts
    #
    #
    # def __extract_texts_from_pdf(
    #         self,
    #         filename: str,
    #         source_texts: list):
    #     file_path = \
    #         os.path.join(
    #                 self.source_texts_folder_path,
    #                 filename)
    #     text = \
    #         self.extract_text_from_pdf(
    #                 pdf_path=file_path)
    #
    #     source_texts.append(
    #             text)
    #
    # @staticmethod
    # def extract_text_from_pdf(
    #         pdf_path: str):
    #     text = \
    #         str()
    #
    #     with open(
    #             pdf_path,
    #             READ_BYTES_ACRONYM) as file:
    #         pdf_reader = \
    #             PyPDF2.PdfReader(
    #                 file)
    #
    #         for page_number \
    #                 in range(len(pdf_reader.pages)):
    #             pdf_page = \
    #                 pdf_reader.pages[page_number]
    #
    #             text += \
    #                 pdf_page.extract_text()
    #
    #     return \
    #         text
    

    