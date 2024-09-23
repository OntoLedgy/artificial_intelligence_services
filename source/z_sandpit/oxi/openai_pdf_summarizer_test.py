from source.code.common_utilities.boro_configurations.nf_open_ai_configurations import NfOpenAiConfigurations
from source.code.services.summarisation.pdf_summariser import PDFSummarizer
from source.z_sandpit.oxi.helpers.nf_open_ai_configurations_overrider_oxi import override_nf_open_ai_configurations_oxi

if __name__ == '__main__':
    override_nf_open_ai_configurations_oxi()

    pdf_path = r'C:\S\OXi\PythonDev\code\artificial_intelligence_services\source\z_sandpit\test_data\QMMQ2015 - Improving Model Quality through Foundational Ontologies (paper).pdf'

    summarizer = PDFSummarizer(
        pdf_path,
        NfOpenAiConfigurations.OPEN_AI_API_KEY)

    summarizer.load_and_split_pdf()
    summary = summarizer.summarize()

    print("Summary of the PDF:")
    print(summary)
