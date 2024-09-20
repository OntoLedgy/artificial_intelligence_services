from source.code.services.summarisation.pdf_summariser import PDFSummarizer

if __name__ == '__main__':
    pdf_path = r'C:\S\OXi\PythonDev\code\artificial_intelligence_services\source\z_sandpit\test_data\QMMQ2015 - Improving Model Quality through Foundational Ontologies (paper).pdf'
    openai_api_key = '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'

    summarizer = PDFSummarizer(
        pdf_path,
        openai_api_key)

    summarizer.load_and_split_pdf()
    summary = summarizer.summarize()

    print("Summary of the PDF:")
    print(summary)
