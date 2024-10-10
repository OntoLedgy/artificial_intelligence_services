from docx import Document


def extract_text_from_word_document_sections(document_file_path: str) -> list:
    document = Document(document_file_path)

    sections = list()

    current_section = {"heading": None, "content": str()}

    for paragraph in document.paragraphs:
        # Check if the paragraph is a heading (Heading 1, Heading 2, etc.)
        if paragraph.style.name.startswith("Heading"):
            # If we have a current section, save it
            if current_section["heading"]:
                sections.append(current_section)

            # Start a new section
            current_section = {"heading": paragraph.text.strip(), "content": ""}
        else:
            # Append the paragraph text to the current section's content
            current_section["content"] += paragraph.text.strip() + "\n"

    # Append the last section
    if current_section["heading"]:
        sections.append(current_section)

    return sections
