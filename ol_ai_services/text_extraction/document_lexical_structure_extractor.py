from pathlib import Path
from unstructured.partition.auto import partition
import spacy

try:
    from typing import List
except ImportError:
    pass

class Word:
    def __init__(self, text: str):
        self.text = text

class Sentence:
    def __init__(self, text: str, words: List[Word]):
        self.text = text
        self.words = words

class Paragraph:
    def __init__(self, text: str, sentences: List[Sentence]):
        self.text = text
        self.sentences = sentences

class Section:
    def __init__(self, title: str, level: int):
        self.title = title
        self.level = level
        self.paragraphs: List[Paragraph] = []
        self.subsections: List[Section] = []

class Document:
    def __init__(self):
        self.sections: List[Section] = []
    
    
    def print_document_structure(
            self) -> None:
        """
        Recursively print the hierarchy of the Document:
        Sections (with levels) -> Paragraphs -> Sentences -> Words.
        """
        
        
        def _print_section(
                section: Section,
                indent: int):
            indent_str = "    " * indent
            print(
                f"{indent_str}- Section (Level {section.level}): {section.title}")
            # Print paragraphs
            for idx_p, para in enumerate(
                    section.paragraphs,
                    1):
                print(
                    f"{indent_str}    - Paragraph {idx_p}: {para.text}")
                # Print sentences
                for idx_s, sent in enumerate(
                        para.sentences,
                        1):
                    print(
                        f"{indent_str}        - Sentence {idx_s}: {sent.text}")
                    # Print words
                    words_str = ", ".join(
                            word.text for word in sent.words)
                    print(
                        f"{indent_str}            - Words: [{words_str}]")
            # Recurse into subsections
            for subsection in section.subsections:
                _print_section(
                    subsection,
                    indent + 1)
        
        
        print(
            "Document Structure:")
        for sec in self.sections:
            _print_section(
                sec,
                0)

class DocumentParser:
    def __init__(self, nlp_model: str = "en_core_web_sm"):
        """
        Initialize parser with spaCy model for tokenization.
        Falls back to a blank English model with sentencizer.
        """
        try:
            self.nlp = spacy.load(nlp_model)
        except OSError:
            self.nlp = spacy.blank("en")
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")

    def parse(self, file_path: str) -> Document:
        """
        Parse document into hierarchical Document -> Sections -> Paragraphs -> Sentences -> Words.

        :param file_path: Path to supported document (PDF, DOCX, HTML, etc.)
        :return: Document object.
        """
        file_path = Path(file_path)
        elements = partition(file_path)

        def get_type(el):
            if hasattr(el, "element_type"):
                return el.element_type
            if hasattr(el, "type") and not isinstance(el, dict):
                return el.type
            if hasattr(el, "category"):
                return el.category
            if isinstance(el, dict):
                return el.get("type")
            return None

        def get_text(el):
            if hasattr(el, "text"):
                return el.text
            if isinstance(el, dict):
                return el.get("text", "")
            return str(el)

        document = Document()
        stack = [(document, 0)]

        for el in elements:
            el_type = get_type(el)
            text = get_text(el).strip()
            if not text or el_type is None:
                continue

            # Handle headings/titles
            if el_type in ("Title", "Heading"):
                # Extract heading level
                if not isinstance(el, dict) and hasattr(el, "metadata"):
                    level = getattr(el.metadata, "heading_level", 1) or 1
                else:
                    meta = el.get("metadata", {}) if isinstance(el, dict) else {}
                    level = meta.get("heading_level", 1)
                section = Section(text, level)
                # Find parent section
                while stack and stack[-1][1] >= level:
                    stack.pop()
                parent, _ = stack[-1]
                if isinstance(parent, Document):
                    parent.sections.append(section)
                else:
                    parent.subsections.append(section)
                stack.append((section, level))

            # Handle narrative text as paragraphs
            elif el_type == "NarrativeText":
                parent, _ = stack[-1]
                sp_doc = self.nlp(text)
                sentences = []
                for sent in sp_doc.sents:
                    words = [Word(tok.text) for tok in sent]
                    sentences.append(Sentence(sent.text.strip(), words))
                paragraph = Paragraph(text, sentences)
                # Attach paragraph to parent section
                if isinstance(parent, Document):
                    if not parent.sections:
                        default_sec = Section("", 1)
                        parent.sections.append(default_sec)
                        stack.append((default_sec, 1))
                        parent = default_sec
                    else:
                        parent = parent.sections[-1]
                parent.paragraphs.append(paragraph)

        return document




# Example:
# parser = DocumentParser()
# doc = parser.parse("path/to/file.pdf")
# Traverse via doc.sections
