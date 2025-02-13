import xml.etree.ElementTree as ET
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re


class LanguageError(Exception):
    pass


def read_application(file_path: str) -> str:
    """
    Reads a patent application in XML format.

    Parameters
    ----------
    filepath : str
        The file path to the patent application.

    Returns
    -------
    raw_text : str
        A string containing the full raw text extracted from the patent application.

    """
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    text_elements = []
    english_section_active = False

    for elem in root.iter():
        # Access 'lang' attribute (check if it's 'en')
        lang = elem.attrib.get("lang")

        # Start collecting text when lang="en"
        if lang == "en":
            english_section_active = True

        # If we encounter a different language after an English section, stop
        elif lang is not None and lang != "en" and english_section_active:
            break

        # If in an English section, collect tail text (text after the element)
        if english_section_active and elem.tail and elem.tail.strip():
            text_elements.append(elem.tail.strip())
        if english_section_active and elem.text and elem.text.strip():
            text_elements.append(elem.text.strip())
    if not text_elements:
        raise LanguageError(
            "No sections with 'lang=\"en\"' found in the patent application."
        )

    # Join all collected text into a single string
    raw_text = " ".join(text_elements)

    return raw_text


if __name__ == "__main__":
    file_path = r"C:\Users\patgjoh\Desktop\exjobb\fulltext_en\SE539880C2.xml"
    raw_text = read_application(file_path)
