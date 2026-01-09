"""
Shared text processing utilities
Eliminates code duplication for text chunking and preprocessing
"""

import re
from typing import List


class TextProcessor:
    """Centralized text processing utilities"""

    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Clean and normalize text for better TTS

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = " ".join(text.split())

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(""", "'").replace(""", "'")

        # Remove markdown formatting
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # Bold
        text = re.sub(r"\*(.+?)\*", r"\1", text)  # Italic
        text = re.sub(r"`(.+?)`", r"\1", text)  # Code
        text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)  # Links

        # Remove URLs
        text = re.sub(r"http[s]?://\S+", "", text)

        # Fix common abbreviations to prevent awkward splits
        text = text.replace("e.g.", "for example")
        text = text.replace("i.e.", "that is")
        text = text.replace("etc.", "etcetera")
        text = text.replace("Dr.", "Doctor")
        text = text.replace("Mr.", "Mister")
        text = text.replace("Mrs.", "Misses")
        text = text.replace("Ms.", "Miss")

        return text.strip()

    @staticmethod
    def split_sentences(text: str, max_length: int = 150) -> List[str]:
        """
        Split text into sentences with improved handling

        Args:
            text: Input text
            max_length: Maximum sentence length before splitting

        Returns:
            List of sentences
        """
        # Preprocess text first
        text = TextProcessor.preprocess_text(text)

        # Split on sentence boundaries
        # Use regex to handle periods followed by space and capital letter
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

        # Further split very long sentences at commas or semicolons
        final_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If sentence is too long, split at natural breaks
            if len(sentence) > max_length:
                # Split at commas, semicolons, or conjunctions
                parts = re.split(r"(?<=,)\s+|(?<=;)\s+|\s+(?:and|but|or)\s+", sentence)
                for part in parts:
                    part = part.strip()
                    if part and len(part) > 10:  # Avoid tiny fragments
                        final_sentences.append(part)
            else:
                final_sentences.append(sentence)

        return final_sentences if final_sentences else [text]

    @staticmethod
    def split_into_chunks(text: str, max_length: int = 200) -> List[str]:
        """
        Split text into chunks of reasonable length
        Useful for models that work best with shorter inputs

        Args:
            text: Input text
            max_length: Maximum chunk length

        Returns:
            List of text chunks
        """
        # First split by sentences
        sentences = []
        for delimiter in [". ", "! ", "? ", "\n"]:
            text = text.replace(delimiter, f"{delimiter}|")

        sentence_list = [s.strip() for s in text.split("|") if s.strip()]

        # Then combine sentences into chunks
        chunks = []
        current_chunk = ""

        for sentence in sentence_list:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [text]

    @staticmethod
    def simple_sentence_split(text: str) -> List[str]:
        """
        Simple sentence splitting for basic use cases

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        sentences = []
        for delimiter in [".", "!", "?"]:
            text = text.replace(delimiter, f"{delimiter}|")

        for sentence in text.split("|"):
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)

        return sentences if sentences else [text]
