import os  # Importing the os module for file operations
from sbert_punc_case_ru import SbertPuncCase  # Importing a model for sentence punctuation and casing in Russian

# Initializing the SbertPuncCase model for sentence punctuation and casing
SbertPuncCase_model = SbertPuncCase()

class ConversationExtractor:
    def __init__(self, file_path):
        self.file_path = file_path

    def extract_conversation(self):
        # Extracting conversation lines from the file
        with open(self.file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            phrases = []
            for line in lines:
                line = line.strip()
                phrases.append(line)
        return phrases

    def extract_data(self, conversation, conversation_id):
        data = []
        # Extracting conversation data from lines
        for phrase in conversation:
            person = phrase.split(":")[0].strip().capitalize()
            message = phrase.split(":")[1].strip()
            data.append((conversation_id, person, message))
        return data