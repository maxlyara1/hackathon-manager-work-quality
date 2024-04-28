from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, Doc
from manager_sentiment_analysis import SentimentAnalyzer
from manager_help_detection import ManagerHelped

class SetMarkForManager:
    def __init__(self):
        self.segmenter = Segmenter()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.sentiment_analyzer = SentimentAnalyzer()

    def set_BadWordsMark(self, text): # частота INTJ и PROPN во всём тексте: от 0 до 1
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        count = 0
        bad_words = []
        num_tokens = len(doc.tokens)
        if num_tokens > 0:
            for token in doc.tokens:
                if ('INTJ' in token.pos) or ('PROPN' in token.pos):
                    bad_words.append(token.text)
                    count += 1
            return count / num_tokens
        else:
            return 0  # If there are no tokens, return 0.

    def set_SentimentMark(self, text): # sentiment score: от -1 до 1
        sentiment_mark = self.sentiment_analyzer.get_sentiment(text, return_type = 'score')
        return sentiment_mark
    def set_Manager_Help_mark(self, text): # не помог или помог менеджер в решении задачи: 0 или 1
        model = ManagerHelped()
        return model.analyze_manager_text(text)
    def result(self, text_manager, text_client): # от 1 до 5
        return 1 + self.set_BadWordsMark(text_manager) + self.set_SentimentMark(text_manager) + self.set_Manager_Help_mark(text_client) * 2