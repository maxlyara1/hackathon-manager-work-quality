# src/support_mark.py
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, Doc
from .sentiment_analysis import SentimentAnalyzer
from .help_detection import ManagerHelped
import logging

logger = logging.getLogger(__name__)


class SetMarkForManager:
    """
    Выставляет итоговую оценку менеджеру.
    """

    def __init__(self) -> None:
        self.segmenter = Segmenter()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.manager_help = ManagerHelped()
        self.bad_words = [
            "пизд",
            "хуй",
            "ебл",
            "ебат",
            "уеб",
            "бля",
            "сука",
            "гондон",
            "мразь",
            "тварь",
            "долбоеб",
            "пидор",
            "педик",
        ]

    def set_BadWordsMark(self, text: str) -> float:
        """Определяем наличие и долю ненормативной лексики (от 0 до 1)"""
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        bad_words_count = 0
        total_tokens = 0

        for token in doc.tokens:
            token_text = token.text.lower()
            if token.pos != "PUNCT":
                total_tokens += 1
                if any(bad_word in token_text for bad_word in self.bad_words):
                    bad_words_count += 1
        return min(bad_words_count / total_tokens, 1) if total_tokens > 0 else 0.0

    def set_SentimentMark(self, text: str) -> float:
        """Определяем сентимент (от -1 до 1)"""
        try:
            sentiment_score = self.sentiment_analyzer.get_sentiment(text, "score")
            return sentiment_score  # Возвращаем "сырую" оценку
        except Exception as e:
            logger.error(f"Ошибка при выставлении оценки за сентимент: {e}")
            return 0.0

    def set_ManagerHelp_mark(self, client_text: str) -> int:
        """Помог или не помог менеджер в решении задачи"""
        return int(self.manager_help.analyze_manager_text(client_text))

    def result(self, text_manager: str, text_client: str) -> float:
        """Итоговая оценка"""
        try:
            badWordsMark = self.set_BadWordsMark(
                text_manager
            )  # от 0 до 1, где 1 - много плохих слов, 0 - нет
            sentimentMark = self.set_SentimentMark(text_manager)  # от -1 до 1
            managerHelpMark = self.set_ManagerHelp_mark(text_client)  # 0 или 1

            # Итоговый результат (можно настроить веса)
            final_score = (
                0.3 * (1 - badWordsMark)
                + 0.4 * (sentimentMark + 1) / 2
                + 0.3 * managerHelpMark
            )
            return round(final_score, 2)  # Округляем до двух знаков после запятой

        except Exception as e:
            logger.error(f"Ошибка при вычислении итоговой оценки: {e}")
            return 0.0
