# src/help_detection.py
import logging

logger = logging.getLogger(__name__)


class ManagerHelped:
    """
    Класс для определения, помог ли менеджер клиенту.
    """

    def __init__(self):
        self.keywords = [
            "спасибо",
            "благодарю",
            "понятно",
            "хорошо",
            "отлично",
            "помогло",
            "вопрос решен",
            "проблема решена",
            "очень помогли",
            "большое спасибо",
            "огромное спасибо",
            "супер",
            "замечательно",
            "прекрасно",
            "все получилось",
            "все работает",
            "здорово",
        ]

    def analyze_manager_text(self, client_text: str) -> bool:
        """
        Анализирует текст клиента, ищет ключевые слова благодарности.

        Args:
            client_text: Текст клиента.

        Returns:
            True, если менеджер помог, False иначе.
        """
        try:
            text_lower = client_text.lower()
            for keyword in self.keywords:
                if keyword in text_lower:
                    return True
            return False
        except Exception as e:
            logger.error(f"Ошибка при анализе текста: {e}")
            return False
