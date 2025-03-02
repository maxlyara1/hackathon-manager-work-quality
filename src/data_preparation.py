# src/data_preparation.py

import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)


class ConversationExtractor:
    """
    Класс для извлечения и структурирования данных из текстовых файлов.
    """

    def __init__(self, file_path: str):
        """
        Инициализирует ConversationExtractor.

        Args:
            file_path: Путь к текстовому файлу.
        """
        self.file_path = file_path
        self.speaker_pattern = re.compile(r"^(.*?):(.*)$")

    def extract_conversation(self) -> List[str]:
        """
        Извлекает строки из файла.

        Returns:
            Список строк.

        Raises:
            FileNotFoundError: Если файл не найден.
            Exception: Если произошла ошибка при чтении.
        """
        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                lines = (line.strip() for line in file)
                return [line for line in lines if line]
        except FileNotFoundError:
            logger.error(f"Файл не найден: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Ошибка при чтении файла {self.file_path}: {e}")
            raise

    def extract_data(
        self, conversation: List[str], conversation_id: str
    ) -> List[Dict[str, str]]:
        """
        Извлекает данные из строк разговора.

        Args:
            conversation: Список строк разговора.
            conversation_id: ID разговора.

        Returns:
            Список словарей с данными реплик.
        """
        data: List[Dict[str, str]] = []
        for phrase in conversation:
            match = self.speaker_pattern.match(phrase)
            if match:
                person, message = match.groups()
                person = person.strip().capitalize()
                message = message.strip()
                if person and message:
                    data.append(
                        {
                            "conversation_id": conversation_id,
                            "person": person,
                            "message": message,
                        }
                    )
            else:
                logger.warning(
                    f"Некорректный формат строки: {phrase} в файле {conversation_id}"
                )
        return data
