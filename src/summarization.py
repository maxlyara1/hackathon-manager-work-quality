# src/summarization.py

import torch
from transformers import GPT2Tokenizer, T5ForConditionalGeneration
import pandas as pd
from .utils import get_text  # Относительный импорт (из текущего пакета)
from sbert_punc_case_ru import SbertPuncCase
import logging

# Настройка логгирования
logger = logging.getLogger(__name__)


class ConversationSummarizer:
    """
    Класс для суммаризации (краткого пересказа) диалогов с использованием модели FRED-T5-Summarizer.
    """

    def __init__(self):
        """
        Инициализирует ConversationSummarizer, загружает модель и токенизатор.
        """
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # Определяем, доступен ли GPU
        logger.info(
            f"Используется устройство: {self.device}"
        )  # Сообщаем, какое устройство используется
        self.sbert_punc_case_model = (
            SbertPuncCase()
        )  # Создаём объект для расстановки пунктуации

        try:
            logger.info("Загрузка токенизатора RussianNLP/FRED-T5-Summarizer...")
            # Загружаем токенизатор.  GPT2Tokenizer используется с моделью FRED-T5.
            # eos_token='</s>' - это специальный токен, который обозначает конец текста.
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                "RussianNLP/FRED-T5-Summarizer", eos_token="</s>"
            )
            logger.info("Токенизатор успешно загружен.")

            logger.info("Загрузка модели RussianNLP/FRED-T5-Summarizer...")
            # Загружаем саму модель для суммаризации.
            self.model = T5ForConditionalGeneration.from_pretrained(
                "RussianNLP/FRED-T5-Summarizer"
            )
            self.model.to(self.device)  # Перемещаем модель на GPU (если есть) или CPU
            self.model.eval()  # Переводим модель в режим "оценки" (evaluation mode).  В этом режиме модель не обучается.
            logger.info("Модель успешно загружена.")

        except Exception as e:
            logger.exception(
                f"Ошибка при загрузке модели: {e}"
            )  # Логируем ошибку (если она возникнет)
            raise  # Передаём ошибку дальше, чтобы остановить выполнение программы

    def _add_punctuation(self, text: str) -> str:
        """
        Добавляет знаки препинания в текст, используя sbert_punc_case_ru.
        Это приватный метод (начинается с _), он нужен только внутри этого класса.
        """
        return self.sbert_punc_case_model.punctuate(text)

    def summarize_conversation(
        self, conversation_df: pd.DataFrame, conversation_id: str
    ) -> str:
        """
        Суммаризирует (делает краткий пересказ) диалога.

        Args:
            conversation_df:  DataFrame (таблица) с данными диалога.
            conversation_id:  Идентификатор (номер) диалога.

        Returns:
            Краткое содержание диалога (строка) или пустую строку, если произошла ошибка.
        """
        try:
            # Получаем текст диалога из DataFrame, используя функцию get_text (из файла utils.py).
            # Извлекаем реплики оператора, сотрудника, менеджера и клиента.
            input_text = get_text(
                conversation_df,
                conversation_id,
                persons=["Оператор", "Сотрудник", "Менеджер", "Клиент"],
                speaker=True,  # Включаем имена говорящих в текст
            )

            # Добавляем специальный префикс, как показано в примере на Hugging Face.
            input_text = "<LM> Сократи текст.\n" + input_text

            with torch.no_grad():  # Отключаем вычисление градиентов (это нужно только при обучении модели)
                # Преобразуем текст в числовой формат (токены), который понимает модель.
                input_ids = torch.tensor([self.tokenizer.encode(input_text)]).to(
                    self.device
                )
                # Генерируем краткое содержание.  Здесь много параметров, которые влияют на результат:
                #   - eos_token_id:  ID токена конца текста.
                #   - num_beams:     "Ширина луча" (beam search).  Влияет на качество, но замедляет работу.
                #   - min_new_tokens, max_new_tokens:  Минимальная и максимальная длина *нового* текста (не считая входного).
                #   - do_sample:      Использовать ли сэмплирование (True) или жадный поиск (False).
                #   - no_repeat_ngram_size:  Не повторять n-граммы (последовательности слов).
                #   - top_p:          Параметр для сэмплирования (nucleus sampling).
                outputs = self.model.generate(
                    input_ids,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_beams=5,
                    min_new_tokens=17,
                    max_new_tokens=200,
                    do_sample=True,
                    no_repeat_ngram_size=4,
                    top_p=0.9,
                )
                # Преобразуем числовой результат обратно в текст.
                result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            return result
        except Exception as e:
            logger.error(f"Ошибка при суммаризации диалога {conversation_id}: {e}")
            return ""  # В случае ошибки возвращаем пустую строку
