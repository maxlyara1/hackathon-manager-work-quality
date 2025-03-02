# src/sentiment_analysis.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

# Настройка логгера
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Класс для анализа тональности текста с помощью предобученной модели Hugging Face.
    """

    def __init__(self, model_checkpoint="cointegrated/rubert-tiny-sentiment-balanced"):
        """
        Инициализирует анализатор тональности.

        Args:
            model_checkpoint: Идентификатор предобученной модели на Hugging Face Hub.
        """
        self.model_checkpoint = model_checkpoint  # Сохраняем идентификатор модели
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # Определяем устройство: GPU (cuda) или CPU
        logger.info(
            f"Используется устройство: {self.device}"
        )  # Логируем, какое устройство используется
        self.tokenizer = None  # Инициализируем токенизатор (будет загружен позже)
        self.model = None  # Инициализируем модель (будет загружена позже)
        self.model_initialized = False  # Флаг, указывающий, загружена ли модель

    def initialize_model(self):
        """
        Загружает токенизатор и модель с Hugging Face Hub.
        Перемещает модель на нужное устройство (GPU/CPU).
        Переводит модель в режим оценки (eval mode).
        """
        if self.model_initialized:
            return  # Если модель уже загружена, ничего не делаем

        try:
            # Загружаем токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
            # Загружаем модель
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_checkpoint
            )
            self.model.to(self.device)  # Перемещаем модель на GPU или CPU
            self.model.eval()  # Переводим модель в режим оценки (отключаем обучение)
            self.model_initialized = True  # Устанавливаем флаг, что модель загружена
            logger.info(f"Модель '{self.model_checkpoint}' успешно загружена.")

        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")  # Логируем ошибку
            self.model_initialized = False  # Если произошла ошибка, сбрасываем флаг

    def get_sentiment(self, text: str, return_type: str = "label") -> str | float:
        """
        Определяет тональность текста.

        Args:
            text: Текст для анализа.
            return_type: Тип возвращаемого значения:
                - 'label' (по умолчанию): Возвращает метку тональности ('POSITIVE', 'NEGATIVE', 'NEUTRAL').
                - 'score': Возвращает числовую оценку тональности (от -1 до 1).
                - 'proba': Возвращает вероятности для каждой метки.

        Returns:
            Результат анализа тональности (метка, оценка или вероятности).
            Если модель не загружена, возвращает 'Unknown' (для label) или 0.0 (для score).

        Raises:
            ValueError: Если указан некорректный return_type.
        """
        if not self.model_initialized:
            self.initialize_model()  # Пытаемся загрузить модель, если она ещё не загружена
            if not self.model_initialized:
                return "Unknown" if return_type == "label" else 0.0

        try:
            with torch.no_grad():  # Отключаем вычисление градиентов (нужно только для оценки)
                # Токенизируем текст и подготавливаем входные данные для модели
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",  # Возвращаем тензоры PyTorch
                    truncation=True,  # Обрезаем текст, если он слишком длинный
                    padding=True,  # Дополняем текст, если он слишком короткий
                    max_length=512,  # Максимальная длина текста
                ).to(
                    self.device
                )  # Перемещаем входные данные на GPU или CPU

                # Получаем выход модели (логиты)
                outputs = self.model(**inputs)
                logits = outputs.logits
                # Преобразуем логиты в вероятности с помощью softmax
                probabilities = (
                    torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
                )

            # Возвращаем результат в зависимости от запрошенного типа
            if return_type == "label":
                return self.model.config.id2label[
                    probabilities.argmax()
                ]  # Возвращаем метку с наибольшей вероятностью
            elif return_type == "score":
                # Возвращаем взвешенную сумму вероятностей (где веса -1, 0, 1)
                return probabilities.dot([-1, 0, 1])
            elif return_type == "proba":
                return probabilities  # Возвращаем вероятности для всех меток
            else:
                raise ValueError(
                    "Неверный return_type. Должен быть 'label', 'score' или 'proba'."
                )

        except Exception as e:
            logger.error(f"Ошибка при анализе тональности: {e}")  # Логируем ошибку
            return (
                "Unknown" if return_type == "label" else 0.0
            )  # В случае ошибки возвращаем дефолт

    def analyze_text_sentiment(self, text: str) -> bool:
        """
        Анализирует тональность текста и возвращает True, если тональность
        положительная или нейтральная, и False, если тональность отрицательная.
        Также выводит в консоль входной текст, предсказанную метку и вероятности.
        """
        label = self.get_sentiment(
            text, return_type="label"
        )  # Получаем метку тональности
        return label.lower() in (
            "positive",
            "neutral",
        )  # Возвращаем True, если метка "positive" или "neutral"
