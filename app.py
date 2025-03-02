# app.py

from src.data_preparation import ConversationExtractor
from src.summarization import ConversationSummarizer
from src.help_detection import ManagerHelped
from src.sentiment_analysis import SentimentAnalyzer
from src.support_mark import SetMarkForManager
from src.utils import get_text
import pandas as pd
import streamlit as st
import os
import numpy as np
import plotly.graph_objects as go
import logging

# Настройка логирования (для записи ошибок и отладочной информации)
logger = logging.getLogger(__name__)


def main():
    """
    Основная функция приложения Streamlit.

    Отображает интерфейс пользователя и управляет логикой приложения.
    """

    # Настройка страницы Streamlit: заголовок, иконка, ширина
    st.set_page_config(
        page_title="Анализ качества работы менеджеров", page_icon="📞", layout="wide"
    )
    st.title("Анализ качества работы менеджеров по продажам")

    # Инициализация переменных состояния (session state)
    # Эти переменные сохраняются между перезагрузками страницы в рамках одной сессии
    if "polite_count" not in st.session_state:
        st.session_state.polite_count = 0  # Счетчик вежливых менеджеров
    if "not_polite_count" not in st.session_state:
        st.session_state.not_polite_count = 0  # Счетчик невежливых менеджеров
    if "manager_helped_count" not in st.session_state:
        st.session_state.manager_helped_count = 0  # Счетчик менеджеров, которые помогли
    if "manager_not_helped_count" not in st.session_state:
        st.session_state.manager_not_helped_count = (
            0  # Счетчик менеджеров, которые не помогли
        )
    if "total_score" not in st.session_state:
        st.session_state.total_score = 0.0  # Сумма оценок всех менеджеров
    if "num_conversations" not in st.session_state:
        st.session_state.num_conversations = 0  # Количество обработанных разговоров

    # Создание временной директории (если она не существует)
    os.makedirs("temp", exist_ok=True)

    # Боковая панель для загрузки файлов
    st.sidebar.header("Загрузка данных")
    uploaded_files = st.sidebar.file_uploader(
        "Загрузите текстовые файлы (.txt)", type="txt", accept_multiple_files=True
    )

    # Если файлы загружены, обрабатываем их
    if uploaded_files:
        table_data = []  # Список для хранения данных из всех файлов
        for uploaded_file in uploaded_files:
            try:
                # Временный файл
                file_path = os.path.join("temp", uploaded_file.name)
                # Сохраняем загруженный файл во временную папку
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Обрабатываем файл с помощью ConversationExtractor
                conversation_extractor = ConversationExtractor(file_path)
                conversation = (
                    conversation_extractor.extract_conversation()
                )  # Извлекаем строки
                # Извлекаем данные и добавляем их в общий список
                table_data.extend(
                    conversation_extractor.extract_data(
                        conversation, uploaded_file.name
                    )
                )
            except Exception as e:
                # В случае ошибки выводим сообщение в интерфейс Streamlit
                st.error(f"Ошибка при обработке файла {uploaded_file.name}: {e}")
                continue  # Переходим к следующему файлу

        # Если данные не были извлечены, выводим сообщение
        if not table_data:
            st.info("Нет данных для анализа.")
            return  # Выходим из функции

        # Создаём DataFrame (таблицу) из извлеченных данных
        df = pd.DataFrame(table_data, columns=["conversation_id", "person", "message"])
        # Объединяем последовательные реплики одного и того же человека в одну строку
        df["message"] = df.groupby(
            ["conversation_id", (df["person"] != df["person"].shift()).cumsum()]
        )["message"].transform(lambda x: " ".join(x))
        # Удаляем дубликаты реплик
        df = df.drop_duplicates(
            subset=["message", "conversation_id", "person"]
        ).reset_index(drop=True)

        # Отображаем таблицу с исходными данными
        st.subheader("Исходные данные")
        st.dataframe(df)

        st.markdown("---")  # Разделитель

        # Создаём экземпляры классов для анализа
        summarizer = ConversationSummarizer()
        set_mark_model = SetMarkForManager()
        manager_help_model = ManagerHelped()
        sentiment_analyzer = SentimentAnalyzer()

        # Перебираем все уникальные ID разговоров
        for conversation_id in df["conversation_id"].unique():
            st.subheader(f"Разговор № {conversation_id}")

            # Извлекаем текст клиента и менеджера (без указания говорящего)
            text_client = get_text(
                df, conversation_id, persons=["Клиент"], speaker=False
            )
            text_manager = get_text(
                df,
                conversation_id,
                persons=["Оператор", "Сотрудник", "Менеджер"],
                speaker=False,
            )

            # Анализируем, помог ли менеджер и был ли он вежлив
            is_manager_helped = manager_help_model.analyze_manager_text(text_client)
            is_manager_was_polite = sentiment_analyzer.analyze_text_sentiment(
                text_manager
            )
            # Вычисляем итоговую оценку
            mark = set_mark_model.result(text_manager, text_client)

            # Обновляем счетчики в session_state
            st.session_state.num_conversations += 1
            st.session_state.total_score += mark

            if is_manager_helped:
                st.session_state.manager_helped_count += 1
            else:
                st.session_state.manager_not_helped_count += 1

            if is_manager_was_polite:
                st.session_state.polite_count += 1
            else:
                st.session_state.not_polite_count += 1

            # Отображаем результаты анализа для текущего разговора
            col1, col2, col3 = st.columns(3)  # Создаём три колонки
            with col1:
                # Используем st.metric для красивого отображения "Да" или "Нет"
                st.metric("Помощь клиенту", "Да" if is_manager_helped else "Нет")
            with col2:
                st.metric(
                    "Вежливость менеджера", "Да" if is_manager_was_polite else "Нет"
                )
            with col3:
                st.metric(
                    "Итоговая оценка", f"{mark:.2f}"
                )  # Отображаем оценку с двумя знаками

            # Блок с кратким содержанием разговора (сворачиваемый)
            with st.expander(f"Краткое содержание разговора № {conversation_id}"):
                conversation_summary = summarizer.summarize_conversation(
                    df, conversation_id
                )
                if conversation_summary:  # Проверяем, что суммаризация прошла успешно
                    st.write(conversation_summary)  # Выводим краткое содержание
                else:
                    st.write("Не удалось получить краткое содержание.")

            # Блок с полным текстом разговора (сворачиваемый)
            with st.expander(f"Полный текст разговора № {conversation_id}"):
                # Извлекаем текст *всех* участников разговора
                text_with_punctuation_all = get_text(
                    df,
                    conversation_id,
                    persons=["Оператор", "Сотрудник", "Менеджер", "Клиент"],
                )
                # Отображаем текст в многострочном поле
                st.text_area(label="", value=text_with_punctuation_all, height=400)

            st.markdown("---")  # Разделитель

        # Общая статистика по всем разговорам (с использованием Plotly)
        st.subheader("Общая статистика по всем разговорам")

        # Проверяем, были ли обработаны разговоры (чтобы избежать деления на ноль)
        if st.session_state.num_conversations > 0:
            # Вычисляем среднюю оценку
            avg_score = (
                st.session_state.total_score / st.session_state.num_conversations
            )
            st.metric("Средняя оценка по всем разговорам", f"{avg_score:.2f}")

            # Создаём две колонки для графиков
            col1, col2 = st.columns(2)
            with col1:
                st.write("Вежливость менеджеров")
                polite_labels = ["Вежлив", "Невежлив"]  # Подписи для графика
                polite_values = [
                    st.session_state.polite_count,
                    st.session_state.not_polite_count,
                ]  # Значения
                # Создаём круговую диаграмму с помощью Plotly
                polite_fig = go.Figure(
                    data=[go.Pie(labels=polite_labels, values=polite_values, hole=0.3)]
                )
                st.plotly_chart(polite_fig, use_container_width=True)  # Отображаем

            with col2:
                st.write("Помощь менеджеров")
                manager_help_labels = ["Помог", "Не помог"]
                manager_help_values = [
                    st.session_state.manager_helped_count,
                    st.session_state.manager_not_helped_count,
                ]
                manager_help_fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=manager_help_labels,
                            values=manager_help_values,
                            hole=0.3,  # Отверстие в центре
                        )
                    ]
                )
                st.plotly_chart(
                    manager_help_fig, use_container_width=True
                )  # Отображаем
        else:
            st.info("Нет данных для отображения общей статистики.")

    else:
        st.info("Пожалуйста, загрузите файлы для анализа.")


if __name__ == "__main__":
    main()
