# Анализ Качества Работы Менеджеров по Продажам на Основе Телефонных Разговоров (с использованием NLP и Streamlit)

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.37.2-yellow.svg)](https://huggingface.co/transformers/)
[![Natasha](https://img.shields.io/badge/Natasha-1.6.0-orange.svg)](https://natasha.github.io/)

**Описание**

Этот проект представляет собой end-to-end систему для автоматического анализа качества работы менеджеров по продажам на основе текстовой расшифровки телефонных разговоров с клиентами.  Система использует передовые методы обработки естественного языка (NLP) и машинного обучения (ML) для извлечения ключевой информации из диалогов и оценки работы менеджера по нескольким параметрам. Проект был разработан как решение для хакатона и занял призовое место.

**Ключевые возможности:**

*   **Извлечение и структурирование данных:** Автоматическое извлечение диалогов из текстовых файлов, разделение реплик по говорящим (менеджер/клиент).
*   **Суммаризация диалогов:** Генерация краткого содержания каждого разговора с использованием state-of-the-art модели `RussianNLP/FRED-T5-large-summarization` (T5-based).
*   **Определение вежливости менеджера:** Анализ тональности (сентимента) текста менеджера с использованием предобученной модели `cointegrated/rubert-tiny-sentiment-balanced` (BERT-based).  Учитываются как положительные, так и нейтральные реплики.
*   **Определение помощи менеджера:** Выявление ключевых слов и фраз в тексте клиента, указывающих на то, что менеджер оказал помощь (например, слова благодарности).
*   **Комплексная оценка:** Вычисление итоговой оценки работы менеджера на основе комбинации метрик (вежливость, помощь, отсутствие ненормативной лексики).
*   **Интерактивный веб-интерфейс (Streamlit):**  Удобный и интуитивно понятный интерфейс, позволяющий загружать файлы с разговорами, просматривать результаты анализа, читать как полные тексты разговоров, так и их краткое содержание.
*   **Визуализация данных:**  Использование библиотеки Plotly для наглядного представления сводной статистики по всем загруженным разговорам (круговые диаграммы).
*   **Надёжный запуск:** Скрипт `launch.py` автоматически создаёт виртуальное окружение (venv), устанавливает все необходимые зависимости и запускает приложение.  Это упрощает развёртывание и использование проекта.
* **Логирование:** В проекте настроено логирование действий.

**Используемые технологии и библиотеки:**

*   **Python:** Основной язык программирования.
*   **Streamlit:** Фреймворк для создания интерактивных веб-приложений для анализа данных и машинного обучения.
*   **Pandas:** Библиотека для обработки и анализа данных (используется для работы с табличными данными разговоров).
*   **Transformers (Hugging Face):**  Библиотека, предоставляющая доступ к предобученным моделям NLP (BERT, T5 и др.).  Ключевые модели:
    *   `RussianNLP/FRED-T5-large-summarization`:  Модель для генерации краткого содержания текста (суммаризации) на русском языке.
    *   `cointegrated/rubert-tiny-sentiment-balanced`:  Модель для анализа тональности (сентимента) текста на русском языке.
    *    `sbert_punc_case_ru`: Модель для расстановки знаков препинания.
*   **Natasha:**  Библиотека для обработки русского языка (морфологический анализ, сегментация текста, работа с именованными сущностями). Используется для выявления ненормативной лексики.
*   **Plotly:**  Библиотека для создания интерактивных графиков (круговые диаграммы для визуализации статистики).
*   **NumPy:**  Библиотека для работы с многомерными массивами (используется в `transformers`).
*   **Logging:**  Стандартный модуль Python для логирования (отслеживание ошибок и хода выполнения).

**Архитектура проекта**

Проект имеет модульную архитектуру, что облегчает понимание, поддержку и расширение кода:

```
hackathon-manager-work-quality/
├── src/                      # Исходный код
│   ├── data_preparation.py   # Извлечение и подготовка данных
│   ├── summarization.py      # Суммаризация диалогов
│   ├── sentiment_analysis.py # Анализ тональности
│   ├── help_detection.py    # Определение помощи менеджера
│   ├── support_mark.py      # Вычисление итоговой оценки
│   └── utils.py             # Вспомогательные функции
├── app.py                  # Основной файл приложения Streamlit
├── requirements.txt        # Список зависимостей
├── launch.py               # Скрипт для запуска приложения
├── README.md               # Этот файл
├── .gitignore              # Файлы, игнорируемые Git'ом
└── data/                   # (Опционально) Примеры данных
    └── example.txt
```

*   **`src/`:**  Содержит модули с основной логикой приложения.  Каждый модуль отвечает за отдельную задачу (извлечение данных, суммаризация, анализ тональности и т.д.).
*   **`app.py`:**  Основной файл Streamlit-приложения.  Отвечает за пользовательский интерфейс и взаимодействие с модулями из `src/`.
*   **`requirements.txt`:**  Содержит список всех необходимых Python-библиотек с указанием версий.
*   **`launch.py`:**  Скрипт, который автоматизирует процесс запуска приложения:
    *   Создаёт виртуальное окружение (venv), если оно ещё не создано.
    *   Устанавливает все зависимости из `requirements.txt`.
    *   Запускает приложение Streamlit.
*   **`README.md`:**  Этот файл. Содержит описание проекта, инструкции по установке и запуску, а также другую полезную информацию.
*   **`.gitignore`:** Определяет, какие файлы и папки не нужно отслеживать системе контроля версий Git (например, виртуальное окружение `venv`, временные файлы).

**Установка и запуск**

0.  **Установите Homebrew (если еще не установлен, для macOS):**

    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

    Homebrew — это менеджер пакетов для macOS, который упрощает установку многих полезных инструментов.

1.  **Клонирование репозитория:**

    ```bash
    git clone https://github.com/maxiyara1/hackathon-manager-work-quality.git
    cd hackathon-manager-work-quality
    ```

2.  **Запуск приложения:**

    ```bash
    python launch.py
    ```

    Скрипт `launch.py` автоматически создаст виртуальное окружение `venv`, установит все необходимые зависимости и запустит приложение Streamlit.  При первом запуске может потребоваться некоторое время на загрузку моделей.

3.  **Откройте приложение в браузере:** После запуска, launch.py покажет в консоли адрес по типу:
    *   Local URL: `http://localhost:8501`

**Использование**

<img width="1680" alt="image" src="https://github.com/user-attachments/assets/a12c9aa3-3cab-445d-9942-7f9d6a8418bb" />

1.  **Загрузка данных:**  В боковой панели приложения нажмите кнопку "Загрузите текстовые файлы (.txt)" и выберите один или несколько файлов с расшифровкой телефонных разговоров.  Каждый файл должен содержать один разговор.
2.  **Просмотр результатов:**  После загрузки файлов приложение автоматически обработает их и отобразит:
    *   Таблицу с исходными данными (реплики разговоров).
    *   Результаты анализа для *каждого* разговора:
        *   **Помощь клиенту:**  "Да" или "Нет".
        *   **Вежливость менеджера:**  "Да" или "Нет" (учитываются как положительные, так и нейтральные реплики).
        *   **Итоговая оценка:**  Число от 0 до 1 (чем выше, тем лучше).
    *   Краткое содержание каждого разговора (в сворачиваемом блоке).
    *   Полный текст каждого разговора с восстановленной пунктуацией (в сворачиваемом блоке).
    *   Общую статистику по всем загруженным разговорам (круговые диаграммы).

**Дополнительная информация**

*   Проект использует **модульную архитектуру**, что делает код более организованным, читаемым и поддерживаемым.
*   В коде используются **аннотации типов (type hints)**, что улучшает читаемость и помогает выявлять ошибки.
*   Используется **логирование** для записи ошибок и отладочной информации.
*   Реализована **обработка ошибок** (например, при чтении файлов, загрузке моделей, анализе текста).
*   Используются **лучшие практики** программирования на Python и Streamlit.

**Этот проект демонстрирует следующие навыки и компетенции:**

*   **Разработка на Python:** Уверенное владение языком Python, использование стандартных библиотек, работа с файлами, обработка ошибок.
*   **NLP (Natural Language Processing):**  Практический опыт применения методов обработки естественного языка для решения реальных задач.
*   **Машинное обучение (ML):**  Использование предобученных моделей машинного обучения (BERT, T5) из библиотеки `transformers`.
*   **Streamlit:**  Создание интерактивных веб-приложений для демонстрации результатов работы ML-моделей.
*   **Pandas:**  Обработка и анализ данных с помощью библиотеки Pandas.
*   **Plotly:**  Визуализация данных с помощью библиотеки Plotly.
*   **Работа с виртуальными окружениями:**  Использование `venv` для изоляции зависимостей проекта.
*   **Git:**  Работа с системой контроля версий Git.
*   **Модульная архитектура:**  Разработка структурированного и поддерживаемого кода.
*   **Решение реальных задач:**  Проект направлен на решение конкретной бизнес-задачи (анализ качества работы менеджеров).
*   **End-to-end разработка:**  Проект представляет собой законченное решение, от загрузки данных до визуализации результатов.
*   **Понимание принципов MLOps:**  Проект демонстрирует базовые принципы MLOps (воспроизводимость, автоматизация, развёртывание).
