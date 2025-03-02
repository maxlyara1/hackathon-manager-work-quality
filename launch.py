# launch.py

import subprocess  # Модуль для запуска внешних команд (как в терминале)
import sys  # Модуль для доступа к системным функциям (например, выход из программы)
import os  # Модуль для работы с операционной системой (пути к файлам и папкам)
import platform  # Модуль для определения типа операционной системы (Windows, macOS, Linux)


def main():
    """
    Основная функция, которая запускает приложение Streamlit.

    Этапы работы:
    1. Создаёт виртуальное окружение (venv), если оно ещё не создано.
    2. Определяет путь к исполняемому файлу Python внутри venv.
    3. Устанавливает/обновляет зависимости, используя pip, *внутри* venv.
    4. Запускает приложение Streamlit, используя Python *из venv*.
    """

    venv_name = "venv"  # Имя папки для виртуального окружения (можно изменить)
    venv_path = os.path.join(os.getcwd(), venv_name)  # Полный путь к папке venv

    # 1. Создаём venv, если его ещё нет.
    if not os.path.exists(venv_path):  # Проверяем, существует ли папка venv
        print(f"Создание виртуального окружения '{venv_name}'...")
        try:
            # Используем sys.executable, чтобы venv создался с тем же Python,
            # который сейчас запущен (это может быть системный Python, Python из conda, и т.д.)
            subprocess.run(
                [sys.executable, "-m", "venv", venv_name],
                check=True,  # Если команда завершится с ошибкой, программа остановится
                capture_output=True,  # Сохраняем вывод команды (и ошибки)
                text=True,  # Вывод будет текстом, а не байтами
            )
            print(f"Виртуальное окружение '{venv_name}' успешно создано.")
        except subprocess.CalledProcessError as e:
            print(
                f"Ошибка при создании виртуального окружения:\n{e.stderr}"
            )  # Печатаем ошибку
            sys.exit(1)  # Завершаем программу с кодом ошибки

    # 2. Определяем путь к Python *внутри* venv.
    if platform.system() == "win32":  # Если операционная система Windows
        python_executable = os.path.join(
            venv_path, "Scripts", "python.exe"
        )  # Путь на Windows
    else:  # Если macOS или Linux
        python_executable = os.path.join(
            venv_path, "bin", "python"
        )  # Путь на macOS/Linux

    # 3. Проверяем, существует ли исполняемый файл Python внутри venv (дополнительная проверка).
    if not os.path.exists(python_executable):
        print(f"Ошибка: Python не найден в venv: {python_executable}")
        print("  Убедитесь, что виртуальное окружение создано правильно.")
        sys.exit(1)

    # 4. Устанавливаем/обновляем зависимости, используя Python *из venv*.
    try:
        print("Installing/upgrading dependencies...")
        subprocess.run(
            [
                python_executable,  # Используем Python *из venv*!
                "-m",
                "pip",  # Запускаем pip *как модуль* Python (это надёжнее)
                "install",
                "-r",
                "requirements.txt",  # Устанавливаем зависимости из файла requirements.txt
                "--upgrade",  # Обновляем зависимости, если они уже установлены
            ],
            check=True,  # Останавливаем программу, если команда завершилась с ошибкой
            capture_output=True,  # Сохраняем вывод (и ошибки)
            text=True,  # Вывод будет текстом
        )
        print("Зависимости успешно установлены/обновлены.")
    except subprocess.CalledProcessError as e:
        print(
            f"Ошибка при установке/обновлении зависимостей:\n{e.stdout}\n{e.stderr}"
        )  # Печатаем и stdout, и stderr
        sys.exit(1)

    # 5. Запускаем приложение Streamlit, используя Python *из venv*.
    try:
        subprocess.run(
            [python_executable, "-m", "streamlit", "run", "app.py"], check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при запуске Streamlit:\n{e}")
        sys.exit(1)
    except KeyboardInterrupt:  # Обрабатываем прерывание пользователем (Ctrl+C)
        print("\nЗапуск приложения прерван пользователем.")
        sys.exit(0)


if __name__ == "__main__":
    main()
