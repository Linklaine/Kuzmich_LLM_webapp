import streamlit as st
import yaml
import os
import requests

st.set_page_config(page_title="Настройки", page_icon="⚙️", layout="wide")

CONFIG_FILE = "config.yaml"

def load_config():
    """Загружает текущую конфигурацию"""
    if not os.path.exists(CONFIG_FILE):
        return None
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(config):
    """Сохраняет конфигурацию в файл"""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

def get_available_models(host, port):
    """Получает список доступных моделей из LM Studio API"""
    try:
        url = f"http://{host}:{port}/v1/models"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            return [model["id"] for model in models_data.get("data", [])]
        else:
            return []
    except Exception:
        return []

def show_settings_page():
    st.title("⚙️ Настройки сервиса")
    st.markdown("На этой странице вы можете изменить параметры подключения и поведения сервиса.")
    
    # Загружаем текущую конфигурацию
    config = load_config()
    
    if config is None:
        st.error(f"Файл конфигурации `{CONFIG_FILE}` не найден!")
        st.info("Создайте файл config.yaml в корневой папке проекта.")
        return
    
    st.divider()
    
    # === Раздел 1: Настройки LM Studio ===
    st.subheader("🔌 Подключение к LM Studio")
    
    col1, col2 = st.columns(2)
    
    with col1:
        host = st.text_input(
            "Хост сервера",
            value=config["lm_studio"].get("host", "localhost"),
            help="IP-адрес или доменное имя сервера LM Studio",
            key="host_input"
        )
    
    with col2:
        port = st.number_input(
            "Порт сервера",
            min_value=1,
            max_value=65535,
            value=config["lm_studio"].get("port", 1234),
            help="Порт, на котором запущен Local Server LM Studio",
            key="port_input"
        )
    
    # Обновляем конфиг новыми значениями хоста и порта
    config["lm_studio"]["host"] = host
    config["lm_studio"]["port"] = port
    
    config["lm_studio"]["api_key"] = st.text_input(
        "API Key",
        value=config["lm_studio"].get("api_key", "lm-studio"),
        help="Ключ API (для LM Studio может быть любым)"
    )
    
    # === Загрузка списка моделей ===
    st.markdown("**Доступные модели:**")
    available_models = get_available_models(host, port)
    
    if available_models:
        # Определяем индекс текущей модели по умолчанию
        current_default = config["app"].get("default_model", "")
        default_index = 0
        if current_default in available_models:
            default_index = available_models.index(current_default)
        
        # Селектор модели
        selected_model = st.selectbox(
            "Модель по умолчанию",
            options=available_models,
            index=default_index,
            help="Имя модели, которая будет выбрана при запуске"
        )
        
        config["app"]["default_model"] = selected_model
        
        st.success(f"✅ Найдено моделей: {len(available_models)}")
    else:
        st.warning("⚠️ Не удалось получить список моделей. Проверьте подключение к LM Studio.")
        # Резервное поле ввода вручную
        config["app"]["default_model"] = st.text_input(
            "Модель по умолчанию (вручную)",
            value=config["app"].get("default_model", ""),
            help="Введите имя модели вручную, если она не обнаружена"
        )
    
    st.divider()
    
    # === Раздел 2: Настройки приложения ===
    st.subheader("📦 Настройки приложения")
    
    col1, col2 = st.columns(2)
    
    with col1:
        config["app"]["history_file_name"] = st.text_input(
            "Файл истории",
            value=config["app"].get("history_file_name", "history.json"),
            help="Имя файла для сохранения истории запросов"
        )
    
    with col2:
        config["app"]["history_length"] = st.number_input(
            "Максимум записей в истории",
            min_value=1,
            max_value=1000,
            value=config["app"].get("history_length", 10),
            help="Количество последних запросов, сохраняемых в истории"
        )
    
    st.divider()
    
    # === Предпросмотр конфигурации ===
    st.subheader("📋 Предпросмотр конфигурации")
    
    with st.expander("Показать YAML"):
        st.code(yaml.dump(config, allow_unicode=True, default_flow_style=False), language="yaml")
    
    st.divider()
    
    # === Кнопки действий ===
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        if st.button("💾 Сохранить настройки", type="primary", use_container_width=True):
            try:
                save_config(config)
                st.success("✅ Настройки успешно сохранены!")
                st.info("⚠️ Для применения некоторых настроек может потребоваться перезапуск приложения.")
            except Exception as e:
                st.error(f"❌ Ошибка при сохранении: {e}")
    
    with col2:
        if st.button("🔄 Обновить список моделей", use_container_width=True):
            st.rerun()
    
    with col3:
        if st.button("🗑 Сброс", use_container_width=True):
            default_config = {
                "lm_studio": {"host": "localhost", "port": 1234, "api_key": "lm-studio"},
                "app": {"default_model": "llama3.1:8b-instruct", "history_file_name": "history.json", "history_length": 10}
            }
            save_config(default_config)
            st.success("Настройки сброшены!")
            st.rerun()

if __name__ == "__main__":
    show_settings_page()