import streamlit as st
import json
import os
from datetime import datetime
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

HISTORY_FILE = config["app"]["history_file_name"]

def load_history():
    """Загружает историю из файла"""
    if not os.path.exists(HISTORY_FILE):
        return []
    
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def show_history_page():
    st.title("📚 История запросов")
    
    # Загружаем историю
    history = load_history()
    
    if not history:
        st.info("История запросов пуста. Отправьте несколько запросов для сохранения.")
        return
    
    # Настройки отображения
    st.subheader("Настройки отображения")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_items = st.number_input(
            "Количество последних запросов",
            min_value=1,
            max_value=len(history),
            value=min(5, len(history)),
            step=1
        )
    
    with col2:
        fields = st.multiselect(
            "Поля для отображения и экспорта",
            options=["timestamp", "system_prompt", "prompt", "result"],
            default=["timestamp", "prompt", "result"]
        )
    
    with col3:
        sort_order = st.selectbox(
            "Сортировка",
            options=["Новые сверху", "Старые сверху"],
            index=0
        )
    
    # Применяем настройки
    displayed_history = history[-max_items:] if sort_order == "Новые сверху" else history[:max_items]
    if sort_order == "Старые сверху":
        displayed_history = list(reversed(displayed_history))
    
    # Фильтруем поля для отображения и экспорта
    filtered_history = []
    for entry in displayed_history:
        filtered_entry = {field: entry.get(field, "") for field in fields}
        filtered_history.append(filtered_entry)
    
    # Отображаем историю
    st.subheader(f"Последние {len(displayed_history)} запросов")
    
    for i, entry in enumerate(displayed_history):
        with st.expander(f"Запрос {i+1}: {entry.get('timestamp', 'Без времени')[:19]}"):
            
            if "timestamp" in fields:
                st.markdown(f"**Время:** {entry.get('timestamp', '—')}")
            
            if "system_prompt" in fields:
                st.markdown("**Системный промпт:**")
                st.text_area(
                    "Системный промпт", 
                    value=entry.get('system_prompt', ''), 
                    height=80, 
                    disabled=True, 
                    key=f"sys_{i}",
                    label_visibility="hidden"  # Скрывает label, но сохраняет его для доступности
                )
            
            if "prompt" in fields:
                st.markdown("**Запрос:**")
                st.text_area(
                    "Запрос",
                    value=entry.get('prompt', ''), 
                    height=100, 
                    disabled=True, 
                    key=f"prompt_{i}",
                    label_visibility="hidden"
                )
            
            if "result" in fields:
                st.markdown("**Результат:**")
                st.text_area(
                    "Результат",
                    value=entry.get('result', ''), 
                    height=150, 
                    disabled=True, 
                    key=f"result_{i}",
                    label_visibility="hidden"
                )
    
    # Кнопка экспорта (соответствует выбранным параметрам)
    st.divider()
    if filtered_history:
        export_json = json.dumps(filtered_history, ensure_ascii=False, indent=2)
        st.download_button(
            label="📤 Скачать выбранную историю (JSON)",
            data=export_json,
            file_name=f"linguaflow_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    else:
        st.info("Нет данных для экспорта с текущими настройками.")

# Вызов функции (если нужно протестировать отдельно)
if __name__ == "__main__":
    show_history_page()