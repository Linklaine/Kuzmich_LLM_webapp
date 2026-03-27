import streamlit as st
import yaml
import os

st.set_page_config(page_title="Управление промптами", page_icon="📝", layout="wide")

CONFIG_FILE = "config.yaml"

def load_config():
    if not os.path.exists(CONFIG_FILE):
        return None
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(config):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

def show_prompts_page():
    st.title("📝 Управление системными промптами")
    st.markdown("Здесь вы можете добавлять, редактировать и удалять шаблоны системных промптов.")
    
    config = load_config()
    if config is None:
        st.error(f"Файл `{CONFIG_FILE}` не найден!")
        return
    
    # Инициализируем секцию prompts если нет
    if "prompts" not in config:
        config["prompts"] = {}
    
    st.divider()
    
    # === Список существующих промптов ===
    st.subheader("📋 Существующие промпты")
    
    prompts = config.get("prompts", {})
    
    if not prompts:
        st.info("Нет сохранённых промптов. Добавьте первый!")
    else:
        # Отображение в виде карточек
        for name, text in prompts.items():
            with st.expander(f"📌 {name}"):
                st.text_area(
                    "Текст промпта:",
                    value=text,
                    height=100,
                    key=f"prompt_{name}",
                    label_visibility="collapsed"
                )
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("🗑 Удалить", key=f"delete_{name}"):
                        del config["prompts"][name]
                        save_config(config)
                        st.rerun()
                with col2:
                    if st.button("💾 Сохранить изменения", key=f"save_{name}"):
                        config["prompts"][name] = st.session_state[f"prompt_{name}"]
                        save_config(config)
                        st.success("Сохранено!")
    
    st.divider()
    
    # === Добавление нового промпта ===
    st.subheader("➕ Добавить новый промпт")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        new_name = st.text_input("Название", key="new_prompt_name")
    
    with col2:
        new_text = st.text_area("Текст промпта", height=100, key="new_prompt_text")
    
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        if st.button("Добавить", type="primary", use_container_width=True):
            if new_name.strip() and new_text.strip():
                if new_name in prompts:
                    st.error(f"Промпт с именем '{new_name}' уже существует!")
                else:
                    config["prompts"][new_name] = new_text
                    save_config(config)
                    st.success(f"Промпт '{new_name}' добавлен!")
                    st.rerun()
            else:
                st.error("Заполните название и текст промпта")

if __name__ == "__main__":
    show_prompts_page()