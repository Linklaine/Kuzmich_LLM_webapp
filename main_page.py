import streamlit as st
import pandas as pd
import requests
from openai import OpenAI
import yaml
import json
import os
from datetime import datetime
from semantic_analyzer import SemanticHeatmap
import numpy as np
import streamlit.components.v1 as components


#TODO: загружать дефолтную модель из конфига
#TODO: Добавить кнопку изменения конфига
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

if "show_heatmap" not in st.session_state:
    st.session_state.show_heatmap = False

LM_STUDIO_HOST = config["lm_studio"]["host"]
LM_STUDIO_PORT = config["lm_studio"]["port"]
LM_STUDIO_KEY = config["lm_studio"]["api_key"]
DEFAULT_MODEL = config["app"]["default_model"]

HISTORY_FILE = config["app"]["history_file_name"]
MAX_HISTORY_ITEMS = config["app"]["history_length"]

default_prompt="Вы — полезный и точный инструмент исправляющий текст. Проведите операции над текстом выше не добавляя ничего лишнего."
EVAL_STRUCTURE={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "score",
                            "schema": {
                        "properties": {
                            "grammar": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100
                            },
                            "coherence": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100
                            },
                            "clarity": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100
                            },
                            "engagement": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100
                            }
                        },
                        "required": [
                            "grammar",
                            "coherence",
                            "clarity",
                            "engagement"
                        ]
                        },
                            }
                        }

                # Формируем промпт для оценки
evaluation_prompt = f"""Вы — эксперт-филолог. Объективно оцените представленный текст по следующим критериям по шкале от 1 до 100:

- grammar: грамматическая корректность текста
- coherence: связность и логичность изложения  
- clarity: понятность и ясность формулировок
- engagement: увлекательность и выразительность текста

Выведите ТОЛЬКО JSON в указанном формате."""
SYSTEM_PROMPTS = config.get("prompts", {
        "Базовая помощь": default_prompt
    })
MAIN_PROMPTS=["Перевод  RU-EN","Перевод EN-RU","Академический стиль","Деловая переписка","Структуризация","Сокращение","Энергичный текст","Исправление"]
ADDITIONAL_PROMPTS = [p for p in SYSTEM_PROMPTS.keys() if p not in MAIN_PROMPTS]



client = OpenAI(base_url=f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/v1", api_key=LM_STUDIO_KEY)



def get_available_models():
    """Получает список доступных моделей из LM Studio API"""
    try:
        response = requests.get(f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/v1/models", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            # Извлекаем имена моделей из ответа
            model_names = [model["id"] for model in models_data.get("data", [])]
            return model_names
        else:
            return ["Ошибка загрузки моделей"]
    except Exception as e:
        st.error(f"Не удалось подключиться к LM Studio: {e}")
        return ["LM Studio не запущена"]
    
def combine_system_prompts(selected_templates: list) -> str:
    """Объединяет несколько системных промптов в один"""
    if not selected_templates:
        return default_prompt
    
    # Берем тексты выбранных промптов
    prompt_parts = [SYSTEM_PROMPTS[template] for template in selected_templates]
    
    # Объединяем их в четкую инструкцию
    combined_prompt = "Вы — точный инструмент по изменению текста, пожалуйста проведите операции над текстом выше учитывая все следующие инструкции не добавляя ничего лишнего:\n\n" + "\n".join(
        f"{i+1}. {part}" for i, part in enumerate(prompt_parts)
    )
    
    return combined_prompt

def save_to_history(system_prompt: str,prompt: str, result: str):
    """Сохраняет запрос и результат в историю (с ограничением количества записей)"""
    # Загружаем существующую историю или создаем новую
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
    else:
        history = []
    
    # Добавляем новую запись
    new_entry = {
        "timestamp": datetime.now().isoformat(),
        "system_prompt":system_prompt,
        "prompt": prompt,
        "result": result
    }
    history.append(new_entry)
    
    # Ограничиваем количество записей (оставляем последние MAX_HISTORY_ITEMS)
    if len(history) > MAX_HISTORY_ITEMS:
        history = history[-MAX_HISTORY_ITEMS:]
    
    # Сохраняем обратно в файл
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def copy_button(text: str, button_label: str = "📋 Копировать"):
    """Создаёт кнопку копирования в буфер обмена"""
    # Экранируем текст для JavaScript
    escaped_text = text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
    
    components.html(f"""
        <script>
        function copyToClipboard() {{
            navigator.clipboard.writeText("{escaped_text}").then(() => {{
                alert("✅ Скопировано!");
            }}).catch((err) => {{
                alert("❌ Ошибка копирования");
            }});
        }}
        </script>
        
        <button onclick="copyToClipboard()" 
                style="background-color: #4CAF50; color: white; padding: 10px 20px; 
                       border: none; border-radius: 5px; cursor: pointer; font-size: 14px;">
            {button_label}
        </button>
    """, height=70)

def evaluate_text(text):
    try:
        # Выполняем структурированный запрос
        response = client.chat.completions.create(
            model=st.session_state.selected_model,
            messages=[{"role": "system", "content": evaluation_prompt},{"role": "user", "content": text}],
            temperature=0.0,  # Низкая температура для консистентности
            response_format=EVAL_STRUCTURE
                                )
        
        # Парсим результат
        evaluation_result = response.choices[0].message.content
        scores = json.loads(evaluation_result)

        # st.markdown(scores)
        
        # Отображаем результаты
        # col1, col2, col3 = st.columns(3)
        # col1.metric("Читаемость", f"{scores['readability']}/100")
        # col2.metric("Качество", f"{scores['quality']}/100")  
        # col3.metric("Креативность", f"{scores['creativity']}/100")
        
        # Сохраняем для дальнейшего использования
        st.session_state.evaluation_scores = scores

        if 'evaluation_scores' in st.session_state:
            scores = st.session_state.evaluation_scores
            
            st.subheader("📊 Оценка качества результата")
            
            # Читаемость
            st.markdown(f"**Грамматика:** {get_score_color(scores['grammar'])} {scores['grammar']}/100")
            st.progress(scores['grammar'] / 100)

            st.markdown(f"**Связность:** {get_score_color(scores['coherence'])} {scores['coherence']}/100")
            st.progress(scores['coherence'] / 100)

            st.markdown(f"**Понятность:** {get_score_color(scores['clarity'])} {scores['clarity']}/100")
            st.progress(scores['clarity'] / 100)

            st.markdown(f"**Увлекательность:** {get_score_color(scores['engagement'])} {scores['engagement']}/100")
            st.progress(scores['engagement'] / 100)
        
    except Exception as e:
        st.error(f"Ошибка при оценке: {str(e)}")

def get_score_color(score):

        if score >= 80:
            return "🟢"  # Зеленый
        elif score >= 60:
            return "🟡"  # Желтый  
        else:
            return "🔴"  # Красный    


st.set_page_config(
    page_title="Кузьмич",
    page_icon=r"app\static\logo.svg",  # Можно использовать emoji или путь к файлу
    layout="wide"
)

#Убирает панель streamlit
# st.markdown("""
#     <style>
#         /* Убираем верхнюю панель с меню */
#         [data-testid="stHeader"] {
#             display: none;
#         }
        
#         /* Убираем отступ сверху */
#         .block-container {
#             padding-top: 1rem;
#         }
#     </style>
# """, unsafe_allow_html=True)

import base64
from pathlib import Path

# Загрузка и кодирование SVG
svg_path = Path(r"app\static\logo.svg")
with open(svg_path, "rb") as f:
    svg_data = base64.b64encode(f.read()).decode()
    svg_base64 = f"data:image/svg+xml;base64,{svg_data}"

# Использование в HTML
st.markdown(f"""
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 15px;">
    <img src="{svg_base64}" alt="Кузьмич" style="width: 48px; height: 48px;">
    <div>
        <h2 style="margin: 0; font-size: 1.6em;">Кузьмич</h2>
        <p style="margin: 3px 0 0 0; color: #666; font-size: 0.9em;">Поддержка письма</p>
    </div>
</div>
""", unsafe_allow_html=True)



# 1. Инициализация состояния сессии (в начале скрипта)
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = SYSTEM_PROMPTS["По умолчанию"]

# 2. Функция-коллбэк для обновления промпта
def update_combined_prompt(prompt_name=None, new_value=None):
    """Обновляет комбинированный промпт при изменении выбора"""

    # Если вызвано из тумблера — обновляем конкретное значение
    if prompt_name is not None and new_value is not None:
        st.session_state[f"toggle_{prompt_name}"] = new_value
        # print(f"🔄 Тумблер: {prompt_name} → {new_value}")
    else:
        # print(f"🔄 Мультиселект: изменение")
        pass

    main_selected = [p for p in MAIN_PROMPTS if st.session_state.get(f"toggle_{p}", False)]
    additional_selected = st.session_state.get("additional_prompts_select", [])
    
    combined = combine_system_prompts(main_selected + additional_selected)
    #print(st.session_state)
    # print(st.session_state)

    st.session_state.system_prompt = combined

# Получаем список моделей (выполняется при каждом запуске/обновлении)
available_models = get_available_models()

# Выбор модели
selected_model = st.selectbox(
    "Выберите модель:",
    options=available_models,
    index=available_models.index(DEFAULT_MODEL) if DEFAULT_MODEL in available_models else 0,
    key="selected_model"
)


# 3. Сворачиваемый блок
with st.expander("⚙️ Системные инструкции"):

    # --- Тумблеры для основных промптов ---
    st.markdown("**Основные инструкции:**")
    
    # Отображение тумблеров в сетке
    cols = st.columns(2)
    for idx, prompt_name in enumerate(MAIN_PROMPTS):
        col = cols[idx % 2]
        with col:
            toggle_key = f"toggle_{prompt_name}"
            # Инициализируем если нет
            if toggle_key not in st.session_state:
                st.session_state[toggle_key] = False
            
            # Получаем ТЕКУЩЕЕ значение (до переключения)
            current_value = st.session_state[toggle_key]
            is_active = st.toggle(
                prompt_name,
                value=current_value,
                key=toggle_key,
                help=SYSTEM_PROMPTS.get(prompt_name, "")[:100] + "...",
                on_change=update_combined_prompt,
                args=(prompt_name, not current_value)
            )
    
    # --- Мультиселект для дополнительных ---
    if ADDITIONAL_PROMPTS:
        st.markdown("**Дополнительные инструкции:**")
        additional_selected = st.multiselect(
            "Выберите из списка:",
            options=ADDITIONAL_PROMPTS,
            default=[],
            key="additional_prompts_select",
            help="Эти инструкции добавятся к основным",
            on_change=update_combined_prompt
        )
    else:
        additional_selected = []
    
    # --- Предпросмотр ---
    # Редактируемое поле (автоматически обновляется)
    system_prompt = st.text_area(
        "Комбинированный промпт:",
        value=st.session_state.get("system_prompt", ""),
        height=150,
        key="system_prompt"
    )
    # print(st.session_state.system_prompt)

user_input = st.text_area(
    "Введите ваш текст",
    height=200,
    key="user_input")

if st.session_state.get("user_input"):
    if st.button("📊 Оценить качество исходного текста"):
        with st.spinner("Оценка качества..."):
            original_text = st.session_state.get("user_input", "")
            evaluate_text(original_text)

if st.button("📤 Отправить на обработку", type="primary"):
    if not user_input.strip():
        st.warning("Пожалуйста, введите текст для обработки.")
    else:
        with st.spinner("Обработка..."):
            # Получаем текущие значения из session_state
            system_prompt = st.session_state.system_prompt
            user_prompt = user_input
            
            # Формируем сообщения для модели
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "system", "content": system_prompt}
            ]
            
            try:
                # Выполняем запрос к модели (ваш существующий код)
                print(f"Отправлен запрос с промптом:\n{messages}")
                stream = client.chat.completions.create(
                    model=st.session_state.selected_model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4000,
                    stream=True
                )
                
                # Потоковый вывод результата
                response_container = st.empty()
                full_response = ""
                
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        response_container.markdown(full_response + "▌")
                
                # Финальный результат без курсора
                response_container.markdown(full_response)
                
                # Сохраняем результат в session_state (опционально)
                st.session_state.last_result = full_response
                st.session_state.processed_text = full_response

                save_to_history(system_prompt,user_input,full_response)
                
            except Exception as e:
                st.error(f"Ошибка при обработке: {str(e)}")
elif "last_result" in st.session_state:
    st.markdown(st.session_state.last_result)

# Использование
if "processed_text" in st.session_state and st.session_state.processed_text: 
    # Кнопка копирования
    copy_button(st.session_state.processed_text, "📋 Копировать результат")

# После отображения результата обработки ПРОВЕРИТЬ
if st.session_state.get("last_result"):
    if st.button("📊 Оценить качество сгенерированного текста"):
        with st.spinner("Оценка качества..."):
            processed_text = st.session_state.last_result
            evaluate_text(processed_text)

st.divider()
st.subheader("🔍 Семантический поиск")

col1, col2 = st.columns([1, 4])

with col1:
    if st.button("🎨 Поиск по смыслу", use_container_width=True):
        st.session_state.show_heatmap = not st.session_state.show_heatmap
        st.rerun()

if st.session_state.show_heatmap:
    # Поле для поискового запроса
    search_query = st.text_area(
        "🔎 Введите запрос для поиска:",
        placeholder="Например: усталость, дела, помощь...",
        height=80,
        key="semantic_query"
    )
    
    # Выбор текста для поиска
    search_target = st.radio(
        "Где искать:",
        options=["Исходный текст", "Обработанный текст", "Оба текста"],
        index=0,
        key="search_target"
    )
    
    if search_query.strip():
        with st.spinner("Анализ..."):
            try:
                analyzer = SemanticHeatmap()
                
                # Определяем, в каком тексте искать
                texts_to_search = []
                if search_target == "Исходный текст" or search_target == "Оба текста":
                    if "user_input" in st.session_state and st.session_state.user_input.strip():
                        texts_to_search.append(("Исходный текст", st.session_state.user_input))
                
                if search_target == "Обработанный текст" or search_target == "Оба текста":
                    if "processed_text" in st.session_state and st.session_state.processed_text.strip():
                        texts_to_search.append(("Обработанный текст", st.session_state.processed_text))
                
                
                if not texts_to_search:
                    st.warning("⚠️ Нет текста для поиска.")
                else:
                    for text_name, text_content in texts_to_search:
                        sentences, similarities, colors = analyzer.search_in_text(search_query, text_content)

                        
                        
                        st.markdown(f"**{text_name}:**")
                        
                        # Сортировка по релевантности (опционально)
                        if st.checkbox(f"Сортировать по релевантности", key=f"sort_{text_name}"):
                            sorted_indices = np.argsort(similarities)[::-1]
                            for idx in sorted_indices:
                                if similarities[idx]:  # Показываем только релевантные
                                    st.markdown(
                                        f"""<div style="background-color: {colors[idx]}20; 
                                                       border-left: 5px solid {colors[idx]}; 
                                                       padding: 12px; 
                                                       margin: 8px 0; 
                                                       border-radius: 4px;">
                                            <small style="color: #666;">Сходство: {similarities[idx]:.2f}</small><br>
                                            {sentences[idx]}
                                            </div>""",
                                        unsafe_allow_html=True
                                    )
                        else:
                            # Показываем в исходном порядке
                            for i, (sentence, color, score) in enumerate(zip(sentences, colors, similarities)):
                                st.markdown(
                                    f"""<div style="background-color: {color}20; 
                                                   border-left: 5px solid {color}; 
                                                   padding: 12px; 
                                                   margin: 8px 0; 
                                                   border-radius: 4px;">
                                        <small style="color: #666;">Сходство: {score:.2f}</small><br>
                                        {sentence}
                                        </div>""",
                                    unsafe_allow_html=True
                                )
                        
                        # Статистика
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Среднее сходство", f"{np.mean(similarities):.2f}")
                        col2.metric("Наилучшее совпадение", f"{np.max(similarities):.2f}")
                        col3.metric("Наименьшее совпадение", f"{np.min(similarities):.2f}")
                        
                        st.divider()
                
            except Exception as e:
                st.error(f"Ошибка анализа: {e}")
                st.exception(e)
    else:
        st.info("👉 Введите поисковый запрос выше для начала анализа.")
    
    # Кнопка скрытия
    if st.button("Скрыть поиск", key="hide_heatmap"):
        st.session_state.show_heatmap = False
        st.rerun()
#full_response="response"

#st.markdown(f"**Результат:**\n\n{st.session_state.system_prompt}")