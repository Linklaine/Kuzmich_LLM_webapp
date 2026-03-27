import json
import yaml
import requests
from openai import OpenAI
from datetime import datetime
from pathlib import Path



# --- КОНФИГУРАЦИЯ ---
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

BASE_URL = f"http://{config['lm_studio']['host']}:{config['lm_studio']['port']}/v1"
client = OpenAI(base_url=BASE_URL, api_key=config['lm_studio']['api_key'])

# Модели для теста
MODELS = [
    "qwen/qwen3-32b",
    "openai/gpt-oss-20b", 
    "mistralai/devstral-small-2-2512",
    "qwen/qwen3-30b-a3b"
]

# Тестовые задачи (по 1 примеру на задачу)
TEST_TASKS = [
    {
        "id": "correction_ru",
        "name": "Коррекция текста (RU)",
        "prompt": "Исправь все ошибки в тексте, сохранив смысл, не пиши ничего лишнего:",
        "input": "Вчира я быль в магазе и купил малоко, хлеб, и яица погода была хорошая и я решил прогулятся"
    },
    {
        "id": "correction_en",
        "name": "Коррекция текста (EN)",
        "prompt": "Correct all errors in the text while preserving the meaning, don't write anything else:",
        "input": "Yestarday I go to the store and buyed milk, bread, and eggs. The weather was good so I decided to take a walk."
    },
    {
        "id": "paraphrase_ru",
        "name": "Перефразирование (RU)",
        "prompt": "Перефразируй текст, сохранив смысл но используя другие слова, не пиши ничего лишнего:",
        "input": "Компания планирует запустить новый продукт в следующем квартале. Ожидается высокий спрос на рынке."
    },
    {
        "id": "paraphrase_en",
        "name": "Перефразирование (EN)",
        "prompt": "Paraphrase the text using different words while keeping the same meaning, don't write anything else:",
        "input": "The company plans to launch a new product next quarter. High market demand is expected."
    },
    {
        "id": "translate_ru_en",
        "name": "Перевод RU→EN",
        "prompt": "Переведи текст на английский язык, сохраняя деловой стиль, не пиши ничего лишнего:",
        "input": "Уважаемые коллеги, напоминаем о необходимости сдать отчеты до пятницы."
    },
    {
        "id": "translate_en_ru",
        "name": "Перевод EN→RU",
        "prompt": "Translate the text to Russian, maintaining formal business style, don't write anything else:",
        "input": "Dear colleagues, we remind you that reports must be submitted by Friday."
    },
    {
        "id": "style_change",
        "name": "Смена стиля (разговорный→деловой)",
        "prompt": "Перепиши текст в формальном деловом стиле, не пиши ничего лишнего:",
        "input": "Привет! Короче, нам надо встретиться на следующей неделе и обсудить проект. Напиши когда сможешь."
    },
    {
        "id": "structuring",
        "name": "Структуризация в JSON",
        "prompt": "Извлеки информацию и оформи строго в формате JSON, не пиши ничего лишнего: {\"tasks\": [\"задача1\", \"задача2\"]}",
        "input": "Нужно сделать три вещи: купить продукты, позвонить клиенту и подготовить презентацию к завтрашнему дню."
    },
    {
        "id": "summarization",
        "name": "Сокращение текста (50%)",
        "prompt": "Сократи текст в 2 раза, сохранив ключевые идеи, не пиши ничего лишнего:",
        "input": "Вчера состоялась ежегодная конференция компании, на которой присутствовало более ста сотрудников из различных отделов. Генеральный директор выступил с докладом о результатах работы за прошедший год и поделился планами на будущее. Также были награждены лучшие сотрудники за их вклад в развитие организации."
    },
    {
        "id": "continuation",
        "name": "Генерация продолжения",
        "prompt": "Продолжи текст, сохраняя стиль и тему (2-3 предложения), не пиши ничего лишнего:",
        "input": "Искусственный интеллект активно развивается в последние годы. Новые модели становятся всё более мощными и эффективными."
    },
    {
        "id": "multistep_complex",
        "name": "Многоступенчатая инструкция",
        "prompt": "Выполните последовательно три шага: 1) Переведите текст на английский, 2) Сократите результат на 50%, 3) Оформите оставшуюся информацию в виде нумерованного списка. Верните только финальный результат, не пиши ничего лишнего.",
        "input": "Компания планирует запустить новый продукт в следующем квартале. Для этого необходимо провести маркетинговое исследование, подготовить техническую документацию, обучить персонал и договориться с поставщиками. Ожидается, что продукт займет значительную долю рынка."
    },
    {
        "id": "mixed_language",
        "name": "Смешанный язык (RU/EN)",
        "prompt": "Обработайте текст: исправьте ошибки, сохраняя переключение языков (английские слова должны остатся английскими и т.д.), и сделайте текст более формальным, не пиши ничего лишнего.",
        "input": "Вчира у нась быль meting с инвесторами. Они был impressed нашим результатами. We discused планы на следующий quarter и agred что нужно увеличить бюджет на marketing. Overal, это был productiv день."
    },
    {
        "id": "contradictory_instructions",
        "name": "Противоречивые инструкции",
        "prompt": "Сделайте текст максимально кратким (не более 10 слов), но при этом добавьте конкретные примеры и детали, раскрывающие каждую идею, не пиши ничего лишнего.",
        "input": "Искусственный интеллект развивается очень быстро. Новые модели становятся всё более мощными. Это влияет на различные отрасли."
    },
    {
        "id": "medical_terminology",
        "name": "Специфичная терминология (Медицина)",
        "prompt": "Исправьте ошибки в медицинском тексте, сохраняя профессиональную терминологию, не пиши ничего лишнего:",
        "input": "Пациенту был поставлен диагноз: острый инфаркт миокарда с подъемом сегмента ST. Была проведена экстренная коронароангиопластика со стентированием правой коронарной артерии. В послеоперационном периоде назначена двойная антиагрегантная терапия."
    },
    {
        "id": "legal_terminology",
        "name": "Специфичная терминология (Юриспруденция)",
        "prompt": "Отформатируйте юридический текст, добавьте структуру (пункты, подпункты), сохраняя точность формулировок, не пиши ничего лишнего.:",
        "input": "Стороны договорились что в случае форс-мажорных обстоятельств включая но ограничиваясь стихийные бедствия, военные действия, акты терроризма, стороны освобождаются от ответственности за неисполнение обязательств. Срок уведомления о форс-мажоре составляет 5 рабочих дней."
    },
    {
        "id": "technical_code",
        "name": "Технический текст с кодом",
        "prompt": "Объясните что делает код простыми словами, но сохраните технические детали для разработчиков, не пиши ничего лишнего:",
        "input": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2) # O(2^n) time complexity"
    },
    {
        "id": "contextual_reasoning",
        "name": "Контекстуальное рассуждение",
        "prompt": "Прочитайте контекст и ответьте на вопрос, используя только информацию из текста. Если ответа нет, скажите 'Недостаточно информации':",
        "input": "КОНТЕКСТ: Компания 'ТехноПрогресс' была основана в 2015 году. В 2018 году она выпустила свой первый продукт - платформу для анализа данных. К 2020 году компания привлекла $50 млн инвестиций. В 2022 году был запущен второй продукт - система машинного обучения. \n\nВОПРОС: Сколько продуктов выпустила компания до 2021 года?"
    },
    {
        "id": "cultural_nuances",
        "name": "Культурные особенности и идиомы",
        "prompt": "Переведите текст на английский, адаптируя идиомы и культурные отсылки для носителя языка, не пиши ничего лишнего.:",
        "input": "Этот проект - как медвежья услуга: хотели помочь, а только хуже сделали. Теперь придётся расхлёбывать заваренную кашу."
    }
]

print("\n" + "="*60)
print("🔍 ОТЛАДОЧНАЯ ИНФОРМАЦИЯ")
print("="*60)
print(f"📂 Рабочая директория: {Path.cwd()}")
print(f"📄 Файл скрипта: {__file__}")

# Проверка config.yaml
config_path = Path("config.yaml")
if config_path.exists():
    print(f"✅ config.yaml найден: {config_path.absolute()}")
else:
    print(f"❌ config.yaml НЕ найден в: {config_path.absolute()}")
    print("💡 Создайте файл config.yaml или запустите скрипт из правильной папки!")
    exit(1)

# Проверка доступности библиотек
try:
    import openai
    print(f"✅ openai: версия {openai.__version__}")
except ImportError as e:
    print(f"❌ openai не установлен: {e}")
    exit(1)

print("="*60 + "\n")

def generate_response(model, system_prompt, user_input):
    """Генерирует ответ от модели"""
    print(f"      [DEBUG] Запрос к модели: {model}")
    
    try:
        # Для GPT-OSS используем режим thinking
        extra_params = {}
        if "gpt-oss" in model.lower():
            print("      [DEBUG] Активирован режим thinking для GPT-OSS")
            extra_params["extra_body"] = {"thinking": True}
        
        print(f"      [DEBUG] Отправка запроса к {BASE_URL}...")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.3,
            **extra_params
        )
        
        output = response.choices[0].message.content
        print(f"      [DEBUG] Ответ получен (длина: {len(output)} симв.)")
        return output
    
    except Exception as e:
        print(f"      [DEBUG] ❌ ОШИБКА: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"❌ ОШИБКА: {str(e)}"

def run_tests():
    """Запускает тестирование всех моделей"""
    print("\n" + "="*60)
    print("🚀 ЗАПУСК БЫСТРОГО ТЕСТИРОВАНИЯ МОДЕЛЕЙ")
    print("="*60)
    print(f"⏰ Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📡 URL API: {BASE_URL}")
    print(f"🤖 Моделей: {len(MODELS)}")
    print(f"📝 Задач: {len(TEST_TASKS)}")
    print("="*60)
    
    # Проверка подключения к API
    print("\n🔍 Проверка подключения к LM Studio...")
    try:
        test_response = requests.get(f"{BASE_URL}/models", timeout=5)
        if test_response.status_code == 200:
            print("✅ Подключение успешно!")
        else:
            print(f"⚠️ Странный ответ сервера: {test_response.status_code}")
    except Exception as e:
        print(f"❌ ОШИБКА ПОДКЛЮЧЕНИЯ: {e}")
        print("💡 Убедитесь, что LM Studio запущен и Local Server включён!")
        return  # Прерываем выполнение
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "models": {},
        "tasks": TEST_TASKS
    }
    
    total_tasks = len(MODELS) * len(TEST_TASKS)
    current_task_num = 0
    
    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"🧪 ТЕСТИРОВАНИЕ МОДЕЛИ: {model}")
        print(f"{'='*60}")
        results["models"][model] = {}
        
        for task in TEST_TASKS:
            current_task_num += 1
            print(f"\n[{current_task_num}/{total_tasks}] Задача: {task['name']}")
            print(f"      [DEBUG] ID задачи: {task['id']}")
            
            try:
                output = generate_response(model, task["prompt"], task["input"])
                
                results["models"][model][task["id"]] = {
                    "input": task["input"],
                    "output": output,
                    "evaluation": ""
                }
                
                if "❌ ОШИБКА" in output:
                    print(f"      [РЕЗУЛЬТАТ] ❌ Ошибка генерации")
                else:
                    print(f"      [РЕЗУЛЬТАТ] ✅ Успешно ({len(output)} симв.)")
                
            except Exception as e:
                print(f"      [РЕЗУЛЬТАТ] ❌ Критическая ошибка: {e}")
                results["models"][model][task["id"]] = {
                    "input": task["input"],
                    "output": f"❌ ОШИБКА: {str(e)}",
                    "evaluation": ""
                }
    
    # Сохранение результатов
    print(f"\n{'='*60}")
    print("💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print(f"{'='*60}")
    
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Директория: {output_dir.absolute()}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # JSON с данными
    json_file = output_dir / f"quick_test_{timestamp}.json"
    try:
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✅ JSON сохранён: {json_file.name}")
    except Exception as e:
        print(f"❌ Ошибка сохранения JSON: {e}")
    
    # HTML отчёт
    html_file = output_dir / f"quick_test_{timestamp}.html"
    try:
        generate_html_report(results, html_file)
        print(f"✅ HTML отчёт сохранён: {html_file.name}")
    except Exception as e:
        print(f"❌ Ошибка генерации HTML: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
    print(f"{'='*60}")
    print(f"📁 Откройте файл в браузере:")
    print(f"   {html_file.absolute()}")
    print(f"{'='*60}\n")

def generate_html_report(results, output_file):
    """Генерирует HTML отчёт с формой для оценки"""
    
    # Экранируем timestamp для JavaScript
    js_timestamp = results['timestamp']
    
    html = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Отчёт по тестированию моделей</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1400px; margin: 0 auto; }}
            h1 {{ color: #333; }}
            .model-section {{ background: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .task {{ border: 1px solid #ddd; margin: 15px 0; padding: 15px; border-radius: 5px; }}
            .task-header {{ font-weight: bold; color: #2c3e50; margin-bottom: 10px; }}
            .input-box {{ background: #f8f9fa; padding: 10px; border-left: 3px solid #3498db; margin: 10px 0; }}
            .output-box {{ background: #e8f6f3; padding: 10px; border-left: 3px solid #27ae60; margin: 10px 0; white-space: pre-wrap; }}
            .evaluation {{ margin-top: 15px; }}
            .eval-btn {{ 
                padding: 8px 16px; 
                margin-right: 5px; 
                border: none; 
                border-radius: 4px; 
                cursor: pointer;
                font-size: 14px;
            }}
            .btn-success {{ background: #27ae60; color: white; }}
            .btn-warning {{ background: #f39c12; color: white; }}
            .btn-danger {{ background: #e74c3c; color: white; }}
            .btn-selected {{ opacity: 0.5; }}
            .summary {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-top: 30px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background: #34495e; }}
            .label {{ font-size: 12px; color: #7f8c8d; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Отчёт по тестированию языковых моделей</h1>
            <p><strong>Дата:</strong> {results['timestamp'][:19]}</p>
            <p><strong>Моделей:</strong> {len(results['models'])} | <strong>Задач:</strong> {len(results['tasks'])}</p>
            
            <div class="summary">
                <h2>📋 Инструкция по оценке</h2>
                <p>Для каждой задачи выберите оценку:</p>
                <ul>
                    <li>✅ <strong>Успешно</strong> — задача выполнена полностью, результат пригоден</li>
                    <li>⚠️ <strong>Частично</strong> — есть замечания, требуется правка</li>
                    <li>❌ <strong>Не успешно</strong> — задача не выполнена</li>
                </ul>
                <p>После оценки всех задач нажмите <strong>"Сохранить оценки"</strong> внизу страницы.</p>
            </div>
    """
    
    # Таблица для сводки
    html += """
            <div class="summary">
                <h2>📈 Сводная таблица</h2>
                <table id="summaryTable">
                    <tr>
                        <th>Задача</th>
    """
    for model in results['models'].keys():
        model_short = model.split('/')[-1][:20]
        html += f"<th>{model_short}</th>"
    html += """
                    </tr>
    """
    for task in results['tasks']:
        html += f"<tr><td style='text-align:left;'>{task['name']}</td>"
        for model in results['models'].keys():
            html += f"<td id='cell-{model}-{task['id']}'>-</td>"
        html += "</tr>"
    html += """
                </table>
            </div>
    """
    
    # Детали по моделям
    for model, tasks_data in results['models'].items():
        model_short = model.split('/')[-1]
        html += f"""
            <div class="model-section">
                <h2>🤖 Модель: {model_short}</h2>
        """
        
        for task in results['tasks']:
            task_id = task['id']
            data = tasks_data.get(task_id, {})
            
            html += f"""
                <div class="task" id="task-{model}-{task_id}">
                    <div class="task-header">{task['name']}</div>
                    
                    <div class="input-box">
                        <div class="label">ИСХОДНЫЙ ТЕКСТ:</div>
                        {data.get('input', '')}
                    </div>
                    
                    <div class="output-box">
                        <div class="label">ОТВЕТ МОДЕЛИ:</div>
                        {data.get('output', '')}
                    </div>
                    
                    <div class="evaluation">
                        <div class="label">ОЦЕНКА:</div>
                        <button class="eval-btn btn-success" onclick="setRating('{model}', '{task_id}', '✅')">✅ Успешно</button>
                        <button class="eval-btn btn-warning" onclick="setRating('{model}', '{task_id}', '⚠️')">⚠️ Частично</button>
                        <button class="eval-btn btn-danger" onclick="setRating('{model}', '{task_id}', '❌')">❌ Не успешно</button>
                    </div>
                </div>
            """
        
        html += "</div>"
    
    # Кнопка сохранения (ИСПРАВЛЕНО!)
    html += f"""
            <div class="summary">
                <h2>💾 Сохранение результатов</h2>
                <button class="eval-btn btn-success" style="font-size: 18px; padding: 15px 30px;" onclick="saveResults()">💾 Сохранить оценки в JSON</button>
                <p id="saveStatus"></p>
            </div>
        </div>
        
        <script>
            const evaluations = {{}};
            const reportTimestamp = "{js_timestamp}";
            
            function setRating(model, taskId, value) {{
                evaluations[model + '-' + taskId] = value;
                
                // Обновляем кнопку
                const taskDiv = document.getElementById('task-' + model + '-' + taskId);
                taskDiv.querySelectorAll('.eval-btn').forEach(btn => {{
                    btn.classList.remove('btn-selected');
                }});
                event.target.classList.add('btn-selected');
                
                // Обновляем сводную таблицу
                document.getElementById('cell-' + model + '-' + taskId).textContent = value;
                
                // Подсветка
                if (value === '✅') document.getElementById('cell-' + model + '-' + taskId).style.background = '#27ae60';
                else if (value === '⚠️') document.getElementById('cell-' + model + '-' + taskId).style.background = '#f39c12';
                else if (value === '❌') document.getElementById('cell-' + model + '-' + taskId).style.background = '#e74c3c';
            }}
            
            function saveResults() {{
                const data = {{
                    timestamp: reportTimestamp,
                    evaluations: evaluations
                }};
                
                const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'evaluations_' + new Date().toISOString().slice(0,19).replace(/:/g,'-') + '.json';
                a.click();
                
                document.getElementById('saveStatus').textContent = '✅ Оценки сохранены!';
            }}
        </script>
    </body>
    </html>
    """
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    run_tests()