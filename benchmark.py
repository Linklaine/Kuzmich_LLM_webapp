from openai import OpenAI
import yaml
import json
import yaml
import requests
import re
from openai import OpenAI
from datetime import datetime

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

LM_STUDIO_HOST = config["lm_studio"]["host"]
LM_STUDIO_PORT = config["lm_studio"]["port"]
LM_STUDIO_KEY = config["lm_studio"]["api_key"]
DEFAULT_MODEL = config["app"]["default_model"]

HISTORY_FILE = config["app"]["history_file_name"]
MAX_HISTORY_ITEMS = config["app"]["history_lemgth"]

client = OpenAI(base_url=f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/v1", api_key=LM_STUDIO_KEY)


def schema_response(model:str,prompt:str,text:str):
    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": prompt},{"role": "user", "content": text}],
                        temperature=0.0,  # Низкая температура для консистентности
                        response_format={
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
                                        )
    return parse_json_from_text(response.choices[0].message.content)

def thinking_response(model:str,prompt:str,text:str):
    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": prompt},{"role": "user", "content": text}],
                        temperature=0.0,  # Низкая температура для консистентности
                        reasoning_effort="on",
                        extra_body={"thinking":True}
                        )
    return parse_json_from_text(response.choices[0].message.content)

import json
import yaml
from openai import OpenAI
from datetime import datetime

# Загрузка конфигурации
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

LM_STUDIO_HOST = config["lm_studio"]["host"]
LM_STUDIO_PORT = config["lm_studio"]["port"]
LM_STUDIO_KEY = config["lm_studio"]["api_key"]
# Список моделей для теста
MODELS = [
    "qwen/qwen3-30b-a3b-2507",
    "qwen/qwen3-30b-a3b",
    "openai/gpt-oss-20b"
]

client = OpenAI(
    base_url=f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/v1", 
    api_key=LM_STUDIO_KEY
)

# Тестовые тексты разного характера
TEST_TEXTS = [
    {
        "name": "Лев Толстой — «Анна Каренина»",
        "text": "Весь его [Левина] наружный вид был чрезвычайно нервичен и взволнован. Он сидел за столом, быстро писал, потом вставал, подходил к Дарье Александровне и твердил ей что-то, потом опять садился и продолжал писать. Он просил ее приехать к ним в деревню с детьми, обещал ей все удобства и умолял ее согласиться. Дарья Александровна слушала его с недоумением и не знала, что ему ответить. Она чувствовала, что он был совершенно не в своей тарелке."
    },
    {
        "name": "Фёдор Достоевский — «Преступление и наказание»",
        "text": "Он [Раскольников] боялся встречи с каждым человеком, но еще более боялся оставаться дома. В комнате его было невыносимо душно, и он не мог переносить этого запаха. Он вышел на лестницу, надеясь найти там свежий воздух, но и там было душно. Тогда он решил выйти на улицу, хотя бы для того, чтобы почувствовать себя живым"
    },
    {
        "name": "Иван Тургенев — «Отцы и дети»",
        "text": "Базаров был нигилист. Он не кланялся святыням, не верил авторитетам, не принимал ни одного принципа на веру. Его отец, отставной армейский лекарь, был человек мягкий и добродушный. Мать его, простая дворянка, обожала сына до страсти. Базаров же относился к родителям с добродушным снисхождением."
    },
    {
        "name": "Александр Пушкин — «Евгений Онегин»",
        "text": """Но вот прошла зима; весна
Вступила в права свои;
И сад, недавно так пустынный,
Уже зелен, густ и тенист.
Природа вновь воскресла вся...
Онегин, скучный и унылый,
Выходит в сад; но скука с ним.
Он равнодушен ко всему:
К цветам, к деревьям, к небесам,
К весне, к природе, к жизни самой."""
    },
    {
        "name":"Антон Чехов — «Человек в футляре»",
        "text":"В городе С. жил один учитель греческого языка, Беликов. Он был замечателен тем, что всегда носил очки, пальто с поднятым воротником и держался особняком. Когда он шел по улице, казалось, что он прячется от всего мира. Даже в хорошую погоду он носил калоши и держал зонтик. Он боялся любого изменения и старался окружить себя защитой от реальной жизни."
    },
    {
        "name":"1. О влиянии цифровизации на коммуникацию",
        "text":"""Современное общество переживает трансформацию коммуникативных практик под воздействием цифровых технологий.
Данное явление обусловлено ростом доступности интернет-платформ и мобильных устройств.
Исследования демонстрируют, что изменяется не только форма, но и содержание межличностного взаимодействия.
Особое внимание уделяется снижению качества письменной речи среди молодёжи.
Таким образом, возникает необходимость в разработке инструментов лингвистической поддержки, адаптированных к новым условиям."""
    },
    {
        "name":"2. О роли искусственного интеллекта в лингвистике",
        "text":"""Применение методов искусственного интеллекта в области компьютерной лингвистики открывает новые горизонты для автоматизированной обработки естественного языка.
Большие языковые модели демонстрируют способность к контекстуальному пониманию и генерации связного текста.
Однако их эффективность напрямую зависит от качества обучающих корпусов и архитектурных решений.
Особую сложность представляет двуязычная обработка, требующая баланса между универсальностью и спецификой языков.
В связи с этим актуальным становится вопрос оценки качества генерируемых текстов по объективным метрикам."""
    },
    {
        "name":"3. О проблемах машинного перевода",
        "text":"""Несмотря на значительный прогресс в области машинного перевода, сохраняются системные ограничения, связанные с передачей культурно-специфических реалий.
Синтаксические и семантические различия между языками затрудняют точную передачу исходного смысла.
Современные нейросетевые архитектуры частично компенсируют эти недостатки за счёт контекстного анализа.
Тем не менее, в текстах высокого регистра (академических, юридических) требуется обязательная постредактура.
Следовательно, автоматизированные системы должны рассматриваться как вспомогательный, а не замещающий инструмент."""
    },
    {
        "name":"4. О читаемости текста",
        "text":"""Читаемость представляет собой метрику, отражающую степень когнитивной нагрузки при восприятии текста.
Она определяется совокупностью факторов: лексической сложностью, длиной предложений, плотностью информации.
Существуют формализованные индексы (например, Флеша–Кинкейда), позволяющие количественно оценить данный параметр.
Оптимизация читаемости особенно важна в образовательных и информационных материалах.
Таким образом, управление читаемостью является важным аспектом редакторской деятельности."""
    },
    {
        "name":"5. О генеративных моделях",
        "text":"""Генеративные большие языковые модели функционируют на основе вероятностного прогнозирования последовательностей токенов.
Их обучение осуществляется на масштабных текстовых корпусах, что обеспечивает широкий охват лингвистических паттернов.
Ключевым преимуществом данных систем является способность к few-shot и zero-shot обучению.
Однако они склонны к галлюцинациям и воспроизведению предвзятостей из обучающих данных.
Поэтому критически важно внедрять механизмы контроля фактической достоверности генерируемых ответов."""
    },
    {
        "name":"Разговорный",
        "text":"вчера было довольно много дел и я сильно устал столько всего было что даже не знаю с чего начать надеюсь ты успел доделать свои дела потому что мне придется еще и мне помогать кстати сгоняй в магаз"
    },
    {
        "name": "Деловой / RU / Короткий",
        "text": "Уважаемые партнеры, напоминаем вам о необходимости предоставить финансовую отчетность за четвертый квартал до 25 декабря текущего года. В случае возникновения вопросов просим обращаться в бухгалтерию."
    },
    {
        "name": "Деловой / RU / Средний",
        "text": "Коллеги, добрый день. Напоминаем, что срок подачи заявок на участие в корпоративной программе обучения истекает в эту пятницу. Пожалуйста, убедитесь, что все необходимые документы заполнены корректно и подписаны руководителем подразделения. Заявки, полученные после указанного срока, рассматриваться не будут. Список доступных курсов находится во внутренней системе компании."
    },
    {
        "name": "Деловой / RU / Длинный",
        "text": "Уважаемые руководители подразделений. Настоящим уведомляем вас о проведении планового аудита бизнес-процессов компании, который состоится в период с 15 по 30 января следующего года. В рамках аудита будет проведена оценка эффективности текущих рабочих процедур, выявлены узкие места и разработаны рекомендации по оптимизации. От каждого департамента требуется назначить ответственного сотрудника для взаимодействия с аудиторской группой. Подробный план мероприятий и список необходимых документов будут направлены отдельным письмом. Просим обеспечить доступ к запрашиваемой информации в установленные сроки."
    },
    {
        "name": "Business / EN / Short",
        "text": "Dear Team, please submit your weekly status reports by end of day Friday. Late submissions will affect project tracking and resource allocation."
    },
    {
        "name": "Business / EN / Medium",
        "text": "Dear Colleagues, we are pleased to announce the launch of our new customer feedback system. Starting next month, all client interactions should be logged through the updated portal. Training sessions will be scheduled for each department. Please coordinate with your team lead to select a convenient time slot. Your participation is essential for a smooth transition."
    },
    {
        "name": "Разговорный / RU / Короткий",
        "text": "Привет, ты где сейчас? Мы уже в кафе сидим, заказ сделали. Подходи быстрее, а то остынет всё. И захвати зарядку, у меня телефон садится."
    },
    {
        "name": "Разговорный / RU / Средний",
        "text": "Слушай, вчера такое было, просто ужас. Ехал на работу, попал в жуткую пробку, опоздал на встречу. Начальник потом весь день мозг выносил. Ещё и телефон разрядился в самый неподходящий момент. Короче, день не задался с самого утра. Надеюсь, сегодня будет получше. Ты как, нормально добрался?"
    },
    {
        "name": "Разговорный / RU / Длинный",
        "text": "Привет, как дела? Давно не виделись, надо бы встретиться как-нибудь. У меня куча новостей, даже не знаю с чего начать. Во-первых, я наконец-то переехал в новую квартиру, ремонт ещё не закончил но уже жить можно. Во-вторых, сменил работу, теперь работаю в другой компании, условия намного лучше. В-третьих, записался в спортзал, пытаюсь хоть немного привести себя в форму. Короче, жизнь кипит. А у тебя что нового? Расскажи потом подробно, когда созвонимся. Кстати, в выходные планируем шашлыки за городом, если погода позволит, присоединяйся, будет весело."
    },
    {
        "name": "Conversational / EN / Short",
        "text": "Hey, are you coming tonight? Everyone's already here. Bring some snacks if you can, we're running low. Text me when you're close."
    },
    {
        "name": "Conversational / EN / Medium",
        "text": "Oh man, you won't believe what happened yesterday. I was walking my dog and ran into my old high school teacher. We started talking and lost track of time. She still remembers me, can you believe it? Anyway, we should catch up soon. Are you free this weekend?"
    },
    {
        "name": "Conversational / EN / Long",
        "text": "Hey, long time no talk! How have you been? I've been meaning to call you for weeks but life just got crazy. Work has been insane, my boss keeps piling on new projects. On the bright side, I finally started learning guitar, though I'm terrible at it. My neighbors probably hate me. Also, I'm planning a trip to Europe next month, still figuring out the itinerary. You've been there before, right? Any recommendations? We should definitely grab coffee soon and catch up properly. Let me know when you're free."
    }
]

# Системный промпт для оценки (единый для всех режимов)
EVALUATION_PROMPT = """Вы — эксперт-филолог. Оцените представленный текст по следующим критериям по шкале от 1 до 100:

- grammar: грамматическая корректность текста
- coherence: связность и логичность изложения  
- clarity: понятность и ясность формулировок
- engagement: увлекательность и выразительность текста

Выведите ТОЛЬКО JSON в указанном формате."""

def parse_json_from_text(text):
    """Извлекает JSON из текста ответа модели"""
    try:
        # Попытка найти JSON {...} даже если вокруг есть текст
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(text)
    except Exception as e:
        return {"error": f"JSON Parse Error: {str(e)}"}

def evaluate_structured_mode(model_name, text):
    """Режим 1: Structured Output (JSON Schema) через стандартный API"""
    try:
        # Для openai/gpt-oss-20b стандартный JSON Schema может не работать так же, 
        # но попробуем через стандартный client сначала. Если модель требует спец. формат,
        # она может проигнорировать schema или выдать ошибку.
        # В данном случае мы используем стандартный вызов, так как это универсальный подход LM Studio.
        
        response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "system", "content": EVALUATION_PROMPT},{"role": "user", "content": text}],
                        temperature=0.0,  # Низкая температура для консистентности
                        response_format={
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
                                        )
        content = response.choices[0].message.content
        return parse_json_from_text(content)
    
    except Exception as e:
        return {"error": str(e)}

def evaluate_thinking_mode(model_name, text):
    """Режим 2: Thinking / Reasoning (Нативный API для gpt-oss, эмуляция для других)"""
    
    # Специфичная логика для openai/gpt-oss-20b
    if model_name == "openai/gpt-oss-20b":
        try:
            payload = {
                "model": model_name,
                "input": f"{EVALUATION_PROMPT}\n\nТекст для оценки:\n{text}",
                "reasoning": "high", # Ключевой параметр для этой модели
                "temperature": 0.0
            }
            
            # Прямой запрос к нативному эндпоинту (предположительно /api/v1/chat или аналогичный в вашей версии LM Studio)
            # Внимание: Эндпоинт может отличаться в зависимости от версии LM Studio плагина для этой модели.
            # Стандартный Chat Completion endpoint обычно /v1/chat/completions, но ваш пример показывает /api/v1/chat
            url = f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/api/v1/chat" 
            
            headers = {
                "Authorization": f"Bearer {LM_STUDIO_KEY}",
                "Content-Type": "application/json"
            }
            
            resp = requests.post(url, json=payload, headers=headers, timeout=120)
            
            if resp.status_code == 200:
                data = resp.json()
                # Парсинг структуры ответа согласно вашему примеру: result["output"][1]["content"]
                # Обратите внимание: индексы могут меняться, лучше проверить реальный ответ
                if "output" in data and len(data["output"]) > 1:
                    content = data["output"][1]["content"]
                    return parse_json_from_text(content)
                elif "output" in data and len(data["output"]) > 0:
                    # Попытка найти контент в первом элементе, если структура другая
                    content = data["output"][0].get("content", "")
                    return parse_json_from_text(content)
                else:
                    return {"error": "Неверная структура ответа API"}
            else:
                return {"error": f"HTTP {resp.status_code}: {resp.text}"}
                
        except Exception as e:
            return {"error": f"Request Error: {str(e)}"}

    else:
        # Для Qwen и других моделей пробуем эмулировать "мышление" через системный промпт или параметры
        # Так как у них нет такого же нативного API параметра в вашем примере
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": f"{EVALUATION_PROMPT}\n\nПодумай пошагово перед выставлением оценки."},
                    {"role": "user", "content": text}
                ],
                temperature=0.1,
                # Некоторые версии LM Studio поддерживают extra_body для thinking
                extra_body={"thinking": True} 
            )
            content = response.choices[0].message.content
            return parse_json_from_text(content)
        except Exception as e:
            return {"error": str(e)}

def run_benchmark():
    print(f"🚀 Запуск бенчмарка: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Модели: {MODELS}")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "models": {}
    }

    for model in MODELS:
        print(f"\n🧪 Тестирование модели: {model}")
        results["models"][model] = {"structured": {}, "thinking": {}}
        
        for item in TEST_TEXTS:
            t_name = item["name"]
            t_text = item["text"]
            print(f"   └─ Текст: {t_name}...")
            
            # 1. Structured Mode
            print(f"      ├─ Structured...", end=" ")
            res_struct = evaluate_structured_mode(model, t_text)
            results["models"][model]["structured"][t_name] = res_struct
            print("OK" if "error" not in res_struct else "FAIL")
            
            # 2. Thinking Mode
            print(f"      └─ Thinking...", end=" ")
            res_think = evaluate_thinking_mode(model, t_text)
            results["models"][model]["thinking"][t_name] = res_think
            print("OK" if "error" not in res_think else "FAIL")

    # Сохранение
    out_file = "benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Готово! Результаты сохранены в {out_file}")

if __name__ == "__main__":
    run_benchmark()