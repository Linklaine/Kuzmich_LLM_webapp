import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Настройка стиля и шрифтов
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

# --- КОНФИГУРАЦИЯ ---
# ID текстов по стилям (адаптируйте под ваш датасет)
TEXT_IDS = {
    "literary": list(range(0, 5)),        # 0-4
    "business": list(range(5, 10)) + list(range(11, 16)),  # 5-9, 11-15
    "conversational": [10] + list(range(16, 23))  # 10, 16-22
}

# Ожидаемые диапазоны оценок для каждого стиля (для визуального сравнения)
EXPECTED_RANGES = {
    "literary": {"grammar": (85, 100), "coherence": (85, 100), 
                 "clarity": (80, 100), "engagement": (80, 100)},
    "business": {"grammar": (85, 100), "coherence": (85, 100), 
                 "clarity": (85, 100), "engagement": (50, 75)},
    "conversational": {"grammar": (60, 85), "coherence": (60, 85), 
                       "clarity": (70, 90), "engagement": (60, 85)}
}

CRITERIA = ["grammar", "coherence", "clarity", "engagement"]
CRITERIA_RU = {"grammar": "Грамматика", "coherence": "Связность", 
               "clarity": "Понятность", "engagement": "Увлекательность"}

def load_results(filepath="benchmark_results.json"):
    """Загрузка результатов бенчмарка"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_scores_by_style(results, mode="structured"):
    """Извлечение оценок по стилям текстов"""
    
    # 1. Создаем явное сопоставление: НАЗВАНИЕ ТЕКСТА -> СТИЛЬ
    # Впишите сюда точные названия из вашего файла benchmark_results.json
    TEXT_NAME_TO_STYLE = {
        # Литературные (ID 0-4)
        "Лев Толстой — «Анна Каренина»": "literary",
        "Фёдор Достоевский — «Преступление и наказание»": "literary",
        "Иван Тургенев — «Отцы и дети»": "literary",
        "Александр Пушкин — «Евгений Онегин»": "literary",
        "Антон Чехов — «Человек в футляре»": "literary",
        
        # Деловые (ID 5-9, 11-15)
        "1. О влиянии цифровизации на коммуникацию": "business",
        "2. О роли искусственного интеллекта в лингвистике": "business",
        "3. О проблемах машинного перевода": "business",
        "4. О читаемости текста": "business",
        "5. О генеративных моделях": "business",
        "Деловой / RU / Короткий": "business",
        "Деловой / RU / Средний": "business",
        "Деловой / RU / Длинный": "business",
        "Business / EN / Short": "business",
        "Business / EN / Medium": "business",
        # ... добавьте остальные названия деловых текстов ...
        
        # Разговорные (ID 10, 16-22)
        "Разговорный": "conversational",
        "Разговорный / RU / Короткий": "conversational",
        "Разговорный / RU / Средний": "conversational",
        "Разговорный / RU / Длинный": "conversational",
        "Conversational / EN / Short": "conversational",
        "Conversational / EN / Medium": "conversational",
        "Conversational / EN / Long": "conversational",
        # ... добавьте остальные названия разговорных текстов ...
    }
    
    # Инициализация структуры для хранения оценок
    style_scores = {style: {model: [] for model in results["models"].keys()} 
                    for style in ["literary", "business", "conversational"]}
    
    for model in results["models"].keys():
        for text_name, text_data in results["models"][model][mode].items():
            
            # 2. Определяем стиль по названию через словарь
            style = TEXT_NAME_TO_STYLE.get(text_name)
            
            # Если название не найдено в словаре — пропускаем
            if style is None:
                print(f"⚠️ Предупреждение: Неизвестный текст '{text_name}', пропускаем.")
                continue
            
            if "error" not in text_data:
                for criterion in CRITERIA:
                    score = text_data.get(criterion, 0)
                    style_scores[style][model].append({
                        "criterion": criterion,
                        "score": score,
                        "text_name": text_name  # Сохраняем имя для отладки
                    })
    
    return style_scores

def calculate_style_differentiation(style_scores):
    """Расчет метрики различимости стилей (насколько хорошо модель видит разницу)"""
    differentiation_scores = {}
    
    for model, scores in style_scores["literary"].items():
        # Средние оценки по каждому стилю
        literary_avg = np.mean([s["score"] for s in style_scores["literary"][model]])
        business_avg = np.mean([s["score"] for s in style_scores["business"][model]])
        conversational_avg = np.mean([s["score"] for s in style_scores["conversational"][model]])
        
        # Разница между стилями (чем больше, тем лучше модель различает)
        diff = abs(literary_avg - business_avg) + abs(business_avg - conversational_avg) + abs(literary_avg - conversational_avg)
        differentiation_scores[model] = diff
    
    return differentiation_scores

def visualize_average_scores_by_style(style_scores, output_file="average_scores_by_style.png"):
    """Визуализация средних оценок по каждому стилю для всех критериев"""
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # 4 графика по критериям
    
    styles_ru = {"literary": "Художественный", "business": "Деловой", "conversational": "Разговорный"}
    colors = plt.cm.Set2(np.linspace(0, 1, len(style_scores["literary"].keys())))
    
    for idx, criterion in enumerate(CRITERIA):
        ax = axes[idx]
        
        data_rows = []
        for style in TEXT_IDS.keys():
            for model in style_scores[style].keys():
                scores = [s["score"] for s in style_scores[style][model] if s["criterion"] == criterion]
                if scores:
                    data_rows.append({
                        "Стиль": styles_ru[style],
                        "Модель": model.split("/")[-1],
                        "Оценка": np.mean(scores)
                    })
        
        df = pd.DataFrame(data_rows)
        
        if df.empty:
            ax.text(0.5, 0.5, "Нет данных", transform=ax.transAxes, ha='center')
            continue
        
        # Группировка данных
        pivot_data = df.pivot_table(index="Стиль", columns="Модель", values="Оценка")
        
        # Построение столбчатой диаграммы
        pivot_data.plot(kind='bar', ax=ax, color=colors, width=0.8, legend=False)
        
        # Добавление ожидаемого диапазона для этого критерия
        for style_name, style_key in zip(["Художественный", "Деловой", "Разговорный"], 
                                          ["literary", "business", "conversational"]):
            if style_name in df["Стиль"].values:
                expected = EXPECTED_RANGES[style_key][criterion]
                ax.axhspan(expected[0], expected[1], alpha=0.15, color='gray')
        
        ax.set_ylim(0, 100)
        ax.set_ylabel("Средняя оценка")
        ax.set_title(CRITERIA_RU[criterion])
        ax.tick_params(axis='x', rotation=0)
        ax.grid(axis='y', alpha=0.3)
    
    # Общая легенда для всех графиков
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Модель", bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.suptitle("Средние оценки по стилям текста (по каждому критерию)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ График сохранен: {output_file}")

def visualize_style_comparison(style_scores, output_file="style_comparison.png"):
    """Визуализация сравнения оценок по стилям"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = {"structured": "#2E86AB", "thinking": "#A23B72"}
    styles_ru = {"literary": "Художественный", "business": "Деловой", "conversational": "Разговорный"}
    
    for idx, criterion in enumerate(CRITERIA):
        ax = axes[idx]
        
        data_rows = []
        for style in TEXT_IDS.keys():
            for model in style_scores[style].keys():
                scores = [s["score"] for s in style_scores[style][model] if s["criterion"] == criterion]
                if scores:
                    data_rows.append({
                        "Стиль": styles_ru[style],
                        "Модель": model.split("/")[-1],  # Короткое имя
                        "Критерий": CRITERIA_RU[criterion],
                        "Оценка": np.mean(scores),
                        "Стд. отклонение": np.std(scores)
                    })
        
        df = pd.DataFrame(data_rows)
        
        if df.empty:
            ax.text(0.5, 0.5, "Нет данных", transform=ax.transAxes, ha='center')
            continue
        
        # Группировка по стилю и модели
        pivot_data = df.pivot_table(index="Стиль", columns="Модель", values="Оценка")
        
        # Построение столбчатой диаграммы
        pivot_data.plot(kind='bar', ax=ax, colormap='Set2', width=0.8)
        
        # Добавление ожидаемого диапазона
        expected = EXPECTED_RANGES[
            "literary" if "Художественный" in df["Стиль"].values else 
            "business" if "Деловой" in df["Стиль"].values else "conversational"
        ]
        ax.axhspan(expected[criterion][0], expected[criterion][1], 
                  alpha=0.2, color='gray', label='Ожидаемый диапазон')
        
        ax.set_ylim(0, 100)
        ax.set_ylabel("Оценка")
        ax.set_title(CRITERIA_RU[criterion])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.tick_params(axis='x', rotation=0)
    
    plt.suptitle("Сравнение оценок моделей по стилям текста", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ График сохранен: {output_file}")

def visualize_differentiation_ranking(differentiation_scores, output_file="differentiation_ranking.png"):
    """Рейтинг моделей по способности различать стили"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Сортировка по убыванию
    sorted_models = sorted(differentiation_scores.items(), key=lambda x: x[1], reverse=True)
    models = [m[0].split("/")[-1] for m in sorted_models]
    scores = [m[1] for m in sorted_models]
    
    # Построение графика
    bars = ax.barh(models, scores, color=sns.color_palette("viridis", len(models)))
    
    # Добавление значений на столбцы
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
               f'{score:.1f}', va='center', fontsize=10)
    
    ax.set_xlabel("Суммарная разница между стилями (чем больше, тем лучше)")
    ax.set_title("Рейтинг моделей по способности различать стили текста", fontsize=12, fontweight='bold')
    ax.set_xlim(0, max(scores) * 1.2)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ График сохранен: {output_file}")

def generate_analysis_report(style_scores, differentiation_scores, output_file="analysis_report.txt"):
    """Генерация текстового отчета с выводами"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ОТЧЕТ ПО АНАЛИЗУ СПОСОБНОСТИ МОДЕЛЕЙ РАЗЛИЧАТЬ СТИЛИ\n")
        f.write("=" * 60 + "\n\n")
        
        # Лучшие модели
        best_model = max(differentiation_scores, key=differentiation_scores.get)
        f.write(f"🏆 Лучшая модель для различения стилей: {best_model}\n")
        f.write(f"   Score: {differentiation_scores[best_model]:.2f}\n\n")
        
        # Средние оценки по стилям
        f.write("Средние оценки по стилям (по всем критериям):\n")
        f.write("-" * 60 + "\n")
        
        for style in ["literary", "business", "conversational"]:
            f.write(f"\n{style.upper()}:\n")
            for model in style_scores[style].keys():
                scores = [s["score"] for s in style_scores[style][model]]
                if scores:
                    avg = np.mean(scores)
                    f.write(f"  {model.split('/')[-1]:<25} {avg:.1f}\n")
        
        # Выводы
        f.write("\n" + "=" * 60 + "\n")
        f.write("ВЫВОДЫ:\n")
        f.write("=" * 60 + "\n")
        f.write("1. Модели с высоким score лучше различают стилистические особенности.\n")
        f.write("2. Сравните ожидаемые диапазоны с фактическими оценками.\n")
        f.write("3. Режим 'thinking' может улучшить различение для сложных моделей.\n")
    
    print(f"✅ Отчет сохранен: {output_file}")

# --- ОСНОВНОЙ ЗАПУСК ---
if __name__ == "__main__":
    print("🔍 Загрузка результатов бенчмарка...")
    results = load_results()
    
    print("📊 Анализ режима: structured")
    style_scores_struct = extract_scores_by_style(results, mode="structured")
    diff_struct = calculate_style_differentiation(style_scores_struct)
    
    print("📊 Анализ режима: thinking")
    style_scores_think = extract_scores_by_style(results, mode="thinking")
    diff_think = calculate_style_differentiation(style_scores_think)
    
    # Визуализация
    print("\n📈 Построение графиков...")
    visualize_style_comparison(style_scores_struct, "style_comparison_structured.png")
    visualize_style_comparison(style_scores_think, "style_comparison_thinking.png")
    # Визуализация средних оценок
    print("\n📈 Построение графика средних оценок...")
    visualize_average_scores_by_style(style_scores_struct, "average_scores_structured.png")
    visualize_average_scores_by_style(style_scores_think, "average_scores_thinking.png")
    
    # Рейтинг моделей
    print("\n🏆 Построение рейтинга моделей...")
    all_diff = {**{f"{k}_struct": v for k, v in diff_struct.items()}, 
                **{f"{k}_think": v for k, v in diff_think.items()}}
    visualize_differentiation_ranking(all_diff, "differentiation_ranking.png")
    
    # Отчет
    print("\n📝 Генерация отчета...")
    generate_analysis_report(style_scores_struct, diff_struct, "analysis_report_structured.txt")
    generate_analysis_report(style_scores_think, diff_think, "analysis_report_thinking.txt")
    
    print("\n✅ Анализ завершен!")