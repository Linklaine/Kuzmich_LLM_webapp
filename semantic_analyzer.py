# semantic_analyzer.py
from sentence_transformers import SentenceTransformer
import numpy as np
import re

class SemanticHeatmap:
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
    
    def split_into_sentences(self, text):
        """Разбивает текст на предложения"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def search_in_text(self, query: str, text: str):
        """
        Сравнивает поисковый запрос с каждым предложением текста.
        Возвращает предложения, оценки сходства и цвета.
        """
        if not text.strip() or not query.strip():
            return [], [], []
        
        # Разбиваем текст на предложения
        sentences = self.split_into_sentences(text)
        
        # Эмбеддинг запроса
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Эмбеддинги предложений
        sentence_embeddings = self.model.encode(sentences, convert_to_numpy=True)
        sentence_embeddings = sentence_embeddings / np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
        
        # Косинусное сходство
        similarities = np.dot(sentence_embeddings, query_embedding.T).flatten()
        
        if len(similarities) == 0:
            return []
        
        max_score = np.max(similarities)
        
        # Если все значения нулевые или текст пустой
        if max_score == 0:
            return ["#4575b4"] * len(similarities)
        
        colors = []
        for score in similarities:
            # Отрицательные значения (семантическая противоположность)
            if score < 0:
                # Нормализуем от -1 до 0
                relative_neg = abs(score)  # 0 to 1
                if relative_neg >= 0.50:
                    colors.append("#9b59b6")  # Тёмно-фиолетовый (сильная противоположность)
                elif relative_neg >= 0.30:
                    colors.append("#be90d4")  # Фиолетовый
                else:
                    colors.append("#d2b4de")  # Светло-фиолетовый (слабая противоположность)
            # Положительные значения (нормализуем от максимума)
            else:
                if max_score == 0:
                    colors.append("#808080")  # Серый если максимум 0
                else:
                    relative_score = score / max_score
                    
                    if relative_score >= 0.85:
                        colors.append("#d73027")  # Тёмно-красный
                    elif relative_score >= 0.75:
                        colors.append("#fc8d59")  # Оранжевый
                    elif relative_score >= 0.65:
                        colors.append("#fee08b")  # Жёлтый
                    elif relative_score >= 0.50:
                        colors.append("#91bfdb")  # Голубой
                    else:
                        colors.append("#4575b4")  # Тёмно-синий
    
        
        return sentences, similarities, colors