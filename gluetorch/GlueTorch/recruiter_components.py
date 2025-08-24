# recruiter_components.py
import torch
import torch.nn as nn
import re
from typing import Any, Callable, Optional, Dict, List, Tuple, Union, Set, Type
from gluetorch import Component, PortSpec, PortType, LinearComponent
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ResumeParserComponent(Component):
    """Компонент для парсинга резюме и извлечения структурированной информации"""
    def __init__(self, name: str = None):
        super().__init__(name)
        self.input_ports = {
            'resume_text': PortSpec('resume_text', PortType.ANY)  # Изменено на ANY
        }
        self.output_ports = {
            'skills': PortSpec('skills', PortType.SEQUENCE),
            'experience': PortSpec('experience', PortType.SEQUENCE),
            'education': PortSpec('education', PortType.SEQUENCE)
        }
        
    def forward(self, resume_text: Any) -> Dict[str, List[str]]:  # Изменен тип параметра
        # Преобразуем входные данные в строку
        text = str(resume_text)
        
        # Извлечение навыков (простая реализация)
        skills = self._extract_skills(text)
        
        # Извлечение опыта работы
        experience = self._extract_experience(text)
        
        # Извлечение образования
        education = self._extract_education(text)
        
        return {
            'skills': skills,
            'experience': experience,
            'education': education
        }
    
    def _extract_skills(self, text: str) -> List[str]:
        # Простое извлечение навыков по ключевым словам
        skill_keywords = ['python', 'java', 'sql', 'machine learning', 'docker', 'kubernetes']
        found_skills = []
        for skill in skill_keywords:
            if skill in text.lower():
                found_skills.append(skill)
        return found_skills
    
    def _extract_experience(self, text: str) -> List[str]:
        # Простое извлечение опыта работы
        experience_patterns = [
            r'(\d+)\s* years? of experience',
            r'experience.*?(\d+)\s* years?',
        ]
        experiences = []
        for pattern in experience_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            experiences.extend(matches)
        return experiences if experiences else ['3']  # Значение по умолчанию
    
    def _extract_education(self, text: str) -> List[str]:
        # Простое извлечение образования
        education_keywords = ['bachelor', 'master', 'phd', 'degree', 'diploma']
        education_levels = []
        for education in education_keywords:
            if education in text.lower():
                education_levels.append(education)
        return education_levels if education_levels else ['bachelor']  # Значение по умолчанию

class JDProcessorComponent(Component):
    """Компонент для обработки описания вакансии"""
    def __init__(self, name: str = None):
        super().__init__(name)
        self.input_ports = {
            'jd_text': PortSpec('jd_text', PortType.ANY)  # Изменено на ANY
        }
        self.output_ports = {
            'required_skills': PortSpec('required_skills', PortType.SEQUENCE),
            'required_experience': PortSpec('required_experience', PortType.SEQUENCE),
            'required_education': PortSpec('required_education', PortType.SEQUENCE)
        }
        
    def forward(self, jd_text: Any) -> Dict[str, List[str]]:  # Изменен тип параметра
        # Преобразуем входные данные в строку
        text = str(jd_text)
        
        required_skills = self._extract_required_skills(text)
        required_experience = self._extract_required_experience(text)
        required_education = self._extract_required_education(text)
        
        return {
            'required_skills': required_skills,
            'required_experience': required_experience,
            'required_education': required_education
        }
    
    def _extract_required_skills(self, text: str) -> List[str]:
        skill_keywords = ['python', 'java', 'sql', 'machine learning', 'docker', 'kubernetes']
        found_skills = []
        for skill in skill_keywords:
            if skill in text.lower():
                found_skills.append(skill)
        return found_skills
    
    def _extract_required_experience(self, text: str) -> List[str]:
        experience_patterns = [
            r'(\d+)\s* years? of experience',
            r'experience.*?(\d+)\s* years?',
        ]
        experiences = []
        for pattern in experience_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            experiences.extend(matches)
        return experiences if experiences else ['3']  # Значение по умолчанию
    
    def _extract_required_education(self, text: str) -> List[str]:
        education_keywords = ['bachelor', 'master', 'phd', 'degree', 'diploma']
        education_levels = []
        for education in education_keywords:
            if education in text.lower():
                education_levels.append(education)
        return education_levels if education_levels else ['bachelor']  # Значение по умолчанию

class SkillMatcherComponent(Component):
    """Компонент для сравнения навыков кандидата и требований вакансии"""
    def __init__(self, name: str = None):
        super().__init__(name)
        self.input_ports = {
            'candidate_skills': PortSpec('candidate_skills', PortType.SEQUENCE),
            'required_skills': PortSpec('required_skills', PortType.SEQUENCE)
        }
        self.output_ports = {
            'skill_match_score': PortSpec('skill_match_score', PortType.TENSOR, shape=(1,))
        }
        
    def forward(self, candidate_skills: List[str], required_skills: List[str]) -> Dict[str, torch.Tensor]:
        # Вычисление соответствия навыков
        if not required_skills:
            return {'skill_match_score': torch.tensor([1.0])}
        
        matched_skills = set(candidate_skills) & set(required_skills)
        match_score = len(matched_skills) / len(required_skills)
        
        return {'skill_match_score': torch.tensor([match_score])}

class ExperienceMatcherComponent(Component):
    """Компонент для сравнения опыта кандидата и требований вакансии"""
    def __init__(self, name: str = None):
        super().__init__(name)
        self.input_ports = {
            'candidate_experience': PortSpec('candidate_experience', PortType.SEQUENCE),
            'required_experience': PortSpec('required_experience', PortType.SEQUENCE)
        }
        self.output_ports = {
            'experience_match_score': PortSpec('experience_match_score', PortType.TENSOR, shape=(1,))
        }
        
    def forward(self, candidate_experience: List[str], required_experience: List[str]) -> Dict[str, torch.Tensor]:
        # Вычисление соответствия опыта
        if not required_experience:
            return {'experience_match_score': torch.tensor([1.0])}
        
        candidate_exp = int(candidate_experience[0]) if candidate_experience else 0
        required_exp = int(required_experience[0]) if required_experience else 0
        
        if candidate_exp >= required_exp:
            return {'experience_match_score': torch.tensor([1.0])}
        else:
            return {'experience_match_score': torch.tensor([candidate_exp / required_exp])}

class EducationMatcherComponent(Component):
    """Компонент для сравнения образования кандидата и требований вакансии"""
    def __init__(self, name: str = None):
        super().__init__(name)
        self.input_ports = {
            'candidate_education': PortSpec('candidate_education', PortType.SEQUENCE),
            'required_education': PortSpec('required_education', PortType.SEQUENCE)
        }
        self.output_ports = {
            'education_match_score': PortSpec('education_match_score', PortType.TENSOR, shape=(1,))
        }
        
    def forward(self, candidate_education: List[str], required_education: List[str]) -> Dict[str, torch.Tensor]:
        # Вычисление соответствия образования
        if not required_education:
            return {'education_match_score': torch.tensor([1.0])}
        
        education_levels = {'bachelor': 1, 'master': 2, 'phd': 3, 'degree': 1, 'diploma': 1}
        
        candidate_edu_level = max([education_levels.get(edu.lower(), 0) for edu in candidate_education]) if candidate_education else 0
        required_edu_level = max([education_levels.get(edu.lower(), 0) for edu in required_education]) if required_education else 0
        
        if candidate_edu_level >= required_edu_level:
            return {'education_match_score': torch.tensor([1.0])}
        else:
            return {'education_match_score': torch.tensor([candidate_edu_level / required_edu_level])}

class OverallScorerComponent(Component):
    """Компонент для вычисления общего score кандидата"""
    def __init__(self, weights: Optional[Dict[str, float]] = None, name: str = None):
        super().__init__(name)
        self.weights = weights or {'skill': 0.5, 'experience': 0.3, 'education': 0.2}
        
        self.input_ports = {
            'skill_score': PortSpec('skill_score', PortType.TENSOR, shape=(1,)),
            'experience_score': PortSpec('experience_score', PortType.TENSOR, shape=(1,)),
            'education_score': PortSpec('education_score', PortType.TENSOR, shape=(1,))
        }
        self.output_ports = {
            'overall_score': PortSpec('overall_score', PortType.TENSOR, shape=(1,)),
            'detailed_scores': PortSpec('detailed_scores', PortType.DICT)
        }
        
    def forward(self, skill_score: torch.Tensor, experience_score: torch.Tensor, 
                education_score: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Вычисление общего score
        overall = (self.weights['skill'] * skill_score + 
                  self.weights['experience'] * experience_score + 
                  self.weights['education'] * education_score)
        
        return {
            'overall_score': overall,
            'detailed_scores': {
                'skill': skill_score,
                'experience': experience_score,
                'education': education_score
            }
        }

class SemanticSimilarityComponent(Component):
    """Компонент для вычисления семантического сходства на основе эмбеддингов"""
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', name: str = None):
        super().__init__(name)
        self.model = SentenceTransformer(model_name)
        
        self.input_ports = {
            'text1': PortSpec('text1', PortType.ANY),  # Изменено на ANY
            'text2': PortSpec('text2', PortType.ANY)   # Изменено на ANY
        }
        self.output_ports = {
            'semantic_similarity': PortSpec('semantic_similarity', PortType.TENSOR, shape=(1,))
        }
        
    def forward(self, text1: Any, text2: Any) -> Dict[str, torch.Tensor]:  # Изменены типы параметров
        # Преобразуем входные данные в строки
        text1_str = str(text1)
        text2_str = str(text2)
            
        embedding1 = self.model.encode(text1_str, convert_to_tensor=True)
        embedding2 = self.model.encode(text2_str, convert_to_tensor=True)
        
        similarity = cosine_similarity(
            embedding1.cpu().numpy().reshape(1, -1), 
            embedding2.cpu().numpy().reshape(1, -1)
        )[0][0]
        
        return {'semantic_similarity': torch.tensor([similarity])}
