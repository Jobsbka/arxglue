# recruiter_pipeline.py
from gluetorch import GlueTorch
from recruiter_components import (
    ResumeParserComponent, 
    JDProcessorComponent,
    SkillMatcherComponent,
    ExperienceMatcherComponent,
    EducationMatcherComponent,
    OverallScorerComponent,
    SemanticSimilarityComponent
)

def create_recruiter_pipeline():
    """Создание пайплайна для анализа кандидатов"""
    system = GlueTorch()
    
    # Создаем компоненты
    resume_parser = ResumeParserComponent(name='resume_parser')
    jd_processor = JDProcessorComponent(name='jd_processor')
    skill_matcher = SkillMatcherComponent(name='skill_matcher')
    experience_matcher = ExperienceMatcherComponent(name='experience_matcher')
    education_matcher = EducationMatcherComponent(name='education_matcher')
    overall_scorer = OverallScorerComponent(name='overall_scorer')
    semantic_similarity = SemanticSimilarityComponent(name='semantic_similarity')
    
    # Регистрируем компоненты
    system.register_component(resume_parser)
    system.register_component(jd_processor)
    system.register_component(skill_matcher)
    system.register_component(experience_matcher)
    system.register_component(education_matcher)
    system.register_component(overall_scorer)
    system.register_component(semantic_similarity)
    
    # Создаем соединения
    # Парсинг резюме и вакансии
    system.add_connection(('input', 'resume_text'), ('resume_parser', 'resume_text'))
    system.add_connection(('input', 'jd_text'), ('jd_processor', 'jd_text'))
    
    # Соединяем парсеры с матчерами
    system.add_connection(('resume_parser', 'skills'), ('skill_matcher', 'candidate_skills'))
    system.add_connection(('jd_processor', 'required_skills'), ('skill_matcher', 'required_skills'))
    
    system.add_connection(('resume_parser', 'experience'), ('experience_matcher', 'candidate_experience'))
    system.add_connection(('jd_processor', 'required_experience'), ('experience_matcher', 'required_experience'))
    
    system.add_connection(('resume_parser', 'education'), ('education_matcher', 'candidate_education'))
    system.add_connection(('jd_processor', 'required_education'), ('education_matcher', 'required_education'))
    
    # Соединяем матчеры с общим scorer'ом
    system.add_connection(('skill_matcher', 'skill_match_score'), ('overall_scorer', 'skill_score'))
    system.add_connection(('experience_matcher', 'experience_match_score'), ('overall_scorer', 'experience_score'))
    system.add_connection(('education_matcher', 'education_match_score'), ('overall_scorer', 'education_score'))
    
    # Добавляем семантическое сходство
    system.add_connection(('input', 'resume_text'), ('semantic_similarity', 'text1'))
    system.add_connection(('input', 'jd_text'), ('semantic_similarity', 'text2'))
    
    # Выходы системы
    system.add_connection(('overall_scorer', 'overall_score'), ('output', 'overall_score'))
    system.add_connection(('overall_scorer', 'detailed_scores'), ('output', 'detailed_scores'))
    system.add_connection(('semantic_similarity', 'semantic_similarity'), ('output', 'semantic_similarity'))
    system.add_connection(('resume_parser', 'skills'), ('output', 'candidate_skills'))
    system.add_connection(('jd_processor', 'required_skills'), ('output', 'required_skills'))
    
    # Компилируем систему
    system.compile()
    
    return system

# Пример использования
if __name__ == "__main__":
    # Создаем пайплайн
    pipeline = create_recruiter_pipeline()
    
    # Тестовые данные
    resume_text = """
    John Doe, Python Developer
    Skills: Python, Machine Learning, SQL, Docker
    Experience: 5 years of software development
    Education: Master's degree in Computer Science
    """
    
    jd_text = """
    We are looking for a Senior Python Developer with:
    - Strong Python skills
    - Experience with Machine Learning
    - Knowledge of Docker and Kubernetes
    - At least 3 years of experience
    - Bachelor's degree or higher
    """
    
    # Запускаем пайплайн
    result = pipeline(resume_text=resume_text, jd_text=jd_text)
    
    print("Overall Score:", result['overall_score'].item())
    print("Detailed Scores:", result['detailed_scores'])
    print("Semantic Similarity:", result['semantic_similarity'].item())
    print("Candidate Skills:", result['candidate_skills'])
    print("Required Skills:", result['required_skills'])