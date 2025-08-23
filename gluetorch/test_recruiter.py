from recruiter_pipeline import create_recruiter_pipeline

def test_recruiter_pipeline():
    print("Testing Recruiter Pipeline...")
    
    # Создаем пайплайн
    pipeline = create_recruiter_pipeline()
    
    # Тестовые данные
    test_cases = [
        {
            'resume': "Python developer with 5 years of experience. Skills: Python, ML, Docker. Education: Master's degree.",
            'jd': "Looking for Python developer with ML experience and Docker knowledge. Minimum 3 years of experience. Bachelor's degree required."
        },
        {
            'resume': "Java developer with 2 years of experience. Skills: Java, Spring. Education: Bachelor's degree.",
            'jd': "Senior Python developer needed with 5+ years of experience. Master's degree preferred."
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        result = pipeline(
            resume_text=test_case['resume'],
            jd_text=test_case['jd']
        )
        
        print(f"Overall Score: {result['overall_score'].item():.3f}")
        print(f"Semantic Similarity: {result['semantic_similarity'].item():.3f}")
        print(f"Candidate Skills: {result['candidate_skills']}")
        print(f"Required Skills: {result['required_skills']}")

if __name__ == "__main__":
    test_recruiter_pipeline()