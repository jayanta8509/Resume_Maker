from Scraper.resume_scraper import get_resume_content
from chat_section.vectordata import FAISSVectorDB
from chat_section.question_generation import QuestionGenerator
import asyncio


def collect_resume_andlinkdin_data(user_id , resume_path , linkedin_file_path):
    try:
        resume_profile_data = get_resume_content(resume_path)
        linkedin_profile_data = get_resume_content(linkedin_file_path) if linkedin_file_path else None
        db = FAISSVectorDB(db_path=f"./{user_id}_faiss_db")

        store = db.store_user_data(resume_profile_data, linkedin_profile_data, user_id)
        return 200          
    except Exception as e:
        print(f"Error in collect_resume_data: {e}")
        raise

def collect_resume_andlinkdin_data_text(user_id , resume_path):
    try:
        db = FAISSVectorDB(db_path=f"./{user_id}_faiss_db")

        store = db.store_user_data(resume_path, user_id)
        return 200          
    except Exception as e:
        print(f"Error in collect_resume_data: {e}")
        raise

def generate_questions(user_id):
    try:
        vector_db = FAISSVectorDB(db_path=f"./{user_id}_faiss_db")
        generator = QuestionGenerator(vector_db)
        all_questions = generator.generate_questions_for_experience(user_id)
        return all_questions
    except Exception as e:
        print(f"Error in generate_questions: {e}")
        raise