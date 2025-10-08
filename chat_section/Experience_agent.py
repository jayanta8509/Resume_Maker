import os
import re
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directory to path to import shared_client
sys.path.append(str(Path(__file__).parent.parent))
from shared_client import get_async_client

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

class Duration(BaseModel):
    StartDate: str
    EndDate: str

class Projects(BaseModel):
    Project_title: str
    Role: str
    technologies_used: list[str]
    Description: str

class ExperienceItem(BaseModel):
    CompanyName: str
    Position: str
    Duration: Duration
    Location: str
    SkillSet: list[str]
    Projects: list[Projects]

class ExperienceData(BaseModel):
    Experience: list[ExperienceItem]


async def improve_experience_description(
    user_id: str,
    experience_data: dict,
    question_answers: Dict[str, Dict[str, str]],
    vector_db
) -> tuple:
    """
    Improves project descriptions in experience data based on user's answers to questions,
    vector database data, and current experience data.
    
    Args:
        user_id (str): User's unique identifier to retrieve data from vector database
        experience_data (dict): Current experience data in JSON format with structure:
                               {
                                   "Experience": [
                                       {
                                           "CompanyName": "...",
                                           "Position": "...",
                                           "Duration": {"StartDate": "...", "EndDate": "..."},
                                           "Location": "...",
                                           "SkillSet": [...],
                                           "Projects": [
                                               {
                                                   "Project_title": "...",
                                                   "Role": "...",
                                                   "technologies_used": [...],
                                                   "Description": "..."
                                               }
                                           ]
                                       }
                                   ]
                               }
        question_answers (Dict[str, Dict[str, str]]): Questions and answers organized by company:
                                                      {
                                                          "Company Name": {
                                                              "Question 1": "Answer 1",
                                                              "Question 2": "Answer 2",
                                                              ...
                                                          }
                                                      }
        vector_db: Instance of FAISSVectorDB to retrieve user's resume and LinkedIn data
    
    Returns:
        tuple: (ExperienceData object with improved project descriptions, total_tokens_used)
    """
    
    # Retrieve user data from vector database
    print(f"Retrieving data for user_id: {user_id} from vector database...")
    user_data = vector_db.retrieve_user_data(user_id)
    
    if not user_data:
        print(f"Error: User {user_id} not found in vector database")
        return None, 0
    
    resume_data = user_data['resume_data']
    linkedin_data = user_data['linkedin_data']
    
    # Format question-answers for the prompt
    qa_formatted = ""
    for company, qa_dict in question_answers.items():
        qa_formatted += f"\n\n=== {company} - Questions & Answers ===\n"
        for question, answer in qa_dict.items():
            qa_formatted += f"Q: {question}\nA: {answer}\n\n"
    
    # Create comprehensive prompt
    prompt_template = """You are an expert resume writer and career coach specializing in creating compelling, achievement-focused project descriptions. 

Your task is to improve the Description field for EACH PROJECT under EACH company experience by incorporating:
1. Information from the user's resume and LinkedIn profile
2. Detailed insights from the user's answers to targeted questions about their work at that company
3. The existing project description

**CRITICAL INSTRUCTIONS:**

For each company's projects:
1. **Locate the matching company** in the question-answers data
2. **Read all questions and answers** for that company carefully
3. **For each project under that company**, improve its Description by:
   - Integrating relevant information from the company's Q&A answers
   - Preserving all existing accurate information from current description
   - Adding quantifiable metrics mentioned in answers (numbers, percentages, impact metrics, etc.)
   - Highlighting specific achievements and impact mentioned in answers
   - Including technical details, challenges, and solutions from answers
   - Incorporating problem-solving examples relevant to the project
   - Adding business impact, user metrics, or performance improvements from answers

**Project Description Writing Guidelines:**
- Write in excellent, professional English with proper grammar
- Use strong action verbs to start the description (Developed, Engineered, Built, Implemented, Created, Designed)
- Include specific, quantifiable achievements from answers (X% improvement, handled Y users, reduced Z by N%)
- Mention the project's purpose, technologies used, and measurable outcomes
- Highlight impact on business, product, users, or team performance from answers
- Write 2-4 well-crafted sentences that tell a complete project story
- Make each project description unique and showcase specific contributions
- Balance technical details with business/user impact
- Maintain authenticity - only include information supported by the data
- Create descriptions that are compelling, detailed, and achievement-focused

**Important Notes:**
- If no questions/answers exist for a company, improve project descriptions based on resume, LinkedIn, and existing data
- Maintain the EXACT same JSON structure
- ONLY improve the Description field in each Project
- Keep all other fields unchanged (CompanyName, Position, Duration, Location, SkillSet, Project_title, Role, technologies_used)
- Each project description should be substantial and information-rich based on the answers provided
- Focus on concrete achievements, measurable outcomes, and specific technical contributions
- Ensure descriptions showcase the complexity and impact of each project
"""

    # Prepare the input context
    input_context = f"""
**User's Resume Data:**
{resume_data}

**User's LinkedIn Data:**
{linkedin_data}

**Current Experience Data:**
{experience_data}

**Questions & Answers (Company-wise):**
{qa_formatted}
"""

    # Get the async client
    client = await get_async_client()
    
    print("Generating improved experience descriptions...")
    
    completion = await client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": input_context}
        ],
        response_format=ExperienceData,
    )

    analysis_response = completion.choices[0].message
    total_tokens = completion.usage.total_tokens
    
    if hasattr(analysis_response, 'refusal') and analysis_response.refusal:
        print(f"Model refused to respond: {analysis_response.refusal}")
        return None, total_tokens
    else:
        parsed_data = ExperienceData(Experience=analysis_response.parsed.Experience)
        print(f"Successfully improved descriptions. Total tokens used: {total_tokens}")
        return parsed_data, total_tokens


# # Example usage
# async def main():
#     from vectordata import FAISSVectorDB
    
#     # Initialize vector database
#     vector_db = FAISSVectorDB(db_path="./my_faiss_db")
    
#     # User ID (data already in vector database)
#     user_id = "user_12345"
    
#     # Current experience data
#     experience_data = {
#         "Experience": [
#             {
#                 "CompanyName": "Iksen India Pvt Ltd",
#                 "Position": "AI/ML Engineer",
#                 "Duration": {
#                     "StartDate": "Jul 2024",
#                     "EndDate": "Present"
#                 },
#                 "Location": "Kolkata, India",
#                 "SkillSet": [
#                     "FastAPI",
#                     "OpenAI GPT-4O mini",
#                     "OpenCV",
#                     "Fooocus",
#                     "RunPod",
#                     "GPT-40",
#                     "Eleven Labs",
#                     "Stable Diffusion",
#                     "WAN 2.1",
#                     "OpenAI GPT-4o",
#                     "Pydantic"
#                 ],
#                 "Projects": [
#                     {
#                         "Project_title": "AI-powered Question Generation System",
#                         "Role": "",
#                         "technologies_used": [
#                             "OpenAI's GPT-4O mini",
#                             "OpenCV",
#                             "AWS S3"
#                         ],
#                         "Description": "Developed an AI-powered Question Generation System leveraging OpenAI's GPT-4O mini and OpenCV for pattern recognition, with AWS S3 integration for scalable data management."
#                     },
#                     {
#                         "Project_title": "Virtual Try-On AI System",
#                         "Role": "",
#                         "technologies_used": [
#                             "Fooocus",
#                             "FastAPI",
#                             "RunPod"
#                         ],
#                         "Description": "Engineered a Virtual Try-On AI System, by utilizing the Fooocus image generation model, FastAPI framework, and RunPod for scalable model deployment."
#                     },
#                     {
#                         "Project_title": "AI-powered video generation platform",
#                         "Role": "",
#                         "technologies_used": [
#                             "GPT-40",
#                             "Eleven Labs",
#                             "Stable Diffusion",
#                             "WAN 2.1",
#                             "RunPod"
#                         ],
#                         "Description": "Developed an AI-powered video generation platform leveraging GPT-40 for script interpretation, ensuring scalable and efficient performance."
#                     }
#                 ]
#             },
#             {
#                 "CompanyName": "Tech Innovations Corp",
#                 "Position": "Software Developer",
#                 "Duration": {
#                     "StartDate": "Jan 2023",
#                     "EndDate": "Jun 2024"
#                 },
#                 "Location": "Mumbai, India",
#                 "SkillSet": [
#                     "Python",
#                     "React",
#                     "Node.js",
#                     "PostgreSQL",
#                     "Docker"
#                 ],
#                 "Projects": [
#                     {
#                         "Project_title": "E-commerce Platform",
#                         "Role": "Full Stack Developer",
#                         "technologies_used": [
#                             "React",
#                             "Node.js",
#                             "PostgreSQL"
#                         ],
#                         "Description": "Built a full-stack e-commerce platform with real-time inventory management."
#                     }
#                 ]
#             }
#         ]
#     }
    
#     # Question-answers organized by company
#     question_answers = {
#         "Iksen India Pvt Ltd": {
#             "What specific impact did the Question Generation System have?": "The system improved question quality by 45% and reduced manual question creation time by 60%. It processes 10,000+ questions daily with 92% accuracy.",
#             "How did you optimize the Virtual Try-On system?": "Optimized inference time by 40% through model compression and caching strategies. The system now handles 500 concurrent users with 98% uptime.",
#             "What were the key challenges in the video generation platform?": "Integrated 4 different AI models (GPT-40, Eleven Labs, Stable Diffusion, WAN 2.1) into a unified pipeline. Achieved 25% faster video generation while maintaining quality.",
#             "What technologies did you work with?": "Extensively used FastAPI for building scalable APIs, integrated OpenAI models, worked with RunPod for GPU-accelerated deployments, and implemented AWS S3 for storage.",
#             "What was the scale of your projects?": "Projects served 50,000+ users collectively, processing over 1 million API requests per month with 99.5% reliability."
#         },
#         "Tech Innovations Corp": {
#             "What was the impact of the e-commerce platform?": "Platform handled 15,000+ daily transactions with 99.9% uptime. Reduced cart abandonment by 35% through optimized checkout flow.",
#             "What technical challenges did you solve?": "Implemented real-time inventory sync across 5 warehouses, built caching layer reducing database queries by 70%, and optimized React performance for 3x faster page loads."
#         }
#     }
    
#     # Improve descriptions
#     improved_experience, tokens = await improve_experience_description(
#         user_id=user_id,
#         experience_data=experience_data,
#         question_answers=question_answers,
#         vector_db=vector_db
#     )
    
#     if improved_experience:
#         print("\n" + "="*80)
#         print("IMPROVED PROJECT DESCRIPTIONS")
#         print("="*80)
        
#         for exp in improved_experience.Experience:
#             print(f"\n{exp.CompanyName} - {exp.Position}")
#             print(f"Duration: {exp.Duration.StartDate} to {exp.Duration.EndDate}")
#             print(f"Location: {exp.Location}")
#             print(f"\nProjects:")
            
#             for project in exp.Projects:
#                 print(f"\n  Project: {project.Project_title}")
#                 if project.Role:
#                     print(f"  Role: {project.Role}")
#                 print(f"  Technologies: {', '.join(project.technologies_used)}")
#                 print(f"  Improved Description:")
#                 print(f"  {project.Description}")
            
#             print(f"\nSkills: {', '.join(exp.SkillSet)}")
#             print("-" * 80)
        
#         # Print JSON output
#         print("\n" + "="*80)
#         print("JSON OUTPUT")
#         print("="*80)
#         import json
#         json_output = improved_experience.model_dump()
#         print(json.dumps(json_output, indent=2))
        
#         return improved_experience, tokens


# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())