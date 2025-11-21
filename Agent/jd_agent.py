import os
import re
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add parent directory to path to import shared_client
sys.path.append(str(Path(__file__).parent.parent))
from shared_client import get_async_client

load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")

    
class JDAnalysis(BaseModel):
    job_title: str
    hard_skills: list[str]
    soft_skills: list[str]
    tools_and_technologies: list[str]
    responsibilities: list[str]
    required_qualifications: list[str]
    preferred_qualifications: list[str]
    action_verbs: list[str]

class jd_data(BaseModel):
    analysis: JDAnalysis


async def analyze_jd(job_description):
    """
    Analyzes job descriptions and extracts comprehensive, structured information for ATS optimization.
    
    Args:
        job_description (str): The complete job description text to analyze
    
    Returns:
        tuple: (jd_data object with comprehensive JD analysis, total_tokens_used)
    """    
    prompt_template = """You are an expert Job Description Analyst AI with deep expertise in talent acquisition and resume optimization. Your task is to comprehensively analyze job descriptions and extract structured, actionable information that can be used for ATS optimization and resume tailoring.

            **Core Extraction Requirements:**
            Extract and structure the following information with high accuracy and completeness:

            **1. job_title**: 
            - Extract the exact official job title as stated in the posting
            - If multiple titles are mentioned, use the primary/main title
            - Maintain proper capitalization and formatting

            **2. hard_skills**: 
            - Technical skills, programming languages, frameworks, methodologies
            - Role-specific competencies and technical expertise requirements
            - Industry-specific knowledge and specialized skills
            - Include variations and synonyms (e.g., "JS", "JavaScript")
            - Focus on skills that would appear on resumes

            **3. soft_skills**: 
            - Interpersonal abilities, behavioral qualities, character traits
            - Communication, leadership, teamwork, adaptability skills
            - Problem-solving, analytical thinking, creativity requirements
            - Work style preferences and personality attributes

            **4. tools_and_technologies**: 
            - Specific software platforms, applications, and tools
            - Cloud services, development environments, databases
            - Hardware, systems, and infrastructure technologies
            - Business tools, project management platforms, design software
            - Include version numbers or specifications when mentioned

            **5. responsibilities**: 
            - Core job duties and day-to-day tasks
            - Key performance expectations and deliverables
            - Project types and work scope descriptions
            - Collaborative activities and team interactions
            - Write as concise, action-oriented phrases

            **6. required_qualifications**: 
            - Educational requirements (degrees, fields of study)
            - Mandatory certifications and licenses
            - Years of experience requirements (specific numbers)
            - Must-have technical competencies
            - Legal or compliance requirements

            **7. preferred_qualifications**: 
            - Nice-to-have educational background
            - Optional certifications and additional training
            - Desired experience levels or domains
            - Bonus skills and competencies
            - Additional languages or cultural requirements

            **8. action_verbs**: 
            - Strong action words used to describe tasks and expectations
            - Leadership and management verbs (lead, manage, oversee)
            - Technical execution verbs (develop, implement, design)
            - Collaboration verbs (coordinate, collaborate, facilitate)
            - Impact and achievement verbs (optimize, improve, deliver)

            **Analysis Guidelines:**
            - **Accuracy First**: Only extract information explicitly stated or clearly implied
            - **Comprehensive Coverage**: Capture all relevant details without redundancy
            - **ATS Optimization**: Focus on keywords and phrases that ATS systems recognize
            - **Resume Relevance**: Prioritize information useful for resume tailoring
            - **Industry Context**: Consider industry-specific terminology and standards
            - **Skill Variations**: Include different ways the same skill might be expressed

            **Quality Standards:**
            - Use industry-standard terminology and proper capitalization
            - Avoid vague or generic terms when specific ones are available
            - Ensure extracted items are actionable and specific
            - Maintain consistency in formatting and style
            - Focus on the most important and relevant information first

            **Output Requirements:**
            - Provide structured data in the exact format specified
            - Ensure all lists contain specific, actionable items
            - Avoid duplicates across different categories
            - Order items by importance and relevance when possible

            Process the job description thoroughly and extract comprehensive, accurate information for optimal ATS resume matching.
    """

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": f"Extract structured data from this job description: {job_description}"}
        ],
        response_format=jd_data,
    )

    analysis_response = completion.choices[0].message
    total_tokens = completion.usage.total_tokens

    if hasattr(analysis_response, 'refusal') and analysis_response.refusal:
        print(f"Model refused to respond: {analysis_response.refusal}")
        return None, total_tokens
    else:
        parsed_data = jd_data(analysis=analysis_response.parsed.analysis)
        return parsed_data, total_tokens