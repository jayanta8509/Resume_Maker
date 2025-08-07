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
openai_api_key = os.getenv("OPENAI_API_KEY")

class Duration(BaseModel):
    StartDate: str
    EndDate: str

class ExperienceItem(BaseModel):
    CompanyName: str
    Position: str
    Duration: Duration
    Location: str
    Description: str
    SkillSet: list[str]

class Step(BaseModel):
    Experience: list[ExperienceItem]
   

class Experience_data(BaseModel):
    steps: list[Step]


async def analyze_experience(input_question):
    """
    Analyzes and extracts professional experience from multiple data sources.
    
    Args:
        input_question (str): Combined experience data from multiple sources including:
                             - Resume experience data
                             - LinkedIn experience data  
                             - GitHub experience/projects data
                             - Portfolio experience data
                             - Other link experience data
    
    Returns:
        tuple: (Experience_data object with well-written descriptions, total_tokens_used)
    """

    prompt_template = """ You are an expert career analyst and professional writer. You will receive experience data from multiple sources including Resume, LinkedIn, GitHub, Portfolio, and Other Links. Your task is to analyze, cross-reference, and create comprehensive, well-written experience entries.

        **Data Processing Instructions:**
        - Analyze all provided sources to get a complete picture of each experience
        - Cross-reference information across sources to ensure accuracy and completeness
        - If data conflicts between sources, prioritize the most detailed and recent information
        - Merge related experiences from different sources (e.g., GitHub projects that relate to work experience)
        - Fill gaps using information from any available source

        **Experience Extraction Requirements:**
        For each company/organization experience, extract and structure:

        1. **Company name**: 
           - Use the most official/complete company name found across all sources
           - Check email domains, LinkedIn URLs, official company names, subsidiaries
           - Cross-reference between resume and LinkedIn for accuracy

        2. **Position/role**: 
           - Use the most detailed and current job title
           - Consider title evolution if multiple sources show progression

        3. **Duration**: 
           - Maintain exact format as it appears in the primary source (resume/LinkedIn)
           - Ensure consistency across similar formats (e.g., "Jan 2020 - Mar 2022", "2019-Present")
           - Use the most accurate dates available

        4. **Company location**: 
           - Extract from resume/LinkedIn, use most complete format
           - Include city, state/country as available

        5. **Detailed description**: 
           - **CRITICAL**: Write in excellent, professional English with proper grammar and sentence structure
           - Create compelling, action-oriented descriptions that showcase achievements and impact
           - Combine insights from all sources (resume, LinkedIn, GitHub projects, portfolio work)
           - Use strong action verbs (developed, implemented, led, optimized, designed, etc.)
           - Include specific technologies, methodologies, and quantifiable results when available
           - Write 3-5 well-crafted sentences that tell a complete story of the role
           - Focus on accomplishments and value delivered, not just duties
           - Ensure each sentence flows naturally and professionally
           - Example quality: "Led the development of a scalable microservices architecture that improved system performance by 40% and reduced deployment time from 2 hours to 15 minutes. Collaborated with cross-functional teams to implement CI/CD pipelines using Docker and Kubernetes, resulting in 99.9% uptime across production environments."

        6. **SkillSet**: 
           - Extract and list all relevant technical and professional skills used in this specific experience
           - Include programming languages, frameworks, tools, technologies, methodologies, and soft skills
           - Combine skills mentioned across all sources (resume, LinkedIn, GitHub, portfolio)
           - Use industry-standard terminology and proper capitalization (e.g., "JavaScript", "React.js", "AWS", "Agile")
           - Include both technical skills (Python, Docker, Kubernetes) and professional skills (Leadership, Project Management)
           - Prioritize skills that are directly relevant to the role and demonstrated through the experience
           - Remove duplicates and ensure each skill is listed only once per experience
           - Order skills by relevance and importance to the role
           - Include 5-15 skills per experience depending on the complexity and scope of the role

        **Writing Quality Standards:**
        - Use professional, engaging language that highlights expertise and impact
        - Vary sentence structure for readability and flow
        - Include specific technical details and business outcomes when available
        - Ensure descriptions are concise yet comprehensive
        - Maintain consistency in tone and style across all experiences

        """

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": input_question}
    ],
    response_format=Experience_data,
    )

    analysis_response = completion.choices[0].message
    total_tokens = completion.usage.total_tokens
    if hasattr(analysis_response, 'refusal') and analysis_response.refusal:
        print(f"Model refused to respond: {analysis_response.refusal}")
        return None, total_tokens
    else:
        parsed_data = Experience_data(steps=analysis_response.parsed.steps)
        return parsed_data, total_tokens