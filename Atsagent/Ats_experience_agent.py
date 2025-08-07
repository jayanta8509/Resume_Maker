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


async def analyze_experience(input_question, jd_input):
    """
    Analyzes and extracts professional experience from multiple data sources,
    optimized for ATS matching based on job description requirements.
    
    Args:
        input_question (str): Combined experience data from multiple sources including:
                             - Resume experience data
                             - LinkedIn experience data  
                             - GitHub experience/projects data
                             - Portfolio experience data
                             - Other link experience data
        jd_input (dict): Job description data containing:
                        - hard_skills: List of technical skills
                        - tools_and_technologies: List of required tools/tech
                        - responsibilities: List of job responsibilities
                        - action_verbs: List of action verbs from JD
    
    Returns:
        tuple: (Experience_data object with ATS-optimized descriptions, total_tokens_used)
    """

    prompt_template = """ You are an expert ATS optimization specialist and career analyst. You will receive experience data from multiple sources and job description requirements. Your task is to analyze, cross-reference, and create ATS-optimized experience entries that maximize keyword matching and alignment with the target job.

        **Data Processing Instructions:**
        - Analyze all provided sources to get a complete picture of each experience
        - Cross-reference information across sources to ensure accuracy and completeness
        - If data conflicts between sources, prioritize the most detailed and recent information
        - Merge related experiences from different sources (e.g., GitHub projects that relate to work experience)
        - Fill gaps using information from any available source
        - **ATS Optimization**: Strategically align experience descriptions with job description requirements

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

        5. **ATS-Optimized Detailed Description**: 
           - **CRITICAL**: Write in excellent, professional English with proper grammar and sentence structure
           - **ATS Strategy**: Strategically incorporate job description keywords, especially hard skills, tools, and action verbs
           - **Action Verb Matching**: Prioritize using action verbs from the JD when describing accomplishments
           - **Technical Alignment**: Highlight technologies and tools from the JD that were used in the experience
           - **Responsibility Matching**: Frame achievements to align with target job responsibilities when relevant
           - **Keyword Density**: Naturally weave in JD keywords while maintaining readability and authenticity
           - Create compelling, action-oriented descriptions that showcase achievements and impact
           - Combine insights from all sources (resume, LinkedIn, GitHub projects, portfolio work)
           - Include specific technologies, methodologies, and quantifiable results that match JD requirements
           - Write 3-5 well-crafted sentences that tell a complete story while maximizing ATS compatibility
           - Focus on accomplishments and value delivered that relate to target job responsibilities
           - Ensure each sentence flows naturally and professionally while being ATS-friendly
           - Example approach: "[JD Action Verb] [JD-relevant technology/process] that [quantified achievement matching JD responsibilities]. [JD Action Verb] with [JD tools/technologies] to [outcome aligning with JD requirements]."

        6. **ATS-Optimized SkillSet**: 
           - **Primary Focus**: Prioritize skills from the JD (hard_skills and tools_and_technologies) that appear in candidate's experience
           - Extract and list all relevant technical and professional skills used in this specific experience
           - **JD Alignment**: Include skills from JD hard_skills and tools_and_technologies lists when demonstrated in the experience
           - **Exact Matching**: Use exact terminology from JD when the candidate has that skill/experience
           - Combine skills mentioned across all sources (resume, LinkedIn, GitHub, portfolio)
           - Use industry-standard terminology and proper capitalization matching JD format
           - Include both technical skills and professional skills that align with JD requirements
           - Prioritize skills that directly match JD requirements and are demonstrated through the experience
           - **ATS Keywords**: Ensure critical JD skills appear when genuinely demonstrated
           - Remove duplicates and ensure each skill is listed only once per experience
           - Order skills by JD relevance first, then by importance to the role
           - Include 5-15 skills per experience, prioritizing JD-matching skills

        **ATS Optimization Guidelines:**
        - **Keyword Integration**: Naturally incorporate JD hard_skills, tools_and_technologies, and action_verbs
        - **Responsibility Alignment**: Frame experience to match target job responsibilities
        - **Technical Matching**: Highlight JD technologies and tools used in the experience
        - **Action Verb Usage**: Use JD action_verbs when describing achievements and tasks
        - **Skills Prioritization**: Emphasize skills that appear in both candidate experience and JD requirements
        - **Natural Flow**: Maintain professional readability while optimizing for ATS keyword matching
        - **Authenticity**: Only include JD elements that can be genuinely supported by candidate's experience

        **Writing Quality Standards:**
        - Use professional, engaging language that highlights expertise and ATS optimization
        - Strategically vary sentence structure while incorporating JD keywords
        - Include specific technical details that match JD requirements when available
        - Ensure descriptions are ATS-optimized yet comprehensive and natural
        - Maintain consistency in tone and style while maximizing keyword alignment

        """

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": f"Experience Information: {input_question}\n\nJob Description Requirements for ATS Optimization:\nHard Skills: {jd_input.get('hard_skills', [])}\nTools & Technologies: {jd_input.get('tools_and_technologies', [])}\nResponsibilities: {jd_input.get('responsibilities', [])}\nAction Verbs: {jd_input.get('action_verbs', [])}"}
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