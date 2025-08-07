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

class Skills(BaseModel):
    Skill_Category: str
    Skills: list[str]


class Step(BaseModel):
    Skills: list[Skills]


class Skills_data(BaseModel):
    steps: list[Step]


async def analyze_skills(input_question, jd_input):
    """
    Analyzes and extracts skills information from multiple data sources,
    optimized for ATS matching based on job description skill requirements.
    
    Args:
        input_question (str): Combined skills data from multiple sources including:
                             - Resume skills data
                             - GitHub skills/technologies data  
                             - Portfolio skills information
        jd_input (dict): Job description data containing:
                        - hard_skills: List of technical skills
                        - soft_skills: List of soft skills and interpersonal abilities
                        - tools_and_technologies: List of required tools/technologies
    
    Returns:
        tuple: (Skills_data object with ATS-optimized skills categorization, total_tokens_used)
    """

    prompt_template = """ You are an expert ATS optimization specialist and technical skills analyst. You will receive skills data from multiple sources and job description skill requirements. Your task is to analyze, cross-reference, and create ATS-optimized skills profiles that maximize alignment with target job technical and soft skill requirements.

        **Data Processing Instructions:**
        - Analyze all provided sources to get a complete picture of technical and professional skills
        - Cross-reference skills across sources to ensure accuracy and completeness
        - Merge related skills from different sources (e.g., GitHub repository technologies with resume skills)
        - Extract skills demonstrated through projects, work experience, and stated competencies
        - Validate skills authenticity by checking consistency across sources
        - Remove duplicates and consolidate similar skills under standard terminology
        - **ATS Optimization**: Strategically prioritize and categorize skills based on job description requirements

        **Skills Extraction and Categorization Requirements:**
        Analyze and categorize skills into appropriate groups with strategic focus on JD alignment:

        **ATS-Optimized Skill Categories:**
        
        1. **Programming Languages (JD Priority)**: 
           - **Primary Focus**: Prioritize programming languages from JD hard_skills list
           - Languages actually used in projects or work that match JD requirements
           - Include evidence from GitHub repositories and project descriptions
           - Use industry-standard names and proper capitalization matching JD format
           - **JD Alignment**: List JD-required languages first, then additional languages

        2. **Frameworks & Libraries (JD Priority)**: 
           - **Primary Focus**: Emphasize frameworks from JD hard_skills and tools_and_technologies
           - Web frameworks, mobile frameworks, data science libraries that match JD needs
           - Cross-reference with GitHub repository usage and prioritize JD matches
           - **JD Alignment**: Highlight frameworks mentioned in job requirements first

        3. **Tools & Technologies (JD Alignment)**: 
           - **Critical Focus**: Prioritize tools from JD tools_and_technologies list
           - Development tools, cloud platforms, databases, DevOps tools matching JD requirements
           - Include design tools, project management tools that align with job needs
           - **ATS Strategy**: Ensure JD-mentioned tools appear prominently when candidate has them

        4. **Cloud & DevOps (JD Relevant)**: 
           - Cloud platforms (AWS, Azure, Google Cloud) matching JD requirements
           - DevOps tools (Docker, Kubernetes, Jenkins) from JD specifications
           - Infrastructure and deployment technologies aligned with job needs
           - **JD Priority**: Emphasize cloud/DevOps skills mentioned in job requirements

        5. **Databases & Data Management (JD Focused)**: 
           - Relational and NoSQL databases that match JD technical requirements
           - Data warehousing and big data tools aligned with job specifications
           - **JD Alignment**: Prioritize database technologies mentioned in job requirements

        6. **Soft Skills & Leadership (JD Matching)**: 
           - **Strategic Focus**: Prioritize soft skills from JD soft_skills list
           - Communication, teamwork, leadership skills that match job requirements
           - Project management and organizational abilities aligned with JD needs
           - Problem-solving and analytical thinking as specified in job description
           - **ATS Optimization**: Include JD soft skills when genuinely demonstrated

        7. **Domain Expertise (JD Relevant)**: 
           - Industry-specific knowledge that aligns with job domain
           - Business analysis, data analysis, methodologies matching JD requirements
           - Specialized expertise relevant to target role and industry
           - **JD Alignment**: Focus on domain skills that support job requirements

        **Dynamic Categories Based on JD Requirements:**
        Create additional categories if JD emphasizes specific areas:
        - **Mobile Development** (if JD requires mobile skills)
        - **Data Science & Analytics** (if JD focuses on data roles)
        - **Security & Cybersecurity** (if JD emphasizes security)
        - **AI & Machine Learning** (if JD requires AI/ML expertise)
        - **Other JD-Specific Categories** based on unique job requirements

        **ATS Optimization Guidelines:**
        - **JD Skills Priority**: Prioritize skills that directly match JD hard_skills, soft_skills, and tools_and_technologies
        - **Keyword Matching**: Use exact terminology from JD when candidate possesses those skills
        - **Skills Ordering**: Within each category, list JD-matching skills first
        - **Category Relevance**: Emphasize categories most relevant to job requirements
        - **Competitive Advantage**: Highlight skills that exceed basic JD requirements
        - **Professional Validation**: Focus on skills with concrete evidence and JD alignment

        **Quality Standards:**
        - Use industry-standard terminology matching JD format and capitalization
        - Group related skills logically while prioritizing JD-relevant skills
        - Include only skills with clear evidence and emphasize JD matches
        - Prioritize skills by JD relevance first, then by demonstrated proficiency
        - Ensure each skill appears only once across all categories
        - Order skills within categories by JD importance and proficiency level
        - Include 3-8 skills per category, prioritizing JD-matching skills
        - Focus on skills that add maximum value for the target role

        **Validation and Prioritization Guidelines:**
        - **JD Skill Validation**: Cross-validate candidate skills against JD requirements
        - **Technical Verification**: Confirm technical skills with GitHub repository evidence
        - **Soft Skills Evidence**: Ensure soft skills align with demonstrated experience and JD needs
        - **Relevance Filtering**: Remove skills not relevant to job requirements
        - **Strategic Presentation**: Present skills in order of importance to target role

        """

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": f"Skills Information: {input_question}\n\nJob Description Skills Requirements for ATS Optimization:\nHard Skills: {jd_input.get('hard_skills', [])}\nSoft Skills: {jd_input.get('soft_skills', [])}\nTools & Technologies: {jd_input.get('tools_and_technologies', [])}"}
    ],
    response_format=Skills_data,
    )

    analysis_response = completion.choices[0].message
    total_tokens = completion.usage.total_tokens
    if hasattr(analysis_response, 'refusal') and analysis_response.refusal:
        print(f"Model refused to respond: {analysis_response.refusal}")
        return None, total_tokens
    else:
        parsed_data = Skills_data(steps=analysis_response.parsed.steps)
        return parsed_data, total_tokens