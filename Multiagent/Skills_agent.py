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


async def analyze_skills(input_question):
    """
    Analyzes and extracts skills information from multiple data sources with intelligent categorization.
    
    Args:
        input_question (str): Combined skills data from multiple sources including:
                             - Resume skills data
                             - GitHub skills/technologies data  
                             - Portfolio skills information
    
    Returns:
        tuple: (Skills_data object with well-categorized skills, total_tokens_used)
    """

    prompt_template = """ You are an expert technical skills analyst and career development specialist. You will receive skills data from multiple sources including Resume, GitHub repositories, and Portfolio. Your task is to analyze, cross-reference, and create comprehensive, well-categorized skills profiles.

        **Data Processing Instructions:**
        - Analyze all provided sources to get a complete picture of technical and professional skills
        - Cross-reference skills across sources to ensure accuracy and completeness
        - Merge related skills from different sources (e.g., GitHub repository technologies with resume skills)
        - Extract skills demonstrated through projects, work experience, and stated competencies
        - Validate skills authenticity by checking consistency across sources
        - Remove duplicates and consolidate similar skills under standard terminology

        **Skills Extraction and Categorization Requirements:**
        Analyze and categorize skills into appropriate groups. Create relevant categories based on the candidate's profile:

        **Standard Skill Categories:**
        
        1. **Programming Languages**: 
           - Languages actually used in projects or work (Python, JavaScript, Java, C++, etc.)
           - Include evidence from GitHub repositories and project descriptions
           - Use industry-standard names and proper capitalization

        2. **Frameworks & Libraries**: 
           - Web frameworks (React, Angular, Django, Flask, etc.)
           - Mobile frameworks (React Native, Flutter, etc.)
           - Data science libraries (TensorFlow, PyTorch, Pandas, etc.)
           - Cross-reference with GitHub repository usage

        3. **Databases & Data Management**: 
           - Relational databases (MySQL, PostgreSQL, Oracle, etc.)
           - NoSQL databases (MongoDB, Redis, Cassandra, etc.)
           - Data warehousing and big data tools

        4. **Cloud & DevOps**: 
           - Cloud platforms (AWS, Azure, Google Cloud, etc.)
           - DevOps tools (Docker, Kubernetes, Jenkins, etc.)
           - Infrastructure as Code, CI/CD pipelines

        5. **Tools & Technologies**: 
           - Development tools (Git, VS Code, IntelliJ, etc.)
           - Design tools (Figma, Adobe Creative Suite, etc.)
           - Project management tools (Jira, Trello, etc.)

        6. **Soft Skills & Leadership**: 
           - Communication, teamwork, leadership skills
           - Project management and organizational abilities
           - Problem-solving and analytical thinking

        7. **Domain Expertise**: 
           - Industry-specific knowledge (Finance, Healthcare, E-commerce, etc.)
           - Business analysis, data analysis, machine learning
           - Specialized methodologies (Agile, Scrum, etc.)

        **Additional Categories (if applicable):**
        - **Mobile Development** (if significant mobile experience)
        - **Data Science & Analytics** (if data-focused role)
        - **Security & Cybersecurity** (if security experience)
        - **Game Development** (if gaming industry experience)
        - **AI & Machine Learning** (if AI/ML focus)

        **Quality Standards:**
        - Use industry-standard terminology and proper capitalization (e.g., "JavaScript", "React.js", "AWS")
        - Group related skills logically and avoid over-categorization
        - Include only skills with clear evidence from at least one source
        - Prioritize skills by relevance and demonstrated proficiency
        - Ensure each skill appears only once across all categories
        - Order skills within categories by importance and proficiency level
        - Include 3-8 skills per category for optimal readability
        - Focus on skills that add value to the candidate's profile

        **Validation Guidelines:**
        - Cross-validate technical skills with GitHub repository languages and technologies
        - Ensure claimed skills align with project descriptions and work experience
        - Remove vague or unsubstantiated skill claims
        - Prioritize skills with concrete evidence over self-reported skills

        """

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": input_question}
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