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

class Projects(BaseModel):
    ProjectName: str
    Description: str
    Technologies: list[str]
    YourRole: str
    Duration: Duration


class Step(BaseModel):
    Projects: list[Projects]
    


class Projects_data(BaseModel):
    steps: list[Step]


async def analyze_projects(input_question):
    """
    Analyzes and extracts project information from multiple data sources with comprehensive descriptions.
    
    Args:
        input_question (str): Combined project data from multiple sources including:
                             - Resume projects data
                             - LinkedIn projects data  
                             - GitHub repository analysis and summaries
    
    Returns:
        tuple: (Projects_data object with well-written project descriptions, total_tokens_used)
    """

    prompt_template = """ You are an expert project analyst and technical writer. You will receive project data from multiple sources including Resume, LinkedIn, and GitHub repositories. Your task is to analyze, cross-reference, and create comprehensive, well-written project entries that showcase technical expertise and achievements.

        **Data Processing Instructions:**
        - Analyze all provided sources to get a complete picture of each project
        - Cross-reference project information across sources to ensure accuracy and completeness
        - Merge related project data from different sources (e.g., GitHub repositories with resume projects)
        - Extract technical details, achievements, and impact from all available sources
        - Validate project authenticity by checking consistency across sources
        - Prioritize projects that demonstrate significant technical skills and accomplishments

        **Projects Extraction Requirements:**
        For each project, extract and structure:

        1. **Project name**: 
           - Use the most descriptive and professional project name found across all sources
           - Cross-reference between resume, LinkedIn, and GitHub for accuracy
           - If GitHub repository name differs, use the more professional/descriptive version
           - Ensure name reflects the project's purpose and scope

        2. **Description**: 
           - **CRITICAL**: Write in excellent, professional English with proper grammar and sentence structure
           - Create compelling, detailed descriptions that showcase technical expertise and project impact
           - Combine insights from all sources (resume, LinkedIn, GitHub README, repository analysis)
           - Use action-oriented language that highlights problem-solving and achievements
           - Include project objectives, challenges overcome, and results achieved
           - Write 3-5 well-crafted sentences that tell a complete story of the project
           - Focus on technical accomplishments and business value delivered
           - Include specific metrics, performance improvements, or user impact when available
           - Example quality: "Developed a full-stack e-commerce platform using React and Node.js that processes over 1,000 daily transactions. Implemented secure payment integration with Stripe API and optimized database queries, reducing page load times by 40%. Designed responsive UI components and deployed the application on AWS with CI/CD pipeline automation."

        3. **Technologies**: 
           - Extract and list all relevant technologies used in the project
           - Combine technologies mentioned across all sources (resume, LinkedIn, GitHub analysis)
           - Use industry-standard terminology and proper capitalization (e.g., "JavaScript", "React.js", "MongoDB")
           - Include programming languages, frameworks, databases, cloud services, and tools
           - Prioritize core technologies and avoid listing every minor library
           - Validate technologies against GitHub repository analysis when available
           - Group related technologies logically (e.g., frontend, backend, database, deployment)

        4. **Your role**: 
           - Describe the specific role and responsibilities in the project
           - Use professional titles when appropriate (e.g., "Lead Developer", "Full-Stack Developer", "Project Lead")
           - If role is not explicitly stated, infer from project scope and technical complexity
           - For solo projects, use terms like "Developer", "Creator", or "Lead Developer"
           - For team projects, specify collaborative aspects and leadership responsibilities

        5. **Duration**: 
           - Maintain exact format as it appears in the primary source (resume/LinkedIn)
           - Ensure consistency across similar formats (e.g., "Jan 2020 - Mar 2022", "2019-Present")
           - Use the most accurate dates available across all sources
           - For ongoing projects, use "Present" or "Ongoing"
           - If duration is not specified, infer reasonable timeframes based on project complexity

        **Writing Quality Standards:**
        - Use professional, engaging language that highlights technical expertise and innovation
        - Vary sentence structure for readability and flow
        - Include specific technical details and quantifiable results when available
        - Ensure descriptions are detailed yet concise
        - Maintain consistency in tone and style across all projects
        - Focus on unique aspects and achievements of each project
        - Demonstrate problem-solving skills and technical depth

        **Project Prioritization:**
        - Prioritize projects that demonstrate significant technical skills
        - Include projects that show career progression and learning
        - Focus on projects with clear business impact or innovation
        - Ensure a good mix of different types of projects (web, mobile, data, etc.)
        - Validate project claims against available evidence (GitHub commits, deployment links, etc.)

        """

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": input_question}
    ],
    response_format=Projects_data,
    )

    analysis_response = completion.choices[0].message
    total_tokens = completion.usage.total_tokens
    if hasattr(analysis_response, 'refusal') and analysis_response.refusal:
        print(f"Model refused to respond: {analysis_response.refusal}")
        return None, total_tokens
    else:
        parsed_data = Projects_data(steps=analysis_response.parsed.steps)
        return parsed_data, total_tokens