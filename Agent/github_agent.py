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

    
class GitHubAnalysis(BaseModel):
    summary_of_all_repositories: str
    overall_analysis: str
    skills: list[str]

class github_profile_data(BaseModel):
    analysis: GitHubAnalysis


async def analyze_github_profile(profile_data):
    """
    Analyze GitHub profile and provide comprehensive developer assessment
    
    Args:
        profile_data: GitHub profile data (JSON string or dict)
        
    Returns:
        Comprehensive analysis with repository summary, open source contributions, and overall developer assessment
    """
    
    prompt_template = """You are an expert GitHub profile analyst and developer assessor. 
    Analyze the provided GitHub profile data and provide a comprehensive assessment.
    
    Your task is to analyze the GitHub profile and provide exactly the following three components:
    
    1. **Summary of All Repositories**: 
       Provide a comprehensive summary that includes:
       - Overview of the types of projects (web development, data science, algorithms, etc.)
       - Programming languages most frequently used
       - Notable projects or repositories that stand out
       - Patterns in repository creation (consistent coding practice, project diversity, etc.)
       - Assessment of code quality and project complexity
       - Repository count and mix of original vs forked projects
       - Any interesting trends in the developer's coding journey
    
    2. **Overall Analysis**: 
       Provide a comprehensive developer assessment including:
       - Developer skill level and experience assessment
       - Primary areas of expertise and technical strengths
       - Career stage and development trajectory
       - Coding consistency and commitment to programming
       - Unique strengths or standout qualities as a developer
       - Areas of specialization or focus
       - Evidence of collaboration and open source involvement
       - Overall impression of the developer's GitHub presence and capabilities
       - Assessment of contribution quality and community engagement
    
    3. **Skills**: 
       Extract and list the technical skills demonstrated in the profile:
       - Programming languages used across repositories
       - Frameworks and technologies evident from projects
       - Development tools and platforms
       - Technical domains and specializations
       - Any other relevant technical competencies
    
    **Analysis Guidelines**:
    - Be thorough and insightful in your summaries
    - Focus on meaningful patterns and insights rather than just listing facts
    - Assess technical competency based on repository quality, complexity, and diversity
    - Consider both quantity and quality of contributions
    - Look for evidence of continuous learning and skill development
    - Provide honest and balanced assessments
    - Use specific examples from repositories when relevant
    
    **Writing Style**:
    - Write in clear, professional language
    - Be specific and evidence-based in your assessments
    - Provide actionable insights about the developer's capabilities
    - Focus on what makes this developer unique or noteworthy
    - Keep summaries comprehensive but concise
    
    Return your analysis in the structured JSON format with exactly these three fields:
    - summary_of_all_repositories (string)
    - overall_analysis (string)  
    - skills (array of strings)
    """

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": f"Extract structured data from this GitHub profile: {profile_data}"}
        ],
        response_format=github_profile_data,
    )

    analysis_response = completion.choices[0].message
    total_tokens = completion.usage.total_tokens

    if hasattr(analysis_response, 'refusal') and analysis_response.refusal:
        print(f"Model refused to respond: {analysis_response.refusal}")
        return None, total_tokens
    else:
        parsed_data = github_profile_data(analysis=analysis_response.parsed.analysis)
        return parsed_data, total_tokens