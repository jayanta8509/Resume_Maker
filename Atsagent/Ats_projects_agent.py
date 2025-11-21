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


async def analyze_projects(input_question, jd_input):
    """
    Analyzes and extracts project information from multiple data sources,
    optimized for ATS matching based on job description technical requirements.
    
    Args:
        input_question (str): Combined project data from multiple sources including:
                             - Resume projects data
                             - LinkedIn projects data  
                             - GitHub repository analysis and summaries
        jd_input (dict): Job description data containing:
                        - hard_skills: List of technical skills
                        - tools_and_technologies: List of required tools/technologies
                        - preferred_qualifications: List of preferred qualifications
    
    Returns:
        tuple: (Projects_data object with ATS-optimized project descriptions, total_tokens_used)
    """

    prompt_template = """ You are an expert ATS optimization specialist and technical project analyst. You will receive project data from multiple sources and job description technical requirements. Your task is to analyze, cross-reference, and create ATS-optimized project entries that maximize alignment with target job technical skills, tools, and qualifications.

        **Data Processing Instructions:**
        - Analyze all provided sources to get a complete picture of each project
        - Cross-reference project information across sources to ensure accuracy and completeness
        - Merge related project data from different sources (e.g., GitHub repositories with resume projects)
        - Extract technical details, achievements, and impact from all available sources
        - Validate project authenticity by checking consistency across sources
        - Prioritize projects that demonstrate significant technical skills and accomplishments
        - **ATS Optimization**: Strategically align project presentation with job description requirements

        **Projects Extraction Requirements:**
        For each project, extract and structure:

        1. **Project name**: 
           - Use the most descriptive and professional project name found across all sources
           - Cross-reference between resume, LinkedIn, and GitHub for accuracy
           - If GitHub repository name differs, use the more professional/descriptive version
           - Ensure name reflects the project's purpose and scope

        2. **ATS-Optimized Description**: 
           - **CRITICAL**: Write in excellent, professional English with proper grammar and sentence structure
           - **Technical Skills Integration**: Strategically incorporate JD hard skills demonstrated in the project
           - **Tools & Technology Alignment**: Highlight JD tools and technologies used in the project
           - **Qualification Demonstration**: Show how project demonstrates preferred qualifications from JD
           - **Keyword Optimization**: Naturally weave in JD technical keywords while maintaining authenticity
           - Create compelling, detailed descriptions that showcase both technical expertise and JD alignment
           - Combine insights from all sources while optimizing for JD technical requirements
           - Use action-oriented language that highlights problem-solving using JD-relevant skills
           - Include project objectives, challenges overcome using JD technologies, and results achieved
           - Write 3-5 well-crafted sentences that tell a complete story while maximizing ATS compatibility
           - Focus on technical accomplishments using JD hard skills and business value delivered
           - Include specific metrics, performance improvements, or user impact achieved with JD technologies
           - **ATS Strategy**: Ensure descriptions include JD technical terms and demonstrate relevant qualifications
           - Example approach: "Developed [project type] using [JD tools/technologies] that [quantified achievement]. Implemented [JD hard skills] to [solve problem/deliver value] and [JD preferred qualification demonstration]. Applied [JD technical expertise] resulting in [measurable outcome]."

        3. **ATS-Optimized Technologies**: 
           - **Primary Focus**: Prioritize technologies from JD hard skills and tools_and_technologies lists
           - Extract and list all relevant technologies used in the project that align with JD requirements
           - **JD Alignment**: Emphasize technologies from JD requirements when used in the project
           - **Exact Matching**: Use exact terminology from JD when the project used those technologies
           - Combine technologies mentioned across all sources while prioritizing JD-relevant ones
           - Use industry-standard terminology and proper capitalization matching JD format
           - Include programming languages, frameworks, databases, cloud services, and tools that match JD needs
           - **ATS Keywords**: Ensure critical JD technologies appear when genuinely used in projects
           - Validate technologies against GitHub repository analysis and prioritize JD matches
           - Order technologies by JD relevance first, then by project importance
           - Group related technologies logically while highlighting JD-matching technologies first

        4. **Your role**: 
           - Describe the specific role and responsibilities in the project
           - Use professional titles when appropriate (e.g., "Lead Developer", "Full-Stack Developer", "Project Lead")
           - If role is not explicitly stated, infer from project scope and technical complexity
           - For solo projects, use terms like "Developer", "Creator", or "Lead Developer"
           - For team projects, specify collaborative aspects and leadership responsibilities
           - **JD Alignment**: Emphasize role aspects that demonstrate JD preferred qualifications

        5. **Duration**: 
           - Maintain exact format as it appears in the primary source (resume/LinkedIn)
           - Ensure consistency across similar formats (e.g., "Jan 2020 - Mar 2022", "2019-Present")
           - Use the most accurate dates available across all sources
           - For ongoing projects, use "Present" or "Ongoing"
           - If duration is not specified, infer reasonable timeframes based on project complexity

        **ATS Optimization Guidelines:**
        - **Technical Skills Matching**: Highlight projects that demonstrate JD hard skills
        - **Technology Alignment**: Emphasize JD tools and technologies used in projects
        - **Qualification Evidence**: Show how projects demonstrate JD preferred qualifications
        - **Keyword Integration**: Naturally incorporate JD technical terminology
        - **Skills Validation**: Connect project outcomes to JD technical requirements
        - **Professional Relevance**: Focus on projects most relevant to target role requirements
        - **Innovation Showcase**: Present projects that exceed basic JD technical expectations

        **Writing Quality Standards:**
        - Use professional, engaging language that highlights both technical expertise and JD alignment
        - Strategically vary sentence structure while incorporating JD technical keywords
        - Include specific technical details that match JD requirements when available
        - Ensure descriptions are ATS-optimized yet detailed and authentic
        - Maintain consistency in tone while maximizing technical keyword alignment
        - Focus on unique aspects and achievements that demonstrate JD-relevant expertise
        - Demonstrate problem-solving skills using JD technical competencies

        **Project Prioritization Strategy:**
        - **Tier 1**: Projects demonstrating multiple JD hard skills and using JD tools/technologies
        - **Tier 2**: Projects showing JD preferred qualifications and relevant technical expertise
        - **Tier 3**: Projects with significant technical skills that support overall JD alignment
        - **Tier 4**: Additional projects that demonstrate career progression and learning
        - Validate project claims against available evidence while emphasizing JD-relevant aspects
        - Focus on projects with clear business impact achieved through JD-relevant technologies

        """

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
    model="gpt-5.1",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": f"Project Information: {input_question}\n\nJob Description Technical Requirements for ATS Optimization:\nHard Skills: {jd_input.get('hard_skills', [])}\nTools & Technologies: {jd_input.get('tools_and_technologies', [])}\nPreferred Qualifications: {jd_input.get('preferred_qualifications', [])}"}
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