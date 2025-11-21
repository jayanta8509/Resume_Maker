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

class Achievements(BaseModel):
    Achievement_Titlee: str
    Issuing_Organization: str
    Date_Received: str
    Description : str

class Step(BaseModel):

    Achievements: list[Achievements]



class Achievements_data(BaseModel):
    steps: list[Step]


async def analyze_achievements(input_question, jd_input):
    """
    Analyzes and extracts achievement information from multiple data sources,
    optimized for ATS matching based on job description soft skills and action verbs.
    
    Args:
        input_question (str): Combined achievement data from multiple sources including:
                             - Resume achievements data
                             - LinkedIn achievements data  
                             - Portfolio achievements information
                             - Other link achievement data
        jd_input (dict): Job description data containing:
                        - soft_skills: List of soft skills and interpersonal abilities
                        - action_verbs: List of action verbs from JD
    
    Returns:
        tuple: (Achievements_data object with ATS-optimized achievement descriptions, total_tokens_used)
    """

    prompt_template = """ You are an expert ATS optimization specialist and achievement analyst. You will receive achievement data from multiple sources and job description requirements focusing on soft skills and action verbs. Your task is to analyze, cross-reference, and create ATS-optimized achievement entries that maximize alignment with target job soft skills and leadership qualities.

        **Data Processing Instructions:**
        - Analyze all provided sources to get a complete picture of professional achievements and recognitions
        - Cross-reference achievement information across sources to ensure accuracy and completeness
        - Merge related achievement data from different sources
        - Validate achievement authenticity by checking consistency across sources
        - Prioritize achievements that demonstrate significant professional impact and recognition
        - Focus on achievements that add measurable value to the candidate's professional profile
        - **ATS Optimization**: Strategically align achievement descriptions with job description soft skills and action verbs

        **Achievements Extraction Requirements:**
        For each achievement/award/recognition, extract and structure:

        1. **Achievement title**: 
           - Use the full, official achievement/award name as it appears from the issuing organization
           - Cross-reference between resume, LinkedIn, and other sources for accuracy
           - Include proper capitalization and official terminology
           - Ensure title reflects the exact nature and scope of the achievement
           - Use professional, descriptive titles that convey significance

        2. **Issuing organization**: 
           - Use the official, complete organization name (e.g., "Google", "IEEE", "Forbes", "Harvard Business Review")
           - Cross-reference across sources to ensure accuracy and credibility
           - Use standardized organization names rather than abbreviations when possible
           - Verify organization credibility and recognition in the industry
           - Include department or division if it adds context and credibility

        3. **Date received**: 
           - Maintain exact format as it appears in the primary source
           - Use the most accurate and complete date available across all sources
           - Include month and year when available (e.g., "March 2023", "2023")
           - For ongoing recognitions or recurring achievements, indicate appropriately
           - Ensure chronological accuracy and consistency

        4. **ATS-Optimized Description**: 
           - **CRITICAL**: Write in excellent, professional English with proper grammar and sentence structure
           - **Soft Skills Integration**: Strategically incorporate JD soft skills that the achievement demonstrates
           - **Action Verb Alignment**: Use JD action verbs when describing how the achievement was earned or its impact
           - **Leadership Qualities**: Highlight soft skills like leadership, communication, teamwork that led to the achievement
           - **Interpersonal Abilities**: Emphasize collaborative, communication, and relationship-building aspects of achievements
           - **Professional Qualities**: Showcase soft skills like problem-solving, adaptability, creativity demonstrated through achievements
           - Create compelling, impactful descriptions that highlight both achievement significance and relevant soft skills
           - Combine insights from all sources while optimizing for JD soft skills and action verbs
           - Explain the achievement's significance while demonstrating JD-relevant interpersonal qualities
           - Include the criteria, competition level, or selection process while highlighting relevant soft skills
           - Highlight the achievement's impact using JD action verbs and soft skill terminology
           - Mention quantifiable metrics while emphasizing leadership, communication, or collaborative achievements
           - Write 2-4 well-crafted sentences that demonstrate both achievement value and JD-aligned soft skills
           - Focus on what the achievement represents in terms of leadership, communication, teamwork, and other JD soft skills
           - **ATS Strategy**: Ensure descriptions naturally include JD soft skills and action verbs while maintaining authenticity
           - Example approach: "[JD Action Verb] [achievement context] that required [JD soft skills] to [quantified result]. This recognition demonstrates exceptional [JD soft skills] and [JD action verbs] in [professional context]."

        **Achievement Validation Guidelines:**
        - Verify achievement names and details against official sources when possible
        - Cross-check dates for accuracy and chronological consistency
        - Validate organization credibility and industry recognition
        - Ensure achievements are relevant to the candidate's professional field and demonstrate JD soft skills
        - Prioritize prestigious, competitive, or industry-recognized achievements that showcase interpersonal abilities
        - Include professional awards, academic honors, industry recognitions, and leadership acknowledgments
        - Focus on achievements that demonstrate excellence, innovation, or significant contribution through soft skills

        **ATS Optimization Guidelines:**
        - **Soft Skills Emphasis**: Highlight achievements that demonstrate JD soft skills (leadership, communication, teamwork, etc.)
        - **Action Verb Integration**: Use JD action verbs when describing achievement processes and outcomes
        - **Interpersonal Focus**: Emphasize collaborative, communication, and relationship aspects of achievements
        - **Leadership Demonstration**: Showcase achievements that required leadership and management soft skills
        - **Problem-Solving Showcase**: Highlight achievements that demonstrate analytical and creative thinking
        - **Adaptability Evidence**: Present achievements that show flexibility and learning agility
        - **Communication Excellence**: Emphasize achievements involving presentation, writing, or public speaking

        **Quality Standards:**
        - Use professional, impactful language that highlights both excellence and JD-aligned soft skills
        - Ensure consistency in formatting while maximizing soft skills integration
        - Include specific details that demonstrate achievement significance and relevant interpersonal abilities
        - Maintain accuracy while presenting achievements in the most ATS-favorable light for soft skills
        - Focus on achievements that add substantial value while showcasing JD-relevant qualities
        - Order achievements by significance, recency, and soft skills relevance

        **Professional Impact Assessment:**
        - Prioritize achievements that demonstrate thought leadership, innovation, and exceptional interpersonal performance
        - Focus on recognitions that validate both expertise and soft skills like leadership and communication
        - Consider the achievement's value in demonstrating JD-required interpersonal and leadership qualities
        - Include both technical achievements and leadership/management recognitions that show soft skills
        - Highlight achievements that differentiate the candidate through demonstrated soft skills and action-oriented leadership

        **Types of Achievements to Prioritize:**
        - Leadership awards and team recognition (showcasing management and collaboration skills)
        - Communication and presentation awards (demonstrating communication excellence)
        - Mentorship and coaching recognitions (showing leadership and interpersonal skills)
        - Cross-functional project awards (highlighting teamwork and collaboration)
        - Innovation awards that required creative problem-solving and teamwork
        - Community service and volunteer leadership (demonstrating empathy and social responsibility)
        - Public speaking and thought leadership recognition (communication and influence skills)
        - Team-building and organizational culture awards
        - Customer service and relationship management recognitions
        - Conflict resolution and negotiation achievements

        """

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
    model="gpt-5.1",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": f"Achievement Information: {input_question}\n\nJob Description Requirements for ATS Optimization:\nSoft Skills: {jd_input.get('soft_skills', [])}\nAction Verbs: {jd_input.get('action_verbs', [])}"}
    ],
    response_format=Achievements_data,
    )

    analysis_response = completion.choices[0].message
    total_tokens = completion.usage.total_tokens
    if hasattr(analysis_response, 'refusal') and analysis_response.refusal:
        print(f"Model refused to respond: {analysis_response.refusal}")
        return None, total_tokens
    else:
        parsed_data = Achievements_data(steps=analysis_response.parsed.steps)
        return parsed_data, total_tokens