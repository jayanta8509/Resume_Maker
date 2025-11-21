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


async def analyze_achievements(input_question):
    """
    Analyzes and extracts achievement information from multiple data sources with impact assessment.
    
    Args:
        input_question (str): Combined achievement data from multiple sources including:
                             - Resume achievements data
                             - LinkedIn achievements data  
                             - Portfolio achievements information
                             - Other link achievement data
    
    Returns:
        tuple: (Achievements_data object with impactful achievement descriptions, total_tokens_used)
    """

    prompt_template = """ You are an expert achievement analyst and professional recognition specialist. You will receive achievement data from multiple sources including Resume, LinkedIn, Portfolio, and Other Links. Your task is to analyze, cross-reference, and create comprehensive, impactful achievement entries that showcase professional excellence and recognition.

        **Data Processing Instructions:**
        - Analyze all provided sources to get a complete picture of professional achievements and recognitions
        - Cross-reference achievement information across sources to ensure accuracy and completeness
        - Merge related achievement data from different sources
        - Validate achievement authenticity by checking consistency across sources
        - Prioritize achievements that demonstrate significant professional impact and recognition
        - Focus on achievements that add measurable value to the candidate's professional profile

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

        4. **Description**: 
           - **CRITICAL**: Write in excellent, professional English with proper grammar and sentence structure
           - Create compelling, impactful descriptions that highlight the significance and value of the achievement
           - Combine insights from all sources to provide comprehensive context
           - Explain the achievement's significance in the professional and industry context
           - Include the criteria, competition level, or selection process when relevant
           - Highlight the achievement's impact on career, organization, or industry
           - Mention quantifiable metrics, recognition scope, or competitive aspects when available
           - Write 2-4 well-crafted sentences that demonstrate the achievement's professional value
           - Focus on what the achievement represents in terms of expertise, leadership, or innovation
           - Example quality: "Recognized as one of the top 30 innovators under 30 in the technology sector by Forbes magazine. Selected from over 5,000 nominees based on exceptional contributions to artificial intelligence and machine learning applications in healthcare. This recognition highlights pioneering work in developing AI-driven diagnostic tools that improved patient outcomes by 35%."

        **Achievement Validation Guidelines:**
        - Verify achievement names and details against official sources when possible
        - Cross-check dates for accuracy and chronological consistency
        - Validate organization credibility and industry recognition
        - Ensure achievements are relevant to the candidate's professional field
        - Prioritize prestigious, competitive, or industry-recognized achievements
        - Include professional awards, academic honors, industry recognitions, and leadership acknowledgments
        - Focus on achievements that demonstrate excellence, innovation, or significant contribution

        **Quality Standards:**
        - Use professional, impactful language that highlights excellence and recognition
        - Ensure consistency in formatting and presentation across all achievements
        - Include specific details that demonstrate the achievement's significance and competitiveness
        - Maintain accuracy while presenting achievements in the most favorable professional light
        - Focus on achievements that add substantial value to the candidate's profile
        - Order achievements by significance, recency, and professional relevance

        **Professional Impact Assessment:**
        - Prioritize achievements from recognized industry leaders, prestigious organizations, or competitive programs
        - Include achievements that demonstrate thought leadership, innovation, or exceptional performance
        - Focus on recognitions that validate expertise and professional standing
        - Consider the achievement's recognition value in the target industry or role
        - Include both technical achievements and leadership/management recognitions as appropriate
        - Highlight achievements that differentiate the candidate from peers

        **Types of Achievements to Include:**
        - Industry awards and recognitions
        - Academic honors and scholarships
        - Professional certifications with distinction
        - Speaking engagements and thought leadership recognition
        - Publication acknowledgments and research awards
        - Leadership awards and team recognition
        - Innovation awards and patent recognitions
        - Community service and volunteer leadership awards
        - Competition winners and contest achievements
        - Performance-based recognitions and company awards

        """

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
    model="gpt-5.1",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": input_question}
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