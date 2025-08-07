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

class EducationItem(BaseModel):
    CollegeUniversity: str
    Location: str
    CourseDegree: str
    GraduationYear: str
    GPAorGrade: str
    AdditionalInformation: str


class Step(BaseModel):
    Education: list[EducationItem]



class Education_data(BaseModel):
    steps: list[Step]

async def analyze_education(input_question):
    """
    Analyzes and extracts education information from multiple data sources.
    
    Args:
        input_question (str): Combined education data from multiple sources including:
                             - Resume education data
                             - LinkedIn education data  
                             - Portfolio education data
                             - Other link education data
    
    Returns:
        tuple: (Education_data object with comprehensive education info, total_tokens_used)
    """

    prompt_template = """ You are an expert education analyst and academic credential specialist. You will receive education data from multiple sources including Resume, LinkedIn, Portfolio, and Other Links. Your task is to analyze, cross-reference, and create comprehensive, accurate education entries.

        **Data Processing Instructions:**
        - Analyze all provided sources to get a complete picture of each educational experience
        - Cross-reference information across sources to ensure accuracy and completeness
        - If data conflicts between sources, prioritize the most detailed and official information
        - Merge related educational information from different sources
        - Fill gaps using information from any available source
        - Ensure academic credentials are accurately represented

        **Education Extraction Requirements:**
        For each educational institution, extract and structure:

        1. **College/University name**: 
           - Use the most official and complete institution name found across all sources
           - Cross-reference between resume and LinkedIn for accuracy
           - Include full official name, not abbreviations when possible

        2. **Location of the institution**: 
           - Extract from resume/LinkedIn, use most complete format
           - Include city, state/country as available
           - Use consistent formatting across entries

        3. **Course/degree name**: 
           - Use the most detailed and accurate degree title
           - Include full degree name (e.g., "Bachelor of Science in Computer Science" not just "Computer Science")
           - Specify major, minor, concentration, or specialization when available

        4. **Graduation year**: 
           - Maintain exact format as it appears in the primary source
           - Use most accurate dates available
           - For ongoing education, use "Expected [Year]" or "In Progress" as appropriate

        5. **GPA or grade**: 
           - Include if mentioned in any source
           - Maintain original format and scale (e.g., "3.8/4.0", "First Class Honours")
           - Only include if explicitly stated, do not estimate

        6. **Additional information**: 
           - **CRITICAL**: Write in excellent, professional English with proper grammar
           - Include honors, awards, dean's list, magna/summa cum laude, etc.
           - List relevant coursework that relates to career goals
           - Include academic projects, thesis topics, or research work
           - Mention extracurricular activities, leadership roles, or academic organizations
           - Include study abroad programs, exchange programs, or special certifications
           - Write in a clear, professional manner that highlights academic achievements
           - Combine insights from all sources to create comprehensive additional information
           - Focus on accomplishments and distinctions that add value

        **Writing Quality Standards:**
        - Use professional, academic language appropriate for educational credentials
        - Ensure consistency in formatting and presentation across all entries
        - Include specific details that demonstrate academic excellence and engagement
        - Maintain accuracy while presenting information in the most favorable light
        - Ensure all information is factual and verifiable

        """

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": input_question}
    ],
    response_format=Education_data,
    )

    analysis_response = completion.choices[0].message
    total_tokens = completion.usage.total_tokens
    if hasattr(analysis_response, 'refusal') and analysis_response.refusal:
        print(f"Model refused to respond: {analysis_response.refusal}")
        return None, total_tokens
    else:
        parsed_data = Education_data(steps=analysis_response.parsed.steps)
        return parsed_data, total_tokens