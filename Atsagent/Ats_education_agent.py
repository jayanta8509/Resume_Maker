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
    GPAorGrade: float
    AdditionalInformation: str


class Step(BaseModel):
    Education: list[EducationItem]



class Education_data(BaseModel):
    steps: list[Step]

async def analyze_education(input_question, jd_input):
    """
    Analyzes and extracts education information from multiple data sources,
    optimized for ATS matching based on job description educational requirements.
    
    Args:
        input_question (str): Combined education data from multiple sources including:
                             - Resume education data
                             - LinkedIn education data  
                             - Portfolio education data
                             - Other link education data
        jd_input (dict): Job description data containing:
                        - required_qualifications: List of educational requirements
    
    Returns:
        tuple: (Education_data object with ATS-optimized education info, total_tokens_used)
    """

    prompt_template = """ You are an expert ATS optimization specialist and academic credential analyst. You will receive education data from multiple sources and job description educational requirements. Your task is to analyze, cross-reference, and create ATS-optimized education entries that maximize alignment with target job qualifications.

        **Data Processing Instructions:**
        - Analyze all provided sources to get a complete picture of each educational experience
        - Cross-reference information across sources to ensure accuracy and completeness
        - If data conflicts between sources, prioritize the most detailed and official information
        - Merge related educational information from different sources
        - Fill gaps using information from any available source
        - Ensure academic credentials are accurately represented
        - **ATS Optimization**: Strategically align education presentation with job description requirements

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
           - **ATS Alignment**: Emphasize degree aspects that match JD required qualifications

        4. **Graduation year**: 
           - Maintain exact format as it appears in the primary source
           - Use most accurate dates available
           - For ongoing education, use "Expected [Year]" or "In Progress" as appropriate

        5. **GPA or grade**: 
           - Include if mentioned in any source
           - Maintain original format and scale (e.g., "3.8/4.0", "First Class Honours")
           - Only include if explicitly stated, do not estimate

        6. **ATS-Optimized Additional Information**: 
           - **CRITICAL**: Write in excellent, professional English with proper grammar
           - **JD Alignment Strategy**: Strategically highlight educational elements that match job requirements
           - **Qualification Matching**: Emphasize coursework, projects, and achievements that align with JD required qualifications
           - **Keyword Integration**: Naturally incorporate terminology from JD educational requirements
           - **Relevant Coursework**: Prioritize and highlight courses that relate to JD requirements
           - **Academic Projects**: Emphasize projects that demonstrate skills mentioned in JD qualifications
           - **Certifications and Training**: Highlight any additional certifications that match JD requirements
           - Include honors, awards, dean's list, magna/summa cum laude, etc. when they support JD alignment
           - Mention research work, thesis topics that relate to job field or requirements
           - Include extracurricular activities and leadership roles that demonstrate JD-relevant skills
           - Highlight study abroad programs, exchange programs that add value for the target role
           - **ATS Strategy**: Ensure educational achievements are presented in ATS-friendly language that matches JD terminology
           - Write in a clear, professional manner that maximizes educational qualification alignment
           - Combine insights from all sources while optimizing for JD educational requirements
           - Focus on accomplishments and distinctions that specifically add value for the target position

        **ATS Optimization Guidelines:**
        - **Qualification Matching**: Emphasize how education meets or exceeds JD required qualifications
        - **Keyword Alignment**: Use terminology and language that matches JD educational requirements
        - **Relevance Prioritization**: Highlight educational aspects most relevant to job requirements
        - **Skill Demonstration**: Showcase how education demonstrates skills and knowledge required by the job
        - **Competitive Advantage**: Present education in a way that differentiates candidate within JD requirements
        - **Natural Integration**: Maintain professional academic language while optimizing for ATS systems

        **Writing Quality Standards:**
        - Use professional, academic language that aligns with job description terminology
        - Ensure consistency in formatting while maximizing JD qualification alignment
        - Include specific details that demonstrate how education meets job requirements
        - Maintain accuracy while presenting information in the most ATS-favorable light
        - Focus on educational achievements that directly support candidacy for the target role
        - Ensure all information supports the candidate's fit for JD educational requirements

        """

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": f"Education Information: {input_question}\n\nJob Description Educational Requirements for ATS Optimization:\nRequired Qualifications: {jd_input.get('required_qualifications', [])}"}
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