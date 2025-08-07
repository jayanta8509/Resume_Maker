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

class Certifications(BaseModel):
    CertificationName: str
    Issuing_Organization: str
    DateObtained: str
    Certification_ID : str
    Description: str


class Step(BaseModel):
    Certifications: list[Certifications]


class Certifications_data(BaseModel):
    steps: list[Step]


async def analyze_certifications(input_question, jd_input):
    """
    Analyzes and extracts certification information from multiple data sources,
    optimized for ATS matching based on job description qualification requirements.
    
    Args:
        input_question (str): Combined certification data from multiple sources including:
                             - Resume certifications data
                             - LinkedIn certifications data  
                             - Portfolio certifications information
                             - Other link certification data
        jd_input (dict): Job description data containing:
                        - required_qualifications: List of required certifications/qualifications
                        - preferred_qualifications: List of preferred certifications/qualifications
    
    Returns:
        tuple: (Certifications_data object with ATS-optimized certification descriptions, total_tokens_used)
    """

    prompt_template = """ You are an expert ATS optimization specialist and certification analyst. You will receive certification data from multiple sources and job description qualification requirements. Your task is to analyze, cross-reference, and create ATS-optimized certification entries that maximize alignment with target job requirements and preferred qualifications.

        **Data Processing Instructions:**
        - Analyze all provided sources to get a complete picture of professional certifications
        - Cross-reference certification information across sources to ensure accuracy and completeness
        - Verify certification details by checking consistency across multiple sources
        - Merge related certification data from different sources
        - Validate certification authenticity and current status when possible
        - Prioritize certifications that are current, relevant, and professionally valuable
        - **ATS Optimization**: Strategically align certification presentation with job description requirements

        **Certifications Extraction Requirements:**
        For each certification, extract and structure:

        1. **Certification name**: 
           - Use the full, official certification name as it appears from the issuing organization
           - Cross-reference between resume, LinkedIn, and other sources for accuracy
           - Include proper capitalization and official terminology
           - Avoid abbreviations unless they are the standard format (e.g., "AWS Certified Solutions Architect")
           - Ensure name reflects the exact certification earned
           - **JD Alignment**: Emphasize certifications that match or relate to JD qualifications

        2. **Issuing organization**: 
           - Use the official, complete organization name (e.g., "Amazon Web Services", "Microsoft", "Google Cloud")
           - Cross-reference across sources to ensure accuracy
           - Use standardized organization names rather than abbreviations when possible
           - Verify organization credibility and recognition in the industry

        3. **Date obtained**: 
           - Maintain exact format as it appears in the primary source
           - Use the most accurate and complete date available across all sources
           - Include month and year when available (e.g., "March 2023", "2023")
           - For ongoing or renewed certifications, indicate current status appropriately
           - If expiration date is mentioned, consider including it

        4. **Certification ID**: 
           - Include if explicitly mentioned in any source
           - Use exact format as provided (alphanumeric codes, URLs, etc.)
           - Cross-reference across sources to ensure accuracy
           - If not available, leave empty rather than guessing
           - Include verification URLs or badge links if provided

        5. **ATS-Optimized Description**: 
           - **CRITICAL**: Write in excellent, professional English with proper grammar
           - **JD Qualification Alignment**: Strategically highlight how the certification meets or supports JD requirements
           - **Requirement Matching**: Emphasize certifications that directly fulfill required or preferred qualifications
           - **Keyword Integration**: Naturally incorporate terminology from JD qualification requirements
           - **Professional Relevance**: Connect certification value to specific job requirements and responsibilities
           - **Skills Validation**: Highlight how certifications validate skills mentioned in JD qualifications
           - Create compelling, informative descriptions that highlight both certification value and JD alignment
           - Combine insights from all sources while optimizing for JD qualification requirements
           - Explain the certification's significance in context of target job requirements
           - Include key skills, knowledge areas, or competencies that match JD qualifications
           - Mention the certification's relevance to JD requirements and industry demands
           - Write 2-3 well-crafted sentences that demonstrate professional value and JD alignment
           - Focus on practical applications and career benefits that support target role requirements
           - **ATS Strategy**: Ensure descriptions include keywords and concepts from JD qualifications
           - Example approach: "This certification validates expertise in [JD qualification area] and demonstrates proficiency in [JD requirement skills]. Meets the [specific JD requirement] and provides advanced knowledge in [relevant JD qualification domain]."

        **Certification Validation Guidelines:**
        - Verify certification names against official issuing organization standards
        - Cross-check dates for reasonableness and consistency
        - Validate organization credibility and industry recognition
        - Ensure certifications are relevant to the candidate's career field and support JD qualifications
        - Prioritize current and renewable certifications that align with job requirements
        - Include professional certifications, technical certifications, and industry-recognized credentials
        - Focus on certifications that demonstrate fulfillment of JD qualification requirements
        - Exclude informal courses unless they directly support JD qualifications

        **ATS Optimization Guidelines:**
        - **Requirement Fulfillment**: Emphasize certifications that directly meet JD required or preferred qualifications
        - **Qualification Matching**: Highlight how certifications support specific job requirements
        - **Keyword Alignment**: Use terminology and language that matches JD qualification descriptions
        - **Professional Credibility**: Present certifications that validate expertise mentioned in JD qualifications
        - **Competitive Advantage**: Showcase certifications that exceed basic JD requirements
        - **Industry Relevance**: Focus on certifications recognized in the target job's industry
        - **Skills Validation**: Connect certifications to specific competencies required by the job

        **Quality Standards:**
        - Use professional language that highlights expertise and alignment with JD qualifications
        - Ensure consistency in formatting while maximizing qualification alignment
        - Include specific details that demonstrate certification relevance to job requirements
        - Maintain accuracy while presenting certifications in the most ATS-favorable light
        - Focus on certifications that add substantial value for the target role
        - Order certifications by JD relevance, requirement fulfillment, and professional impact

        **Professional Value Assessment:**
        - **Primary Focus**: Prioritize certifications that match JD required or preferred qualifications
        - Include certifications from recognized industry leaders that support job requirements
        - Focus on certifications that demonstrate continuous learning in JD-relevant areas
        - Consider certification relevance to specific JD qualification requirements
        - Include both technical and professional certifications that align with job needs
        - Highlight certifications that differentiate the candidate within JD qualification framework

        **Certification Prioritization Strategy:**
        - **Tier 1**: Certifications directly mentioned in JD required qualifications
        - **Tier 2**: Certifications that support or relate to JD preferred qualifications
        - **Tier 3**: Industry-relevant certifications that enhance overall qualification profile
        - **Tier 4**: Additional professional certifications that demonstrate commitment to learning

        """

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": f"Certification Information: {input_question}\n\nJob Description Qualification Requirements for ATS Optimization:\nRequired Qualifications: {jd_input.get('required_qualifications', [])}\nPreferred Qualifications: {jd_input.get('preferred_qualifications', [])}"}
    ],
    response_format=Certifications_data,
    )

    analysis_response = completion.choices[0].message
    total_tokens = completion.usage.total_tokens
    if hasattr(analysis_response, 'refusal') and analysis_response.refusal:
        print(f"Model refused to respond: {analysis_response.refusal}")
        return None, total_tokens
    else:
        parsed_data = Certifications_data(steps=analysis_response.parsed.steps)
        return parsed_data, total_tokens