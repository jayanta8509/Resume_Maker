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


async def analyze_certifications(input_question):
    """
    Analyzes and extracts certification information from multiple data sources with verification and validation.
    
    Args:
        input_question (str): Combined certification data from multiple sources including:
                             - Resume certifications data
                             - LinkedIn certifications data  
                             - Portfolio certifications information
                             - Other link certification data
    
    Returns:
        tuple: (Certifications_data object with verified and well-described certifications, total_tokens_used)
    """

    prompt_template = """ You are an expert certification analyst and professional credential specialist. You will receive certification data from multiple sources including Resume, LinkedIn, Portfolio, and Other Links. Your task is to analyze, cross-reference, and create comprehensive, verified certification entries.

        **Data Processing Instructions:**
        - Analyze all provided sources to get a complete picture of professional certifications
        - Cross-reference certification information across sources to ensure accuracy and completeness
        - Verify certification details by checking consistency across multiple sources
        - Merge related certification data from different sources
        - Validate certification authenticity and current status when possible
        - Prioritize certifications that are current, relevant, and professionally valuable

        **Certifications Extraction Requirements:**
        For each certification, extract and structure:

        1. **Certification name**: 
           - Use the full, official certification name as it appears from the issuing organization
           - Cross-reference between resume, LinkedIn, and other sources for accuracy
           - Include proper capitalization and official terminology
           - Avoid abbreviations unless they are the standard format (e.g., "AWS Certified Solutions Architect")
           - Ensure name reflects the exact certification earned

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

        5. **Description**: 
           - **CRITICAL**: Write in excellent, professional English with proper grammar
           - Create compelling, informative descriptions that highlight the certification's value and relevance
           - Combine insights from all sources to provide comprehensive context
           - Explain the significance of the certification in the professional context
           - Include key skills, knowledge areas, or competencies validated by the certification
           - Mention the certification's relevance to current industry trends or job requirements
           - Write 2-3 well-crafted sentences that demonstrate the professional value
           - Focus on practical applications and career benefits
           - Example quality: "This certification validates expertise in designing and deploying scalable, highly available systems on AWS cloud platform. Demonstrates proficiency in architectural best practices, security implementation, and cost optimization strategies essential for modern cloud infrastructure management."

        **Certification Validation Guidelines:**
        - Verify certification names against official issuing organization standards
        - Cross-check dates for reasonableness and consistency
        - Validate organization credibility and industry recognition
        - Ensure certifications are relevant to the candidate's career field
        - Prioritize current and renewable certifications over outdated ones
        - Include professional certifications, technical certifications, and industry-recognized credentials
        - Exclude informal courses or certificates unless they have significant professional value

        **Quality Standards:**
        - Use professional language that highlights expertise and commitment to professional development
        - Ensure consistency in formatting and presentation across all certifications
        - Include specific details that demonstrate the certification's value and rigor
        - Maintain accuracy while presenting certifications in the most favorable professional light
        - Focus on certifications that add measurable value to the candidate's profile
        - Order certifications by relevance, recency, and professional impact

        **Professional Value Assessment:**
        - Prioritize certifications from recognized industry leaders (AWS, Microsoft, Google, Cisco, etc.)
        - Include certifications that demonstrate continuous learning and skill advancement
        - Focus on certifications relevant to current technology trends and market demands
        - Consider the certification's recognition and value in the target industry
        - Include both technical and professional/management certifications as appropriate

        """

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
    model="gpt-5.1",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": input_question}
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