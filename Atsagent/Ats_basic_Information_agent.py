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

class BasicInformation(BaseModel):
    SuggestedRole: str
    CandidateFullName: str
    EmailAddress: str
    PhoneNumber: str
    ProfessionalTitle: str
    Summary: str

class BasicInformationData(BaseModel):
    steps: list[BasicInformation]


async def analyze_basic_information(input_question, job_title):
    """
    Analyzes and extracts basic information from combined resume and LinkedIn profile data,
    optimized for ATS matching based on target job title.
    
    Args:
        input_question (str): Combined text containing basic information from both 
                             resume and LinkedIn profile (name, email, phone, title, etc.)
        job_title (str): Target job title for ATS optimization and role alignment
    
    Returns:
        tuple: (BasicInformationData object with ATS-optimized content, total_tokens_used)
    """

    prompt_template = """ You are an expert ATS optimization specialist and data analyst. You will receive basic information from both a resume and LinkedIn profile, along with a target job title. Your task is to intelligently merge candidate data and optimize it for ATS systems while ensuring alignment with the target job role.

        **Data Processing Instructions:**
        - If both sources provide the same information, use the most recent or complete version
        - If information conflicts, prioritize LinkedIn for current professional title and resume for contact details
        - If information is missing from one source, use the available data from the other source
        - Ensure all extracted information is accurate and properly formatted
        - **ATS Optimization**: Align extracted information with the target job title and industry standards

        1. **Basic Information:**
           - **Candidate's full name**: Extract from either source, use the most complete version
           - **Email address**: Prioritize personal/professional email from resume, then LinkedIn if not available
           - **Phone number**: Extract from resume primarily, use LinkedIn if resume doesn't contain it
           - **Professional title**: Use the most current title from LinkedIn, fallback to resume if LinkedIn is outdated

        2. **ATS-Optimized Professional Summary:**
           - **CRITICAL**: Create an ATS-friendly summary that aligns with the target job title and industry
           - Write in first person using "I" instead of mentioning the candidate's name
           - **Role Alignment**: Tailor the summary to match the target job title and its typical requirements
           - **Industry Keywords**: Include relevant technical skills, tools, and methodologies common for the target role
           - **Skills Matching**: Highlight experience and skills from candidate's background that align with the target position
           - **Professional Terminology**: Use industry-standard language and terminology associated with the target job title
           - **Action Verbs**: Use strong action verbs commonly associated with the target role responsibilities
           - **Quantifiable Impact**: Include metrics and achievements that relate to typical requirements of the target position
           - Keep it concise but comprehensive (3-4 sentences) while optimizing for role-specific keywords
           - **ATS Strategy**: Ensure the summary includes keywords that ATS systems expect for the target job title
           - Example approach: "I am a [target job title] with [X years] of experience in [role-relevant domains], specializing in [role-specific skills]. I have demonstrated expertise in [technical skills for this role] and [relevant methodologies], delivering [role-relevant achievements]. My background in [industry/domain] positions me to excel in [typical responsibilities of target role]."

        3. **Target Role Alignment:**
           - **Primary Focus**: Use the provided target job title as the suggested role if candidate is qualified
           - Analyze work experience, education, and technical skills from both sources against typical requirements for the target role
           - Assess how well the candidate's background aligns with the target job title
           - Consider industry standards and typical qualifications for the target position
           - If candidate closely matches typical requirements, suggest the exact target role
           - If there's a gap, suggest the closest aligned role or the target role with growth potential

        **ATS Optimization Guidelines:**
        - **Role-Specific Keywords**: Include keywords commonly associated with the target job title
        - **Industry Standards**: Use terminology and skills typical for the target role
        - **Skills Alignment**: Highlight candidate skills that match typical requirements for the target position
        - **Professional Language**: Use industry-standard language for the target role's field
        - **Natural Integration**: Maintain professional, readable content while optimizing for role-specific ATS filters

        **Input Format Expected:**
        The input will contain basic information from both resume and LinkedIn profile, plus a target job title for optimization. Process all sources to align the output with the target role requirements.

        """

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
    model="gpt-5.1",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": f"Candidate Information: {input_question}\n\nTarget Job Title for ATS Optimization: {job_title}"}
    ],
    response_format=BasicInformationData,
    )

    analysis_response = completion.choices[0].message
    total_tokens = completion.usage.total_tokens
    if hasattr(analysis_response, 'refusal') and analysis_response.refusal:
        print(f"Model refused to respond: {analysis_response.refusal}")
        return None, total_tokens
    else:
        parsed_data = BasicInformationData(steps=analysis_response.parsed.steps)
        return parsed_data, total_tokens