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

class BasicInformation(BaseModel):
    SuggestedRole: str
    CandidateFullName: str
    EmailAddress: str
    PhoneNumber: str
    ProfessionalTitle: str
    Summary: str

class BasicInformationData(BaseModel):
    steps: list[BasicInformation]


async def analyze_basic_information(input_question):
    """
    Analyzes and extracts basic information from combined resume and LinkedIn profile data.
    
    Args:
        input_question (str): Combined text containing basic information from both 
                             resume and LinkedIn profile (name, email, phone, title, etc.)
    
    Returns:
        tuple: (BasicInformationData object, total_tokens_used)
    """

    prompt_template = """ You are an expert data analyst. You will receive basic information from both a resume and a LinkedIn profile. Intelligently merge and prioritize this data to extract the most accurate and complete information according to the specified format:

        **Data Processing Instructions:**
        - If both sources provide the same information, use the most recent or complete version
        - If information conflicts, prioritize LinkedIn for current professional title and resume for contact details
        - If information is missing from one source, use the available data from the other source
        - Ensure all extracted information is accurate and properly formatted

        1. **Basic Information:**
           - **Candidate's full name**: Extract from either source, use the most complete version
           - **Email address**: Prioritize personal/professional email from resume, then LinkedIn if not available
           - **Phone number**: Extract from resume primarily, use LinkedIn if resume doesn't contain it
           - **Professional title**: Use the most current title from LinkedIn, fallback to resume if LinkedIn is outdated

        2. **Professional Summary:**
           - Create a comprehensive summary by combining insights from both resume and LinkedIn profile
           - Write in first person using "I" instead of mentioning the candidate's name
           - Highlight key strengths, experience, and career focus mentioned in either source
           - Ensure the summary reflects the candidate's current professional status and aspirations
           - Keep it concise but comprehensive (3-4 sentences)
           - Example format: "I am a [professional title] with [X years] of experience in [domain]..."

        3. **Suggested Role:**
           - Analyze work experience, education, and technical skills from both sources
           - Suggest the most suitable job role based on combined data from resume and LinkedIn
           - Consider career progression patterns visible in both sources
           - Ensure the recommendation reflects realistic career advancement and current industry trends
           - The suggested role should align with the candidate's demonstrated expertise and stated interests

        **Input Format Expected:**
        The input will contain basic information from both resume and LinkedIn profile. Process both sources comprehensively to provide the most accurate output.

        """

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": input_question}
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