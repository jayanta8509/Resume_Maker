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

    
class ATSAnalysis(BaseModel):
    ATS_score: float

class ats_data(BaseModel):
    analysis: ATSAnalysis


async def analyze_ats(resume_text):
    """
    Analyzes resumes and extracts comprehensive, structured information for ATS optimization.
    
    Args:
        resume_text (str): The complete resume text to analyze
    """    
    prompt_template = """You are an ATS (Applicant Tracking System) scoring assistant.

Goal
- Read the provided resume text.
- Produce ONE overall ATS score in the range 0-100 that reflects how well the resume would pass generic ATS screening.

Scoring rubric (concise, generic ATS heuristics)
- Keywords & Skills (≈40%): Presence and natural use of industry-relevant hard/soft skills, role titles, tools, certifications, action verbs. Penalize obvious keyword stuffing.
- Formatting & Structure (≈30%): ATS-parsable layout (single column, standard section headers like “Summary”, “Work Experience”, “Education”, “Skills”), consistent dates, readable bullets, no tables/images/text boxes/unicode icons that break parsing.
- Content Strength (≈30%): Clear, specific bullets; action-oriented language; measurable outcomes/metrics; minimal fluff; correct grammar/spelling.

Rules
- Compute a single overall score using the rubric (weights are guidelines; you may adapt if evidence is strong).
- Score MUST be between 0 and 100 (inclusive). Round to the nearest integer if uncertain.
- Do NOT include any prose, markdown, explanations, or extra fields.
- Do NOT assume a job description; evaluate the resume standalone.
- If the resume text is empty or non-resume content, return 0.

Return only the JSON object.
    """

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": f"Extract structured data from this resume: {resume_text}"}
        ],
        response_format=ats_data,
    )

    analysis_response = completion.choices[0].message
    total_tokens = completion.usage.total_tokens

    if hasattr(analysis_response, 'refusal') and analysis_response.refusal:
        print(f"Model refused to respond: {analysis_response.refusal}")
        return None, total_tokens
    else:
        parsed_data = ats_data(analysis=analysis_response.parsed.analysis)
        return parsed_data, total_tokens