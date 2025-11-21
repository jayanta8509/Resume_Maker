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

class Experience_Projects(BaseModel):
    Project_title: str
    Role: str
    technologies_used: list[str]
    Description: str

class ExperienceItem(BaseModel):
    CompanyName: str
    Position: str
    Duration: Duration
    Location: str
    Projects: list[Experience_Projects]
    SkillSet: list[str]


class Step(BaseModel):
    Experience: list[ExperienceItem]

class resume_experience_data(BaseModel):
    steps: list[Step]


async def analyze_resume_Experience(input_question):
    prompt_template = """ You are a meticulous resume parser. Extract ONLY the candidate’s EXPERIENCE and return data that conforms EXACTLY to this JSON shape:

            CRITICAL RULES
            1) OUTPUT FORMAT: Return exactly one JSON object with the shape above (no prose, no extra fields). If something is missing in the resume, use "" or [] as appropriate.

            2) DATES: Copy the date strings EXACTLY as written in the resume for StartDate and EndDate (preserve casing, separators, and abbreviations). 
            - Use only the explicit date span (e.g., "Jan 2020 – Mar 2022"); EXCLUDE trailing relative notes like "(3 yrs 2 mos)".
            - If current role, keep "Present"/"Current"/"Now" exactly as written.

            3) NON-INVENTION: Do NOT guess or fabricate company names, positions, dates, locations, technologies, or projects. Use resume text only.
            
            4) COMPANY IDENTIFICATION: Prefer official company names found in the resume. Use clues like email domains, LinkedIn/company URLs, and legal names/subsidiaries when explicitly present in the resume.
           
             5) MULTIPLE ROLES / TENURES: 
            - If the candidate held multiple titles at the SAME company with distinct date ranges, create SEPARATE Experience items (one per title/date range).
            - If non-contiguous tenures exist at the same company, create separate items per tenure.
            
            6) LOCATION: Copy location text exactly as written (e.g., "Bengaluru, India", "Remote", "Hybrid – NYC").
            
            7) SKILLSET (per company): Provide a deduplicated list of skills explicitly mentioned in that company’s section (including bullets and project descriptions). Do not pull skills from other parts of the resume.
            
            8) PROJECTS (inside each company):
            - **AGGRESSIVE EXTRACTION**: Find EVERY project mentioned under each company experience, including bullet points, achievements, or descriptions that describe specific work
            - Project_title:
              * If explicitly named (e.g., "Project: Phoenix", "E-commerce Platform"), copy it exactly
              * If not named but clearly described, CREATE a descriptive title based on the work described (e.g., "Customer Management System", "Mobile App Development")
              * **NEVER leave Project_title empty if there's any project work described**
            - Role:
              * Copy exactly if specified (e.g., "Backend Engineer", "Tech Lead")
              * If not specified but implied from the job position, use the job title (e.g., "Software Engineer")
              * **NEVER leave Role empty if there's project work**
            - technologies_used: extract all technologies mentioned in relation to the project or role
            - Description:
              * Create a summary based on the project work described in bullet points or text
              * **NEVER leave Description empty if project work exists**
            
            9) OVERLAPS: If date ranges overlap across roles, do not merge or modify; simply record each role as it appears.
            
            10) CLEANUP: Normalize whitespace; preserve punctuation/case from the resume. Remove inline markdown or HTML if present.

            EDGE CASES TO HANDLE
            - Internships, apprenticeships, contracts, part-time, freelance → include them like other roles.
            - Company acquisitions/renames → use the name as written for the period.
            - Dates shown as "2019-Present", "Jul'19 – Mar'22", "07/2019 – 03/2022" → preserve original forms.
            - If only a year is given (e.g., "2021 – 2023"), keep it as is.

            **CRITICAL: PROJECT EXTRACTION RULES**
            - If a company section has bullet points, achievements, or detailed responsibilities, TREAT EACH as potential projects
            - Look for phrases like: "developed", "built", "created", "led", "implemented", "designed", "managed" - these indicate project work
            - **NEVER return empty Projects list [] if the work description contains specific achievements or deliverables**
            - When in doubt, create a project entry rather than leaving it empty

        """
    
    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
    model="gpt-5.1",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": input_question}
    ],
    response_format=resume_experience_data,
    )

    analysis_response = completion.choices[0].message
    total_tokens = completion.usage.total_tokens
    if hasattr(analysis_response, 'refusal') and analysis_response.refusal:
        print(f"Model refused to respond: {analysis_response.refusal}")
        return None, total_tokens
    else:
        parsed_data = resume_experience_data(steps=analysis_response.parsed.steps)
        print("dataatatata : ", parsed_data)
        return parsed_data, total_tokens





