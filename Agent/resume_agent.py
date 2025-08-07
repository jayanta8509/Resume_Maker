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

class Duration(BaseModel):
    StartDate: str
    EndDate: str

class ExperienceItem(BaseModel):
    CompanyName: str
    Position: str
    Duration: Duration
    Location: str
    Description: str

class EducationItem(BaseModel):
    CollegeUniversity: str
    Location: str
    CourseDegree: str
    GraduationYear: str
    GPAorGrade: str
    AdditionalInformation: str


class Languages(BaseModel):
    Language: str
    Proficiency: str

class Projects(BaseModel):
    ProjectName: str
    Description: str
    Technologies: list[str]
    YourRole: str
    Duration: Duration

class Certifications(BaseModel):
    CertificationName: str
    Issuing_Organization: str
    DateObtained: str
    Certification_ID : str
    Description: str

class Achievements(BaseModel):
    Achievement_Titlee: str
    Issuing_Organization: str
    Date_Received: str
    Description : str

class Skills(BaseModel):
    Skill_Category: str
    Skills: list[str]


class Step(BaseModel):
    SuggestedRole: str
    CandidateFullName: str
    EmailAddress: str
    PhoneNumber: str
    ProfessionalTitle: str
    Summary: str
    YearsOfExperienceRequired: str
    Experience: list[ExperienceItem]
    Education: list[EducationItem]
    Languages: list[Languages]
    Projects: list[Projects]
    Certifications: list[Certifications]
    Achievements: list[Achievements]
    Skills: list[Skills]


class resume_data(BaseModel):
    steps: list[Step]


async def analyze_resume(input_question):

    prompt_template = """ You are an expert resume parser. Extract the following information from the resume and structure it according to the specified format:

        1. **Basic Information:**
           - Candidate's full name
           - Email address (check personal email, work emails, LinkedIn profiles)
           - Phone number
           - Professional title (current or most recent job title)

        2. **Professional Summary:**
           - Create a comprehensive summary highlighting the candidate's key strengths, experience, and career focus

        3. **Suggested Role:**
           - Based on the candidate's work experience, education, and technical skill set, suggest the most suitable job role they are likely to be both qualified for and interested in
           - The suggested role should align with the candidate's career progression, domain expertise, and strengths demonstrated in their resume
           - Ensure the recommendation reflects realistic career advancement and industry relevance

        4. **Years of Experience Required:**
           - **CRITICAL**: Calculate the total professional work experience in years based on all employment history
           - **Calculation Method**: 
             * Analyze each work experience entry and extract start/end dates
             * Convert all date ranges to years and months
             * Sum up the total duration across all positions
             * Handle overlapping employment periods by counting them only once
             * For current roles (ongoing), calculate from start date to present
             * Account for gaps in employment (do not count gap periods)
           - **Output Format**: Provide as a single number followed by "years" (e.g., "3.5 years", "7 years", "1.2 years")
           - **Examples**:
             * Jan 2020 - Dec 2022 = 3 years
             * Mar 2019 - Present (assuming current date is 2024) = 5 years
             * Multiple roles: 2 years + 1.5 years + 3 years = 6.5 years
           - **Edge Cases**:
             * If experience is less than 1 year, use months (e.g., "8 months")
             * If dates are unclear, make reasonable assumptions and note uncertainty
             * For part-time or contract work, count the actual time worked

        5. **Experience:**
           For each company experience, extract:
           - Company name (LOOK CAREFULLY - check email domains, LinkedIn URLs, official company names, subsidiaries)
           - Position/role
           - Duration (specify EXACT start date and end date in the same format they appear in the resume)
           - Company location
           - Detailed description of responsibilities and achievements

        6. **Education:**
           For each educational institution:
           - College/University name
           - Location of the institution
           - Course/degree name
           - Graduation year
           - GPA or grade (if mentioned)
           - Additional information (honors, relevant coursework, etc.)

        7. **Skills:**
           Categorize skills into groups such as:
           - Programming Languages
           - Frameworks & Libraries
           - Databases
           - Tools & Technologies
           - Soft Skills
           - Domain Expertise
           (Create appropriate categories based on the resume content)

        8. **Projects:**
           For each project mentioned:
           - Project name
           - Description of the project
           - Technologies used
           - Your role in the project
           - Duration/timeline

        9. **Certifications:**
           For each certification:
           - Certification name
           - Issuing organization
           - Date obtained
           - Certification ID (if available)
           - Description or relevance

        10. **Achievements:**
           For each achievement/award:
           - Achievement title
           - Issuing organization
           - Date received
           - Description

        11. **Languages:**
            For each language:
            - Language name
            - Proficiency level (e.g., Native, Fluent, Intermediate, Basic)

        IMPORTANT INSTRUCTIONS:
        1. Extract dates EXACTLY as they appear in the resume without reformatting
        2. For Duration, maintain the exact format from the resume (e.g., "Jan 2020 - Mar 2022", "2019-Present")
        3. If a field is not present or cannot be determined, use empty string "" or empty list [] as appropriate
        4. Be aggressive about finding company names from any source in the resume
        5. Group skills logically by category
        6. Extract all projects, certifications, and achievements mentioned
        7. Ensure the professional summary is comprehensive and highlights key strengths
        8. **MANDATORY**: Calculate YearsOfExperienceRequired by analyzing ALL work experience dates and summing the total professional experience
        9. For YearsOfExperienceRequired, be precise in date calculations and handle edge cases properly
        10. If multiple overlapping positions exist, count the time period only once to avoid double-counting

        """

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": input_question}
    ],
    response_format=resume_data,
    )

    analysis_response = completion.choices[0].message
    total_tokens = completion.usage.total_tokens
    if hasattr(analysis_response, 'refusal') and analysis_response.refusal:
        print(f"Model refused to respond: {analysis_response.refusal}")
        return None, total_tokens
    else:
        parsed_data = resume_data(steps=analysis_response.parsed.steps)
        return parsed_data, total_tokens