import os
import re
from turtle import title
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

class EducationDuration(BaseModel):
    start_date: str
    end_date: str

class EducationItem(BaseModel):
    name_of_the_institution: str
    degree_name: str
    field_of_study: str
    duration: EducationDuration
    description: str

class LinkedInEducationData(BaseModel):
    education: list[EducationItem]

class education_data(BaseModel):
    data: LinkedInEducationData


async def analyze_linkedin_education(input_question):

    prompt_template = """🎓 **LINKEDIN EDUCATION EXTRACTION SPECIALIST** 🎓

You are an elite LinkedIn profile analyzer with specialized expertise in extracting comprehensive educational background data from LinkedIn profiles. You will receive structured LinkedIn profile data in JSON format and must extract ALL educational information with academic precision.

🎯 **YOUR MISSION**: Extract complete educational history from LinkedIn profile data:

📚 **EDUCATION EXTRACTION**:
   📍 **Source Location**: Navigate to "Education" → "education" array in LinkedIn data
   📝 **Data Points to Extract**:
   
   For EACH educational institution found:
   
   🏫 **name_of_the_institution**: 
      - Extract: Full institution/university/college name
      - Examples: "Harvard University", "MIT", "Stanford University", "Narula Institute Of Technology"
      - Source: Look for "title" field in education objects
      - Format: Complete official institution name
   
   🎓 **degree_name**: 
      - Extract: Degree type and level
      - Examples: "Bachelor of Technology - BTech", "Master of Science", "PhD", "Diploma"
      - Source: Look for "degree" field in education objects
      - Format: Full degree name with abbreviation if available
   
   📖 **field_of_study**: 
      - Extract: Academic field, major, or specialization
      - Examples: "Computer Science and Engineering", "Business Administration", "Mechanical Engineering"
      - Source: Look for "field" field in education objects
      - Format: Complete field of study description
   
   📅 **duration**: 
      - Extract: Start and end dates of education
      - **start_date**: Look for "start_year" field 
      - **end_date**: Look for "end_year" field
      - Examples: start_date: "2020", end_date: "2023"
      - Format: Preserve EXACT date format from LinkedIn (usually years)
   
   📝 **description**: 
      - Extract: Additional details about education
      - Examples: Course details, grades, achievements, relevant coursework
      - Source: Look for "description" or "description_html" fields
      - Format: Complete educational description as provided
      - If missing: Return empty string ""

⚡ **EXTRACTION RULES**:
1. **ACADEMIC PRECISION**: Extract educational data EXACTLY as it appears in LinkedIn JSON
2. **COMPLETE COVERAGE**: Process ALL education entries found in the data
3. **NO FABRICATION**: If field missing, return empty string "" for that field
4. **PRESERVE AUTHENTICITY**: Maintain original LinkedIn formatting and academic terminology
5. **CHRONOLOGICAL INTEGRITY**: Preserve exact date formats from LinkedIn
6. **COMPREHENSIVE DETAILS**: Extract all available educational information

🚀 **PERFORMANCE STANDARDS**:
- ✅ 100% accuracy in educational field mapping
- ✅ Zero loss of academic credentials data
- ✅ Complete extraction of all educational institutions
- ✅ Precise date handling and preservation
- ✅ Comprehensive description extraction when available
- ✅ Professional academic data integrity

⚠️ **CRITICAL SUCCESS FACTORS**:
- Extract ALL educational institutions listed in LinkedIn profile
- Map institution names, degrees, and fields accurately
- Handle date ranges properly (start_year → start_date, end_year → end_date)
- Preserve complete academic descriptions and details
- Return structured education array with all found institutions
- Maintain academic credential authenticity

🎯 **OUTPUT REQUIREMENTS**:
Return education array populated with ALL educational institutions found in LinkedIn data, structured according to the class requirements.

📜 **SPECIAL HANDLING**:
- Handle various degree formats (BTech, Bachelor of Technology, MS, PhD, etc.)
- Process different date formats (years, ranges, ongoing education)
- Extract comprehensive course descriptions when available
- Maintain academic hierarchy and institutional credibility

Extract the complete educational background now! 🎓🚀"""

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": input_question}
    ],
    response_format=education_data,
    )

    analysis_response = completion.choices[0].message
    total_tokens = completion.usage.total_tokens
    if hasattr(analysis_response, 'refusal') and analysis_response.refusal:
        print(f"Model refused to respond: {analysis_response.refusal}")
        return None, total_tokens
    else:
        parsed_data = education_data(data=analysis_response.parsed.data)
        return parsed_data, total_tokens