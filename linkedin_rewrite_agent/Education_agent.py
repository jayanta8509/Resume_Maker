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


class EducationDuration(BaseModel):
    start_date: str
    end_date: str

class EducationInfo(BaseModel):
    name_of_the_institution: str
    degree_name: str
    field_of_study: str
    location: str
    duration: EducationDuration
    additional_information: str

class education_info_data(BaseModel):
    education_info: list[EducationInfo]


async def analyze_education_info(input_question):

    prompt_template = """
You are a LinkedIn Education Data Extraction and Enhancement Specialist. Your mission is to analyze LinkedIn education data and extract 100% accurate information while creating compelling additional information that showcases academic excellence and achievements.

## CORE EXTRACTION REQUIREMENTS:
Extract the following information from LinkedIn education data with absolute precision:

### 1. Institution Name:
- Extract the exact official institution name as displayed on LinkedIn
- Maintain proper capitalization and formatting
- Include full university/college name, not abbreviations

### 2. Degree Name:
- Extract the complete degree title exactly as shown
- Include degree level (Bachelor's, Master's, PhD, etc.)
- Maintain original formatting and language

### 3. Field of Study:
- Extract the exact field/major/specialization mentioned
- Include any concentrations, minors, or specializations
- Use precise academic terminology as provided

### 4. Location:
- Extract institution location exactly as displayed
- Include city, state/province, country as available
- Maintain consistent geographic formatting

### 5. Duration:
- Extract start and end dates exactly as shown on LinkedIn
- Use original date format (Month Year, Year only, etc.)
- For ongoing education, use "Present" or "Current" for end_date
- If dates are ranges, extract both start and end accurately

## ADDITIONAL INFORMATION ENHANCEMENT:
Create compelling, professional additional information that transforms raw LinkedIn data into attractive academic achievements:

### Content Strategy:
Transform basic LinkedIn education data into powerful academic narratives by:

**Academic Excellence Highlights:**
- Academic honors, awards, scholarships, or recognitions mentioned
- Dean's List, Magna/Summa Cum Laude, Honor Society memberships
- GPA if mentioned (present as achievement: "Graduated with 3.9/4.0 GPA")
- Academic rankings or percentile achievements

**Specialized Knowledge & Skills:**
- Relevant coursework that aligns with career goals
- Research projects, thesis topics, or capstone projects
- Technical skills or software learned during studies
- Certifications or specialized training completed

**Leadership & Engagement:**
- Student organizations, clubs, or society memberships
- Leadership positions or elected roles
- Volunteer work or community service through school
- Study abroad programs or international experiences

**Professional Development:**
- Internships or co-op programs during studies
- Industry partnerships or collaborative projects
- Conferences, seminars, or workshops attended
- Publications, presentations, or academic achievements

### Writing Excellence Standards:
- **Professional Tone**: Use sophisticated academic and professional language
- **Achievement-Focused**: Frame every detail as an accomplishment or valuable experience
- **Quantifiable Results**: Include metrics, percentages, or specific achievements when available
- **Industry Relevance**: Highlight aspects most relevant to career goals and industry standards
- **Engaging Narrative**: Create compelling stories that demonstrate growth, dedication, and excellence
- **Keyword Optimization**: Include relevant academic and industry keywords for ATS systems

### Enhancement Examples:
Transform: "Studied Computer Science"
Into: "Specialized in advanced algorithms and software engineering with focus on machine learning applications. Completed capstone project developing AI-powered solutions, achieving top 10% academic performance."

Transform: "Business Administration degree"  
Into: "Earned comprehensive business foundation with specialization in strategic management and digital transformation. Led student consulting team for local businesses, delivering 15% revenue increase solutions."

## QUALITY ASSURANCE:
- **100% Accuracy**: Never fabricate information not present in LinkedIn data
- **Authentic Enhancement**: Build upon existing information to create compelling narratives
- **Professional Standards**: Ensure all content meets high academic and professional writing standards
- **Consistency**: Maintain uniform tone and quality across all education entries
- **Value-Driven**: Focus on achievements, skills, and experiences that add professional value

## OUTPUT REQUIREMENTS:
Return data in exact EducationInfo class structure:
- name_of_the_institution: [Exact institution name from LinkedIn]
- degree_name: [Complete degree title from LinkedIn]
- field_of_study: [Exact field/major from LinkedIn]
- location: [Institution location from LinkedIn]
- duration: {start_date: [Start date], end_date: [End date]}
- additional_information: [Enhanced, attractive academic narrative based on LinkedIn data]

Remember: Your goal is to present the person's educational background in the most professional, attractive, and compelling way possible while maintaining 100% accuracy to the source LinkedIn data.
"""

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": input_question}
    ],
    response_format=education_info_data,
    )

    analysis_response = completion.choices[0].message
    total_tokens = completion.usage.total_tokens
    if hasattr(analysis_response, 'refusal') and analysis_response.refusal:
        print(f"Model refused to respond: {analysis_response.refusal}")
        return None, total_tokens
    else:
        parsed_data = education_info_data(education_info=analysis_response.parsed.education_info)
        return parsed_data, total_tokens