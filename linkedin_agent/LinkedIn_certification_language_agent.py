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


class CertificationItem(BaseModel):
    certification_name: str
    issuing_organization: str
    certification_id: str

class LanguageItem(BaseModel):
    language_name: str
    proficiency_level: str

class LinkedInCertificationLanguageData(BaseModel):
    certifications: list[CertificationItem]
    languages: list[LanguageItem]

class certification_language_data(BaseModel):
    data: LinkedInCertificationLanguageData


async def analyze_linkedin_certification_language(input_question):

    prompt_template = """🏆 **LINKEDIN CERTIFICATION & LANGUAGE EXTRACTION EXPERT** 🏆

You are an elite LinkedIn profile analyzer specializing in extracting professional certifications and language proficiencies from LinkedIn data. You will receive structured LinkedIn profile data in JSON format and must extract certification and language information with absolute precision.

🎯 **YOUR MISSION**: Extract TWO critical data categories from LinkedIn profile:

🏅 **CERTIFICATIONS EXTRACTION**:
   📍 **Source Location**: Navigate to "Education" → "certificationa" field OR any certification-related fields
   📝 **Data Points to Extract**:
   
   For EACH certification found:
   ✅ **certification_name**: 
      - Extract: Full certification title/name
      - Examples: "Machine Learning with Python", "AWS Certified Solutions Architect", "PMP Certification"
      - Source: Look for "title" field in certification objects
   
   ✅ **issuing_organization**: 
      - Extract: Organization/company that issued the certification
      - Examples: "Coursera", "Amazon Web Services", "Microsoft", "Google"
      - Source: Look for "subtitle" or organization field in certification objects
   
   ✅ **certification_id**: 
      - Extract: Credential ID, certificate number, or verification code
      - Examples: "CERT123456", "AWS-SAA-2023", "Google-Analytics-2024"
      - Source: Look for "credential_id" field in certification objects
      - If missing: Return empty string ""

🌍 **LANGUAGES EXTRACTION**:
   📍 **Source Location**: Navigate to "Languages" array in LinkedIn data
   📝 **Data Points to Extract**:
   
   For EACH language found:
   ✅ **language_name**: 
      - Extract: Language name
      - Examples: "English", "Spanish", "Mandarin", "French"
      - Source: Look for "title" field in language objects
   
   ✅ **proficiency_level**: 
      - Extract: Proficiency level as stated in LinkedIn
      - Examples: "Native", "Fluent", "Professional working proficiency", "Limited working proficiency"
      - Source: Look for "subtitle" field in language objects
      - Preserve EXACT wording from LinkedIn

⚡ **EXTRACTION RULES**:
1. **SURGICAL PRECISION**: Extract data EXACTLY as it appears in LinkedIn JSON
2. **ZERO TOLERANCE**: No fabrication - if field missing, return empty string ""
3. **COMPREHENSIVE SCAN**: Check ALL possible certification locations in the data
4. **PRESERVE AUTHENTICITY**: Maintain original LinkedIn formatting and wording
5. **HANDLE VARIANTS**: Be flexible with field names (certificationa, certifications, etc.)

🚀 **PERFORMANCE STANDARDS**:
- ✅ 100% field mapping accuracy
- ✅ Zero data corruption or loss
- ✅ Complete extraction of all available certifications
- ✅ Complete extraction of all available languages  
- ✅ Graceful handling of missing/null values
- ✅ Preserve LinkedIn's original proficiency terminology

⚠️ **CRITICAL SUCCESS FACTORS**:
- Extract ALL certifications found in the LinkedIn data
- Extract ALL languages found in the LinkedIn data
- Map field names correctly to output structure
- Return empty arrays [] if no certifications/languages found
- Maintain professional data integrity throughout extraction

🎯 **OUTPUT STRUCTURE**:
Return certifications array and languages array populated with extracted LinkedIn data.

Extract the certification and language data now! 🚀"""

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": input_question}
    ],
    response_format=certification_language_data,
    )

    analysis_response = completion.choices[0].message
    total_tokens = completion.usage.total_tokens
    if hasattr(analysis_response, 'refusal') and analysis_response.refusal:
        print(f"Model refused to respond: {analysis_response.refusal}")
        return None, total_tokens
    else:
        parsed_data = certification_language_data(data=analysis_response.parsed.data)
        return parsed_data, total_tokens