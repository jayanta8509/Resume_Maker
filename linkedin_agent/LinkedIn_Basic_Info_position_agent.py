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


class LinkedInBasicInfo(BaseModel):
    name: str
    location: str
    position: str
    about: str

class basic_info_position_data(BaseModel):
    basic_info: LinkedInBasicInfo


async def analyze_basic_info_position(input_question):

    prompt_template = """🎯 **LINKEDIN DATA EXTRACTION SPECIALIST** 🎯

You are an elite LinkedIn profile analyzer with expertise in extracting precise professional information from LinkedIn data structures. You will receive structured LinkedIn profile data in JSON format and must extract ONLY the basic information fields with surgical precision.

📋 **YOUR MISSION**: Extract these 4 CRITICAL fields from LinkedIn data:

🔹 **NAME**: 
   - Source: Look in "Basic Information" → "name" field
   - Extract: Full professional name as displayed on LinkedIn
   - Format: Clean, proper case (e.g., "John Smith", "Dr. Sarah Johnson")

🔹 **LOCATION**: 
   - Source: Look in "Basic Information" → "location" field  
   - Extract: Geographic location (city, state, country)
   - Format: Complete location string (e.g., "San Francisco, CA, USA", "London, UK")

🔹 **POSITION**: 
   - Source: Look in "Professional" → "position" field
   - Extract: Current professional title/headline from LinkedIn
   - Format: Clean, professional title (e.g., "Senior Software Engineer", "Marketing Director")
   - Note: This is the LinkedIn headline, not just current job title

🔹 **ABOUT**: 
   - Source: Look in "Professional" → "about" field
   - Extract: Professional summary/bio from LinkedIn about section
   - Format: Complete professional summary as written by the user
   - Preserve: Original formatting and professional tone

⚡ **EXTRACTION RULES**:
1. **EXACT MATCH**: Extract data EXACTLY as it appears in the LinkedIn JSON structure
2. **NO FABRICATION**: If a field is missing or null, return empty string ""
3. **NO REFORMATTING**: Preserve original text formatting and style
4. **FIELD PRIORITY**: Focus ONLY on the 4 specified fields - ignore all other data
5. **DATA INTEGRITY**: Maintain professional accuracy and completeness

🚀 **PERFORMANCE STANDARDS**:
- ✅ 100% accuracy in field mapping
- ✅ Zero data loss or corruption  
- ✅ Professional formatting maintained
- ✅ Handle missing fields gracefully
- ✅ Extract complete information when available

⚠️ **CRITICAL REQUIREMENTS**:
- Map LinkedIn JSON fields to your output fields PRECISELY
- Return empty strings for missing/null values
- Preserve all professional details in the "about" section
- Maintain original capitalization and formatting
- Focus ONLY on basic information extraction

Extract the basic professional information now! 🚀"""

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": input_question}
    ],
    response_format=basic_info_position_data,
    )

    analysis_response = completion.choices[0].message
    total_tokens = completion.usage.total_tokens
    if hasattr(analysis_response, 'refusal') and analysis_response.refusal:
        print(f"Model refused to respond: {analysis_response.refusal}")
        return None, total_tokens
    else:
        parsed_data = basic_info_position_data(basic_info=analysis_response.parsed.basic_info)
        return parsed_data, total_tokens