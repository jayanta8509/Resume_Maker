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


class PersonalInfo(BaseModel):
    full_name: str
    location: str
    description: str
    about: str
    current_company: str

class personal_info_data(BaseModel):
    personal_info: PersonalInfo


async def analyze_personal_info(input_question):

    prompt_template = """
You are a LinkedIn Profile Optimization Expert and Data Extraction Specialist. Your task is to analyze LinkedIn profile data and extract accurate information while creating compelling, professional descriptions.

## EXTRACTION REQUIREMENTS:
Extract the following information from the LinkedIn data with 100% accuracy:
- Full Name: Extract the exact name as displayed on the profile
- Location: Extract the current location/city mentioned
- Current Company: Extract the most recent/current company name
- Raw Profile Data: Use this as the foundation for creating enhanced content

## CONTENT ENHANCEMENT GUIDELINES:

### For "description" field:
Create a powerful, concise professional headline (1-2 lines) that:
- Captures the person's core professional identity
- Highlights their primary expertise/role
- Uses industry-relevant keywords
- Shows value proposition
- Is attention-grabbing and memorable
- Examples: "Senior Software Engineer | Full-Stack Developer | Building Scalable Solutions" or "Digital Marketing Strategist | Growth Hacker | Driving 300% ROI for Tech Startups"

### For "about" field:
Craft a compelling professional summary (3-5 sentences) that:
- Tells a cohesive professional story
- Highlights key achievements with specific metrics when available
- Showcases unique value proposition
- Demonstrates expertise and passion
- Uses action-oriented language
- Includes relevant industry keywords
- Ends with a call-to-action or future goals
- Example structure: "Proven [role] with [X years] experience in [industry/field]. Successfully [key achievement with metrics]. Expert in [core skills/technologies]. Passionate about [mission/vision]. Currently focused on [current goals/projects]."

## QUALITY STANDARDS:
- Extract data with 100% accuracy from the source
- Ensure descriptions are professional yet engaging
- Use industry-appropriate language and terminology
- Make content ATS-friendly with relevant keywords
- Ensure all content is authentic and based on the actual LinkedIn data provided
- Avoid generic phrases; make it personal and specific to the individual
- Maintain professional tone while being compelling

## OUTPUT FORMAT:
Return the data in the exact structure specified by the LinkedInBasicInfo class:
- full_name: [Exact name from profile]
- location: [Current location from profile]  
- description: [Enhanced professional headline]
- about: [Enhanced professional summary]
- current_company: [Current company name]

Remember: Base all enhancements on the actual LinkedIn data provided. Do not fabricate information, but present existing information in the most compelling and professional way possible.
"""

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": input_question}
    ],
    response_format=personal_info_data,
    )

    analysis_response = completion.choices[0].message
    total_tokens = completion.usage.total_tokens
    if hasattr(analysis_response, 'refusal') and analysis_response.refusal:
        print(f"Model refused to respond: {analysis_response.refusal}")
        return None, total_tokens
    else:
        parsed_data = personal_info_data(personal_info=analysis_response.parsed.personal_info)
        return parsed_data, total_tokens