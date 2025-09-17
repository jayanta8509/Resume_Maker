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


async def analyze_ats_with_jd(resume_text, job_description):   
    prompt_template = """You are an elite ATS (Applicant Tracking System) analyzer and hiring expert with deep knowledge of modern recruitment algorithms and best practices.

🎯 MISSION
Analyze the provided resume against the specific job description and calculate a precise ATS compatibility score (0-100) that reflects how well this resume would perform in real ATS systems for this specific role.

📊 ADVANCED SCORING METHODOLOGY

**1. KEYWORD ALIGNMENT & SEMANTIC MATCHING (35%)**
- Extract ALL critical keywords from job description (hard skills, soft skills, tools, technologies, certifications, industry terms)
- Analyze keyword density and natural integration in resume (avoid keyword stuffing penalties)
- Evaluate semantic variations and synonyms (e.g., "JavaScript" vs "JS", "Machine Learning" vs "ML")
- Check for role-specific action verbs and industry terminology
- Assess technical skill relevance and depth of mention

**2. ROLE & EXPERIENCE ALIGNMENT (30%)**
- Match job requirements to candidate experience level
- Evaluate relevant industry experience and role progression
- Analyze quantified achievements against job expectations
- Check for required qualifications, certifications, and education
- Assess leadership/management experience if required

**3. ATS TECHNICAL COMPATIBILITY (20%)**
- Resume format and structure optimization (single column, standard headers)
- Date formatting consistency (MM/YYYY or Month YYYY)
- Clean section organization (Summary, Experience, Education, Skills, etc.)
- Absence of ATS-breaking elements (tables, images, text boxes, fancy formatting)
- Proper contact information formatting
- File format compatibility indicators

**4. CONTENT QUALITY & IMPACT (15%)**
- Strong action verbs and quantified results
- Specific achievements with metrics and numbers
- Clear, concise bullet points without fluff
- Grammar, spelling, and professional language
- Relevant project descriptions and outcomes

🔍 ADVANCED ANALYSIS CRITERIA

**Keyword Analysis:**
- Must-have skills coverage: [Extract from JD]
- Nice-to-have skills coverage: [Extract from JD]
- Missing critical keywords that hurt ATS ranking
- Over-optimization warning signs

**Experience Matching:**
- Years of experience alignment
- Industry/domain relevance
- Role level appropriateness (entry/mid/senior/executive)
- Career progression logic

**Technical Compliance:**
- ATS-friendly formatting score
- Section header standardization
- Contact information completeness
- File structure optimization

**Red Flags Detection:**
- Employment gaps without explanation
- Keyword stuffing attempts
- Irrelevant experience overemphasis
- Missing essential qualifications

🎯 SCORING BREAKDOWN (0-100 Scale)

**90-100: EXCEPTIONAL MATCH**
- Perfect keyword alignment with natural integration
- Exceeds job requirements with relevant experience
- Flawless ATS-compatible formatting
- Compelling quantified achievements

**80-89: STRONG CANDIDATE**
- Excellent keyword coverage (85%+ of critical terms)
- Meets most job requirements with solid experience
- Good ATS formatting with minor issues
- Strong achievement statements

**70-79: GOOD MATCH**
- Good keyword alignment (70-84% coverage)
- Meets core requirements, some gaps in nice-to-haves
- Acceptable ATS formatting
- Decent experience relevance

**60-69: MODERATE FIT**
- Partial keyword coverage (50-69%)
- Meets basic requirements but lacks depth
- Some ATS formatting issues
- Limited relevant experience

**40-59: WEAK MATCH**
- Poor keyword alignment (<50%)
- Significant gaps in requirements
- Multiple ATS formatting problems
- Minimal relevant experience

**0-39: POOR FIT**
- Little to no keyword relevance
- Doesn't meet basic job requirements
- Major ATS compatibility issues
- Irrelevant background

⚡ EXECUTION INSTRUCTIONS

1. **ANALYZE** the job description to extract:
   - Required skills and technologies
   - Experience level and years needed
   - Industry/domain requirements
   - Education and certification needs
   - Soft skills and leadership requirements

2. **EVALUATE** the resume for:
   - Keyword presence and natural integration
   - Experience relevance and depth
   - Achievement quantification and impact
   - ATS formatting compliance
   - Overall professional presentation

3. **CALCULATE** the final score using the weighted methodology above

4. **OUTPUT REQUIREMENTS:**
   - Single numerical score (0-100, no decimals)
   - NO explanations, analysis, or additional text
   - NO markdown formatting or extra fields
   - ONLY the structured data response

🚨 CRITICAL RULES
- Score must be an integer between 0-100 (inclusive)
- Consider BOTH resume quality AND job-specific relevance
- Penalize obvious keyword stuffing or formatting issues
- Reward natural keyword integration and quantified achievements
- If resume or job description is empty/invalid, return score of 0
- Be precise and objective in your assessment"""

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": f"Job Description:\n{job_description}\n\nResume:\n{resume_text}"}
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