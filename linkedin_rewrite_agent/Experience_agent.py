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


class ExperienceDuration(BaseModel):
    start_date: str
    end_date: str

class ExperienceInfo(BaseModel):
    company_name: str
    position: str
    location: str
    skill_set: list[str]
    job_type: str
    duration: ExperienceDuration
    additional_information: str

class experience_info_data(BaseModel):
    experience_info: list[ExperienceInfo]


async def analyze_experience_info(input_question):

    prompt_template = """
You are a LinkedIn Professional Experience Data Extraction and Enhancement Specialist. Your mission is to analyze LinkedIn experience data and extract 100% accurate information while creating compelling additional information that showcases professional excellence, achievements, and career impact.

## CORE EXTRACTION REQUIREMENTS:
Extract the following information from LinkedIn experience data with absolute precision:

### 1. Company Name:
- Extract the exact official company name as displayed on LinkedIn
- Maintain proper capitalization and formatting
- Include full company name, subsidiaries, or divisions as shown
- Use the most complete and official version available

### 2. Position/Job Title:
- Extract the complete position title exactly as shown on LinkedIn
- Include seniority level (Senior, Lead, Principal, Manager, etc.)
- Maintain original formatting and terminology
- Capture any title evolution or promotions within the same company

### 3. Location:
- Extract company/office location exactly as displayed
- Include city, state/province, country as available
- Maintain consistent geographic formatting
- Use format shown on LinkedIn profile

### 4. Job Type:
- Extract employment type from LinkedIn (Full-time, Part-time, Contract, Freelance, Internship, etc.)
- Use exact terminology as displayed on the profile
- If not explicitly stated, infer from context but mark as "Full-time" as default

### 5. Duration:
- Extract start and end dates exactly as shown on LinkedIn
- Use original date format (Month Year, Year only, etc.)
- For current positions, use "Present" or "Current" for end_date
- Maintain consistency with LinkedIn's date display format

### 6. Skill Set:
- Extract all relevant technical and professional skills demonstrated in this role
- Include technologies, programming languages, frameworks, tools, methodologies
- Capture skills mentioned in job descriptions, achievements, or project details
- Use industry-standard terminology and proper capitalization
- Include both hard skills (Python, AWS, React) and soft skills (Leadership, Communication)
- List 5-15 relevant skills based on role complexity and scope

## ADDITIONAL INFORMATION ENHANCEMENT:
Create compelling, professional additional information that transforms raw LinkedIn experience data into powerful career narratives:

### Content Strategy:
Transform basic LinkedIn experience data into impactful professional stories by focusing on:

**Achievement & Impact Highlights:**
- Quantifiable results, metrics, and business outcomes mentioned
- Revenue growth, cost savings, efficiency improvements, or performance gains
- Awards, recognitions, promotions, or special achievements
- Team leadership accomplishments and people management success
- Project completions, launches, or milestone achievements

**Technical Excellence & Innovation:**
- Complex technical projects, system implementations, or architecture designs
- Technology stack expertise and innovative solutions developed
- Process improvements, automation, or optimization initiatives
- Research and development contributions or innovative approaches
- Problem-solving examples with measurable technical impact

**Leadership & Collaboration:**
- Team leadership, mentoring, or cross-functional collaboration
- Stakeholder management and client relationship achievements
- Training, knowledge sharing, or team building initiatives
- Change management or organizational transformation contributions
- Strategic planning participation or decision-making involvement

**Professional Growth & Learning:**
- Skill development, certifications, or training completed during the role
- Expanding responsibilities or scope increases over time
- Industry expertise gained or domain knowledge developed
- Professional network expansion or industry recognition
- Career progression or advancement within the organization

### Writing Excellence Standards:
- **Results-Oriented Language**: Focus on achievements, impact, and measurable outcomes
- **Action-Driven Narratives**: Use powerful action verbs (led, developed, implemented, optimized, delivered)
- **Quantifiable Success**: Include specific metrics, percentages, dollar amounts, timeframes
- **Technical Depth**: Showcase relevant technical expertise and industry knowledge
- **Professional Impact**: Demonstrate value delivered to organization, team, or clients
- **Career Progression**: Show growth, learning, and increasing responsibilities
- **Industry Keywords**: Include relevant terminology for ATS optimization and industry recognition

### Enhancement Examples:

Transform: "Software Developer at Tech Company"
Into: "Spearheaded development of scalable microservices architecture serving 100K+ daily users, resulting in 40% performance improvement and 99.9% uptime. Led cross-functional team of 5 engineers in implementing CI/CD pipelines using Docker and Kubernetes, reducing deployment time from 2 hours to 15 minutes. Mentored junior developers and established coding standards that improved code quality by 35%."

Transform: "Marketing Manager role"
Into: "Drove comprehensive digital marketing strategy that increased lead generation by 150% and revenue by $2.3M annually. Managed $500K advertising budget across multiple channels, achieving 300% ROI through data-driven optimization. Led rebranding initiative that improved brand recognition by 45% and established thought leadership through content strategy reaching 50K+ professionals monthly."

Transform: "Project Manager position"
Into: "Successfully delivered 15+ complex technical projects worth $5M+ total value, consistently meeting deadlines and staying within budget. Implemented Agile methodologies that improved team productivity by 30% and stakeholder satisfaction by 25%. Coordinated cross-functional teams of 20+ members across 3 time zones, ensuring seamless communication and project alignment."

## QUALITY ASSURANCE:
- **100% Accuracy**: Never fabricate information not present in LinkedIn data
- **Authentic Enhancement**: Build compelling narratives based solely on actual LinkedIn content
- **Professional Excellence**: Ensure all content meets highest professional writing standards
- **Consistency**: Maintain uniform tone, style, and quality across all experience entries
- **Value-Driven Focus**: Emphasize achievements, impact, and professional value delivered
- **ATS Optimization**: Include relevant keywords and industry terminology for applicant tracking systems

## OUTPUT REQUIREMENTS:
Return data in exact ExperienceInfo class structure:
- company_name: [Exact company name from LinkedIn]
- position: [Complete job title from LinkedIn]
- location: [Company location from LinkedIn]
- skill_set: [Array of relevant skills extracted and inferred from experience]
- job_type: [Employment type from LinkedIn or inferred]
- duration: {start_date: [Start date], end_date: [End date or "Present"]}
- additional_information: [Enhanced, compelling professional narrative based on LinkedIn experience data]

Remember: Your goal is to present the person's professional experience in the most impactful, attractive, and compelling way possible while maintaining 100% accuracy to the source LinkedIn data. Focus on achievements, growth, and the value they brought to each organization.
"""

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": input_question}
    ],
    response_format=experience_info_data,
    )

    analysis_response = completion.choices[0].message
    total_tokens = completion.usage.total_tokens
    if hasattr(analysis_response, 'refusal') and analysis_response.refusal:
        print(f"Model refused to respond: {analysis_response.refusal}")
        return None, total_tokens
    else:
        parsed_data = experience_info_data(experience_info=analysis_response.parsed.experience_info)
        return parsed_data, total_tokens