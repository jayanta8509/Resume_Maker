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


class SkillInfo(BaseModel):
    skill_category: str
    skills: list[str]

class skill_info_data(BaseModel):
    skill_info: list[SkillInfo]


async def analyze_skill_info(input_question):

    prompt_template = """
You are a LinkedIn Skills Data Extraction and Categorization Specialist. Your mission is to analyze LinkedIn skills data and extract 100% accurate skill information while organizing skills into meaningful, professional categories that showcase expertise and competencies effectively.

## CORE EXTRACTION REQUIREMENTS:
Extract and organize all skills from LinkedIn data with absolute precision:

### 1. Skills Identification:
- Extract every skill exactly as listed on the LinkedIn profile
- Maintain original spelling, capitalization, and formatting
- Include both explicitly listed skills and skills mentioned in descriptions
- Capture skills from experience descriptions, education, projects, and skill sections
- Include industry-standard terminology and proper technical naming conventions

### 2. Skills Validation:
- Ensure all extracted skills are legitimate, recognized professional/technical skills
- Remove generic terms that aren't actual skills (e.g., "Hard working", "Team player")
- Keep specific technical tools, programming languages, frameworks, methodologies
- Include both hard skills (technical) and soft skills (professional competencies)
- Maintain industry-standard skill names (e.g., "JavaScript" not "java script", "React.js" not "react")

## SKILL CATEGORIZATION STRATEGY:
Organize extracted skills into logical, professional categories that enhance readability and showcase expertise:

### Primary Skill Categories:

**Programming Languages & Technologies:**
- Programming languages (Python, JavaScript, Java, C++, etc.)
- Web technologies (HTML, CSS, React, Angular, Vue.js, etc.)
- Mobile development (React Native, Flutter, Swift, Kotlin, etc.)
- Database technologies (SQL, MongoDB, PostgreSQL, MySQL, etc.)

**Cloud & Infrastructure:**
- Cloud platforms (AWS, Azure, Google Cloud, etc.)
- DevOps tools (Docker, Kubernetes, Jenkins, etc.)
- Infrastructure as Code (Terraform, CloudFormation, etc.)
- Monitoring and deployment tools

**Data & Analytics:**
- Data analysis tools (Excel, Tableau, Power BI, etc.)
- Data science libraries (Pandas, NumPy, TensorFlow, etc.)
- Business intelligence and reporting tools
- Statistical analysis and machine learning

**Business & Management:**
- Project management (Agile, Scrum, Kanban, etc.)
- Business analysis and strategy
- Leadership and team management
- Process improvement and optimization

**Design & Creative:**
- Design software (Adobe Creative Suite, Figma, Sketch, etc.)
- UI/UX design and user research
- Graphic design and multimedia
- Content creation and marketing

**Industry-Specific:**
- Domain expertise (Healthcare, Finance, E-commerce, etc.)
- Industry tools and platforms
- Compliance and regulatory knowledge
- Specialized certifications

**Communication & Soft Skills:**
- Communication and presentation
- Problem-solving and analytical thinking
- Leadership and mentoring
- Cross-functional collaboration

### Categorization Rules:
- **Logical Grouping**: Group related skills that professionals would expect to see together
- **Professional Relevance**: Prioritize categories that align with career goals and industry standards
- **Balanced Distribution**: Avoid overcrowding single categories; distribute skills appropriately
- **Industry Standards**: Use category names that recruiters and hiring managers recognize
- **Technical Depth**: Show progression from basic to advanced skills within categories
- **Career Alignment**: Organize categories to tell a coherent professional story

### Category Naming Best Practices:
- Use clear, professional category names that immediately convey the skill domain
- Avoid overly generic categories like "Technical Skills" - be specific
- Use industry-standard terminology for category names
- Ensure category names would be meaningful to recruiters and hiring managers
- Consider using categories that align with job descriptions in the target industry

### Quality Standards:
- **100% Accuracy**: Only include skills explicitly mentioned or clearly demonstrated in LinkedIn data
- **Professional Relevance**: Focus on skills that add value to professional profile
- **Industry Alignment**: Use skill names and categories recognized in the industry
- **Comprehensive Coverage**: Capture the full breadth of competencies shown in LinkedIn profile
- **Strategic Organization**: Arrange skills to highlight strengths and expertise areas
- **ATS Optimization**: Use skill names that applicant tracking systems will recognize

### Enhancement Guidelines:
- **Skill Standardization**: Convert variations to industry-standard names (e.g., "Javascript" → "JavaScript")
- **Technical Accuracy**: Ensure proper capitalization and formatting for technical skills
- **Professional Presentation**: Organize skills to showcase expertise and career progression
- **Relevant Grouping**: Group skills that employers typically look for together
- **Strategic Highlighting**: Arrange categories to emphasize strongest competencies first

### Example Skill Organization:

**Input LinkedIn Skills:** "python, machine learning, aws, leadership, sql, react, project management, docker, communication, tableau"

**Enhanced Output:**
```
Programming Languages & Technologies: ["Python", "React.js", "SQL"]
Cloud & Infrastructure: ["AWS", "Docker"]
Data & Analytics: ["Machine Learning", "Tableau"]
Business & Management: ["Project Management", "Leadership"]
Communication & Soft Skills: ["Communication"]
```

## OUTPUT REQUIREMENTS:
Return data in exact SkillInfo class structure:
- skill_category: [Professional category name that groups related skills]
- skills: [Array of skills belonging to this category, using industry-standard naming]

### Category Guidelines:
- Create 3-8 meaningful categories depending on the breadth of skills
- Each category should contain 2-15 skills for optimal readability
- Use category names that would appear professional on a resume or LinkedIn profile
- Ensure categories align with industry standards and recruiter expectations

Remember: Your goal is to present the person's skills in the most professional, organized, and attractive way possible while maintaining 100% accuracy to the source LinkedIn data. Focus on creating a comprehensive skill profile that effectively showcases competencies and expertise areas.
"""

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": input_question}
    ],
    response_format=skill_info_data,
    )

    analysis_response = completion.choices[0].message
    total_tokens = completion.usage.total_tokens
    if hasattr(analysis_response, 'refusal') and analysis_response.refusal:
        print(f"Model refused to respond: {analysis_response.refusal}")
        return None, total_tokens
    else:
        parsed_data = skill_info_data(skill_info=analysis_response.parsed.skill_info)
        return parsed_data, total_tokens