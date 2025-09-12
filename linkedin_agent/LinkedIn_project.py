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

class ProjectDuration(BaseModel):
    start_date: str
    end_date: str

class ProjectItem(BaseModel):
    project_name: str
    description: str
    technologies_used: str
    role: str
    duration: ProjectDuration

class LinkedInProjectData(BaseModel):
    projects: list[ProjectItem]

class project_data(BaseModel):
    data: LinkedInProjectData


async def analyze_linkedin_projects(input_question):

    prompt_template = """🚀 **LINKEDIN PROJECT EXTRACTION SPECIALIST** 🚀

You are an elite LinkedIn profile analyzer with specialized expertise in extracting comprehensive project portfolio data from LinkedIn profiles. You will receive structured LinkedIn profile data in JSON format and must extract ALL project information with technical precision and innovation focus.

🎯 **YOUR MISSION**: Extract complete project portfolio from LinkedIn profile data:

🛠️ **PROJECT PORTFOLIO EXTRACTION**:
   📍 **Source Location**: Navigate to "Projects" array in LinkedIn data
   📝 **Data Points to Extract**:
   
   For EACH project found:
   
   📋 **project_name**: 
      - Extract: Full project title/name
      - Examples: "Loan Prediction Using Machine Learning", "Password Strength Checker", "E-commerce Platform", "AI Chatbot"
      - Source: Look for "title" field in project objects
      - Format: Complete project name as listed on LinkedIn
   
   📝 **description**: 
      - Extract: Comprehensive project description, objectives, and outcomes
      - Examples: Detailed project overview, problem solved, features implemented, impact achieved
      - Source: Look for "description" field in project objects
      - Format: Complete project description as provided on LinkedIn
      - Technical Details: Include algorithms, methodologies, frameworks used
      - If missing: Return empty string ""
   
   🔧 **technologies_used**: 
      - Extract: Technical stack, programming languages, frameworks, tools, and technologies
      - Examples: "Python, Flask, HTML, CSS, JavaScript, Scikit-learn, Pandas", "React, Node.js, MongoDB"
      - Source: Parse from description or dedicated technology fields
      - Format: Comma-separated list of technologies
      - Include: Programming languages, frameworks, databases, tools, libraries
      - If missing or unclear: Extract from description context
   
   👨‍💻 **role**: 
      - Extract: Your specific role and contribution in the project
      - Examples: "Full Stack Developer", "Machine Learning Engineer", "Project Lead", "Solo Developer"
      - Source: Look for role information in project description or dedicated role field
      - Format: Clear role designation
      - Default: If unclear, use "Developer" or extract from context
   
   📅 **duration**: 
      - Extract: Project start and end dates or timeline
      - **start_date**: Look for "start_date" field or extract from description
      - **end_date**: Look for "end_date" field or extract from description
      - Examples: start_date: "May 2022", end_date: "Jun 2022"
      - Format: Preserve date format from LinkedIn (usually "Mon YYYY" format)
      - Handle: Single date entries, ongoing projects, unclear timelines

⚡ **EXTRACTION RULES**:
1. **TECHNICAL PRECISION**: Extract project data EXACTLY as it appears in LinkedIn JSON
2. **COMPREHENSIVE COVERAGE**: Process ALL projects found in the data
3. **NO FABRICATION**: If field missing, return empty string "" for that field
4. **PRESERVE INNOVATION**: Maintain original project descriptions and technical details
5. **TECHNOLOGY FOCUS**: Aggressively extract all technical information from descriptions
6. **OUTCOME EMPHASIS**: Capture project achievements and impact when mentioned

🚀 **PERFORMANCE STANDARDS**:
- ✅ 100% accuracy in project field mapping
- ✅ Zero loss of technical project data
- ✅ Complete extraction of all project entries
- ✅ Comprehensive technology stack identification
- ✅ Detailed project description preservation
- ✅ Professional project portfolio integrity

⚠️ **CRITICAL SUCCESS FACTORS**:
- Extract ALL projects listed in LinkedIn profile
- Map project names, descriptions, and technical details accurately
- Identify and extract complete technology stacks from descriptions
- Handle missing date information gracefully
- Preserve technical innovation and project impact narratives
- Return structured project array with all found projects

🎯 **OUTPUT REQUIREMENTS**:
Return projects array populated with ALL projects found in LinkedIn data, structured according to the class requirements.

🔍 **SPECIAL PROJECT HANDLING**:
- Academic projects, personal projects, professional projects
- Open source contributions and GitHub projects
- Hackathon projects and competitions
- Client projects and freelance work
- Research projects and publications

💡 **TECHNICAL INTELLIGENCE**:
- Parse technology mentions from natural language descriptions
- Identify frameworks, libraries, and tools from project context
- Extract programming languages from technical descriptions
- Recognize development methodologies and approaches

🏆 **PROJECT VALUE EXTRACTION**:
- Capture project outcomes and measurable results
- Preserve innovation aspects and unique features
- Maintain technical complexity and challenge descriptions
- Extract collaboration details and team contributions

Extract the complete project portfolio now! 🚀💻"""

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": input_question}
    ],
    response_format=project_data,
    )

    analysis_response = completion.choices[0].message
    total_tokens = completion.usage.total_tokens
    if hasattr(analysis_response, 'refusal') and analysis_response.refusal:
        print(f"Model refused to respond: {analysis_response.refusal}")
        return None, total_tokens
    else:
        parsed_data = project_data(data=analysis_response.parsed.data)
        return parsed_data, total_tokens