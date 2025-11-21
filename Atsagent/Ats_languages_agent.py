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
# openai_api_key = os.getenv("OPENAI_API_KEY")

class Languages(BaseModel):
    Language: str
    Proficiency: str


class Step(BaseModel):
    Languages: list[Languages]


class Languages_data(BaseModel):
    steps: list[Step]


async def analyze_languages(input_question):
    """
    Analyzes and extracts language proficiency information from multiple data sources.
    
    Args:
        input_question (str): Combined language data from multiple sources including:
                             - Resume languages data
                             - LinkedIn languages data  
                             - Portfolio languages information
    
    Returns:
        tuple: (Languages_data object with accurate proficiency levels, total_tokens_used)
    """

    prompt_template = """ You are an expert language proficiency analyst and multilingual specialist. You will receive language data from multiple sources including Resume, LinkedIn, and Portfolio. Your task is to analyze, cross-reference, and create accurate, comprehensive language proficiency entries.

        **Data Processing Instructions:**
        - Analyze all provided sources to get a complete picture of language proficiencies
        - Cross-reference information across sources to ensure accuracy and completeness
        - If data conflicts between sources, prioritize the most detailed and realistic assessment
        - Merge related language information from different sources
        - Fill gaps using information from any available source
        - Ensure language proficiency levels are accurately and consistently represented

        **Languages Extraction Requirements:**
        For each language, extract and structure:

        1. **Language name**: 
           - Use the standard, full language name (e.g., "English", "Spanish", "Mandarin Chinese", "French")
           - Avoid abbreviations or informal names
           - Cross-reference between resume and LinkedIn for accuracy
           - Include all languages mentioned across any source

        2. **Proficiency level**: 
           - Use standardized proficiency levels: "Native", "Fluent", "Advanced", "Intermediate", "Basic", "Beginner"
           - Cross-reference proficiency claims across sources for realistic assessment
           - If specific certifications or test scores are mentioned (TOEFL, IELTS, etc.), factor them into proficiency determination
           - Consider context clues from education, work experience, or portfolio that might indicate language use
           - Be conservative and realistic in proficiency assessment - avoid overestimating
           - Prioritize self-reported levels but adjust if they seem unrealistic based on other evidence

        **Proficiency Level Guidelines:**
        - **Native**: Mother tongue or equivalent native-level fluency
        - **Fluent**: Can communicate effectively in all situations, near-native proficiency
        - **Advanced**: High proficiency in most contexts, minor limitations in specialized areas
        - **Intermediate**: Good working proficiency, can handle most everyday situations
        - **Basic**: Limited working proficiency, can handle simple communications
        - **Beginner**: Elementary proficiency, very limited communication ability

        **Quality Standards:**
        - Ensure consistency in language names and proficiency terminology
        - Include only languages where there is clear evidence of some proficiency level
        - Avoid listing languages without any proficiency indication
        - Maintain professional and accurate representation of multilingual abilities
        - Consider cultural and professional context when assessing proficiency claims

        """

    # Get the async client
    client = await get_async_client()
    
    completion = await client.beta.chat.completions.parse(
    model="gpt-5.1",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": input_question}
    ],
    response_format=Languages_data,
    )

    analysis_response = completion.choices[0].message
    total_tokens = completion.usage.total_tokens
    if hasattr(analysis_response, 'refusal') and analysis_response.refusal:
        print(f"Model refused to respond: {analysis_response.refusal}")
        return None, total_tokens
    else:
        parsed_data = Languages_data(steps=analysis_response.parsed.steps)
        return parsed_data, total_tokens