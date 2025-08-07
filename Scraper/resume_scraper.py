from markitdown import MarkItDown
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

def get_resume_content(resume_path):
    md = MarkItDown(enable_plugins=False) 
    result = md.convert(resume_path)
    text_content = result.text_content
    if text_content == "":
        md = MarkItDown(llm_client=client, llm_model="gpt-4o-mini")
        result = md.convert(resume_path)
        text_content = result.text_content
        return text_content
    else:
        return text_content

# print(get_resume_content("../Ujjwal_resume.pdf"))










