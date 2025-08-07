# test2.py - Optimized version
import os
import re
import tiktoken
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import logging
import time
import sys
from pathlib import Path

# Add parent directory to path to import shared_client
sys.path.append(str(Path(__file__).parent.parent))
from shared_client import get_async_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define Pydantic models for structured output
class ProtflowAnalysis(BaseModel):
    summary_of_portfolio: str

class PortfolioProfileData(BaseModel):
    analysis: ProtflowAnalysis

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count the number of tokens in a text string for a specific model.
    
    Args:
        text (str): Text to count tokens for
        model (str): Model name to use for tokenization
        
    Returns:
        int: Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        # Fallback to rough estimate (1 token ≈ 4 characters)
        return len(text) // 4

def chunk_text(text: str, max_tokens: int = 8000, model: str = "gpt-4o-mini") -> List[str]:
    """
    Split text into chunks that fit within token limits.
    
    Args:
        text (str): Text to split into chunks
        max_tokens (int): Maximum tokens per chunk
        model (str): Model name to use for tokenization
        
    Returns:
        List[str]: List of text chunks
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        
        chunks = []
        current_chunk = []
        current_count = 0
        
        for token in tokens:
            current_chunk.append(token)
            current_count += 1
            
            if current_count >= max_tokens:
                chunks.append(encoding.decode(current_chunk))
                current_chunk = []
                current_count = 0
        
        if current_chunk:
            chunks.append(encoding.decode(current_chunk))
            
        return chunks
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        # Fallback to simple character-based chunking
        chunk_size = max_tokens * 4  # Rough estimate
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

async def analyze_portfolio_website(html_data: str, max_retries: int = 3) -> Tuple[Optional[PortfolioProfileData], Optional[int]]:
    """
    Analyze portfolio website HTML content and provide comprehensive summary
    
    Args:
        html_data (str): HTML content from portfolio website
        max_retries (int): Maximum number of retry attempts for API calls
        
    Returns:
        Tuple[Optional[PortfolioProfileData], Optional[int]]: (analysis result, total tokens used)
    """
    # Check if input is valid
    if not html_data or not isinstance(html_data, str):
        logger.error("Invalid HTML data provided")
        return None, None
    
    # Count tokens in the input
    input_tokens = count_tokens(html_data)
    logger.info(f"Input text contains approximately {input_tokens} tokens")
    
    # Define token limits for the model
    max_input_tokens = 8000  # Conservative limit for gpt-4o-mini
    
    # If the content is too large, chunk it and process in parts
    if input_tokens > max_input_tokens:
        logger.info(f"Content exceeds token limit, chunking into smaller parts")
        return await analyze_large_portfolio(html_data, max_retries)
    
    # For smaller content, process directly
    return await analyze_portfolio_direct(html_data, max_retries)

async def analyze_large_portfolio(html_data: str, max_retries: int) -> Tuple[Optional[PortfolioProfileData], Optional[int]]:
    """
    Analyze large portfolio content by chunking and combining results
    
    Args:
        html_data (str): HTML content from portfolio website
        max_retries (int): Maximum number of retry attempts for API calls
        
    Returns:
        Tuple[Optional[PortfolioProfileData], Optional[int]]: (analysis result, total tokens used)
    """
    # Split content into manageable chunks
    chunks = chunk_text(html_data, max_tokens=7000)  # Leave room for prompt and response
    logger.info(f"Split content into {len(chunks)} chunks")
    
    # Analyze each chunk
    chunk_analyses = []
    total_tokens = 0
    
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)}")
        
        # Create a specialized prompt for chunk analysis
        chunk_prompt = f"""
        You are analyzing a chunk of content from a portfolio website. This is chunk {i+1} of {len(chunks)}.
        
        Extract and summarize the key information from this chunk that would be relevant for a comprehensive portfolio analysis.
        Focus on:
        - Professional identity and background
        - Skills and expertise
        - Projects and work
        - Professional experience
        - Personal qualities
        - Portfolio quality
        
        Provide a detailed summary of the relevant information in this chunk:
        """
        
        try:
            # Get the async client
            client = await get_async_client()
            
            completion = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": chunk_prompt},
                    {"role": "user", "content": chunk}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            chunk_analysis = completion.choices[0].message.content
            chunk_analyses.append(chunk_analysis)
            total_tokens += completion.usage.total_tokens
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error processing chunk {i+1}: {e}")
            continue
    
    if not chunk_analyses:
        logger.error("Failed to analyze any chunks")
        return None, None
    
    # Combine the chunk analyses
    combined_analysis = "\n\n".join(chunk_analyses)
    
    # Create a final summary from the combined analyses
    final_prompt = f"""
    You are an expert portfolio website analyst. You have been provided with analyses of {len(chunks)} chunks from a portfolio website.
    
    Combine these analyses into a comprehensive portfolio summary that includes:
    
    1. **Professional Identity & Background**:
       - Name and professional title/role
       - Educational background and qualifications
       - Years of experience and career level
       - Current position or professional status
    
    2. **Skills & Expertise**:
       - Technical skills and programming languages
       - Tools, frameworks, and technologies mentioned
       - Soft skills and professional competencies
       - Certifications or special qualifications
       - Areas of specialization or expertise
    
    3. **Projects & Work Portfolio**:
       - Overview of featured projects and their significance
       - Types of work showcased
       - Complexity and scope of projects
       - Technologies used in projects
       - Notable achievements or project outcomes
    
    4. **Professional Experience**:
       - Work history and career progression
       - Companies worked for and roles held
       - Notable accomplishments and achievements
       - Client testimonials or recommendations (if present)
       - Industry experience and domain knowledge
    
    5. **Personal Qualities & Differentiators**:
       - Unique selling points and what sets them apart
       - Personal interests and passions related to their field
       - Communication style and personality traits evident from the portfolio
       - Professional goals and aspirations mentioned
       - Any awards, recognition, or notable mentions
    
    6. **Portfolio Quality Assessment**:
       - Overall presentation and design quality of the portfolio
       - Completeness and depth of information provided
       - Professional impression and credibility
       - Strengths and standout elements of the portfolio
    
    Create a single comprehensive summary that captures all the important aspects of this portfolio and the professional it represents.
    """
    
    try:
        # Get the async client
        client = await get_async_client()
        
        final_completion = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": final_prompt},
                {"role": "user", "content": combined_analysis}
            ],
            response_format=PortfolioProfileData,
        )
        
        final_response = final_completion.choices[0].message
        total_tokens += final_completion.usage.total_tokens
        
        if hasattr(final_response, 'refusal') and final_response.refusal:
            logger.error(f"Model refused to respond: {final_response.refusal}")
            return None, total_tokens
        else:
            parsed_data = PortfolioProfileData(analysis=final_response.parsed.analysis)
            return parsed_data, total_tokens
            
    except Exception as e:
        logger.error(f"Error creating final summary: {e}")
        return None, total_tokens

async def analyze_portfolio_direct(html_data: str, max_retries: int) -> Tuple[Optional[PortfolioProfileData], Optional[int]]:
    """
    Analyze portfolio content directly without chunking
    
    Args:
        html_data (str): HTML content from portfolio website
        max_retries (int): Maximum number of retry attempts for API calls
        
    Returns:
        Tuple[Optional[PortfolioProfileData], Optional[int]]: (analysis result, total tokens used)
    """
    prompt_template = """You are an expert portfolio website analyst and professional assessor. 
    Analyze the provided HTML content from a portfolio website and create a comprehensive summary.
    
    Your task is to read through the HTML content and create a detailed portfolio summary that includes:
    
    1. **Professional Identity & Background**:
       - Name and professional title/role
       - Educational background and qualifications
       - Years of experience and career level
       - Current position or professional status
    
    2. **Skills & Expertise**:
       - Technical skills and programming languages
       - Tools, frameworks, and technologies mentioned
       - Soft skills and professional competencies
       - Certifications or special qualifications
       - Areas of specialization or expertise
    
    3. **Projects & Work Portfolio**:
       - Overview of featured projects and their significance
       - Types of work showcased (web development, mobile apps, design, etc.)
       - Complexity and scope of projects
       - Technologies used in projects
       - Notable achievements or project outcomes
    
    4. **Professional Experience**:
       - Work history and career progression
       - Companies worked for and roles held
       - Notable accomplishments and achievements
       - Client testimonials or recommendations (if present)
       - Industry experience and domain knowledge
    
    5. **Personal Qualities & Differentiators**:
       - Unique selling points and what sets them apart
       - Personal interests and passions related to their field
       - Communication style and personality traits evident from the portfolio
       - Professional goals and aspirations mentioned
       - Any awards, recognition, or notable mentions
    
    6. **Portfolio Quality Assessment**:
       - Overall presentation and design quality of the portfolio
       - Completeness and depth of information provided
       - Professional impression and credibility
       - Strengths and standout elements of the portfolio
    
    Create a single comprehensive summary that captures all the important aspects of this portfolio and the professional it represents.
    """
    
    for attempt in range(max_retries):
        try:
            # Get the async client
            client = await get_async_client()
            
            completion = await client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt_template},
                    {"role": "user", "content": f"Analyze this portfolio website HTML content and create a comprehensive summary: {html_data}"}
                ],
                response_format=PortfolioProfileData,
            )
            
            analysis_response = completion.choices[0].message
            total_tokens = completion.usage.total_tokens
            
            if hasattr(analysis_response, 'refusal') and analysis_response.refusal:
                logger.error(f"Model refused to respond: {analysis_response.refusal}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                parsed_data = PortfolioProfileData(analysis=analysis_response.parsed.analysis)
                return parsed_data, total_tokens
                
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}: API call failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            continue
    
    logger.error(f"Failed to analyze portfolio after {max_retries} attempts")
    return None, None