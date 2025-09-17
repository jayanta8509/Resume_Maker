from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, Field, HttpUrl
from typing import Optional, List
import uvicorn
from datetime import datetime
import os
import aiofiles
from pathlib import Path
from processing import resume_data, process_all_agents
from ats_processing import resume_data as ats_resume_data, process_all_agents as ats_process_all_agents,collect_jd_data
from Agent.ats_agent import analyze_ats
from Agent.ats_with_jd_agent import analyze_ats_with_jd
from Scraper.resume_scraper import get_resume_content
from linkedin_rewrite_process import linkedin_rewrite_process

# Initialize FastAPI app
app = FastAPI(
    title="Resume Maker API",
    description="A FastAPI application for resume generation with cross validation",
    version="1.0.0"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request validation
class ResumeImprovementData(BaseModel):
    github_profile: Optional[HttpUrl] = Field(None, description="GitHub profile URL")
    linkedin_profile: Optional[HttpUrl] = Field(None, description="LinkedIn profile URL")
    portfolio_link: Optional[HttpUrl] = Field(None, description="Portfolio website URL")
    other_link: Optional[HttpUrl] = Field(None, description="Any other relevant link")
    
    @field_validator('github_profile')
    @classmethod
    def validate_github_url(cls, v):
        if v and 'github.com' not in str(v):
            raise ValueError('GitHub profile must be a valid GitHub URL')
        return v
    
    @field_validator('linkedin_profile')
    @classmethod
    def validate_linkedin_url(cls, v):
        if v and 'linkedin.com' not in str(v):
            raise ValueError('LinkedIn profile must be a valid LinkedIn URL')
        return v

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
async def root():
    """Root endpoint with basic API information"""
    return {
        "message": "Welcome to Resume Maker API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "improvement_resume": "/improvement-resume",
            "ats_resume": "/ATS-resume"
        }
    }
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Resume Maker API"
    }

@app.post("/improvement-resume")
async def improve_resume(
    resume_file: UploadFile = File(..., description="Resume file (PDF, DOC, DOCX)"),
    github_profile: Optional[str] = Form(None, description="GitHub profile URL"),
    linkedin_profile_file: Optional[UploadFile] = File(None, description="LinkedIn profile file (PDF, DOC, DOCX)"),
    portfolio_link: Optional[str] = Form(None, description="Portfolio website URL"),
    other_link: Optional[str] = Form(None, description="Any other relevant link")
):
    """Improve resume using provided links and uploaded resume file"""
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.doc', '.docx', '.txt'}
        file_extension = Path(resume_file.filename).suffix.lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file_extension} not supported. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Validate LinkedIn file only if provided
        if linkedin_profile_file and linkedin_profile_file.filename:
            linkedin_file_extension = Path(linkedin_profile_file.filename).suffix.lower()
            if linkedin_file_extension not in allowed_extensions:
                raise HTTPException(
                    status_code=400, 
                    detail=f"LinkedIn file type {linkedin_file_extension} not supported. Allowed types: {', '.join(allowed_extensions)}"
                )
        
        # Validate URLs if provided
        profile_data = {}
        
        if github_profile:
            if 'github.com' not in github_profile:
                raise HTTPException(status_code=400, detail="Invalid GitHub URL")
            profile_data['github_profile'] = github_profile
            
        if portfolio_link:
            profile_data['portfolio_link'] = portfolio_link
            
        if other_link:
            profile_data['other_link'] = other_link
        
        # Save uploaded resume file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{resume_file.filename}"
        file_path = UPLOAD_DIR / safe_filename
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await resume_file.read()
            await f.write(content)

        # Save LinkedIn profile file if provided
        linkedin_file_path = None
        if linkedin_profile_file and linkedin_profile_file.filename:
            linkedin_safe_filename = f"{timestamp}_linkedin_{linkedin_profile_file.filename}"
            linkedin_file_path = UPLOAD_DIR / linkedin_safe_filename
            
            async with aiofiles.open(linkedin_file_path, 'wb') as f:
                linkedin_content = await linkedin_profile_file.read()
                await f.write(linkedin_content)

        # Process resume data from all sources
        Basic_Information, resume_tokens, github_tokens, protflow_tokens, other_link_tokens = await resume_data(
            file_path, linkedin_file_path, github_profile, other_link, portfolio_link
        )
        
        # Process all agents and get comprehensive analysis
        analysis_results = await process_all_agents(
            Basic_Information, resume_tokens, github_tokens, protflow_tokens, other_link_tokens
        )

        # Clean up: delete the uploaded files after processing
        try:
            if file_path.exists():
                file_path.unlink()
                print(f"Successfully deleted uploaded resume file: {file_path}")
        except Exception as delete_error:
            print(f"Warning: Could not delete resume file {file_path}: {delete_error}")
            
        try:
            if linkedin_file_path and linkedin_file_path.exists():
                linkedin_file_path.unlink()
                print(f"Successfully deleted uploaded LinkedIn file: {linkedin_file_path}")
        except Exception as delete_error:
            print(f"Warning: Could not delete LinkedIn file {linkedin_file_path}: {delete_error}")
            # Continue execution even if file deletion fails

        return {
            "status_code": 200,
            "status": "success",
            "message": "Comprehensive resume analysis completed successfully",
            **analysis_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")
    


@app.post("/ATS-resume")
async def ATS_resume(
    resume_file: UploadFile = File(..., description="Resume file (PDF, DOC, DOCX)"),
    github_profile: Optional[str] = Form(None, description="GitHub profile URL"),
    linkedin_profile_file: Optional[UploadFile] = File(None, description="LinkedIn profile file (PDF, DOC, DOCX)"),
    portfolio_link: Optional[str] = Form(None, description="Portfolio website URL"),
    other_link: Optional[str] = Form(None, description="Any other relevant link"),
    job_description: str = Form(..., description="Job description")
):
    """ATS resume using provided links and uploaded resume file"""
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.doc', '.docx', '.txt'}
        file_extension = Path(resume_file.filename).suffix.lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file_extension} not supported. Allowed types: {', '.join(allowed_extensions)}"
            )

        # Validate LinkedIn file only if provided
        if linkedin_profile_file and linkedin_profile_file.filename:
            linkedin_file_extension = Path(linkedin_profile_file.filename).suffix.lower()
            if linkedin_file_extension not in allowed_extensions:
                raise HTTPException(
                    status_code=400, 
                    detail=f"LinkedIn file type {linkedin_file_extension} not supported. Allowed types: {', '.join(allowed_extensions)}"
                )
        
        # Validate job description
        if not job_description:
            raise HTTPException(status_code=400, detail="Job description is required")
        
        # Validate job description length
        # if len(job_description) < 100:
        #     raise HTTPException(status_code=400, detail="Job description must be at least 100 characters long")
        
        # Validate URLs if provided
        profile_data = {}
        
        if github_profile:
            if 'github.com' not in github_profile:
                raise HTTPException(status_code=400, detail="Invalid GitHub URL")
            profile_data['github_profile'] = github_profile
            
        if portfolio_link:
            profile_data['portfolio_link'] = portfolio_link
            
        if other_link:
            profile_data['other_link'] = other_link
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{resume_file.filename}"
        file_path = UPLOAD_DIR / safe_filename
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await resume_file.read()
            await f.write(content)
        
        # Save LinkedIn profile file if provided
        linkedin_file_path = None
        if linkedin_profile_file and linkedin_profile_file.filename:
            linkedin_safe_filename = f"{timestamp}_linkedin_{linkedin_profile_file.filename}"
            linkedin_file_path = UPLOAD_DIR / linkedin_safe_filename
            
            async with aiofiles.open(linkedin_file_path, 'wb') as f:
                linkedin_content = await linkedin_profile_file.read()
                await f.write(linkedin_content)


        

        # Process job description to extract structured JD data
        print("🔍 Processing job description...")
        jd_data = await collect_jd_data(job_description)
        # jd_tokens = jd_data.get('tokens', 0)
        
        # Process resume data from all sources using ATS processing
        print("📄 Processing resume and external sources...")
        Basic_Information, resume_tokens, github_tokens, protflow_tokens, other_link_tokens = await ats_resume_data(
            file_path, linkedin_file_path, github_profile, other_link, portfolio_link
        )
        
        # Process all ATS agents with JD data for optimization
        print("🤖 Running ATS-optimized agent analysis...")
        analysis_results = await ats_process_all_agents(
            Basic_Information, jd_data, resume_tokens, github_tokens, protflow_tokens, other_link_tokens
        )

        # Clean up: delete the uploaded file after processing
        try:
            if file_path.exists():
                file_path.unlink()
                print(f"Successfully deleted uploaded file: {file_path}")
        except Exception as delete_error:
            print(f"Warning: Could not delete file {file_path}: {delete_error}")
            # Continue execution even if file deletion fails
        
        try:
            if linkedin_file_path and linkedin_file_path.exists():
                linkedin_file_path.unlink()
                print(f"Successfully deleted uploaded LinkedIn file: {linkedin_file_path}")
        except Exception as delete_error:
            print(f"Warning: Could not delete LinkedIn file {linkedin_file_path}: {delete_error}")
            # Continue execution even if file deletion fails

        return {
            "status_code": 200,
            "status": "success",
            "message": "ATS-optimized resume analysis completed successfully",
            **analysis_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")
    

@app.post("/ATS-score")
async def ATS_score(
   resume_file: UploadFile = File(..., description="Resume file (PDF, DOC, DOCX)"),
):
    """ATS score using provided resume text"""
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.doc', '.docx', '.txt'}
        file_extension = Path(resume_file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file_extension} not supported. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{resume_file.filename}"
        file_path = UPLOAD_DIR / safe_filename
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await resume_file.read()
            await f.write(content)
        
        resume_text = get_resume_content(file_path)
        ATS_score_float, totat_tokens = await analyze_ats(resume_text)

        # Clean up: delete the uploaded file after processing
        try:
            if file_path.exists():
                file_path.unlink()
                print(f"Successfully deleted uploaded file: {file_path}")
        except Exception as delete_error:
            print(f"Warning: Could not delete file {file_path}: {delete_error}")
            # Continue execution even if file deletion fails

        return {
            "status_code": 200,
            "status": "success",
            "message": "ATS score completed successfully",
            "ATS_score": ATS_score_float,
            "total_tokens": totat_tokens
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")


@app.post("/Linkedin-rewrite")
async def Linkedin_rewrite(
   linkedin_file: UploadFile = File(..., description="Resume file (PDF, DOC, DOCX)"),
):
    """Linkedin rewrite using provided linkedin file"""
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.doc', '.docx', '.txt'}
        file_extension = Path(linkedin_file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file_extension} not supported. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{linkedin_file.filename}"
        file_path = UPLOAD_DIR / safe_filename
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await linkedin_file.read()
            await f.write(content)
        
        linkedin_rewrite_data = await linkedin_rewrite_process(file_path)

        # Clean up: delete the uploaded file after processing
        try:
            if file_path.exists():
                file_path.unlink()
                print(f"Successfully deleted uploaded file: {file_path}")
        except Exception as delete_error:
            print(f"Warning: Could not delete file {file_path}: {delete_error}")
            # Continue execution even if file deletion fails

        return {
            "status_code": 200,
            "status": "success",
            "message": "Linkedin rewrite completed successfully",
            "linkedin_rewrite_data": linkedin_rewrite_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")
        

@app.post("/ATS-score-with-JD")
async def ATS_score_with_JD(
   resume_file: UploadFile = File(..., description="Resume file (PDF, DOC, DOCX)"),
   job_description: str = Form(..., description="Job description")
):
    """ATS score using provided resume text and job description"""
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.doc', '.docx', '.txt'}
        file_extension = Path(resume_file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file_extension} not supported. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{resume_file.filename}"
        file_path = UPLOAD_DIR / safe_filename
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await resume_file.read()
            await f.write(content)
        
        resume_text = get_resume_content(file_path)
        ATS_score_float, totat_tokens = await analyze_ats_with_jd(resume_text, job_description)

        # Clean up: delete the uploaded file after processing
        try:
            if file_path.exists():
                file_path.unlink()
                print(f"Successfully deleted uploaded file: {file_path}")
        except Exception as delete_error:
            print(f"Warning: Could not delete file {file_path}: {delete_error}")
            # Continue execution even if file deletion fails

        return {
            "status_code": 200,
            "status": "success",
            "message": "ATS score completed successfully",
            "ATS_score": ATS_score_float,
            "total_tokens": totat_tokens
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )