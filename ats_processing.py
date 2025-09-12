"""
Resume Processing Module

This module contains all agent functions for comprehensive resume analysis.
Each function analyzes specific aspects of resume data from multiple sources.
"""
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx
from Atsagent.Ats_basic_Information_agent import analyze_basic_information
from Atsagent.Ats_experience_agent import analyze_experience
from Atsagent.Ats_education_agent import analyze_education
from Atsagent.Ats_skills_agent import analyze_skills
from Atsagent.Ats_languages_agent import analyze_languages
from Atsagent.Ats_projects_agent import analyze_projects
from Atsagent.Ats_certifications_agent import analyze_certifications
from Atsagent.Ats_achievements_agent import analyze_achievements

# from Scraper.linkedin_scraper import get_dataset_snapshot, extract_linkedin_profile_clean

#linkedin_agent
from linkedin_agent.LinkedIn_Basic_Info_position_agent import analyze_basic_info_position
from linkedin_agent.LinkedIn_experience_agent import linkedin_analyze_experience
from linkedin_agent.LinkedIn_eduction_agent import analyze_linkedin_education
from linkedin_agent.LinkedIn_certification_language_agent import analyze_linkedin_certification_language
from linkedin_agent.LinkedIn_project import analyze_linkedin_projects

from Scraper.github_scraper import get_github_profile_info
from Agent.github_agent import analyze_github_profile
from Scraper.protflow_other_link import get_portfolio_content
from Agent.protflow_agent import analyze_portfolio_website
from Scraper.resume_scraper import get_resume_content
from Agent.resume_agent import analyze_resume
from Agent.jd_agent import analyze_jd
from Agent.resume_experince_agent  import analyze_resume_Experience


async def collect_linkedin_data(linkedin_profile_data):
    try:
        linkedin_profile_data = get_resume_content(linkedin_profile_data)
        # linkedin_profile_data_clean = extract_linkedin_profile_clean(linkedin_profile_data)

        (
            (linkedin_profile_basic_info_data, linkedin_profile_basic_info_tokens),
            (linkedin_profile_experience_data, linkedin_profile_experience_tokens),
            (linkedin_profile_education_data, linkedin_profile_education_tokens),
            (linkedin_profile_certification_language_data, linkedin_profile_certification_language_tokens),
            (linkedin_profile_projects_data, linkedin_profile_projects_tokens)

        ) =  await asyncio.gather(
            analyze_basic_info_position(linkedin_profile_data),
            linkedin_analyze_experience(linkedin_profile_data),
            analyze_linkedin_education(linkedin_profile_data),
            analyze_linkedin_certification_language(linkedin_profile_data),
            analyze_linkedin_projects(linkedin_profile_data)
        )
        return {
            'basic_information': linkedin_profile_basic_info_data.basic_info if linkedin_profile_basic_info_data else None,
            'professional_summary': linkedin_profile_basic_info_data.basic_info.position if linkedin_profile_basic_info_data and linkedin_profile_basic_info_data.basic_info else "",
            'experience': linkedin_profile_experience_data.data.experience if linkedin_profile_experience_data and linkedin_profile_experience_data.data else [],
            'education': (linkedin_profile_education_data.data.education if linkedin_profile_education_data and linkedin_profile_education_data.data else []) + (linkedin_profile_certification_language_data.data.certifications if linkedin_profile_certification_language_data and linkedin_profile_certification_language_data.data else []),
            'projects': linkedin_profile_projects_data.data.projects if linkedin_profile_projects_data and linkedin_profile_projects_data.data else [],
            'languages': linkedin_profile_certification_language_data.data.languages if linkedin_profile_certification_language_data and linkedin_profile_certification_language_data.data else [],
            'tokens': linkedin_profile_basic_info_tokens + linkedin_profile_experience_tokens + linkedin_profile_education_tokens + linkedin_profile_certification_language_tokens + linkedin_profile_projects_tokens,
            'error': None
        }
    except Exception as e:
        print(f"LinkedIn processing error: {e}")
        return {
            'basic_information': None,
            'professional_summary': None,
            'experience': None,
            'education': None,
            'projects': None,
            'languages': None,
            'error': str(e)
        }


async def collect_github_data(github_profile_link):
    """
    Collect GitHub profile data in a separate thread.
    Note: This function is async because it calls analyze_github_profile which is async.
    
    Args:
        github_profile_link: GitHub profile URL
    
    Returns:
        dict: Dictionary containing GitHub data or empty values if error
    """
    try:
        github_profile_data = get_github_profile_info(github_profile_link)
        github_profile_data_clean, github_tokens = await analyze_github_profile(github_profile_data)
        
        # Extract data from the analysis object
        if github_profile_data_clean and github_profile_data_clean.analysis:
            return {
                'overall_analysis': github_profile_data_clean.analysis.overall_analysis,
                'summary_repositories': github_profile_data_clean.analysis.summary_of_all_repositories,
                'skills': github_profile_data_clean.analysis.skills,
                'tokens': github_tokens,
                'error': None
            }
        else:
            return {
                'overall_analysis': "",
                'summary_repositories': "",
                'skills': [],
                'tokens': 0,
                'error': None
            }
    except Exception as e:
        print(f"GitHub processing error: {e}")
        return {
            'overall_analysis': "",
            'summary_repositories': "",
            'skills': [],
            'tokens': 0,
            'error': str(e)
        }


async def collect_portfolio_data(protflow_profile_link):
    """
    Collect portfolio website data in a separate thread.
    Note: This function is async because it calls analyze_portfolio_website which is async.
    
    Args:
        protflow_profile_link: Portfolio website URL
    
    Returns:
        dict: Dictionary containing portfolio data or empty values if error
    """
    try:
        protflow_profile_data = get_portfolio_content(protflow_profile_link)
        protflow_profile_data_clean, protflow_tokens = await analyze_portfolio_website(protflow_profile_data)
        
        if protflow_profile_data_clean and protflow_profile_data_clean.analysis:
            return {
                'summary': protflow_profile_data_clean.analysis.summary_of_portfolio,
                'tokens': protflow_tokens,
                'error': None
            }
        else:
            return {
                'summary': "",
                'tokens': 0,
                'error': None
            }
    except Exception as e:
        print(f"Portfolio processing error: {e}")
        return {
            'summary': "",
            'tokens': 0,
            'error': str(e)
        }


async def collect_other_link_data(other_link):
    """
    Collect other link data in a separate thread.
    Note: This function is async because it calls analyze_portfolio_website which is async.
    
    Args:
        other_link: Other relevant link URL
    
    Returns:
        dict: Dictionary containing other link data or empty values if error
    """
    try:
        other_link_data = get_portfolio_content(other_link)
        other_link_data_clean, other_link_tokens = await analyze_portfolio_website(other_link_data)
        
        if other_link_data_clean and other_link_data_clean.analysis:
            return {
                'summary': other_link_data_clean.analysis.summary_of_portfolio,
                'tokens': other_link_tokens,
                'error': None
            }
        else:
            return {
                'summary': "",
                'tokens': 0,
                'error': None
            }
    except Exception as e:
        print(f"Other link processing error: {e}")
        return {
            'summary': "",
            'tokens': 0,
            'error': str(e)
        }
    

async def collect_jd_data(job_description):
    """
    Collect job description data in a separate thread.
    """
    try:
        jd_data, jd_tokens = await analyze_jd(job_description)
        if jd_data and jd_data.analysis:
            jd_step = jd_data.analysis
            return {
                'job_title': jd_step.job_title,
                'hard_skills': jd_step.hard_skills,
                'soft_skills': jd_step.soft_skills,
                'tools_and_technologies': jd_step.tools_and_technologies,
                'responsibilities': jd_step.responsibilities,
                'required_qualifications': jd_step.required_qualifications,
                'preferred_qualifications': jd_step.preferred_qualifications,
                'action_verbs': jd_step.action_verbs,
                'tokens': jd_tokens,
                'error': None
            }
        else:
            # Fallback values if no data is returned
            return {
                'job_title': "",
                'hard_skills': [],
                'soft_skills': [],
                'tools_and_technologies': [],
                'responsibilities': [],
                'required_qualifications': [],
                'preferred_qualifications': [],
                'action_verbs': [],
                'tokens': 0,
                'error': None
            }
    except Exception as e:
        print(f"Job description processing error: {e}")
        return {
            'job_title': "",
            'hard_skills': [],
            'soft_skills': [],
            'tools_and_technologies': [],
            'responsibilities': [],
            'required_qualifications': [],
            'preferred_qualifications': [],
            'action_verbs': [],
            'tokens': 0,
            'error': str(e)
        }



async def collect_resume_data(resume_path):
    """
    Collect resume data - this always runs separately as it's required.
    
    Args:
        resume_path: Path to the uploaded resume file
    
    Returns:
        dict: Dictionary containing resume data and tokens
    """
    try:
        resume_profile_data = get_resume_content(resume_path)
        # resume_profile_data_clean, resume_tokens = await analyze_resume(resume_profile_data)
        # resume_experience_data, resume_experience_total_token = await analyze_resume_Experience(resume_profile_data)

        (
            (resume_profile_data_clean, resume_tokens),
            (resume_experience_data, resume_experience_total_token)
        ) = await asyncio.gather(
            analyze_resume(resume_profile_data),
            analyze_resume_Experience(resume_profile_data)
        )

        # Extract data from the first step (assuming there's at least one step)
        if resume_profile_data_clean and resume_profile_data_clean.steps:
            resume_step = resume_profile_data_clean.steps[0]
            return {
                'suggested_role': resume_step.SuggestedRole,
                'full_name': resume_step.CandidateFullName,
                'email': resume_step.EmailAddress,
                'phone': resume_step.PhoneNumber,
                'professional_title': resume_step.ProfessionalTitle,
                'summary': resume_step.Summary,
                'experience_in_years': resume_step.YearsOfExperienceRequired,
                'experience': resume_experience_data,
                'education': resume_step.Education,
                'languages': resume_step.Languages,
                'projects': resume_step.Projects,
                'certifications': resume_step.Certifications,
                'achievements': resume_step.Achievements,
                'skills': resume_step.Skills,
                'tokens': resume_tokens + resume_experience_total_token,
                'error': None
            }
        else:
            # Fallback values if no data is returned
            return {
                'suggested_role': "",
                'full_name': "",
                'email': "",
                'phone': "",
                'professional_title': "",
                'summary': "",
                'experience': [],
                'education': [],
                'languages': [],
                'projects': [],
                'certifications': [],
                'achievements': [],
                'skills': [],
                'tokens': 0,
                'error': None
            }
    except Exception as e:
        print(f"Resume processing error: {e}")
        return {
            'suggested_role': "",
            'full_name': "",
            'email': "",
            'phone': "",
            'professional_title': "",
            'summary': "",
            'experience': [],
            'education': [],
            'languages': [],
            'projects': [],
            'certifications': [],
            'achievements': [],
            'skills': [],
            'tokens': 0,
            'error': str(e)
        }


async def basic_information_agent(Basic_Information, jd_data, resume_tokens=0, github_tokens=0, protflow_tokens=0, other_link_tokens=0):
    """
    Analyze and compare basic information from multiple sources.
    
    Args:
        Basic_Information: Object containing all scraped data
        resume_tokens: Tokens used for resume processing
        github_tokens: Tokens used for GitHub processing
        protflow_tokens: Tokens used for portfolio processing
        other_link_tokens: Tokens used for other links processing
    
    Returns:
        tuple: (analyzed_basic_info, total_tokens)
    """
    basic_info_input = f"""
        Please analyze and compare the following information from multiple sources to create accurate and comprehensive basic information:

        **Resume Data:**
        - Suggested Role: {Basic_Information.Resume_SuggestedRole}
        - Full Name: {Basic_Information.Resume_CandidateFullName}
        - Email: {Basic_Information.Resume_EmailAddress}
        - Phone: {Basic_Information.Resume_PhoneNumber}
        - Professional Title: {Basic_Information.Resume_ProfessionalTitle}
        - Experience in Years: {Basic_Information.Resume_Experience_in_years}
        - Summary: {Basic_Information.Resume_Summary}
        

        **LinkedIn Profile Data:**
        - Basic Information: {Basic_Information.linkedin_basic_information_data}
        - Professional Summary: {Basic_Information.linkedin_Professional_Summary}

        Please analyze these sources, identify any discrepancies, and provide the most accurate and comprehensive basic information with proper suggestions for role suitability.
        """
    # Call the Basic Information agent for analysis
    job_title = jd_data.get('job_title', '')
    basic_info_analysis, basic_info_tokens = await analyze_basic_information(basic_info_input, job_title)
    
    if basic_info_analysis:
        analyzed_basic_info = basic_info_analysis.steps[0] if basic_info_analysis.steps else None
    else:
        analyzed_basic_info = None

    # Calculate total tokens used
    total_tokens = basic_info_tokens + resume_tokens + github_tokens + protflow_tokens + other_link_tokens
    
    return analyzed_basic_info, total_tokens


async def experience_agent(Basic_Information, jd_data, resume_tokens=0, github_tokens=0, protflow_tokens=0, other_link_tokens=0):
    """
    Analyze and compare experience information from multiple sources.
    
    Args:
        Basic_Information: Object containing all scraped data
        resume_tokens: Tokens used for resume processing
        github_tokens: Tokens used for GitHub processing
        protflow_tokens: Tokens used for portfolio processing
        other_link_tokens: Tokens used for other links processing
    
    Returns:
        tuple: (analyzed_experience, total_tokens)
    """
    experience_input = f"""
        Please analyze and compare the following information from multiple sources to create accurate and comprehensive experience information:

        **Resume Experience Data:**
        {Basic_Information.Resume_Experience}

        **LinkedIn Experience Data:**
        {Basic_Information.linkedin_Experience}

        **GitHub Experience Data:**
        {Basic_Information.github_overall_analysis_data}
        {Basic_Information.github_summary_of_all_repositories}

        **Portfolio Experience Data:**
        {Basic_Information.protflow_summary}

        **Other Link Experience Data:**
        {Basic_Information.other_link_summary}

        Please analyze these sources, cross-reference the experience information, and provide accurate, comprehensive, and well-structured experience data.
        """
    
    # Create JD input for experience analysis
    jd_input = {
        'hard_skills': jd_data.get('hard_skills', []),
        'tools_and_technologies': jd_data.get('tools_and_technologies', []),
        'responsibilities': jd_data.get('responsibilities', []),
        'action_verbs': jd_data.get('action_verbs', [])
    }
    experience_analysis, experience_tokens = await analyze_experience(experience_input, jd_input)
    
    if experience_analysis:
        analyzed_experience = experience_analysis.steps[0] if experience_analysis.steps else None
    else:
        analyzed_experience = None

    total_tokens = experience_tokens + resume_tokens + github_tokens + protflow_tokens + other_link_tokens
    return analyzed_experience, total_tokens


async def education_agent(Basic_Information, jd_data, resume_tokens=0, github_tokens=0, protflow_tokens=0, other_link_tokens=0):
    """
    Analyze and compare education information from multiple sources.
    
    Args:
        Basic_Information: Object containing all scraped data
        resume_tokens: Tokens used for resume processing
        github_tokens: Tokens used for GitHub processing
        protflow_tokens: Tokens used for portfolio processing
        other_link_tokens: Tokens used for other links processing
    
    Returns:
        tuple: (analyzed_education, total_tokens)
    """
    education_input = f"""
        Please analyze and compare the following information from multiple sources to create accurate and comprehensive education information:

        **Resume Education Data:**
        {Basic_Information.Resume_Education}

        **LinkedIn Education Data:**
        {Basic_Information.linkedin_Education}

        **Portfolio Education Data:**
        {Basic_Information.protflow_summary}

        **Other Link Education Data:**
        {Basic_Information.other_link_summary}

        Please analyze these sources, cross-reference the education information, and provide accurate, comprehensive, and well-structured education data.
        """
    
    # Create JD input for education analysis
    jd_input = {
        'required_qualifications': jd_data.get('required_qualifications', [])
    }
    education_analysis, education_tokens = await analyze_education(education_input, jd_input)
    
    if education_analysis:
        analyzed_education = education_analysis.steps[0] if education_analysis.steps else None
    else:
        analyzed_education = None

    total_tokens = education_tokens + resume_tokens + github_tokens + protflow_tokens + other_link_tokens
    return analyzed_education, total_tokens


async def skills_agent(Basic_Information, jd_data, resume_tokens=0, github_tokens=0, protflow_tokens=0, other_link_tokens=0):
    """
    Analyze and compare skills information from multiple sources.
    
    Args:
        Basic_Information: Object containing all scraped data
        resume_tokens: Tokens used for resume processing
        github_tokens: Tokens used for GitHub processing
        protflow_tokens: Tokens used for portfolio processing
        other_link_tokens: Tokens used for other links processing
    
    Returns:
        tuple: (analyzed_skills, total_tokens)
    """
    skills_input = f"""
        Please analyze and compare the following information from multiple sources to create accurate and comprehensive skills information:

        **Resume Skills Data:**
        {Basic_Information.Resume_Skills}

        **GitHub Skills Data:**
        {Basic_Information.github_skills_data}

        **Portfolio Skills Data:**
        {Basic_Information.protflow_summary}

        Please analyze these sources, identify technical and soft skills, categorize them appropriately, and provide accurate, comprehensive, and well-structured skills data.
        """
    
    # Create JD input for skills analysis
    jd_input = {
        'hard_skills': jd_data.get('hard_skills', []),
        'soft_skills': jd_data.get('soft_skills', []),
        'tools_and_technologies': jd_data.get('tools_and_technologies', [])
    }
    skills_analysis, skills_tokens = await analyze_skills(skills_input, jd_input)
    
    if skills_analysis:
        analyzed_skills = skills_analysis.steps[0] if skills_analysis.steps else None
    else:
        analyzed_skills = None

    total_tokens = skills_tokens + resume_tokens + github_tokens + protflow_tokens + other_link_tokens
    return analyzed_skills, total_tokens


async def languages_agent(Basic_Information, resume_tokens=0, github_tokens=0, protflow_tokens=0, other_link_tokens=0):
    """
    Analyze and compare languages information from multiple sources.
    
    Args:
        Basic_Information: Object containing all scraped data
        resume_tokens: Tokens used for resume processing
        github_tokens: Tokens used for GitHub processing
        protflow_tokens: Tokens used for portfolio processing
        other_link_tokens: Tokens used for other links processing
    
    Returns:
        tuple: (analyzed_languages, total_tokens)
    """
    languages_input = f"""
        Please analyze and compare the following information from multiple sources to create accurate and comprehensive languages information:

        **Resume Languages Data:**
        {Basic_Information.Resume_Languages}

        **LinkedIn Languages Data:**
        {Basic_Information.linkedin_Languages}

        **Portfolio Languages Information:**
        - Portfolio Summary: {Basic_Information.protflow_summary}

        Please analyze these sources, cross-reference language proficiency levels, and provide accurate, comprehensive, and well-structured languages data.
        """
    
    languages_analysis, languages_tokens = await analyze_languages(languages_input)
    
    if languages_analysis:
        analyzed_languages = languages_analysis.steps[0] if languages_analysis.steps else None
    else:
        analyzed_languages = None

    total_tokens = languages_tokens + resume_tokens + github_tokens + protflow_tokens + other_link_tokens
    return analyzed_languages, total_tokens


async def projects_agent(Basic_Information, jd_data, resume_tokens=0, github_tokens=0, protflow_tokens=0, other_link_tokens=0):
    """
    Analyze and compare projects information from multiple sources.
    
    Args:
        Basic_Information: Object containing all scraped data
        resume_tokens: Tokens used for resume processing
        github_tokens: Tokens used for GitHub processing
        protflow_tokens: Tokens used for portfolio processing
        other_link_tokens: Tokens used for other links processing
    
    Returns:
        tuple: (analyzed_projects, total_tokens)
    """
    projects_input = f"""
        Please analyze and compare the following information from multiple sources to create accurate and comprehensive projects information:

        **Resume Projects Data:**
        {Basic_Information.Resume_Projects}

        **LinkedIn Projects Data:**
        {Basic_Information.linkedin_Projects}

        **GitHub Repository Analysis:**
        - Repository Summary: {Basic_Information.github_summary_of_all_repositories}
        - Overall Analysis: {Basic_Information.github_overall_analysis_data}

        Please analyze these sources, cross-reference project information, identify technologies used, and provide accurate, comprehensive, and well-structured projects data.
        """
    
    # Create JD input for projects analysis
    jd_input = {
        'hard_skills': jd_data.get('hard_skills', []),
        'tools_and_technologies': jd_data.get('tools_and_technologies', []),
        'preferred_qualifications': jd_data.get('preferred_qualifications', [])
    }
    projects_analysis, projects_tokens = await analyze_projects(projects_input, jd_input)
    
    if projects_analysis:
        analyzed_projects = projects_analysis.steps[0] if projects_analysis.steps else None
    else:
        analyzed_projects = None

    total_tokens = projects_tokens + resume_tokens + github_tokens + protflow_tokens + other_link_tokens
    return analyzed_projects, total_tokens


async def certifications_agent(Basic_Information, jd_data, resume_tokens=0, github_tokens=0, protflow_tokens=0, other_link_tokens=0):
    """
    Analyze and compare certifications information from multiple sources.
    
    Args:
        Basic_Information: Object containing all scraped data
        resume_tokens: Tokens used for resume processing
        github_tokens: Tokens used for GitHub processing
        protflow_tokens: Tokens used for portfolio processing
        other_link_tokens: Tokens used for other links processing
    
    Returns:
        tuple: (analyzed_certifications, total_tokens)
    """
    certifications_input = f"""
        Please analyze and compare the following information from multiple sources to create accurate and comprehensive certifications information:

        **Resume Certifications Data:**
        {Basic_Information.Resume_Certifications}

        **LinkedIn Certifications Data:**
        {Basic_Information.linkedin_Education}

        **Portfolio Certifications Information:**
        - Portfolio Summary: {Basic_Information.protflow_summary}

        **Other Sources:**
        - Other Link Summary: {Basic_Information.other_link_summary}

        Please analyze these sources, verify certification details, and provide accurate, comprehensive, and well-structured certifications data.
        """
    
    # Create JD input for certifications analysis
    jd_input = {
        'preferred_qualifications': jd_data.get('preferred_qualifications', []),
        'required_qualifications': jd_data.get('required_qualifications', [])
    }
    certifications_analysis, certifications_tokens = await analyze_certifications(certifications_input, jd_input)
    
    if certifications_analysis:
        analyzed_certifications = certifications_analysis.steps[0] if certifications_analysis.steps else None
    else:
        analyzed_certifications = None

    total_tokens = certifications_tokens + resume_tokens + github_tokens + protflow_tokens + other_link_tokens
    return analyzed_certifications, total_tokens


async def achievements_agent(Basic_Information, jd_data, resume_tokens=0, github_tokens=0, protflow_tokens=0, other_link_tokens=0):
    """
    Analyze and compare achievements information from multiple sources.
    
    Args:
        Basic_Information: Object containing all scraped data
        resume_tokens: Tokens used for resume processing
        github_tokens: Tokens used for GitHub processing
        protflow_tokens: Tokens used for portfolio processing
        other_link_tokens: Tokens used for other links processing
    
    Returns:
        tuple: (analyzed_achievements, total_tokens)
    """
    achievements_input = f"""
        Please analyze and compare the following information from multiple sources to create accurate and comprehensive achievements information:

        **Resume Achievements Data:**
        {Basic_Information.Resume_Achievements}

        **LinkedIn Achievements Data:**
        {Basic_Information.linkedin_Education}

        **Portfolio Achievements Information:**
        - Portfolio Summary: {Basic_Information.protflow_summary}

        **Other Sources:**
        - Other Link Summary: {Basic_Information.other_link_summary}

        Please analyze these sources, identify notable achievements and recognitions, and provide accurate, comprehensive, and well-structured achievements data.
        """
    
    # Create JD input for achievements analysis
    jd_input = {
        'soft_skills': jd_data.get('soft_skills', []),
        'action_verbs': jd_data.get('action_verbs', [])
    }
    achievements_analysis, achievements_tokens = await analyze_achievements(achievements_input, jd_input)
    
    if achievements_analysis:
        analyzed_achievements = achievements_analysis.steps[0] if achievements_analysis.steps else None
    else:
        analyzed_achievements = None

    total_tokens = achievements_tokens + resume_tokens + github_tokens + protflow_tokens + other_link_tokens
    return analyzed_achievements, total_tokens


async def resume_data(resume_path, linkedin_profile_link=None, github_profile_link=None, other_link=None, protflow_profile_link=None):
    """
    Collect and process data from all sources (resume, LinkedIn, GitHub, portfolio, other links) using concurrent processing.
    
    Args:
        resume_path: Path to the uploaded resume file
        linkedin_profile_link: LinkedIn profile URL (optional)
        github_profile_link: GitHub profile URL (optional)
        other_link: Other relevant link URL (optional)
        protflow_profile_link: Portfolio link URL (optional)
    
    Returns:
        tuple: (Basic_Information_object, resume_tokens, github_tokens, protflow_tokens, other_link_tokens)
    """
    import asyncio
    
    # Initialize token counters
    resume_tokens = 0
    github_tokens = 0
    protflow_tokens = 0
    other_link_tokens = 0
    
    # Initialize default values
    linkedin_basic_information_data = None
    linkedin_Professional_Summary = None
    linkedin_Experience = None
    linkedin_Education = None
    linkedin_Projects = None
    linkedin_Languages = None
    
    github_overall_analysis_data = ""
    github_summary_of_all_repositories = ""
    github_skills_data = []
    
    protflow_summary = ""
    other_link_summary = ""
    
    # Create a list of tasks to run concurrently
    concurrent_tasks = []
    
    # Add LinkedIn task if link provided
    # Add LinkedIn task if link provided
    if linkedin_profile_link:
        # LinkedIn is async, so we create a task for concurrent execution
        linkedin_task = collect_linkedin_data(linkedin_profile_link)
        concurrent_tasks.append(('linkedin', linkedin_task))
    
    # Add GitHub task if link provided
    if github_profile_link:
        github_task = collect_github_data(github_profile_link)
        concurrent_tasks.append(('github', github_task))
    
    # Add Portfolio task if link provided
    if protflow_profile_link:
        portfolio_task = collect_portfolio_data(protflow_profile_link)
        concurrent_tasks.append(('portfolio', portfolio_task))
    
    # Add Other link task if link provided
    if other_link:
        other_task = collect_other_link_data(other_link)
        concurrent_tasks.append(('other', other_task))
    
    # Process resume data separately (always required)
    print("Starting resume processing...")
    resume_data_result = await collect_resume_data(resume_path)
    
    # Extract resume data
    Resume_SuggestedRole = resume_data_result['suggested_role']
    Resume_CandidateFullName = resume_data_result['full_name']
    Resume_EmailAddress = resume_data_result['email']
    Resume_PhoneNumber = resume_data_result['phone']
    Resume_ProfessionalTitle = resume_data_result['professional_title']
    Resume_Summary = resume_data_result['summary']
    Resume_Experience = resume_data_result['experience']
    Resume_Experience_in_years = resume_data_result['experience_in_years']
    Resume_Education = resume_data_result['education']
    Resume_Languages = resume_data_result['languages']
    Resume_Projects = resume_data_result['projects']
    Resume_Certifications = resume_data_result['certifications']
    Resume_Achievements = resume_data_result['achievements']
    Resume_Skills = resume_data_result['skills']
    resume_tokens = resume_data_result['tokens']
    
    # Run all concurrent tasks together
    if concurrent_tasks:
        print(f"Starting concurrent processing of {len(concurrent_tasks)} external sources...")
        print(f"🔍 Debug - External sources being processed:")
        for task_name, _ in concurrent_tasks:
            print(f"   • {task_name}")
        
        # Execute all tasks concurrently
        task_names = [task[0] for task in concurrent_tasks]
        task_coroutines = [task[1] for task in concurrent_tasks]
        
        try:
            # Record start time for concurrency measurement
            concurrent_start = asyncio.get_event_loop().time()
            results = await asyncio.gather(*task_coroutines, return_exceptions=True)
            concurrent_end = asyncio.get_event_loop().time()
            
            print(f"⏱️  Concurrent execution took: {concurrent_end - concurrent_start:.2f} seconds")
            
            # Process results
            for i, (task_name, result) in enumerate(zip(task_names, results)):
                if isinstance(result, Exception):
                    print(f"Error in {task_name} processing: {result}")
                    continue
                    
                if task_name == 'linkedin':
                    linkedin_basic_information_data = result['basic_information']
                    linkedin_Professional_Summary = result['professional_summary']
                    linkedin_Experience = result['experience']
                    linkedin_Education = result['education']
                    linkedin_Projects = result['projects']
                    linkedin_Languages = result['languages']
                    print("✓ LinkedIn processing completed")
                    
                elif task_name == 'github':
                    github_overall_analysis_data = result['overall_analysis']
                    github_summary_of_all_repositories = result['summary_repositories']
                    github_skills_data = result['skills']
                    github_tokens = result['tokens']
                    print("✓ GitHub processing completed")
                    
                elif task_name == 'portfolio':
                    protflow_summary = result['summary']
                    protflow_tokens = result['tokens']
                    print("✓ Portfolio processing completed")
                    
                elif task_name == 'other':
                    other_link_summary = result['summary']
                    other_link_tokens = result['tokens']
                    print("✓ Other link processing completed")
        
        except Exception as e:
            print(f"Error in concurrent processing: {e}")
    
    print("All data collection completed!")

    # Create a structured Basic_Information object
    class BasicInformationData:
        def __init__(self):
            # Resume data
            self.Resume_SuggestedRole = Resume_SuggestedRole
            self.Resume_CandidateFullName = Resume_CandidateFullName
            self.Resume_EmailAddress = Resume_EmailAddress
            self.Resume_PhoneNumber = Resume_PhoneNumber
            self.Resume_ProfessionalTitle = Resume_ProfessionalTitle
            self.Resume_Summary = Resume_Summary
            self.Resume_Experience = Resume_Experience
            self.Resume_Education = Resume_Education
            self.Resume_Languages = Resume_Languages
            self.Resume_Projects = Resume_Projects
            self.Resume_Certifications = Resume_Certifications
            self.Resume_Achievements = Resume_Achievements
            self.Resume_Skills = Resume_Skills
            self.Resume_Experience_in_years = Resume_Experience_in_years
            
            # LinkedIn data
            self.linkedin_basic_information_data = linkedin_basic_information_data
            self.linkedin_Professional_Summary = linkedin_Professional_Summary
            self.linkedin_Experience = linkedin_Experience
            self.linkedin_Education = linkedin_Education
            self.linkedin_Projects = linkedin_Projects
            self.linkedin_Languages = linkedin_Languages
            
            # GitHub data
            self.github_overall_analysis_data = github_overall_analysis_data
            self.github_summary_of_all_repositories = github_summary_of_all_repositories
            self.github_skills_data = github_skills_data
            
            # Portfolio and other links data
            self.protflow_summary = protflow_summary
            self.other_link_summary = other_link_summary

    Basic_Information = BasicInformationData()
    
    return Basic_Information, resume_tokens, github_tokens, protflow_tokens, other_link_tokens


async def process_all_agents_with_batching(Basic_Information, jd_data, resume_tokens, github_tokens, protflow_tokens, other_link_tokens, batch_size=4):
    """
    Process all agent functions in batches to manage concurrency and API limits.
    
    Args:
        Basic_Information: Object containing all scraped data
        resume_tokens: Tokens used for resume processing
        github_tokens: Tokens used for GitHub processing
        protflow_tokens: Tokens used for portfolio processing
        other_link_tokens: Tokens used for other links processing
        batch_size: Number of agents to run concurrently in each batch
    
    Returns:
        dict: Dictionary containing all analysis results and metadata
    """
    import asyncio
    
    print("🤖 Starting batched concurrent agent processing...")
    
    # Define all agent functions
    agent_configs = [
        ('basic_information', basic_information_agent),
        ('experience', experience_agent),
        ('education', education_agent),
        ('skills', skills_agent),
        ('languages', languages_agent),
        ('projects', projects_agent),
        ('certifications', certifications_agent),
        ('achievements', achievements_agent)
    ]
    
    # Split into batches
    batches = [agent_configs[i:i + batch_size] for i in range(0, len(agent_configs), batch_size)]
    
    print(f"📊 Processing {len(agent_configs)} agents in {len(batches)} batches of {batch_size}...")
    
    analysis_results = {}
    total_analysis_tokens = 0
    
    for batch_num, batch in enumerate(batches, 1):
        print(f"\n🔄 Processing Batch {batch_num}/{len(batches)} ({len(batch)} agents)...")
        
        # Create tasks for this batch
        batch_tasks = []
        batch_names = []
        
        for agent_name, agent_func in batch:
            if agent_name == 'languages':
                # Languages agent doesn't need JD data
                task = agent_func(Basic_Information, resume_tokens, github_tokens, protflow_tokens, other_link_tokens)
            else:
                task = agent_func(Basic_Information, jd_data, resume_tokens, github_tokens, protflow_tokens, other_link_tokens)
            batch_tasks.append(task)
            batch_names.append(agent_name)
        
        try:
            # Execute this batch concurrently
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process batch results
            for agent_name, result in zip(batch_names, batch_results):
                if isinstance(result, Exception):
                    print(f"❌ Error in {agent_name} agent: {result}")
                    analysis_results[agent_name] = None
                    continue
                
                # Each agent returns (analysis, tokens)
                analysis, tokens = result
                analysis_results[agent_name] = analysis
                total_analysis_tokens += tokens
                print(f"✓ {agent_name.replace('_', ' ').title()} agent completed ({tokens} tokens)")
            
            # Small delay between batches to respect API limits
            if batch_num < len(batches):
                print("⏳ Brief pause between batches...")
                await asyncio.sleep(1)
                
        except Exception as e:
            print(f"❌ Error in batch {batch_num}: {e}")
            # Continue with next batch
            continue
    
    print(f"🎉 All agent processing completed! Total tokens: {total_analysis_tokens}")
    
    return {
        "analysis_results": analysis_results,
        "total_tokens_consumed": total_analysis_tokens
    }


async def process_all_agents(Basic_Information, jd_data, resume_tokens, github_tokens, protflow_tokens, other_link_tokens):
    """
    Process all agent functions using the most appropriate concurrent strategy.
    
    Args:
        Basic_Information: Object containing all scraped data
        resume_tokens: Tokens used for resume processing
        github_tokens: Tokens used for GitHub processing
        protflow_tokens: Tokens used for portfolio processing
        other_link_tokens: Tokens used for other links processing
    
    Returns:
        dict: Dictionary containing all analysis results and metadata
    """
    import asyncio
    
    print("🤖 Starting optimized concurrent agent processing...")
    
    # Create tasks using asyncio.create_task for true concurrency
    print("⚡ Creating concurrent tasks...")
    
    task_1 = asyncio.create_task(basic_information_agent(Basic_Information, jd_data, resume_tokens, github_tokens, protflow_tokens, other_link_tokens))
    task_2 = asyncio.create_task(experience_agent(Basic_Information, jd_data, resume_tokens, github_tokens, protflow_tokens, other_link_tokens))
    task_3 = asyncio.create_task(education_agent(Basic_Information, jd_data, resume_tokens, github_tokens, protflow_tokens, other_link_tokens))
    task_4 = asyncio.create_task(skills_agent(Basic_Information, jd_data, resume_tokens, github_tokens, protflow_tokens, other_link_tokens))
    task_5 = asyncio.create_task(languages_agent(Basic_Information, resume_tokens, github_tokens, protflow_tokens, other_link_tokens))
    task_6 = asyncio.create_task(projects_agent(Basic_Information, jd_data, resume_tokens, github_tokens, protflow_tokens, other_link_tokens))
    task_7 = asyncio.create_task(certifications_agent(Basic_Information, jd_data, resume_tokens, github_tokens, protflow_tokens, other_link_tokens))
    task_8 = asyncio.create_task(achievements_agent(Basic_Information, jd_data, resume_tokens, github_tokens, protflow_tokens, other_link_tokens))
    
    agent_tasks = [
        ('basic_information', task_1),
        ('experience', task_2),
        ('education', task_3),
        ('skills', task_4),
        ('languages', task_5),
        ('projects', task_6),
        ('certifications', task_7),
        ('achievements', task_8)
    ]
    
    print(f"📊 Processing {len(agent_tasks)} agents with task-based concurrency...")
    
    # Extract task names and tasks
    task_names = [task[0] for task in agent_tasks]
    tasks = [task[1] for task in agent_tasks]
    
    try:
        # Execute all agent tasks concurrently using asyncio.gather
        print("⚡ Starting all agents simultaneously...")
        start_time = asyncio.get_event_loop().time()
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time
        
        # Initialize result containers
        analysis_results = {}
        total_analysis_tokens = 0
        
        # Process results
        completed_count = 0
        for task_name, result in zip(task_names, results):
            if isinstance(result, Exception):
                print(f"❌ Error in {task_name} agent: {result}")
                analysis_results[task_name] = None
                continue
            
            # Each agent returns (analysis, tokens)
            analysis, tokens = result
            analysis_results[task_name] = analysis
            total_analysis_tokens += tokens
            completed_count += 1
            print(f"✓ [{completed_count}/8] {task_name.replace('_', ' ').title()} agent completed ({tokens} tokens)")
        
        print(f"🎉 All agent processing completed in {processing_time:.2f}s! Total tokens: {total_analysis_tokens}")
        
        return {
            "analysis_results": analysis_results,
            "total_tokens_consumed": total_analysis_tokens
        }
        
    except Exception as e:
        print(f"❌ Error in concurrent agent processing: {e}")
        print("🔄 Falling back to batched processing...")
        
        # Fallback to batched processing
        return await process_all_agents_with_batching(Basic_Information, jd_data, resume_tokens, github_tokens, protflow_tokens, other_link_tokens)