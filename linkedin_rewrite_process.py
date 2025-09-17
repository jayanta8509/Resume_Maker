import asyncio
import threading
from Scraper.resume_scraper import get_resume_content
from linkedin_rewrite_agent.Personal_info_agent import analyze_personal_info
from linkedin_rewrite_agent.Experience_agent import analyze_experience_info
from linkedin_rewrite_agent.Education_agent import analyze_education_info
from linkedin_rewrite_agent.Skill_agent import analyze_skill_info
from linkedin_rewrite_agent.Language_agent import analyze_language_info



async def linkedin_rewrite_process(resume_path):

    try:
        resume_profile_data = get_resume_content(resume_path)

        (
            (personal_info, personal_info_tokens),
            (experience_info, experience_info_token),
            (education_info, education_info_token),
            (skill_info, skill_info_token),
            (language_info, language_info_token)
        ) = await asyncio.gather(
            analyze_personal_info(resume_profile_data),
            analyze_experience_info(resume_profile_data),
            analyze_education_info(resume_profile_data),
            analyze_skill_info(resume_profile_data),
            analyze_language_info(resume_profile_data)
        )

        # Extract data from the responses
        if personal_info and personal_info.personal_info:
            personal_info_step = personal_info.personal_info  # This is a single object, not a list
            experience_info_step = experience_info.experience_info if experience_info and experience_info.experience_info else []
            education_info_step = education_info.education_info if education_info and education_info.education_info else []
            skill_info_step = skill_info.skill_info if skill_info and skill_info.skill_info else []
            language_info_step = language_info.language_info if language_info and language_info.language_info else []
            total_tokens = personal_info_tokens + experience_info_token + education_info_token + skill_info_token + language_info_token

            return {
                'personal_info': personal_info_step,
                'experience_info': experience_info_step,
                'education_info': education_info_step,
                'skill_info': skill_info_step,
                'language_info': language_info_step,
                'total_tokens': total_tokens
            }
        else:
            return {
                'personal_info': None,
                'experience_info': [],
                'education_info': [],
                'skill_info': [],
                'language_info': [],
                'total_tokens': 0,
                'error': 'No personal info found'
            }
    except Exception as e:
        print(f"Resume processing error: {e}")
        return "Error: " + str(e)


# if __name__ == "__main__":
#     data = asyncio.run(linkedin_rewrite_process("jayanta_linkedin_Profile.pdf"))
#     print(data)






