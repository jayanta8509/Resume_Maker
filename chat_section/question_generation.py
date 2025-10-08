import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional

class QuestionGenerator:
    def __init__(self, vector_db):
        """
        Initialize Question Generator with OpenAI client and Vector Database
        
        Args:
            vector_db: Instance of FAISSVectorDB class
        """
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"  # You can change to gpt-4 if needed
        self.vector_db = vector_db
    
    def _get_user_data(self, user_id: str) -> Optional[Dict[str, str]]:
        """
        Retrieve user data from vector database
        
        Args:
            user_id: User's unique identifier
            
        Returns:
            Dictionary with resume_data and linkedin_data or None
        """
        user_data = self.vector_db.retrieve_user_data(user_id)
        if not user_data:
            print(f"User {user_id} not found in database")
            return None
        return user_data
    
    def _parse_work_experience(self, resume_data: str, linkedin_data: str) -> List[Dict[str, str]]:
        """
        Parse work experience from resume and LinkedIn data
        
        Args:
            resume_data: Resume text
            linkedin_data: LinkedIn profile text
            
        Returns:
            List of dictionaries containing company and role information
        """
        combined_text = f"{resume_data}\n\n{linkedin_data}"
        
        prompt = f"""
        Analyze the following resume and LinkedIn data and extract all work experiences.
        
        Data:
        {combined_text}
        
        Return the work experiences in the following format (one per line):
        COMPANY: [company name] | ROLE: [job title] | DURATION: [time period]
        
        If you can't find specific information, use "Not specified".
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at parsing professional work experience data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            experiences = []
            content = response.choices[0].message.content
            
            # Parse the response
            for line in content.strip().split('\n'):
                if 'COMPANY:' in line:
                    parts = line.split('|')
                    company = parts[0].replace('COMPANY:', '').strip()
                    role = parts[1].replace('ROLE:', '').strip() if len(parts) > 1 else "Not specified"
                    duration = parts[2].replace('DURATION:', '').strip() if len(parts) > 2 else "Not specified"
                    
                    experiences.append({
                        'company': company,
                        'role': role,
                        'duration': duration
                    })
            
            return experiences
            
        except Exception as e:
            print(f"Error parsing work experience: {str(e)}")
            return []
    
    def generate_questions_for_experience(
        self, 
        user_id: str,
        company_name: Optional[str] = None,
        questions_per_company: int = 1
    ) -> Dict[str, List[str]]:
        """
        Generate comprehensive questions to improve experience descriptions
        
        Args:
            user_id: User's unique identifier
            company_name: Optional - specific company to generate questions for
            questions_per_company: Number of questions per company (default: 1)
            
        Returns:
            Dictionary with company names as keys and list of questions as values
        """
        # Retrieve user data from vector database
        user_data = self._get_user_data(user_id)
        if not user_data:
            return {}
        
        resume_data = user_data['resume_data']
        linkedin_data = user_data['linkedin_data']
        
        # Parse work experiences
        experiences = self._parse_work_experience(resume_data, linkedin_data)
        
        if not experiences:
            print("No work experiences found in the data")
            return {}
        
        # Filter by company if specified
        if company_name:
            experiences = [exp for exp in experiences if company_name.lower() in exp['company'].lower()]
            if not experiences:
                print(f"No experience found for company: {company_name}")
                return {}
        
        all_questions = {}
        total_companies = len(experiences)
        
        print(f"\nFound {total_companies} companies. Generating {questions_per_company} question(s) per company...\n")
        
        # Generate questions for each company
        for experience in experiences:
            company = experience['company']
            role = experience['role']
            duration = experience['duration']
            
            print(f"Generating {questions_per_company} question(s) for {company}...")
            
            prompt = f"""
            Generate exactly {questions_per_company} comprehensive and insightful question(s) to help improve the work experience description for:
            
            Company: {company}
            Role: {role}
            Duration: {duration}
            
            Current Resume/LinkedIn Data:
            {resume_data[:500]}...
            
            Generate EXACTLY {questions_per_company} detailed question(s) that will help the user write a better, more impactful experience description. 
            
            Categories to cover:
            1. Quantifiable achievements and metrics (revenue, users, performance improvements)
            2. Technical skills and tools used
            3. Leadership and team collaboration
            4. Problem-solving and challenges overcome
            5. Impact on business/product/team
            6. Specific projects and initiatives
            7. Recognition and accomplishments
            
            Make the question specific, actionable, and focused on extracting concrete details.
            Format: Return ONLY {questions_per_company} question(s), one per line, numbered from 1 to {questions_per_company}.
            """
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert career coach and resume writer who helps professionals articulate their achievements effectively."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                
                questions_text = response.choices[0].message.content
                
                # Parse questions
                questions = []
                for line in questions_text.strip().split('\n'):
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                        # Remove numbering and bullets
                        question = re.sub(r'^[\d\.\)\-•]+\s*', '', line)
                        if question:
                            questions.append(question)
                
                # Ensure we have exactly the requested number of questions
                if len(questions) > questions_per_company:
                    questions = questions[:questions_per_company]
                elif len(questions) < questions_per_company:
                    print(f"Warning: Only generated {len(questions)} questions for {company}")
                
                all_questions[company] = questions
                
            except Exception as e:
                print(f"Error generating questions for {company}: {str(e)}")
                all_questions[company] = []
        
        return all_questions
