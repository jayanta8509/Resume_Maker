import os
import requests
import time
from dotenv import load_dotenv
load_dotenv()
bright_data_api_key = os.getenv("BRIGHT_DATA_API_KEY")




def trigger_dataset(url):
	api_url = "https://api.brightdata.com/datasets/v3/trigger"
	headers = {
		"Authorization": f"Bearer {bright_data_api_key}",
		"Content-Type": "application/json",
	}
	params = {
		"dataset_id": "gd_l1viktl72bvl7bjuj0",
		"include_errors": "true",
	}
	data = [
		{"url":url},
	]

	response = requests.post(api_url, headers=headers, params=params, json=data)
	return response.json().get("snapshot_id")


def get_dataset_progress(snapshot_id):
	url = f"https://api.brightdata.com/datasets/v3/progress/{snapshot_id}"
	headers = {
		"Authorization": f"Bearer {bright_data_api_key}",
	}

	response = requests.get(url, headers=headers)
	return response.json().get("status")


def get_dataset_snapshot(url):

	tigger_id = trigger_dataset(url)
	progress = get_dataset_progress(tigger_id)

	while progress != "ready":
		time.sleep(30)
		print(f"Snapshot is not ready, waiting for {progress} seconds")
		progress = get_dataset_progress(tigger_id)

	url = f"https://api.brightdata.com/datasets/v3/snapshot/{tigger_id}"
	headers = {
		"Authorization": f"Bearer {bright_data_api_key}",
	}
	params = {
		"format": "json",
	}

	response = requests.get(url, headers=headers, params=params)
	return response.json()


# Alternative function with cleaner output format
def extract_linkedin_profile_clean(json_data):
    """
    Extract LinkedIn profile data and return in a clean format (excluding URLs).
    """
    if isinstance(json_data, list) and len(json_data) > 0:
        data = json_data[0]
    else:
        data = json_data
    
    # Function to remove URL fields from dictionaries
    def clean_urls(item):
        if isinstance(item, dict):
            return {k: clean_urls(v) for k, v in item.items() if not k.endswith('_url') and k not in ['url', 'credential_url', 'company_logo_url', 'institute_logo_url', 'link']}
        elif isinstance(item, list):
            return [clean_urls(i) for i in item]
        else:
            return item
    
    return {
        "Basic Information": {
            "name": data.get("name", ""),
            "location": f"{data.get('city', '')}, {data.get('country_code', '')}" if data.get('city') else data.get('location', "")
        },
        "Professional": {
            "position": data.get("position", ""),
            "about": data.get("about", "")
        },
        "Experience": {
            "current_company": clean_urls(data.get("current_company", {})),
            "experience": clean_urls(data.get("experience", []))
        },
        "Education": {
            "education": clean_urls(data.get("education", [])),
            "certifications": clean_urls(data.get("certifications", []))
        },
        "Projects": clean_urls(data.get("projects", [])),
        "Languages": data.get("languages", [])
    }

#Alternative function with cleaner output format

def extract_linkedin_profile_data(json_data):
    """
    Extract specific fields from LinkedIn profile JSON data (excluding URLs).
    
    Args:
        json_data (dict or list): The LinkedIn profile JSON data
    
    Returns:
        dict: Formatted profile data with the requested sections
    """
    # Handle case where json_data is a list with one item
    if isinstance(json_data, list) and len(json_data) > 0:
        profile_data = json_data[0]
    else:
        profile_data = json_data
    
    # Function to remove URL fields from dictionaries
    def clean_urls(data):
        if isinstance(data, dict):
            return {k: clean_urls(v) for k, v in data.items() if not k.endswith('_url') and k not in ['url', 'credential_url', 'company_logo_url', 'institute_logo_url', 'link']}
        elif isinstance(data, list):
            return [clean_urls(item) for item in data]
        else:
            return data
    
    # Extract and format the data
    extracted_data = {
        "Basic_Information": {
            "name": profile_data.get("name", ""),
            "location": f"{profile_data.get('city', '')}, {profile_data.get('country_code', '')}" if profile_data.get('city') else profile_data.get('location', "")
        },
        "Professional": {
            "position": profile_data.get("position", ""),
            "about": profile_data.get("about", "")
        },
        "Experience": {
            "current_company": clean_urls(profile_data.get("current_company", {})),
            "experience": clean_urls(profile_data.get("experience", []))
        },
        "Education": {
            "education": clean_urls(profile_data.get("education", [])),
            "certifications": clean_urls(profile_data.get("certifications", []))
        },
        "Projects": clean_urls(profile_data.get("projects", [])),
        "Languages": profile_data.get("languages", [])
    }
    
    return extracted_data






