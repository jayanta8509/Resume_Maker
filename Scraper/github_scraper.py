import requests
from datetime import datetime

def get_github_profile_info(github_url):
    """
    Fetches all publicly available information for a given GitHub profile.

    Args:
        github_url (str): The GitHub profile URL (e.g., "https://github.com/jayanta8509")

    Returns:
        dict: A dictionary containing user profile, repositories, and recent activity.
              Returns None if the profile is not found or an error occurs.
    """
    # Extract username from URL
    username = github_url.strip("/").split("/")[-1]
    if not username:
        return None

    base_url = "https://api.github.com"
    headers = {"Accept": "application/vnd.github.v3+json"}

    try:
        # Fetch user profile
        user_response = requests.get(f"{base_url}/users/{username}", headers=headers)
        if user_response.status_code != 200:
            return None
        user_data = user_response.json()

        # Fetch user repositories
        repos_response = requests.get(f"{base_url}/users/{username}/repos?per_page=100", headers=headers)
        repos_data = repos_response.json() if repos_response.status_code == 200 else []

        # Fetch recent public events (activity)
        events_response = requests.get(f"{base_url}/users/{username}/events/public?per_page=30", headers=headers)
        events_data = events_response.json() if events_response.status_code == 200 else []

        # Process repositories
        repos_info = []
        languages_used = set()
        repo_topics = set()
        total_stars = 0

        for repo in repos_data:
            repo_info = {
                "name": repo.get("name"),
                "full_name": repo.get("full_name"),
                "description": repo.get("description"),
                "language": repo.get("language"),
                "fork": repo.get("fork"),
                "stars": repo.get("stargazers_count"),
                "forks": repo.get("forks_count"),
                "created_at": repo.get("created_at"),
                "updated_at": repo.get("updated_at"),
                "url": repo.get("html_url"),
                "topics": repo.get("topics", [])
            }
            repos_info.append(repo_info)
            if repo.get("language"):
                languages_used.add(repo.get("language"))
            repo_topics.update(repo.get("topics", []))
            total_stars += repo.get("stargazers_count", 0)

        # Process recent events
        recent_events = []
        for event in events_data[:10]:  # Last 10 events
            event_info = {
                "type": event.get("type"),
                "repo": event.get("repo", {}).get("name"),
                "created_at": event.get("created_at"),
                "payload": event.get("payload")
            }
            recent_events.append(event_info)

        # Compile final profile dictionary
        profile_info = {
            "username": user_data.get("login"),
            "name": user_data.get("name"),
            "bio": user_data.get("bio"),
            "location": user_data.get("location"),
            "blog": user_data.get("blog"),
            "company": user_data.get("company"),
            "email": user_data.get("email"),
            "public_repos": user_data.get("public_repos"),
            "public_gists": user_data.get("public_gists"),
            "followers": user_data.get("followers"),
            "following": user_data.get("following"),
            "created_at": user_data.get("created_at"),
            "updated_at": user_data.get("updated_at"),
            "avatar_url": user_data.get("avatar_url"),
            "html_url": user_data.get("html_url"),
            "total_stars": total_stars,
            "languages_used": sorted(languages_used),
            "repo_topics": sorted(repo_topics),
            "repositories": repos_info,
            "recent_events": recent_events,
            "api_note": (
                "Contribution graph data is not available via the API. "
                "For detailed contributions, see the profile's contribution graph on GitHub."
            )
        }

        return profile_info

    except Exception as e:
        print(f"Error fetching GitHub profile info: {e}")
        return None


# # Example usage
# if __name__ == "__main__":
#     profile_url = "https://github.com/jayanta8509"
#     profile_data = get_github_profile_info(profile_url)
#     if profile_data:
#         import pprint
#         pp = pprint.PrettyPrinter(indent=2)
#         pp.pprint(profile_data)
#     else:
#         print("Profile not found or error occurred.")