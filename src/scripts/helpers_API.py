import requests

BEARER_TOKEN = "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyMjczZTFjNmRkMjQwNTY2Mzk5MjI2Njc5MTA2OTNmZiIsIm5iZiI6MTczMDgwMDIzOC4yODE4OTEzLCJzdWIiOiI2NzFmNWE2MzI3YmQ1N2Q5MWY2MzFmZjciLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.y9i-dkSKnj-_0R2-pK5Wz-BvyiPXZ5AscPHtCyVZdaY"

def get_actor_popularity(actor_name):
    print(f"Fetching popularity for {actor_name}")
    url = f"https://api.themoviedb.org/3/search/person?query={actor_name}&include_adult=false&language=en-US&page=1"
    headers = {
    "accept": "application/json",
    "Authorization": BEARER_TOKEN
    }
    response = requests.get(url, headers=headers)
    results = response.json()['results']
    score = results[0]['popularity']
    return score
