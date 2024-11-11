import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import json
import helpers_API

# Path to the file containing actor roles
PATH = 'data/CMU_dataset/tvtropes.clusters.txt'

# Download VADER lexicon for sentiment analysis (first-time setup)
nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to parse each line of the file
def parse_line(line):
    role, details = line.strip().split('\t', 1)
    # Pre-process role by replacing underscores with spaces
    role = role.replace('_', ' ')
    details = json.loads(details)
    actor = details['actor']
    return actor, role

# Read and process the file
def process_file(file_path):
    print(f"Processing file: {file_path}")
    actor_roles = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            actor, role = parse_line(line)
            # Analyze sentiment of the role description
            sentiment = sia.polarity_scores(role)['compound']
            
            # Store each role and its sentiment for the actor
            if actor not in actor_roles:
                actor_roles[actor] = []
            actor_roles[actor].append(sentiment)
    
    # Convert to DataFrame for easier manipulation
    actor_sentiments = {actor: sum(scores) / len(scores) for actor, scores in actor_roles.items()}
    df = pd.DataFrame(list(actor_sentiments.items()), columns=['actor', 'sentiment'])
    
    return df

# Main function to run the pipeline
df = process_file(PATH)
df['popularity'] = df['actor'].apply(helpers_API.get_actor_popularity)
df.to_csv('output_data/actor_sentiment_popularity_scores.csv', index=False)


