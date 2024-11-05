import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import json
import mplcursors
import helpers_API

# Path to the file containing actor roles
PATH = 'data/MovieSummaries/tvtropes.clusters.txt'

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
    
    # Save the DataFrame to a CSV file
    # df.to_csv('data/actor_sentiment_scores.csv', index=False)
    
    return df

def plot_sentiment_and_popularity(df, remove_zero_sentiment=True):
    # Filter out actors with a sentiment score of 0
    if (remove_zero_sentiment):
        df = df[df['sentiment'] != 0]
    plt.figure(figsize=(12, 8))

    # Calculate the average popularity for the horizontal line
    average_popularity = df['popularity'].mean()

    # Scatter plot with sentiment score on x-axis and popularity on y-axis
    scatter = plt.scatter(df['sentiment'], df['popularity'], color='skyblue', s=50, alpha=0.6, edgecolor='gray')

    # Add vertical and horizontal lines
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)  # Vertical line at x=0
    plt.axhline(y=average_popularity, color='gray', linestyle='--', alpha=0.7)  # Horizontal line at average popularity

    # Set labels and title
    plt.xlabel('Sentiment Score')
    plt.ylabel('Popularity')
    plt.title('Sentiment Score vs. Popularity of Actors')
    plt.xlim(-1, 1)
    plt.ylim(df['popularity'].min(), df['popularity'].max())
    plt.grid(True, linestyle='--', alpha=0.7)

    # Enable hover annotation with mplcursors
    cursor = mplcursors.cursor(scatter, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(df['actor'].iloc[sel.index]))

    plt.show()

# Main function to run the pipeline
df = process_file(PATH)
df['popularity'] = df['actor'].apply(helpers_API.get_actor_popularity)
df.to_csv('data/actor_sentiment_popularity_scores.csv', index=False)
plot_sentiment_and_popularity(df)


