import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import json

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
    df = pd.DataFrame(list(actor_sentiments.items()), columns=['Actor', 'Sentiment Score'])
    
    # Save the DataFrame to a CSV file
    df.to_csv('data/actor_sentiment_scores.csv', index=False)
    
    return df

# Scatter plot for sentiment scores with names
def plot_sentiment(df):
    # Filter out actors with a sentiment score of 0
    df = df[df['Sentiment Score'] != 0]
    plt.figure(figsize=(12, 8))
    
    # Scatter plot with actors as dots
    plt.scatter(df['Sentiment Score'], df['Actor'], color='skyblue', s=100, alpha=0.6, edgecolor='gray')
    
    # Annotate each dot with the actor's name
    for i in range(len(df)):
        plt.text(df['Sentiment Score'].iloc[i], df['Actor'].iloc[i], df['Actor'].iloc[i],
                 fontsize=9, ha='right' if df['Sentiment Score'].iloc[i] < 0 else 'left')
    
    # Remove y-axis since actor names are displayed next to the dots
    plt.gca().axes.get_yaxis().set_visible(False)
    
    plt.xlabel('Sentiment Score')
    plt.title('Sentiment Score per Actor Based on Roles')
    plt.xlim(-1, 1)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

# Main function to run the pipeline
def main(file_path):
    df = process_file(file_path)
    plot_sentiment(df)

# Run the pipeline with your file
main('data/MovieSummaries/tvtropes.clusters.txt')