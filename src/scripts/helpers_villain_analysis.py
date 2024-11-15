import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
import matplotlib.pyplot as plt
import mplcursors

sia = SentimentIntensityAnalyzer()
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('punkt_tab')

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

# Function to classify a character as 'Villain' or 'Not Villain' based on sentiment score
def classify_villain_via_sentiment(row):
    score = row['character_sentiment_score']
    
    # Define threshold for villain classification
    # Characters with highly negative sentiment scores are classified as 'Villain'
    if score is not None and score < -0.3:  # Adjust this threshold based on observations
        return 'Villain'
    else:
        return 'Not Villain'

def extract_character_context(row):
    character_name_parts = row['character_name'].split()  # Split the character name into parts
    plot_summary = row['plot_summary']
    
    # Split plot summary into sentences
    sentences = nltk.sent_tokenize(plot_summary)
    
    # Filter sentences that mention any part of the character's name
    character_sentences = [
        sentence for sentence in sentences 
        if any(part.lower() in sentence.lower() for part in character_name_parts)
    ]
    
    # Combine relevant sentences back into a shorter summary focused on the character
    return ' '.join(character_sentences)

# Define a function to extract character-centric sentiment, considering partial character name matches
def character_sentiment_extraction(row):
    character_name_parts = row['character_name'].split()  # Split the character name into parts
    plot_summary = row['plot_summary']
    
    # Split the plot summary into sentences
    sentences = nltk.sent_tokenize(plot_summary)
    
    # Filter sentences that mention any part of the character's name
    relevant_sentences = [
        sentence for sentence in sentences 
        if any(part.lower() in sentence.lower() for part in character_name_parts)
    ]
    
    # If relevant sentences are found, calculate the average sentiment score
    if relevant_sentences:
        scores = [sia.polarity_scores(sentence)['compound'] for sentence in relevant_sentences]
        return sum(scores) / len(scores)
    else:
        return None
