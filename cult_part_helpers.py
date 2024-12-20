import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import mplcursors
import matplotlib.pyplot as plt
import re 
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Initialize NLTK components
nltk.download('punkt')
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# Function to normalize character names
def normalize_name(name):
    return re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z0-9\s]', '', name)).strip().lower()

# Function to clean the movie data
def clean_merged_df(merged_df):
    merged_df.rename(columns={'plot': 'plot_summary'}, inplace=True)

    # Ensure all `character_name` values are strings
    merged_df['character_name'] = merged_df['character_name'].fillna('Unknown').astype(str)

    merged_df['character_sentiment_score'] = merged_df.apply(character_sentiment_extraction, axis=1)

    # Classify characters
    merged_df['classification'] = merged_df.apply(classify_villain_via_sentiment, axis=1)

    # Add a dummy 'popularity' column (replace with real data if available)
    merged_df['popularity'] = merged_df['character_sentiment_score'].apply(lambda x: abs(x) * 100 if x is not None else 0)

    # Extract character context
    merged_df['character_context'] = merged_df.apply(extract_character_context, axis=1)

    merged_df['actor_name'] = merged_df['actor_name'].str.strip().str.lower()
    return merged_df

def clean_plots_df(merged_df):
    # Ensure RELEASE_DATE is in datetime format
    merged_df['RELEASE_DATE'] = pd.to_datetime(merged_df['RELEASE_DATE'], errors='coerce')

    # Extract decade from release date
    merged_df['DECADE'] = (merged_df['RELEASE_DATE'].dt.year // 10) * 10

    # Drop rows with missing popularity
    merged_df = merged_df.dropna(subset=['POPULARITY'])

    # Ensure numeric values
    merged_df['REVENUE'] = pd.to_numeric(merged_df['REVENUE'], errors='coerce')
    merged_df['POPULARITY'] = pd.to_numeric(merged_df['POPULARITY'], errors='coerce')
    merged_df['imdb_score'] = pd.to_numeric(merged_df['imdb_score'], errors='coerce')
    return merged_df

# Function to clean the actor scores dataframe
def clean_actor_scores_df(actor_scores_df):
    # Rename 'actor' to 'actor_name' for consistency
    actor_scores_df.rename(columns={'actor': 'actor_name'}, inplace=True)

    # Standardize actor names
    actor_scores_df['actor_name'] = actor_scores_df['actor_name'].str.strip().str.lower()
    return actor_scores_df

def clean_merged_with_actor_df(merged_with_actor_df):
    # Combine sentiment columns
    merged_with_actor_df['sentiment'] = merged_with_actor_df['sentiment'].fillna(merged_with_actor_df['character_sentiment_score'])

    # Combine popularity columns
    merged_with_actor_df['popularity'] = merged_with_actor_df['popularity_y']

    # Drop unnecessary intermediate columns
    merged_with_actor_df.drop(columns=['character_sentiment_score', 'popularity_x', 'popularity_y'], inplace=True)
    return merged_with_actor_df

# Function to extract character-centric sentiment
def character_sentiment_extraction(row):
    character_name_parts = str(row['character_name']).split()  # Ensure `character_name` is a string
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
    
# Function to classify characters as 'Villain' or 'Not Villain'
def classify_villain_via_sentiment(row):
    score = row['character_sentiment_score']
    if score is not None and score < -0.3:  # Adjust threshold as needed
        return 'Villain'
    else:
        return 'Not Villain'

# Extract relevant context for each character
def extract_character_context(row):
    character_name_parts = row['character_name'].split()
    plot_summary = row['plot_summary']

    # Split plot summary into sentences
    sentences = nltk.sent_tokenize(plot_summary)

    # Filter sentences that mention any part of the character's name
    character_sentences = [
        sentence for sentence in sentences 
        if any(part.lower() in sentence.lower() for part in character_name_parts)
    ]

    return ' '.join(character_sentences)



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

# Classify characters into types based on sentiment
def classify_character_type(row):
    score = row['sentiment']
    if score > 0.3:
        return 'Hero'
    elif -0.3 <= score <= 0.3:
        return 'Neutral'
    elif score < -0.3:
        return 'Villain'
    

def plot_nb_of_characters_by_decade(df):
    # Count characters by decade
    df = df.groupby('decade')['character_name'].count()
    # Plot the timeline
    plt.figure(figsize=(10, 6))
    df.plot(kind='bar', color='skyblue')
    plt.title('Number of Characters Across Decades', fontsize=14)
    plt.xlabel('Decade', fontsize=12)
    plt.ylabel('Number of Characters', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_revenue_vs_popularity(df):
    # Scatter plot of revenue vs popularity
    plt.figure(figsize=(10, 6))
    plt.scatter(df['POPULARITY'], df['REVENUE'], alpha=0.6, color='blue')
    plt.xlabel('Popularity')
    plt.ylabel('Revenue')
    plt.title('Revenue vs Popularity')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_avg_imdb_by_decade(df):
    # Group by decade and calculate average IMDb score
    avg_imdb_by_decade = df.groupby('DECADE')['imdb_score'].mean()   

    # Plot IMDb score trends across decades
    plt.figure(figsize=(10, 6))
    avg_imdb_by_decade.plot(kind='line', marker='o', color='green')
    plt.xlabel('Decade')
    plt.ylabel('Average IMDb Score')
    plt.title('Average IMDb Score by Decade')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Extract character-specific context
def extract_character_context_sv(row):
    character_name_parts = row['character_name'].split()
    plot_summary = row['plot_summary']
    sentences = nltk.sent_tokenize(plot_summary)
    character_sentences = [
        sentence for sentence in sentences 
        if any(part.lower() in sentence.lower() for part in character_name_parts)
    ]
    return ' '.join(character_sentences)


# Apply sentiment analysis
def character_sentiment_extraction_sv(row):
    character_name_parts = str(row['character_name']).split()
    plot_summary = row['plot_summary']
    sentences = nltk.sent_tokenize(plot_summary)
    relevant_sentences = [
        sentence for sentence in sentences 
        if any(part.lower() in sentence.lower() for part in character_name_parts)
    ]
    if relevant_sentences:
        scores = [sia.polarity_scores(sentence)['compound'] for sentence in relevant_sentences]
        return sum(scores) / len(scores)
    else:
        return None


# Classify character types
def classify_character_type_sv(row):
    score = row['sentiment']
    if score > 0.3:
        return 'Hero'
    elif -0.3 <= score <= 0.3:
        return 'Neutral'
    elif score < -0.3:
        return 'Villain'
    
def plot_sym_villain_by_decade(character_metadata, plot_summaries):
    # Keep only relevant columns
    character_data = character_metadata.loc[:, ['movie_id', 'character_name', 'release_date']]
    # Merge plot summaries and character data on movie_id
    merged_data = pd.merge(plot_summaries, character_data, on='movie_id')

    # Drop NaN values
    merged_data_clean = merged_data.dropna(subset=['character_name']).reset_index(drop=True)

    # Keep characters that are mentioned in the plot summary
    characters_in_plot = merged_data_clean[
        merged_data_clean.apply(
            lambda row: any(part.lower() in row['plot_summary'].lower() for part in row['character_name'].split()),
            axis=1
        )
    ].reset_index(drop=True)

    characters_in_plot['character_context'] = characters_in_plot.apply(extract_character_context_sv, axis=1)
    characters_in_plot['sentiment'] = characters_in_plot.apply(character_sentiment_extraction_sv, axis=1)
    characters_in_plot['character_type'] = characters_in_plot.apply(classify_character_type_sv, axis=1)

    # Extract decades
    characters_in_plot['release_date'] = pd.to_datetime(characters_in_plot['release_date'], errors='coerce')
    characters_in_plot['decade'] = (characters_in_plot['release_date'].dt.year // 10) * 10

    # Sympathetic villains
    villains = characters_in_plot[characters_in_plot['character_type'] == 'Villain'].copy()
    villains['sympathetic'] = villains['plot_summary'].str.contains(
        'sympathy|tragedy|struggle|misunderstood|loss|trauma', case=False
    )

    # Plot sympathetic villains
    sympathetic_villain_by_decade = villains.groupby(['decade', 'sympathetic']).size().reset_index(name='count')
    sympathetic_villain_by_decade['sympathetic'] = sympathetic_villain_by_decade['sympathetic'].map(
        {True: "Sympathetic Villain", False: "Villain"}
    )
    fig1 = px.bar(
        sympathetic_villain_by_decade,
        x="decade",
        y="count",
        color="sympathetic",
        title="Sympathetic Villains by Decade",
        labels={"count": "Number of Villains", "decade": "Decade", "sympathetic": "Sympathetic"},
        text="count",
    )
    fig1.update_layout(barmode="stack", template="plotly_white")
    fig1.update_traces(texttemplate='%{text}', textposition='inside')
    fig1.show()

    # Plot character type distribution
    character_type_distribution = characters_in_plot.groupby(['decade', 'character_type']).size().reset_index(name='count')
    fig2 = px.bar(
        character_type_distribution,
        x="decade",
        y="count",
        color="character_type",
        title="Distribution of Heroes, Neutrals, and Villains Over Decades",
        labels={"count": "Number of Characters", "decade": "Decade", "character_type": "Character Type"},
        text="count",
    )
    fig2.update_layout(barmode="stack", template="plotly_white")
    fig2.update_traces(texttemplate='%{text}', textposition='inside')
    fig2.show()

def plot_movies(movies_df):
    movies_df['title_year'] = movies_df['title_year']
    movies_df = movies_df.sort_values('title_year')
    movies_df.dropna(subset=['title_year', 'movie_title', 'country'], inplace=True)
    movies_df['decade'] = (movies_df['title_year'] //10 *10).astype(int).astype(str)
    top_movies_per_decade = (
        movies_df.sort_values(by=['decade', 'imdb_score'], ascending=[True, False])  # Sort by decade and IMDb score
        .groupby('decade')  # Group by decade
        .head(5)  # Take the top 5 movies per decade
        .reset_index(drop=True)  # Reset index for clarity
    )

    # Create the first plot (IMDB Score)
    fig1 = px.bar(
        top_movies_per_decade.sort_values(by=['decade', 'imdb_score']),
        x="movie_title",
        y="imdb_score",
        color="decade",
        labels={"movie_title": "Movie Title", "imdb_score": "IMDB Score", "decade": "Decade"},
        hover_data=["director_name"],
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig1.update_traces(marker_line_width=2, hovertemplate="%{x}<extra></extra>")
    fig1.update_layout(
        xaxis=dict(title="Movie Title", tickangle=45),
        yaxis=dict(title="IMDB Score"),
        height=700,
        legend_title="Decade",
        title=dict(text="Top 5 Movies by IMDB Score for Each Decade (1970-2013)", x=0.5)
    )

    # Create the second plot (Number of User Reviews)
    fig2 = px.bar(
        top_movies_per_decade.sort_values(by=['decade', 'num_user_for_reviews']),
        x="movie_title",
        y="num_user_for_reviews",
        color="decade",
        labels={"movie_title": "Movie Title", "num_user_for_reviews": "Number of User Reviews", "decade": "Decade"},
        hover_data=["director_name"],
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig2.update_traces(marker_line_width=2, hovertemplate="%{x}<extra></extra>")
    fig2.update_layout(
        xaxis=dict(title="Movie Title", tickangle=45),
        yaxis=dict(title="Number of User Reviews"),
        height=700,
        legend_title="Decade",
        title=dict(text="Top 5 Movies by Number of User Reviews for Each Decade (1970-2013)", x=0.5)
    )

    # Combine the two plots with toggle buttons
    fig = go.Figure()

    # Add the traces for each plot
    fig.add_traces(fig1.data)
    fig.add_traces(fig2.data)

    # Update the visibility for the toggle effect
    # Initially show only the first set of traces (IMDB Score)
    for i, trace in enumerate(fig.data):
        trace.visible = i < len(fig1.data)

    # Define buttons to toggle between plots
    buttons = [
        dict(
            label="IMDB Score",
            method="update",
            args=[
                {"visible": [i < len(fig1.data) for i in range(len(fig.data))]},
                {"title": "Top 5 Movies by IMDB Score for Each Decade (1970-2013)"},
            ],
        ),
        dict(
            label="Number of Reviews",
            method="update",
            args=[
                {"visible": [i >= len(fig1.data) for i in range(len(fig.data))]},
                {"title": "Top 5 Movies by Number of User Reviews for Each Decade (1970-2013)"},
            ],
        ),
    ]

    # Add the buttons to the layout
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=buttons,
                showactive=True,
                x=0.5,
                xanchor="center",
                y=1.1,
                yanchor="top"
            )
        ],
        height=700
    )

    fig.show()

def oscars_to_directors(oscars, movies_df):
    oscar_winners = oscars[oscars['winner'] == True]
    oscar_winners.dropna(subset='film')
    oscar_winners = oscar_winners.dropna(subset=['film'])
    oscar_winners['film'] = oscar_winners['film'].str.lower()
    movies_df['movie_title'] = movies_df['movie_title'].str.lower()
    oscar_winners['film'] = oscar_winners['film'].str.strip()
    movies_df['movie_title'] = movies_df['movie_title'].str.strip()
    # Merge movie data and oscar on the same movie title
    oscar_and_imdb = oscar_winners.merge(movies_df, left_on='film', right_on='movie_title', how='inner')# Keep only relevant columns
    filtered_oscars = oscar_and_imdb.loc[oscar_and_imdb['year_film'] == oscar_and_imdb['title_year'], 
                                    [ 'category', 'film', 'director_name', 'num_user_for_reviews', 'country', 'title_year', 'imdb_score', 'decade']]
    # Calculate the number of unique oscar winning movies per director
    top_directors = (
        filtered_oscars.groupby('director_name')['film']
        .agg(
            unique_movies_count='nunique',
            unique_movies_list=lambda x: list(x.unique())
        )
        .reset_index()
    )

    # Display the resulting DataFrame
    top_directors.sort_values('unique_movies_count', ascending=False)
    # Adjust dataframe so that each movie gets its own row
    top_directors = top_directors.explode('unique_movies_list')

    # Rename the columns for clarity
    top_directors = top_directors.rename(columns={'unique_movies_list': 'film'})

    # Reorder the columns
    top_directors = top_directors[['director_name', 'film', 'unique_movies_count']]

    # Display the resulting DataFrame
    top_directors.sort_values('unique_movies_count', ascending=False)
    return top_directors