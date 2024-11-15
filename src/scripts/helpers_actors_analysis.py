import ast
import pandas as pd
import numpy as np
import random
from itertools import combinations
from collections import Counter
from pyvis.network import Network
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns


# Function to clean the genre column by extracting genre names
def extract_genres(genre_str):
    try:
        genre_dict = ast.literal_eval(genre_str)
        # Extract and join all genre names
        cleaned_genres = '|'.join(genre_dict.values())
        return cleaned_genres
    except (ValueError, SyntaxError):
        return None
    
def group_and_filter_movies(merged_movie_character, top_n_genres=50):
    """
    Groups movies by Wikipedia movie ID and filters by top genres.

    Parameters:
    - merged_movie_character (DataFrame): DataFrame containing movie and actor information.
    - top_n_genres (int): Number of top genres to filter by.

    Returns:
    - filtered_movies (DataFrame): DataFrame of movies filtered by the top genres.
    - top_genres (list): List of the top genres.
    """
    # Group movies by Wikipedia movie ID
    movies_grouped = merged_movie_character.groupby('Wikipedia movie ID').agg({
        'Movie genres': 'first',  # Get the first non-NaN value for genres
        'Actor name': lambda x: '|'.join(x.dropna().unique())  # Aggregate all actors for each movie
    }).reset_index()

    # Get top genres
    all_genres = movies_grouped['Movie genres'].dropna().str.split('|').sum()
    genre_counts = Counter(all_genres)
    top_genres = [genre for genre, count in genre_counts.most_common(top_n_genres)]

    # Filter by top genres
    filtered_movies = movies_grouped[
        movies_grouped['Movie genres'].apply(lambda genres: any(g in top_genres for g in genres.split('|')))
    ]
    
    return filtered_movies, top_genres

def generate_actor_pairs(filtered_movies):
    """
    Generates actor pairs from the filtered movies DataFrame.

    Parameters:
    - filtered_movies (DataFrame): The filtered DataFrame containing movie and actor information.

    Returns:
    - List of actor pairs.
    """
    actor_pairs = []
    for _, row in filtered_movies.iterrows():
        actors = row['Actor name']

        # Skip if actors are NaN
        if pd.isna(actors):
            continue

        actors_list = actors.split('|') if isinstance(actors, str) else actors

        # Generate all possible pairs of actors for the movie
        actor_pairs.extend(list(combinations(actors_list, 2)))
    
    return actor_pairs

def visualize_top_actor_genre(merged_movie_character, top_n_genres=50, top_n_pairs=500):
    """
    Visualizes the most commonly seen pairs of actors in the filtered movies using PyVis.

    Parameters:
    - merged_movie_character (DataFrame): The merged DataFrame containing movie, genre, and actor information.
    - top_n_genres (int): Number of top genres to filter by. Default is 50.
    - top_n_pairs (int): Number of top actor pairs to visualize. Default is 500.
    """
    # Filter movies
    filtered_movies, top_50_genres = group_and_filter_movies(merged_movie_character, top_n_genres)

    # Generate Actor pairs
    actor_pairs = generate_actor_pairs(filtered_movies)

    # Count top actor pairs
    pair_counts = Counter(actor_pairs)
    top_pairs = pair_counts.most_common(top_n_pairs)

    # Extract nodes, edges, and genres
    nodes = set()
    edges = []
    node_genre_map = {}

    for (actor1, actor2), count in top_pairs:
        nodes.update([actor1, actor2])
        edges.append((actor1, actor2, count))

    # Determine the genres for the actors
    for _, row in filtered_movies.iterrows():
        genres = row['Movie genres']
        actors = row['Actor name']

        if pd.isna(genres) or pd.isna(actors):
            continue

        genres_list = genres.split('|')
        actors_list = actors.split('|') if isinstance(actors, str) else actors

        for actor in actors_list:
            if actor in nodes and actor not in node_genre_map:
                relevant_genres = [genre for genre in genres_list if genre in top_50_genres]
                node_genre_map[actor] = relevant_genres[0] if relevant_genres else 'Other'

    # assign colors to genres
    genre_colors = {genre: f'#{random.randint(0, 0xFFFFFF):06x}' for genre in top_50_genres}
    default_color = '#999999'

    # Initialize the PyVis Network 
    net = Network(notebook=True, height='750px', width='100%', bgcolor='#222222', font_color='white')
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08, damping=0.4, overlap=0)

    # Add nodes to the network with genre-based colors
    for node in nodes:
        genre = node_genre_map.get(node, 'Other')
        color = genre_colors.get(genre, default_color)
        net.add_node(node, label=node, color=color)

    # Add edges to the network with labels for frequency
    for source, target, weight in edges:
        net.add_edge(source, target, value=weight, title=f'Collaborations: {weight}')

    # Visualization
    net.show("actor_collaboration_filtered_genre_optimized.html")

def filter_top_grossing_movies(df, top_n=500):
    """
    Filters the top N grossing movies from the given DataFrame.
    
    Parameters:
    - df (DataFrame): The merged movie-character DataFrame.
    - top_n (int): The number of top grossing movies to filter. Default is 500.
    
    Returns:
    - DataFrame: A DataFrame with the top N grossing movies and aggregated actors.
    """
    # Group movies by 'Wikipedia movie ID' and aggregate information
    movies_grouped = df.groupby('Wikipedia movie ID').agg({
        'Movie genres': 'first',
        'Actor name': lambda x: '|'.join(x.dropna().unique()),
        'Movie box office revenue': 'first'
    }).reset_index()

    # Sort by box office revenue and take the top N movies
    top_grossing_movies = movies_grouped.sort_values(by='Movie box office revenue', ascending=False).head(top_n)

    return top_grossing_movies

def visualize_top_actor_pairs(filtered_movies, top_50_genres, top_n_pairs=500):
    """
    Visualizes the most commonly seen pairs of actors in the top movies using PyVis.

    Parameters:
    - filtered_movies (DataFrame): The filtered DataFrame containing movie, genre, and actor information.
    - top_50_genres (list): List of the top 50 genres by frequency.
    - top_n_pairs (int): Number of top actor pairs to visualize. Default is 500.
    """
    # Generate Actor Pairs from Filtered Movies
    actor_pairs = []
    for _, row in filtered_movies.iterrows():
        actors = row['Actor name']
        if pd.isna(actors):
            continue
        actors_list = actors.split('|') if isinstance(actors, str) else actors
        actor_pairs.extend(list(combinations(actors_list, 2)))

    # Count and Get Top Actor Pairs
    pair_counts = Counter(actor_pairs)
    top_pairs = pair_counts.most_common(top_n_pairs)

    nodes = set()
    edges = []
    node_genre_map = {}

    for (actor1, actor2), count in top_pairs:
        nodes.update([actor1, actor2])
        edges.append((actor1, actor2, count))

        # Determine the genres for the actors
        for _, row in filtered_movies.iterrows():
            genres = row['Movie genres']
            actors = row['Actor name']
            if pd.isna(genres) or pd.isna(actors):
                continue
            genres_list = genres.split('|')
            actors_list = actors.split('|') if isinstance(actors, str) else actors
            if actor1 in actors_list or actor2 in actors_list:
                relevant_genres = [genre for genre in genres_list if genre in top_50_genres]
                if actor1 not in node_genre_map:
                    node_genre_map[actor1] = relevant_genres if relevant_genres else ['Other']
                if actor2 not in node_genre_map:
                    node_genre_map[actor2] = relevant_genres if relevant_genres else ['Other']

    genre_colors = {genre: f'#{random.randint(0, 0xFFFFFF):06x}' for genre in top_50_genres}
    default_color = '#999999'

    net = Network(notebook=True, height='750px', width='100%', bgcolor='#222222', font_color='white')
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08, damping=0.4, overlap=0)

    # Add nodes to the network with genre-based colors
    for node in nodes:
        genres = node_genre_map.get(node, ['Other'])
        color = '#FFFFFF' if len(genres) > 1 else genre_colors.get(genres[0], default_color)
        net.add_node(node, label=node, color=color, title=f'Genres: {", ".join(genres)}')

    # Add edges to the network with labels for frequency
    for source, target, weight in edges:
        net.add_edge(source, target, value=weight, title=f'Collaborations: {weight}')

    # Add legend for different genres
    for genre, color in genre_colors.items():
        net.add_node(genre, label=genre, color=color, shape='box', title=f'Genre: {genre}', size=10)

    net.show("top_actor_pairs_visualization.html")

def perform_genre_statistical_analysis(df):
    """
    Performs statistical analysis on genre preferences.

    Parameters:
    - df (DataFrame): DataFrame containing movie and genre information.
    """
    # Count the occurrence of each genre in the dataset
    all_genres = df['Movie genres'].dropna().str.split('|').sum()
    genre_counts = Counter(all_genres)

    # Convert genre counts to a DataFrame for analysis
    genre_df = pd.DataFrame.from_dict(genre_counts, orient='index', columns=['Count']).reset_index()
    genre_df.rename(columns={'index': 'Genre'}, inplace=True)

    # Plot the distribution of genres
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Genre', y='Count', data=genre_df.sort_values(by='Count', ascending=False).head(20), palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 20 Genres by Frequency')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # Compare the top genres over the entire dataset
    top_genres = genre_df.head(5)['Genre'].tolist()
    genre_counts_by_movie = df['Movie genres'].dropna().str.get_dummies(sep='|')[top_genres]

    # Perform ANOVA to see if there is a significant difference in the popularity of these genres

    genre_values = [genre_counts_by_movie[genre] for genre in top_genres]

    # Perform ANOVA
    f_stat, p_value = f_oneway(*genre_values)
    print(f"ANOVA F-statistic: {f_stat:.3f}, p-value: {p_value:.3e}")

def analyze_genre_preferences_by_actor_pairings(filtered_movies, top_n_pairs=500):
    """
    Analyzes genre preferences for the most commonly seen pairs of actors in the top movies.
    
    Parameters:
    - filtered_movies (DataFrame): The filtered DataFrame containing movie, genre, and actor information.
    - top_n_pairs (int): Number of top actor pairs to analyze. Default is 500.
    
    Returns:
    - actor_pair_genres (dict): A dictionary mapping actor pairs to their genre counts.
    """
    # Generate Actor Pairs and Associated Genres
    actor_pairs = []
    genre_pairings = []

    for _, row in filtered_movies.iterrows():
        actors = row['Actor name']
        genres = row['Movie genres']
        if pd.notna(actors) and pd.notna(genres):
            actors_list = actors.split('|')
            genres_list = genres.split('|')
            actor_pairs.extend(combinations(actors_list, 2))
            genre_pairings.extend([(pair, genres_list) for pair in combinations(actors_list, 2)])

    # Count Top Actor Pairs
    top_pairs = Counter(actor_pairs).most_common(top_n_pairs)

    # Extract Genres for Top Actor Pairs
    actor_pair_genres = {
        pair: Counter(
            genre for (actor_pair, genres) in genre_pairings if set(actor_pair) == set(pair) for genre in genres
        )
        for pair, _ in top_pairs
    }

    return actor_pair_genres

def visualize_genre_preferences_heatmap_extended(actor_pair_genres, top_n_pairs=10):
    """
    Visualizes genre preferences for the top N actor pairs using a heatmap, 
    only including genres that are present for those actor pairs.
    
    Parameters:
    - actor_pair_genres (dict): A dictionary mapping actor pairs to their genre counts.
    - top_n_pairs (int): Number of top actor pairs to visualize. Default is 10.
    """
    # Filter the top N pairs for visualization
    top_pairs = list(actor_pair_genres.items())[:top_n_pairs]

    # Collect only genres that are present for the selected actor pairs
    present_genres = set()
    for _, genre_count in top_pairs:
        present_genres.update(genre_count.keys())
    
    present_genres = sorted(present_genres)
    actor_pair_labels = [f"{actor1} & {actor2}" for (actor1, actor2), _ in top_pairs]
    heatmap_data = []

    for _, genre_count in top_pairs:
        row = [genre_count.get(genre, 0) for genre in present_genres]
        heatmap_data.append(row)

    heatmap_df = pd.DataFrame(heatmap_data, index=actor_pair_labels, columns=present_genres)

    # Plot the heatmap
    plt.figure(figsize=(16, 10))
    sns.heatmap(heatmap_df, annot=True, linewidths=.5, fmt='g')
    plt.xlabel('Genres')
    plt.ylabel('Actor Pairs')
    plt.title(f'Top {top_n_pairs} Actor Pairs by Genre Preferences (Filtered Heatmap)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def calculate_genre_diversity(actor_pair_genres):
    """
    Calculate diversity scores for actor pairs across genres.

    Parameters:
    - actor_pair_genres (dict): A dictionary where keys are actor pairs and values are counters of genres.

    Returns:
    - DataFrame: A DataFrame containing actor pairs, unique genre count, and Shannon diversity score.
    """
    diversity_data = []

    for pair, genre_count in actor_pair_genres.items():
        unique_genres = len(genre_count)

        # Calculate Shannon Diversity Index
        total_movies = sum(genre_count.values())
        genre_frequencies = np.array(list(genre_count.values())) / total_movies
        shannon_diversity = -np.sum(genre_frequencies * np.log(genre_frequencies))

        # Store the results
        diversity_data.append({
            'Actor Pair': f"{pair[0]} & {pair[1]}",
            'Unique Genres': unique_genres,
            'Shannon Diversity Score': shannon_diversity
        })

    diversity_df = pd.DataFrame(diversity_data)

    # Sort by Shannon Diversity Score for easier interpretation
    diversity_df = diversity_df.sort_values(by='Shannon Diversity Score', ascending=False)

    return diversity_df