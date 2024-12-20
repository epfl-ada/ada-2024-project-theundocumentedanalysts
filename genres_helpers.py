# Imports 
import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interactive_output, VBox, HBox, Layout, widgets, interact, IntSlider
import plotly.graph_objs as go
from IPython.display import display, clear_output
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import linregress
from collections import defaultdict
from itertools import combinations
import re


def extract_human_readable_genres(genre_str):
    # Use regular expression to find all human-readable genres
    genres = re.findall(r'": "([^"]+)"', genre_str)
    return '|'.join(genres)

# Function to clean release dates
def clean_release_date(date):
    if "-" in str(date):
        return str(date).split("-")[0]  # Keep only the year
    return str(date)  # If the date has only the year, return as is

def normalize_name(name):
    return re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z0-9\s]', '', name)).strip().lower()

# Prepare genre data by year groups and select top N genres
def prepare_genre_data(interval, top_n, movies_expanded):
    # Group movies into year intervals
    movies_expanded['year_group'] = (movies_expanded['title_year'] // interval) * interval
    
    # Count movies by year group and genre
    genre_counts = movies_expanded.groupby(['year_group', 'genres']).size().unstack(fill_value=0)
    
    # Get the most frequent genres across all years
    top_genres = genre_counts.sum().sort_values(ascending=False).head(top_n).index
    
    # Return counts for the top N genres
    return genre_counts[top_genres]

# Plot trends of top genres over time
def plot_genre_trends(metric, top_n, interval, movies_expanded):
    # Prepare data for the plot
    genre_counts_top = prepare_genre_data(interval, top_n, movies_expanded)
    
    # Use percentage or absolute counts based on the selected metric
    if metric == 'Percentage':
        genre_counts_plot = genre_counts_top.div(genre_counts_top.sum(axis=1), axis=0) * 100
        ylabel = 'Percentage (%)'
    else:
        genre_counts_plot = genre_counts_top
        ylabel = 'Number of Movies'
    
    # Clear any existing figure and set size
    plt.close('all')
    plt.figure(figsize=(14, 8))
    
    # Create a stacked bar chart
    ax = genre_counts_plot.plot(kind='bar', stacked=True, colormap='viridis', ax=plt.gca())
    plt.title(f'Genre Trends Over Time ({metric})', fontsize=16)
    plt.xlabel('Year Group', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    # Adjust legend and layout
    plt.legend(title='Genres', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(left=0.05, right=0.75, top=0.9, bottom=0.1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_frequencies(genre_counts_reduced, expected_df, residuals):
    # Visualize observed vs. expected frequencies
    plt.figure(figsize=(14, 8))
    sns.heatmap(genre_counts_reduced, cmap='YlGnBu', annot=True, fmt=".0f", cbar_kws={"label": "Observed Count"})
    plt.title('Observed Frequencies (Genres vs. 10-Year Intervals)', fontsize=16)
    plt.xlabel('Genres', fontsize=12)
    plt.ylabel('10-Year Intervals', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(expected_df, cmap='YlGnBu', annot=True, fmt=".0f", cbar_kws={"label": "Expected Count"})
    plt.title('Expected Frequencies (Genres vs. 10-Year Intervals)', fontsize=16)
    plt.xlabel('Genres', fontsize=12)
    plt.ylabel('10-Year Intervals', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Visualize residuals
    plt.figure(figsize=(14, 8))
    sns.heatmap(residuals, cmap='coolwarm', annot=True, fmt=".2f", cbar_kws={"label": "Residuals"})
    plt.title('Residuals: (Observed - Expected) / sqrt(Expected)', fontsize=16)
    plt.xlabel('Genres', fontsize=12)
    plt.ylabel('10-Year Intervals', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Create Interactive Visualization
def create_genre_revenue_plot(exploded_df):
    # Calculate the top 5 and worst 5 genres by count
    genre_counts = exploded_df['genres'].value_counts()
    top_5_genres = genre_counts.head(5).index.tolist()

    # Check presence of genres in the dataset before adding them to the worst 5 list
    worst_5_genres = [
        genre for genre in genre_counts.tail(8).index.tolist()
        if genre in exploded_df['genres'].unique() and genre in genre_counts.index
    ][:5]  # Limit to the first 5 valid genres

    print("Top 5 genres:", top_5_genres)
    print("Filtered Worst 5 genres:", worst_5_genres)

    # Create year groups (decades)
    exploded_df['decade'] = (exploded_df['title_year'] // 10) * 10

    # Aggregate revenue by decade and genre and calculate average revenue
    genre_revenue_by_decade = exploded_df.groupby(['decade', 'genres']).agg({'REVENUE': ['sum', 'count']}).reset_index()
    genre_revenue_by_decade.columns = ['decade', 'genres', 'total_revenue', 'count']
    genre_revenue_by_decade['average_revenue'] = genre_revenue_by_decade['total_revenue'] / genre_revenue_by_decade['count']

    # Filter datasets for top 5 and worst 5 genres
    top_5_revenue = genre_revenue_by_decade[genre_revenue_by_decade['genres'].isin(top_5_genres)]
    worst_5_revenue = genre_revenue_by_decade[genre_revenue_by_decade['genres'].isin(worst_5_genres)]
    # Create traces for top 5 genres
    top_traces = []
    for genre in top_5_genres:
        genre_data = top_5_revenue[top_5_revenue['genres'] == genre]
        trace = go.Scatter(
            x=genre_data['decade'],
            y=genre_data['average_revenue'],
            mode='lines+markers',
            name=f"Top: {genre}",
            visible=True  # Initially visible
        )
        top_traces.append(trace)

    # Create traces for worst 5 genres
    worst_traces = []
    for genre in worst_5_genres:
        genre_data = worst_5_revenue[worst_5_revenue['genres'] == genre]
        trace = go.Scatter(
            x=genre_data['decade'],
            y=genre_data['average_revenue'],
            mode='lines+markers',
            name=f"Worst: {genre}",
            visible=False  # Initially hidden
        )
        worst_traces.append(trace)

    # Combine all traces
    all_traces = top_traces + worst_traces

    # Create buttons for toggling between top 5 and worst 5 genres
    buttons = [
        dict(
            label="Top 5 Genres",
            method="update",
            args=[
                {"visible": [True] * len(top_traces) + [False] * len(worst_traces)},
                {"title.text": "Average Revenue Evolution Over Time "}
            ]
        ),
        dict(
            label="Worst 5 Genres",
            method="update",
            args=[
                {"visible": [False] * len(top_traces) + [True] * len(worst_traces)},
                {"title.text": "Average Revenue Evolution Over Time "}
            ]
        )
    ]

    # Build the figure
    fig = go.Figure(data=all_traces)
    fig.update_layout(
        title="Average Revenue Evolution Over Time ",
        xaxis_title="Decade",
        yaxis_title="Average Revenue",
        template="plotly_white",
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "x": 0.8,
            "xanchor": "center",
            "y": 1.2,
            "yanchor": "top",
            "showactive": True
        }],
        margin=dict(l=50, r=50, t=100, b=50)
    )

    # Display the plot
    fig.show()

# Function to calculate diversity score for each actor pair per genre
def calculate_diversity_score_by_genre(movies_df):
    all_genres_in_dataset = set(movies_df['genres'].str.cat(sep='|').split('|'))  # Calculate once for all genres
    genre_actor_pairs = defaultdict(lambda: defaultdict(set))

    for _, row in movies_df.dropna(subset=['actor_1_name', 'actor_2_name', 'actor_3_name', 'genres']).iterrows():
        actors = [row['actor_1_name'], row['actor_2_name'], row['actor_3_name']]
        genres = row['genres'].split('|')
        for genre in genres:
            for pair in combinations(actors, 2):
                pair = tuple(sorted(pair))
                genre_actor_pairs[genre][pair].update(genres)  # Update with all genres

    # Calculate diversity scores
    diversity_scores = []
    for genre, actor_pairs in genre_actor_pairs.items():
        for pair, genres in actor_pairs.items():
            diversity_score = len(genres) / len(all_genres_in_dataset)
            diversity_scores.append({'genre': genre, 'actor1': pair[0], 'actor2': pair[1], 'diversity_score': diversity_score})

    return pd.DataFrame(diversity_scores)

# Calculate the top actor pairs based on movie score with specific filters and genres
def calculate_top_actor_pairs_by_movie_score(movies_df, genre_filter=None, min_movies_together=3, min_movies_total=5):
    actor_pairs = defaultdict(list)  # Store movie scores for each actor pair

    for _, row in movies_df.dropna(subset=['actor_1_name', 'actor_2_name', 'actor_3_name', 'imdb_score', 'genres']).iterrows():
        actors = [row['actor_1_name'], row['actor_2_name'], row['actor_3_name']]
        imdb_score = row['imdb_score']
        genres = row['genres'].split('|')

        # Apply genre filter if specified
        if genre_filter and not any(g in genres for g in genre_filter):
            continue

        for pair in combinations(actors, 2):  # Generate all combinations of actor pairs
            pair = tuple(sorted(pair))  # Sort the pair to avoid duplicates
            actor_pairs[pair].append(imdb_score)

    # Filter pairs based on the number of movies together and separately
    filtered_scores = []
    for pair, scores in actor_pairs.items():
        if len(scores) >= min_movies_together:  # Minimum movies together
            actor1_total_movies = len(movies_df[(movies_df['actor_1_name'] == pair[0]) | (movies_df['actor_2_name'] == pair[0])])
            actor2_total_movies = len(movies_df[(movies_df['actor_1_name'] == pair[1]) | (movies_df['actor_2_name'] == pair[1])])
            if actor1_total_movies >= min_movies_total and actor2_total_movies >= min_movies_total:
                avg_score = sum(scores) / len(scores)  # Average IMDb score
                filtered_scores.append({'actor1': pair[0], 'actor2': pair[1], 'average_score': avg_score})

    return pd.DataFrame(filtered_scores)

# Interactive plot for top actor pairs by movie score with genre filtering
def plot_top_actor_pairs_by_score_with_genre(movies_df, genre_filter=None):
    # Calculate scores with the genre filter
    filtered_scores = calculate_top_actor_pairs_by_movie_score(movies_df, genre_filter=genre_filter)
    
    # Check if filtered_scores is empty
    if filtered_scores.empty:
        print(f"No data available for the selected genre(s): {', '.join(genre_filter) if genre_filter else 'None'}")
        return
    
    # Sort and select top pairs
    top_pairs = filtered_scores.sort_values(by='average_score', ascending=False).head(10)

    # Plot the top 10 actor pairs by average movie score
    plt.figure(figsize=(10, 6))
    plt.barh(
        y=top_pairs.apply(lambda x: f"{x['actor1']} & {x['actor2']}", axis=1), 
        width=top_pairs['average_score'], 
        color='lightgreen'
    )
    plt.xlabel('Average IMDb Score')
    plt.ylabel('Actor Pairs')
    plt.title('Top 10 Actor Pairs by Average Movie Score')
    plt.tight_layout()
    plt.show()

# Function to calculate average IMDb scores for actor pairs
def calculate_avg_imdb_scores(movies_df):
    actor_pair_scores = defaultdict(list)

    for _, row in movies_df.dropna(subset=['actor_1_name', 'actor_2_name', 'actor_3_name', 'genres', 'imdb_score']).iterrows():
        actors = [row['actor_1_name'], row['actor_2_name'], row['actor_3_name']]
        genres = row['genres'].split('|')
        for genre in genres:
            for pair in combinations(actors, 2):
                pair = tuple(sorted(pair))
                actor_pair_scores[(genre, pair)].append(row['imdb_score'])

    # Aggregate average scores
    imdb_scores = []
    for (genre, pair), scores in actor_pair_scores.items():
        avg_score = sum(scores) / len(scores)
        imdb_scores.append({'genre': genre, 'actor1': pair[0], 'actor2': pair[1], 'avg_imdb_score': avg_score})

    return pd.DataFrame(imdb_scores)

# Merge and analyze correlation
def analyze_combined_scores(movies_df):
    diversity_df = calculate_diversity_score_by_genre(movies_df)
    imdb_df = calculate_avg_imdb_scores(movies_df)

    # Merge datasets
    combined_df = pd.merge(diversity_df, imdb_df, on=['genre', 'actor1', 'actor2'], how='inner')

    # Display correlation
    correlation = combined_df['diversity_score'].corr(combined_df['avg_imdb_score'])
    print(f"Correlation between Diversity Score and Average IMDb Score: {correlation:.3f}")

    # Persistent output area for the figure
    output_area = widgets.Output()
    display(output_area)

    # Scatter plot to visualize the relationship
    def plot_combined_scores(genre_filter):
        with output_area:
            clear_output(wait=True)  # Clear the existing plot
            filtered_data = combined_df if genre_filter == "All" else combined_df[combined_df['genre'] == genre_filter]

            if filtered_data.empty:
                print(f"No data available for genre: {genre_filter}")
                return

            fig = px.scatter(
                filtered_data,
                x='diversity_score',
                y='avg_imdb_score',
                color='genre',
                title=f"Diversity Score vs IMDb Score ({genre_filter})",
                labels={'diversity_score': 'Diversity Score', 'avg_imdb_score': 'Average IMDb Score'},
                hover_data=['actor1', 'actor2']
            )
            fig.show()

    # Dropdown for genre selection
    available_genres = list(combined_df['genre'].unique()) + ["All"]
    interact(plot_combined_scores, genre_filter=widgets.Dropdown(options=available_genres, description='Genre:'))

    return combined_df

# Function to calculate actor pair scores for heatmap
def calculate_actor_pair_movies(movies_df, genre_filter=None, decade_filter=None, min_movies_together=1, min_movies_total=3):
    actor_pairs = defaultdict(list)  # Store IMDb scores for each actor pair

    # Filter the top 1000 movies based on IMDb scores
    top_movies_df = movies_df.nlargest(500, 'imdb_score')

    # Iterate through rows with valid actor and score data
    for _, row in top_movies_df.dropna(subset=['actor_1_name', 'actor_2_name', 'actor_3_name', 'imdb_score', 'genres', 'title_year']).iterrows():
        actors = [row['actor_1_name'], row['actor_2_name'], row['actor_3_name']]
        genres = row['genres'].split('|')
        decade = (row['title_year'] // 10) * 10  # Calculate decade

        # Apply genre and decade filters
        if genre_filter and not any(g in genres for g in genre_filter):
            continue
        if decade_filter and decade != decade_filter:
            continue

        for pair in combinations(actors, 2):  # Generate all combinations of actor pairs
            pair = tuple(sorted(pair))  # Sort the pair to avoid duplicates
            actor_pairs[pair].append(1)  # Increment count of movies together

    # Calculate the number of movies made together and filter pairs
    filtered_scores = []
    for pair, counts in actor_pairs.items():
        total_movies = len(counts)  # Number of movies made together
        if total_movies >= min_movies_together:
            actor1_total_movies = len(movies_df[(movies_df['actor_1_name'] == pair[0]) | (movies_df['actor_2_name'] == pair[0])])
            actor2_total_movies = len(movies_df[(movies_df['actor_1_name'] == pair[1]) | (movies_df['actor_2_name'] == pair[1])])
            if actor1_total_movies >= min_movies_total and actor2_total_movies >= min_movies_total:
                filtered_scores.append({'actor1': pair[0], 'actor2': pair[1], 'total_movies': total_movies})

    return pd.DataFrame(filtered_scores)

# Heatmap for Actor Pair Movie Counts using Plotly
def plot_actor_pair_heatmap(movies_df, genre_filter=None, decade_filter=None):
    scores_df = calculate_actor_pair_movies(movies_df, genre_filter=genre_filter, decade_filter=decade_filter)

    if scores_df.empty:
        print("No data available for the selected genres or decades.")
        return

    # Create pivot table for heatmap
    heatmap_data = pd.pivot_table(
        scores_df,
        values='total_movies',  # Use the count of movies as the values
        index='actor1',
        columns='actor2',
        aggfunc='sum',
        fill_value=0
    )

    # Convert pivot table to long format for Plotly
    heatmap_data_long = heatmap_data.reset_index().melt(id_vars='actor1', var_name='actor2', value_name='total_movies')

    # Plot interactive heatmap using Plotly
    fig = px.imshow(
        heatmap_data.values,
        labels=dict(x="Actor 2", y="Actor 1", color="Number of Movies Together"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale="viridis",  # Use a sequential colorscale
        aspect="auto",
        title=f"Actor Pair Collaboration Heatmap ({decade_filter if decade_filter else 'All Decades'}) - Top 500 Movies",
    )

    # Add hover data
    fig.update_traces(
        hovertemplate="<b>Actor 1:</b> %{y}<br>" +
                      "<b>Actor 2:</b> %{x}<br>" +
                      "<b>Total Movies Together:</b> %{z}<extra></extra>"
    )

    fig.show()

# Function to calculate average IMDb scores based on the number of movies done together
def calculate_average_scores_by_collaboration(movies_df, genre_filter=None, decade_filter=None):
    collaboration_scores = defaultdict(list)  # Store IMDb scores for each collaboration count

    for _, row in movies_df.dropna(subset=['actor_1_name', 'actor_2_name', 'actor_3_name', 'imdb_score', 'genres', 'title_year']).iterrows():
        actors = [row['actor_1_name'], row['actor_2_name'], row['actor_3_name']]
        imdb_score = row['imdb_score']
        genres = row['genres'].split('|')
        decade = (row['title_year'] // 10) * 10  # Calculate decade

        # Apply genre filter if provided
        if genre_filter and not any(g in genres for g in genre_filter):
            continue

        # Apply decade filter if provided
        if decade_filter and decade != decade_filter:
            continue

        for pair in combinations(actors, 2):  # Generate all combinations of actor pairs
            pair = tuple(sorted(pair))  # Sort the pair to avoid duplicates
            collaboration_scores[pair].append(imdb_score)

    # Aggregate scores by the number of movies done together
    aggregated_scores = defaultdict(list)
    for pair, scores in collaboration_scores.items():
        num_movies = len(scores)
        avg_score = sum(scores) / len(scores)
        aggregated_scores[num_movies].append(avg_score)

    # Prepare DataFrame for plotting
    data = []
    for num_movies, scores in aggregated_scores.items():
        data.append({"num_movies": num_movies, "avg_score": sum(scores) / len(scores)})

    return pd.DataFrame(data)

# 3D Plot for Average Scores by Collaboration Count
def plot_3d_collaboration_scores(movies_df):
    scores_df = []

    for genre_filter in movies_df['genres'].str.split('|').explode().unique():
        for decade_filter in sorted((movies_df['title_year'] // 10 * 10).dropna().unique().astype(int)):
            scores = calculate_average_scores_by_collaboration(movies_df, genre_filter=[genre_filter], decade_filter=decade_filter)
            if not scores.empty:
                scores['genre'] = genre_filter
                scores['decade'] = decade_filter
                scores_df.append(scores)

    # Combine all data
    final_df = pd.concat(scores_df, ignore_index=True)

    # Plot 3D chart
    fig = px.scatter_3d(
        final_df,
        x='num_movies',
        y='avg_score',
        z='decade',
        color='genre',
        title="Average IMDb Scores by Collaboration Count (3D)",
        labels={"num_movies": "Number of Movies Together", "avg_score": "Average IMDb Score", "decade": "Decade"},
    )
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    fig.show()

def plot_3d_surface_collaboration_scores(movies_df):
    scores_df = []

    for genre_filter in movies_df['genres'].str.split('|').explode().unique():
        for decade_filter in sorted((movies_df['title_year'] // 10 * 10).dropna().unique().astype(int)):
            scores = calculate_average_scores_by_collaboration(movies_df, genre_filter=[genre_filter], decade_filter=decade_filter)
            if not scores.empty:
                scores['genre'] = genre_filter
                scores['decade'] = decade_filter
                scores_df.append(scores)

    # Combine all data
    final_df = pd.concat(scores_df, ignore_index=True)

    # Pivot data for surface plot
    surface_data = final_df.pivot_table(values='avg_score', index='num_movies', columns='decade', aggfunc='mean').fillna(0)
    x = surface_data.columns
    y = surface_data.index
    z = surface_data.values

    # Create the surface plot
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis')])
    fig.update_layout(
        title="Average IMDb Scores by Collaboration Count (3D Surface)",
        scene=dict(
            xaxis_title="Decade",
            yaxis_title="Number of Movies Together",
            zaxis_title="Average IMDb Score",
        )
    )
    fig.show()


def calculate_star_power(movie_data, awards_data):
    """
    Calculate the Star Power Index for actors based on IMDb scores, awards, longevity, and consistency.
    """
    actor_columns = ['actor_1_name', 'actor_2_name', 'actor_3_name']
    actor_movies = pd.melt(movie_data, id_vars=['movie_title', 'imdb_score', 'title_year'], 
                           value_vars=actor_columns, value_name='actor').dropna()

    # Calculate actor metrics
    actor_ratings = actor_movies.groupby('actor')['imdb_score'].mean().reset_index(name='avg_imdb_score')
    actor_longevity = actor_movies.groupby('actor')['title_year'].nunique().reset_index(name='longevity')
    actor_consistency = actor_movies.groupby('actor')['imdb_score'].std().reset_index(name='consistency')

    # Process awards data
    awards_count = awards_data.groupby('name').agg(
        nominations=('winner', 'count'),
        wins=('winner', 'sum')
    ).reset_index().rename(columns={'name': 'actor'})

    # Merge actor metrics
    actor_metrics = actor_ratings.merge(actor_longevity, on='actor', how='left') \
                                 .merge(actor_consistency, on='actor', how='left')
    star_power_df = actor_metrics.merge(awards_count, on='actor', how='left').fillna(0)

    # Calculate star power
    star_power_df['star_power_index'] = (
        0.4 * star_power_df['avg_imdb_score'] +
        0.3 * star_power_df['wins'] +
        0.2 * star_power_df['longevity'] +
        0.1 * (1 / (1 + star_power_df['consistency'].fillna(0.1)))
    )
    return star_power_df.sort_values('star_power_index', ascending=False)

def calculate_producer_metrics(movie_data):
    """
    Calculate the Producer Index based on average IMDb scores and longevity.
    """
    producer_movies = movie_data[['movie_title', 'imdb_score', 'title_year', 'director_name']].dropna()
    producer_ratings = producer_movies.groupby('director_name')['imdb_score'].mean().reset_index(name='avg_producer_score')
    producer_longevity = producer_movies.groupby('director_name')['title_year'].nunique().reset_index(name='producer_longevity')

    # Combine producer metrics
    producer_metrics = producer_ratings.merge(producer_longevity, on='director_name', how='left')
    producer_metrics['producer_index'] = (
        0.6 * producer_metrics['avg_producer_score'] + 
        0.4 * producer_metrics['producer_longevity']
    )
    return producer_metrics

def merge_actor_producer_data(movie_data, star_power_df, producer_metrics):
    """
    Combine actor and producer data, linking actors with their star power and producers with their index.
    """
    combined_df = pd.merge(
        movie_data[['movie_title', 'actor_1_name', 'actor_2_name', 'actor_3_name', 'director_name']],
        producer_metrics[['director_name', 'producer_index']],
        on='director_name',
        how='left'
    )

    # Melt actor columns and merge star power
    combined_df = pd.melt(
        combined_df, 
        id_vars=['movie_title', 'director_name', 'producer_index'],
        value_vars=['actor_1_name', 'actor_2_name', 'actor_3_name'],
        value_name='actor'
    ).dropna()

    combined_df = pd.merge(
        combined_df, 
        star_power_df[['actor', 'star_power_index']], 
        on='actor', 
        how='left'
    )
    return combined_df

def plot_actor_producer_relationship(data):
    """
    Create a scatter plot showing the relationship between Actor Star Power and Producer Index.
    """
    fig = px.scatter(
        data,
        x='star_power_index',
        y='producer_index',
        color='director_name',
        hover_data=['actor', 'movie_title'],
        title="Actor Star Power vs Producer Index"
    )
    fig.show()

def plot_trends_over_time(data, movie_data):
    """
    Plot the trends in Actor Star Power and Producer Index over time.
    """
    data['year'] = movie_data['title_year']
    trend_df = data.groupby('year')[['star_power_index', 'producer_index']].mean().reset_index()

    fig = px.line(
        trend_df,
        x='year',
        y=['star_power_index', 'producer_index'],
        title="Trends in Actor Star Power and Producer Index Over Time",
        labels={"value": "Index Score", "variable": "Metric"}
    )
    fig.show()