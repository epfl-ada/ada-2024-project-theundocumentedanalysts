import re 
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





