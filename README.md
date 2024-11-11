# Hollywood Evolution: A Deep Dive into Actor Career Paths, Genre Shifts, and the Rise of the Sympathetic Villain

# Unfolding the Actor-Genre Constellation in Cinema: Relationships, Sentiment, and the Evolution of the Sympathetic Villain

## Abstract

This project explores the evolution of actors’ careers and the portrayal of antagonists in cinema, using a blend of **network analysis**, **natural language processing (NLP)**, and **sentiment analysis**. We aim to uncover how actor career trajectories evolve across genres, the collaborative networks that shape successful film outcomes, and the rise of the "sympathetic villain" in popular cinema. Through the CMU Movie Summary Corpus and supplementary datasets, we will analyze genre shifts, actor collaboration clusters, and changing emotional tones associated with antagonists. The project intends to provide visualizations and insights into the key elements that drive cinematic success, while also offering an interactive component where users can simulate potential movie plots based on actor profiles. Our work will reveal trends in Hollywood's storytelling dynamics and demonstrate the interconnectedness of genre evolution, actor choices, and character portrayal.

## Project Structure

The directory structure of new project looks like this:

```
├── data                        <- Project data files #IGNORED
│
├── src                         <- Source code
│   ├── scripts                         <- Data pipelines and plots
      ├── helpers_API.py               <- TMDB database API GET functions
      ├── SP_plot.py                   <- Sentiment/Popularity score plot for actors
      ├── tvtropes_pipeline.py         <- Data pipeline that processes tvtropes file
      ├── sympathetic_villain.ipynb    <- Sentiment analysis pipeline on character_metadata
      ├── sami
      ├── sami
│
│
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- List of used libraries
├── install_requirements.ipynb  <- Notebook to install or update python dependencies
└── README.md
```

Check install_requirements.ipynb to update required libraries.

## Research Questions

1. **How do actors' careers evolve over time in terms of genre and emotional tone?**
2. **What collaborative actor constellations are associated with higher box office success or critical acclaim?**
3. **How has the portrayal of villains, especially sympathetic or complex antagonists, evolved over time across different genres?**
4. **Can we identify patterns between actor collaboration networks and the rise of specific character archetypes, like the sympathetic villain?**
5. **What potential movie plotlines can be generated based on an actor’s career history in genres and emotional tones?**

## Proposed Additional Datasets

In addition to the **CMU Movie Summary Corpus**, we propose using data from:

1. **IMDb Collaborations Data**:
   - **Content**: This dataset offers insights into actor pairings, collaboration frequency, and success metrics (e.g., box office revenue, IMDb ratings).
   - **Processing Approach**: We will integrate IMDb collaboration data to analyze actor constellations. Using NetworkX, we’ll map actor networks and calculate centrality metrics for identifying influential nodes and clusters within Hollywood.
   - **Expected Data Size and Format**: CSV/TSV format; expected to contain millions of rows detailing movie and actor relationships. Efficient processing with filtering and sampling will be necessary to manage size.

2. **The Movie Database (TMDb) API**:
   - **Content**: TMDb provides enriched metadata for each film, including genre, keywords, and actor bios.
   - **Processing Approach**: Using Python API requests, we will gather additional genre and character information for sentiment analysis. TMDb's documentation suggests the need for API keys and throttling limits, which we’ll accommodate in our scheduling.
   - **Expected Data Size and Format**: JSON format; results are manageable in size due to TMDb's structured pagination.

These datasets will be processed to handle memory management and enrichment tasks, such as merging genres with sentiment scores and managing network data.

## Methods

1. **NLP and Sentiment Analysis**:
   - We will conduct sentiment analysis on the `plot_summaries.txt` to identify emotional tones and sympathetic/antagonistic character traits. Using Python's `NLTK` and `TextBlob`, we will extract sentiment scores and categorize them according to genres and characters.
   
2. **Network Analysis of Actor Constellations**:
   - Using `NetworkX` and `Gephi`, we will build collaboration networks, identifying clusters where actor partnerships correlate with box office or critical success. Metrics such as betweenness and eigenvector centrality will highlight influential constellations.
   
3. **Visualization**:
   - Through `Matplotlib` and `Plotly`, we will create visual graphs for actor genre evolution and sympathetic villain portrayal. Time series plots will show trends in genres and sentiment tones over time.
   
4. **Interactive Movie Plot Generator**:
   - Using sentiment and genre data, we will create a machine-learning-based generator that proposes new movie titles and summaries tailored to an actor’s typical roles. This will involve a generative model using text data from plot summaries, focusing on genre and emotional alignment.

## Proposed Timeline - A revoir


| Phase                              | Tasks                                                                                                 | Date                         |
|------------------------------------|-------------------------------------------------------------------------------------------------------|------------------------------|
| **Week 1: Initial Setup**          | Dataset acquisition, initial analyses, and preprocessing setup                                        | Nov 5 - Nov 12               |
| **Week 2: Data Exploration & Pipeline Setup** | Perform descriptive statistics, sentiment analysis setup, and pipeline documentation  | Nov 12 - Nov 15 *(P2 Deadline)* |
| **Week 3: Network Mapping**        | Build the actor constellation network and analyze collaboration impact on success metrics             | Nov 16 - Nov 24              |
| **Week 4: Sentiment Analysis**     | Conduct sentiment analysis on plot summaries to identify sympathetic villains and genre trends        | Nov 25 - Dec 1               |
| **Week 5: Visualization**          | Create visualizations for genre evolution, constellations, and sympathetic villain trends             | Dec 2 - Dec 8                |
| **Week 6: Interactive Feature**    | Develop and test the interactive movie plot generator, integrate into final presentation              | Dec 9 - Dec 15               |
| **Week 7: Finalization**           | Complete documentation, README update, data story creation, and project repository review             | Dec 16 - Dec 20 *(P3 Deadline)* |

---


## Organization within the Team - A déterminer

faut répartir les tâches !

**Internal Milestones**:
- **Nov 12**: Dataset acquisition and preprocessing setup completed, with initial analyses and descriptive statistics in place.
- **Nov 15**: **P2 Submission** – README.md with detailed project proposal, preliminary analyses in a Jupyter notebook.
- **Nov 24**: Completion of constellation network mapping and success analysis.
- **Dec 1**: Completion of sentiment analysis and draft visualizations.
- **Dec 8**: Completion of interactive feature and final visualization drafts.
- **Dec 15**: Finalize all features, complete documentation, and prepare for **P3 submission**.
- **Dec 20**: **P3 Submission** – Final project code, results notebook, and data story URL.

## Questions for TAs

1. Would the integration of TMDb API data be feasible given the time and API rate limits, or should we explore a static alternative?
2. Are there recommended best practices for sentiment analysis specific to character-based plot data?
3. For the interactive component, would it be better to pursue a pre-trained generative model, or should we create a simple template-based text generator?
