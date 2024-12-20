
# Unfolding the Actor-Genre Constellation in Cinema: Relationships, Sentiment, and the Rise of the Sympathetic Villain

# You can find the data story on our website [here](https://mehdi1704.github.io/jekyll-theme-yat/)

## Abstract

This project explores the evolution of actors’ careers and the portrayal of antagonists in cinema, using a blend of **network analysis**, **natural language processing (NLP)**, and **sentiment analysis**. We aim to uncover how actor career trajectories evolve across genres, the collaborative networks that shape successful film outcomes, and the rise of the "sympathetic villain" in popular cinema. Through the CMU Movie Summary Corpus and supplementary datasets, we will analyze genre shifts, actor collaboration clusters, and changing emotional tones associated with antagonists. The project intends to provide visualizations and insights into the key elements that drive cinematic success, while also offering an interactive component where users can simulate potential movie plots based on actor profiles. Our work will reveal trends in Hollywood's storytelling dynamics and demonstrate the interconnectedness of genre evolution, actor choices, and character portrayal.

## Project Structure

The directory structure of new project looks like this:

```
├── data/                     <- Project data files #IGNORED
   ├── CMU_dataset/           <- Chosen dataset
   ├── TMDB_dataset/          <- TMDB local dataset to avoid API requests
   ├── TMDB_dataset_csv/      <- TMDB local dataset to avoid API requests
   ├── the_oscar_award.csv    <- Academy Awards: 1927 - 2024 nominees and winners dataset
   └── movie_data.csv         <- Directors, Actors, Genres, and Movies ratings
│
│
├── output_data/                                <- Processed data files
   └── actor_sentiment_popularity_scores.csv    <- tvtropes_pipeline.py output
│
│
├── src/                               <- Source code
   ├── results_P2.ipynb                <- Old results file, containing all Milestone 2 analysis
   ├── helpers_actors_analysis.py      <- Helper functions to Actors, Movies and Oscars analysis
   ├── helpers_villain_analysis.py     <- Helper functions for villain sentiment analysis
   ├── helpers_API.py                  <- TMDB database API GET functions
   │
   └── drafts/                         <- Separate data pipelines and plots
      ├── SP_plot.ipynb                <- Sentiment/Popularity score plot for actors
      ├── tvtropes_pipeline.py         <- Data pipeline that processes tvtropes file
      ├── sympathetic_villain.ipynb    <- Sentiment analysis pipeline on character_metadata
      └── oscars_movies_analysis.ipynb <- Actor/Genres constellations analysis and additional oscars implementations
│
│
│
├── results.ipynb               <- New main file, containing all Milestone 3 analysis
├── model.app                   <- Application containing our machine learning model deployed with the website
│
├── notebook_merger.py          <- notebooks merger script
├── .gitignore                  <- List of files ignored by git
├── requirements.txt            <- List of used libraries
├── install_requirements.ipynb  <- Notebook to install or update python dependencies
└── README.md
```

⚠️ **Important:** Refer to `install_requirements.ipynb` to ensure all required libraries are installed.

## Research Questions

**1. How have genre preferences and trends evolved over the decades?**

**2. What is the impact of actor collaborations and director influence on movie success?**

**3. How has the portrayal of villains, particularly sympathetic antagonists, changed over time?**

**4. Can sentiment analysis reveal patterns in character portrayals, especially for antagonists?**

**5. How do combined factors (actors, genres, directors) predict movie success metrics (IMDb ratings, box office revenues)?**
## Proposed Additional Datasets

## Proposed Additional Datasets

1. **[IMDb Collaborations Data](https://www.kaggle.com/rounakbanik/the-movies-dataset)**  
   - **Content**: Insights into actor pairings, collaboration frequency, and success metrics (e.g., box office revenue, IMDb ratings).
   - **Processing Approach**: Integration of IMDb collaboration data to analyze actor constellations. Using NetworkX, we’ll map actor networks and calculate centrality metrics for identifying influential nodes and clusters within Hollywood.

2. **[The Movie Database (TMDb) API](https://developers.themoviedb.org/3/getting-started)**  
   - **Content**: Metadata for films, including genres, keywords, and actor bios.
   - **Processing Approach**: Using Python API requests to gather additional genre and character information for sentiment analysis. Pagination and API throttling will be managed during requests.

3. **[Oscars Dataset](https://www.kaggle.com/unanimad/the-oscar-award)**  
   - **Content**: Oscar nomination and award data from 1927 to 2024, including details on award categories, nominees, and winners.
   - **Processing Approach**: Analysis of Oscar data to correlate critically acclaimed performances with sentiment trends and genre shifts.

4. **[IMDb Ratings](https://www.kaggle.com/thedevastator/imdb-movie-ratings-dataset)**  
   - **Content**: IMDb ratings, votes, and reviews for movies.
   - **Processing Approach**: Correlation of actor clusters and genres with box office success metrics, exploring how actor networks and genres contribute to success.

Here is the revised and more detailed version of your methods section reflecting the updates and additional content from the newly provided notebook:

---

## Methods

### 1. NLP and Sentiment Analysis
- **Text Tokenization**: Using `NLTK` to tokenize plot summaries for sentiment analysis.
- **Sentiment Categorization**: Employing VADER Sentiment Analysis to classify characters as sympathetic villains based on polarity scores.

### 2. Data Manipulation and Integration
- **Data Cleaning**: Extracting relevant features from `character.metadata.tsv` and `plot_summaries.txt`.
- **Character Analysis**: Using `pandas` to isolate and analyze prominent characters and their portrayals.

### 3. Network Analysis of Actor Collaborations
- **Graph Construction**: Using `NetworkX` to map actor collaborations and clusters.
- **Metrics**: Evaluating Actor Collaboration Frequency, Genre Diversity Score, and Network Centrality.

### 4. Predictive Modeling
- **Revenue and Ratings Prediction**: Using `XGBoost` and `sklearn` to predict box office revenues and IMDb ratings based on actor collaborations, genres, and sentiment.

### 5. Data Visualization
- **Interactive Plots**: Created with `matplotlib`, `mplcursors`, and `matplotlib-venn`.

## Project Contributions

### Visualizations
- **Genre Trends and Evolution:** Visualizing how genres have evolved over time.
- **Top Directors:** Highlighting directors and their most reviewed movies.
- **Actor Collaboration Networks:** Exploring diversity and connectivity among actors.
- **Sentiment Trends:** Analyzing antagonist portrayals and their emotional impact.

### Machine Learning Models
- **Predictive Models:** Developed models to forecast IMDb ratings and box office revenues based on:
  - Director influence
  - Actor networks
  - Genre attributes

### Interactive Features
- **Actor Collaboration Explorer:** A tool to explore actor networks and their collaborations.
- **Movie Success Simulator:** Simulates the probability of success for a potential movie.

---

## Deliverables

### Visualizations
- Genre Trends and Evolution
- Actor Collaboration Networks
- Sentiment Trends of Antagonists

### Machine Learning Models
- Predictive models for IMDb ratings and box office revenues.

### Interactive Features
- Actor Collaboration Explorer
- Movie Success Simulator

### Final Deliverables
- **Data Story**: [Here](https://mehdi1704.github.io/jekyll-theme-yat/).
- **Final Notebook**: `results.ipynb`.
- **Supporting Scripts**: For modular and clean implementation.

⚠️ **Notebook Viewer Compatibility Issue**: Interactive widgets may not render on static viewers like GitHub.

## Contributions

| Team Member      | Contribution                                   |
|-------------------|-----------------------------------------------|
| **Karine Rafla** | Cult movies research and analysis |
| **Mehdi Bouchoucha** | Project layout and website creation |
| **Mohamed Hedi Hidri** | Interactive predictor implementation |
| **Sami Amrouche** | Genres and actors research and analysis |
| **Tamara Antoun** | Cult characters research and analysis |

## Challenges and Adjusted Plans

### 1. Shift in Focus
- Original Goal: **Interactive Movie Plot Generator**.
- Final Goal: **Revenue and Ratings Predictor**. 
- **Reason**: Complexity of implementing a generative model locally with limited resources.

### 2. Freebase Data Challenges
- Solution: Mapped Freebase IDs to Wikidata entries for partial data recovery.

### 3. CoreNLP Dataset Utilization
- Plan: Study the CoreNLP pipeline further for efficient integration.

### Conclusion

This project, completed as part of the **CS-401 Applied Data Analysis course at EPFL (2024)**, explores the intricate relationships between actors, genres, and sentiment in cinema, providing unique insights into the evolving dynamics of storytelling in Hollywood. Through a combination of **network analysis**, **sentiment analysis**, and **predictive modeling**, we achieved:

- **Key Insights**: Understanding the rise of sympathetic villains, the evolution of genre trends, and the role of actor collaborations in movie success.
- **Practical Tools**: Interactive visualizations and predictive models for exploring cinematic success factors, such as IMDb ratings and box office revenues.

Despite challenges, including handling deprecated Freebase data and integrating unconventional CoreNLP datasets, our team implemented innovative solutions to deliver meaningful results.

#### Future Directions:
1. Expanding datasets to include **global cinema**, allowing for a more diverse and inclusive analysis.
2. Developing a **generative model for movie plot creation**, tailoring summaries and titles to actor profiles and sentiment trends.
3. Enhancing **interactivity in visual tools**, enabling deeper user engagement and exploration.

This work lays a foundation for further exploration of the connections between storytelling, character sentiment, and cinematic success, contributing to the broader understanding of the evolving film industry. 
