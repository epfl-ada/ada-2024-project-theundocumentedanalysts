
# Unfolding the Actor-Genre Constellation in Cinema: Relationships, Sentiment, and the Rise of the Sympathetic Villain

## Abstract

This project explores the evolution of actors’ careers and the portrayal of antagonists in cinema, using a blend of **network analysis**, **natural language processing (NLP)**, and **sentiment analysis**. We aim to uncover how actor career trajectories evolve across genres, the collaborative networks that shape successful film outcomes, and the rise of the "sympathetic villain" in popular cinema. Through the CMU Movie Summary Corpus and supplementary datasets, we will analyze genre shifts, actor collaboration clusters, and changing emotional tones associated with antagonists. The project intends to provide visualizations and insights into the key elements that drive cinematic success, while also offering an interactive component where users can simulate potential movie plots based on actor profiles. Our work will reveal trends in Hollywood's storytelling dynamics and demonstrate the interconnectedness of genre evolution, actor choices, and character portrayal.

## Project Structure

The directory structure of new project looks like this:

```
├── data/                     <- Project data files #IGNORED
   ├── CMU_dataset/           <- Chosen dataset
   ├── TMDB_dataset/          <- TMDB local dataset to avoid API requests
   ├── coreNLP                <- Additionnal Stanford CoreNLP-processed summaries dataset
   └── the_oscar_award.csv    <- Academy Awards: 1927 - 2024 nominees and winners dataset
   └── movie_data.csv         <- Directors, Actors, Genres, and Movies ratings
│
│
├── output_data/                                <- Processed data files
   ├── actor_sentiment_popularity_scores.csv    <- tvtropes_pipeline.py output
│
│
├── src/                               <- Source code
   ├── scripts/                        <- Data pipelines and plots
      ├── helpers_API.py               <- TMDB database API GET functions
      ├── SP_plot.py                   <- Sentiment/Popularity score plot for actors
      ├── tvtropes_pipeline.py         <- Data pipeline that processes tvtropes file
      ├── sympathetic_villain.ipynb    <- Sentiment analysis pipeline on character_metadata
      ├── oscars_movies_analysis.ipynb <- Actor/Genres constellations analysis and additional oscars implementations
      └── Hedi's file
│
│
├── freebasetowiki/             <- External Freebase converter to wikidata IDs
│
│
├── freebase_convert.py         <- Converter script from freebase to wikidata
├── .gitignore                  <- List of files ignored by git
├── requirements.txt            <- List of used libraries
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

A snapshot of this dataset was downloaded to avoid the delays associated with making multiple API requests, allowing for quicker access to data without waiting for individual responses. Since the data is not fully up-to-date, we will still use the TMDB API to supplement and update the dataset as needed. This approach combines the efficiency of using a local dataset for initial analyses with the accuracy of API updates to ensure we have the latest information on movie genres, release dates, and actor bios.

3. **Oscars**:
https://www.kaggle.com/datasets/unanimad/the-oscar-award
   - **Content**: This dataset contains Oscar nomination and award data from 1927 to 2024, including details on award categories, nominees, and winners.
   - **Processing Approach**: This dataset offers valuable insights into critically acclaimed performances. Our analysis will focus on correlating Oscar data with sentiment trends and genre shifts, particularly to examine how award-winning performances align with genre evolution and the portrayal of sympathetic villains.

4. **Movie ratings**:
https://www.kaggle.com/datasets/thedevastator/imdb-movie-ratings-dataset
   - **Content**: This dataset provides movie ratings, including IMDb ratings, votes, and reviews.
   - **Processing Approach**: This data will be used to correlate actor clusters and genres with box office success metrics, allowing us to explore how specific actor networks and genres contribute to financial and critical success. It may also help assess how sentimental tones or villain portrayals affect audience reception.

5. **Stanford CoreNLP-processed summaries**:
   - **Content**: This complementary dataset contains all of the plot summaries from above, run through the Stanford CoreNLP pipeline.
   - **Processing Approach**: We are still exploring how to use this dataset due to its unusual format and plan to study its pipeline to understand its storage structure.


These datasets will be processed to handle memory management and enrichment tasks, such as merging genres with sentiment scores and managing network data.

## Methods

1. **NLP and Sentiment Analysis**:
   - We will conduct sentiment analysis on the `plot_summaries.txt` to identify emotional tones and sympathetic/antagonistic character traits. Using Python's `NLTK` and `TextBlob`, we will extract sentiment scores and categorize them according to genres and characters.
   
2. **Network Analysis of Actor Constellations**:
   - Using `NetworkX` and `Gephi`, we will build collaboration networks, identifying clusters where actor partnerships correlate with box office or critical success. Metrics such as betweenness and eigenvector centrality will highlight influential constellations.
   
3. **Visualization**:
   - Through `Matplotlib` and `Plotly`, we will create visual graphs for actor genre evolution and sympathetic villain portrayal. Time series plots will show trends in genres and sentiment tones over time.
   
4. **Revenue and Ratings Predictor**:
   - We will develop a machine-learning-based model that predicts a movie’s potential revenue and ratings based on the past work of its actors and director. 

## Proposed Timeline - A revoir

| Phase                              | Tasks                                                                                                 | Date                         |
|------------------------------------|-------------------------------------------------------------------------------------------------------|------------------------------|
| **Week 1: Initial Setup**          | Dataset acquisition, initial analyses, and preprocessing setup                                        | Nov 5 - Nov 12               |
| **Week 2: Data Exploration & Pipeline Setup** | Perform descriptive statistics, sentiment analysis setup, and pipeline documentation. HW2 initial work begins.  | Nov 12 - Nov 15 *(P2 Deadline)* |
| **Week 3: HW2 Completion & Network Mapping**  | Complete HW2 tasks, including descriptive statistics and initial visualizations. Begin actor network mapping.  | Nov 16 - Nov 24              |
| **Week 4: HW2 Submission & Sentiment Analysis** | Finalize HW2 deliverables. Conduct sentiment analysis on plot summaries to identify sympathetic villains and genre trends.   | Nov 25 - Nov 29 *(HW2 Deadline)* |
| **Week 5: Visualization Refinement**          | Create visualizations for genre evolution, constellations, and sympathetic villain trends             | Dec 2 - Dec 8                |
| **Week 6: Interactive Feature**    | Develop and test the interactive movie plot generator, integrate into final presentation              | Dec 9 - Dec 15               |
| **Week 7: Finalization**           | Complete documentation, README update, data story creation, and project repository review             | Dec 16 - Dec 20 *(P3 Deadline)* |



## Organization within the Team

<table class="tg" style="table-layout: fixed; width: 342px">
<colgroup>
<col style="width: 160px">
<col style="width: 280px">
</colgroup>
<thead>
  <tr>
    <th class="tg-0lax">Teammate Name</th>
    <th class="tg-0lax">Contributions</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">Karine Rafla </td>
    <td class="tg-0lax">Reports redaction<br>Research on Freebase</td>
  </tr>
  <tr>
    <td class="tg-0lax">Mehdi Bouchoucha </td>
    <td class="tg-0lax">Project layout<br>Additionnal datasets research</td>
  </tr>
    <tr>
    <td class="tg-0lax">Mohamed Hedi Hidri </td>
    <td class="tg-0lax"> Work on interactive predictor<br>Additionnal datasets research</td>
  </tr>
  <tr>
    <td class="tg-0lax">Sami Amrouche </td>
    <td class="tg-0lax">Constellation work<br>Oscars analysis</td>
  </tr>
  <tr>
    <td class="tg-0lax">Tamara Antoun </td>
    <td class="tg-0lax">Work on sympathetic villain<br>Reports redaction</td>
  </tr>
</tbody>
</table>

**Internal Milestones**:
- **Nov 12**: Dataset acquisition and preprocessing setup completed, with initial analyses and descriptive statistics in place.
- **Nov 15**: **P2 Submission** – README.md with detailed project proposal, preliminary analyses in a Jupyter notebook.
- **Nov 24**: Completion of constellation network mapping and success analysis.
- **Dec 1**: Completion of sentiment analysis and draft visualizations.
- **Dec 8**: Completion of interactive feature and final visualization drafts.
- **Dec 15**: Finalize all features, complete documentation, and prepare for **P3 submission**.
- **Dec 20**: **P3 Submission** – Final project code, results notebook, and data story URL.

## Challenges Faced and Adjusted Plans

**Shift in Focus: From Interactive Movie Plot Generator to Revenue and Ratings Predictor**:

Our original idea was to develop an Interactive Movie Plot Generator: a machine-learning-based tool that would propose new movie titles and summaries tailored to an actor’s typical roles by analyzing sentiment and genre data. The plan involved a generative model using text from plot summaries to align with an actor’s emotional and genre-based patterns.

However, we decided to pivot to a Revenue and Ratings Predictor instead. This new approach focuses on forecasting potential revenue and ratings by analyzing historical data from actors and directors, leveraging sentiment, genre, and other key attributes. The switch was driven by the realization that implementing a generative model locally would require extensive resources and introduce technical complexities that were outside our current scope.

This adjusted approach allowed us to focus on achievable predictive insights while still working with rich data on actors, directors, and genre alignment.

**Handling Freebase Data in Our Dataset**

Our initial dataset includes numerous Freebase IDs. Freebase, originally owned by a different company, was acquired by Google and shut down in 2015. Although there is no longer API access to Freebase, we were able to locate a data dump from the last snapshot of the service. This dump contains links that refer to other Freebase IDs, but without direct API access, it posed challenges for retrieving relevant data.

After extensive research, we implemented an external converter that maps Freebase IDs to Wikidata entries. While this converter cannot resolve all references due to the loss of some data after Freebase’s shutdown, it still enables us to retrieve a portion of the information that would otherwise be inaccessible. This workaround allows us to recover some of the lost data and continue enriching our dataset with valuable external information.

**CoreNLP Dataset**

This complementary dataset contains all of the plot summaries mentioned earlier, processed through the Stanford CoreNLP pipeline. We are still exploring how to best utilize this dataset due to its unconventional format. Our plan is to study the pipeline further in order to understand its storage structure and determine the most efficient way to integrate it into our project.
