import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.preprocessing import StandardScaler
import xgboost

# Load the best XGBoost model
with open('src/drafts/best_xgb_model.pkl', 'rb') as f:
    best_xgb = pickle.load(f)

with open('src/drafts/feature_columns.json', 'rb') as f:
    feature_columns = json.load(f)

# Load the scaler used during training
with open('src/drafts/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


numeric_features = ['movie_runtime', 'BUDGET']

# Identify actor and director columns from the feature_columns
actor_dummy_cols = [c for c in feature_columns if c.startswith('dummy_actor_')]
director_dummy_cols = [c for c in feature_columns if c.startswith('dummy_director_')]

# Strip the dummy prefix for display purposes only
actor_columns = [c.replace('dummy_actor_', '') for c in actor_dummy_cols]
director_columns = [c.replace('dummy_director_', '') for c in director_dummy_cols]

# Create sets for genres, languages, and countries

genre_set = {
    "Absurdism","Action","Action Comedy","Action Thrillers","Action/Adventure","Addiction Drama","Adult",
    "Adventure","Adventure Comedy","Airplanes and airports","Albino bias","Alien Film","Alien invasion",
    "Americana","Animal Picture","Animals","Animated Musical","Animated cartoon","Animation","Anime",
    "Anthology","Anti-war","Anti-war film","Apocalyptic and post-apocalyptic fiction","Archives and records",
    "Art film","Auto racing","Avant-garde","B-movie","Backstage Musical","Baseball","Beach Film","Biker Film",
    "Biographical film","Biography","Biopic [feature]","Black comedy","Black-and-white","Blaxploitation",
    "Bollywood","Boxing","British Empire Film","British New Wave","Buddy cop","Buddy film","Camp","Caper story",
    "Cavalry Film","Chase Movie","Childhood Drama","Children","Chinese Movies","Christian film","Christmas movie",
    "Cold War","Combat Films","Comedy","Comedy Thriller","Comedy Western","Comedy film","Comedy horror",
    "Comedy of Errors","Comedy of manners","Comedy-drama","Coming of age","Computer Animation","Concert film",
    "Conspiracy fiction","Costume Adventure","Costume Horror","Costume drama","Courtroom Comedy","Courtroom Drama",
    "Creature Film","Crime","Crime Comedy","Crime Drama","Crime Fiction","Crime Thriller","Cult","Culture & Society",
    "Cyberpunk","Dance","Demonic child","Detective","Detective fiction","Disaster","Docudrama","Documentary","Dogme 95",
    "Domestic Comedy","Doomsday film","Drama","Dystopia","Educational","Ensemble Film","Environmental Science","Epic",
    "Epic Western","Erotic Drama","Erotic thriller","Erotica","Escape Film","Existentialism","Experimental film",
    "Extreme Sports","Fairy tale","Family Drama","Family Film","Family-Oriented Adventure","Fantasy","Fantasy Adventure",
    "Fantasy Comedy","Fantasy Drama","Female buddy film","Feminist Film","Film \\u00e0 clef","Film adaptation",
    "Film noir","Future noir","Gangster Film","Gay","Gay Interest","Gay Themed","Gay pornography","Glamorized Spy Film",
    "Gothic Film","Gross out","Gross-out film","Gulf War","Hagiography","Haunted House Film","Heaven-Can-Wait Fantasies",
    "Heavenly Comedy","Heist","Hip hop movies","Historical Epic","Historical drama","Historical fiction","History",
    "Holiday Film","Horror","Horror Comedy","Horse racing","Humour","Hybrid Western","Indian Western","Indie",
    "Inspirational Drama","Interpersonal Relationships","Inventions & Innovations","Japanese Movies","Jukebox musical",
    "Jungle Film","Juvenile Delinquency Film","Kafkaesque","LGBT","Legal drama","Live action","Marriage Drama",
    "Martial Arts Film","Master Criminal Films","Media Satire","Medical fiction","Melodrama","Mockumentary","Monster",
    "Monster movie","Mumblecore","Music","Musical","Musical Drama","Musical comedy","Mystery","Mythological Fantasy",
    "Natural disaster","Natural horror films","Nature","Neo-noir","New Hollywood","Ninja movie","Outlaw","Parkour in popular culture",
    "Parody","Period piece","Plague","Point of view shot","Political cinema","Political drama","Political satire",
    "Political thriller","Pornographic movie","Pre-Code","Prison","Prison film","Private military company","Propaganda film",
    "Psycho-biddy","Psychological horror","Psychological thriller","Punk rock","Reboot","Religious Film","Remake",
    "Revisionist Western","Road movie","Road-Horror","Roadshow theatrical release","Rockumentary","Romance Film","Romantic comedy",
    "Romantic drama","Romantic fantasy","Samurai cinema","Satire","School story","Sci-Fi Adventure","Sci-Fi Horror","Sci-Fi Thriller",
    "Science Fiction","Screwball comedy","Sex comedy","Sexploitation","Short Film","Slapstick","Slasher","Slice of life story",
    "Social issues","Social problem film","Space opera","Space western","Spaghetti Western","Splatter film","Sports","Spy",
    "Stand-up comedy","Star vehicle","Steampunk","Stoner film","Stop motion","Superhero","Superhero movie","Supermarionation",
    "Supernatural","Surrealism","Suspense","Swashbuckler films","Sword and Sandal","Sword and sorcery","Sword and sorcery films",
    "Teen","Television movie","The Netherlands in World War II","Therimin music","Thriller","Time travel","Tragedy","Tragicomedy",
    "War film","Werewolf fiction","Western","Whodunit","Workplace Comedy","World cinema","Wuxia","Zombie Film"
}

language_set = {
    'Aboriginal Malay languages','Afrikaans Language','Albanian language','Algonquin Language','American English',
    'American Sign Language','Amharic Language','Ancient Greek','Apache, Western Language','Arabic Language','Aramaic language',
    'Armenian Language','Assyrian Neo-Aramaic Language','Assyrian language','Bengali Language','Bosnian language','Brazilian Portuguese',
    'Bulgarian Language','Cantonese','Catalan language','Chewa language','Chinese language','Corsican Language','Croatian language',
    'Czech Language','Danish Language','Dari','Dutch Language','English Language','Esperanto Language','Estonian Language',
    'Filipino language','Finnish Language','French Language','Gaelic','Galician Language','Georgian Language','German','German Language',
    'Greek Language','Hawaiian language','Hebrew Language','Hindi Language','Hmong language','Hokkien','Hungarian language','Icelandic Language',
    'Inuktitut','Irish','Italian','Italian Language','Japanese Language','Khmer language','Klingon language','Korean Language',
    'Korean Sign Language','Krio Language','Kurdish language','Latin Language','Luxembourgish language','M\\u0101ori language',
    'Malay Language','Malayalam Language','Mandarin Chinese','Maya, Yucat\\u00e1n Language','Mende Language','Min Nan','Navajo Language',
    'Norwegian Language','Old English language','Papiamento language','Pawnee Language','Persian Language','Polish Language',
    'Portuguese Language','Punjabi language','Romani language','Romanian Language','Russian Language','Scottish Gaelic language',
    'Serbian language','Serbo-Croatian','Shanghainese','Sicilian Language','Sinhala Language','Sioux language','Slovak Language',
    'Somali Language','Sotho language','Spanish Language','Standard Cantonese','Standard Mandarin','Standard Tibetan','Sumerian',
    'Swahili Language','Swedish Language','Swiss German Language','Tagalog language','Taiwanese','Tamil Language','Thai Language',
    'Tibetan languages','Turkish Language','Ukrainian Language','Urdu Language','Vietnamese Language','Welsh Language','Xhosa Language',
    'Yiddish Language','Zulu Language'
}

country_set = {
    'Argentina','Australia','Austria','Belgium','Brazil','Bulgaria','Canada','China','Colombia','Costa Rica','Cuba','Czech Republic',
    'Denmark','England','Finland','Germany','Greece','Hong Kong','Hungary','Iceland','India','Indonesia','Iran','Ireland','Israel',
    'Italy','Japan','Kingdom of Great Britain','Korea','Libya','Luxembourg','Malaysia','Malta','Mexico','Netherlands','New Zealand',
    'Norway','Pakistan','Panama','Peru','Poland','Portugal','Romania','Russia','Serbia','Singapore','Slovakia','Slovenia','South Africa',
    'South Korea','Soviet Union','Spain','Sweden','Switzerland','Taiwan','Thailand','Tunisia','Turkey','United Arab Emirates','United Kingdom',
    'United States of America','Uruguay','West Germany'
}

genre_set = {g for g in genre_set if g in feature_columns}
language_set = {l for l in language_set if l in feature_columns}
country_set = {c for c in country_set if c in feature_columns}
# Extract actual columns from feature_columns for genres, languages, and countries
genre_columns = [c for c in feature_columns if c in genre_set]
language_columns = [c for c in feature_columns if c in language_set]
country_columns = [c for c in feature_columns if c in country_set]

st.title("IMDb Score Predictor")
st.markdown("Provide movie details and get a predicted IMDb score.")

# Numeric Inputs
st.header("🔢 Basic Movie Info")
movie_runtime_input = st.number_input("Movie Runtime (minutes)", min_value=1, value=100)
budget_input = st.number_input("Budget (USD)", min_value=1000, value=5000000)

# Actor and Director Selection
st.header("🎭 Cast and Crew")
selected_actors = st.multiselect("Select Actors", actor_columns)
selected_directors = st.multiselect("Select Directors", director_columns)

# Genre Selection
st.header("🎥 Genres")
selected_genres = st.multiselect("Select Genres", genre_columns)

# Language Selection
st.header("🌐 Languages")
selected_languages = st.multiselect("Select Languages", language_columns)

# Country Selection
st.header("🏳️ Production Country")
selected_country = st.selectbox("Select Country", [None] + country_columns)

def prepare_input():
    # Create a single-row DataFrame with all features set to 0
    input_data = pd.DataFrame(columns=feature_columns, data=np.zeros((1, len(feature_columns))))
    
    # Assign numeric values
    input_data['movie_runtime'] = movie_runtime_input
    input_data['BUDGET'] = budget_input
    
    # Actors
    for a in selected_actors:
        dummy_col = f'dummy_actor_{a}'
        if dummy_col in input_data.columns:
            input_data[dummy_col] = 1
        
    # Directors
    for d in selected_directors:
        dummy_col = f'dummy_director_{d}'
        if dummy_col in input_data.columns:
            input_data[dummy_col] = 1
        
    # Genres
    for g in selected_genres:
        if g in input_data.columns:
            input_data[g] = 1
        
    # Languages
    for lang in selected_languages:
        if lang in input_data.columns:
            input_data[lang] = 1
        
    # Country
    if selected_country and selected_country in input_data.columns:
        input_data[selected_country] = 1
    
    # Reorder columns to match training
    input_data = input_data[feature_columns]
    
    input_data_scaled = scaler.transform(input_data)
    input_data = pd.DataFrame(input_data_scaled, columns=feature_columns)

    return input_data

# Prediction Button
st.markdown(
    """
    <style>
    .predict-button {
        background-color: #FF0000; /* Red color */
        color: white;
        padding: 10px 15px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
    }
    .predict-button:hover {
        background-color: #D00000; /* Slightly darker red on hover */
    }
    </style>
    <button class="predict-button" onclick="predict()">📊 Predict IMDb Score</button>
    <script>
    function predict() {
        const el = document.querySelector('button[kind="secondary"]');
        if (el) el.click();
    }
    </script>
    """,
    unsafe_allow_html=True
)

if st.button("", key="secondary"):
    if best_xgb is None:
        st.error("Model not loaded. Please load the model before predicting.")
    else:
        input_df = prepare_input()
        prediction = best_xgb.predict(input_df)[0]
        st.success(f"Predicted IMDb score: {prediction:.2f}")

# Direct Navigation Button
st.markdown(
    """<a href="https://mehdi1704.github.io/jekyll-theme-yat/" target="_blank" style="text-decoration:none;">
    <button style="background-color:#4CAF50; color:white; padding:10px 15px; border:none; border-radius:5px; cursor:pointer;">
    🌐 Come Back to the Site
    </button>
    </a>""",
    unsafe_allow_html=True
)
