import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import altair as alt

st.set_page_config(
    page_title="PlayerxG Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Theme & Colors */
    .stApp { background-color: #0E1117; }
    [data-testid="stSidebar"] { background-color: #262730; border-right: 1px solid #333; }

    /* Header & Text */
    h1, h2, h3 { color: #FFFFFF !important; }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #FAFAFA !important; }

    /* Metric Cards */
    .metric-card {
        background-color: white;
        border: 1px solid #e0e0e0;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value { font-size: 28px; font-weight: bold; color: #007AFF; }
    .metric-label { font-size: 14px; color: #333; margin-top: 5px; font-weight: 600; }

    /* Inputs & Tags */
    span[data-baseweb="tag"] { background-color: #007AFF !important; color: white; }

    /* Chart Container */
    [data-testid="stAltairChart"] {
        background-color: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }

    /* Top 5 Buttons Styling */
    div.stButton > button {
        width: 100%;
        border: 1px solid #444;
        background-color: #333;
        color: white;
        text-align: left;
        padding: 10px;
        margin-bottom: 5px;
        border-radius: 8px;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        border-color: #007AFF;
        color: #007AFF;
        background-color: #222;
    }
    div.stButton > button:focus {
        border-color: #007AFF;
        background-color: #007AFF;
        color: white;
    }

    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@st.cache_resource
def load_resources():
    try:
        model = joblib.load('xg_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        df = pd.read_csv('shots_data.csv')

        with st.spinner('Initializing AI Engine...'):
            cleaned_texts = [clean_text(t) for t in df['text'].astype(str).tolist()]
            vectors = vectorizer.transform(cleaned_texts)
            df['nlp_xg'] = model.predict_proba(vectors)[:, 1]

        return model, vectorizer, df
    except FileNotFoundError:
        return None, None, None


model, vectorizer, df = load_resources()

if model is None:
    st.error("System Error: Model files not found. Please run 'train_model.py' first.")
    st.stop()

def extract_technique(text):
    text = text.lower()
    if 'head' in text: return 'Header'
    if 'left foot' in text: return 'Left Foot'
    if 'right foot' in text: return 'Right Foot'
    return 'Other'


def extract_zone(text):
    text = text.lower()
    if 'outside the box' in text or 'long range' in text or '30 yards' in text or '35 yards' in text:
        return 'Outside Box'
    elif 'six yard box' in text or '6 yard box' in text:
        return 'Six Yard Box'
    elif 'penalty area' in text or 'the box' in text or '18 yards' in text or '12 yards' in text:
        return 'Penalty Area'
    return 'Other'


def explain_prediction(text, model, vectorizer):
    cleaned_text = clean_text(text)
    vector = vectorizer.transform([cleaned_text])
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]

    word_contributions = []
    for col_idx in vector.nonzero()[1]:
        word = feature_names[col_idx]
        weight = coefs[col_idx]
        if weight > 0:
            word_contributions.append((word, weight))

    word_contributions.sort(key=lambda x: x[1], reverse=True)
    return word_contributions[:3]


def set_player(player_name):
    st.session_state['selected_player_name'] = player_name

with st.sidebar:
    try:
        st.image("logo.png", use_container_width=True)
    except:
        st.warning("Logo file 'logo.png' not found.")
        st.title("PlayerxG")

    st.markdown("---")
    st.markdown("<h4 style='color: #FAFAFA;'>Scout Filters</h4>", unsafe_allow_html=True)

    all_players = sorted(df['player'].dropna().unique())

    if 'selected_player_name' not in st.session_state:
        if "lionel messi" in all_players:
            st.session_state['selected_player_name'] = "lionel messi"
        else:
            st.session_state['selected_player_name'] = all_players[0]

    try:
        current_index = all_players.index(st.session_state['selected_player_name'])
    except ValueError:
        current_index = 0

    selected_player = st.selectbox(
        "Select Player:",
        all_players,
        index=current_index,
        key='selected_player_name'  # Syncs with Top 5 buttons
    )

    seasons = sorted(df['season'].unique(), reverse=True)
    selected_season = st.multiselect("Select Season(s)", seasons, default=seasons[:2])

    st.markdown("---")
    st.markdown("<h4 style='color: #007AFF;'> Top 5 xG Performers</h4>", unsafe_allow_html=True)
    st.caption(f"Based on selected season(s)")

    if selected_season:
        season_df = df[df['season'].isin(selected_season)]
    else:
        season_df = df

    top_players = season_df.groupby('player')['nlp_xg'].sum().reset_index()
    top_players = top_players.sort_values('nlp_xg', ascending=False).head(5)  # เอาแค่ 5 คน

    for i, row in top_players.iterrows():
        rank = i + 1
        player_name = row['player']
        xg_val = row['nlp_xg']
        # ปุ่มกดเพื่อเปลี่ยน Player ด้านบน
        if st.button(f"{rank}. {player_name.title()} ({xg_val:.1f} xG)", key=f"btn_{rank}", on_click=set_player,
                     args=(player_name,)):
            pass

if not selected_season:
    player_data = df[df['player'] == selected_player]
else:
    player_data = df[(df['player'] == selected_player) & (df['season'].isin(selected_season))]

st.title(f"Player Analysis: {selected_player.title()}")

if player_data.empty:
    st.info("No data found for this player in the selected seasons.")
else:
    player_data = player_data.copy()
    player_data['technique'] = player_data['text'].apply(extract_technique)
    player_data['zone'] = player_data['text'].apply(extract_zone)

    col1, col2, col3, col4 = st.columns(4)
    total_shots = len(player_data)
    total_goals = player_data['is_goal'].sum()
    total_xg = player_data['nlp_xg'].sum()
    efficiency = total_goals - total_xg

    with col1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{total_shots}</div><div class="metric-label">Total Shots</div></div>',
            unsafe_allow_html=True)
    with col2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{total_goals}</div><div class="metric-label">Actual Goals</div></div>',
            unsafe_allow_html=True)
    with col3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{total_xg:.2f}</div><div class="metric-label">Total xG (Text-Based)</div></div>',
            unsafe_allow_html=True)
    with col4:
        color = "#27ae60" if efficiency >= 0 else "#c0392b"
        sign = "+" if efficiency > 0 else ""
        st.markdown(
            f'<div class="metric-card"><div class="metric-value" style="color:{color}">{sign}{efficiency:.2f}</div><div class="metric-label">Performance vs xG</div></div>',
            unsafe_allow_html=True)

    st.markdown("---")

    col_tech, col_zone = st.columns(2)

    with col_tech:
        st.subheader("Technique Analysis (Regex)")
        tech_df = player_data['technique'].value_counts().reset_index()
        tech_df.columns = ['Technique', 'Count']

        domain_tech = ['Left Foot', 'Right Foot', 'Header', 'Other']
        range_tech = ['#3498db', '#e67e22', '#2ecc71', '#95a5a6']

        bar_chart = alt.Chart(tech_df).mark_bar(cornerRadius=3).encode(
            x=alt.X('Count', title='Number of Shots'),
            y=alt.Y('Technique', sort='-x', title=None),
            color=alt.Color('Technique', legend=None, scale=alt.Scale(domain=domain_tech, range=range_tech)),
            tooltip=['Technique', 'Count']
        ).properties(height=300)
        st.altair_chart(bar_chart + bar_chart.mark_text(align='left', dx=3).encode(text='Count'),
                        use_container_width=True)

    with col_zone:
        st.subheader("Zone Analysis (Regex)")
        zone_df = player_data['zone'].value_counts().reset_index()
        zone_df.columns = ['Zone', 'Count']

        domain_zone = ['Penalty Area', 'Outside Box', 'Six Yard Box', 'Other']
        range_zone = ['#1abc9c', '#3498db', '#f1c40f', '#95a5a6']

        zone_chart = alt.Chart(zone_df).mark_bar(cornerRadius=3).encode(
            x=alt.X('Count', title='Number of Shots'),
            y=alt.Y('Zone', sort='-x', title=None),
            color=alt.Color('Zone', legend=None, scale=alt.Scale(domain=domain_zone, range=range_zone)),
            tooltip=['Zone', 'Count']
        ).properties(height=300)
        st.altair_chart(zone_chart + zone_chart.mark_text(align='left', dx=3).encode(text='Count'),
                        use_container_width=True)

    st.markdown("---")
    st.subheader("Shot Quality Distribution (NLP xG)")

    hist_data = player_data.copy()
    hist_data['Outcome'] = hist_data['is_goal'].map({1: 'Goal', 0: 'Miss'})

    hist_chart = alt.Chart(hist_data).mark_bar().encode(
        x=alt.X('nlp_xg', bin=alt.Bin(maxbins=30), title='Goal Probability (xG)'),
        y=alt.Y('count()', title='Frequency'),
        color=alt.Color('Outcome', scale={'domain': ['Goal', 'Miss'], 'range': ['#27ae60', '#bdc3c7']}),
        tooltip=['Outcome', alt.Tooltip('count()', title='Count')]
    ).properties(height=350).interactive()

    st.altair_chart(hist_chart, use_container_width=True)

    st.markdown("---")
    st.subheader("Top 5 Highest xG Chances")

    top_chances = player_data.nlargest(5, 'nlp_xg')[['season', 'opponent', 'text', 'nlp_xg', 'is_goal']]

    for i, row in top_chances.iterrows():
        outcome_text = "GOAL" if row['is_goal'] == 1 else "MISS"
        outcome_bg = "#d4efdf" if row['is_goal'] == 1 else "#fadbd8"
        outcome_color = "#145a32" if row['is_goal'] == 1 else "#78281f"
        border_color = "#27ae60" if row['is_goal'] == 1 else "#c0392b"

        top_words = explain_prediction(row['text'], model, vectorizer)

        explanation_html = ""
        for word, weight in top_words:
            explanation_html += f'<span style="background-color: #e3f2fd; color: #1565c0; padding: 2px 8px; border-radius: 4px; margin-right: 6px; font-size: 12px; font-weight: 600; border: 1px solid #bbdefb;">{word} (+{weight:.1f})</span>'

        if not explanation_html:
            explanation_html = '<span style="color: #999; font-size: 12px;">No specific keywords detected</span>'

        st.markdown(f"""
        <div style="background-color: white; padding: 16px; border-radius: 8px; margin-bottom: 12px; border-left: 5px solid {border_color}; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 600; color: #555; font-size: 14px;">{row['season']} vs {row['opponent']}</span>
                <span style="font-weight: 700; color: {outcome_color}; background-color: {outcome_bg}; padding: 4px 10px; border-radius: 4px; font-size: 11px; letter-spacing: 0.5px;">{outcome_text}</span>
            </div>
            <div style="font-size: 15px; color: #34495e; line-height: 1.5; font-style: italic; margin-bottom: 12px;">
                "{row['text']}"
            </div>
            <div style="background-color: #f8f9fa; padding: 8px 12px; border-radius: 6px; display: flex; justify-content: space-between; align-items: center;">
                <div style="font-size: 13px; color: #7f8c8d;">
                    <strong style="color: #2c3e50;">Key Factors:</strong> {explanation_html}
                </div>
                <div style="font-size: 14px; font-weight: 700; color: #007AFF;">
                    xG: {row['nlp_xg'] * 100:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)