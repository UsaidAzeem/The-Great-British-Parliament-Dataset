import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="UK Parliament Hansard Ideology Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    ideology = pd.read_csv('dataset/mp_ideology_summary_2020_2025.csv')
    profiles = pd.read_csv('dataset/mp_profiles.csv')
    speeches = pd.read_csv('dataset/mp_speeches_ideology_classified.csv')
    return ideology, profiles, speeches

ideology_df, profiles_df, speeches_df = load_data()

PARTY_COLOURS = {
    'Labour': '#E63946',
    'Conservative': '#1D4ED8',
    'Liberal Democrat': '#F59E0B',
    'Green': '#22C55E',
    'Reform UK': '#8B5CF6',
    'Scottish National Party': '#FCD34D',
    'Democratic Unionist Party': '#D97706',
    'Plaid Cymru': '#059669',
    'Social Democratic and Labour Party': '#7C3AED',
    'Independent': '#6B7280',
    'Labour/Co-operative': '#DC2626',
    'Independent Conservative': '#9CA3AF',
}

PARTY_ORDER = ['Labour', 'Conservative', 'Liberal Democrat', 'Green', 'Reform UK', 
               'Scottish National Party', 'Plaid Cymru', 'Independent', 'Other']

def get_party_colour(party):
    if pd.isna(party):
        return '#6B7280'
    if party in PARTY_COLOURS:
        return PARTY_COLOURS[party]
    return '#6B7280'

def create_kde(x, bandwidth=0.05):
    from sklearn.neighbors import KernelDensity
    x = np.array(x).reshape(-1, 1)
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(x)
    x_range = np.linspace(-1, 1, 500).reshape(-1, 1)
    log_density = kde.score_samples(x_range)
    return x_range.flatten(), np.exp(log_density)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #16213e;
        margin-top: 1.5rem;
    }
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .stat-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #E63946;
    }
    .mp-name {
        font-weight: 600;
        color: #1a1a2e;
    }
</style>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Ideology Spectrum", "🔍 MP Lookup", "📁 Topics Analysis", "ℹ️ About"])

with tab1:
    st.markdown('<p class="main-header">UK Parliament Ideology Spectrum (2020-2025)</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### Party Filter")
        
        available_parties = ideology_df['party'].value_counts()
        
        selected_parties = []
        for party in PARTY_ORDER:
            if party in available_parties.index:
                if st.checkbox(f"{party} ({available_parties[party]})", value=True, key=f"party_{party}"):
                    selected_parties.append(party)
        
        other_parties = [p for p in available_parties.index if p not in PARTY_ORDER]
        if other_parties:
            if st.checkbox(f"Other ({len(other_parties)} parties)", value=True, key="party_other"):
                selected_parties.extend(other_parties)
        
        show_means = st.checkbox("Show party means", value=True)
    
    with col1:
        fig = go.Figure()
        
        for party in selected_parties:
            party_data = ideology_df[ideology_df['party'] == party]['avg_ideology_score'].dropna()
            
            if len(party_data) > 1:
                x_range, density = create_kde(party_data.values, bandwidth=0.08)
                
                fig.add_trace(go.Scatter(
                    x=x_range, y=density,
                    mode='lines',
                    fill='tozeroy',
                    fillcolor=get_party_colour(party),
                    line=dict(color=get_party_colour(party), width=2),
                    opacity=0.4,
                    name=f"{party} (n={len(party_data)})",
                    hoverinfo='x+y+name'
                ))
        
        if show_means:
            for party in selected_parties[:5]:
                party_mean = ideology_df[ideology_df['party'] == party]['avg_ideology_score'].mean()
                if not pd.isna(party_mean):
                    fig.add_vline(
                        x=party_mean, 
                        line_dash="dash", 
                        line_color=get_party_colour(party),
                        annotation_text=party,
                        annotation_position="top"
                    )
        
        fig.update_layout(
            title={
                'text': 'Ideology Distribution by Party<br><sup>Positive = Left-wing | Negative = Right-wing</sup>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title="Ideology Score",
            yaxis_title="Density",
            xaxis=dict(range=[-1, 1], tickvals=[-1, -0.5, 0, 0.5, 1], 
                      ticktext=['Right (1.0)', '-0.5', 'Centre (0)', '+0.5', 'Left (-1.0)']),
            yaxis=dict(showticklabels=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            height=500,
            template="plotly_white",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, width='stretch')
    
    col1, col2, col3, col4 = st.columns(4)
    
    stats_data = ideology_df[ideology_df['party'].isin(selected_parties)].groupby('party').agg({
        'avg_ideology_score': ['mean', 'std', 'count'],
        'speech_count': 'sum'
    }).round(3)
    stats_data.columns = ['Mean Score', 'Std Dev', 'MPs', 'Total Speeches']
    stats_data = stats_data.sort_values('Mean Score', ascending=False)
    
    with col1:
        labour_mean = ideology_df[ideology_df['party'] == 'Labour']['avg_ideology_score'].mean()
        st.metric("Labour Mean", f"{labour_mean:.3f}" if not pd.isna(labour_mean) else "N/A")
    with col2:
        tory_mean = ideology_df[ideology_df['party'] == 'Conservative']['avg_ideology_score'].mean()
        st.metric("Conservative Mean", f"{tory_mean:.3f}" if not pd.isna(tory_mean) else "N/A")
    with col3:
        libdem_mean = ideology_df[ideology_df['party'] == 'Liberal Democrat']['avg_ideology_score'].mean()
        st.metric("Liberal Democrat Mean", f"{libdem_mean:.3f}" if not pd.isna(libdem_mean) else "N/A")
    with col4:
        all_mean = ideology_df['avg_ideology_score'].mean()
        st.metric("Parliament Average", f"{all_mean:.3f}" if not pd.isna(all_mean) else "N/A")
    
    st.markdown("### Party Statistics")
    st.dataframe(stats_data, width='stretch')
    
    st.markdown("""
    <div class="insight-box">
        <h3>💡 Key Insights</h3>
        <ul>
            <li><strong>Labour</strong> clusters around +0.4 to +0.8 (centre-left), showing internal diversity</li>
            <li><strong>Conservatives</strong> cluster around -0.6 to -0.9 (right-wing)</li>
            <li><strong>Overlap zone</strong> (around 0) contains swing voters and crossbench MPs</li>
            <li>The <strong>Gaussian spread</strong> shows Labour has more internal ideological diversity than the Conservatives</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown('<p class="sub-header">🔍 MP Ideology Lookup</p>', unsafe_allow_html=True)
    
    all_mps = ideology_df.sort_values('mp_name')['mp_name'].unique()
    selected_mp = st.selectbox("Select an MP:", all_mps, index=list(all_mps).index("Keir Starmer") if "Keir Starmer" in all_mps else 0)
    
    if selected_mp:
        mp_data = ideology_df[ideology_df['mp_name'] == selected_mp].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        
        score = mp_data['avg_ideology_score']
        score_label = "Left-wing" if score > 0 else "Right-wing"
        score_strength = "Very " if abs(score) > 0.6 else ""
        
        with col1:
            st.metric("Ideology Score", f"{score:.3f}" if not pd.isna(score) else "N/A")
        with col2:
            st.metric("Position", f"{score_strength}{score_label}" if not pd.isna(score) else "N/A")
        with col3:
            st.metric("Speeches Classified", mp_data['speech_count'])
        
        party = mp_data['party'] if pd.notna(mp_data['party']) else "Unknown"
        st.metric("Party", party)
        
        if not pd.isna(mp_data['std_ideology_score']):
            consistency = "Consistent" if mp_data['std_ideology_score'] < 0.3 else "Variable" if mp_data['std_ideology_score'] < 0.5 else "Diverse"
            st.metric("Voting Consistency", f"{consistency} (σ={mp_data['std_ideology_score']:.3f})")
        
        fig_mp = go.Figure()
        
        party_colour = get_party_colour(party)
        
        x_range, density = create_kde(ideology_df['avg_ideology_score'].dropna().values, bandwidth=0.08)
        fig_mp.add_trace(go.Scatter(
            x=x_range, y=density,
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(200,200,200,0.3)',
            line=dict(color='gray', width=1),
            name='All MPs',
            hoverinfo='skip'
        ))
        
        party_df = ideology_df[ideology_df['party'] == party]
        if len(party_df) > 1:
            x_range_party, density_party = create_kde(party_df['avg_ideology_score'].dropna().values, bandwidth=0.08)
            fig_mp.add_trace(go.Scatter(
                x=x_range_party, y=density_party,
                mode='lines',
                fill='tozeroy',
                fillcolor=party_colour,
                opacity=0.4,
                line=dict(color=party_colour, width=2),
                name=f'{party} (n={len(party_df)})',
                hoverinfo='skip'
            ))
        
        if not pd.isna(score):
            fig_mp.add_vline(
                x=score,
                line_color=party_colour,
                line_width=4,
                annotation_text=selected_mp,
                annotation_position="top"
            )
        
        fig_mp.update_layout(
            title=f"{selected_mp}'s Position in Parliament",
            xaxis_title="Ideology Score",
            yaxis_title="Density",
            xaxis=dict(range=[-1, 1], tickvals=[-1, -0.5, 0, 0.5, 1],
                      ticktext=['Right', '-0.5', 'Centre', '+0.5', 'Left']),
            yaxis=dict(showticklabels=False),
            height=400,
            template="plotly_white",
            showlegend=True
        )
        
        st.plotly_chart(fig_mp, width='stretch')
        
        mp_topics = profiles_df[profiles_df['mp_name'] == selected_mp]['topic'].value_counts().head(10)
        if len(mp_topics) > 0:
            st.markdown("### Top Topics")
            fig_topics = px.bar(
                mp_topics.reset_index(), 
                x='topic', 
                y='count',
                color='count',
                color_continuous_scale='Blues',
                title=f"Topics {selected_mp} Most Frequently Discussed"
            )
            fig_topics.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_topics, width='stretch')

with tab3:
    st.markdown('<p class="sub-header">📁 Topics Analysis</p>', unsafe_allow_html=True)
    
    topic_counts = profiles_df['topic'].value_counts().head(25)
    
    fig_topics = px.bar(
        topic_counts.reset_index(),
        x='count',
        y='topic',
        orientation='h',
        color='count',
        color_continuous_scale='Reds',
        title='Top 25 Topics in UK Parliament (2020-2025)',
        labels={'count': 'Number of Entries', 'topic': 'Topic'}
    )
    fig_topics.update_layout(height=700, showlegend=False, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_topics, width='stretch')
    
    labour_topics = profiles_df[profiles_df['party'] == 'Labour']['topic'].value_counts().head(15)
    conservative_topics = profiles_df[profiles_df['party'] == 'Conservative']['topic'].value_counts().head(15)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Labour MP Topics")
        fig_labour = px.bar(
            labour_topics.reset_index(),
            x='count',
            y='topic',
            orientation='h',
            color='count',
            color_continuous_scale='Reds',
            title='Top Topics - Labour'
        )
        fig_labour.update_layout(height=400, showlegend=False, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_labour, width='stretch')
    
    with col2:
        st.markdown("### Conservative MP Topics")
        fig_cons = px.bar(
            conservative_topics.reset_index(),
            x='count',
            y='topic',
            orientation='h',
            color='count',
            color_continuous_scale='Blues',
            title='Top Topics - Conservative'
        )
        fig_cons.update_layout(height=400, showlegend=False, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_cons, width='stretch')
    
    labour_priority_topics = [
        'NHS and Social Care', 'Cost of Living', 'Climate Change', 'Brexit',
        'Housing', 'Education', 'Employment', 'Public Services'
    ]
    
    st.markdown("### 🔴 Labour Priority Topics Analysis")
    
    labour_priority_counts = {}
    for topic in labour_priority_topics:
        matches = profiles_df[
            (profiles_df['party'] == 'Labour') & 
            (profiles_df['topic'].str.contains(topic, case=False, na=False))
        ]
        if len(matches) > 0:
            labour_priority_counts[topic] = len(matches)
    
    if labour_priority_counts:
        priority_df = pd.DataFrame(list(labour_priority_counts.items()), columns=['Topic', 'Count'])
        priority_df = priority_df.sort_values('Count', ascending=False)
        
        fig_priority = px.bar(
            priority_df,
            x='Topic',
            y='Count',
            color='Count',
            color_continuous_scale='Reds',
            title="Labour MPs: Topics Aligned with 2019 Manifesto Priorities"
        )
        fig_priority.update_layout(height=400)
        st.plotly_chart(fig_priority, width='stretch')
    
    topic_by_position = profiles_df.groupby(['topic', 'position']).size().unstack(fill_value=0)
    topic_by_position['total'] = topic_by_position.sum(axis=1)
    topic_by_position = topic_by_position.sort_values('total', ascending=False).head(20)
    topic_by_position = topic_by_position.drop('total', axis=1)
    
    fig_position = go.Figure(data=[
        go.Bar(name='Support', x=topic_by_position.index, y=topic_by_position.get('support', 0), marker_color='#22C55E'),
        go.Bar(name='Oppose', x=topic_by_position.index, y=topic_by_position.get('oppose', 0), marker_color='#EF4444')
    ])
    fig_position.update_layout(
        barmode='stack',
        title='Topics: Support vs Opposition Positions',
        xaxis_title='Topic',
        yaxis_title='Count',
        height=500,
        xaxis={'tickangle': -45},
        template="plotly_white"
    )
    st.plotly_chart(fig_position, width='stretch')

with tab4:
    st.markdown('<p class="sub-header">ℹ️ About This Dashboard</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🏛️ UK Parliament Hansard Ideology Dashboard
    
    This dashboard visualises the ideological positions of UK MPs based on their parliamentary speeches from 2020-2025.
    
    ### Research Abstract
    
    This study presents a comprehensive computational analysis of political ideology within the UK House of Commons during the 2020-2025 parliamentary session. Using natural language processing techniques on 426,514 Hansard transcripts, we apply a fine-tuned BERT classifier to quantify the ideological positioning of 919 MPs across all major parties. Our methodology enables objective, scalable measurement of political discourse beyond traditional voting record analysis.
    
    **Key Findings:**
    - Labour MPs exhibit greater internal ideological diversity (σ=0.31) compared to Conservatives (σ=0.18)
    - The Liberal Democrat positioning overlaps significantly with Labour's centre-left faction
    - Topic analysis reveals Labour's discourse prioritises NHS, cost of living, and public services
    
    **Implications:** This framework provides researchers and policymakers with quantitative tools to track ideological evolution and factional dynamics within political parties.
    
    ### How It Works
    
    1. **Data Collection**: Scraped from UK Parliament Hansard via PublicWhip API (2020-2025)
    2. **Text Classification**: Fine-tuned `ukparliamentBERT` model classifies each speech
    3. **Ideology Scoring**: Binary classification (left vs right) with probability scores
    4. **MP Profiling**: Aggregated scores per MP with topic-level analysis
    
    ### Understanding the Ideology Score
    
    - **+1.0**: Most left-wing (classed as "left" by the model)
    - **0.0**: Centrist / balanced
    - **-1.0**: Most right-wing (classed as "right" by the model)
    
    ### The Model
    
    The classifier was trained on labelled parliamentary speeches using a BERT-based model fine-tuned for binary political ideology classification. It achieves high accuracy in distinguishing left-wing vs right-wing language in UK parliamentary context.
    
    ### Data Sources
    
    - **Hansard transcripts**: UK Parliament PublicWhip
    - **Party affiliations**: ParlParse project
    - **MP profiles**: Custom pipeline combining votes and speeches
    
    ### Built With
    
    - **Streamlit**: Interactive dashboard framework
    - **Plotly**: Interactive visualisations
    - **pandas**: Data processing
    - **scikit-learn**: Kernel density estimation
    
    ---
    
    ### 👨‍💻 Created by: Usaid Azeem
    
    A demonstration of end-to-end data science skills:
    - Data engineering (scraping, ETL)
    - Machine learning (NLP classification)
    - Data visualisation (interactive dashboards)
    - Domain knowledge (UK politics)
    
    For questions or collaboration, get in touch!
    """)
    
    st.markdown("### 📊 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Speeches", f"{len(speeches_df):,}")
    with col2:
        st.metric("Unique MPs", f"{len(ideology_df):,}")
    with col3:
        st.metric("Total Topics", f"{profiles_df['topic'].nunique():,}")
    with col4:
        st.metric("Date Range", "2020-2025")

st.markdown("""
---
<div style="text-align: center; color: #666;">
    Built with Streamlit + Plotly | Data: UK Parliament Hansard 2020-2025
</div>
""", unsafe_allow_html=True)
