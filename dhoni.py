import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
def dhoni_title():
  return st.title("MS Dhoni T20 International Analysis")

@st.cache_data
def load_data():
    return pd.read_csv('ball_by_ball_it20.csv')

@st.cache_data
def dhoni_performance():
    t20i = load_data()
    full_name = ''
    for name in t20i['Batter'].unique():
        if 'dhoni' in name.lower():
            full_name = name
            break
    st.write(full_name)

    dhoni_t20i = t20i[t20i['Batter'] == full_name]

    recommended_columns = ['Match ID', 'Date', 'Venue', 'Bat First', 'Bat Second', 'Innings', 'Over', 'Ball', 'Bowler', 
                        'Batter Runs', 'Runs From Ball', 'Batter Balls Faced', 'Wicket', 'Method', 'Player Out',
                        'Player Out Runs', 'Player Out Balls Faced', 'Winner', 'Chased Successfully']

    dhoni_t20i = dhoni_t20i[recommended_columns]
    st.dataframe(dhoni_t20i.sample(5))
    st.write(dhoni_t20i.shape)

    match_id_mapping = {}
    match_id_counter = 1

    for match_id in dhoni_t20i['Match ID'].unique():
        match_id_mapping[match_id] = f"Match {match_id_counter}"
        match_id_counter += 1

    dhoni_t20i['Match ID'] = dhoni_t20i['Match ID'].map(match_id_mapping)

    matches_played = dhoni_t20i['Match ID'].nunique()
    total_runs_scored = dhoni_t20i['Batter Runs'].sum()

    dhoni_t20i['Date'] = pd.to_datetime(dhoni_t20i['Date']).dt.date
    dhoni_t20i['year'] = dhoni_t20i['Date'].apply(lambda x: x.year)
    matches_played_by_year = dhoni_t20i.groupby(by='year')['Match ID'].nunique().reset_index()
    # st.write("MS Dhoni's Match Played Year by Year")
    matches_played_by_year.columns = ['year', 'matches_played']

    st.title("MS Dhoni's Matches Played Year by Year")
    plt.figure(figsize=(9, 4))
    sns.set_style("darkgrid")
    sns.lineplot(data=matches_played_by_year, x='year', y='matches_played')
    plt.title("Matches Played by Year")
    st.pyplot(plt)

    runs_by_year = dhoni_t20i.pivot_table(index='year', values='Batter Runs', aggfunc='sum').reset_index()
    runs_by_year['Percentage Change'] = np.round(runs_by_year['Batter Runs'].pct_change() * 100)

    st.title("MS Dhoni's Runs Year by Year")
    plt.figure(figsize=(9, 4))
    sns.set_style("darkgrid")
    sns.lineplot(data=runs_by_year, x='year', y='Batter Runs')
    plt.title("Runs Scored by Year")
    st.pyplot(plt)

    venue_stats = dhoni_t20i.groupby('Venue').agg({
        'Batter Runs': 'sum',
        'Batter Balls Faced': 'count'
    }).reset_index()
    venue_stats['Strike Rate'] = np.round(100 * venue_stats['Batter Runs'] / venue_stats['Batter Balls Faced'])
    venue_stats = venue_stats.sort_values(by='Batter Runs', ascending=False)

    st.title("MS Dhoni's Top 5 Favourite Stadiums to Score")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=venue_stats.head(5), y='Venue', x='Batter Runs', orient='h')
    sns.despine(bottom=True, trim=True)
    st.pyplot(plt)

    runs_by_innings = dhoni_t20i.groupby(by='Innings').agg({'Batter Runs': 'sum', 'Batter Balls Faced': 'count'})
    runs_by_innings['Strike Rate'] = np.round((100 * runs_by_innings['Batter Runs'] / runs_by_innings['Batter Balls Faced']))
    labels = ['Innings 1', 'Innings 2']
    x = runs_by_innings['Batter Runs']
    explode = [0.1, 0]
    st.title("MS Dhoni's Run Distribution Innings Wise")
    plt.figure(figsize=(8, 4))
    plt.pie(labels=labels, x=x, autopct='%1.1f%%', explode=explode)
    st.pyplot(plt)

    over_runs = dhoni_t20i.groupby(by='Over').agg({"Batter Runs": 'sum'}).fillna(0).reset_index()
    st.title("MS Dhoni's Top 5 Overs Where He Scored Maximum Runs")
    plt.figure(figsize=(8, 4))
    over_runs = over_runs.sort_values(by='Batter Runs', ascending=False)
    sns.barplot(data=over_runs.head(5), x='Over', y='Batter Runs')
    st.pyplot(plt)

    runs_bowlers = dhoni_t20i.groupby(by='Bowler').agg({
        'Batter Runs': 'sum',
        'Match ID': 'count'
    }).reset_index()
    runs_bowlers['Strike Rate'] = np.round(100 * runs_bowlers['Batter Runs'] / runs_bowlers['Match ID'])
    runs_bowlers.columns = ['Bowler', 'Batter Runs', 'Balls Faced', 'Strike Rate']
    runs_bowlers = runs_bowlers.sort_values(by=['Batter Runs', 'Strike Rate'], ascending=False)

    st.title("MS Dhoni's Most Runs Against Different Bowlers")
    plt.figure(figsize=(8, 4))
    sns.barplot(data=runs_bowlers.head(5), y='Bowler', x='Batter Runs', orient='h', palette='viridis')
    st.pyplot(plt)

    total_fours = len(dhoni_t20i[dhoni_t20i['Batter Runs'] == 4])
    four_percentage = np.round(100 * total_fours * 4 / dhoni_t20i['Batter Runs'].sum(), 2)
    total_sixes = len(dhoni_t20i[dhoni_t20i['Batter Runs'] == 6])
    six_percentage = np.round(100 * total_sixes * 6 / dhoni_t20i['Batter Runs'].sum(), 2)

    runs_dist = pd.DataFrame(
        dhoni_t20i['Batter Runs'].value_counts().reset_index().sort_values(by='index')
    )
    runs_dist.columns = ['Batter Runs', 'count']
    runs_dist['Runs'] = runs_dist['Batter Runs'] * runs_dist['count']
    runs_dist['Percentage Contribution'] = np.round(100 * runs_dist['Runs'] / total_runs_scored, 2)

    st.title("MS Dhoni's Total Run Distribution with Singles, Doubles, Triples, Fours, and Sixes")
    plt.figure(figsize=(8, 4))
    labels = ['Dots', 'Singles', 'Doubles', 'Triples', 'Fours', 'Sixes']
    plt.pie(x=runs_dist['Runs'], labels=labels, autopct='%.2f', pctdistance=0.5)
    st.pyplot(plt)

    dismissal_type = pd.DataFrame(dhoni_t20i['Method'].value_counts()).reset_index()
    dismissal_type.columns = ['Method', 'count']

    st.title("MS Dhoni's Dismissals Throughout His Career")
    plt.figure(figsize=(13, 4))
    sns.barplot(data=dismissal_type, x='Method', y='count')
    st.pyplot(plt)

    results_df = pd.DataFrame(columns=['Match ID', 'Runs Scored', 'India Won'])
    match_data = {}

    for index, row in dhoni_t20i.iterrows():
        match_id = row['Match ID']
        runs = row['Batter Runs']
        winner = row['Winner']
        
        if match_id not in match_data:
            match_data[match_id] = {
                'total_runs': 0,
                'winner': winner
            }
        
        if not pd.isna(runs):
            match_data[match_id]['total_runs'] += runs
        
        if not pd.isna(winner) and 'India' in winner:
            match_data[match_id]['India_won'] = True
        else:
            match_data[match_id]['India_won'] = False

    for match_id, match_info in match_data.items():
        total_runs = match_info['total_runs']
        india_won = match_info['India_won']
        results_df = pd.concat([results_df, pd.DataFrame({'Match ID': [match_id], 'Runs Scored': [total_runs], 'India Won': [india_won]})], ignore_index=True)

    st.title("India Wins with Respect to Dhoni's Scores")
    plt.figure(figsize=(8, 4))
    sns.barplot(data=results_df, x='India Won', y='Runs Scored')
    st.pyplot(plt)

    matches_won_with_fifties = results_df[(results_df['Runs Scored'] >= 50)]

    runs_by_opponent = dhoni_t20i.groupby(by='Bat Second').agg({'Batter Runs': 'sum'}).reset_index()
    runs_by_opponent = runs_by_opponent[runs_by_opponent['Bat Second'] != 'India'].sort_values('Batter Runs', ascending=False)

    st.title("MS Dhoni's Highest Scores Against Different Teams by Batting Second")
    plt.figure(figsize=(8, 4))
    sns.barplot(data=runs_by_opponent.head(10), x='Bat Second', y='Batter Runs')
    plt.xticks(rotation=90)
    plt.title('Runs Against Opponents')
    plt.xlabel('Opponent Teams')
    plt.ylabel('Runs Scored')
    st.pyplot(plt)

    runs_by_opponent = dhoni_t20i.groupby(by='Bat First').agg({'Batter Runs': 'sum'}).reset_index()
    runs_by_opponent = runs_by_opponent[runs_by_opponent['Bat First'] != 'India'].sort_values('Batter Runs', ascending=False)

    st.title("MS Dhoni's Highest Scores Against Different Teams by Batting First")
    plt.figure(figsize=(8, 4))
    sns.barplot(data=runs_by_opponent.head(10), x='Bat First', y='Batter Runs')
    plt.xticks(rotation=90)
    plt.title('Runs Against Opponents')
    plt.xlabel('Opponent Teams')
    plt.ylabel('Runs Scored')
    st.pyplot(plt)

