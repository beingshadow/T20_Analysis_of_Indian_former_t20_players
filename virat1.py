import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
def virat_title():
    return st.title("Virat Kohli T20 International Analysis")


@st.cache_data
def load_data():
    return pd.read_csv('ball_by_ball_it20.csv')

@st.cache_data
def virat_kohli_performance():
    t20i = load_data()
    full_name = ''
    for name in t20i['Batter'].unique():
        if 'kohli' in name.lower():
            full_name = name
            break

    kohli_t20i = t20i[t20i['Batter'] == full_name]

    recommended_columns = ['Match ID', 'Date', 'Venue', 'Bat First', 'Bat Second', 'Innings', 'Over', 'Ball', 'Bowler', 
                        'Batter Runs', 'Runs From Ball', 'Batter Balls Faced', 'Wicket', 'Method', 'Player Out',
                        'Player Out Runs', 'Player Out Balls Faced', 'Winner', 'Chased Successfully']

    kohli_t20i = kohli_t20i[recommended_columns]
    st.dataframe(kohli_t20i.sample(5))
    st.write(kohli_t20i.shape)

    match_id_mapping = {}
    match_id_counter = 1

    for match_id in kohli_t20i['Match ID'].unique():
        match_id_mapping[match_id] = f"Match {match_id_counter}"
        match_id_counter += 1

    kohli_t20i['Match ID'] = kohli_t20i['Match ID'].map(match_id_mapping)

    matches_played = kohli_t20i['Match ID'].nunique()
    total_runs_scored = kohli_t20i['Batter Runs'].sum()

    kohli_t20i['Date'] = pd.to_datetime(kohli_t20i['Date']).dt.date
    kohli_t20i['year'] = kohli_t20i['Date'].apply(lambda x: x.year)
    matches_played_by_year = kohli_t20i.groupby(by='year')['Match ID'].nunique().reset_index()
    st.write("Kohli Matches Played Year by Year")
    matches_played_by_year.columns = ['year', 'matches_played']

    st.title("Kohli's Matches Played Year by Year")
    plt.figure(figsize=(9, 4))
    sns.set_style("darkgrid")
    sns.lineplot(data=matches_played_by_year, x='year', y='matches_played')
    plt.title("Matches Played by Year")
    st.pyplot(plt)

    runs_by_year = kohli_t20i.pivot_table(index='year', values='Batter Runs', aggfunc='sum').reset_index()
    runs_by_year['Percentage Change'] = np.round(runs_by_year['Batter Runs'].pct_change() * 100)

    st.title("Kohli's Runs Year by Year")
    plt.figure(figsize=(9, 4))
    sns.set_style("darkgrid")
    sns.lineplot(data=runs_by_year, x='year', y='Batter Runs')
    plt.title("Runs Scored by Year")
    st.pyplot(plt)

    venue_stats = kohli_t20i.groupby('Venue').agg({
        'Batter Runs': 'sum',
        'Batter Balls Faced': 'count'
    }).reset_index()
    venue_stats['Strike Rate'] = np.round(100 * venue_stats['Batter Runs'] / venue_stats['Batter Balls Faced'])
    venue_stats = venue_stats.sort_values(by='Batter Runs', ascending=False)

    st.title("Virat Kohli's Top 5 Favourite Stadiums to Score")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=venue_stats.head(5), y='Venue', x='Batter Runs', orient='h')
    sns.despine(bottom=True, trim=True)
    st.pyplot(plt)

    runs_by_innings = kohli_t20i.groupby(by='Innings').agg({'Batter Runs': 'sum', 'Batter Balls Faced': 'count'})
    runs_by_innings['Strike Rate'] = np.round((100 * runs_by_innings['Batter Runs'] / runs_by_innings['Batter Balls Faced']))
    labels = ['Innings 1', 'Innings 2']
    x = runs_by_innings['Batter Runs']
    explode = [0.1, 0]
    st.title("Virat Kohli Run Distribution Innings Wise")
    plt.figure(figsize=(8, 4))
    plt.pie(labels=labels, x=x, autopct='%1.1f%%', explode=explode)
    st.pyplot(plt)

    over_runs = kohli_t20i.groupby(by='Over').agg({"Batter Runs": 'sum'}).fillna(0).reset_index()

    st.title("Kohli Top 5 Overs Where He Scored Maximum Runs")
    plt.figure(figsize=(8, 4))
    over_runs = over_runs.sort_values(by='Batter Runs', ascending=False)
    sns.barplot(data=over_runs.head(5), x='Over', y='Batter Runs')
    st.pyplot(plt)

    runs_bowlers = kohli_t20i.groupby(by='Bowler').agg({
        'Batter Runs': 'sum',
        'Match ID': 'count'
    }).reset_index()
    runs_bowlers['Strike Rate'] = np.round(100 * runs_bowlers['Batter Runs'] / runs_bowlers['Match ID'])
    runs_bowlers.columns = ['Bowler', 'Batter Runs', 'Balls Faced', 'Strike Rate']
    runs_bowlers = runs_bowlers.sort_values(by=['Batter Runs', 'Strike Rate'], ascending=False)

    st.title("Kohli Most Runs Against Different Bowlers")
    plt.figure(figsize=(8, 4))
    sns.barplot(data=runs_bowlers.head(5), y='Bowler', x='Batter Runs', orient='h', palette='viridis')
    st.pyplot(plt)

    total_fours = len(kohli_t20i[kohli_t20i['Batter Runs'] == 4])
    four_percentage = np.round(100 * total_fours * 4 / kohli_t20i['Batter Runs'].sum(), 2)
    total_sixes = len(kohli_t20i[kohli_t20i['Batter Runs'] == 6])
    six_percentage = np.round(100 * total_sixes * 6 / kohli_t20i['Batter Runs'].sum(), 2)

    runs_dist = pd.DataFrame(
        kohli_t20i['Batter Runs'].value_counts().reset_index().sort_values(by='index')
    )
    runs_dist.columns = ['Batter Runs', 'count']
    runs_dist['Runs'] = runs_dist['Batter Runs'] * runs_dist['count']
    runs_dist['Percentage Contribution'] = np.round(100 * runs_dist['Runs'] / total_runs_scored, 2)

    st.title("Kohli Total Run Distribution with Singles, Doubles, Triples, Fours, and Sixes")
    plt.figure(figsize=(8, 4))
    labels = ['Dots', 'Singles', 'Doubles', 'Triples', 'Fours', 'Sixes']
    plt.pie(x=runs_dist['Runs'], labels=labels, autopct='%.2f', pctdistance=0.5)
    st.pyplot(plt)

    dismissal_type = pd.DataFrame(kohli_t20i['Method'].value_counts()).reset_index()
    dismissal_type.columns = ['Method', 'count']

    st.title("Kohli's Dismissals Throughout His Career")
    plt.figure(figsize=(13, 4))
    sns.barplot(data=dismissal_type, x='Method', y='count')
    st.pyplot(plt)

    results_df = pd.DataFrame(columns=['Match ID', 'Runs Scored', 'India Won'])
    match_data = {}

    for index, row in kohli_t20i.iterrows():
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

    st.title("India Wins with Respect to Kohli Scores")
    plt.figure(figsize=(8, 4))
    sns.barplot(data=results_df, x='India Won', y='Runs Scored')
    st.pyplot(plt)

    matches_won_with_fifties = results_df[(results_df['Runs Scored'] >= 50)]

    runs_by_opponent = kohli_t20i.groupby(by='Bat Second').agg({'Batter Runs': 'sum'}).reset_index()
    runs_by_opponent = runs_by_opponent[runs_by_opponent['Bat Second'] != 'India'].sort_values('Batter Runs', ascending=False)

    st.title("Virat Kohli's Highest Scores Against Different Teams by Batting Second")
    plt.figure(figsize=(8, 4))
    sns.barplot(data=runs_by_opponent.head(10), x='Bat Second', y='Batter Runs')
    plt.xticks(rotation=90)
    plt.title('Runs Against Opponents')
    plt.xlabel('Opponent Teams')
    plt.ylabel('Runs Scored')
    st.pyplot(plt)

    runs_by_opponent = kohli_t20i.groupby(by='Bat First').agg({'Batter Runs': 'sum'}).reset_index()
    runs_by_opponent = runs_by_opponent[runs_by_opponent['Bat First'] != 'India'].sort_values('Batter Runs', ascending=False)

    st.title("Virat Kohli's Highest Scores Against Different Teams by Batting First")
    plt.figure(figsize=(8, 4))
    sns.barplot(data=runs_by_opponent.head(10), x='Bat First', y='Batter Runs')
    plt.xticks(rotation=90)
    plt.title('Runs Against Opponents')
    plt.xlabel('Opponent Teams')
    plt.ylabel('Runs Scored')
    st.pyplot(plt)

