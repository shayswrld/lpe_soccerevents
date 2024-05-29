
#Shaylin Velen
#Program to clean and restructure soccer event data, 
#get all open source events and ensure that the data is in a format that is easy to use for word2vec

import dask.dataframe as dd
import pandas as pd
import numpy as np
from statsbombpy import sb
import multiprocessing as mp
import time
import math


#Define pitch size and dimensions from statsbombpy
X_DIM = 120  # Max x coordinate - const 
Y_DIM = 80   # Max y coordinate - const

def get_matches(path: str, delim = ';'):
    # Read in the set of all available
    pd.set_option('display.max_columns', None)
    matches =  pd.read_csv(path, delimiter=delim)
    # print(len(matches))

    return matches

def get_comps(matches: pd.DataFrame):
    # Create series of all open source matches available across all seasons
    unique_comps = pd.Series(matches['competition'].unique())
    selected_comps = pd.Series(unique_comps[0:9]) # Choose selected competitions

    selected_comps = pd.concat([selected_comps, pd.Series(unique_comps[12]), pd.Series(unique_comps[17:19])]) # Add selected competitions
    selected_comps = selected_comps[selected_comps != 'England - FA Women\'s Super League'] # Remove women's league

    selected_matches = matches[matches['competition'].isin(selected_comps)]
    # Matches from these competitions selected for example
    '''
    0            England - Premier League
    1                    France - Ligue 1
    2             Germany - 1. Bundesliga
    3                     Spain - La Liga
    4                     Italy - Serie A
    5           Europe - Champions League
    6         Europe - UEFA Europa League
    8      International - FIFA World Cup
    0                  Europe - UEFA Euro
    17               Spain - Copa del Rey
    18    Africa - African Cup of Nations
    dtype: object
    '''

    return selected_matches

def get_match_set(league: str, selected_matches: pd.DataFrame) -> dd:
    # Pass league as argument for competition that data is to be taken from
    # return partitioned dask dataframe for parallel computation (optional)
    competition = selected_matches[selected_matches['competition'] == league]
    competition.reset_index(drop=True)

    #Split dataframe up vertically through partitions equal to num CPU cores so that operations on dataframe computed in parallel
    competition = dd.from_pandas(competition, npartitions=mp.cpu_count())

    competition_result = competition.compute() # Collect all threads and compute the result
    return competition_result

def get_iterators(selected_matches: pd.DataFrame):
    # Create iterators to go over matches in a passed set
    iterator = pd.Series(selected_matches['match_id'])
    return iterator


def get_events(match) -> dd.DataFrame:
    # Pass match as the match_id
    # Get all events for the last match in the premier league into a dask dataframe

    # Include all attributes and columns for each event, 'split=False'
    events_df = sb.events(match_id=match, split=False)
    
    # Drop columns that are not useful for learning from 
    pd_events_keys = pd.Series(events_df.keys().unique())
    dropped_types = ['Starting XI',
                    'Half Start',
                    'Half End',
                    'Injury Stoppage',
                    'Player Off',
                    'Referee Ball-Drop',
                    'Player On',
                    'Substitution']
    events_df = events_df[~events_df['type'].isin(dropped_types)]
    events_df_new = events_df.fillna('') # Fill NaN values with empty strings
    event_types = pd.Series(events_df['type'].unique())

    #Create and process dask dataframe in parallel with npartitions = number cpus
    # Should allow roughly 10x speedup for events_result processes
    dd_events = dd.from_pandas(events_df_new, npartitions=mp.cpu_count())
    events_result = dd_events.compute()

    return events_result

def map_field(x_coord, y_coord, x_dim, y_dim):

    # Input is a list location of this form [61.0, 40.1]
    # Output is a tuple of the mapped location
    # print(x_coord, y_coord)
    # Round up since coords are from 1;5
    mapped_x = math.ceil(x_coord / x_dim * 5)
    mapped_y = math.ceil(y_coord / y_dim * 5)

    return f'({mapped_x},{mapped_y})'

def map_loc(location):
    # Map the location of each event to a 5x5 grid
    # Input is a list location of this form [61.0, 40.1] as a string
    if location == '':
        return location
    else:
        # print(list(location)[1])
        location = location.strip('[]').split(',')
        x_coord = float(location[0])
        y_coord = float(location[1])
        
    mapped_x = math.ceil((x_coord / X_DIM) * 5)
    mapped_y = math.ceil((y_coord / Y_DIM) * 5)
    return f'({mapped_x},{mapped_y})'

def select_events(events, filters):
    # Obtain dataframe with only selected events
    df = events[filters]
    return df
    

def mod_pseudo_frame(pseudo_frame: pd.DataFrame):
    # Modify the pseudo frame to ensure that the data is in a format that is easy to use for word2vec
    pseudo_frame.loc[pseudo_frame['ball_receipt_outcome'] != '', 'ball_receipt_outcome'] = 'ball_receive_failed'

    carry_col = pseudo_frame['carry_end_location']
    pseudo_frame.loc[:, 'carry_end_location'] = carry_col.map(map_loc)
    pseudo_frame.loc[pseudo_frame['carry_end_location'] != '', 'carry_end_location'] = 'ball_carry_to ' + (pseudo_frame['carry_end_location'])

    pseudo_frame.loc[pseudo_frame['clearance_body_part'] != '', 'clearance_body_part'] = pseudo_frame['clearance_body_part'].str.replace(' ', '_') + ' clearance'
    pseudo_frame.loc[pseudo_frame['dribble_nutmeg'] != '', 'dribble_nutmeg'] = 'nutmegged'
    pseudo_frame.loc[pseudo_frame['dribble_outcome'] != '', 'dribble_outcome'] = pseudo_frame['dribble_outcome'] + '_dribble'

    pseudo_frame.loc[pseudo_frame['duel_outcome'] != '', 'duel_outcome'] = 'Duel ' + pseudo_frame['duel_outcome'].str.replace(' ', '_')
    pseudo_frame.loc[pseudo_frame['duel_type'] != '', 'duel_type'] = pseudo_frame['duel_type'].str.replace(' ', '_')
    for column in pseudo_frame.columns[7:9]:
        pseudo_frame.loc[pseudo_frame[column] != '', column] = pseudo_frame[column].str.replace(' ', '_')
    pseudo_frame.loc[pseudo_frame['interception_outcome'] != '', 'interception_outcome'] = 'Interception ' + pseudo_frame['interception_outcome'].str.replace(' ', '_')

    pseudo_frame.loc[pseudo_frame['pass_cross'] != '', 'pass_cross'] = 'Cross'
    pass_end_col = pseudo_frame['pass_end_location']
    
    pseudo_frame.loc[pseudo_frame['pass_end_location'] != '', 'pass_end_location'] = 'Pass_to ' + pass_end_col.map(map_loc)
    pseudo_frame.loc[pseudo_frame['pass_height'] != '', 'pass_height'] = pseudo_frame['pass_height'].str.replace(' ', '_')

    pass_lengths = pseudo_frame['pass_length']
    index_iter = list(pseudo_frame.index)
    for j in index_iter:
        pass_sent = pass_lengths[j]
        if pass_sent == '':
            pseudo_frame.loc[j, 'pass_length'] = ''
        else:
            pass_dist = round(float(pass_sent), 0)
            if pass_dist < 15:
                pseudo_frame.loc[j, 'pass_length'] = 'short_pass'
            elif pass_dist < 30:
                pseudo_frame.loc[j, 'pass_length'] = 'medium_pass'
            else:
                pseudo_frame.loc[j, 'pass_length'] = 'long_pass'

    pseudo_frame.loc[pseudo_frame['pass_recipient'] != '', 'pass_recipient'] = 'Pass_received_by ' + pseudo_frame['pass_recipient'].str.replace(' ', '_')
    pseudo_frame.loc[pseudo_frame['pass_technique'] != '', 'pass_technique'] = pseudo_frame['pass_technique'].str.replace(' ', '_') + '_pass'
    pseudo_frame.loc[pseudo_frame['player'] != '', 'player'] = pseudo_frame['player'].str.replace(' ', '_')
    pseudo_frame.loc[pseudo_frame['shot_outcome'] != '', 'shot_outcome'] = 'Shot_' + pseudo_frame['shot_outcome'].str.replace(' ', '_')
    pseudo_frame.loc[pseudo_frame['goalkeeper_outcome'] != '', 'goalkeeper_outcome'] = pseudo_frame['goalkeeper_outcome'].str.replace(' ', '_')
    
    # Shot one on one and dribble nutmeg columns not present in every dataframe

    return pseudo_frame

def set_anon(pseudo_frame):
    
    # Assuming pseudo_frame is your DataFrame
    print(pseudo_frame.index)
    for i in range(len(pseudo_frame) - 1, -1, -1):
        if pseudo_frame.loc[i, 'pass_recipient'] != '':
            # Duplicate the row
            new_row = pseudo_frame.loc[i].copy()
            new_row['player'] = ''
            pseudo_frame.loc[i, 'pass_recipient'] = 'Pass_received_by '
            # Insert the new row one position below the original row
            pseudo_frame = pd.concat([pseudo_frame.iloc[:i+1], new_row.to_frame().T, pseudo_frame.iloc[i+1:]]).reset_index(drop=True)

    # Reindex the DataFrame to correct the index after inserting rows
    pseudo_frame.index = np.arange(len(pseudo_frame))
    return pseudo_frame

def write_corpus(pseudo_frame, k: int):

    # Form the corpus for the word2vec model

    corpus = ''
    # pseudo_frame = pseudo_frame.compute()
    writepath = 'match_coms_new/'+ 'corpus' + str(k) + '.txt'
    vocabsize = 0 
    with open(writepath, 'w') as f:

        for row in pseudo_frame.values:
        # print(row)
            line = ''
            str(row).strip()
            for column in row:
                if column != '':
                    line += str(column) + ' '
                    vocabsize += 1
            line += '\n'
            f.write(line)
    f.close()
    return vocabsize

def has_more_than_three_filled(row):
    filled_count = sum(row != '')
    return filled_count > 3

def main():
    start_time = time.time()
    # Get all matches
    path_to_matches = '/home/vlnsha004/CSC2005Z/player2vec/data_events/available_matches.csv'
    # delim = ';' # Delimiter for csv file
    matches = get_matches(path_to_matches)

    # Obtain selected matches from competitions selected
    selected_matches = get_comps(matches)
    # match_set = get_match_set('Spain - La Liga', selected_matches) #match_set is a dask dataframe - can be computed in parallel effectively
    matches_iter = dd.from_pandas(get_iterators(selected_matches), npartitions=mp.cpu_count())
    param_size = 0
    for k in matches_iter:

        start_single = time.time()
        # Get all events for the matches in the premier league
        # try:

        events = get_events(k) # Obtain the events dataframe as a dask dataframe, iterate through all matches
    
        location_col = events['location']
        events['location'] = location_col.map(map_loc) # Map the location of each event to a 5x5 grid
        filters = ['index', 'player', 'location', 'ball_receipt_outcome', 'carry_end_location', 'clearance_body_part', 'dribble_outcome','duel_outcome',
                'duel_type', 'goalkeeper_outcome', 'goalkeeper_technique', 'interception_outcome',
                'pass_cross', 'pass_end_location', 'pass_height', 'pass_length', 'pass_recipient', 'pass_technique', 'shot_outcome']
        pseudo_frame = select_events(events, filters) # Obtain dataframe with only selected events
        # Convert the 'timestamp' column to datetime format
        # Filter rows with more than 2 filled columns
        pseudo_frame = pseudo_frame[pseudo_frame.apply(has_more_than_three_filled, axis=1)]

        pseudo_frame = mod_pseudo_frame(pseudo_frame)
        pseudo_frame.loc[:, 'index'] = pd.to_numeric(pseudo_frame['index'])

        # Sort the DataFrame by the 'timestamp' column
        pseudo_frame = pseudo_frame.sort_values('index')
        pseudo_frame = pseudo_frame.reset_index(drop=True)
        anon_pf = set_anon(pseudo_frame)
        pseudo_frame = anon_pf.iloc[:, 1:]
        param_size += write_corpus(pseudo_frame, k) # Form the corpus for the word2vec model
        # except Exception as e:
        #     print(e)
        #     continue

        end_single = time.time()
        execution_one = end_single - start_single
        print(f"Program executed in: {execution_one} seconds")

    print(param_size)
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Program executed in: {execution_time} seconds")

if __name__ == '__main__':
    main()
