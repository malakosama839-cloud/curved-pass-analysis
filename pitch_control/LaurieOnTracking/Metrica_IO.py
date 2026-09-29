#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 11:18:49 2020

Module for reading in Metrica sample data.

Data can be found at: https://github.com/metrica-sports/sample-data

@author: Laurie Shaw (@EightyFivePoint)
"""

import pandas as pd
import csv as csv
import numpy as np
import json
import os
from typing import Tuple
def read_match_data(DATADIR,gameid):
    '''
    read_match_data(DATADIR,gameid):
    read all Metrica match data (tracking data for home & away teams, and ecvent data)
    '''
    tracking_home = tracking_data(DATADIR,gameid,'Home')
    tracking_away = tracking_data(DATADIR,gameid,'Away')
    events = read_event_data(DATADIR,gameid)
    return tracking_home,tracking_away,events

def read_event_data(DATADIR,game_id):
    '''
    read_event_data(DATADIR,game_id):
    read Metrica event data  for game_id and return as a DataFrame
    '''
    eventfile = '/Sample_Game_%d/Sample_Game_%d_RawEventsData.csv' % (game_id,game_id) # filename
    events = pd.read_csv('{}/{}'.format(DATADIR, eventfile)) # read data
    return events

def tracking_data(DATADIR,game_id,teamname):
    '''
    tracking_data(DATADIR,game_id,teamname):
    read Metrica tracking data for game_id and return as a DataFrame. 
    teamname is the name of the team in the filename. For the sample data this is either 'Home' or 'Away'.
    '''
    teamfile = '/Sample_Game_%d/Sample_Game_%d_RawTrackingData_%s_Team.csv' % (game_id,game_id,teamname)
    # First:  deal with file headers so that we can get the player names correct
    csvfile =  open('{}/{}'.format(DATADIR, teamfile), 'r') # create a csv file reader
    reader = csv.reader(csvfile) 
    teamnamefull = next(reader)[3].lower()
    print("Reading team: %s" % teamnamefull)
    # construct column names
    jerseys = [x for x in next(reader) if x != ''] # extract player jersey numbers from second row
    columns = next(reader)
    for i, j in enumerate(jerseys): # create x & y position column headers for each player
        columns[i*2+3] = "{}_{}_x".format(teamname, j)
        columns[i*2+4] = "{}_{}_y".format(teamname, j)
    columns[-2] = "ball_x" # column headers for the x & y positions of the ball
    columns[-1] = "ball_y"
    # Second: read in tracking data and place into pandas Dataframe
    tracking = pd.read_csv('{}/{}'.format(DATADIR, teamfile), names=columns, index_col='Frame', skiprows=3)
    return tracking


class DataFormatConverter:
    """
    Converts between different football data formats for pitch control analysis
    """
    
    def __init__(self):
        """Initialize the converter with default field dimensions and mappings"""
        self.field_dimensions = (106.0, 68.0)  # FIFA standard pitch dimensions (length, width) in meters
        
    def read_csv_tracking_data(self, file_path: str) -> pd.DataFrame:
        """
        Read CSV tracking data and convert to expected format
        
        Expected CSV format:
        - gameid, gameeventid, possessioneventid, starttime, endtime, duration, 
          eventtime, sequence, playerid, positiongrouptype, jerseynum, team, x, y, ...
        """
        print(f"Reading CSV tracking data from: {file_path}")
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Group by eventtime (frame equivalent) and team to create tracking format
        tracking_frames = {}
        
        for frame_time, group in df.groupby('eventtime'):
            frame_data = {'Time [s]': frame_time, 'Period': group.iloc[0]['period']}
            
            # Process each team
            for team_name, team_group in group.groupby('team'):
                for _, player in team_group.iterrows():
                    jersey = str(player['jerseynum'])
                    x_col = f"{team_name}_{jersey}_x"
                    y_col = f"{team_name}_{jersey}_y"
                    
                    # Convert coordinates to normalized format (0-1)
                    # Assuming input coordinates are in meters, convert to normalized
                    frame_data[x_col] = (player['x'] + 53) / 106.0  # Center field and normalize
                    frame_data[y_col] = (player['y'] + 34) / 68.0   # Center field and normalize
            
            tracking_frames[len(tracking_frames) + 1] = frame_data
        
        # Convert to DataFrame
        tracking_df = pd.DataFrame.from_dict(tracking_frames, orient='index')
        tracking_df.index.name = 'Frame'
        
        return tracking_df
    
    def read_ball_tracking_data(self, file_path: str) -> pd.DataFrame:
        """
        Read ball tracking data from CSV and integrate with player tracking
        """
        print(f"Reading ball tracking data from: {file_path}")
        
        df = pd.read_csv(file_path)
        ball_data = {}
        
        for _, row in df.iterrows():
            frame = int(row['eventtime'] * 25)  # Convert time to frame (assuming 25 FPS)
            
            # Convert ball coordinates to normalized format
            ball_data[frame] = {
                'ball_x': (row['ball_x'] + 53) / 106.0,
                'ball_y': (row['ball_y'] + 34) / 68.0,
                'Time [s]': row['eventtime'],
                'Period': row['period']
            }
        
        return pd.DataFrame.from_dict(ball_data, orient='index')
    
    def read_json_event_data(self, file_path: str) -> pd.DataFrame:
        """
        Read JSON event data and convert to expected CSV-like format
        
        Expected format:
        - Team, Type, Subtype, Period, Start Frame, Start Time [s], End Frame, End Time [s], 
          From, To, Start X, Start Y, End X, End Y
        """
        print(f"Reading JSON event data from: {file_path}")
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse JSON - handle different possible formats
        events_list = []
        
        # Try parsing as a single JSON object first
        try:
            data = json.loads(content)
            if isinstance(data, list):
                events_list = data
            else:
                events_list = [data]
        except json.JSONDecodeError:
            # Try parsing line by line
            for line in content.strip().split('\n'):
                if line.strip():
                    try:
                        event = json.loads(line)
                        events_list.append(event)
                    except json.JSONDecodeError:
                        continue
        
        if not events_list:
            print("Warning: No valid JSON events found")
            return pd.DataFrame(columns=['Team', 'Type', 'Subtype', 'Period', 'Start Frame', 
                                       'Start Time [s]', 'End Frame', 'End Time [s]', 'From', 'To', 
                                       'Start X', 'Start Y', 'End X', 'End Y'])
        
        converted_events = []
        
        for event in events_list:
            try:
                # Extract basic event information
                game_events = event.get('gameEvents', {})
                possession_events = event.get('possessionEvents', {})
                stadium_metadata = event.get('stadiumMetadata', {})
                
                # Determine team name and map to Home/Away for pitch control
                team_name = game_events.get('teamName', 'Unknown')
                if team_name == 'Unknown' and game_events.get('homeTeam') is not None:
                    team_name = 'Home' if game_events.get('homeTeam') else 'Away'
                else:
                    # Map actual team names to Home/Away for pitch control compatibility
                    # First team found becomes "Home", second becomes "Away"
                    if team_name in ['Netherlands', 'Home']:
                        team_name = 'Home'
                    elif team_name in ['Senegal', 'Away']:
                        team_name = 'Away'
                
                # Map event types to expected format
                event_type = possession_events.get('possessionEventType', 'Unknown')
                if event_type == 'PA':  # PA means Possession Action (similar to PASS)
                    mapped_type = 'PASS'
                elif event_type == 'SH':  # SH means Shot
                    mapped_type = 'SHOT'
                else:
                    mapped_type = event_type
                
                event_data = {
                    'Team': team_name,
                    'Type': mapped_type,
                    'Subtype': '',
                    'Period': game_events.get('period', 1),
                    'Start Frame': int(event.get('startTime', 0) * 25) if event.get('startTime') else 0,
                    'Start Time [s]': event.get('startTime', 0),
                    'End Frame': int(event.get('endTime', 0) * 25) if event.get('endTime') else 0,
                    'End Time [s]': event.get('endTime', 0),
                    'From': f"Player{possession_events.get('passerPlayerId', '')}",
                    'To': f"Player{possession_events.get('receiverPlayerId', '')}",
                }
                
                # Extract ball position for start coordinates
                ball_data = event.get('ball', [])
                if ball_data and isinstance(ball_data, list) and len(ball_data) > 0:
                    ball_pos = ball_data[0]
                    # Convert from meters to normalized coordinates
                    event_data['Start X'] = (ball_pos.get('x', 0) + 53) / 106.0
                    event_data['Start Y'] = (ball_pos.get('y', 0) + 34) / 68.0
                else:
                    event_data['Start X'] = np.nan
                    event_data['Start Y'] = np.nan
                
                # For now, set end coordinates same as start
                event_data['End X'] = event_data['Start X']
                event_data['End Y'] = event_data['Start Y']
                
                # Only add events with meaningful data
                if event_data['Type'] != 'Unknown' or not pd.isna(event_data['Start X']):
                    converted_events.append(event_data)
                
            except Exception as e:
                print(f"Warning: Could not process event: {e}")
                continue
        
        events_df = pd.DataFrame(converted_events)
        print(f"Successfully converted {len(events_df)} events from JSON")
        return events_df
    
    def split_tracking_by_team(self, tracking_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split combined tracking data into separate home and away team dataframes
        """
        # Find home and away team columns
        home_cols = [col for col in tracking_df.columns if col.startswith('H_') or col.startswith('Home_')]
        away_cols = [col for col in tracking_df.columns if col.startswith('A_') or col.startswith('Away_')]
        
        # Include time and period columns
        time_cols = ['Time [s]', 'Period']
        ball_cols = ['ball_x', 'ball_y']
        
        # Create separate dataframes
        home_df = tracking_df[time_cols + home_cols + ball_cols].copy()
        away_df = tracking_df[time_cols + away_cols + ball_cols].copy()
        
        return home_df, away_df
    
    def convert_csv_to_metrica_format(self, 
                                    player_csv_path: str, 
                                    ball_csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert CSV tracking data to Metrica-like format with separate home/away dataframes
        """
        print("Converting CSV tracking data to Metrica format...")
        
        # Read player tracking data
        player_data = pd.read_csv(player_csv_path)
        
        # Read ball tracking data
        ball_data = pd.read_csv(ball_csv_path)
        
        # Map team codes to full names
        team_mapping = {'H': 'Home', 'A': 'Away'}
        
        # Create frame-based data structure
        frames_data = {}
        
        # Process player data by event time (frame equivalent)
        for event_time, group in player_data.groupby('eventtime'):
            frame_idx = int(event_time * 25)  # Convert time to frame number
            
            frame_data = {
                'Period': group.iloc[0]['period'],
                'Time [s]': event_time
            }
            
            # Add player positions grouped by team
            for _, player in group.iterrows():
                team_code = player['team']
                team_name = team_mapping.get(team_code, team_code)
                jersey = player['jerseynum']
                
                # Convert coordinates to normalized (0-1) format expected by Metrica
                x_normalized = (player['x'] + 53) / 106.0  # Assuming input is in meters
                y_normalized = (player['y'] + 34) / 68.0
                
                frame_data[f"{team_name}_{jersey}_x"] = x_normalized
                frame_data[f"{team_name}_{jersey}_y"] = y_normalized
            
            frames_data[frame_idx] = frame_data
        
        # Add ball data
        for _, ball_row in ball_data.iterrows():
            frame_idx = int(ball_row['eventtime'] * 25)
            if frame_idx in frames_data:
                frames_data[frame_idx]['ball_x'] = (ball_row['ball_x'] + 53) / 106.0
                frames_data[frame_idx]['ball_y'] = (ball_row['ball_y'] + 34) / 68.0
        
        # Convert to DataFrame
        combined_df = pd.DataFrame.from_dict(frames_data, orient='index')
        combined_df.index.name = 'Frame'
        combined_df = combined_df.sort_index()
        
        # Split into home and away teams
        home_cols = ['Period', 'Time [s]'] + [col for col in combined_df.columns if col.startswith('Home_')] + ['ball_x', 'ball_y']
        away_cols = ['Period', 'Time [s]'] + [col for col in combined_df.columns if col.startswith('Away_')] + ['ball_x', 'ball_y']
        
        home_df = combined_df[home_cols].copy()
        away_df = combined_df[away_cols].copy()
        
        print(f"Created tracking data with {len(combined_df)} frames")
        print(f"  - Home players: {len([c for c in home_df.columns if c.startswith('Home_') and c.endswith('_x')])}")
        print(f"  - Away players: {len([c for c in away_df.columns if c.startswith('Away_') and c.endswith('_x')])}")
        
        return home_df, away_df


def convert_csv_to_metrica_format(self, player_data, ball_data) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        # Map team codes to full names
        team_mapping = {'H': 'Home', 'A': 'Away'}
        
        # Create frame-based data structure
        frames_data = {}
        
        # Process player data by event time (frame equivalent)
        for event_time, group in player_data.groupby('eventtime'):
            frame_idx = int(event_time * 25)  # Convert time to frame number
            
            frame_data = {
                'Period': group.iloc[0]['period'],
                'Time [s]': event_time
            }
            
            # Add player positions grouped by team
            for _, player in group.iterrows():
                team_code = player['team']
                team_name = team_mapping.get(team_code, team_code)
                jersey = player['jerseynum']
                
                # Convert coordinates to normalized (0-1) format expected by Metrica
                x_normalized = (player['x'] + 53) / 106.0  # Assuming input is in meters
                y_normalized = (player['y'] + 34) / 68.0
                
                frame_data[f"{team_name}_{jersey}_x"] = x_normalized
                frame_data[f"{team_name}_{jersey}_y"] = y_normalized
            
            frames_data[frame_idx] = frame_data
        
        # Add ball data
        for _, ball_row in ball_data.iterrows():
            frame_idx = int(ball_row['eventtime'] * 25)
            if frame_idx in frames_data:
                frames_data[frame_idx]['ball_x'] = (ball_row['ball_x'] + 53) / 106.0
                frames_data[frame_idx]['ball_y'] = (ball_row['ball_y'] + 34) / 68.0
        
        # Convert to DataFrame
        combined_df = pd.DataFrame.from_dict(frames_data, orient='index')
        combined_df.index.name = 'Frame'
        combined_df = combined_df.sort_index()
        
        # Split into home and away teams
        home_cols = ['Period', 'Time [s]'] + [col for col in combined_df.columns if col.startswith('Home_')] + ['ball_x', 'ball_y']
        away_cols = ['Period', 'Time [s]'] + [col for col in combined_df.columns if col.startswith('Away_')] + ['ball_x', 'ball_y']
        
        home_df = combined_df[home_cols].copy()
        away_df = combined_df[away_cols].copy()
        
        print(f"Created tracking data with {len(combined_df)} frames")
        print(f"  - Home players: {len([c for c in home_df.columns if c.startswith('Home_') and c.endswith('_x')])}")
        print(f"  - Away players: {len([c for c in away_df.columns if c.startswith('Away_') and c.endswith('_x')])}")
        
        return home_df, away_df

def load_new_data(data_directory: str, 
                 match_id: str = "3812") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and convert new data format to work with existing pitch control system
    
    Args:
        data_directory: Root directory containing the new data
        match_id: Match identifier (e.g., "3812")
    
    Returns:
        tracking_home, tracking_away, events (in Metrica-compatible format)
    """
    converter = DataFormatConverter()
    
    # Define file paths
    player_csv = os.path.join(data_directory, f"match_{match_id}_players.csv")
    ball_csv = os.path.join(data_directory,  f"match_{match_id}_ball.csv")
    event_json = os.path.join(data_directory, f"{match_id}.json")
    
    # Check if files exist
    if not all(os.path.exists(f) for f in [player_csv, ball_csv, event_json]):
        missing = [f for f in [player_csv, ball_csv, event_json] if not os.path.exists(f)]
        raise FileNotFoundError(f"Missing data files: {missing}")
    
    # Convert tracking data
    tracking_home, tracking_away = converter.convert_csv_to_metrica_format(player_csv, ball_csv)
    
    # Convert event data
    events = converter.read_json_event_data(event_json)
    
    # Fix frame alignment - ensure all events have valid frames
    print("Fixing frame alignment...")
    tracking_frames = set(tracking_home.index) & set(tracking_away.index)
    
    # Filter events to only include those with valid tracking frames
    valid_events = []
    for idx, event in events.iterrows():
        start_frame = event['Start Frame']
        if start_frame in tracking_frames:
            valid_events.append(idx)
    
    if valid_events:
        events = events.loc[valid_events].copy()
        print(f"  ✓ Filtered to {len(events)} events with valid tracking frames")
    else:
        print("  ⚠ No events found with matching tracking frames")
    
    print(f"Successfully loaded data for match {match_id}")
    print(f"  - Home tracking: {tracking_home.shape}")
    print(f"  - Away tracking: {tracking_away.shape}")
    print(f"  - Events: {events.shape}")
    
    return tracking_home, tracking_away, events


def merge_tracking_data(home,away):
    '''
    merge home & away tracking data files into single data frame
    '''
    return home.drop(columns=['ball_x', 'ball_y']).merge( away, left_index=True, right_index=True )
    
def to_metric_coordinates(data,field_dimen=(106.,68.) ):
    '''
    Convert positions from Metrica units to meters (with origin at centre circle)
    '''
    x_columns = [c for c in data.columns if c[-1].lower()=='x']
    y_columns = [c for c in data.columns if c[-1].lower()=='y']
    data[x_columns] = ( data[x_columns]-0.5 ) * field_dimen[0]
    data[y_columns] = -1 * ( data[y_columns]-0.5 ) * field_dimen[1]
    ''' 
    ------------ ***NOTE*** ------------
    Metrica actually define the origin at the *top*-left of the field, not the bottom-left, as discussed in the YouTube video. 
    I've changed the line above to reflect this. It was originally:
    data[y_columns] = ( data[y_columns]-0.5 ) * field_dimen[1]
    ------------ ********** ------------
    '''
    return data

def to_single_playing_direction(home,away,events):
    '''
    Flip coordinates in second half so that each team always shoots in the same direction through the match.
    '''
    for team in [home,away,events]:
        second_half_idx = team.Period.idxmax(2)
        columns = [c for c in team.columns if c[-1].lower() in ['x','y']]
        team.loc[second_half_idx:,columns] *= -1
    return home,away,events

def find_playing_direction(team,teamname):
    '''
    Find the direction of play for the team (based on where the goalkeepers are at kickoff). +1 is left->right and -1 is right->left
    '''    
    GK_column_x = teamname+"_"+find_goalkeeper(team)+"_x"
    # +ve is left->right, -ve is right->left
    return -np.sign(team.iloc[0][GK_column_x])
    
def find_goalkeeper(team):
    '''
    Find the goalkeeper in team, identifying him/her as the player closest to goal at kick off
    ''' 
    x_columns = [c for c in team.columns if c[-2:].lower()=='_x' and c[:4] in ['Home','Away']]
    GK_col = team.iloc[0][x_columns].abs().idxmax()
    return GK_col.split('_')[1]
    