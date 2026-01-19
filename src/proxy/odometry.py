# odometry stuff, kinda messy but works

import sqlite3
import re
import math
import json

def find_column_payload(conn):
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(messages)")
    cols = []
    for row in cursor.fetchall():
        cols.append(row[1])
    
    # check some common names
    possible = ["data", "msg", "message", "payload", "raw"]
    for p in possible:
        if p in cols:
            return p
    return None

def calc_distance(x1, y1, x2, y2):
    """pythagoras thing"""
    dx = x2 - x1
    dy = y2 - y1
    return math.sqrt(dx**2 + dy**2)

def get_distance_km_from_topic(db_path, topic_name_substring="local_odometry", verbose=False, start_ns=None, end_ns=None):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute("SELECT id, name FROM topics")
    all_topics = cur.fetchall()
    
    # find the odometry topic
    topic_id = None
    topic_name = None
    for tid, name in all_topics:
        if topic_name_substring.lower() in name.lower():
            topic_id = tid
            topic_name = name
            break
    
    if topic_id is None:
        conn.close()
        if verbose:
            print("couldn't find odometry topic")
        return None
    
    # figure out data column
    data_col = find_column_payload(conn)
    if data_col is None:
        conn.close()
        if verbose:
            print("no data column?")
        return None
    
    if verbose:
        print(f"using topic {topic_id} '{topic_name}' with column '{data_col}'")
    
    # slice-lvl or run-lvl
    if start_ns is not None and end_ns is not None:
        sql = f"SELECT timestamp, {data_col} FROM messages WHERE topic_id = ? AND timestamp >= ? AND timestamp < ? ORDER BY timestamp"
        params = (topic_id, int(start_ns), int(end_ns))
    else:
        sql = f"SELECT timestamp, {data_col} FROM messages WHERE topic_id = ? ORDER BY timestamp"
        params = (topic_id,)
    
    cur.execute(sql, params)
    messages = cur.fetchall()
    conn.close()
    
    if not messages:
        if verbose:
            print("no messages for this topic")
        return None
    
    # regex for numbers
    num_pattern = re.compile(r'[-+]?\d*\.?\d+')
    
    prev_pos = None
    total_dist = 0.0
    
    for timestamp, payload in messages:
        if payload is None:
            continue
        
        position = None

        # regex on raw text
        if position is None:
            try:
                if isinstance(payload, bytes):
                    text = payload.decode('utf-8', errors='ignore') # ros2 normalization
                else:
                    text = str(payload)
                
                numbers = [float(m.group()) for m in num_pattern.finditer(text)]
                if len(numbers) >= 2:
                    position = (numbers[0], numbers[1])
            except:
                pass
        
        if position is None:
            continue
        
        if prev_pos is None:
            prev_pos = position
            continue
        
        dist = calc_distance(prev_pos[0], prev_pos[1], position[0], position[1])
        total_dist += dist
        prev_pos = position
    
    if total_dist <= 0:
        if verbose:
            print("no distance calculated")
        return None
    
    # m -> km
    return total_dist / 1000.0