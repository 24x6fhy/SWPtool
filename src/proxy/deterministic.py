# proxy functions
import sqlite3
import re


def match_weights(topic, weights):
    for pattern in weights:
        if re.match(pattern, topic):
            return weights[pattern]
    return 1.0


def weighted_msg_count(db_path, weights, start_ns=None, end_ns=None):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT id, name FROM topics")
    topics = {row[0]: row[1] for row in cur.fetchall()}

    if start_ns is not None and end_ns is not None:
        cur.execute(
            "SELECT topic_id, COUNT(*) FROM messages WHERE timestamp >= ? AND timestamp < ? GROUP BY topic_id",
            (int(start_ns), int(end_ns))
        )
    else:
        cur.execute("SELECT topic_id, COUNT(*) FROM messages GROUP BY topic_id")
    
    counts = cur.fetchall()
    conn.close()

    result = {}
    for tid, cnt in counts:
        name = topics.get(tid, "unknown")
        w = match_weights(name, weights)
        result[name] = cnt * w
    
    return result


def simple_msg_count(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM messages")
    n = cur.fetchone()[0]
    conn.close()
    return n


def get_drive_duration(db_path, time_unit="seconds"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT MIN(timestamp), MAX(timestamp) from messages")
    row = cur.fetchone()
    conn.close()

    t1, t2 = row[0], row[1]
    if t1 is None or t2 is None:
        return 0
    
    secs = (t2 - t1) / 1e9

    if time_unit == "seconds":
        return secs
    elif time_unit == "minutes":
        return secs / 60
    elif time_unit == "hours":
        return secs / 3600
    else:
        raise ValueError(f"unknown time_unit: {time_unit}")


def print_topics(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM topics")
    for tid, name in cur.fetchall():
        print(f"  {tid}: {name}")
    conn.close()

