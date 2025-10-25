import pandas as pd
import json
import math
from collections import Counter, defaultdict


# ---------------------------- Helper Functions ----------------------------

def seconds_to_dhms(seconds: float) -> str:
    seconds = abs(float(seconds))
    days = math.floor(seconds / (3600 * 24))
    seconds %= (3600 * 24)
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{round(seconds, 2)}s")
    return " ".join(parts)


def infer_step_index(df, case_id_col, start_time_col):
    df = df.sort_values([case_id_col, start_time_col]).copy()
    df["step_index"] = df.groupby(case_id_col).cumcount() + 1
    return df


def infer_ideal_positions(df, case_id_col, activity_col):
    pos_counts = defaultdict(Counter)
    for _, row in df[[activity_col, "step_index"]].iterrows():
        pos_counts[row[activity_col]][int(row["step_index"])] += 1

    ideal_positions = {}
    for act, counter in pos_counts.items():
        most_common = counter.most_common()
        if not most_common:
            continue
        top_freq = most_common[0][1]
        candidates = [pos for pos, freq in most_common if freq == top_freq]
        ideal_positions[act] = min(candidates)
    return ideal_positions


def compute_ideal_times(df, activity_col, ideal_positions):
    ideal_times = {}
    for act, pos in ideal_positions.items():
        mask = (df[activity_col] == act) & (df["step_index"] == pos)
        ideal_times[act] = float(df.loc[mask, "Event_Duration_Seconds"].median()) if mask.any() else None
    return ideal_times


def detect_loops(df, case_id_col, activity_col):
    loop_records = []
    loop_activities = set()
    for case_id, group in df.groupby(case_id_col):
        last_seen = {}
        for _, row in group.iterrows():
            act = row[activity_col]
            idx = int(row["step_index"])
            if act in last_seen:
                loop_records.append({"case_id": case_id, "activity": act, "from": last_seen[act], "to": idx})
                loop_activities.add(act)
            last_seen[act] = idx
    return loop_records, loop_activities


def detect_dropouts(df, activity_col, case_id_col, ideal_positions):
    activities = df[activity_col].unique().tolist()
    dropout_cases_by_activity = {a: set() for a in activities}

    ideal_sequence = sorted(ideal_positions.items(), key=lambda x: x[1])
    ideal_steps = [a for a, _ in ideal_sequence]

    for case_id, group in df.groupby(case_id_col):
        performed_steps = set(group[activity_col].tolist())
        missing = [s for s in ideal_steps if s not in performed_steps]
        for m in missing:
            dropout_cases_by_activity[m].add(case_id)

    is_dropout_map = {a: len(cases) > 0 for a, cases in dropout_cases_by_activity.items()}
    return is_dropout_map, dropout_cases_by_activity


def detect_bottlenecks(activity_metrics, ideal_times, activity_col, threshold_sec: int = 3600):
    """
    Detect activities that take significantly longer than their ideal duration.
    A bottleneck is flagged only if the delay (actual - ideal) > threshold_sec.
    Default threshold: 3600 seconds (1 hour).
    """
    is_bottleneck_map = {}
    bottleneck_delta_map = {}

    for _, r in activity_metrics.iterrows():
        act = r[activity_col]
        actual_time = float(r["avg_duration"]) if pd.notnull(r["avg_duration"]) else 0.0
        ideal_time_sec = ideal_times.get(act)

        if ideal_time_sec is None or ideal_time_sec == 0:
            is_bottleneck_map[act] = False
            bottleneck_delta_map[act] = 0.0
            continue

        delay = actual_time - ideal_time_sec

        if delay > threshold_sec:
            is_bottleneck_map[act] = True
            bottleneck_delta_map[act] = round(delay, 2)
        else:
            is_bottleneck_map[act] = False
            bottleneck_delta_map[act] = 0.0

    return is_bottleneck_map, bottleneck_delta_map


# ---------------------------- MAIN FUNCTION ----------------------------

def analyze_and_structure_process_data(
    event_log_data,
    case_id_col='case_id',
    activity_col='activity_name',
    start_time_col='start_time',
    complete_time_col='complete_time',
    owner_map=None,
    description_map=None,
    extras_map=None,
):
    if not event_log_data:
        return json.dumps({"Error": "Event log data is empty."}, indent=4)

    try:
        df = pd.DataFrame(event_log_data)
        df[start_time_col] = pd.to_datetime(df[start_time_col], utc=True)
        df[complete_time_col] = pd.to_datetime(df[complete_time_col], utc=True)
        df["Event_Duration_Seconds"] = (df[complete_time_col] - df[start_time_col]).dt.total_seconds()

        df = infer_step_index(df, case_id_col, start_time_col)
        ideal_positions = infer_ideal_positions(df, case_id_col, activity_col)
        ideal_times = compute_ideal_times(df, activity_col, ideal_positions)

        activity_metrics = (
            df.groupby(activity_col)
            .agg(
                avg_duration=("Event_Duration_Seconds", "mean"),
                total_count=(activity_col, "size"),
            )
            .reset_index()
        )

        loop_records, loop_activities = detect_loops(df, case_id_col, activity_col)
        is_dropout_map, dropout_cases_by_activity = detect_dropouts(df, activity_col, case_id_col, ideal_positions)
        is_bottleneck_map, bottleneck_delta_map = detect_bottlenecks(activity_metrics, ideal_times, activity_col)

        def sort_key(act_name):
            return (ideal_positions.get(act_name, 999999), act_name.lower())

        ordered_activities = sorted(activity_metrics[activity_col].tolist(), key=sort_key)
        loop_by_activity = defaultdict(list)
        for rec in loop_records:
            loop_by_activity[rec["activity"]].append(
                {"case_id": rec["case_id"], "from": rec["from"], "to": rec["to"]}
            )

        process_flow_nodes = []
        for idx, act in enumerate(ordered_activities, start=1):
            metrics = activity_metrics.loc[activity_metrics[activity_col] == act].iloc[0]
            avg_duration = float(metrics["avg_duration"])
            total_count = int(metrics["total_count"])

            descriptions = [f"{act} occurred {total_count} times in the log."]
            if is_bottleneck_map.get(act):
                delay = bottleneck_delta_map.get(act, 0.0)
                descriptions.append(f"There is a bottleneck in this step ({act}) — delay: {delay} sec.")
            if is_dropout_map.get(act):
                ex_cases = sorted(list(dropout_cases_by_activity.get(act, [])))
                if ex_cases:
                    case_list = ", ".join(ex_cases[:5])
                    descriptions.append(f"There is dropout in case(s): {case_list} — \"{act}\" was missed from these processes.")

            node = {
                "id": str(idx),
                "label": act,
                "value": str(round(avg_duration / 60, 2)),
                "status": "in-progress",
                "owner": owner_map.get(act, "Unassigned") if owner_map else "Unassigned",
                "descriptions": descriptions,
                "Is_Bottlenecks": "Yes" if is_bottleneck_map.get(act) else "No",
                "Is_Dropout": "Yes" if is_dropout_map.get(act) else "No",
                "hasLoop": act in loop_activities,
                "extras": extras_map.get(act, []) if extras_map else [],
            }

            if loop_by_activity.get(act):
                node["loopConnections"] = loop_by_activity[act]
            process_flow_nodes.append(node)

        return json.dumps({"process_flow_nodes": process_flow_nodes}, indent=4)

    except Exception as e:
        return json.dumps({"Error": f"An error occurred: {e}"}, indent=4)


# ---------------------------- CSV Loader & Entry Point ----------------------------

def load_event_log_from_csv(csv_path: str):
    """Load CSV and return list of dict records expected by analyzer."""
    df = pd.read_csv(csv_path)
    required_cols = ['case_id', 'activity_name', 'start_time', 'complete_time']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")
    return df.to_dict(orient='records')


if __name__ == '__main__':
    csv_file = 'invoice_process_expanded (1).csv'
    try:
        event_log = load_event_log_from_csv(csv_file)
        result = analyze_and_structure_process_data(event_log)
        print(result)
    except Exception as e:
        print(json.dumps({"Error": str(e)}, indent=4))
