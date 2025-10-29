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
    # NOTE: This original function compared activity avg to an "ideal" time and a fixed threshold.
    # For simplified bottleneck analysis (used in global metrics), we'll instead compare activity
    # average duration against the overall average duration in the log. This function remains as a
    # compatibility wrapper but will return empty maps if ideal_times is None.
    is_bottleneck_map = {}
    bottleneck_delta_map = {}

    # If ideal_times is provided and non-empty, try to keep existing behavior fallback; otherwise
    # leave maps empty and caller can use alternate detection based on overall averages.
    if not ideal_times:
        return is_bottleneck_map, bottleneck_delta_map

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

        # loops, dropouts (per-activity), and the original bottleneck map (kept for compatibility)
        loop_records, loop_activities = detect_loops(df, case_id_col, activity_col)
        is_dropout_map, dropout_cases_by_activity = detect_dropouts(df, activity_col, case_id_col, ideal_positions)
        # keep original detect_bottlenecks behavior available but we'll compute a simplified variant below
        is_bottleneck_map_orig, bottleneck_delta_map = detect_bottlenecks(activity_metrics, ideal_times, activity_col)

        # ---- Global metrics calculations ----
        total_cases = int(df[case_id_col].nunique())
        # throughput: cases per day over the log time span
        min_start = df[start_time_col].min()
        max_end = df[complete_time_col].max()
        total_days = (max_end - min_start).total_seconds() / (3600 * 24) if pd.notnull(min_start) and pd.notnull(max_end) and max_end > min_start else 0
        throughput_per_day = round((total_cases / total_days) if total_days > 0 else float(total_cases), 2)

        # max steps in a case
        max_steps = int(df.groupby(case_id_col)["step_index"].max().max()) if not df.empty else 0

        # per-case sequences and cycle times
        case_groups = df.sort_values([case_id_col, "step_index"]).groupby(case_id_col)
        case_sequences = {}
        case_cycle_seconds = {}
        for case_id, grp in case_groups:
            seq = grp[activity_col].tolist()
            case_sequences[case_id] = tuple(seq)
            # cycle time for case: from first start to last complete
            start = grp[start_time_col].min()
            end = grp[complete_time_col].max()
            case_cycle_seconds[case_id] = (end - start).total_seconds() if pd.notnull(start) and pd.notnull(end) else 0

        variant_counts = Counter(case_sequences.values())
        total_variants = len(variant_counts)
        top_variants = variant_counts.most_common(5)
        top_variants_list = []
        for seq, cnt in top_variants:
            pct = round((cnt / total_cases) * 100, 2) if total_cases > 0 else 0.0
            top_variants_list.append({
                "sequence": list(seq),
                "count": int(cnt),
                "percentage": pct,
            })

        # cycle time statistics
        cycle_values = list(case_cycle_seconds.values())
        avg_cycle = float(pd.Series(cycle_values).mean()) if cycle_values else 0.0
        med_cycle = float(pd.Series(cycle_values).median()) if cycle_values else 0.0
        min_cycle = float(min(cycle_values)) if cycle_values else 0.0
        max_cycle = float(max(cycle_values)) if cycle_values else 0.0

        # simplified bottleneck detection: flag activity as bottleneck if its avg_duration > overall log avg
        overall_avg = float(df["Event_Duration_Seconds"].mean()) if not df["Event_Duration_Seconds"].isna().all() else 0.0
        is_bottleneck_map_overall = {}
        for _, r in activity_metrics.iterrows():
            act = r[activity_col]
            avg_dur = float(r["avg_duration"]) if pd.notnull(r["avg_duration"]) else 0.0
            is_bottleneck_map_overall[act] = avg_dur > overall_avg

        # happy path heuristic: if top variant share >= 40% consider happy path true
        top_pct = top_variants_list[0]["percentage"] if top_variants_list else 0.0
        happy_path = bool(top_pct >= 40.0)

        # ---- Build process_flow_nodes in requested format ----
        def sort_key(act_name):
            return (ideal_positions.get(act_name, 999999), act_name.lower())

        ordered_activities = sorted(activity_metrics[activity_col].tolist(), key=sort_key)
        loop_by_activity = defaultdict(list)
        for rec in loop_records:
            loop_by_activity[rec["activity"]].append({"case_id": rec["case_id"], "from": rec["from"], "to": rec["to"]})

        process_flow_nodes = []
        for idx, act in enumerate(ordered_activities, start=1):
            metrics = activity_metrics.loc[activity_metrics[activity_col] == act].iloc[0]
            avg_duration = float(metrics["avg_duration"]) if pd.notnull(metrics["avg_duration"]) else 0.0
            total_count = int(metrics["total_count"])

            descriptions = [f"Activity occurs {total_count} times.", f"Average processing time: {seconds_to_dhms(avg_duration)}"]
            if is_bottleneck_map_overall.get(act):
                descriptions.append("Flagged as bottleneck based on overall log average duration.")
            if is_dropout_map.get(act):
                ex_cases = sorted(list(dropout_cases_by_activity.get(act, [])))
                if ex_cases:
                    case_list = ", ".join(map(str, ex_cases[:5]))
                    descriptions.append(f"There is dropout in case(s): {case_list} — \"{act}\" was missed from these processes.")

            node = {
                "label": act,
                "hasLoop": act in loop_activities,
                "isBottleneck": bool(is_bottleneck_map_overall.get(act)),
                "isDropout": bool(is_dropout_map.get(act)),
                "id": str(idx),
                "value": str(round(avg_duration / 60, 2)),
                "loopConnections": None,
                "status": "final" if idx == len(ordered_activities) else "in-progress",
                "owner": owner_map.get(act, "N/A") if owner_map else "N/A",
                "descriptions": descriptions,
                "extras": extras_map.get(act, []) if extras_map else [],
            }

            # attach a single representative loop connection when present (first one)
            if loop_by_activity.get(act):
                first_conn = loop_by_activity[act][0]
                node["loopConnections"] = {"from": str(first_conn.get("from")), "to": str(first_conn.get("to"))}

            process_flow_nodes.append(node)

        global_metrics = {
            "Total_Completed_Cases": total_cases,
            "Case_Throughput_Rate_Per_Day": throughput_per_day,
            "Max_Steps_in_a_Case": max_steps,
            "Most_Frequent_Path": " -> ".join(top_variants_list[0]["sequence"]) if top_variants_list else "",
            "Cycle_Time": {
                "Average": seconds_to_dhms(avg_cycle),
                "Median": seconds_to_dhms(med_cycle),
                "Min": seconds_to_dhms(min_cycle),
                "Max": seconds_to_dhms(max_cycle),
            },
            "Variant_Analysis": {
                "Total_Number_of_Process_Variants": total_variants,
                "Top_5_Process_Variants": top_variants_list,
            },
            "Rework_Analysis_Simplified": {
                "Activities_In_Loops_Count": len(loop_activities),
                "Time_Lost_Simplified_Basis": "Loop/Rework analysis is now embedded in 'process_flow_nodes' loopConnections."
            },
            "Bottleneck_Analysis_Simplified": {
                "Bottlenecks_Based_on_Avg_Duration_Count": int(sum(1 for v in is_bottleneck_map_overall.values() if v)),
                "Time_Lost_Simplified_Basis": "Bottlenecks flagged if avg duration > overall log avg duration."
            },
            "happy_path": happy_path,
        }

        return json.dumps({"global_metrics": global_metrics, "process_flow_nodes": process_flow_nodes}, indent=4)

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