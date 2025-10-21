import pandas as pd
import json
import math
from datetime import timedelta


# --- Helper Function for Formatting Time ---
def seconds_to_dhms(seconds):
    """Converts total seconds into a human-readable days, hours, minutes, seconds format."""
    seconds = abs(seconds)
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


# --- MAIN FUNCTION ---
def analyze_and_structure_process_data(
    event_log_data,
    case_id_col='case_id',
    activity_col='activity_name',
    start_time_col='start_time',
    complete_time_col='complete_time',
    bottleneck_top_n=3,
    owner_map=None,
    description_map=None,
    extras_map=None,
):
    """
    Analyzes process event logs and produces a business-friendly structured process flow output.
    Includes:
    - Global metrics summary (cycle times, variants, loops, bottlenecks)
    - Detailed node-level structure suitable for visualization or workflow diagrams
    """

    if not event_log_data:
        return json.dumps({"Error": "Event log data is empty."}, indent=4)

    try:
        df = pd.DataFrame(event_log_data)

        # --- 1. Prepare and Calculate Durations ---
        df[start_time_col] = pd.to_datetime(df[start_time_col], utc=True)
        df[complete_time_col] = pd.to_datetime(df[complete_time_col], utc=True)

        df['Event_Duration_Seconds'] = (
            df[complete_time_col] - df[start_time_col]
        ).dt.total_seconds()

        # Case-level cycle times
        case_start = df.groupby(case_id_col)[start_time_col].min().rename('Case_Start')
        case_end = df.groupby(case_id_col)[complete_time_col].max().rename('Case_End')
        df_cycle_times = pd.merge(case_start, case_end, on=case_id_col).reset_index()
        df_cycle_times['Cycle_Time_Seconds'] = (
            df_cycle_times['Case_End'] - df_cycle_times['Case_Start']
        ).dt.total_seconds()

        # --- Assign per-case positional labels like 1a,1b,... ---
        # We will create a column 'step_label' that denotes the ideal position within the case.
        def idx_to_label(idx):
            # idx is 0-based index within a case
            # map 0 -> 'a', 1 -> 'b', etc. For indices beyond 'z' we'll continue with 'aa', 'ab', ...
            label = ''
            n = idx
            while True:
                label = chr(ord('a') + (n % 26)) + label
                n = n // 26 - 1
                if n < 0:
                    break
            return label

        # For each case, enumerate the activities in chronological order and assign step numbers
        df = df.sort_values(by=[case_id_col, start_time_col])
        case_positions = []
        for case_id, group in df.groupby(case_id_col):
            # enumerate events in order
            for pos, (_, row) in enumerate(group.iterrows()):
                step_num = str(list(df[case_id_col].unique()).tolist().index(case_id) + 1) if False else None

        # Instead of the above incorrect approach, we'll build labels per case using a mapping loop
        df['step_within_case_idx'] = df.groupby(case_id_col).cumcount()
        df['step_within_case_label'] = df['step_within_case_idx'].apply(lambda x: idx_to_label(int(x)))
        # Compose full step label like '1a' where '1' is the case ordinal within the dataset (stable ordering by case id appearance)
        # Build an ordered list of unique case ids in appearance order
        unique_case_order = df[case_id_col].drop_duplicates().tolist()
        case_order_map = {case_id: (i + 1) for i, case_id in enumerate(unique_case_order)}
        df['case_order'] = df[case_id_col].map(case_order_map)
        df['step_label'] = df['case_order'].astype(str) + df['step_within_case_label']

        # --- 2. Global Metrics ---
        total_cases = df_cycle_times[case_id_col].nunique()
        avg_cycle_time_sec = df_cycle_times['Cycle_Time_Seconds'].mean()
        med_cycle_time_sec = df_cycle_times['Cycle_Time_Seconds'].median()
        min_cycle_time_sec = df_cycle_times['Cycle_Time_Seconds'].min()
        max_cycle_time_sec = df_cycle_times['Cycle_Time_Seconds'].max()

        # Throughput rate (cases per day)
        time_span_days = (df[complete_time_col].max() - df[start_time_col].min()).total_seconds() / (3600 * 24)
        throughput_rate = total_cases / time_span_days if time_span_days > 0 else 0

        max_steps_count = df.groupby(case_id_col).size().max()

        # Variants
        df_sequences = df.sort_values(by=[case_id_col, start_time_col]).groupby(case_id_col)[activity_col].apply(lambda x: tuple(x.tolist()))
        variant_counts = df_sequences.value_counts()
        total_variants = len(variant_counts)
        top_5_variants = [
            {"sequence": list(variant), "count": int(count), "percentage": round((count / total_cases) * 100, 2)}
            for variant, count in variant_counts.nlargest(5).items()
        ]

        # --- 3. Activity-Level Metrics ---
        activity_metrics = df.groupby(activity_col).agg(
            avg_duration=('Event_Duration_Seconds', 'mean'),
            total_count=(activity_col, 'size')
        ).reset_index()

        bottleneck_activities = activity_metrics.nlargest(bottleneck_top_n, 'avg_duration')[activity_col].tolist()

        # Loops
        # We'll detect loops per case by finding activities that appear more than once.
        loop_records = []  # list of dicts {case_id, activity, first_label, later_label}
        loop_activities = set()
        loop_records = []  # list of dicts {case_id, from_activity, to_activity, from_label, to_label}
        all_loop_activities = set()

        for case_id, group in df.groupby(case_id_col):
            group = group.sort_values(by=start_time_col).reset_index(drop=True)
            # Map activity to its most recent occurrence (index and label)
            last_occurrence = {}
            for idx, row in group.iterrows():
                act = row[activity_col]
                label = row['step_label']

                # If we've seen this activity before, a loop has occurred.
                if act in last_occurrence:
                    # The activity just before this repeated one is the 'from' node of the loop.
                    # Ensure we don't go out of bounds on the very first event.
                    if idx == 0: continue

                    from_activity_row = group.iloc[idx - 1]

                    loop_records.append({
                        'case_id': case_id,
                        'from_activity': from_activity_row[activity_col],
                        'to_activity': act, # The activity that is being repeated
                        'from_label': from_activity_row['step_label'],
                        'to_label': label
                    })
                    all_loop_activities.add(from_activity_row[activity_col])
                    all_loop_activities.add(act)

                # Update the last seen position of this activity
                last_occurrence[act] = (idx, label)

        # --- 4. Structure Output: Process Flow Nodes (Enhanced) ---

        process_flow_nodes = []
        unique_activities = df[activity_col].unique()

        # Create a mapping from activity name to the node ID that will be generated.
        # This allows us to reference the node ID before the node list is fully built.
        activity_to_node_id = {activity: str(i + 1) for i, activity in enumerate(unique_activities)}

        # --- Predefined mappings (owner, descriptions, optional extras) ---
        if owner_map is None:
            owner_map = {
                "Invoice Created": "Finance Team",
                "Invoice Sent": "Sales Team",
                "Payment Monitoring": "Finance Team",
                "Dispute Raised": "Customer Service",
                "Invoice Adjusted": "Finance Team",
                "Receipt Reconciled": "Finance Team",
                "Archive": "Finance Team",
            }

        if description_map is None:
            description_map = {
                "Invoice Created": [
                    "The invoice is generated in the system.",
                    "Finance team ensures all details are correct."
                ],
                "Invoice Sent": [
                    "The invoice is forwarded to the client.",
                    "Sales team tracks client receipt and acknowledgment."
                ],
                "Payment Monitoring": [
                    "Finance team monitors incoming payments.",
                    "System checks for overdue or pending transactions."
                ],
                "Dispute Raised": [
                    "Customer service reviews reported payment disputes.",
                    "Follow-up actions are initiated for resolution."
                ],
                "Invoice Adjusted": [
                    "Finance team adjusts invoice details after dispute review.",
                    "Corrections are documented for audit tracking."
                ],
                "Receipt Reconciled": [
                    "Payment receipt is verified and recorded.",
                    "Finance team ensures transaction accuracy."
                ],
                "Archive": [
                    "Process is finalized and archived for compliance.",
                    "No further user actions required."
                ],
            }

        if extras_map is None:
            extras_map = {}

        for i, activity in enumerate(unique_activities):
            metrics = activity_metrics[activity_metrics[activity_col] == activity].iloc[0]

            node = {
                "id": str(i + 1),
                "label": activity,
                "value": str(round(metrics["avg_duration"] / 60, 2)),  # avg duration in minutes
                "status": "completed" if activity == unique_activities[-1] else "in-progress",
                "owner": owner_map.get(activity, "Unassigned"),
                "descriptions": description_map.get(activity, [
                    f"{activity} occurred {int(metrics['total_count'])} times in the log."
                ]),
                "hasLoop": activity in all_loop_activities,
                "extras": extras_map.get(activity, []),
            }

            # If activity participates in any loops, attach loop connections listing all loop mappings where this activity is repeated
            if activity in loop_activities:
                print("")
                # gather unique loop mappings for this activity
            if activity in all_loop_activities:
                print("")
            # If this activity is the START of a loop, attach the connection details.
            if activity in all_loop_activities and any(r['from_activity'] == activity for r in loop_records):
                # Find all loops that start from the current activity
                mappings = [r for r in loop_records if r['from_activity'] == activity]

                import re

                def alpha_suffix_to_number(suffix: str) -> int:
                    # convert 'a'->1, 'z'->26, 'aa'->27, etc.
                    n = 0
                    for ch in suffix.lower():
                        n = n * 26 + (ord(ch) - ord('a') + 1)
                    return n

                def extract_suffix_number(label: str):
                    # match labels like '8g', '12aa' -> extract trailing letters and convert
                    if not label:
                        return None
                    m = re.match(r'^\d+([a-z]+)$', label, flags=re.I)
                    if m:
                        return alpha_suffix_to_number(m.group(1))
                    # fallback: take any trailing letters
                    m2 = re.search(r'([a-z]+)$', label, flags=re.I)
                    if m2:
                        return alpha_suffix_to_number(m2.group(1))
                    return None

                node_mappings = []
                for m in mappings:
                    from_num = extract_suffix_number(m.get('from_label', ''))
                    to_num = extract_suffix_number(m.get('to_label', ''))
                    node_mappings.append({
                        "case_id": m['case_id'],
                        # "from_label": m['from_label'],
                        # "to_label": m['to_label'],
                        "from": from_num,   # e.g. '8g' -> 7
                        "to": to_num,       # e.g. '8h' -> 8
                        # "from_node_id": activity_to_node_id.get(m['from_activity']),
                        # "to_node_id": activity_to_node_id.get(m['to_activity'])
                    })

                if node_mappings:
                    node["loopConnections"] = node_mappings
                node["loopConnections"] = node_mappings

            process_flow_nodes.append(node)

        # Prepare derived loop summary values
        cases_with_loops = sorted({r['case_id'] for r in loop_records})

        # --- 5. Final JSON Output ---
        final_output = {
            "global_metrics": {
                "Total_Completed_Cases": int(total_cases),
                "Case_Throughput_Rate_Per_Day": round(throughput_rate, 2),
                "Max_Steps_in_a_Case": int(max_steps_count),

                "Cycle_Time": {
                    "Average": seconds_to_dhms(avg_cycle_time_sec),
                    "Median": seconds_to_dhms(med_cycle_time_sec),
                    "Min": seconds_to_dhms(min_cycle_time_sec),
                    "Max": seconds_to_dhms(max_cycle_time_sec),
                },

                "Variant_Analysis": {
                    "Total_Number_of_Process_Variants": int(total_variants),
                    "Top_5_Process_Variants": top_5_variants
                },

                "Rework_Analysis_Simplified": {
                    "Activities_In_Loops": sorted(list(all_loop_activities)),
                    "Cases_With_Loops_Count": len(cases_with_loops),
                    "Cases_With_Loops": cases_with_loops,
                    "Loop_Details": loop_records,
                },

                "Bottleneck_Analysis_Simplified": {
                    "Top_Bottleneck_Activities": bottleneck_activities,
                    "Time_Lost_Simplified_Basis": "Calculated based on average activity duration."
                }
            },
            "process_flow_nodes": process_flow_nodes
        }

        return json.dumps(final_output, indent=4)

    except Exception as e:
        return json.dumps({"Error": f"An error occurred during process analysis: {e}"}, indent=4)


def load_event_log_from_csv(csv_path: str):
    """Load CSV and return list of dict records expected by analyzer.
    This function will attempt to use columns named in the default function
    signature (case_id, activity_name, start_time, complete_time). If any of
    those columns are missing it will raise a ValueError with a helpful message.
    """
    df = pd.read_csv(csv_path)

    required_cols = ['case_id', 'activity_name', 'start_time', 'complete_time']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Convert to dict records for the analyzer
    records = df.to_dict(orient='records')
    return records


if __name__ == '__main__':
    csv_file = 'invoice_process_expanded (1).csv'
    try:
        event_log = load_event_log_from_csv(csv_file)
        result = analyze_and_structure_process_data(event_log)
        print(result)
    except Exception as e:
        print(json.dumps({"Error": str(e)}, indent=4))
