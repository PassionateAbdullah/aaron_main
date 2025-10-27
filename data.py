# Sample data for testing purposes
test_data = {
    "Current_Project_Data": {
        "Metadata": {
            "project_id": 39,
            "process_name": "second process 3",
            "department": "admin 1",
            "team": "Team 2"
        },
        "KPIs": {
            "cycle_time_metrics": {
                "avg_cycle_time": 2072.57,
                "median_cycle_time": 183.15,
                "min_cycle_time": 13.2,
                "max_cycle_time": 13919.78,
                "time_unit": "hours"
            },
            "idle_time_metrics": {
                "total_idle_time_hours": 1200.5,
                "idle_time_ratio": 0.65
            },
            "loop_metrics": {
                "total_loops": 55,
                "looping_case_ratio": 0.25
            },
            "bottleneck_metrics": {
                "bottleneck_activity": "Payment Received",
                "bottleneck_duration_sum": 50.8
            },
            "steps_per_case": {
                "avg_steps_per_case": 8.5,
                "total_cases_analyzed": 1500,
                "total_steps": 12750
            },
            "dropout_metrics": {
                "dropout_rate": 0.032,
                "completed_cases": 1452
            },
            "activity_duration_metrics": {
                "average_activity_durations": [
                    {"activity": "Invoice Created", "avg_duration_hours": 0.5},
                    {"activity": "Payment Monitoring", "avg_duration_hours": 1.2}
                ],
                "avg_activity_time_hours": 0.9
            },
            "top_variants": {
                "top_variants": [
                    {
                        "variant_path": "Invoice Created > Invoice Sent > Payment Received",
                        "count": 250,
                        "percentage": 0.35
                    },
                    {
                        "variant_path": "Invoice Created > Dispute Raised > Invoice Resent",
                        "count": 150,
                        "percentage": 0.21
                    }
                ]
            },
            "first_pass_rate": {
                "first_pass_rate": 0.75,
                "total_cases_analyzed": 1500
            },
            "longest_waiting_time": {
                "longest_waiting_activity": "Archive",
                "max_waiting_time": 45.3
            },
            "complexity_index": {"complexity_index": 2.5},
            "time_lost_to_bottleneck": {
                "time_lost_to_bottleneck_hours": 1200.5
            },
            "process_efficiency_ratio": 0.35
        }
    },
    "Related_Project_Data": {
        "Metadata": {
            "project_id": 42,
            "process_name": "process 4",
            "department": "finance 1",
            "team": "Team 1"
        },
        "KPIs": {
            "cycle_time_metrics": {
                "avg_cycle_time": 1450.2,
                "median_cycle_time": 120.4,
                "min_cycle_time": 10.8,
                "max_cycle_time": 9800.5,
                "time_unit": "hours"
            },
            "idle_time_metrics": {
                "total_idle_time_hours": 800.3,
                "idle_time_ratio": 0.48
            },
            "loop_metrics": {
                "total_loops": 38,
                "looping_case_ratio": 0.17
            },
            "bottleneck_metrics": {
                "bottleneck_activity": "Invoice Approval",
                "bottleneck_duration_sum": 30.6
            },
            "steps_per_case": {
                "avg_steps_per_case": 7.3,
                "total_cases_analyzed": 1600,
                "total_steps": 11680
            },
            "dropout_metrics": {
                "dropout_rate": 0.018,
                "completed_cases": 1571
            },
            "activity_duration_metrics": {
                "average_activity_durations": [
                    {"activity": "Invoice Created", "avg_duration_hours": 0.4},
                    {"activity": "Payment Monitoring", "avg_duration_hours": 1.0}
                ],
                "avg_activity_time_hours": 0.7
            },
            "top_variants": {
                "top_variants": [
                    {
                        "variant_path": "Invoice Created > Invoice Sent > Payment Received",
                        "count": 300,
                        "percentage": 0.41
                    },
                    {
                        "variant_path": "Invoice Created > Payment Pending > Payment Received",
                        "count": 120,
                        "percentage": 0.16
                    }
                ]
            },
            "first_pass_rate": {
                "first_pass_rate": 0.83,
                "total_cases_analyzed": 1600
            },
            "longest_waiting_time": {
                "longest_waiting_activity": "Verification",
                "max_waiting_time": 32.8
            },
            "complexity_index": {"complexity_index": 1.9},
            "time_lost_to_bottleneck": {
                "time_lost_to_bottleneck_hours": 820.7
            },
            "process_efficiency_ratio": 0.49
        }
    }
}
