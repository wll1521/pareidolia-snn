import os
import csv

def log_experiment(logfile: str, metrics: dict) -> None:
    file_exists = os.path.isfile(logfile)

    with open(logfile, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

    print(f"Logged run to {logfile}")
