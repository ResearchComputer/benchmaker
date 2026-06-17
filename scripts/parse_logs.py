import os
import json
import pandas as pd

JOB_LOGS_DIR = "jobs/"

# list all dirs under JOB_LOGS_DIR
job_dirs = [d for d in os.listdir(JOB_LOGS_DIR) if os.path.isdir(os.path.join(JOB_LOGS_DIR, d))]
data = []

for job_dir in job_dirs:
    datum = {}
    if "lock.json" in os.listdir(os.path.join(JOB_LOGS_DIR, job_dir)):
        with open(os.path.join(JOB_LOGS_DIR, job_dir, "lock.json"), "r") as f:
            record = json.load(f)
            datum["mode"]=record['trials'][0]['agent']['import_path']
            datum["concurrency"] = record['n_concurrent_trials']
    
    if "result.json" in os.listdir(os.path.join(JOB_LOGS_DIR, job_dir)):
        with open(os.path.join(JOB_LOGS_DIR, job_dir, "result.json"), "r") as f:
            record = json.load(f)
            if "pi-host__GLM-4.7-Flash__swebench-verified" in record['stats']['evals']:
                datum["solved_tasks"] = len(record['stats']['evals']['pi-host__GLM-4.7-Flash__swebench-verified']['reward_stats']['reward']['1.0'])
            elif "pi-container__GLM-4.7-Flash__swebench-verified" in record['stats']['evals']:
                datum["solved_tasks"] = len(record['stats']['evals']['pi-container__GLM-4.7-Flash__swebench-verified']['reward_stats']['reward']['1.0'])
            datum["total_tasks"] = record['n_total_trials']
            job_start_time = record['started_at']
            job_end_time = record['finished_at']
            datum["duration"] = (pd.to_datetime(job_end_time) - pd.to_datetime(job_start_time)).total_seconds()
    datum['id'] = job_dir
    data.append(datum)
    
with open(".local/parsed_logs.csv", "w") as f:
    df = pd.DataFrame(data)
    df.to_csv(f, index=False)