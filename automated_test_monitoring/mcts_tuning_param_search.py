import csv
from functools import partial
import subprocess
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import logging
import time
import optparse
import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob
import os
from datetime import datetime
import logging

log_dir = 'results/benchmark_1/logs'
logfile_name = f'{log_dir}/logfile_MCTS_tunning_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.log'

os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(filename=logfile_name,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def run_command(date, command: list[str], arguments: list[str]):
    year, month, day = date.split('-')
    dates = ["--year", str(int(year)), "--month", str(int(month)), "--day", str(int(day))]
    full_command = command + dates + arguments

    logging.info(f"Starting: {full_command}")
    start_time = time.time()
    result = subprocess.run(full_command, shell=False, capture_output=True, text=True)
    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"Completed: {full_command} (Duration: {duration:.2f} seconds)")
    return result.stdout, result.stderr, duration


def main():

    start_time_script = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    parser = optparse.OptionParser()
    parser.add_option("--run_only_imperfect", action='store_true', dest='run_only_imperfect', default=False, help='decrese computaiton')
    (options, args) = parser.parse_args()

    
    MCTS_SELECTED_DATE: str = '2022-10-13' #'2022-11-06' 
    NUM_BUSES = 3
    K_MAX = 10
    MCTS_DEPTH = 10
    SINGLE_MCTREE_ITERATIONS = 500
    TIMEOUT = 60
    N_CHAINS = options. #set to 1 for perfect and 25 for imperfect

    arguments_for_buses = partial(generate_args, NUM_BUSES=NUM_BUSES, K_MAX=K_MAX, MCTS_DEPTH=MCTS_DEPTH, SINGLE_MCTREE_ITERATIONS=SINGLE_MCTREE_ITERATIONS, N_CHAINS=N_CHAINS, TIMEOUT=TIMEOUT)

    if options.run_only_imperfect:
        #COMPUTER 1, NO GPU with monitor
        WORKERS = 5
        commands = [
            # ["python3" ,"src/Evaluate_routing.py", "--mode", "2", "--cost_mode", "0", "--IMPERFECT_GENERATIVE_MODEL", "--save_MCTS_tuning_param"],
            ["python3", "src/Evaluate_routing.py", "--mode", "2", "--cost_mode", "1", "--IMPERFECT_GENERATIVE_MODEL", "--save_MCTS_tuning_param"],
            # ["python3", "src/Evaluate_routing.py", "--mode", "2", "--cost_mode", "2", "--IMPERFECT_GENERATIVE_MODEL", "--save_MCTS_tuning_param"],
        ]
    else:
        #COMPUTER 2 ssh here
        WORKERS = 10
        commands = [
            # ["python3", "src/Evaluate_routing.py", "--mode", "1", "--cost_mode", "0" ,"--save_MCTS_tuning_param"],
            # ["python3", "src/Evaluate_routing.py", "--mode", "2", "--cost_mode", "0" ,"--save_MCTS_tuning_param"],
            ["python3", "src/Evaluate_routing.py", "--mode", "1", "--cost_mode", "1", "--save_MCTS_tuning_param"],
            ["python3", "src/Evaluate_routing.py", "--mode", "2", "--cost_mode", "1", "--save_MCTS_tuning_param"],
            # ["python3", "src/Evaluate_routing.py", "--mode", "1", "--cost_mode", "2", "--save_MCTS_tuning_param"],
            # ["python3", "src/Evaluate_routing.py", "--mode", "2", "--cost_mode", "2", "--save_MCTS_tuning_param"],
        ]

    tasks = [(MCTS_SELECTED_DATE, cmd, arguments_for_buses(mcts_tuning_param)) for cmd in commands for mcts_tuning_param in [1, 10, 50, 100, 1000]]

    logging.info(f"Starting execution of {len(tasks)} tasks")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        future_to_task = {executor.submit(run_command, date, cmd, arguments): (date, cmd, arguments) for date, cmd, arguments in tasks}
        completed = 0
        for future in as_completed(future_to_task):
            date, cmd, arguments = future_to_task[future]
            try:
                stdout, stderr, duration = future.result()
                completed += 1
                with open(f'results/benchmark_1/logs/std_out/std_output_MCTS_tuning_{start_time_script}.txt', 'a') as f:
                    year, month, day = date.split('-')
                    full_command = f"{' '.join(cmd)} --year {int(year)} --month {int(month)} --day {int(day)} " + ' '.join(arguments)
                    f.write('----------------------------------------------------------------------------------------------------\n')
                    f.write(full_command + '\n')
                    f.write('----------------------------------------------------------------------------------------------------\n')
                    
                    f.write(stdout)
                if stderr:
                    logging.error(f"Error in command '{cmd}' for date {date}:")
                    logging.error(stderr)
                logging.info(f"Progress: {completed}/{len(tasks)} tasks completed")
            except Exception as exc:
                logging.error(f"Task {cmd} for date {date} generated an exception: {exc}")

    end_time = time.time()
    total_duration = end_time - start_time
    hours, remainder = divmod(total_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f"All tasks completed. Total execution time: {hours} hours, {minutes} minutes, and {seconds:.2f} seconds")



def generate_args(mcts_tuning_param, NUM_BUSES, K_MAX, MCTS_DEPTH, SINGLE_MCTREE_ITERATIONS, N_CHAINS, TIMEOUT):
    arguments = ["--num_buses", str(NUM_BUSES), "--K_MAX", str(K_MAX), "--MCTS_DEPTH", str(MCTS_DEPTH) ,"--SINGLE_MCTREE_ITERATIONS" ,str(SINGLE_MCTREE_ITERATIONS) ,"--MCTS_TUNING_PARAM", str(mcts_tuning_param) ,"--N_CHAINS", str(N_CHAINS), "--TIMEOUT", str(TIMEOUT)]
    return arguments

if __name__ == "__main__":
    main()