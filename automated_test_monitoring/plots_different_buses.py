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
logfile_name = f'{log_dir}/logfile_different_buses_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.log'

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
    parser.add_option("--NUM_CHAINS", action='store', type='int', dest='NUM_CHAINS', default=1, help='decrease computation')    
    (options, args) = parser.parse_args()

    
    with open('data/test_days_first_four.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip the header
        dates = list(reader)

    K_MAX = 10
    MCTS_DEPTH = 10
    SINGLE_MCTREE_ITERATIONS = 500
    TIMEOUT = 60
    MCTS_TUNING_PARAM = 100
    
    if options.run_only_imperfect:
        N_CHAINS = 5 #set to 1 for perfect and 25 for imperfect
    else:
        N_CHAINS = 1
        

    arguments_for_buses = partial(generate_args, K_MAX=K_MAX, MCTS_DEPTH=MCTS_DEPTH, SINGLE_MCTREE_ITERATIONS=SINGLE_MCTREE_ITERATIONS, MCTS_TUNING_PARAM=MCTS_TUNING_PARAM, N_CHAINS=N_CHAINS, TIMEOUT=TIMEOUT)

    if options.run_only_imperfect:
        #COMPUTER 1, NO GPU with monitor
        WORKERS = 16
        commands = [
            ["python3", "src/Evaluate_routing.py", "--mode", "2", "--cost_mode", "1", "--IMPERFECT_GENERATIVE_MODEL", "--save_MCTS_tuning_param"],
        ]

    else:
        #COMPUTER 2 ssh here
        WORKERS = 16 #its ok greedy will run super fast
        commands = [
            ["python3", "src/Evaluate_routing.py", "--mode", "1", "--cost_mode", "1", "--save_MCTS_tuning_param"],
            ["python3", "src/Evaluate_routing.py", "--mode", "2", "--cost_mode", "1", "--save_MCTS_tuning_param"],
        ]

    tasks = [(date[0], cmd, arguments_for_buses(bus_number)) for date in dates for cmd in commands for bus_number in [2, 3, 4, 5]] 

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
                with open(f'results/benchmark_1/logs/std_out/std_output_DIFFERENT_BUSES_{start_time_script}.txt', 'a') as f:
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


def generate_args(bus_number, K_MAX, MCTS_DEPTH, SINGLE_MCTREE_ITERATIONS, MCTS_TUNING_PARAM, N_CHAINS, TIMEOUT):
    arguments = ["--num_buses", str(bus_number), "--K_MAX", str(K_MAX), "--MCTS_DEPTH", str(MCTS_DEPTH), "--SINGLE_MCTREE_ITERATIONS", str(SINGLE_MCTREE_ITERATIONS), "--MCTS_TUNING_PARAM", str(MCTS_TUNING_PARAM), "--N_CHAINS", str(N_CHAINS), "--TIMEOUT", str(TIMEOUT)]
    return arguments

if __name__ == "__main__":
    main()