#!/usr/bin/python3
import csv
import os
import optparse
import datetime
import traceback

from Map_graph import Map_graph
import pandas as pd

from Trajectory import evaluate_trajectory
import os

from Data_structures import Config_flags, Data_folders, Simulator_config, Date_operational_range

import logging
from datetime import datetime, timedelta
from benchmark_1.logger_file import logger_dict


# current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# log_file_name = f'./logs/logging/logfile_{current_time}.log'
# logging.basicConfig(filename=log_file_name, level=logging.INFO)

def create_video(policy_name : str = 'base_policy'):
    os.system('ffmpeg -r 1 -start_number 0 -i data/results/'+policy_name+'/frame%0d.png -pix_fmt yuvj420p -vcodec mjpeg -f mov data/results/'+policy_name+'/trajectory.mov')

def evaluate_all_policies(map_object: Map_graph, policy_names: str, date_operational_range: Date_operational_range, 
                          data_folders: Data_folders, simulator_config: Simulator_config, config_flags: Config_flags) -> None:
    for key in policy_names.keys():
        policy_name=policy_names[key]
        evaluate_trajectory(map_object=map_object, 
                            policy_name=policy_name,
                            date_operational_range=date_operational_range,
                            data_folders=data_folders,
                            simulator_config=simulator_config,
                            config_flags=config_flags)
        if config_flags.create_vid:
            create_video(policy_name=policy_names[key])

def run_experiment(options=None) -> None:
    
    date_operational_range = Date_operational_range(year=options.year, 
                                                    month=options.month,
                                                    day=options.day,
                                                    start_hour=options.start_hour,
                                                    end_hour=options.end_hour)
    
    data_folders = Data_folders()
    
    policy_names = {0: "proactive_routing_perfect",
                    1: "greedy",
                    2: "MCTS"}
    
    cost_types = {0: "wait_time",
                  1: "ptt",
                  2: "budget"}
    
    
    map_object = Map_graph(initialize_shortest_path=options.init_shortest_path, 
                           routing_data_folder=data_folders.routing_data_folder,
                            area_text_file=data_folders.area_text_file, 
                            use_saved_map=options.use_saved_map, 
                            save_map_structure=options.save_map)

        
    simulator_config = Simulator_config(map_object=map_object,
                                        num_buses=options.num_buses,
                                        bus_capacity=options.bus_capacity,
                                        rejection_penalty=options.rejection_penalty)
    
    config_flags = Config_flags(
        consider_route_time=options.consider_route_time,
        include_scaling=options.include_scaling,
        verbose=options.verbose,
        process_requests=options.process_requests,
        plot_initial_routes=options.plot_initial_routes,
        plot_final_routes=options.plot_final_routes,
        create_vid=options.create_vid,
        include_rejected_requests=options.include_rejected_requests,
        cost_type=cost_types[options.cost_mode],
        MCTS_TUNING_PARAM=options.MCTS_TUNING_PARAM,
        DEBUGGING=options.DEBUGGING,
        VERBOSE_MCTS=options.VERBOSE_MCTS,
        VERBOSE_ALGO1=options.VERBOSE_ALGO1,
        VERBOSE_RV=options.VERBOSE_RV,
        VERBOSE_VV=options.VERBOSE_VV,
        IMPERFECT_GENERATIVE_MODEL=options.IMPERFECT_GENERATIVE_MODEL,
        num_buses=options.num_buses,
        TIME_TO_GET_INTO_BUS=options.TIME_TO_GET_INTO_BUS,
        MAX_WAIT_TIME_AT_STATION=options.MAX_WAIT_TIME_AT_STATION,
        MAX_WAIT_TIME_INSIDE_BUS=options.MAX_WAIT_TIME_INSIDE_BUS,
        MAX_ROUTE_TIME=options.MAX_ROUTE_TIME,
        START_HOUR=options.START_HOUR,
        START_OPERATIONAL_HOUR=options.START_OPERATIONAL_HOUR,
        K_MAX=options.K_MAX,
        MCTS_DEPTH=options.MCTS_DEPTH,
        SINGLE_MCTREE_ITERATIONS=options.SINGLE_MCTREE_ITERATIONS,
        SAMPLED_BANK_SIZE=options.SAMPLED_BANK_SIZE,
        N_CHAINS=options.N_CHAINS,
        TIMEOUT=options.TIMEOUT,
        PARALLEL=options.PARALLEL,
        RECOMPUTE_MEAN_STD=options.RECOMPUTE_MEAN_STD,
        SAVE_TEST_DAYS=options.SAVE_TEST_DAYS,
        save_MCTS_tuning_param = options.save_MCTS_tuning_param,
        save_different_buses = options.save_different_buses,
        BETTER_GEN_MODEL = options.BETTER_GEN_MODEL,
        year = options.year,
        month = options.month,
        day = options.day,
        FINAL_RESULTS_CSV_NAME = options.FINAL_RESULTS_CSV_NAME,
        WHICH_PARALLEL = options.WHICH_PARALLEL
    )

    logger_dict['num_buses'] = options.num_buses
    logger_dict['Imperfect generative model'] = options.IMPERFECT_GENERATIVE_MODEL
    logger_dict['Time to get into bus'] = options.TIME_TO_GET_INTO_BUS
    logger_dict['Max wait time at station'] = options.MAX_WAIT_TIME_AT_STATION
    logger_dict['Max wait time inside bus'] = options.MAX_WAIT_TIME_INSIDE_BUS
    logger_dict['Max route time'] = options.MAX_ROUTE_TIME
    logger_dict['Start hour'] = options.START_HOUR
    logger_dict['Start operational hour'] = options.START_OPERATIONAL_HOUR
    logger_dict['K max'] = options.K_MAX
    logger_dict['MCTS depth'] = options.MCTS_DEPTH
    logger_dict['MCTS iterations'] = options.SINGLE_MCTREE_ITERATIONS
    logger_dict['MCTS tuning parameter'] = options.MCTS_TUNING_PARAM
    logger_dict['MCTS_TUNING_PARAM'] = options.MCTS_TUNING_PARAM
    logger_dict['Sampled bank size'] = options.SAMPLED_BANK_SIZE
    logger_dict['N chains'] = options.N_CHAINS
    logger_dict['Timeout'] = options.TIMEOUT
    logger_dict['Parallel'] = options.PARALLEL
    logger_dict['Recompute mean std'] = options.RECOMPUTE_MEAN_STD
    logger_dict['Save test days'] = options.SAVE_TEST_DAYS
    logger_dict['MODE'] = options.mode
    logger_dict['POLICY'] = policy_names[options.mode].upper()
    logger_dict['COST_MODE'] = options.cost_mode
    logger_dict['COST TYPE'] = cost_types[options.cost_mode].upper()
    logger_dict['num buses'] = options.num_buses
    logger_dict['BUS_CAPACITY'] = options.bus_capacity
    logger_dict['START DATETIME'] = datetime(options.year, options.month, options.day, options.start_hour)
    logger_dict['END DATETIME'] = datetime(options.year, options.month, options.day, options.end_hour)
    generative_model = {True: "PERFECT", False: "IMPERFECT"}
    logger_dict['perfect model'] = generative_model[not options.IMPERFECT_GENERATIVE_MODEL]
    logger_dict['save_MCTS_tuning_param'] = options.save_MCTS_tuning_param
    logger_dict['save_different_buses'] = options.save_different_buses
    logger_dict['BETTER_GEN_MODEL'] = options.BETTER_GEN_MODEL
    logger_dict['year'] = options.year
    logger_dict['month'] = options.month
    logger_dict['day'] = options.day
    logger_dict['IMPERFECT_GENERATIVE_MODEL'] = options.IMPERFECT_GENERATIVE_MODEL
    logger_dict['cost_mode'] = options.cost_mode
    logger_dict['mode'] = options.mode
    logger_dict['WHICH_PARALLEL'] = options.WHICH_PARALLEL

    if options.mode < 5:
        policy_name=policy_names[options.mode]
        evaluate_trajectory(map_object=map_object, 
                            policy_name=policy_name,
                            date_operational_range=date_operational_range,
                            data_folders=data_folders,
                            simulator_config=simulator_config,
                            config_flags=config_flags,
                            truncation_horizon=options.truncation_horizon)
    
    elif options.mode == 5:
        evaluate_all_policies(map_object=map_object, 
                            policy_names=policy_names,
                            date_operational_range=date_operational_range,
                            data_folders=data_folders,
                            simulator_config=simulator_config,
                            config_flags=config_flags)
        
        
    else:
        print("Wrong mode selected. Please try again...")

def main() -> int:
    ##############################################
    # Main function, Options
    ##############################################
    parser = optparse.OptionParser()
    parser.add_option("--use_saved_map", action='store_true', dest='use_saved_map', default=True, help='Flag for using saved map data instead of populating new structure')
    parser.add_option("--save_map", action='store_true', dest='save_map', default=False, help='Flag for saving the map structure')
    parser.add_option("--verbose", action='store_true', dest='verbose', default=False, help='Flag for printing additional route information')
    parser.add_option("--create_vid", action='store_true', dest='create_vid', default=False, help='Flag for generating video after plotting')
    parser.add_option("--init_shortest_path", action='store_true', dest='init_shortest_path', default=False, help='Flag for initializing pairwise shortest path distances in the map_graph structure')
    parser.add_option("--plot_initial_routes", action='store_true', dest='plot_initial_routes', default=False, help='Flag for graphing the initial bus routes')
    parser.add_option("--plot_final_routes", action='store_true', dest='plot_final_routes', default=False, help='Flag for graphing the final bus routes')
    parser.add_option("--process_requests", action='store_true', dest='process_requests', default=False, help='Flag for processing requests before loading them')
    parser.add_option("--consider_route_time", action='store_true', dest='consider_route_time', default=False, help='Flag for considering route time in the cost')
    parser.add_option("--include_scaling", action='store_true', dest='include_scaling', default=False, help='Flag for considering scaling coefficient for the wait time cost calculation')
    parser.add_option("--approximate", action='store_true', dest='approximate', default=False, help='Flag for approximating request insertion cost')
    parser.add_option("--num_buses", type=int, dest='num_buses', default=3, help='Number of buses to be ran in the simulation')
    parser.add_option("--truncation_horizon", type=int, dest='truncation_horizon', default=20, help='Number of requests to be considered in rollout planning horizon before truncation.')
    parser.add_option("--bus_capacity", type=int, dest='bus_capacity', default=20, help='Bus capacity for the buses in the fleet')
    

    
    parser.add_option("--start_hour", type=int, dest='start_hour', default=19, help='Select start hour for the time range of interest') #19
    parser.add_option("--end_hour", type=int, dest='end_hour', default=2, help='Select end hour for the time range of interest')
    parser.add_option("--rejection_penalty", type=int, dest='rejection_penalty', default=28800, help='Penalty for rejecting a request. Default value set to the number of seconds in 8 hours.')
    parser.add_option("--include_rejected_requests", action='store_true', dest='include_rejected_requests', default=False, help='Flag for considering rejected requests in the cost calculation for rollout.')
    parser.add_option("--DEBUGGING", action='store_true', dest='DEBUGGING', default=False, help='Running file with small parameters for debugging')
    parser.add_option("--VERBOSE_MCTS", action='store_true', dest='VERBOSE_MCTS', default=False, help='Running file with small parameters for debugging')
    parser.add_option("--VERBOSE_ALGO1", action='store_true', dest='VERBOSE_ALGO1', default=False, help='Running file with small parameters for debugging')
    parser.add_option("--VERBOSE_RV", action='store_true', dest='VERBOSE_RV', default=False, help='Running file with small parameters for debugging')
    parser.add_option("--VERBOSE_VV", action='store_true', dest='VERBOSE_VV', default=False, help='Running file with small parameters for debugging')
    parser.add_option("--TIME_TO_GET_INTO_BUS", type="int", dest='TIME_TO_GET_INTO_BUS', default=60, help='Time to get into bus')
    parser.add_option("--MAX_ROUTE_TIME", type="int", dest='MAX_ROUTE_TIME', default=8*3600, help='Max route time')
    parser.add_option("--START_HOUR", type="int", dest='START_HOUR', default=19, help='Start hour')
    parser.add_option("--START_OPERATIONAL_HOUR", type="int", dest='START_OPERATIONAL_HOUR', default=19, help='Start operational hour')



    parser.add_option("--SAMPLED_BANK_SIZE", type="int", dest='SAMPLED_BANK_SIZE', default=10000, help='Sampled bank size')
    parser.add_option("--PARALLEL", type="int", dest='PARALLEL', default=0, help='Parallel')
    parser.add_option("--RECOMPUTE_MEAN_STD", action='store_true', dest='RECOMPUTE_MEAN_STD', default=False, help='Recompute mean std')
    parser.add_option("--SAVE_TEST_DAYS", action='store_true', dest='SAVE_TEST_DAYS', default=False, help='Save test days')
    parser.add_option("--save_MCTS_tuning_param", action='store_true', dest='save_MCTS_tuning_param', default=False, help='Save test days')
    parser.add_option("--save_different_buses", action='store_true', dest='save_different_buses', default=False, help='Save test days')
    
    #python3 src/Evaluate_routing.py --mode 2 --cost_mode 1 --IMPERFECT_GENERATIVE_MODEL True --BETTER_GEN_MODEL True
    parser.add_option("--MCTS_TUNING_PARAM", type="int", dest='MCTS_TUNING_PARAM', default=1000, help='MCTS tuning parameter')
    parser.add_option("--TIMEOUT", type="int", dest='TIMEOUT', default=60, help='Timeout')
    parser.add_option("--N_CHAINS", type="int", dest='N_CHAINS', default=1, help='Number of chains')
    parser.add_option("--K_MAX", type="int", dest='K_MAX', default=5, help='K max')
    parser.add_option("--MCTS_DEPTH", type="int", dest='MCTS_DEPTH', default=6, help='MCTS depth')
    parser.add_option("--SINGLE_MCTREE_ITERATIONS", type="int", dest='SINGLE_MCTREE_ITERATIONS', default=100, help='Single MCTree iterations')
    
    #this flag is not working enable this ayutomaticaly in the insertion procedure file
    parser.add_option("--MAX_WAIT_TIME_AT_STATION", type="int", dest='MAX_WAIT_TIME_AT_STATION', default=15*60, help='Max wait time at station')
    parser.add_option("--MAX_WAIT_TIME_INSIDE_BUS", type="int", dest='MAX_WAIT_TIME_INSIDE_BUS', default=15*60, help='Max wait time inside bus')

    
    parser.add_option("--year", type=int, dest='year', default=2023, help='Select year of interest.')
    parser.add_option("--month", type=int, dest='month', default=3, help='Select month of interest.')
    parser.add_option("--day", type=int, dest='day', default=1, help='Select day of interest.')
    
    
    parser.add_option("--mode", type=int, dest='mode', default=2, help='Select mode of operation.') #policy 2 ==> MCTS  and 1 ==> GREEDY
    parser.add_option("--cost_mode", type=int, dest='cost_mode', default=1, help='Select mode for cost type.') #cost 1 ==> PTT 
    parser.add_option("--IMPERFECT_GENERATIVE_MODEL", action='store_true', dest='IMPERFECT_GENERATIVE_MODEL', default=False, help='Running file with small parameters for debugging') #TRUE use gen model
    parser.add_option("--BETTER_GEN_MODEL", action='store_true', dest='BETTER_GEN_MODEL', default=False , help='Save test days') # True my model #TODO make this work
    
    parser.add_option("--FINAL_RESULTS_CSV_NAME", action='store', dest='FINAL_RESULTS_CSV_NAME', default='final_results.csv', help='Save test days')    
    parser.add_option("--WHICH_PARALLEL", type=int, dest='WHICH_PARALLEL', default=0, help='Select year of interest.')   

    
    (options, args) = parser.parse_args()
    print("date", options.year, options.month, options.day, "start_hour", options.start_hour, "end_hour", options.end_hour)
    print("mode", options.mode, "cost_mode", options.cost_mode, "IMPERFECT_GENERATIVE_MODEL", options.IMPERFECT_GENERATIVE_MODEL, "BETTER_GEN_MODEL", options.BETTER_GEN_MODEL)
    ##############################################
    # Main
    ##############################################
    run_experiment(options)
    return 0


if __name__ == '__main__':
    """Performs execution delta of the process."""
    pStart = datetime.now()
    logger_dict['experiment start time'] = pStart
    try:
        main()
    except Exception as errorMainContext:
        print("Fail End Process: ", errorMainContext)
        traceback.print_exc()
    qStop = datetime.now()
    time_difference=  qStop-pStart
    print("Execution time: " + str(qStop-pStart))

    logger_dict['Execution time'] = time_difference.total_seconds() / 60
    logger_dict = {k: [v] for k, v in logger_dict.items()}
    df = pd.DataFrame.from_dict(logger_dict)
    df = df[['START DATETIME',
             'POLICY',
             'COST TYPE',
             'perfect model',
             'FREEDOM COST',
             'Total route time',
             'serviced requests',
             'all requests',
             'Average Passenger Trip Time',
             'experiment start time',
             'END DATETIME',
             'Average Passenger Wait Time inside the bus',
             'Average Passenger Wait Time at the origin station',
             'K max',
             'MCTS depth',
             'MCTS iterations',
             'MCTS tuning parameter',
             'N chains',
             'Timeout',
             'Number of requests serviced',
             'Number of passengers serviced',
             'too long wait time',
             'num buses',
             'Execution time',
            ]
            ]
    
    directory_name = './results/benchmark_1/'
    if logger_dict['save_different_buses'][0]:
        directory_name ='./results/benchmark_1/different_buses/'
    if logger_dict['save_MCTS_tuning_param'][0]:
        directory_name = './results/benchmark_1/mcts_tuning/'
    
    if logger_dict['POLICY'][0] == 'GREEDY':
        file_name = directory_name + f'{logger_dict["START DATETIME"][0].date()}_{logger_dict["POLICY"][0]}_{logger_dict["COST TYPE"][0]}_BUSES_{logger_dict["num buses"][0]}_MCTS_TUNING_{logger_dict["MCTS tuning parameter"][0]}.csv'
    else:
        file_name = directory_name + f'{logger_dict["START DATETIME"][0].date()}_{logger_dict["POLICY"][0]}_{logger_dict["COST TYPE"][0]}_{logger_dict["perfect model"][0]}_BUSES_{logger_dict["num buses"][0]}_MCTS_TUNING_{logger_dict["MCTS tuning parameter"][0]}.csv'
    df.to_csv(file_name, header=True, index=False)
    
    
