from collections import Counter
import os
import functools
import osmnx as ox
from frozendict import frozendict
from Map_graph import Map_graph

class Config_flags:
    def __init__(self, consider_route_time: bool, include_scaling: bool, verbose: bool,
                 process_requests: bool, plot_initial_routes: bool, plot_final_routes: bool,
                 create_vid: bool, include_rejected_requests: bool,
                 cost_type: str, MCTS_TUNING_PARAM, DEBUGGING, VERBOSE_MCTS, 
                 VERBOSE_ALGO1, VERBOSE_RV, VERBOSE_VV, IMPERFECT_GENERATIVE_MODEL,
                 num_buses, TIME_TO_GET_INTO_BUS, MAX_WAIT_TIME_AT_STATION,
                 MAX_WAIT_TIME_INSIDE_BUS, MAX_ROUTE_TIME, START_HOUR,
                 START_OPERATIONAL_HOUR, K_MAX, MCTS_DEPTH, SINGLE_MCTREE_ITERATIONS,
                 SAMPLED_BANK_SIZE, N_CHAINS, TIMEOUT, PARALLEL, RECOMPUTE_MEAN_STD,
                 SAVE_TEST_DAYS, save_MCTS_tuning_param, save_different_buses,BETTER_GEN_MODEL, 
                 year, month, day, FINAL_RESULTS_CSV_NAME, WHICH_PARALLEL) -> None:
        self.consider_route_time = consider_route_time
        self.include_scaling = include_scaling
        self.verbose = verbose
        self.process_requests = process_requests
        self.plot_initial_routes = plot_initial_routes
        self.plot_final_routes = plot_final_routes
        self.create_vid = create_vid
        self.include_rejected_requests = include_rejected_requests
        self.cost_type = cost_type
        self.MCTS_TUNING_PARAM = MCTS_TUNING_PARAM
        self.DEBUGGING = DEBUGGING
        self.VERBOSE_MCTS = VERBOSE_MCTS
        self.VERBOSE_ALGO1 = VERBOSE_ALGO1
        self.VERBOSE_RV = VERBOSE_RV
        self.VERBOSE_VV = VERBOSE_VV
        self.IMPERFECT_GENERATIVE_MODEL = IMPERFECT_GENERATIVE_MODEL
        self.num_buses = num_buses
        self.TIME_TO_GET_INTO_BUS = TIME_TO_GET_INTO_BUS
        self.MAX_WAIT_TIME_AT_STATION = MAX_WAIT_TIME_AT_STATION
        self.MAX_WAIT_TIME_INSIDE_BUS = MAX_WAIT_TIME_INSIDE_BUS
        self.MAX_ROUTE_TIME = MAX_ROUTE_TIME
        self.START_HOUR = START_HOUR
        self.START_OPERATIONAL_HOUR = START_OPERATIONAL_HOUR
        self.K_MAX = K_MAX
        self.MCTS_DEPTH = MCTS_DEPTH
        self.SINGLE_MCTREE_ITERATIONS = SINGLE_MCTREE_ITERATIONS
        self.SAMPLED_BANK_SIZE = SAMPLED_BANK_SIZE
        self.N_CHAINS = N_CHAINS
        self.TIMEOUT = TIMEOUT
        self.PARALLEL = PARALLEL
        self.RECOMPUTE_MEAN_STD = RECOMPUTE_MEAN_STD
        self.SAVE_TEST_DAYS = SAVE_TEST_DAYS
        self.save_MCTS_tuning_param = save_MCTS_tuning_param
        self.save_different_buses = save_different_buses
        self.BETTER_GEN_MODEL = BETTER_GEN_MODEL
        self.year = year
        self.month = month
        self.day = day
        self.FINAL_RESULTS_CSV_NAME = FINAL_RESULTS_CSV_NAME
        self.WHICH_PARALLEL = WHICH_PARALLEL
        

    
    def __hash__(self) -> int:
        return hash(tuple((self.consider_route_time, self.include_scaling, self.verbose, \
                          self.process_requests, self.plot_initial_routes, self.plot_final_routes, \
                          self.create_vid)))

class Data_folders:

    def __init__(self, processed_requests_folder_path: str = "data/processed_requests/", 
                 predicted_requests_folder_path: str = "data/predicted_requests/",
                 request_folder_path: str = "data/requests/",
                 routing_data_folder: str = "data/routing_data",
                 area_text_file: str = "data/Evening_Van_polygon.txt",
                 static_results_folder: str = "results/static_routes",
                 dynamic_results_folder: str = "results/dynamic_routes",
                 weather_folder: str = "data/weather",
                 model_data_folder: str = "data/model_data") -> None:
                 
        self._check_and_create_folder(processed_requests_folder_path)
        self.processed_requests_folder_path = processed_requests_folder_path

        self._check_and_create_folder(predicted_requests_folder_path)
        self.predicted_requests_folder_path = predicted_requests_folder_path

        self._check_and_create_folder(routing_data_folder)
        self.routing_data_folder = routing_data_folder

        self.area_text_file = area_text_file
        self.request_folder_path = request_folder_path

        self._check_and_create_folder(static_results_folder)
        self.static_results_folder = static_results_folder
        self._check_and_create_folder(dynamic_results_folder)
        self.dynamic_results_folder = dynamic_results_folder

        self.dynamic_policy_results_folder = dynamic_results_folder

        self.weather_folder = weather_folder

        self._check_and_create_folder(model_data_folder)
        self.model_data_folder = model_data_folder
    
    def _check_and_create_folder(self, folder_path: str):
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
    
    def __repr__(self) -> str:
        folder_str = ""
        return folder_str

class Simulator_config:

    def __init__(self, map_object: Map_graph, num_buses: int, bus_capacity: int, rejection_penalty: int) -> None:
        # Alston
        depot_latitude = 42.3614251
        depot_longitude = -71.1283633

        # # Cambridge
        # depot_latitude = 42.3818293
        # depot_longitude = -71.1292293

        self.num_buses = num_buses
        self.bus_capacities = [bus_capacity] * num_buses
        self.rejection_penalty = rejection_penalty
        depot_longitudes = [depot_longitude] * num_buses
        depot_latitudes = [depot_latitude] * num_buses
        self.initial_bus_locations = ox.nearest_nodes(map_object.G, depot_longitudes, depot_latitudes)
    
    def __repr__(self) -> str:
        operational_range_str = ""
        return operational_range_str

class Temporal_config:
    def __init__(self, requests_locations_folder: str = "data/model_data/requests_locations",
                 requests_number_folder: str = "data/model_data/requests_numbers", weather_output_file: str = "weather_features.csv",
                 interval_length: int = 5, start_hour: int = 19, end_hour: int = 2, number_of_requests_in_context: int = 4, # consider 4 or 8
                 number_of_testing_days: int  = 25):
        self.weather_fields = [ 'temperature_2m', 'precipitation', 'windspeed_10m']
        self.time_fields = ['year', 'month', 'day', 'weekday', 'hour', 'interval_index']
        self.request_index_fields = ['request_index']
        self.location_target_fields = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        self.num_requests_target_fields = ['number_requests']
        self.weather_output_file = weather_output_file
        self.interval_length = interval_length
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.number_of_requests_in_context = number_of_requests_in_context
        self.number_of_testing_days = number_of_testing_days
        self._check_and_create_folder(requests_locations_folder)
        self.requests_locations_folder = requests_locations_folder
        self._check_and_create_folder(requests_number_folder)
        self.requests_number_folder = requests_number_folder
    
    def _check_and_create_folder(self, folder_path: str):
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

class Date_operational_range:
    def __init__(self, year: int, month: int, day: int, start_hour: int, end_hour: int) -> None:
        self.year: int = year
        self.month: int = month
        self.day: int = day
        self.start_hour: int = start_hour
        self.end_hour: int = end_hour

    def __repr__(self) -> str:
        operational_range_str = ""
        return operational_range_str

    def __hash__(self) -> int:
        return hash(tuple((self.year, self.month, self.day, self.start_hour, self.end_hour)))
    
class Requests_info:
    def __init__(self, requests_df, start_hour: int) -> None:
        self.requests_pickup_times = {}
        self.request_capacities = {}
        for index, row in requests_df.iterrows():
            if row["Requested Pickup Time"].hour < start_hour:
                self.requests_pickup_times[index] = (((((row["Requested Pickup Time"].hour + 24) - start_hour) * 60) \
                                                    + row["Requested Pickup Time"].minute) * 60) + row["Requested Pickup Time"].second
            else:
                self.requests_pickup_times[index] = ((((row["Requested Pickup Time"].hour - start_hour) * 60) \
                                                    + row["Requested Pickup Time"].minute) * 60) + row["Requested Pickup Time"].second
            self.request_capacities[index] = row["Number of Passengers"]

    def __repr__(self) -> str:
        pass

    def __hash__(self) -> int:
        return hash(tuple((frozendict(self.requests_pickup_times), frozendict(self.request_capacities))))

class Dataframe_row:
    def __init__(self, data) -> None:
        self.data = data

    def __hash__(self) -> int:
        return hash(tuple(self.data))

class Bus_stop_request_pairings:
    def __init__(self, stops_request_pairing: list[dict[str, list[int]]]) -> None:
        self.data: list[dict[str, list[int]]] = stops_request_pairing

    def __repr__(self) -> str:
        out = ''
        for stop in self.data:
            pick = '{p' + str([i for i in stop["pickup"]]) + ' '
            drop = 'd' + str([i for i in stop["dropoff"]]) + '}, '
            out += pick+drop
        return out 

    def __hash__(self) -> int:
        content_list = []
        for entry in self.data:
            dict_tuple = tuple(sorted((k, tuple(v)) for k, v in entry.items()))
            content_list.append(dict_tuple)
        return hash(tuple(content_list))
    
    def __iter__(self):
        return iter(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        if index < len(self.data):
            return self.data[index]
        else:
            raise IndexError
        
    def pop(self, index: int):
        return self.data.pop(index)
    

class Routing_plan:

    def __init__(self, bus_stops: list[int], stops_wait_times: list[int], 
                 stops_request_pairing: Bus_stop_request_pairings, 
                 assignment_cost: int, start_time: int, route: list[int], 
                 route_edge_times: list[int], route_stop_wait_time: list[int], newest_assignment_cost: int = 0) -> None:
        self.bus_stops = bus_stops
        self.stops_wait_times = stops_wait_times
        self.stops_request_pairing = stops_request_pairing
        self.assignment_cost = assignment_cost
        self.newest_assignment_cost = newest_assignment_cost
        self.start_time = start_time
        self.route = route
        self.route_edge_time = route_edge_times
        self.route_stop_wait_time = route_stop_wait_time
    
    def update_routes(self, route: list[int], route_edge_times: list[int], route_stop_wait_time: list[int]):
        self.route = route
        self.route_edge_time = route_edge_times
        self.route_stop_wait_time = route_stop_wait_time

    def __hash__(self) -> int:
        return hash(tuple((tuple(self.bus_stops), tuple(self.stops_wait_times), self.stops_request_pairing, \
                           self.assignment_cost, self.start_time, tuple(self.route), tuple(self.route_edge_time), \
                            tuple(self.route_stop_wait_time))))
    
    def __lt__(self, other: 'Routing_plan') -> bool:
        return self.assignment_cost < other.assignment_cost
    
    def __repr__(self) -> str:
        repr_str = ''
        repr_str += '--(' + str(self.stops_request_pairing) + str(self.assignment_cost) + ')-- \n'
        repr_str += "stop_wait_times = " + str(self.stops_wait_times) + "\n"
        repr_str += "start_time = " + str(self.start_time)
        return repr_str   
   
class Bus_fleet:

    def __init__(self, routing_plans: list[Routing_plan]) -> None:
        self.routing_plans = routing_plans

    def __hash__(self) -> int:
        return hash(tuple(self.routing_plans))
    
class Completed_requests_stats:

    def __init__(self, combined_requests_df, request_status_label = "Request Status", completed_field = "Completed",
                 ride_duration_label = "Ride Duration", number_of_passengers_label = "Number of Passengers", 
                 passenger_wait_time_label = "On-demand ETA"):
        completed_requests_df = combined_requests_df[combined_requests_df[request_status_label] == completed_field]
        number_of_requests = completed_requests_df[number_of_passengers_label].count()
        number_of_serviced_passengers = completed_requests_df[number_of_passengers_label].sum()

        avg_wait_time_at_station = completed_requests_df[passenger_wait_time_label].sum()/completed_requests_df[passenger_wait_time_label].count()
        avg_wait_time_at_station = avg_wait_time_at_station * 60

        avg_time_in_bus = completed_requests_df[ride_duration_label].sum()/completed_requests_df[ride_duration_label].count()
        avg_time_in_bus = avg_time_in_bus * 60

        self.number_of_serviced_requests = number_of_requests
        self.number_of_serviced_passengers = number_of_serviced_passengers
        self.passenger_wait_times = (completed_requests_df[passenger_wait_time_label]* 60).tolist()
        self.avg_wait_time_at_station = avg_wait_time_at_station
        self.avg_time_in_bus = avg_time_in_bus
    
    def __repr__(self) -> str:
        repr_str = "Number of requests serviced = " + str(self.number_of_serviced_requests) + "\n"
        repr_str += "Number of passengers serviced = " + str(self.number_of_serviced_passengers) + "\n"
        repr_str += "Average Passenger Wait Time at the origin station = " + str(self.avg_wait_time_at_station) + "\n"
        repr_str += "Average Passenger Trip Time = " + str(self.avg_time_in_bus) + "\n"
        repr_str += "Passenger Wait Times at the station = " + str(self.passenger_wait_times) + "\n"
        repr_str + "\n"
        
        return repr_str
    
class Failed_requests_stats:

    def __init__(self, combined_requests_df, request_status_label = "Request Status", unaccepted_proposal_field = "Unaccepted Proposal",
                 number_of_passengers_label = "Number of Passengers", passenger_wait_time_label = "On-demand ETA",
                 cancel_field = "Cancel") -> None:
        
        unaccepted_requests_df = combined_requests_df[combined_requests_df[request_status_label] == unaccepted_proposal_field]
        number_of_unaccepted_requests = unaccepted_requests_df[number_of_passengers_label].count()
        number_of_unaccepted_passengers = unaccepted_requests_df[number_of_passengers_label].sum()
        if unaccepted_requests_df[passenger_wait_time_label].count() == 0:
            print('No unaccepted requests')
            avg_wait_time_for_unaccepted_requests = float('inf')
        else:
            avg_wait_time_for_unaccepted_requests = unaccepted_requests_df[passenger_wait_time_label].sum()/unaccepted_requests_df[passenger_wait_time_label].count()
        avg_wait_time_for_unaccepted_requests = avg_wait_time_for_unaccepted_requests * 60

        fleet_canceled_requests_df = combined_requests_df[(combined_requests_df[request_status_label]) == cancel_field]
        number_of_fleet_cancel_requests = fleet_canceled_requests_df[number_of_passengers_label].count()
        number_of_fleet_cancel_passengers = fleet_canceled_requests_df[number_of_passengers_label].sum()
        if fleet_canceled_requests_df[passenger_wait_time_label].count() == 0:
            print('No fleet canceled requests')
            avg_wait_time_for_fleet_canceled_requests = float('inf')
        else:
            avg_wait_time_for_fleet_canceled_requests = fleet_canceled_requests_df[passenger_wait_time_label].sum()/fleet_canceled_requests_df[passenger_wait_time_label].count()
        avg_wait_time_for_fleet_canceled_requests = avg_wait_time_for_fleet_canceled_requests * 60

        canceled_requests_df = combined_requests_df[(combined_requests_df[request_status_label]).isin([unaccepted_proposal_field, cancel_field])]
        number_of_canceled_requests = canceled_requests_df[number_of_passengers_label].count()
        number_of_canceled_passengers = canceled_requests_df[number_of_passengers_label].sum()
        if canceled_requests_df[passenger_wait_time_label].count() == 0:
            print('No canceled requests')
            avg_wait_time_for_canceled_requests = float('inf')
        else:
            avg_wait_time_for_canceled_requests = canceled_requests_df[passenger_wait_time_label].sum()/canceled_requests_df[passenger_wait_time_label].count()
        avg_wait_time_for_canceled_requests = avg_wait_time_for_canceled_requests * 60
        wait_times_for_canceled_requests = (canceled_requests_df[passenger_wait_time_label] * 60).to_list()

        self.number_of_unaccepted_requests = number_of_unaccepted_requests
        self.number_of_unaccepted_passengers = number_of_unaccepted_passengers
        self.avg_wait_time_for_unaccepted_requests = avg_wait_time_for_unaccepted_requests

        self.number_of_fleet_cancel_requests = number_of_fleet_cancel_requests
        self.number_of_fleet_cancel_passengers = number_of_fleet_cancel_passengers
        self.avg_wait_time_for_canceled_requests = avg_wait_time_for_fleet_canceled_requests

        self.total_number_of_failed_requests = number_of_canceled_requests
        self.total_number_of_failed_passengers = number_of_canceled_passengers
        self.avg_wait_time_for_failed_requests = avg_wait_time_for_canceled_requests
        self.wait_times_for_canceled_requests = wait_times_for_canceled_requests
    
    def __repr__(self) -> str:
        repr_str = "Number of unaccepted  requests = " + str(self.number_of_unaccepted_requests) + "\n"
        repr_str += "Number of unaccepted passengers = " + str(self.number_of_unaccepted_passengers) + "\n"
        repr_str += "Average Wait Time for unaccepted requests = " + str(self.avg_wait_time_for_unaccepted_requests) + "\n"
        repr_str + "\n"
        repr_str +="Number of fleet canceled requests = " + str(self.number_of_fleet_cancel_requests) + "\n"
        repr_str += "Number of fleet canceled passengers = " + str(self.number_of_fleet_cancel_passengers) + "\n"
        repr_str += "Average Wait Time for fleet canceled requests = " + str(self.avg_wait_time_for_canceled_requests) + "\n"
        repr_str + "\n"
        repr_str +="Total number of canceled requests = " + str(self.total_number_of_failed_requests) + "\n"
        repr_str += "Total number of canceled passengers = " + str(self.total_number_of_failed_passengers) + "\n"
        repr_str += "Average Wait Time for canceled requests = " + str(self.avg_wait_time_for_failed_requests) + "\n"
        repr_str += "Wait Times at the station for canceled requests = " + str(self.wait_times_for_canceled_requests) + "\n"
        repr_str + "\n"

        return repr_str

