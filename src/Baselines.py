"""Policies.py

This module has the classes that define all routing baselines that we compare in the paper.

"""
import os
import copy
import pandas as pd
from functools import partial
from collections import Counter

from MCTS_state import MCTS_state
from benchmark_1.MCTS import MCForest
from benchmark_1.VV_Graph import VVEdge
from benchmark_1.new_DS import SimRequest

from State import State
from Plot_utils import Plot_utils
from Requests_prediction_handler import Request_Prediction_Handler
from Insertion_procedure import Request_Insertion_Procedure, Request_Insertion_Procedure_MCTS
from Data_structures import Bus_stop_request_pairings, Config_flags, Data_folders, Dataframe_row 
from Data_structures import Requests_info, Simulator_config, Date_operational_range, Bus_fleet, Routing_plan


class Greedy_static_insertion:
    
    def __init__(self, map_graph, data_folders: Data_folders, simulator_config: Simulator_config, perfect_accuracy: bool = True):
        
        self.map_graph = map_graph
        self.initial_bus_locations = simulator_config.initial_bus_locations
        self.bus_capacities = simulator_config.bus_capacities
        self.num_buses = simulator_config.num_buses
        new_results_folder = os.path.join(data_folders.static_results_folder, "greedy")
        if not os.path.isdir(new_results_folder):
            os.mkdir(new_results_folder)
        self.results_folder = new_results_folder
        self.total_cost = 0
        self.req_pred_handler = Request_Prediction_Handler(data_folders=data_folders,
                                                           perfect_accuracy=perfect_accuracy)
        self.req_insert = Request_Insertion_Procedure(map_graph=map_graph)
        self.plot_utils = Plot_utils(num_buses=simulator_config.num_buses)
        self.request_assignment = []
        self.current_assignment = []
        self.rejected_requests = []
        self.rejection_penalty = simulator_config.rejection_penalty
        routing_plans = self._initialize_buses()
        self.bus_fleet = Bus_fleet(routing_plans=routing_plans)

    def _initialize_buses(self) -> list[Routing_plan]:
        routing_plans = []
        for bus_index in range(self.num_buses):
            self.request_assignment.append({})
            self.current_assignment.append([])
            initial_stops = [self.initial_bus_locations[bus_index], self.initial_bus_locations[bus_index]]
            initial_wait_times = [0, 0]
            initial_stops_request_pairing_list = [{"pickup": [-1], "dropoff": [-1]}, {"pickup": [-1], "dropoff": [-1]}]
            initial_stops_request_pairing = Bus_stop_request_pairings(stops_request_pairing=initial_stops_request_pairing_list)
            initial_assignment_cost = 0
            initial_start_time = 0
            initial_route = [self.initial_bus_locations[bus_index], self.initial_bus_locations[bus_index]]
            initial_route_edge_time = [0]
            initial_route_stop_wait_time = [0, 0]

            bus_routing_plan = Routing_plan(bus_stops=initial_stops,
                                            stops_wait_times=initial_wait_times,
                                            stops_request_pairing=initial_stops_request_pairing,
                                            assignment_cost=initial_assignment_cost,
                                            start_time=initial_start_time,
                                            route=initial_route,
                                            route_edge_times=initial_route_edge_time,
                                            route_stop_wait_time=initial_route_stop_wait_time)
            
            routing_plans.append(bus_routing_plan)
        
        return routing_plans
    
    def retrieve_route_info_for_bus(self, bus_index) -> Routing_plan:
        return self.bus_fleet.routing_plans[bus_index]
    
    def retrieve_all_info(self):
        return self.bus_fleet, self.request_assignment
    
    def _get_static_insertion_cost_for_single_request(self, bus_index: int, request_index: int, request_row: Dataframe_row, 
                                                      requests_info: Requests_info, local_routing_plan: Routing_plan, 
                                                      config_flags: Config_flags):
        request_origin = request_row.data["Origin Node"]
        request_destination = request_row.data["Destination Node"]
        
        insertion_result = self.req_insert.static_insertion(current_start_time=local_routing_plan.start_time,
                                                            bus_capacity=self.bus_capacities[bus_index],
                                                            stops_sequence=local_routing_plan.bus_stops, 
                                                            stops_wait_time=local_routing_plan.stops_wait_times, 
                                                            stop_request_pairing=local_routing_plan.stops_request_pairing.data,
                                                            request_capacities=requests_info.request_capacities, 
                                                            request_origin=request_origin, 
                                                            request_destination=request_destination,
                                                            request_index=request_index,
                                                            requests_pickup_times=requests_info.requests_pickup_times,
                                                            consider_route_time=config_flags.consider_route_time,
                                                            include_scaling=config_flags.include_scaling)
            
        total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair, min_start_time = insertion_result

        if total_dev_cost == float("inf"):
            new_routing_plan = None
        else:
            new_assignment_cost = local_routing_plan.assignment_cost + total_dev_cost
            new_stop_req_pairings = Bus_stop_request_pairings(full_stop_req_pair)
            new_routing_plan = Routing_plan(bus_stops=full_stop_sequence,
                                            stops_wait_times=full_stops_wait_time,
                                            stops_request_pairing=new_stop_req_pairings,
                                            assignment_cost=new_assignment_cost,
                                            start_time=min_start_time,
                                            route=[],
                                            route_edge_times=[],
                                            route_stop_wait_time=[])

        return total_dev_cost, new_routing_plan
    
    def _insert_requests(self, requests_df, requests_info: Requests_info, config_flags: Config_flags):

        for index, row, in requests_df.iterrows():
            min_assignment_cost = float("inf")
            min_bus_index = 0
            min_routing_plan = None
            for bus_index in range(self.num_buses):
                local_routing_plan = self.bus_fleet.routing_plans[bus_index]
                request_row = Dataframe_row(data=row)
                total_dev_cost, new_routing_plan = self._get_static_insertion_cost_for_single_request(bus_index=bus_index, 
                                                                                        request_index=index,
                                                                                        request_row=request_row,
                                                                                        requests_info=requests_info,
                                                                                        local_routing_plan=local_routing_plan,
                                                                                        config_flags=config_flags)
                
                if total_dev_cost < min_assignment_cost:
                    min_assignment_cost = total_dev_cost
                    min_bus_index = bus_index
                    min_routing_plan = new_routing_plan
            
            if min_assignment_cost == float("inf"):
                print("Request " + str(index) + " can't be accomodated.")
                self.rejected_requests.append(index)
            else:
                if config_flags.verbose:
                    print("Cost for assigning request " + str(index) + " to bus " + str(min_bus_index) + " = " + str(min_assignment_cost))
                
                self.request_assignment[min_bus_index][index] = row
                self.current_assignment[min_bus_index] = [index]

                if config_flags.plot_initial_routes:
                    prev_bus_stops = []
                    prev_bus_routes = []
                    for bus_index in range(self.num_buses):
                        current_bus_stops = self.bus_fleet.routing_plans[bus_index].bus_stops
                        current_routes = self.bus_fleet.routing_plans[bus_index].route
                        prev_bus_routes.append(current_routes)
                        prev_bus_stops.append(current_bus_stops)
                    self.plot_utils.plot_routes_before_assignment_offline(map_object=self.map_graph, 
                                                                        current_assignment=self.current_assignment, 
                                                                        request_assignment=self.request_assignment, 
                                                                        prev_bus_stops=prev_bus_stops,
                                                                        prev_bus_routes=prev_bus_routes, 
                                                                        bus_locations=self.initial_bus_locations,
                                                                        folder_path=self.results_folder)
                
                self.bus_fleet.routing_plans[min_bus_index] = min_routing_plan
                self._generate_routes_from_stops()

                if config_flags.plot_initial_routes:
                    current_bus_stops_list = []
                    current_bus_routes_list = []
                    for bus_index in range(self.num_buses):
                        current_bus_stops = self.bus_fleet.routing_plans[bus_index].bus_stops
                        current_routes = self.bus_fleet.routing_plans[bus_index].route
                        current_bus_routes_list.append(current_routes)
                        current_bus_stops_list.append(current_bus_stops)
                    self.plot_utils.plot_routes_after_assignment_offline(map_object=self.map_graph, 
                                                                        outstanding_requests={}, 
                                                                        current_bus_stops=current_bus_stops_list,
                                                                        current_bus_routes=current_bus_routes_list, 
                                                                        bus_locations=self.initial_bus_locations,
                                                                        folder_path=self.results_folder)
    
    def _generate_routes_from_stops(self):
        for bus_index in range(self.num_buses):
            bus_route = []
            bus_route_edge_time = []
            routes_stop_wait_time= []
            current_bus_stops = self.bus_fleet.routing_plans[bus_index].bus_stops
            current_stops_wait_time = self.bus_fleet.routing_plans[bus_index].stops_wait_times
            for bus_stop_index in range(len(current_bus_stops)-1):
                origin_stop = current_bus_stops[bus_stop_index]
                destination_bus_stop = current_bus_stops[bus_stop_index+1]
                origin_wait_time = current_stops_wait_time[bus_stop_index]

                shortest_path = self.map_graph.shortest_paths[origin_stop, destination_bus_stop]
                shortest_path_wait_time = []
                stops_wait_time = []
                for node_index in range(len(shortest_path)-1):
                    edge_origin = shortest_path[node_index]
                    edge_destination = shortest_path[node_index + 1]
                    edge_time = self.map_graph.obtain_shortest_paths_time(edge_origin, edge_destination)
                    shortest_path_wait_time.append(edge_time)
                    if node_index == 0:
                        stops_wait_time.append(origin_wait_time)
                    else:
                        stops_wait_time.append(0)

                if len(shortest_path_wait_time) == 0:
                    shortest_path_wait_time.append(0)
                    stops_wait_time.append(origin_wait_time)
                    bus_route += shortest_path
                else:
                    bus_route += shortest_path[:-1]
                bus_route_edge_time += shortest_path_wait_time
                routes_stop_wait_time += stops_wait_time
            

            bus_route += [current_bus_stops[-1]]
            routes_stop_wait_time += [current_stops_wait_time[-1]]
            self.bus_fleet.routing_plans[bus_index].update_routes(route=bus_route, 
                                                                  route_edge_times=bus_route_edge_time,
                                                                  route_stop_wait_time=routes_stop_wait_time)
    
    def _extract_requests(self, date_operational_range: Date_operational_range):
        scheduled_requests_df, online_requests_df = self.req_pred_handler.\
            get_requests_for_given_date_and_hour_range(date_operational_range=date_operational_range)
        
        combined_requests_df = pd.concat([scheduled_requests_df, online_requests_df])

        return combined_requests_df
        
        
    def assign_requests_and_create_routes(self, date_operational_range: Date_operational_range, config_flags: Config_flags):
        if config_flags.plot_initial_routes:
            self.plot_utils.reset_frame_number()

        requests_to_be_serviced = self._extract_requests(date_operational_range=date_operational_range)

        requests_info = Requests_info(requests_df=requests_to_be_serviced, start_hour=date_operational_range.start_hour)

        self._insert_requests(requests_df=requests_to_be_serviced,
                              requests_info=requests_info,
                              config_flags=config_flags)

        assignment_costs = []

        for routing_plan in self.bus_fleet.routing_plans:
            current_assignment_cost = routing_plan.assignment_cost
            assignment_costs.append(current_assignment_cost)

        self.total_cost = sum(assignment_costs)

        return self.total_cost, assignment_costs, requests_info, self.rejected_requests


class Greedy_dynamic_insertion:

    def __init__(self, map_graph, data_folders: Data_folders, simulator_config: Simulator_config, config_flags: Config_flags, perfect_requests = None):
        self.map_graph = map_graph
        self.initial_bus_locations = simulator_config.initial_bus_locations
        self.num_buses = simulator_config.num_buses
        self.results_folder = data_folders.dynamic_results_folder
        self.req_insert = Request_Insertion_Procedure(map_graph=map_graph)
        self.plot_utils = Plot_utils(num_buses=self.num_buses)
        self.include_scaling = config_flags.include_scaling

    
    def _dropoff_prev_passengers(self, state_object: State, bus_index: int, passengers_in_bus, prev_passengers):
        new_passengers_in_bus = copy.deepcopy(passengers_in_bus)
        new_prev_passengers = copy.deepcopy(prev_passengers)

        next_bus_stop_index = state_object.bus_stop_index[bus_index] + 1

        dropoff_request_index_list = state_object.bus_fleet.routing_plans[bus_index].stops_request_pairing.data[next_bus_stop_index]["dropoff"]
        for dropoff_request_index in dropoff_request_index_list:
            if dropoff_request_index >= 0:
                if dropoff_request_index in new_prev_passengers:
                    del(new_prev_passengers[dropoff_request_index])
                    new_passengers_in_bus -= state_object.request_capacities[dropoff_request_index]
        
        return new_passengers_in_bus, new_prev_passengers
    
    def _pickup_prev_passengers(self, state_object: State, bus_index: int, current_start_time: int, passengers_in_bus, prev_passengers):
        new_passengers_in_bus = copy.deepcopy(passengers_in_bus)
        new_prev_passengers = copy.deepcopy(prev_passengers)

        current_bus_stop_index = state_object.bus_stop_index[bus_index]

        pickup_request_index_list = state_object.bus_fleet.routing_plans[bus_index].stops_request_pairing.data[current_bus_stop_index]["pickup"]
        for pickup_request_index in pickup_request_index_list:
            if pickup_request_index >= 0:
                if pickup_request_index not in new_prev_passengers:
                    new_prev_passengers[pickup_request_index] = [state_object.bus_fleet.routing_plans[bus_index].bus_stops[current_bus_stop_index], current_start_time]
                    new_passengers_in_bus += state_object.request_capacities[pickup_request_index]
        
        return new_passengers_in_bus, new_prev_passengers

    def _get_bus_parameters_of_interest(self, state_object: State, bus_index: int):
        current_step_index = state_object.step_index[bus_index]
        next_bus_location = state_object.bus_fleet.routing_plans[bus_index].route[current_step_index+1]
        current_bus_location = state_object.bus_locations[bus_index]

        current_stop_wait_time = state_object.bus_fleet.routing_plans[bus_index].route_stop_wait_time[current_step_index]
        current_edge_time = state_object.bus_fleet.routing_plans[bus_index].route_edge_time[current_step_index]

        current_bus_stop_index = state_object.bus_stop_index[bus_index]
        next_bus_stop = state_object.bus_fleet.routing_plans[bus_index].bus_stops[current_bus_stop_index+1]
        current_bus_stop = state_object.bus_fleet.routing_plans[bus_index].bus_stops[current_bus_stop_index]

        if len(state_object.bus_fleet.routing_plans[bus_index].bus_stops) == 2 and \
            state_object.bus_fleet.routing_plans[bus_index].bus_stops[current_bus_stop_index] == state_object.bus_fleet.routing_plans[bus_index].bus_stops[current_bus_stop_index+1]:
            new_bus_location = current_bus_location
            current_start_time = state_object.bus_fleet.routing_plans[bus_index].start_time
            current_stop_index = current_bus_stop_index
            passengers_in_bus = state_object.passengers_in_bus[bus_index]
            prev_passengers = state_object.prev_passengers[bus_index]
        else:
            if current_bus_location == current_bus_stop:
                if state_object.wait_time_at_the_station[bus_index] < current_stop_wait_time:
                    new_bus_location = current_bus_location
                    current_start_time = state_object.bus_fleet.routing_plans[bus_index].start_time
                    current_stop_index = current_bus_stop_index
                    passengers_in_bus = state_object.passengers_in_bus[bus_index]
                    prev_passengers = state_object.prev_passengers[bus_index]

                else:
                    new_bus_location = next_bus_location
                    current_start_time = state_object.bus_fleet.routing_plans[bus_index].start_time + current_stop_wait_time
                    new_passengers_in_bus, new_prev_passengers = self._pickup_prev_passengers(state_object=state_object, 
                                                                                                bus_index=bus_index,
                                                                                                current_start_time=current_start_time,
                                                                                                passengers_in_bus=state_object.passengers_in_bus[bus_index],
                                                                                                prev_passengers=state_object.prev_passengers[bus_index])
                    current_start_time += current_edge_time
                    if next_bus_location == next_bus_stop:
                        current_stop_index = current_bus_stop_index + 1
                        new_passengers_in_bus, new_prev_passengers = self._dropoff_prev_passengers(state_object=state_object, 
                                                                                                bus_index=bus_index,
                                                                                                passengers_in_bus=new_passengers_in_bus,
                                                                                                prev_passengers=new_prev_passengers)
                        passengers_in_bus = new_passengers_in_bus
                        prev_passengers = new_prev_passengers
                    else:
                        current_stop_index = current_bus_stop_index
                        passengers_in_bus = new_passengers_in_bus
                        prev_passengers = new_prev_passengers
            else:
                new_bus_location = next_bus_location
                current_start_time = state_object.bus_fleet.routing_plans[bus_index].start_time + current_edge_time
                if next_bus_location == next_bus_stop:
                    current_stop_index = current_bus_stop_index + 1
                    new_passengers_in_bus, new_prev_passengers = self._dropoff_prev_passengers(state_object=state_object, 
                                                                                            bus_index=bus_index,
                                                                                            passengers_in_bus=state_object.passengers_in_bus[bus_index],
                                                                                            prev_passengers=state_object.prev_passengers[bus_index])
                    passengers_in_bus = new_passengers_in_bus
                    prev_passengers = new_prev_passengers
                else:
                    current_stop_index = current_bus_stop_index
                    passengers_in_bus = state_object.passengers_in_bus[bus_index]
                    prev_passengers = state_object.prev_passengers[bus_index]

        return current_start_time, current_stop_index, new_bus_location, passengers_in_bus, prev_passengers

    def _get_dynamic_insertion_cost_for_request(self, state_object: State, bus_index: int, request_index: int, request_row: pd.Series, 
                                                config_flags: Config_flags):
        request_origin = request_row["Origin Node"]
        request_destination = request_row["Destination Node"]

        bus_parameters = self._get_bus_parameters_of_interest(state_object=state_object, bus_index=bus_index)

        current_start_time, current_stop_index, current_location, passengers_in_bus, prev_passengers = bus_parameters

        insertion_result = self.req_insert.dynamic_insertion(current_start_time=current_start_time,
                                                             current_stop_index=current_stop_index,
                                                             bus_location=current_location,
                                                             bus_capacity=state_object.bus_capacities[bus_index],
                                                             passengers_in_bus=passengers_in_bus,
                                                             prev_passengers=prev_passengers,
                                                             stops_sequence=state_object.bus_fleet.routing_plans[bus_index].bus_stops, 
                                                            stops_wait_time=state_object.bus_fleet.routing_plans[bus_index].stops_wait_times, 
                                                            stop_request_pairing=state_object.bus_fleet.routing_plans[bus_index].stops_request_pairing.data,
                                                            request_capacities=state_object.request_capacities,
                                                            request_origin=request_origin, 
                                                            request_destination=request_destination, 
                                                            requests_pickup_times=state_object.requests_pickup_times,
                                                            request_index=request_index,
                                                            consider_route_time=config_flags.consider_route_time,
                                                            include_scaling=config_flags.include_scaling,
                                                            cost_type=config_flags.cost_type)
        
        total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair = insertion_result

        if total_dev_cost == float("inf"):
            new_routing_plan = None
        else:
            new_assignment_cost = state_object.bus_fleet.routing_plans[bus_index].assignment_cost + total_dev_cost
            new_stop_req_pairings = Bus_stop_request_pairings(full_stop_req_pair)
            new_routing_plan = Routing_plan(bus_stops=full_stop_sequence,
                                            stops_wait_times=full_stops_wait_time,
                                            stops_request_pairing=new_stop_req_pairings,
                                            assignment_cost=new_assignment_cost,
                                            start_time=state_object.bus_fleet.routing_plans[bus_index].start_time,
                                            route=[],
                                            route_edge_times=[],
                                            route_stop_wait_time=[])

        
        return total_dev_cost, new_routing_plan

    def _determine_assignment(self, state_object, current_request_index, current_request_row, config_flags: Config_flags):
        min_assignment_cost = float("inf")
        min_bus_index = 0
        min_routing_plan = None
        for bus_index in range(self.num_buses):
            total_dev_cost, new_routing_plan = self._get_dynamic_insertion_cost_for_request(state_object=state_object, 
                                                                                            bus_index=bus_index,
                                                                                            request_index=current_request_index,
                                                                                            request_row=current_request_row,
                                                                                            config_flags=config_flags)

            if total_dev_cost < min_assignment_cost:
                min_assignment_cost = total_dev_cost
                min_bus_index = bus_index
                min_routing_plan = new_routing_plan
        
        return min_bus_index, min_assignment_cost, min_routing_plan
    
    def _generate_route_from_stops(self, current_bus_location, bus_stops, stops_wait_times):
        bus_route = []
        bus_route_edge_time = []
        routes_stop_wait_time= []
        for bus_stop_index in range(len(bus_stops)-1):
            if bus_stop_index == 0:
                origin_stop = current_bus_location
                if bus_stops[bus_stop_index] == current_bus_location:
                    origin_wait_time = stops_wait_times[bus_stop_index]
                else:
                    if bus_stops[bus_stop_index+1] == current_bus_location:
                        continue
                    origin_wait_time = 0
            else:
                origin_stop = bus_stops[bus_stop_index]
                origin_wait_time = stops_wait_times[bus_stop_index]
            destination_bus_stop = bus_stops[bus_stop_index+1]

            shortest_path = self.map_graph.shortest_paths[origin_stop, destination_bus_stop]
            shortest_path_wait_time = []
            stops_wait_time = []
            for node_index in range(len(shortest_path)-1):
                edge_origin = shortest_path[node_index]
                edge_destination = shortest_path[node_index + 1]
                edge_time = self.map_graph.obtain_shortest_paths_time(edge_origin, edge_destination)
                shortest_path_wait_time.append(edge_time)
                if node_index == 0:
                    stops_wait_time.append(origin_wait_time)
                else:
                    stops_wait_time.append(0)

            if len(shortest_path_wait_time) == 0:
                shortest_path_wait_time.append(0)
                stops_wait_time.append(origin_wait_time)
                bus_route += shortest_path
            else:
                bus_route += shortest_path[:-1]
            
            bus_route_edge_time += shortest_path_wait_time
            routes_stop_wait_time += stops_wait_time
        
        bus_route += [bus_stops[-1]]
        routes_stop_wait_time += [stops_wait_times[-1]]
        
        return bus_route, bus_route_edge_time, routes_stop_wait_time

    def assign_requests_and_create_routes(self, state_object: State, requests, config_flags: Config_flags):

        if config_flags.plot_final_routes:
            prev_bus_stops = []
            prev_bus_routes = []
            for bus_index in range(state_object.num_buses):
                current_bus_stops = state_object.bus_fleet.routing_plans[bus_index].bus_stops[state_object.bus_stop_index[bus_index]:]
                current_routes = state_object.bus_fleet.routing_plans[bus_index].route[state_object.step_index[bus_index]:]
                prev_bus_routes.append(current_routes)
                prev_bus_stops.append(current_bus_stops)

            self.plot_utils.plot_routes_before_assignment_online(map_object=self.map_graph, 
                                                                 requests=requests, 
                                                                 prev_bus_stops=prev_bus_stops,
                                                                 prev_bus_routes=prev_bus_routes, 
                                                                 bus_locations=state_object.bus_locations,
                                                                 folder_path=self.results_folder)
        
        for request_index, request_row, in requests.iterrows():
            if request_row["Requested Pickup Time"].hour < state_object.date_operational_range.start_hour:
                state_object.requests_pickup_times[request_index] = (((((request_row["Requested Pickup Time"].hour + 24) - state_object.date_operational_range.start_hour) * 60) \
                                                                    + request_row["Requested Pickup Time"].minute) * 60) + request_row["Requested Pickup Time"].second
            else:
                state_object.requests_pickup_times[request_index] = ((((request_row["Requested Pickup Time"].hour - state_object.date_operational_range.start_hour) * 60) \
                                                                    + request_row["Requested Pickup Time"].minute) * 60) + request_row["Requested Pickup Time"].second
            state_object.request_capacities[request_index] = request_row["Number of Passengers"]

        for request_index, request_row, in requests.iterrows():
            assignment_result = self._determine_assignment(state_object=state_object,
                                                           current_request_index=request_index,
                                                           current_request_row=request_row, 
                                                           config_flags=config_flags)
            bus_index, assignment_cost, new_routing_plan = assignment_result

            if assignment_cost != float("inf"):
                current_step_index = state_object.step_index[bus_index]
                current_stop_wait_time = state_object.bus_fleet.routing_plans[bus_index].route_stop_wait_time[current_step_index]
                if state_object.wait_time_at_the_station[bus_index] < current_stop_wait_time:
                    current_bus_location=state_object.bus_fleet.routing_plans[bus_index].route[state_object.step_index[bus_index]]
                else:
                    current_bus_location=state_object.bus_fleet.routing_plans[bus_index].route[state_object.step_index[bus_index]+1]

                route, bus_route_edge_time, routes_stop_wait_time = self._generate_route_from_stops(current_bus_location=current_bus_location, 
                                                                                                    bus_stops=new_routing_plan.bus_stops, 
                                                                                                    stops_wait_times=new_routing_plan.stops_wait_times)
                
                new_routing_plan.update_routes(route=route,
                                            route_edge_times=bus_route_edge_time,
                                            route_stop_wait_time=routes_stop_wait_time)
                
                state_object.update_state(bus_index, 
                                        request_index=request_index,
                                        request_row=request_row,
                                        assignment_cost=assignment_cost,
                                        new_routing_plan=new_routing_plan)
            else:
                state_object.update_state(bus_index,
                                          request_index=request_index,
                                          request_row=request_row,
                                          assignment_cost=0,
                                          new_routing_plan=None)
        
        if config_flags.plot_final_routes and not requests.empty:
                current_bus_stops_list = []
                current_bus_routes_list = []
                for bus_index in range(state_object.num_buses):
                    current_bus_stops = state_object.bus_fleet.routing_plans[bus_index].bus_stops[state_object.bus_stop_index[bus_index]:]
                    current_routes = state_object.bus_fleet.routing_plans[bus_index].route[state_object.step_index[bus_index]:]
                    current_bus_routes_list.append(current_routes)
                    current_bus_stops_list.append(current_bus_stops)
                self.plot_utils.plot_routes_after_assignment_online(map_object=self.map_graph, 
                                                                    outstanding_requests={}, 
                                                                    current_bus_stops=current_bus_stops_list,
                                                                    current_bus_routes=current_bus_routes_list, 
                                                                    bus_locations=state_object.bus_locations,
                                                                    folder_path=self.results_folder)


class MCTS:
    def __init__(self, 
                 map_graph, 
                 data_folders: Data_folders, 
                 simulator_config: Simulator_config, 
                 config_flags: Config_flags) -> None:
        self.map_graph = map_graph
        self.initial_bus_locations = simulator_config.initial_bus_locations
        self.num_buses = simulator_config.num_buses
        self.data_folders = data_folders
        self.results_folder = data_folders.dynamic_results_folder
        self.req_insert = Request_Insertion_Procedure_MCTS(map_graph=map_graph, 
                                                           MAX_WAIT_TIME_AT_STATION = config_flags.MAX_WAIT_TIME_AT_STATION, 
                                                           MAX_WAIT_TIME_INSIDE_BUS = config_flags.MAX_WAIT_TIME_INSIDE_BUS)
        self.plot_utils = Plot_utils(num_buses=self.num_buses)
        self.include_scaling = config_flags.include_scaling
    
    def _dropoff_prev_passengers(self, state_object: MCTS_state, bus_index: int, passengers_in_bus, prev_passengers):
        new_passengers_in_bus = copy.deepcopy(passengers_in_bus)
        new_prev_passengers = copy.deepcopy(prev_passengers)

        next_bus_stop_index = state_object.bus_stop_index[bus_index] + 1

        dropoff_request_index_list = state_object.bus_fleet.routing_plans[bus_index].stops_request_pairing.data[next_bus_stop_index]["dropoff"]
        for dropoff_request_index in dropoff_request_index_list:
            if dropoff_request_index >= 0:
                if dropoff_request_index in new_prev_passengers:
                    del(new_prev_passengers[dropoff_request_index])
                    new_passengers_in_bus -= state_object.request_capacities[dropoff_request_index]
        
        return new_passengers_in_bus, new_prev_passengers
    
    def _pickup_prev_passengers(self, state_object: MCTS_state, bus_index: int, current_start_time: int, passengers_in_bus, prev_passengers):
        new_passengers_in_bus = copy.deepcopy(passengers_in_bus)
        new_prev_passengers = copy.deepcopy(prev_passengers)

        current_bus_stop_index = state_object.bus_stop_index[bus_index]

        pickup_request_index_list = state_object.bus_fleet.routing_plans[bus_index].stops_request_pairing.data[current_bus_stop_index]["pickup"]
        for pickup_request_index in pickup_request_index_list:
            if pickup_request_index >= 0:
                if pickup_request_index not in new_prev_passengers:
                    new_prev_passengers[pickup_request_index] = [state_object.bus_fleet.routing_plans[bus_index].bus_stops[current_bus_stop_index], current_start_time]
                    new_passengers_in_bus += state_object.request_capacities[pickup_request_index]
        
        return new_passengers_in_bus, new_prev_passengers

    def _get_bus_parameters_of_interest(self, state_object: MCTS_state, bus_index: int):
        current_step_index = state_object.step_index[bus_index]
        next_bus_location = state_object.bus_fleet.routing_plans[bus_index].route[current_step_index+1]
        current_bus_location = state_object.bus_locations[bus_index]

        current_stop_wait_time = state_object.bus_fleet.routing_plans[bus_index].route_stop_wait_time[current_step_index]
        current_edge_time = state_object.bus_fleet.routing_plans[bus_index].route_edge_time[current_step_index]

        current_bus_stop_index = state_object.bus_stop_index[bus_index]
        next_bus_stop = state_object.bus_fleet.routing_plans[bus_index].bus_stops[current_bus_stop_index+1]
        current_bus_stop = state_object.bus_fleet.routing_plans[bus_index].bus_stops[current_bus_stop_index]

        if len(state_object.bus_fleet.routing_plans[bus_index].bus_stops) == 2 and \
            state_object.bus_fleet.routing_plans[bus_index].bus_stops[current_bus_stop_index] == state_object.bus_fleet.routing_plans[bus_index].bus_stops[current_bus_stop_index+1]:
            new_bus_location = current_bus_location
            current_start_time = state_object.bus_fleet.routing_plans[bus_index].start_time
            current_stop_index = current_bus_stop_index
            passengers_in_bus = state_object.passengers_in_bus[bus_index]
            prev_passengers = state_object.prev_passengers[bus_index]
        else:
            if current_bus_location == current_bus_stop:
                if state_object.wait_time_at_the_station[bus_index] < current_stop_wait_time:
                    new_bus_location = current_bus_location
                    current_start_time = state_object.bus_fleet.routing_plans[bus_index].start_time
                    current_stop_index = current_bus_stop_index
                    passengers_in_bus = state_object.passengers_in_bus[bus_index]
                    prev_passengers = state_object.prev_passengers[bus_index]

                else:
                    new_bus_location = next_bus_location
                    current_start_time = state_object.bus_fleet.routing_plans[bus_index].start_time + current_stop_wait_time
                    new_passengers_in_bus, new_prev_passengers = self._pickup_prev_passengers(state_object=state_object, 
                                                                                                bus_index=bus_index,
                                                                                                current_start_time=current_start_time,
                                                                                                passengers_in_bus=state_object.passengers_in_bus[bus_index],
                                                                                                prev_passengers=state_object.prev_passengers[bus_index])
                    current_start_time += current_edge_time
                    if next_bus_location == next_bus_stop:
                        current_stop_index = current_bus_stop_index + 1
                        new_passengers_in_bus, new_prev_passengers = self._dropoff_prev_passengers(state_object=state_object, 
                                                                                                bus_index=bus_index,
                                                                                                passengers_in_bus=new_passengers_in_bus,
                                                                                                prev_passengers=new_prev_passengers)
                        passengers_in_bus = new_passengers_in_bus
                        prev_passengers = new_prev_passengers
                    else:
                        current_stop_index = current_bus_stop_index
                        passengers_in_bus = new_passengers_in_bus
                        prev_passengers = new_prev_passengers
            else:
                new_bus_location = next_bus_location
                current_start_time = state_object.bus_fleet.routing_plans[bus_index].start_time + current_edge_time
                if next_bus_location == next_bus_stop:
                    current_stop_index = current_bus_stop_index + 1
                    new_passengers_in_bus, new_prev_passengers = self._dropoff_prev_passengers(state_object=state_object, 
                                                                                            bus_index=bus_index,
                                                                                            passengers_in_bus=state_object.passengers_in_bus[bus_index],
                                                                                            prev_passengers=state_object.prev_passengers[bus_index])
                    passengers_in_bus = new_passengers_in_bus
                    prev_passengers = new_prev_passengers
                else:
                    current_stop_index = current_bus_stop_index
                    passengers_in_bus = state_object.passengers_in_bus[bus_index]
                    prev_passengers = state_object.prev_passengers[bus_index]

        return current_start_time, current_stop_index, new_bus_location, passengers_in_bus, prev_passengers

    def _get_dynamic_insertion_cost_for_request(self, state_object: MCTS_state, bus_index: int, request_index: int, 
                                                request_row, config_flags: Config_flags, planning: bool):
        request_origin = request_row["Origin Node"]
        request_destination = request_row["Destination Node"]

        bus_parameters = self._get_bus_parameters_of_interest(state_object=state_object, bus_index=bus_index)

        current_start_time, current_stop_index, current_location, passengers_in_bus, prev_passengers = bus_parameters

        insertion_result = self.req_insert.dynamic_insertion(current_start_time=current_start_time,
                                                             current_stop_index=current_stop_index,
                                                             bus_location=current_location,
                                                             bus_capacity=state_object.bus_capacities[bus_index],
                                                             passengers_in_bus=passengers_in_bus,
                                                             prev_passengers=prev_passengers,
                                                            stops_sequence=state_object.bus_fleet.routing_plans[bus_index].bus_stops,
                                                            stops_wait_time=state_object.bus_fleet.routing_plans[bus_index].stops_wait_times,
                                                            stop_request_pairing=state_object.bus_fleet.routing_plans[bus_index].stops_request_pairing.data,
                                                            request_capacities=state_object.request_capacities,
                                                            request_origin=request_origin, 
                                                            request_destination=request_destination, 
                                                            requests_pickup_times=state_object.requests_pickup_times,
                                                            request_index=request_index,
                                                            include_scaling=config_flags.include_scaling,
                                                            bus_index = bus_index,
                                                            cost_type=config_flags.cost_type,
                                                            prev_assignment_cost=state_object.bus_fleet.routing_plans[bus_index].assignment_cost,
                                                            planning=planning)
                                                            
        
        total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair, rv_list = insertion_result

        new_stop_req_pairings = Bus_stop_request_pairings(full_stop_req_pair)
        if total_dev_cost != float("inf"):
            new_assignment_cost = state_object.bus_fleet.routing_plans[bus_index].assignment_cost + total_dev_cost
            new_routing_plan = Routing_plan(bus_stops=full_stop_sequence,
                                            stops_wait_times=full_stops_wait_time,
                                            stops_request_pairing=new_stop_req_pairings,
                                            assignment_cost=new_assignment_cost,
                                            newest_assignment_cost=total_dev_cost,
                                            start_time=state_object.bus_fleet.routing_plans[bus_index].start_time,
                                            route=[],
                                            route_edge_times=[],
                                            route_stop_wait_time=[])
        else:
            new_routing_plan = None
        
        return total_dev_cost, new_routing_plan, rv_list

    def _determine_assignment(self, state_object: MCTS_state, current_request_index: int, current_request_row, config_flags: Config_flags,
                              planning: bool):
        min_assignment_cost = float("inf")
        min_bus_index = 0
        min_routing_plan = None
        all_possible_rv_paths: list[tuple[int, Routing_plan]] = []

        for bus_index in range(config_flags.num_buses):
            total_dev_cost, new_routing_plan, rv_list = self._get_dynamic_insertion_cost_for_request(state_object=state_object, 
                                                                                            bus_index=bus_index,
                                                                                            request_index=current_request_index,
                                                                                            request_row=current_request_row,
                                                                                            config_flags=config_flags,
                                                                                            planning=planning)
            all_possible_rv_paths.extend(rv_list)

            if total_dev_cost < min_assignment_cost:
                min_assignment_cost = total_dev_cost
                min_bus_index = bus_index
                min_routing_plan = new_routing_plan
        
        return min_bus_index, min_assignment_cost, min_routing_plan, all_possible_rv_paths
    
    def _get_requests_that_can_be_unallocated(self, current_stop_index, stop_request_pairing):
        next_available_stop_index = current_stop_index + 1

        if next_available_stop_index < len(stop_request_pairing)-1:
            sliced_stop_request_pairings = stop_request_pairing[next_available_stop_index:]

            combine = [stop['pickup'] + stop['dropoff']
                    for stop in sliced_stop_request_pairings]
            flatten = [item for sublist in combine for item in sublist if item != -1]
            counts = Counter(flatten)
            repeated_values = [request for request,
                            count in counts.items() if count == 2]
        else:
            repeated_values = []

        return repeated_values
    
    def _get_unallocated_routing_plan(self, real_start_time, current_start_time, current_stop_index, bus_location, passengers_in_bus, prev_passengers,
                                      stops_sequence, stops_wait_time, stop_request_pairing, request_capacities, requests_pickup_times,
                                      request_index, cost_type, prev_assignment_cost, planning=False):
        out = self.req_insert.unallocate(current_start_time=current_start_time,
                                         current_stop_index=current_stop_index,
                                         bus_location=bus_location,
                                         passengers_in_bus=passengers_in_bus,
                                         prev_passengers=prev_passengers,
                                         stops_sequence=stops_sequence,
                                         stops_wait_time=stops_wait_time,
                                         stop_request_pairing=stop_request_pairing,
                                         request_capacities=request_capacities,
                                         requests_pickup_times=requests_pickup_times,
                                         request_index=request_index,
                                         cost_type=cost_type,
                                         planning=planning)
        
        if out:
            total_dev_cost = out[0]
            node_origin = out[1] 
            node_destination = out[2] 
            new_stops_sequence = out[3] 
            new_stops_wait_time = out[4]
            new_stop_request_pairing = out[5]

            new_assignment_cost = prev_assignment_cost + total_dev_cost
            new_stop_req_pairings = Bus_stop_request_pairings(new_stop_request_pairing)
            new_routing_plan = Routing_plan(bus_stops=new_stops_sequence,
                                            stops_wait_times=new_stops_wait_time,
                                            stops_request_pairing=new_stop_req_pairings,
                                            assignment_cost=new_assignment_cost,
                                            newest_assignment_cost=total_dev_cost,
                                            start_time=real_start_time,
                                            route=[],
                                            route_edge_times=[],
                                            route_stop_wait_time=[])
            
            sim_request = SimRequest(origin=node_origin, 
                                     destination=node_destination, 
                                     index=request_index,
                                     pickup_time=requests_pickup_times[request_index])
            
            return new_routing_plan, sim_request
        else:
            return None

    
    def _determine_unallocation(self, m: int, n: int, state_object: MCTS_state, config_flags: Config_flags, planning: bool):

        bus_m = state_object.bus_fleet.routing_plans[m]
        bus_n = state_object.bus_fleet.routing_plans[n] 
        original_cost = bus_m.assignment_cost + bus_n.assignment_cost

        bus_parameters = self._get_bus_parameters_of_interest(state_object=state_object, bus_index=m)
        current_start_time, current_stop_index, current_location, passengers_in_bus, prev_passengers = bus_parameters

        cur_min_edge = VVEdge(bus_m, bus_n, float('inf'))
        potential_requests_to_unallocate = self._get_requests_that_can_be_unallocated(current_stop_index=current_stop_index,
                                                                                      stop_request_pairing=state_object.bus_fleet.routing_plans[m].stops_request_pairing.data)
        for request_index in potential_requests_to_unallocate:
            out = self._get_unallocated_routing_plan(real_start_time=state_object.bus_fleet.routing_plans[m].start_time,
                                            current_start_time=current_start_time,
                                            current_stop_index=current_stop_index,
                                            bus_location=current_location,
                                            passengers_in_bus=passengers_in_bus,
                                            prev_passengers=prev_passengers,
                                            stops_sequence=state_object.bus_fleet.routing_plans[m].bus_stops,
                                            stops_wait_time=state_object.bus_fleet.routing_plans[m].stops_wait_times,
                                            stop_request_pairing=state_object.bus_fleet.routing_plans[m].stops_request_pairing.data,
                                            request_capacities=state_object.request_capacities,
                                            requests_pickup_times=state_object.requests_pickup_times,
                                            request_index=request_index,
                                            cost_type=config_flags.cost_type,
                                            prev_assignment_cost=state_object.bus_fleet.routing_plans[m].assignment_cost,
                                            planning=planning)
            if out is None:
                continue
            new_bus_m, unallocated_request = out
            _, new_bus_n, _ = self._get_dynamic_insertion_cost_for_request(state_object=state_object,
                                                                           bus_index=n,
                                                                           request_index=unallocated_request.index,
                                                                           request_row={"Origin Node": unallocated_request.origin,
                                                                                        "Destination Node": unallocated_request.destination},  
                                                                           config_flags=config_flags,
                                                                           planning=planning)
            if new_bus_n is None:
                continue
            new_cost = new_bus_m.assignment_cost + new_bus_n.assignment_cost
            final_swap_cost = new_cost - original_cost 
            new_edge = VVEdge(new_bus_m, new_bus_n, final_swap_cost)
            cur_min_edge = min(new_edge, cur_min_edge, key=lambda x: x.swap_utility)
        
        if cur_min_edge.swap_utility != float('inf'):
            return cur_min_edge
        else:
            return None
    
    def _generate_route_from_stops(self, current_bus_location, bus_stops, stops_wait_times):
        bus_route = []
        bus_route_edge_time = []
        routes_stop_wait_time= []
        for bus_stop_index in range(len(bus_stops)-1):
            if bus_stop_index == 0:
                origin_stop = current_bus_location
                if bus_stops[bus_stop_index] == current_bus_location:
                    origin_wait_time = stops_wait_times[bus_stop_index]
                else:
                    if bus_stops[bus_stop_index+1] == current_bus_location:
                        continue
                    origin_wait_time = 0
            else:
                origin_stop = bus_stops[bus_stop_index]
                origin_wait_time = stops_wait_times[bus_stop_index]
            destination_bus_stop = bus_stops[bus_stop_index+1]

            shortest_path = self.map_graph.shortest_paths[origin_stop, destination_bus_stop]
            shortest_path_wait_time = []
            stops_wait_time = []
            for node_index in range(len(shortest_path)-1):
                edge_origin = shortest_path[node_index]
                edge_destination = shortest_path[node_index + 1]
                edge_time = self.map_graph.obtain_shortest_paths_time(edge_origin, edge_destination)
                shortest_path_wait_time.append(edge_time)
                if node_index == 0:
                    stops_wait_time.append(origin_wait_time)
                else:
                    stops_wait_time.append(0)

            if len(shortest_path_wait_time) == 0:
                shortest_path_wait_time.append(0)
                stops_wait_time.append(origin_wait_time)
                bus_route += shortest_path
            else:
                bus_route += shortest_path[:-1]
            
            bus_route_edge_time += shortest_path_wait_time
            routes_stop_wait_time += stops_wait_time
        
        bus_route += [bus_stops[-1]]
        routes_stop_wait_time += [stops_wait_times[-1]]
        
        return bus_route, bus_route_edge_time, routes_stop_wait_time
    
    def _generate_sliced_routing_plans(self, current_state_object: MCTS_state, next_state_object: MCTS_state, config_flags: Config_flags):
        routing_plans = []
        for bus_index in range(config_flags.num_buses):
            bus_parameters = self._get_bus_parameters_of_interest(state_object=current_state_object, bus_index=bus_index)
            current_start_time, current_stop_index, current_location, passengers_in_bus, prev_passengers = bus_parameters

            local_stops_sequence = copy.deepcopy(next_state_object.bus_fleet.routing_plans[bus_index].bus_stops[current_stop_index:])
            local_stops_wait_time = copy.deepcopy(next_state_object.bus_fleet.routing_plans[bus_index].stops_wait_times[current_stop_index:])
            local_stop_request_pairing = copy.deepcopy(next_state_object.bus_fleet.routing_plans[bus_index].stops_request_pairing.data[current_stop_index:])

            new_stop_req_pairings = Bus_stop_request_pairings(local_stop_request_pairing)

            new_assignment_cost = next_state_object.bus_fleet.routing_plans[bus_index].assignment_cost
            new_routing_plan = Routing_plan(bus_stops=local_stops_sequence,
                                            stops_wait_times=local_stops_wait_time,
                                            stops_request_pairing=new_stop_req_pairings,
                                            assignment_cost=new_assignment_cost,
                                            newest_assignment_cost=0,
                                            start_time=current_state_object.bus_fleet.routing_plans[bus_index].start_time,
                                            route=[],
                                            route_edge_times=[],
                                            route_stop_wait_time=[])
            routing_plans.append(new_routing_plan)

        return routing_plans

    def assign_requests_and_create_routes(self, state_object: State, requests, config_flags: Config_flags):
     
        if config_flags.plot_final_routes:
            prev_bus_stops = []
            prev_bus_routes = []
            for bus_index in range(state_object.num_buses):
                current_bus_stops = state_object.bus_fleet.routing_plans[bus_index].bus_stops[state_object.bus_stop_index[bus_index]:]
                current_routes = state_object.bus_fleet.routing_plans[bus_index].route[state_object.step_index[bus_index]:]
                prev_bus_routes.append(current_routes)
                prev_bus_stops.append(current_bus_stops)

            self.plot_utils.plot_routes_before_assignment_online(map_object=self.map_graph, 
                                                                requests=requests, 
                                                                prev_bus_stops=prev_bus_stops,
                                                                prev_bus_routes=prev_bus_routes, 
                                                                bus_locations=state_object.bus_locations,
                                                                folder_path=self.results_folder)
        for request_index, request_row, in requests.iterrows():
            if request_row["Requested Pickup Time"].hour < state_object.date_operational_range.start_hour:
                state_object.requests_pickup_times[request_index] = (((((request_row["Requested Pickup Time"].hour + 24) - state_object.date_operational_range.start_hour) * 60) \
                                                                    + request_row["Requested Pickup Time"].minute) * 60) + request_row["Requested Pickup Time"].second
            else:
                state_object.requests_pickup_times[request_index] = ((((request_row["Requested Pickup Time"].hour - state_object.date_operational_range.start_hour) * 60) \
                                                                    + request_row["Requested Pickup Time"].minute) * 60) + request_row["Requested Pickup Time"].second
            state_object.request_capacities[request_index] = request_row["Number of Passengers"]

        for request_index, request_row, in requests.iterrows():
            print(f"Request Index: {request_index}")
            request = SimRequest(origin = request_row['Origin Node'],
                                 destination = request_row['Destination Node'],
                                 index = request_index,
                                 pickup_time = state_object.requests_pickup_times[request_index])
            
            mcts_state_object = MCTS_state(state=state_object)

            greedy_assignment_rv = partial(self._determine_assignment, 
                                           config_flags=config_flags)

            unallocate_method_vv = partial(self._determine_unallocation,
                                           config_flags=config_flags)
            
            create_routes_method = self._generate_route_from_stops

            slicing_method = partial(self._generate_sliced_routing_plans,
                                     config_flags=config_flags)

            out = MCForest(initial_request = request,
                           unallocate_method_vv = unallocate_method_vv, 
                           greedy_assignment_rv = greedy_assignment_rv, 
                           create_routes_method = create_routes_method,
                           slicing_method = slicing_method,
                           config_flags = config_flags,
                           data_folders = self.data_folders, 
                           mcts_state_object = mcts_state_object).get_best_action()
            
            if out:
                for bus_index, new_routing_plan in enumerate(out):
                    current_step_index = state_object.step_index[bus_index]
                    current_stop_wait_time = state_object.bus_fleet.routing_plans[bus_index].route_stop_wait_time[current_step_index]
                    if state_object.wait_time_at_the_station[bus_index] < current_stop_wait_time:
                        current_bus_location=state_object.bus_fleet.routing_plans[bus_index].route[state_object.step_index[bus_index]]
                    else:
                        current_bus_location=state_object.bus_fleet.routing_plans[bus_index].route[state_object.step_index[bus_index]+1]

                    route, bus_route_edge_time, routes_stop_wait_time = self._generate_route_from_stops(current_bus_location=current_bus_location, 
                                                                                                        bus_stops=new_routing_plan.bus_stops, 
                                                                                                        stops_wait_times=new_routing_plan.stops_wait_times)
                    
                    new_routing_plan.update_routes(route=route,
                                                route_edge_times=bus_route_edge_time,
                                                route_stop_wait_time=routes_stop_wait_time)
                    
                    # print(delta_cost)
                    state_object.update_state(bus_index, 
                                            request_index=request_index,
                                            request_row=request_row,
                                            assignment_cost=new_routing_plan.newest_assignment_cost,
                                            new_routing_plan=new_routing_plan)
            else:
                state_object.update_state(None, 
                        request_index=request_index,
                        request_row=None,
                        assignment_cost=None,
                        new_routing_plan=None)

        if config_flags.plot_final_routes and not requests.empty:
                current_bus_stops_list = []
                current_bus_routes_list = []
                for bus_index in range(state_object.num_buses):
                    current_bus_stops = state_object.bus_fleet.routing_plans[bus_index].bus_stops[state_object.bus_stop_index[bus_index]:]
                    current_routes = state_object.bus_fleet.routing_plans[bus_index].route[state_object.step_index[bus_index]:]
                    current_bus_routes_list.append(current_routes)
                    current_bus_stops_list.append(current_bus_stops)
                self.plot_utils.plot_routes_after_assignment_online(map_object=self.map_graph, 
                                                                    outstanding_requests={}, 
                                                                    current_bus_stops=current_bus_stops_list,
                                                                    current_bus_routes=current_bus_routes_list, 
                                                                    bus_locations=state_object.bus_locations,
                                                                    folder_path=self.results_folder)
