"""State.py

This module contains all the required methods to initialize and update the state of the routing system. The state keeps track
of the variables described in the paper, and it is used to describe the locations of requests, buses, and the time left in the time
horizon.

"""
import copy
from State import State
from Data_structures import Bus_stop_request_pairings, Routing_plan

class MCTS_state:
    def __init__(self, state: State):
        self.initial_bus_locations = state.initial_bus_locations
        self.bus_capacities = state.bus_capacities
        self.num_buses = state.num_buses
        self.date_operational_range = state.date_operational_range
        self.time_horizon = state.time_horizon
        self.bus_locations = copy.deepcopy(state.bus_locations)
        self.bus_fleet = copy.deepcopy(state.bus_fleet)
        self.time_spent_at_intersection = copy.deepcopy(state.time_spent_at_intersection)
        self.wait_time_at_the_station = copy.deepcopy(state.wait_time_at_the_station)
        self.step_index = copy.deepcopy(state.step_index)
        self.bus_stop_index = copy.deepcopy(state.bus_stop_index)
        self.passengers_in_bus = copy.deepcopy(state.passengers_in_bus)
        self.route_time = copy.deepcopy(state.route_time)
        self.rejected_requests = []
        self.prev_passengers = copy.deepcopy(state.prev_passengers)
        self.requests_pickup_times = copy.deepcopy(state.requests_pickup_times)
        self.request_capacities = copy.deepcopy(state.request_capacities)
        self.state_num = copy.deepcopy(state.state_num)

    def _dropoff_passengers(self, bus_index: int, next_location: int):
        next_bus_stop_index = self.bus_stop_index[bus_index] + 1
        if next_bus_stop_index < len(self.bus_fleet.routing_plans[bus_index].bus_stops):
            next_bus_stop = self.bus_fleet.routing_plans[bus_index].bus_stops[next_bus_stop_index]

            if next_bus_stop == next_location:
                self.bus_stop_index[bus_index] += 1
                request_index_list = self.bus_fleet.routing_plans[bus_index].stops_request_pairing.data[self.bus_stop_index[bus_index]]["dropoff"]
                
                for request_index in request_index_list:
                    if request_index != -1:
                        if request_index in self.prev_passengers[bus_index]:
                            del(self.prev_passengers[bus_index][request_index])
                            self.passengers_in_bus[bus_index] -= self.request_capacities[request_index]
    
    def _pickup_passengers(self, bus_index: int):
        current_bus_stop_index = self.bus_stop_index[bus_index]
        current_bus_location = self.bus_locations[bus_index]
        current_bus_stop = self.bus_fleet.routing_plans[bus_index].bus_stops[current_bus_stop_index]

        if current_bus_location == current_bus_stop:
            request_index_list = self.bus_fleet.routing_plans[bus_index].stops_request_pairing.data[current_bus_stop_index]["pickup"]

            for request_index in request_index_list:
                if request_index != -1:
                    if request_index not in self.prev_passengers[bus_index]:
                        self.prev_passengers[bus_index][request_index] = [self.bus_fleet.routing_plans[bus_index].bus_stops[current_bus_stop_index], copy.deepcopy(self.state_num)]
                        self.passengers_in_bus[bus_index] += self.request_capacities[request_index]
                        if self.state_num < self.requests_pickup_times[request_index]:
                            print("Current request index = " + str(request_index))
                            print("Current bus stop index = " + str(current_bus_stop_index))
                            print("Current bus stop = " + str(current_bus_stop))
                            current_stop_wait_time = self.bus_fleet.routing_plans[bus_index].stops_wait_times[current_bus_stop_index]
                            print("Current start time = " + str(current_stop_wait_time))
                            print(self.bus_fleet.routing_plans[bus_index].start_time)
                            print("Passenger was picked up before they were available")
    
    def _terminate_route(self, bus_index: int):
        self.step_index[bus_index] = 0
        self.bus_stop_index[bus_index] = 0
        self.bus_locations[bus_index] = self.initial_bus_locations[bus_index]
        self.time_spent_at_intersection[bus_index] = 0
        self.wait_time_at_the_station[bus_index] = 0

        new_routing_plan = Routing_plan(bus_stops=[self.initial_bus_locations[bus_index], self.initial_bus_locations[bus_index]],
                                        stops_wait_times=[0, 0],
                                        stops_request_pairing=Bus_stop_request_pairings([{"pickup": [-1], "dropoff": [-1]}, {"pickup": [-1], "dropoff": [-1]}]),
                                        assignment_cost=self.bus_fleet.routing_plans[bus_index].assignment_cost,
                                        newest_assignment_cost=0,
                                        start_time=self.state_num,
                                        route=[self.initial_bus_locations[bus_index], self.initial_bus_locations[bus_index]],
                                        route_edge_times=[0],
                                        route_stop_wait_time=[0, 0])

        self.bus_fleet.routing_plans[bus_index] = new_routing_plan
    
    def _check_traversal_times(self, bus_index: int):
        current_stop_wait_time = self.bus_fleet.routing_plans[bus_index].route_stop_wait_time[self.step_index[bus_index]]
        current_edge_time = self.bus_fleet.routing_plans[bus_index].route_edge_time[self.step_index[bus_index]]
        
        if self.wait_time_at_the_station[bus_index] < current_stop_wait_time:
            self.wait_time_at_the_station[bus_index] += 1
        else:
            if self.time_spent_at_intersection[bus_index] == 0:
                self.time_spent_at_intersection[bus_index] += 1
                self._pickup_passengers(bus_index=bus_index)
            elif self.time_spent_at_intersection[bus_index] < current_edge_time:
                self.time_spent_at_intersection[bus_index] += 1
            else:
                self.step_index[bus_index] += 1
                next_location = self.bus_fleet.routing_plans[bus_index].route[self.step_index[bus_index]]
                self.bus_locations[bus_index] = next_location
                self.bus_fleet.routing_plans[bus_index].start_time = copy.deepcopy(self.state_num)
                self.time_spent_at_intersection[bus_index] = 0
                self.wait_time_at_the_station[bus_index] = 0
                
                self._dropoff_passengers(bus_index=bus_index, next_location=next_location)
                if self.step_index[bus_index] == len(self.bus_fleet.routing_plans[bus_index].route) - 1:
                    self._terminate_route(bus_index=bus_index)
                else:
                    self._check_traversal_times(bus_index=bus_index)


    def _advance_along_route(self, bus_index: int):
        if self.state_num >= self.bus_fleet.routing_plans[bus_index].start_time:
            if len(self.bus_fleet.routing_plans[bus_index].route) > 2:
                self._check_traversal_times(bus_index=bus_index)
                self.route_time[bus_index] += 1
            else:
                self.bus_fleet.routing_plans[bus_index].start_time = self.state_num + 1

    def update_state(self, bus_index: int, request_index: int, new_routing_plan: Routing_plan | None):
        if new_routing_plan is None:
            self.rejected_requests.append(request_index)
        else:
            self.bus_fleet.routing_plans[bus_index].assignment_cost = new_routing_plan.assignment_cost
            self.bus_fleet.routing_plans[bus_index].newest_assignment_cost = new_routing_plan.newest_assignment_cost

            if len(self.bus_fleet.routing_plans[bus_index].bus_stops) == 2 and \
                self.bus_fleet.routing_plans[bus_index].bus_stops[self.bus_stop_index[bus_index]] == self.bus_fleet.routing_plans[bus_index].bus_stops[self.bus_stop_index[bus_index]+1]:
                self.bus_fleet.routing_plans[bus_index].bus_stops = self.bus_fleet.routing_plans[bus_index].bus_stops[:self.bus_stop_index[bus_index]] + new_routing_plan.bus_stops
                self.bus_fleet.routing_plans[bus_index].stops_wait_times = self.bus_fleet.routing_plans[bus_index].stops_wait_times[:self.bus_stop_index[bus_index]] + new_routing_plan.stops_wait_times
                self.bus_fleet.routing_plans[bus_index].stops_request_pairing.data = self.bus_fleet.routing_plans[bus_index].stops_request_pairing.data[:self.bus_stop_index[bus_index]] + new_routing_plan.stops_request_pairing.data
            else:
                if new_routing_plan.route[0] == self.bus_fleet.routing_plans[bus_index].bus_stops[self.bus_stop_index[bus_index] + 1]:
                    self.bus_fleet.routing_plans[bus_index].bus_stops = self.bus_fleet.routing_plans[bus_index].bus_stops[:self.bus_stop_index[bus_index] + 1] + new_routing_plan.bus_stops
                    self.bus_fleet.routing_plans[bus_index].stops_wait_times = self.bus_fleet.routing_plans[bus_index].stops_wait_times[:self.bus_stop_index[bus_index] + 1] + new_routing_plan.stops_wait_times
                    self.bus_fleet.routing_plans[bus_index].stops_request_pairing.data = self.bus_fleet.routing_plans[bus_index].stops_request_pairing.data[:self.bus_stop_index[bus_index] + 1] + new_routing_plan.stops_request_pairing.data
                else:
                    self.bus_fleet.routing_plans[bus_index].bus_stops = self.bus_fleet.routing_plans[bus_index].bus_stops[:self.bus_stop_index[bus_index]] + new_routing_plan.bus_stops
                    self.bus_fleet.routing_plans[bus_index].stops_wait_times = self.bus_fleet.routing_plans[bus_index].stops_wait_times[:self.bus_stop_index[bus_index]] + new_routing_plan.stops_wait_times
                    self.bus_fleet.routing_plans[bus_index].stops_request_pairing.data = self.bus_fleet.routing_plans[bus_index].stops_request_pairing.data[:self.bus_stop_index[bus_index]] + new_routing_plan.stops_request_pairing.data
            
            if len(self.bus_fleet.routing_plans[bus_index].route) > 2:
                current_step_index = self.step_index[bus_index]
                current_stop_wait_time = self.bus_fleet.routing_plans[bus_index].route_stop_wait_time[current_step_index]

                if self.wait_time_at_the_station[bus_index] < current_stop_wait_time:
                    self.bus_fleet.routing_plans[bus_index].route = self.bus_fleet.routing_plans[bus_index].route[:self.step_index[bus_index]] + new_routing_plan.route
                    self.bus_fleet.routing_plans[bus_index].route_edge_time = self.bus_fleet.routing_plans[bus_index].route_edge_time[:self.step_index[bus_index]] + new_routing_plan.route_edge_time
                    self.bus_fleet.routing_plans[bus_index].route_stop_wait_time = self.bus_fleet.routing_plans[bus_index].route_stop_wait_time[:self.step_index[bus_index]] + new_routing_plan.route_stop_wait_time
                else:
                    self.bus_fleet.routing_plans[bus_index].route = self.bus_fleet.routing_plans[bus_index].route[:self.step_index[bus_index]+1] + new_routing_plan.route
                    self.bus_fleet.routing_plans[bus_index].route_edge_time = self.bus_fleet.routing_plans[bus_index].route_edge_time[:self.step_index[bus_index]+1] + new_routing_plan.route_edge_time
                    self.bus_fleet.routing_plans[bus_index].route_stop_wait_time = self.bus_fleet.routing_plans[bus_index].route_stop_wait_time[:self.step_index[bus_index]+1] + new_routing_plan.route_stop_wait_time
            else:
                self.bus_fleet.routing_plans[bus_index].route = new_routing_plan.route
                self.bus_fleet.routing_plans[bus_index].route_edge_time= new_routing_plan.route_edge_time
                self.bus_fleet.routing_plans[bus_index].route_stop_wait_time = new_routing_plan.route_stop_wait_time
    
    def next_state(self):
        for j in range(len(self.bus_locations)):
            self._advance_along_route(j)
        
        self.state_num += 1