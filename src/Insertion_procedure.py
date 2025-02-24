import copy
from Data_structures import Bus_stop_request_pairings, Routing_plan

# from benchmark_1.logger_file import logger_dict

# print('wait times', logger_dict["MAX_WAIT_TIME_AT_STATION"], logger_dict["MAX_WAIT_TIME_INSIDE_BUS"])
# MAX_WAIT_TIME_AT_STATION = logger_dict["MAX_WAIT_TIME_AT_STATION"]
# MAX_WAIT_TIME_INSIDE_BUS = logger_dict["MAX_WAIT_TIME_INSIDE_BUS"]

MAX_WAIT_TIME_AT_STATION = 4 * 60
MAX_WAIT_TIME_INSIDE_BUS = 4 * 60


class Request_Insertion_Procedure:
    def __init__(self, map_graph):
        self.map_graph = map_graph
    
    def _log_negative_wait_time_at_station(self, actual_pickup_time, request_desired_pickup_time, i, 
                                           current_start_time, pickup_request_index, stops_request_pair,
                                           stops_wait_time):
        print("Pickup time for request is before desired pickup time!!!")
        print("Actual pickup time = " + str(actual_pickup_time))
        print("Desired pickup time = " + str(request_desired_pickup_time))
        print(i)
        print(current_start_time)
        print(pickup_request_index)
        print(stops_request_pair)
        print(stops_wait_time)
    
    def _log_negative_wait_time_inside_bus(self, dropoff_request_index, stops_request_pair, stops_sequence, 
                                           stops_wait_time, initial_station, final_station, time_in_bus_route,
                                           direct_route_time, current_start_time, i):
        print("Time inside the bus is negative !!!")
        print("Request index = " + str(dropoff_request_index))
        start_collection = False
        stops_in_between = []
        wait_times_in_between = []
        for new_index in range(i+1):
            failed_request_index_dict = stops_request_pair[new_index]
            failed_pickup_requests_list = failed_request_index_dict["pickup"]
            if start_collection:
                stops_in_between.append(stops_sequence[new_index])
                wait_times_in_between.append(stops_wait_time[new_index])
            elif dropoff_request_index in failed_pickup_requests_list:
                start_collection = True
                stops_in_between.append(stops_sequence[new_index])
            else:
                continue
        
        direct_route = self.map_graph.shortest_paths[initial_station, final_station]
        actual_bus_route = []
        actual_bus_route_times = []
        for stop_in_between_index in range(len(stops_in_between)-1):
            current_station = stops_in_between[stop_in_between_index]
            next_station = stops_in_between[stop_in_between_index+1]
            route_piece = self.map_graph.shortest_paths[current_station, next_station]
            route_piece_time = self.map_graph.obtain_shortest_paths_time(current_station, next_station)
            if stop_in_between_index == 0:
                actual_bus_route += route_piece
            elif stop_in_between_index == len(stops_in_between)-2:
                actual_bus_route += route_piece[1:]
            else:
                actual_bus_route += route_piece[1:-1]
            actual_bus_route_times.append(route_piece_time)

        print("Initial station according to the simulator = " + str(initial_station))
        print("Initial station according to the iterator = " + str(stops_in_between[0]))
        print("Stops in between = " + str(stops_in_between))
        print("Wait times in between = " + str(wait_times_in_between))
        print("Traversal times in between = " + str(actual_bus_route_times))
        print("Actual bus route = " + str(actual_bus_route))
        print("Time in bus = " + str(time_in_bus_route) + "\n")
        print("Direct route = " + str(direct_route))
        print("Direct route time = " + str(direct_route_time))
        print("Start Time = " + str(current_start_time))
        print(stops_sequence)
        print(stops_wait_time)
        print(stops_request_pair)

    def _calculate_cost_of_route_wait_time(self, current_start_time, stops_sequence, stops_wait_time, stops_request_pair, bus_location,
                                           requests_pickup_times, request_capacities, prev_passengers, consider_route_time=False, include_scaling=False, 
                                           maximize=False, bus_capacity=20, max_wait_time_at_station=MAX_WAIT_TIME_AT_STATION, max_wait_time_inside_bus=MAX_WAIT_TIME_INSIDE_BUS,
                                           max_route_time=28800):
        route_time = 0
        requests_wait_time = 0
        real_time = current_start_time
        serviced_requests = copy.deepcopy(prev_passengers)

        for i in range(len(stops_sequence)-1):
            if i == 0 and bus_location != stops_sequence[0]:
                pickup_requests_list = []
                dropoff_requests_list = []
                current_location = bus_location
            else:
                current_request_index_dict = stops_request_pair[i]
                pickup_requests_list = current_request_index_dict["pickup"]
                dropoff_requests_list = current_request_index_dict["dropoff"]
                current_location = stops_sequence[i]
            
            next_location = stops_sequence[i+1]

            for pickup_request_index in pickup_requests_list:
                if pickup_request_index != -1:
                    request_desired_pickup_time = requests_pickup_times[pickup_request_index]
                    actual_pickup_time = real_time + stops_wait_time[i]
                    if actual_pickup_time - request_desired_pickup_time < 0:
                        self._log_negative_wait_time_at_station(actual_pickup_time=actual_pickup_time,
                                                                request_desired_pickup_time=request_desired_pickup_time,
                                                                i=i,
                                                                current_start_time=current_start_time,
                                                                pickup_request_index=pickup_request_index,
                                                                stops_request_pair=stops_request_pair,
                                                                stops_wait_time=stops_wait_time)
                        
                    if actual_pickup_time - request_desired_pickup_time > max_wait_time_at_station:
                        return None
                    
                    wait_time_at_the_station = (actual_pickup_time - request_desired_pickup_time) * request_capacities[pickup_request_index]
                    requests_wait_time += wait_time_at_the_station

                    serviced_requests[pickup_request_index] = [current_location, actual_pickup_time]
            
            for dropoff_request_index in dropoff_requests_list:
                if dropoff_request_index != -1:
                    if dropoff_request_index in serviced_requests:
                        initial_station = serviced_requests[dropoff_request_index][0]
                        final_station = current_location
                        direct_route_time = self.map_graph.obtain_shortest_paths_time(initial_station, final_station)
                        time_in_bus_route = real_time - serviced_requests[dropoff_request_index][1]
                        if (time_in_bus_route - direct_route_time) < 0:
                            self._log_negative_wait_time_inside_bus(dropoff_request_index, stops_request_pair, stops_sequence, 
                                           stops_wait_time, initial_station, final_station, time_in_bus_route,
                                           direct_route_time, current_start_time, i)

                        if (time_in_bus_route - direct_route_time) > max_wait_time_inside_bus:
                            return None
                        
                        wait_time_inside_bus = (time_in_bus_route - direct_route_time) * request_capacities[dropoff_request_index]
                        requests_wait_time += wait_time_inside_bus
                        del(serviced_requests[dropoff_request_index])
            
            current_edge_cost = self.map_graph.obtain_shortest_paths_time(current_location, next_location)

            if i == 0 and bus_location != stops_sequence[0]:
                stop_wait_time = 0
            else:
                stop_wait_time = stops_wait_time[i]

            real_time += (stop_wait_time + current_edge_cost)
            route_time += (stop_wait_time + current_edge_cost)

        if consider_route_time:
            bus_assignment_cost = route_time + requests_wait_time
        else:
            bus_assignment_cost = requests_wait_time
        
        if real_time > max_route_time:
            return None
        else:
            return bus_assignment_cost
    
    def _calculate_cost_of_route_ptt(self, current_start_time, stops_sequence, stops_wait_time, stops_request_pair, bus_location,
                                    requests_pickup_times, request_capacities, prev_passengers, max_wait_time_at_station=MAX_WAIT_TIME_AT_STATION, 
                                    max_wait_time_inside_bus=MAX_WAIT_TIME_INSIDE_BUS, max_route_time=28800):
        
        real_time = current_start_time
        serviced_requests = copy.deepcopy(prev_passengers)

        ptt_cost = 0

        for i in range(len(stops_sequence)-1):
            if i == 0 and bus_location != stops_sequence[0]:
                pickup_requests_list = []
                dropoff_requests_list = []
                current_location = bus_location
            else:
                current_request_index_dict = stops_request_pair[i]
                pickup_requests_list = current_request_index_dict["pickup"]
                dropoff_requests_list = current_request_index_dict["dropoff"]
                current_location = stops_sequence[i]
            
            next_location = stops_sequence[i+1]

            for pickup_request_index in pickup_requests_list:
                if pickup_request_index != -1:
                    request_desired_pickup_time = requests_pickup_times[pickup_request_index]
                    actual_pickup_time = real_time + stops_wait_time[i]
                    if actual_pickup_time - request_desired_pickup_time < 0:
                        self._log_negative_wait_time_at_station(actual_pickup_time=actual_pickup_time,
                                                                request_desired_pickup_time=request_desired_pickup_time,
                                                                i=i,
                                                                current_start_time=current_start_time,
                                                                pickup_request_index=pickup_request_index,
                                                                stops_request_pair=stops_request_pair,
                                                                stops_wait_time=stops_wait_time)
                        
                    if actual_pickup_time - request_desired_pickup_time > max_wait_time_at_station:
                        return None

                    serviced_requests[pickup_request_index] = [current_location, actual_pickup_time]
            
            for dropoff_request_index in dropoff_requests_list:
                if dropoff_request_index != -1:
                    if dropoff_request_index in serviced_requests:
                        initial_station = serviced_requests[dropoff_request_index][0]
                        final_station = current_location
                        direct_route_time = self.map_graph.obtain_shortest_paths_time(initial_station, final_station)
                        time_in_bus_route = real_time - serviced_requests[dropoff_request_index][1]
                        if (time_in_bus_route - direct_route_time) < 0:
                            self._log_negative_wait_time_inside_bus()

                        if (time_in_bus_route - direct_route_time) > max_wait_time_inside_bus:
                            return None
                        
                        ptt_cost += time_in_bus_route * request_capacities[dropoff_request_index]
                        del(serviced_requests[dropoff_request_index])
            
            current_edge_cost = self.map_graph.obtain_shortest_paths_time(current_location, next_location)

            if i == 0 and bus_location != stops_sequence[0]:
                stop_wait_time = 0
            else:
                stop_wait_time = stops_wait_time[i]

            real_time += (stop_wait_time + current_edge_cost)
        
        if real_time > max_route_time:
            return None
        else:
            return ptt_cost
    
    def _calculate_cost_of_route_budget(self, current_start_time, stops_sequence, stops_wait_time, stops_request_pair, bus_location,
                                    requests_pickup_times, request_capacities, prev_passengers, passengers_in_bus, max_wait_time_at_station=MAX_WAIT_TIME_AT_STATION, 
                                    max_wait_time_inside_bus=MAX_WAIT_TIME_INSIDE_BUS, max_route_time=28800):
        
        real_time = current_start_time
        serviced_requests = copy.deepcopy(prev_passengers)
        local_passengers_in_bus = copy.deepcopy(passengers_in_bus)

        budget_cost = max_route_time - current_start_time

        for i in range(len(stops_sequence)-1):
            if i == 0 and bus_location != stops_sequence[0]:
                pickup_requests_list = []
                dropoff_requests_list = []
                current_location = bus_location
            else:
                current_request_index_dict = stops_request_pair[i]
                pickup_requests_list = current_request_index_dict["pickup"]
                dropoff_requests_list = current_request_index_dict["dropoff"]
                current_location = stops_sequence[i]
            
            next_location = stops_sequence[i+1]
            current_edge_cost = self.map_graph.obtain_shortest_paths_time(current_location, next_location)

            for dropoff_request_index in dropoff_requests_list:
                if dropoff_request_index != -1:
                    if dropoff_request_index in serviced_requests:
                        initial_station = serviced_requests[dropoff_request_index][0]
                        final_station = current_location
                        direct_route_time = self.map_graph.obtain_shortest_paths_time(initial_station, final_station)
                        time_in_bus_route = real_time - serviced_requests[dropoff_request_index][1]
                        if (time_in_bus_route - direct_route_time) < 0:
                            self._log_negative_wait_time_inside_bus()

                        if (time_in_bus_route - direct_route_time) > max_wait_time_inside_bus:
                            return None
                        
                        local_passengers_in_bus -= request_capacities[dropoff_request_index]
                        del(serviced_requests[dropoff_request_index])
            
            number_of_picked_up_passengers = 0

            for pickup_request_index in pickup_requests_list:
                if pickup_request_index != -1:
                    request_desired_pickup_time = requests_pickup_times[pickup_request_index]
                    actual_pickup_time = real_time + stops_wait_time[i]
                    if actual_pickup_time - request_desired_pickup_time < 0:
                        self._log_negative_wait_time_at_station(actual_pickup_time=actual_pickup_time,
                                                                request_desired_pickup_time=request_desired_pickup_time,
                                                                i=i,
                                                                current_start_time=current_start_time,
                                                                pickup_request_index=pickup_request_index,
                                                                stops_request_pair=stops_request_pair,
                                                                stops_wait_time=stops_wait_time)
                        
                    if actual_pickup_time - request_desired_pickup_time > max_wait_time_at_station:
                        return None
                    
                    number_of_picked_up_passengers += request_capacities[pickup_request_index]

                    serviced_requests[pickup_request_index] = [current_location, actual_pickup_time]

            if i == 0 and bus_location != stops_sequence[0]:
                stop_wait_time = 0
            else:
                stop_wait_time = stops_wait_time[i]

            real_time += (stop_wait_time + current_edge_cost)

            if local_passengers_in_bus > 0:
                passengers_in_bus_indicator = 1
                budget_cost -= ((current_edge_cost + stop_wait_time) * passengers_in_bus_indicator)
            else:
                if number_of_picked_up_passengers > 0:
                    passengers_in_bus_indicator = 1
                    budget_cost -= ((current_edge_cost) * passengers_in_bus_indicator)
                else:
                    budget_cost -= 0

            local_passengers_in_bus += number_of_picked_up_passengers
        
        if real_time > max_route_time:
            return None
        else:
            return -1 * budget_cost
    
    def _calculate_cost_of_route(self, current_start_time, stops_sequence, stops_wait_time, stops_request_pair, bus_location,
                                 requests_pickup_times, request_capacities, passengers_in_bus, prev_passengers, 
                                 consider_route_time=False, include_scaling=False, maximize=False, bus_capacity=20, 
                                 max_wait_time_at_station=MAX_WAIT_TIME_AT_STATION, max_wait_time_inside_bus=MAX_WAIT_TIME_INSIDE_BUS,
                                 max_route_time=28800, cost_type="wait_time"):
        
        match cost_type:
            case "wait_time":
                cost_value = self._calculate_cost_of_route_wait_time(current_start_time=current_start_time,
                                                        stops_sequence=stops_sequence,
                                                        stops_wait_time=stops_wait_time,
                                                        stops_request_pair=stops_request_pair,
                                                        bus_location=bus_location,
                                                        requests_pickup_times=requests_pickup_times,
                                                        request_capacities=request_capacities,
                                                        prev_passengers=prev_passengers,
                                                        consider_route_time=consider_route_time,
                                                        include_scaling=include_scaling,
                                                        maximize=maximize,
                                                        bus_capacity=bus_capacity,
                                                        max_wait_time_at_station=max_wait_time_at_station,
                                                        max_wait_time_inside_bus=max_wait_time_inside_bus,
                                                        max_route_time=max_route_time)
            case "ptt":
                cost_value = self._calculate_cost_of_route_ptt(current_start_time=current_start_time,
                                                    stops_sequence=stops_sequence,
                                                    stops_wait_time=stops_wait_time,
                                                    stops_request_pair=stops_request_pair,
                                                    bus_location=bus_location,
                                                    requests_pickup_times=requests_pickup_times,
                                                    request_capacities=request_capacities,
                                                    prev_passengers=prev_passengers,
                                                    max_wait_time_at_station=max_wait_time_at_station,
                                                    max_wait_time_inside_bus=max_wait_time_inside_bus,
                                                    max_route_time=max_route_time)
            case "budget":
                cost_value = self._calculate_cost_of_route_budget(current_start_time=current_start_time,
                                                    stops_sequence=stops_sequence,
                                                    stops_wait_time=stops_wait_time,
                                                    stops_request_pair=stops_request_pair,
                                                    bus_location=bus_location,
                                                    requests_pickup_times=requests_pickup_times,
                                                    request_capacities=request_capacities,
                                                    passengers_in_bus=passengers_in_bus,
                                                    prev_passengers=prev_passengers,
                                                    max_wait_time_at_station=max_wait_time_at_station,
                                                    max_wait_time_inside_bus=max_wait_time_inside_bus,
                                                    max_route_time=max_route_time)
            case _:
                raise ValueError("Cost function type not implemented")
        
        return cost_value

        
    
    def _place_request_inside_stop(self, local_stop_request_pairings, stop_index, request_index, label):
        if local_stop_request_pairings[stop_index][label][0] == -1:
            local_stop_request_pairings[stop_index][label][0] = request_index
        else:
            request_placed = False
            for current_list_index, current_request_index in enumerate(local_stop_request_pairings[stop_index][label]):
                if current_request_index == -1*request_index:
                    local_stop_request_pairings[stop_index][label][current_list_index] = request_index
                    request_placed = True
                    break

            if not request_placed:
                local_stop_request_pairings[stop_index][label].append(request_index)
    
    def _update_stop_request_pairings(self, stop_request_pairings, stop_index, request_index, pickup=False, insert=False):
        if insert:
            if pickup:
                local_stop_request_pairings = stop_request_pairings[:stop_index] + [{"pickup": [request_index], "dropoff": [-1]}] + stop_request_pairings[stop_index:]
            else:
                local_stop_request_pairings = stop_request_pairings[:stop_index] + [{"pickup": [-1], "dropoff": [request_index]}] + stop_request_pairings[stop_index:]
        else:
            local_stop_request_pairings = copy.deepcopy(stop_request_pairings)
            if pickup:
                label = "pickup"
            else:
                label = "dropoff"
            self._place_request_inside_stop(local_stop_request_pairings=local_stop_request_pairings,
                                            stop_index=stop_index,
                                            request_index=request_index,
                                            label=label)
        
        return local_stop_request_pairings
    
    def _update_stop_wait_times(self, local_travel_time, stop_index, stops_sequence, stop_request_pairings, stops_wait_time, requests_pickup_times,
                                default_stop_wait_time=15):
        new_stops_wait_time = copy.deepcopy(stops_wait_time)
        for i in range(stop_index, len(new_stops_wait_time)-1):
            current_request_index_dict = stop_request_pairings[i]
            pickup_requests_list = current_request_index_dict["pickup"]

            for list_index, current_request_index in enumerate(pickup_requests_list):
                if current_request_index == -1:
                    continue
                else:
                    if list_index == 0:
                        new_stops_wait_time[i] = default_stop_wait_time
                    current_request_pickup_time = requests_pickup_times[current_request_index]
                    new_stops_wait_time[i] = max(new_stops_wait_time[i], (current_request_pickup_time-local_travel_time)+default_stop_wait_time)
            current_location = stops_sequence[i]
            wait_time = new_stops_wait_time[i]
            next_location = stops_sequence[i+1]
            current_edge_cost = self.map_graph.obtain_shortest_paths_time(current_location, next_location)
            local_travel_time += (wait_time + current_edge_cost)
        
        current_request_index_dict = stop_request_pairings[len(new_stops_wait_time)-1]
        pickup_requests_list = current_request_index_dict["pickup"]

        for list_index, current_request_index in enumerate(pickup_requests_list):
            if current_request_index == -1:
                continue
            else:
                if list_index == 0:
                    new_stops_wait_time[len(new_stops_wait_time)-1] = default_stop_wait_time
                current_request_pickup_time = requests_pickup_times[current_request_index]
                new_stops_wait_time[len(new_stops_wait_time)-1] = max(new_stops_wait_time[len(new_stops_wait_time)-1], 
                                                                      (current_request_pickup_time-local_travel_time)+default_stop_wait_time)
        
        return new_stops_wait_time
    
    def _create_new_stop_lists(self, requests_pickup_times, new_travel_time, next_index, stops_wait_time, current_stop_wait_time, 
                               stops_sequence, request_node, request_index, 
                               stop_request_pairings, pickup=False, default_stop_wait_time=60):
        if request_node == stops_sequence[next_index]:
            local_travel_time = new_travel_time + stops_wait_time[next_index-1]
            local_stops_sequence = stops_sequence
            local_stops_wait_time = stops_wait_time
            insertion_index = next_index
            insert_flag = False
        
        elif request_node == stops_sequence[next_index-1]:
            local_travel_time = new_travel_time
            local_stops_sequence = stops_sequence
            local_stops_wait_time = stops_wait_time
            insertion_index = next_index-1
            insert_flag = False

        else:
            local_travel_time = new_travel_time + stops_wait_time[next_index-1]
            local_stops_sequence = stops_sequence[:next_index] + [request_node] + stops_sequence[next_index:]
            local_stops_wait_time = stops_wait_time[:next_index] + [current_stop_wait_time] + stops_wait_time[next_index:]
            insertion_index = next_index
            insert_flag = True

        local_stop_request_pairings = self._update_stop_request_pairings(stop_request_pairings=stop_request_pairings,
                                                                        stop_index=insertion_index,
                                                                        request_index=request_index,
                                                                        pickup=pickup,
                                                                        insert=insert_flag)

        new_stops_wait_time = self._update_stop_wait_times(local_travel_time=local_travel_time, 
                                                            stop_index=insertion_index,
                                                            stops_sequence=local_stops_sequence,
                                                            stop_request_pairings=local_stop_request_pairings,
                                                            stops_wait_time=local_stops_wait_time,
                                                            requests_pickup_times=requests_pickup_times,
                                                            default_stop_wait_time=default_stop_wait_time)

        return new_stops_wait_time, local_stops_sequence, local_stop_request_pairings, insertion_index

    def _create_new_stop_lists_online(self, requests_pickup_times, new_travel_time, next_index, stops_wait_time, current_stop_wait_time, 
                               stops_sequence, request_node, request_index, bus_location,
                               stop_request_pairings, pickup=False, default_stop_wait_time=60, mismatched_flag=False):
        
        if request_node == stops_sequence[next_index]:
            if mismatched_flag:
                local_travel_time = new_travel_time
            else:
                local_travel_time = new_travel_time + stops_wait_time[next_index-1]
            local_stops_sequence = stops_sequence
            local_stops_wait_time = stops_wait_time
            insertion_index = next_index
            insert_flag = False
        
        elif request_node == stops_sequence[next_index-1]:
            local_travel_time = new_travel_time
            if mismatched_flag:
                local_stops_sequence = stops_sequence[:next_index] + [request_node] + stops_sequence[next_index:]
                local_stops_wait_time = stops_wait_time[:next_index] + [current_stop_wait_time] + stops_wait_time[next_index:]
                insertion_index = next_index
                insert_flag = True
            else:
                local_stops_sequence = stops_sequence
                local_stops_wait_time = stops_wait_time
                insertion_index = next_index-1
                insert_flag = False

        else:
            if mismatched_flag:
                local_travel_time = new_travel_time
            else:
                local_travel_time = new_travel_time + stops_wait_time[next_index-1]
            local_stops_sequence = stops_sequence[:next_index] + [request_node] + stops_sequence[next_index:]
            local_stops_wait_time = stops_wait_time[:next_index] + [current_stop_wait_time] + stops_wait_time[next_index:]
            insertion_index = next_index
            insert_flag = True

        local_stop_request_pairings = self._update_stop_request_pairings(stop_request_pairings=stop_request_pairings,
                                                                        stop_index=insertion_index,
                                                                        request_index=request_index,
                                                                        pickup=pickup,
                                                                        insert=insert_flag)

        new_stops_wait_time = self._update_stop_wait_times(local_travel_time=local_travel_time, 
                                                            stop_index=insertion_index,
                                                            stops_sequence=local_stops_sequence,
                                                            stop_request_pairings=local_stop_request_pairings,
                                                            stops_wait_time=local_stops_wait_time,
                                                            requests_pickup_times=requests_pickup_times,
                                                            default_stop_wait_time=default_stop_wait_time)

        return new_stops_wait_time, local_stops_sequence, local_stop_request_pairings, insertion_index
    
    def _insert_pickup_in_route(self, current_start_time, total_travel_time, stops_sequence, stops_wait_time, current_location,
                                         next_index, request_origin, requests_pickup_times, stop_request_pairings, request_index, 
                                         default_stop_wait_time=60):
        time_to_pickup = self.map_graph.obtain_shortest_paths_time(current_location, request_origin)

        if total_travel_time == 0:
            time_until_request_available = max(0, requests_pickup_times[request_index]-time_to_pickup)
            new_start_time = time_until_request_available
            current_stop_wait_time = default_stop_wait_time
        else:
            new_start_time = current_start_time
            current_travel_time = total_travel_time + current_start_time + time_to_pickup + stops_wait_time[next_index - 1]
            current_request_wait_time = (requests_pickup_times[request_index] - current_travel_time) + default_stop_wait_time
            current_stop_wait_time = max(default_stop_wait_time, current_request_wait_time)
        
        new_travel_time = total_travel_time + new_start_time + time_to_pickup

        new_lists = self._create_new_stop_lists(requests_pickup_times=requests_pickup_times,
                                                new_travel_time=new_travel_time,
                                                next_index=next_index,
                                                stops_wait_time=stops_wait_time,
                                                current_stop_wait_time=current_stop_wait_time,
                                                stops_sequence=stops_sequence,
                                                request_node=request_origin,
                                                request_index=request_index,
                                                stop_request_pairings=stop_request_pairings,
                                                pickup=True,
                                                default_stop_wait_time=default_stop_wait_time)
        
        stops_wait_time_with_insertion, stop_sequence_with_insertion, stop_request_pair_with_insertion, insertion_index = new_lists

        return stop_sequence_with_insertion, stops_wait_time_with_insertion, stop_request_pair_with_insertion, new_start_time, insertion_index
    
    def _insert_pickup_in_route_online(self, current_start_time, total_travel_time, stops_sequence, stops_wait_time, current_location,
                                         next_index, request_origin, requests_pickup_times, stop_request_pairings, request_index, 
                                         default_stop_wait_time=60, mismatched_flag=False):
        time_to_pickup = self.map_graph.obtain_shortest_paths_time(current_location, request_origin)

        new_travel_time = total_travel_time + current_start_time + time_to_pickup
        current_request_wait_time = (requests_pickup_times[request_index] - new_travel_time) + default_stop_wait_time
        current_stop_wait_time = max(default_stop_wait_time, current_request_wait_time)

        new_lists = self._create_new_stop_lists_online(requests_pickup_times=requests_pickup_times,
                                                new_travel_time=new_travel_time,
                                                next_index=next_index,
                                                stops_wait_time=stops_wait_time,
                                                current_stop_wait_time=current_stop_wait_time,
                                                stops_sequence=stops_sequence,
                                                request_node=request_origin,
                                                request_index=request_index,
                                                bus_location=current_location,
                                                stop_request_pairings=stop_request_pairings,
                                                pickup=True,
                                                default_stop_wait_time=default_stop_wait_time,
                                                mismatched_flag=mismatched_flag)
        
        stops_wait_time_with_insertion, stop_sequence_with_insertion, stop_request_pair_with_insertion, insertion_index = new_lists

        return stop_sequence_with_insertion, stops_wait_time_with_insertion, stop_request_pair_with_insertion, current_start_time, insertion_index
    
    def _insert_dropoff_in_route(self, current_start_time, total_travel_time, stops_sequence, stops_wait_time, current_location,
                                next_index, request_destination, requests_pickup_times, stop_request_pairings, request_index, 
                                default_dropoff_wait_time=20, default_stop_wait_time=60):
        
        time_to_dropoff = self.map_graph.obtain_shortest_paths_time(current_location, request_destination)
        
        current_stop_wait_time = default_dropoff_wait_time

        new_travel_time = total_travel_time + current_start_time + time_to_dropoff
        
        new_lists = self._create_new_stop_lists(requests_pickup_times=requests_pickup_times,
                                                new_travel_time=new_travel_time,
                                                next_index=next_index,
                                                stops_wait_time=stops_wait_time,
                                                current_stop_wait_time=current_stop_wait_time,
                                                stops_sequence=stops_sequence,
                                                request_node=request_destination,
                                                request_index=request_index,
                                                stop_request_pairings=stop_request_pairings,
                                                pickup=False,
                                                default_stop_wait_time=default_stop_wait_time)
        
        stops_wait_time_with_insertion, stop_sequence_with_insertion, stop_request_pair_with_insertion, _ = new_lists

        return stop_sequence_with_insertion, stops_wait_time_with_insertion, stop_request_pair_with_insertion
    
    def _obtain_passengers_in_bus(self, stop_index, travel_time, bus_stops, stops_wait_time, passenger_in_bus, stop_request_pairings, 
                                                   serviced_requests, request_capacities):
        current_request_index_dict = stop_request_pairings[stop_index]
        pickup_requests_list = current_request_index_dict["pickup"]
        dropoff_requests_list = current_request_index_dict["dropoff"]

        for pickup_request_index in pickup_requests_list:
            if pickup_request_index >= 0:
                if pickup_request_index not in serviced_requests:
                    serviced_requests[pickup_request_index] = [bus_stops[stop_index], travel_time+stops_wait_time[stop_index]]
                    passenger_in_bus += request_capacities[pickup_request_index]
        
        for dropoff_request_index in dropoff_requests_list:
            if dropoff_request_index >= 0:
                if dropoff_request_index in serviced_requests:
                    del(serviced_requests[dropoff_request_index])
                    passenger_in_bus -= request_capacities[dropoff_request_index]
        
        return passenger_in_bus
    
    def _obtain_passengers_in_bus_online(self, stop_index, travel_time, bus_stops, stops_wait_time, passenger_in_bus, stop_request_pairings, 
                                                   serviced_requests, request_capacities):
        current_request_index_dict = stop_request_pairings[stop_index]
        pickup_requests_list = current_request_index_dict["pickup"]
        dropoff_requests_list = current_request_index_dict["dropoff"]

        for pickup_request_index in pickup_requests_list:
            if pickup_request_index >= 0:
                if pickup_request_index not in serviced_requests:
                    serviced_requests[pickup_request_index] = [bus_stops[stop_index], travel_time+stops_wait_time[stop_index]]
                    passenger_in_bus += request_capacities[pickup_request_index]
        
        for dropoff_request_index in dropoff_requests_list:
            if dropoff_request_index >= 0:
                if dropoff_request_index in serviced_requests:
                    del(serviced_requests[dropoff_request_index])
                    passenger_in_bus -= request_capacities[dropoff_request_index]
        
        return passenger_in_bus
    
    def _place_request_offline_exact(self, current_start_time, bus_capacity, stops_sequence, stops_wait_time, request_origin, request_destination, requests_pickup_times, 
                                     stop_request_pairings, request_index, request_capacities, consider_route_time=True, include_scaling=False,
                                     cost_type="wait_time"):
        total_travel_time = 0
        min_cost = float("inf")
        min_start_time = 0
        min_stop_sequence = []
        min_stop_wait_times = []
        min_stop_request_pairings = []
        serviced_requests = {}
        passenger_in_bus = 0
        original_route_cost = self._calculate_cost_of_route(current_start_time=current_start_time,
                                                            stops_sequence=stops_sequence, 
                                                            stops_wait_time=stops_wait_time,
                                                            stops_request_pair=stop_request_pairings,
                                                            bus_location=stops_sequence[0],
                                                            requests_pickup_times=requests_pickup_times,
                                                            request_capacities=request_capacities, 
                                                            passengers_in_bus=0,
                                                            prev_passengers={},
                                                            consider_route_time=consider_route_time,
                                                            include_scaling=include_scaling,
                                                            bus_capacity=bus_capacity,
                                                            cost_type=cost_type)
        assert original_route_cost is not None

        for i in range(len(stops_sequence)-1):
            passenger_in_bus = self._obtain_passengers_in_bus(stop_index=i, 
                                                            travel_time=total_travel_time+current_start_time,
                                                            bus_stops=stops_sequence,
                                                            stops_wait_time=stops_wait_time,
                                                            passenger_in_bus=passenger_in_bus,
                                                            stop_request_pairings=stop_request_pairings,
                                                            serviced_requests=serviced_requests,
                                                            request_capacities=request_capacities)
            current_location = stops_sequence[i]
            next_location = stops_sequence[i+1]
            next_index = i+1

            if passenger_in_bus + request_capacities[request_index] <= bus_capacity:
                deviation_result = self._insert_pickup_in_route(current_start_time=current_start_time,
                                                                total_travel_time=total_travel_time,
                                                                stops_sequence=stops_sequence,
                                                                stops_wait_time=stops_wait_time, 
                                                                current_location=current_location,
                                                                next_index=next_index,  
                                                                request_origin=request_origin, 
                                                                requests_pickup_times=requests_pickup_times,
                                                                stop_request_pairings=stop_request_pairings,
                                                                request_index=request_index)
                
                new_stop_sequence, new_stops_wait_time, new_stop_req_pair, new_start_time, insertion_index = deviation_result

                new_serviced_stops = new_stop_sequence[:insertion_index]
                new_planned_stops = new_stop_sequence[insertion_index:]
                new_serviced_stop_wait_times = new_stops_wait_time[:insertion_index]
                new_planned_stop_wait_times = new_stops_wait_time[insertion_index:]
                new_serviced_stop_req_pair = new_stop_req_pair[:insertion_index]
                new_planned_stop_req_pair =  new_stop_req_pair[insertion_index:]

                # drop_off_deviation_cost
                local_passengers_in_bus = copy.deepcopy(passenger_in_bus)
                local_serviced_requests = copy.deepcopy(serviced_requests)

                dropoff_travel_time = 0

                if insertion_index == next_index:
                    time_to_pickup = self.map_graph.obtain_shortest_paths_time(current_location, request_origin)
                    new_total_travel_time = total_travel_time + stops_wait_time[i] + time_to_pickup
                else:
                    new_total_travel_time = total_travel_time

                for j in range(len(new_planned_stops)-1):
                    full_travel_time = dropoff_travel_time + new_total_travel_time
                    new_next_index = j+1
                    new_current_location = new_planned_stops[j]
                    new_next_location = new_planned_stops[j+1]
                    
                    local_passengers_in_bus = self._obtain_passengers_in_bus(stop_index=j,
                                                                            travel_time=full_travel_time+new_start_time,
                                                                            bus_stops=new_planned_stops,
                                                                            stops_wait_time=new_planned_stop_wait_times,
                                                                            passenger_in_bus=local_passengers_in_bus,
                                                                            stop_request_pairings=new_planned_stop_req_pair,
                                                                            serviced_requests=local_serviced_requests,
                                                                            request_capacities=request_capacities)

                    total_passengers_in_bus = local_passengers_in_bus
                    if total_passengers_in_bus > bus_capacity:
                        break
                    else:
                        # deviation cost
                        destination_dev_result = self._insert_dropoff_in_route(current_start_time=new_start_time,
                                                                                total_travel_time=full_travel_time,
                                                                                stops_sequence=new_planned_stops,
                                                                                stops_wait_time=new_planned_stop_wait_times,
                                                                                current_location=new_current_location,
                                                                                next_index=new_next_index,
                                                                                request_destination=request_destination,
                                                                                requests_pickup_times=requests_pickup_times,
                                                                                stop_request_pairings=new_planned_stop_req_pair,
                                                                                request_index=request_index)

                        final_planned_stops, final_planned_stops_wait_time, final_planned_stop_req_pair = destination_dev_result

                        full_stop_sequence = new_serviced_stops + final_planned_stops
                        full_stops_wait_time = new_serviced_stop_wait_times + final_planned_stops_wait_time
                        full_stop_req_pair = new_serviced_stop_req_pair + final_planned_stop_req_pair

                        new_route_cost = self._calculate_cost_of_route(current_start_time=new_start_time,
                                                            stops_sequence=full_stop_sequence,
                                                            stops_wait_time=full_stops_wait_time,
                                                            stops_request_pair=full_stop_req_pair,
                                                            bus_location=full_stop_sequence[0],
                                                            requests_pickup_times=requests_pickup_times,
                                                            request_capacities=request_capacities,
                                                            passengers_in_bus=0,
                                                            prev_passengers={},
                                                            consider_route_time=consider_route_time,
                                                            include_scaling=include_scaling,
                                                            bus_capacity=bus_capacity,
                                                            cost_type=cost_type)
                        
                        if new_route_cost is None:
                            total_dev_cost = float("inf")
                        else:
                            total_dev_cost =  new_route_cost - original_route_cost

                        if total_dev_cost < min_cost:
                            min_cost = total_dev_cost
                            min_start_time = new_start_time
                            min_stop_sequence = full_stop_sequence
                            min_stop_wait_times = full_stops_wait_time
                            min_stop_request_pairings = full_stop_req_pair
                            
                    new_current_edge_cost = self.map_graph.obtain_shortest_paths_time(new_current_location, new_next_location)
                    dropoff_travel_time += (new_planned_stop_wait_times[j] + new_current_edge_cost)

            current_edge_cost = self.map_graph.obtain_shortest_paths_time(current_location, next_location)
            total_travel_time += (stops_wait_time[i] + current_edge_cost)

        return min_cost, min_stop_sequence, min_stop_wait_times, min_stop_request_pairings, min_start_time
    
    def _place_request_online_exact(self, current_start_time, bus_capacity, bus_location, planned_stops, stops_wait_time, request_origin, 
                                    request_destination, requests_pickup_times, stop_request_pairings, passengers_in_bus, 
                                    prev_passengers, request_index, request_capacities, consider_route_time=True, include_scaling=False,
                                    cost_type="wait_time"):
        total_travel_time = 0
        min_cost = float("inf")
        min_stop_sequence = []
        min_stop_wait_times = []
        min_stop_request_pairings = []
        min_start_time = 0
        serviced_requests = copy.deepcopy(prev_passengers)
        local_passengers_in_bus = copy.deepcopy(passengers_in_bus)

        if cost_type == "wait_time":
            original_route_cost = self._calculate_cost_of_route(current_start_time=current_start_time,
                                                                stops_sequence=planned_stops, 
                                                                stops_wait_time=stops_wait_time,
                                                                stops_request_pair=stop_request_pairings,
                                                                bus_location=bus_location,
                                                                requests_pickup_times=requests_pickup_times,
                                                                request_capacities=request_capacities, 
                                                                passengers_in_bus=passengers_in_bus,
                                                                prev_passengers=prev_passengers,
                                                                consider_route_time=consider_route_time,
                                                                include_scaling=include_scaling,
                                                                bus_capacity=bus_capacity,
                                                                cost_type=cost_type)
            
            assert original_route_cost is not None
        else:
            original_route_cost = 0
        
        for i in range(len(planned_stops)-1):
            if i == 0 and bus_location != planned_stops[0]:
                local_passengers_in_bus = local_passengers_in_bus
                current_location = bus_location
            else:
                local_passengers_in_bus = self._obtain_passengers_in_bus_online(stop_index=i, 
                                                                        travel_time=total_travel_time+current_start_time,
                                                                        bus_stops=planned_stops,
                                                                        stops_wait_time=stops_wait_time,
                                                                        passenger_in_bus=local_passengers_in_bus,
                                                                        stop_request_pairings=stop_request_pairings,
                                                                        serviced_requests=serviced_requests,
                                                                        request_capacities=request_capacities)
                current_location = planned_stops[i]

            next_location = planned_stops[i+1]
            next_index = i+1
            
            if local_passengers_in_bus  + request_capacities[request_index] <= bus_capacity:
                if i == 0 and bus_location != planned_stops[0]:
                    mismatched_flag = True
                else:
                    mismatched_flag = False
                
                deviation_result = self._insert_pickup_in_route_online(current_start_time=current_start_time,
                                                                total_travel_time=total_travel_time,
                                                                stops_sequence=planned_stops,
                                                                stops_wait_time=stops_wait_time, 
                                                                current_location=current_location,
                                                                next_index=next_index,
                                                                request_origin=request_origin, 
                                                                requests_pickup_times=requests_pickup_times,
                                                                stop_request_pairings=stop_request_pairings,
                                                                request_index=request_index,
                                                                mismatched_flag=mismatched_flag)
                
                new_stop_sequence, new_stops_wait_time, new_stop_req_pair, new_start_time, insertion_index = deviation_result

                new_serviced_stops = new_stop_sequence[:insertion_index]
                new_planned_stops = new_stop_sequence[insertion_index:]
                new_serviced_stop_wait_times = new_stops_wait_time[:insertion_index]
                new_planned_stop_wait_times = new_stops_wait_time[insertion_index:]
                new_serviced_stop_req_pair = new_stop_req_pair[:insertion_index]
                new_planned_stop_req_pair =  new_stop_req_pair[insertion_index:]

                # drop_off_deviation_cost
                new_local_passengers_in_bus = copy.deepcopy(local_passengers_in_bus)
                new_local_serviced_requests = copy.deepcopy(serviced_requests)

                dropoff_travel_time = 0

                if insertion_index == next_index:
                    time_to_pickup = self.map_graph.obtain_shortest_paths_time(current_location, request_origin)
                    if i == 0 and bus_location != planned_stops[0]:
                        stop_time = 0
                    else:
                        stop_time = stops_wait_time[i]
                    new_total_travel_time = total_travel_time + stop_time + time_to_pickup
                else:
                    new_total_travel_time = total_travel_time

                for j in range(len(new_planned_stops)-1):
                    new_full_travel_time = dropoff_travel_time + new_total_travel_time
                    new_next_index = j+1
                    new_current_location = new_planned_stops[j]
                    new_next_location = new_planned_stops[j+1]

                    new_local_passengers_in_bus = self._obtain_passengers_in_bus_online(stop_index=j,
                                                                                travel_time=new_full_travel_time+current_start_time,
                                                                                bus_stops=new_planned_stops,
                                                                                stops_wait_time=new_planned_stop_wait_times,
                                                                                passenger_in_bus=new_local_passengers_in_bus,
                                                                                stop_request_pairings=new_planned_stop_req_pair,
                                                                                serviced_requests=new_local_serviced_requests,
                                                                                request_capacities=request_capacities)

                    total_passengers_in_bus = new_local_passengers_in_bus
                    if total_passengers_in_bus > bus_capacity:
                        break
                    else:
                        # deviation cost
                        destination_dev_result = self._insert_dropoff_in_route(current_start_time=current_start_time,
                                                                            total_travel_time=new_full_travel_time,
                                                                            stops_sequence=new_planned_stops,
                                                                            stops_wait_time=new_planned_stop_wait_times,
                                                                            current_location=new_current_location,
                                                                            next_index=new_next_index,
                                                                            request_destination=request_destination,
                                                                            requests_pickup_times=requests_pickup_times,
                                                                            stop_request_pairings=new_planned_stop_req_pair,
                                                                            request_index=request_index)

                        final_planned_stops, final_planned_stops_wait_time, final_planned_stop_req_pair = destination_dev_result

                        full_stop_sequence = new_serviced_stops + final_planned_stops
                        full_stops_wait_time = new_serviced_stop_wait_times + final_planned_stops_wait_time
                        full_stop_req_pair = new_serviced_stop_req_pair + final_planned_stop_req_pair

                        new_route_cost = self._calculate_cost_of_route(current_start_time=current_start_time,
                                                                        stops_sequence=full_stop_sequence, 
                                                                        stops_wait_time=full_stops_wait_time,
                                                                        stops_request_pair=full_stop_req_pair,
                                                                        bus_location=bus_location,
                                                                        requests_pickup_times=requests_pickup_times,
                                                                        request_capacities=request_capacities, 
                                                                        passengers_in_bus=passengers_in_bus,
                                                                        prev_passengers=prev_passengers,
                                                                        consider_route_time=consider_route_time,
                                                                        include_scaling=include_scaling,
                                                                        bus_capacity=bus_capacity,
                                                                        cost_type=cost_type)
                        
                        if new_route_cost is None:
                            total_dev_cost = float("inf")
                        else:
                            total_dev_cost =  new_route_cost - original_route_cost

                        if total_dev_cost < 0 and cost_type == "wait_time":
                            print("Request index = " + str(request_index))
                            print("New bus stops = " + str(full_stop_sequence))


                        if total_dev_cost < min_cost:
                            min_cost = total_dev_cost
                            min_start_time = new_start_time
                            min_stop_sequence = full_stop_sequence
                            min_stop_wait_times = full_stops_wait_time
                            min_stop_request_pairings = full_stop_req_pair

                    new_current_edge_cost = self.map_graph.obtain_shortest_paths_time(new_current_location, new_next_location)
                    dropoff_travel_time += (new_planned_stop_wait_times[j] + new_current_edge_cost)

            if i == 0 and bus_location != planned_stops[0]:
                current_wait_time = 0
            else:
                current_wait_time = stops_wait_time[i]
            current_edge_cost = self.map_graph.obtain_shortest_paths_time(current_location, next_location)
            total_travel_time +=  (current_wait_time + current_edge_cost)

        return min_cost, min_stop_sequence, min_stop_wait_times, min_stop_request_pairings, min_start_time
    
    def static_insertion(self, current_start_time, bus_capacity, stops_sequence, stops_wait_time, stop_request_pairing, requests_pickup_times, request_capacities, 
                         request_origin, request_destination, request_index, consider_route_time=True, approximate=False, include_scaling=True):
        
        local_stops_sequence = copy.deepcopy(stops_sequence)
        local_stops_wait_time = copy.deepcopy(stops_wait_time)
        local_stop_request_pairing = copy.deepcopy(stop_request_pairing)

        deviation_result = self._place_request_offline_exact(current_start_time=current_start_time,
                                                             bus_capacity=bus_capacity,
                                                             stops_sequence=local_stops_sequence,
                                                             stops_wait_time=local_stops_wait_time,
                                                             request_origin=request_origin,
                                                             request_destination=request_destination,
                                                             requests_pickup_times=requests_pickup_times,
                                                             stop_request_pairings=local_stop_request_pairing,
                                                             request_index=request_index,
                                                             request_capacities=request_capacities,
                                                             consider_route_time=consider_route_time,
                                                             include_scaling=include_scaling)
        
        total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair, min_start_time = deviation_result

        return total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair, min_start_time
    
    def dynamic_insertion(self, current_start_time, current_stop_index, bus_capacity, passengers_in_bus, prev_passengers, bus_location,
                          stops_sequence, stops_wait_time, stop_request_pairing, request_capacities, request_origin, request_destination, 
                          requests_pickup_times, request_index, consider_route_time=True, approximate=False, include_scaling=True,
                          cost_type="wait_time"):
        
        local_stops_sequence = copy.deepcopy(stops_sequence[current_stop_index:])
        local_stops_wait_time = copy.deepcopy(stops_wait_time[current_stop_index:])
        local_stop_request_pairing = copy.deepcopy(stop_request_pairing[current_stop_index:])

        if len(local_stops_sequence) == 1:
            local_stops_sequence = local_stops_sequence + local_stops_sequence
            local_stops_wait_time = local_stops_wait_time + local_stops_wait_time
            local_stop_request_pairing = local_stop_request_pairing + local_stop_request_pairing

        deviation_result = self._place_request_online_exact(current_start_time=current_start_time, 
                                                            bus_capacity=bus_capacity,
                                                            bus_location=bus_location,
                                                             planned_stops=local_stops_sequence,
                                                             stops_wait_time=local_stops_wait_time,
                                                             request_origin=request_origin,
                                                             request_destination=request_destination,
                                                             requests_pickup_times=requests_pickup_times,
                                                             passengers_in_bus=passengers_in_bus,
                                                             prev_passengers=prev_passengers,
                                                             stop_request_pairings=local_stop_request_pairing,
                                                             request_index=request_index,
                                                             request_capacities=request_capacities,
                                                             consider_route_time=consider_route_time,
                                                             include_scaling=include_scaling,
                                                             cost_type=cost_type)
        
        total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair, _ = deviation_result

        return total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair

class Request_Insertion_Procedure_MCTS:
    def __init__(self, map_graph, MAX_WAIT_TIME_AT_STATION = 4 * 60, MAX_WAIT_TIME_INSIDE_BUS = 4 * 60):
        self.map_graph = map_graph
        self.MAX_WAIT_TIME_AT_STATION = MAX_WAIT_TIME_AT_STATION
        self.MAX_WAIT_TIME_INSIDE_BUS = MAX_WAIT_TIME_INSIDE_BUS
    
    def _log_negative_wait_time_at_station(self, actual_pickup_time, request_desired_pickup_time, i, 
                                           current_start_time, pickup_request_index, stops_request_pair,
                                           stops_wait_time):
        print("Pickup time for request is before desired pickup time!!!")
        print("Actual pickup time = " + str(actual_pickup_time))
        print("Desired pickup time = " + str(request_desired_pickup_time))
        print(i)
        print(current_start_time)
        print(pickup_request_index)
        print(stops_request_pair)
        print(stops_wait_time)
    
    def _log_negative_wait_time_inside_bus(self, dropoff_request_index, stops_request_pair, stops_sequence, 
                                           stops_wait_time, initial_station, final_station, time_in_bus_route,
                                           direct_route_time, current_start_time, i):
        print("Time inside the bus is negative !!!")
        print("Request index = " + str(dropoff_request_index))
        # start_collection = False
        # stops_in_between = []
        # wait_times_in_between = []
        # for new_index in range(i+1):
        #     failed_request_index_dict = stops_request_pair[new_index]
        #     failed_pickup_requests_list = failed_request_index_dict["pickup"]
        #     if start_collection:
        #         stops_in_between.append(stops_sequence[new_index])
        #         wait_times_in_between.append(stops_wait_time[new_index])
        #     elif dropoff_request_index in failed_pickup_requests_list:
        #         start_collection = True
        #         stops_in_between.append(stops_sequence[new_index])
        #     else:
        #         continue
        
        # direct_route = self.map_graph.shortest_paths[initial_station, final_station]
        # actual_bus_route = []
        # actual_bus_route_times = []
        # for stop_in_between_index in range(len(stops_in_between)-1):
        #     current_station = stops_in_between[stop_in_between_index]
        #     next_station = stops_in_between[stop_in_between_index+1]
        #     route_piece = self.map_graph.shortest_paths[current_station, next_station]
        #     route_piece_time = self.map_graph.obtain_shortest_paths_time(current_station, next_station)
        #     if stop_in_between_index == 0:
        #         actual_bus_route += route_piece
        #     elif stop_in_between_index == len(stops_in_between)-2:
        #         actual_bus_route += route_piece[1:]
        #     else:
        #         actual_bus_route += route_piece[1:-1]
        #     actual_bus_route_times.append(route_piece_time)

        print("Initial station according to the simulator = " + str(initial_station))
        # print("Initial station according to the iterator = " + str(stops_in_between[0]))
        # print("Stops in between = " + str(stops_in_between))
        # print("Wait times in between = " + str(wait_times_in_between))
        # print("Traversal times in between = " + str(actual_bus_route_times))
        # print("Actual bus route = " + str(actual_bus_route))
        print("Time in bus = " + str(time_in_bus_route) + "\n")
        # print("Direct route = " + str(direct_route))
        print("Direct route time = " + str(direct_route_time))
        print("Start Time = " + str(current_start_time))
        print(stops_sequence)
        print(stops_wait_time)
        print(stops_request_pair)

    def _calculate_cost_of_route_wait_time(self, current_start_time, stops_sequence, stops_wait_time, stops_request_pair, bus_location,
                                           requests_pickup_times, request_capacities, prev_passengers, consider_route_time=False, include_scaling=False, 
                                           maximize=False, bus_capacity=20, max_wait_time_at_station=MAX_WAIT_TIME_AT_STATION, max_wait_time_inside_bus=MAX_WAIT_TIME_INSIDE_BUS,
                                           max_route_time=28800, planning=False):
        route_time = 0
        requests_wait_time = 0
        real_time = current_start_time
        serviced_requests = copy.deepcopy(prev_passengers)

        for i in range(len(stops_sequence)-1):
            if i == 0 and bus_location != stops_sequence[0]:
                pickup_requests_list = []
                dropoff_requests_list = []
                current_location = bus_location
            else:
                current_request_index_dict = stops_request_pair[i]
                pickup_requests_list = current_request_index_dict["pickup"]
                dropoff_requests_list = current_request_index_dict["dropoff"]
                current_location = stops_sequence[i]
            
            next_location = stops_sequence[i+1]

            for pickup_request_index in pickup_requests_list:
                if pickup_request_index != -1:
                    request_desired_pickup_time = requests_pickup_times[pickup_request_index]
                    actual_pickup_time = real_time + stops_wait_time[i]
                    if actual_pickup_time - request_desired_pickup_time < 0:
                        self._log_negative_wait_time_at_station(actual_pickup_time=actual_pickup_time,
                                                                request_desired_pickup_time=request_desired_pickup_time,
                                                                i=i,
                                                                current_start_time=current_start_time,
                                                                pickup_request_index=pickup_request_index,
                                                                stops_request_pair=stops_request_pair,
                                                                stops_wait_time=stops_wait_time)
                        
                    if (actual_pickup_time - request_desired_pickup_time > self.MAX_WAIT_TIME_AT_STATION) and not planning:
                        return None
                    
                    wait_time_at_the_station = (actual_pickup_time - request_desired_pickup_time) * request_capacities[pickup_request_index]
                    requests_wait_time += wait_time_at_the_station

                    serviced_requests[pickup_request_index] = [current_location, actual_pickup_time]
            
            for dropoff_request_index in dropoff_requests_list:
                if dropoff_request_index != -1:
                    if dropoff_request_index in serviced_requests:
                        initial_station = serviced_requests[dropoff_request_index][0]
                        final_station = current_location
                        direct_route_time = self.map_graph.obtain_shortest_paths_time(initial_station, final_station)
                        time_in_bus_route = real_time - serviced_requests[dropoff_request_index][1]
                        if (time_in_bus_route - direct_route_time) < 0:
                            self._log_negative_wait_time_inside_bus(dropoff_request_index, stops_request_pair, stops_sequence, 
                                           stops_wait_time, initial_station, final_station, time_in_bus_route,
                                           direct_route_time, current_start_time, i)

                        if (time_in_bus_route - direct_route_time) > self.MAX_WAIT_TIME_INSIDE_BUS:
                            return None
                        
                        wait_time_inside_bus = (time_in_bus_route - direct_route_time) * request_capacities[dropoff_request_index]
                        requests_wait_time += wait_time_inside_bus
                        del(serviced_requests[dropoff_request_index])
            
            current_edge_cost = self.map_graph.obtain_shortest_paths_time(current_location, next_location)

            if i == 0 and bus_location != stops_sequence[0]:
                stop_wait_time = 0
            else:
                stop_wait_time = stops_wait_time[i]

            real_time += (stop_wait_time + current_edge_cost)
            route_time += (stop_wait_time + current_edge_cost)

        if consider_route_time:
            bus_assignment_cost = route_time + requests_wait_time
        else:
            bus_assignment_cost = requests_wait_time
        
        if real_time > max_route_time:
            return None
        else:
            return bus_assignment_cost
    
    def _calculate_cost_of_route_ptt(self, current_start_time, stops_sequence, stops_wait_time, stops_request_pair, bus_location,
                                    requests_pickup_times, request_capacities, prev_passengers, max_wait_time_at_station=MAX_WAIT_TIME_AT_STATION, 
                                    max_wait_time_inside_bus=MAX_WAIT_TIME_INSIDE_BUS, max_route_time=28800, planning=False):
        
        real_time = current_start_time
        serviced_requests = copy.deepcopy(prev_passengers)

        ptt_cost = 0

        for i in range(len(stops_sequence)-1):
            if i == 0 and bus_location != stops_sequence[0]:
                pickup_requests_list = []
                dropoff_requests_list = []
                current_location = bus_location
            else:
                current_request_index_dict = stops_request_pair[i]
                pickup_requests_list = current_request_index_dict["pickup"]
                dropoff_requests_list = current_request_index_dict["dropoff"]
                current_location = stops_sequence[i]
            
            next_location = stops_sequence[i+1]

            for pickup_request_index in pickup_requests_list:
                if pickup_request_index != -1:
                    request_desired_pickup_time = requests_pickup_times[pickup_request_index]
                    actual_pickup_time = real_time + stops_wait_time[i]
                    if actual_pickup_time - request_desired_pickup_time < 0:
                        self._log_negative_wait_time_at_station(actual_pickup_time=actual_pickup_time,
                                                                request_desired_pickup_time=request_desired_pickup_time,
                                                                i=i,
                                                                current_start_time=current_start_time,
                                                                pickup_request_index=pickup_request_index,
                                                                stops_request_pair=stops_request_pair,
                                                                stops_wait_time=stops_wait_time)
                        
                    if (actual_pickup_time - request_desired_pickup_time > self.MAX_WAIT_TIME_AT_STATION) and not planning:
                        return None

                    serviced_requests[pickup_request_index] = [current_location, actual_pickup_time]
            
            for dropoff_request_index in dropoff_requests_list:
                if dropoff_request_index != -1:
                    if dropoff_request_index in serviced_requests:
                        initial_station = serviced_requests[dropoff_request_index][0]
                        final_station = current_location
                        direct_route_time = self.map_graph.obtain_shortest_paths_time(initial_station, final_station)
                        time_in_bus_route = real_time - serviced_requests[dropoff_request_index][1]
                        if (time_in_bus_route - direct_route_time) < 0:
                            self._log_negative_wait_time_inside_bus(dropoff_request_index, stops_request_pair, stops_sequence, 
                                           stops_wait_time, initial_station, final_station, time_in_bus_route,
                                           direct_route_time, current_start_time, i)

                        if (time_in_bus_route - direct_route_time) > self.MAX_WAIT_TIME_INSIDE_BUS:
                            return None
                        
                        ptt_cost += time_in_bus_route * request_capacities[dropoff_request_index]
                        del(serviced_requests[dropoff_request_index])
            
            current_edge_cost = self.map_graph.obtain_shortest_paths_time(current_location, next_location)

            if i == 0 and bus_location != stops_sequence[0]:
                stop_wait_time = 0
            else:
                stop_wait_time = stops_wait_time[i]

            real_time += (stop_wait_time + current_edge_cost)
        
        if real_time > max_route_time:
            return None
        else:
            return ptt_cost
    
    def _calculate_cost_of_route_budget(self, current_start_time, stops_sequence, stops_wait_time, stops_request_pair, bus_location,
                                    requests_pickup_times, request_capacities, prev_passengers, passengers_in_bus, max_wait_time_at_station=MAX_WAIT_TIME_AT_STATION, 
                                    max_wait_time_inside_bus=MAX_WAIT_TIME_INSIDE_BUS, max_route_time=28800):
        
        real_time = current_start_time
        serviced_requests = copy.deepcopy(prev_passengers)
        local_passengers_in_bus = copy.deepcopy(passengers_in_bus)

        budget_cost = max_route_time - current_start_time

        for i in range(len(stops_sequence)-1):
            if i == 0 and bus_location != stops_sequence[0]:
                pickup_requests_list = []
                dropoff_requests_list = []
                current_location = bus_location
            else:
                current_request_index_dict = stops_request_pair[i]
                pickup_requests_list = current_request_index_dict["pickup"]
                dropoff_requests_list = current_request_index_dict["dropoff"]
                current_location = stops_sequence[i]
            
            next_location = stops_sequence[i+1]
            current_edge_cost = self.map_graph.obtain_shortest_paths_time(current_location, next_location)

            for dropoff_request_index in dropoff_requests_list:
                if dropoff_request_index != -1:
                    if dropoff_request_index in serviced_requests:
                        initial_station = serviced_requests[dropoff_request_index][0]
                        final_station = current_location
                        direct_route_time = self.map_graph.obtain_shortest_paths_time(initial_station, final_station)
                        time_in_bus_route = real_time - serviced_requests[dropoff_request_index][1]
                        if (time_in_bus_route - direct_route_time) < 0:
                            self._log_negative_wait_time_inside_bus(dropoff_request_index, stops_request_pair, stops_sequence, 
                                           stops_wait_time, initial_station, final_station, time_in_bus_route,
                                           direct_route_time, current_start_time, i)

                        if (time_in_bus_route - direct_route_time) > self.MAX_WAIT_TIME_INSIDE_BUS:
                            return None
                        
                        local_passengers_in_bus -= request_capacities[dropoff_request_index]
                        del(serviced_requests[dropoff_request_index])
            
            number_of_picked_up_passengers = 0

            for pickup_request_index in pickup_requests_list:
                if pickup_request_index != -1:
                    request_desired_pickup_time = requests_pickup_times[pickup_request_index]
                    actual_pickup_time = real_time + stops_wait_time[i]
                    if actual_pickup_time - request_desired_pickup_time < 0:
                        self._log_negative_wait_time_at_station(actual_pickup_time=actual_pickup_time,
                                                                request_desired_pickup_time=request_desired_pickup_time,
                                                                i=i,
                                                                current_start_time=current_start_time,
                                                                pickup_request_index=pickup_request_index,
                                                                stops_request_pair=stops_request_pair,
                                                                stops_wait_time=stops_wait_time)
                        
                    if (actual_pickup_time - request_desired_pickup_time > self.MAX_WAIT_TIME_AT_STATION):
                        return None
                    
                    number_of_picked_up_passengers += request_capacities[pickup_request_index]

                    serviced_requests[pickup_request_index] = [current_location, actual_pickup_time]

            if i == 0 and bus_location != stops_sequence[0]:
                stop_wait_time = 0
            else:
                stop_wait_time = stops_wait_time[i]

            real_time += (stop_wait_time + current_edge_cost)

            if local_passengers_in_bus > 0:
                passengers_in_bus_indicator = 1
                budget_cost -= ((current_edge_cost + stop_wait_time) * passengers_in_bus_indicator)
            else:
                if number_of_picked_up_passengers > 0:
                    passengers_in_bus_indicator = 1
                    budget_cost -= ((current_edge_cost) * passengers_in_bus_indicator)
                else:
                    budget_cost -= 0

            local_passengers_in_bus += number_of_picked_up_passengers
        
        if real_time > max_route_time:
            return None
        else:
            return -1 * budget_cost
    
    def _calculate_cost_of_route(self, current_start_time, stops_sequence, stops_wait_time, stops_request_pair, bus_location,
                                 requests_pickup_times, request_capacities, passengers_in_bus, prev_passengers, 
                                 consider_route_time=False, include_scaling=False, maximize=False, bus_capacity=20, 
                                 max_wait_time_at_station=MAX_WAIT_TIME_AT_STATION, max_wait_time_inside_bus=MAX_WAIT_TIME_INSIDE_BUS,
                                 max_route_time=28800, cost_type="wait_time", planning=False):
        
        match cost_type:
            case "wait_time":
                cost_value = self._calculate_cost_of_route_wait_time(current_start_time=current_start_time,
                                                        stops_sequence=stops_sequence,
                                                        stops_wait_time=stops_wait_time,
                                                        stops_request_pair=stops_request_pair,
                                                        bus_location=bus_location,
                                                        requests_pickup_times=requests_pickup_times,
                                                        request_capacities=request_capacities,
                                                        prev_passengers=prev_passengers,
                                                        consider_route_time=consider_route_time,
                                                        include_scaling=include_scaling,
                                                        maximize=maximize,
                                                        bus_capacity=bus_capacity,
                                                        max_wait_time_at_station=self.MAX_WAIT_TIME_AT_STATION,
                                                        max_wait_time_inside_bus=self.MAX_WAIT_TIME_INSIDE_BUS,
                                                        max_route_time=max_route_time,
                                                        planning=planning)
            case "ptt":
                cost_value = self._calculate_cost_of_route_ptt(current_start_time=current_start_time,
                                                    stops_sequence=stops_sequence,
                                                    stops_wait_time=stops_wait_time,
                                                    stops_request_pair=stops_request_pair,
                                                    bus_location=bus_location,
                                                    requests_pickup_times=requests_pickup_times,
                                                    request_capacities=request_capacities,
                                                    prev_passengers=prev_passengers,
                                                    max_wait_time_at_station=self.MAX_WAIT_TIME_AT_STATION,
                                                    max_wait_time_inside_bus=self.MAX_WAIT_TIME_INSIDE_BUS,
                                                    max_route_time=max_route_time,
                                                    planning=planning)
            case "budget":
                cost_value = self._calculate_cost_of_route_budget(current_start_time=current_start_time,
                                                    stops_sequence=stops_sequence,
                                                    stops_wait_time=stops_wait_time,
                                                    stops_request_pair=stops_request_pair,
                                                    bus_location=bus_location,
                                                    requests_pickup_times=requests_pickup_times,
                                                    request_capacities=request_capacities,
                                                    passengers_in_bus=passengers_in_bus,
                                                    prev_passengers=prev_passengers,
                                                    max_wait_time_at_station=max_wait_time_at_station,
                                                    max_wait_time_inside_bus=self.MAX_WAIT_TIME_INSIDE_BUS,
                                                    max_route_time=max_route_time)
            case _:
                raise ValueError("Cost function type not implemented")
        
        return cost_value

        
    
    def _place_request_inside_stop(self, local_stop_request_pairings, stop_index, request_index, label):
        if local_stop_request_pairings[stop_index][label][0] == -1:
            local_stop_request_pairings[stop_index][label][0] = request_index
        else:
            request_placed = False
            for current_list_index, current_request_index in enumerate(local_stop_request_pairings[stop_index][label]):
                if current_request_index == -1*request_index:
                    local_stop_request_pairings[stop_index][label][current_list_index] = request_index
                    request_placed = True
                    break

            if not request_placed:
                local_stop_request_pairings[stop_index][label].append(request_index)
    
    def _update_stop_request_pairings(self, stop_request_pairings, stop_index, request_index, pickup=False, insert=False):
        if insert:
            if pickup:
                local_stop_request_pairings = stop_request_pairings[:stop_index] + [{"pickup": [request_index], "dropoff": [-1]}] + stop_request_pairings[stop_index:]
            else:
                local_stop_request_pairings = stop_request_pairings[:stop_index] + [{"pickup": [-1], "dropoff": [request_index]}] + stop_request_pairings[stop_index:]
        else:
            local_stop_request_pairings = copy.deepcopy(stop_request_pairings)
            if pickup:
                label = "pickup"
            else:
                label = "dropoff"
            self._place_request_inside_stop(local_stop_request_pairings=local_stop_request_pairings,
                                            stop_index=stop_index,
                                            request_index=request_index,
                                            label=label)
        
        return local_stop_request_pairings
    
    def _update_stop_wait_times(self, local_travel_time, stop_index, stops_sequence, stop_request_pairings, stops_wait_time, requests_pickup_times,
                                default_stop_wait_time=15):
        new_stops_wait_time = copy.deepcopy(stops_wait_time)
        for i in range(stop_index, len(new_stops_wait_time)-1):
            current_request_index_dict = stop_request_pairings[i]
            pickup_requests_list = current_request_index_dict["pickup"]

            for list_index, current_request_index in enumerate(pickup_requests_list):
                if current_request_index == -1:
                    continue
                else:
                    if list_index == 0:
                        new_stops_wait_time[i] = default_stop_wait_time
                    current_request_pickup_time = requests_pickup_times[current_request_index]
                    new_stops_wait_time[i] = max(new_stops_wait_time[i], (current_request_pickup_time-local_travel_time)+default_stop_wait_time)
            current_location = stops_sequence[i]
            wait_time = new_stops_wait_time[i]
            next_location = stops_sequence[i+1]
            current_edge_cost = self.map_graph.obtain_shortest_paths_time(current_location, next_location)
            local_travel_time += (wait_time + current_edge_cost)
        
        current_request_index_dict = stop_request_pairings[len(new_stops_wait_time)-1]
        pickup_requests_list = current_request_index_dict["pickup"]

        for list_index, current_request_index in enumerate(pickup_requests_list):
            if current_request_index == -1:
                continue
            else:
                if list_index == 0:
                    new_stops_wait_time[len(new_stops_wait_time)-1] = default_stop_wait_time
                current_request_pickup_time = requests_pickup_times[current_request_index]
                new_stops_wait_time[len(new_stops_wait_time)-1] = max(new_stops_wait_time[len(new_stops_wait_time)-1], 
                                                                      (current_request_pickup_time-local_travel_time)+default_stop_wait_time)
        
        return new_stops_wait_time
    
    def _create_new_stop_lists(self, requests_pickup_times, new_travel_time, next_index, stops_wait_time, current_stop_wait_time, 
                               stops_sequence, request_node, request_index, 
                               stop_request_pairings, pickup=False, default_stop_wait_time=60):
        if request_node == stops_sequence[next_index]:
            local_travel_time = new_travel_time + stops_wait_time[next_index-1]
            local_stops_sequence = stops_sequence
            local_stops_wait_time = stops_wait_time
            insertion_index = next_index
            insert_flag = False
        
        elif request_node == stops_sequence[next_index-1]:
            local_travel_time = new_travel_time
            local_stops_sequence = stops_sequence
            local_stops_wait_time = stops_wait_time
            insertion_index = next_index-1
            insert_flag = False

        else:
            local_travel_time = new_travel_time + stops_wait_time[next_index-1]
            local_stops_sequence = stops_sequence[:next_index] + [request_node] + stops_sequence[next_index:]
            local_stops_wait_time = stops_wait_time[:next_index] + [current_stop_wait_time] + stops_wait_time[next_index:]
            insertion_index = next_index
            insert_flag = True

        local_stop_request_pairings = self._update_stop_request_pairings(stop_request_pairings=stop_request_pairings,
                                                                        stop_index=insertion_index,
                                                                        request_index=request_index,
                                                                        pickup=pickup,
                                                                        insert=insert_flag)

        new_stops_wait_time = self._update_stop_wait_times(local_travel_time=local_travel_time, 
                                                            stop_index=insertion_index,
                                                            stops_sequence=local_stops_sequence,
                                                            stop_request_pairings=local_stop_request_pairings,
                                                            stops_wait_time=local_stops_wait_time,
                                                            requests_pickup_times=requests_pickup_times,
                                                            default_stop_wait_time=default_stop_wait_time)

        return new_stops_wait_time, local_stops_sequence, local_stop_request_pairings, insertion_index

    def _create_new_stop_lists_online(self, requests_pickup_times, new_travel_time, next_index, stops_wait_time, current_stop_wait_time, 
                               stops_sequence, request_node, request_index, bus_location,
                               stop_request_pairings, pickup=False, default_stop_wait_time=60, mismatched_flag=False):
        
        if request_node == stops_sequence[next_index]:
            if mismatched_flag:
                local_travel_time = new_travel_time
            else:
                local_travel_time = new_travel_time + stops_wait_time[next_index-1]
            local_stops_sequence = stops_sequence
            local_stops_wait_time = stops_wait_time
            insertion_index = next_index
            insert_flag = False
        
        elif request_node == stops_sequence[next_index-1]:
            local_travel_time = new_travel_time
            if mismatched_flag:
                local_stops_sequence = stops_sequence[:next_index] + [request_node] + stops_sequence[next_index:]
                local_stops_wait_time = stops_wait_time[:next_index] + [current_stop_wait_time] + stops_wait_time[next_index:]
                insertion_index = next_index
                insert_flag = True
            else:
                local_stops_sequence = stops_sequence
                local_stops_wait_time = stops_wait_time
                insertion_index = next_index-1
                insert_flag = False

        else:
            if mismatched_flag:
                local_travel_time = new_travel_time
            else:
                local_travel_time = new_travel_time + stops_wait_time[next_index-1]
            local_stops_sequence = stops_sequence[:next_index] + [request_node] + stops_sequence[next_index:]
            local_stops_wait_time = stops_wait_time[:next_index] + [current_stop_wait_time] + stops_wait_time[next_index:]
            insertion_index = next_index
            insert_flag = True

        local_stop_request_pairings = self._update_stop_request_pairings(stop_request_pairings=stop_request_pairings,
                                                                        stop_index=insertion_index,
                                                                        request_index=request_index,
                                                                        pickup=pickup,
                                                                        insert=insert_flag)

        new_stops_wait_time = self._update_stop_wait_times(local_travel_time=local_travel_time, 
                                                            stop_index=insertion_index,
                                                            stops_sequence=local_stops_sequence,
                                                            stop_request_pairings=local_stop_request_pairings,
                                                            stops_wait_time=local_stops_wait_time,
                                                            requests_pickup_times=requests_pickup_times,
                                                            default_stop_wait_time=default_stop_wait_time)

        return new_stops_wait_time, local_stops_sequence, local_stop_request_pairings, insertion_index
    
    def _insert_pickup_in_route(self, current_start_time, total_travel_time, stops_sequence, stops_wait_time, current_location,
                                         next_index, request_origin, requests_pickup_times, stop_request_pairings, request_index, 
                                         default_stop_wait_time=60):
        time_to_pickup = self.map_graph.obtain_shortest_paths_time(current_location, request_origin)

        if total_travel_time == 0:
            time_until_request_available = max(0, requests_pickup_times[request_index]-time_to_pickup)
            new_start_time = time_until_request_available
            current_stop_wait_time = default_stop_wait_time
        else:
            new_start_time = current_start_time
            current_travel_time = total_travel_time + current_start_time + time_to_pickup + stops_wait_time[next_index - 1]
            current_request_wait_time = (requests_pickup_times[request_index] - current_travel_time) + default_stop_wait_time
            current_stop_wait_time = max(default_stop_wait_time, current_request_wait_time)
        
        new_travel_time = total_travel_time + new_start_time + time_to_pickup

        new_lists = self._create_new_stop_lists(requests_pickup_times=requests_pickup_times,
                                                new_travel_time=new_travel_time,
                                                next_index=next_index,
                                                stops_wait_time=stops_wait_time,
                                                current_stop_wait_time=current_stop_wait_time,
                                                stops_sequence=stops_sequence,
                                                request_node=request_origin,
                                                request_index=request_index,
                                                stop_request_pairings=stop_request_pairings,
                                                pickup=True,
                                                default_stop_wait_time=default_stop_wait_time)
        
        stops_wait_time_with_insertion, stop_sequence_with_insertion, stop_request_pair_with_insertion, insertion_index = new_lists

        return stop_sequence_with_insertion, stops_wait_time_with_insertion, stop_request_pair_with_insertion, new_start_time, insertion_index
    
    def _insert_pickup_in_route_online(self, current_start_time, total_travel_time, stops_sequence, stops_wait_time, current_location,
                                         next_index, request_origin, requests_pickup_times, stop_request_pairings, request_index, 
                                         default_stop_wait_time=60, mismatched_flag=False):
        time_to_pickup = self.map_graph.obtain_shortest_paths_time(current_location, request_origin)

        new_travel_time = total_travel_time + current_start_time + time_to_pickup
        current_request_wait_time = (requests_pickup_times[request_index] - new_travel_time) + default_stop_wait_time
        current_stop_wait_time = max(default_stop_wait_time, current_request_wait_time)

        new_lists = self._create_new_stop_lists_online(requests_pickup_times=requests_pickup_times,
                                                new_travel_time=new_travel_time,
                                                next_index=next_index,
                                                stops_wait_time=stops_wait_time,
                                                current_stop_wait_time=current_stop_wait_time,
                                                stops_sequence=stops_sequence,
                                                request_node=request_origin,
                                                request_index=request_index,
                                                bus_location=current_location,
                                                stop_request_pairings=stop_request_pairings,
                                                pickup=True,
                                                default_stop_wait_time=default_stop_wait_time,
                                                mismatched_flag=mismatched_flag)
        
        stops_wait_time_with_insertion, stop_sequence_with_insertion, stop_request_pair_with_insertion, insertion_index = new_lists

        return stop_sequence_with_insertion, stops_wait_time_with_insertion, stop_request_pair_with_insertion, current_start_time, insertion_index
    
    def _insert_dropoff_in_route(self, current_start_time, total_travel_time, stops_sequence, stops_wait_time, current_location,
                                next_index, request_destination, requests_pickup_times, stop_request_pairings, request_index, 
                                default_dropoff_wait_time=20, default_stop_wait_time=60):
        
        time_to_dropoff = self.map_graph.obtain_shortest_paths_time(current_location, request_destination)
        
        current_stop_wait_time = default_dropoff_wait_time

        new_travel_time = total_travel_time + current_start_time + time_to_dropoff
        
        new_lists = self._create_new_stop_lists(requests_pickup_times=requests_pickup_times,
                                                new_travel_time=new_travel_time,
                                                next_index=next_index,
                                                stops_wait_time=stops_wait_time,
                                                current_stop_wait_time=current_stop_wait_time,
                                                stops_sequence=stops_sequence,
                                                request_node=request_destination,
                                                request_index=request_index,
                                                stop_request_pairings=stop_request_pairings,
                                                pickup=False,
                                                default_stop_wait_time=default_stop_wait_time)
        
        stops_wait_time_with_insertion, stop_sequence_with_insertion, stop_request_pair_with_insertion, _ = new_lists

        return stop_sequence_with_insertion, stops_wait_time_with_insertion, stop_request_pair_with_insertion
    
    def _obtain_passengers_in_bus(self, stop_index, travel_time, bus_stops, stops_wait_time, passenger_in_bus, stop_request_pairings, 
                                                   serviced_requests, request_capacities):
        current_request_index_dict = stop_request_pairings[stop_index]
        pickup_requests_list = current_request_index_dict["pickup"]
        dropoff_requests_list = current_request_index_dict["dropoff"]

        for pickup_request_index in pickup_requests_list:
            if pickup_request_index >= 0:
                if pickup_request_index not in serviced_requests:
                    serviced_requests[pickup_request_index] = [bus_stops[stop_index], travel_time+stops_wait_time[stop_index]]
                    passenger_in_bus += request_capacities[pickup_request_index]
        
        for dropoff_request_index in dropoff_requests_list:
            if dropoff_request_index >= 0:
                if dropoff_request_index in serviced_requests:
                    del(serviced_requests[dropoff_request_index])
                    passenger_in_bus -= request_capacities[dropoff_request_index]
        
        return passenger_in_bus
    
    def _obtain_passengers_in_bus_online(self, stop_index, travel_time, bus_stops, stops_wait_time, passenger_in_bus, stop_request_pairings, 
                                                   serviced_requests, request_capacities):
        current_request_index_dict = stop_request_pairings[stop_index]
        pickup_requests_list = current_request_index_dict["pickup"]
        dropoff_requests_list = current_request_index_dict["dropoff"]

        for pickup_request_index in pickup_requests_list:
            if pickup_request_index >= 0:
                if pickup_request_index not in serviced_requests:
                    serviced_requests[pickup_request_index] = [bus_stops[stop_index], travel_time+stops_wait_time[stop_index]]
                    passenger_in_bus += request_capacities[pickup_request_index]
        
        for dropoff_request_index in dropoff_requests_list:
            if dropoff_request_index >= 0:
                if dropoff_request_index in serviced_requests:
                    del(serviced_requests[dropoff_request_index])
                    passenger_in_bus -= request_capacities[dropoff_request_index]
        
        return passenger_in_bus
    
    def _place_request_offline_exact(self, current_start_time, bus_capacity, stops_sequence, stops_wait_time, request_origin, request_destination, requests_pickup_times, 
                                     stop_request_pairings, request_index, request_capacities, consider_route_time=False, include_scaling=False,
                                     cost_type="wait_time"):
        total_travel_time = 0
        min_cost = float("inf")
        min_start_time = 0
        min_stop_sequence = []
        min_stop_wait_times = []
        min_stop_request_pairings = []
        serviced_requests = {}
        passenger_in_bus = 0
        original_route_cost = self._calculate_cost_of_route(current_start_time=current_start_time,
                                                            stops_sequence=stops_sequence, 
                                                            stops_wait_time=stops_wait_time,
                                                            stops_request_pair=stop_request_pairings,
                                                            bus_location=stops_sequence[0],
                                                            requests_pickup_times=requests_pickup_times,
                                                            request_capacities=request_capacities, 
                                                            passengers_in_bus=0,
                                                            prev_passengers={},
                                                            consider_route_time=consider_route_time,
                                                            include_scaling=include_scaling,
                                                            bus_capacity=bus_capacity,
                                                            cost_type=cost_type)
        assert original_route_cost is not None

        for i in range(len(stops_sequence)-1):
            passenger_in_bus = self._obtain_passengers_in_bus(stop_index=i, 
                                                            travel_time=total_travel_time+current_start_time,
                                                            bus_stops=stops_sequence,
                                                            stops_wait_time=stops_wait_time,
                                                            passenger_in_bus=passenger_in_bus,
                                                            stop_request_pairings=stop_request_pairings,
                                                            serviced_requests=serviced_requests,
                                                            request_capacities=request_capacities)
            current_location = stops_sequence[i]
            next_location = stops_sequence[i+1]
            next_index = i+1

            if passenger_in_bus + request_capacities[request_index] <= bus_capacity:
                deviation_result = self._insert_pickup_in_route(current_start_time=current_start_time,
                                                                total_travel_time=total_travel_time,
                                                                stops_sequence=stops_sequence,
                                                                stops_wait_time=stops_wait_time, 
                                                                current_location=current_location,
                                                                next_index=next_index,  
                                                                request_origin=request_origin, 
                                                                requests_pickup_times=requests_pickup_times,
                                                                stop_request_pairings=stop_request_pairings,
                                                                request_index=request_index)
                
                new_stop_sequence, new_stops_wait_time, new_stop_req_pair, new_start_time, insertion_index = deviation_result

                new_serviced_stops = new_stop_sequence[:insertion_index]
                new_planned_stops = new_stop_sequence[insertion_index:]
                new_serviced_stop_wait_times = new_stops_wait_time[:insertion_index]
                new_planned_stop_wait_times = new_stops_wait_time[insertion_index:]
                new_serviced_stop_req_pair = new_stop_req_pair[:insertion_index]
                new_planned_stop_req_pair =  new_stop_req_pair[insertion_index:]

                # drop_off_deviation_cost
                local_passengers_in_bus = copy.deepcopy(passenger_in_bus)
                local_serviced_requests = copy.deepcopy(serviced_requests)

                dropoff_travel_time = 0

                if insertion_index == next_index:
                    time_to_pickup = self.map_graph.obtain_shortest_paths_time(current_location, request_origin)
                    new_total_travel_time = total_travel_time + stops_wait_time[i] + time_to_pickup
                else:
                    new_total_travel_time = total_travel_time

                for j in range(len(new_planned_stops)-1):
                    full_travel_time = dropoff_travel_time + new_total_travel_time
                    new_next_index = j+1
                    new_current_location = new_planned_stops[j]
                    new_next_location = new_planned_stops[j+1]
                    
                    local_passengers_in_bus = self._obtain_passengers_in_bus(stop_index=j,
                                                                            travel_time=full_travel_time+new_start_time,
                                                                            bus_stops=new_planned_stops,
                                                                            stops_wait_time=new_planned_stop_wait_times,
                                                                            passenger_in_bus=local_passengers_in_bus,
                                                                            stop_request_pairings=new_planned_stop_req_pair,
                                                                            serviced_requests=local_serviced_requests,
                                                                            request_capacities=request_capacities)

                    total_passengers_in_bus = local_passengers_in_bus
                    if total_passengers_in_bus > bus_capacity:
                        break
                    else:
                        # deviation cost
                        destination_dev_result = self._insert_dropoff_in_route(current_start_time=new_start_time,
                                                                                total_travel_time=full_travel_time,
                                                                                stops_sequence=new_planned_stops,
                                                                                stops_wait_time=new_planned_stop_wait_times,
                                                                                current_location=new_current_location,
                                                                                next_index=new_next_index,
                                                                                request_destination=request_destination,
                                                                                requests_pickup_times=requests_pickup_times,
                                                                                stop_request_pairings=new_planned_stop_req_pair,
                                                                                request_index=request_index)

                        final_planned_stops, final_planned_stops_wait_time, final_planned_stop_req_pair = destination_dev_result

                        full_stop_sequence = new_serviced_stops + final_planned_stops
                        full_stops_wait_time = new_serviced_stop_wait_times + final_planned_stops_wait_time
                        full_stop_req_pair = new_serviced_stop_req_pair + final_planned_stop_req_pair

                        new_route_cost = self._calculate_cost_of_route(current_start_time=new_start_time,
                                                            stops_sequence=full_stop_sequence,
                                                            stops_wait_time=full_stops_wait_time,
                                                            stops_request_pair=full_stop_req_pair,
                                                            bus_location=full_stop_sequence[0],
                                                            requests_pickup_times=requests_pickup_times,
                                                            request_capacities=request_capacities,
                                                            passengers_in_bus=0,
                                                            prev_passengers={},
                                                            consider_route_time=consider_route_time,
                                                            include_scaling=include_scaling,
                                                            bus_capacity=bus_capacity,
                                                            cost_type=cost_type)
                        
                        if new_route_cost is None:
                            total_dev_cost = float("inf")
                        else:
                            total_dev_cost =  new_route_cost - original_route_cost

                        if total_dev_cost < min_cost:
                            min_cost = total_dev_cost
                            min_start_time = new_start_time
                            min_stop_sequence = full_stop_sequence
                            min_stop_wait_times = full_stops_wait_time
                            min_stop_request_pairings = full_stop_req_pair
                            
                    new_current_edge_cost = self.map_graph.obtain_shortest_paths_time(new_current_location, new_next_location)
                    dropoff_travel_time += (new_planned_stop_wait_times[j] + new_current_edge_cost)

            current_edge_cost = self.map_graph.obtain_shortest_paths_time(current_location, next_location)
            total_travel_time += (stops_wait_time[i] + current_edge_cost)

        return min_cost, min_stop_sequence, min_stop_wait_times, min_stop_request_pairings, min_start_time
    
    def _place_request_online_exact(self, current_start_time, bus_capacity, bus_location, planned_stops, stops_wait_time, request_origin, 
                                    request_destination, requests_pickup_times, stop_request_pairings, passengers_in_bus, 
                                    prev_passengers, request_index, request_capacities, bus_index, cost_type, prev_assignment_cost, 
                                    consider_route_time=False, include_scaling=False, planning=False):
        total_travel_time = 0
        min_cost = float("inf")
        min_stop_sequence = []
        min_stop_wait_times = []
        min_stop_request_pairings = []
        min_start_time = 0
        serviced_requests = copy.deepcopy(prev_passengers)
        local_passengers_in_bus = copy.deepcopy(passengers_in_bus)
        rv_list: list[tuple[int, Routing_plan]] = []

        original_route_cost = self._calculate_cost_of_route(current_start_time=current_start_time,
                                                            stops_sequence=planned_stops, 
                                                            stops_wait_time=stops_wait_time,
                                                            stops_request_pair=stop_request_pairings,
                                                            bus_location=bus_location,
                                                            requests_pickup_times=requests_pickup_times,
                                                            request_capacities=request_capacities, 
                                                            passengers_in_bus=passengers_in_bus,
                                                            prev_passengers=prev_passengers,
                                                            consider_route_time=consider_route_time,
                                                            include_scaling=include_scaling,
                                                            bus_capacity=bus_capacity,
                                                            cost_type=cost_type, 
                                                            planning=planning)
        
        if original_route_cost is None and not planning:
            return min_cost, min_stop_sequence, min_stop_wait_times, min_stop_request_pairings, min_start_time, rv_list
        
        for i in range(len(planned_stops)-1):
            if i == 0 and bus_location != planned_stops[0]:
                local_passengers_in_bus = local_passengers_in_bus
                current_location = bus_location
            else:
                local_passengers_in_bus = self._obtain_passengers_in_bus_online(stop_index=i, 
                                                                        travel_time=total_travel_time+current_start_time,
                                                                        bus_stops=planned_stops,
                                                                        stops_wait_time=stops_wait_time,
                                                                        passenger_in_bus=local_passengers_in_bus,
                                                                        stop_request_pairings=stop_request_pairings,
                                                                        serviced_requests=serviced_requests,
                                                                        request_capacities=request_capacities)
                current_location = planned_stops[i]

            next_location = planned_stops[i+1]
            next_index = i+1
            

            if local_passengers_in_bus  + request_capacities[request_index] <= bus_capacity:
                if i == 0 and bus_location != planned_stops[0]:
                    mismatched_flag = True
                else:
                    mismatched_flag = False
                
                deviation_result = self._insert_pickup_in_route_online(current_start_time=current_start_time,
                                                                total_travel_time=total_travel_time,
                                                                stops_sequence=planned_stops,
                                                                stops_wait_time=stops_wait_time, 
                                                                current_location=current_location,
                                                                next_index=next_index,
                                                                request_origin=request_origin, 
                                                                requests_pickup_times=requests_pickup_times,
                                                                stop_request_pairings=stop_request_pairings,
                                                                request_index=request_index,
                                                                mismatched_flag=mismatched_flag)
                
                new_stop_sequence, new_stops_wait_time, new_stop_req_pair, new_start_time, insertion_index = deviation_result

                new_serviced_stops = new_stop_sequence[:insertion_index]
                new_planned_stops = new_stop_sequence[insertion_index:]
                new_serviced_stop_wait_times = new_stops_wait_time[:insertion_index]
                new_planned_stop_wait_times = new_stops_wait_time[insertion_index:]
                new_serviced_stop_req_pair = new_stop_req_pair[:insertion_index]
                new_planned_stop_req_pair =  new_stop_req_pair[insertion_index:]

                # drop_off_deviation_cost
                new_local_passengers_in_bus = copy.deepcopy(local_passengers_in_bus)
                new_local_serviced_requests = copy.deepcopy(serviced_requests)

                dropoff_travel_time = 0

                if insertion_index == next_index:
                    time_to_pickup = self.map_graph.obtain_shortest_paths_time(current_location, request_origin)
                    if i == 0 and bus_location != planned_stops[0]:
                        stop_time = 0
                    else:
                        stop_time = stops_wait_time[i]
                    new_total_travel_time = total_travel_time + stop_time + time_to_pickup
                else:
                    new_total_travel_time = total_travel_time

                for j in range(len(new_planned_stops)-1):
                    new_full_travel_time = dropoff_travel_time + new_total_travel_time
                    new_next_index = j+1
                    new_current_location = new_planned_stops[j]
                    new_next_location = new_planned_stops[j+1]

                    new_local_passengers_in_bus = self._obtain_passengers_in_bus_online(stop_index=j,
                                                                                travel_time=new_full_travel_time+current_start_time,
                                                                                bus_stops=new_planned_stops,
                                                                                stops_wait_time=new_planned_stop_wait_times,
                                                                                passenger_in_bus=new_local_passengers_in_bus,
                                                                                stop_request_pairings=new_planned_stop_req_pair,
                                                                                serviced_requests=new_local_serviced_requests,
                                                                                request_capacities=request_capacities)

                    total_passengers_in_bus = new_local_passengers_in_bus
                    if total_passengers_in_bus > bus_capacity:
                        break
                    else:
                        # deviation cost
                        destination_dev_result = self._insert_dropoff_in_route(current_start_time=current_start_time,
                                                                            total_travel_time=new_full_travel_time,
                                                                            stops_sequence=new_planned_stops,
                                                                            stops_wait_time=new_planned_stop_wait_times,
                                                                            current_location=new_current_location,
                                                                            next_index=new_next_index,
                                                                            request_destination=request_destination,
                                                                            requests_pickup_times=requests_pickup_times,
                                                                            stop_request_pairings=new_planned_stop_req_pair,
                                                                            request_index=request_index)

                        final_planned_stops, final_planned_stops_wait_time, final_planned_stop_req_pair = destination_dev_result

                        full_stop_sequence = new_serviced_stops + final_planned_stops
                        full_stops_wait_time = new_serviced_stop_wait_times + final_planned_stops_wait_time
                        full_stop_req_pair = new_serviced_stop_req_pair + final_planned_stop_req_pair

                        new_route_cost = self._calculate_cost_of_route(current_start_time=current_start_time,
                                                                        stops_sequence=full_stop_sequence, 
                                                                        stops_wait_time=full_stops_wait_time,
                                                                        stops_request_pair=full_stop_req_pair,
                                                                        bus_location=bus_location,
                                                                        requests_pickup_times=requests_pickup_times,
                                                                        request_capacities=request_capacities, 
                                                                        passengers_in_bus=passengers_in_bus,
                                                                        prev_passengers=prev_passengers,
                                                                        consider_route_time=consider_route_time,
                                                                        include_scaling=include_scaling,
                                                                        bus_capacity=bus_capacity,
                                                                        cost_type=cost_type,
                                                                        planning=planning)
                        
                        if new_route_cost is None:
                            total_dev_cost = float("inf")
                        else:
                            total_dev_cost =  new_route_cost - original_route_cost

                            new_routing_plan = Routing_plan(bus_stops = full_stop_sequence,
                                stops_wait_times = full_stops_wait_time,
                                stops_request_pairing = Bus_stop_request_pairings(full_stop_req_pair),
                                assignment_cost = total_dev_cost + prev_assignment_cost,
                                newest_assignment_cost=total_dev_cost,
                                start_time = current_start_time,
                                route = [],
                                route_edge_times = [],
                                route_stop_wait_time = [])
                            
                            rv_list.append((bus_index, new_routing_plan)) 

                        if total_dev_cost < 0 and cost_type == "wait_time":
                            print("Request index = " + str(request_index))
                            print("New bus stops = " + str(full_stop_sequence))


                        if total_dev_cost < min_cost:
                            min_cost = total_dev_cost
                            min_start_time = new_start_time
                            min_stop_sequence = full_stop_sequence
                            min_stop_wait_times = full_stops_wait_time
                            min_stop_request_pairings = full_stop_req_pair

                    new_current_edge_cost = self.map_graph.obtain_shortest_paths_time(new_current_location, new_next_location)
                    dropoff_travel_time += (new_planned_stop_wait_times[j] + new_current_edge_cost)

            if i == 0 and bus_location != planned_stops[0]:
                current_wait_time = 0
            else:
                current_wait_time = stops_wait_time[i]
            current_edge_cost = self.map_graph.obtain_shortest_paths_time(current_location, next_location)
            total_travel_time +=  (current_wait_time + current_edge_cost)

        return min_cost, min_stop_sequence, min_stop_wait_times, min_stop_request_pairings, min_start_time, rv_list
    
    def static_insertion(self, current_start_time, bus_capacity, stops_sequence, stops_wait_time, stop_request_pairing, requests_pickup_times, request_capacities, 
                         request_origin, request_destination, request_index, consider_route_time=False, approximate=False, include_scaling=True):
        
        local_stops_sequence = copy.deepcopy(stops_sequence)
        local_stops_wait_time = copy.deepcopy(stops_wait_time)
        local_stop_request_pairing = copy.deepcopy(stop_request_pairing)

        deviation_result = self._place_request_offline_exact(current_start_time=current_start_time,
                                                             bus_capacity=bus_capacity,
                                                             stops_sequence=local_stops_sequence,
                                                             stops_wait_time=local_stops_wait_time,
                                                             request_origin=request_origin,
                                                             request_destination=request_destination,
                                                             requests_pickup_times=requests_pickup_times,
                                                             stop_request_pairings=local_stop_request_pairing,
                                                             request_index=request_index,
                                                             request_capacities=request_capacities,
                                                             consider_route_time=consider_route_time,
                                                             include_scaling=include_scaling)
        
        total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair, min_start_time = deviation_result

        return total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair, min_start_time
    
    def dynamic_insertion(self, current_start_time, current_stop_index, bus_capacity, passengers_in_bus, prev_passengers, bus_location,
                          stops_sequence, stops_wait_time, stop_request_pairing, request_capacities, request_origin, request_destination, 
                          requests_pickup_times, request_index, bus_index: int, cost_type, prev_assignment_cost,
                          consider_route_time=False, include_scaling=True, planning=False):
        
        local_stops_sequence = copy.deepcopy(stops_sequence[current_stop_index:])
        local_stops_wait_time = copy.deepcopy(stops_wait_time[current_stop_index:])
        local_stop_request_pairing = copy.deepcopy(stop_request_pairing[current_stop_index:])

        if len(local_stops_sequence) == 1:
            local_stops_sequence = local_stops_sequence + local_stops_sequence
            local_stops_wait_time = local_stops_wait_time + local_stops_wait_time
            local_stop_request_pairing = local_stop_request_pairing + local_stop_request_pairing

        deviation_result = self._place_request_online_exact(current_start_time=current_start_time, 
                                                            bus_capacity=bus_capacity,
                                                            bus_location=bus_location,
                                                             planned_stops=local_stops_sequence,
                                                             stops_wait_time=local_stops_wait_time,
                                                             request_origin=request_origin,
                                                             request_destination=request_destination,
                                                             requests_pickup_times=requests_pickup_times,
                                                             passengers_in_bus=passengers_in_bus,
                                                             prev_passengers=prev_passengers,
                                                             stop_request_pairings=local_stop_request_pairing,
                                                             request_index=request_index,
                                                             request_capacities=request_capacities,
                                                             bus_index = bus_index,
                                                             consider_route_time=consider_route_time,
                                                             include_scaling=include_scaling,
                                                             cost_type=cost_type,
                                                             prev_assignment_cost=prev_assignment_cost,
                                                             planning=planning)
        
        total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair, _, rv_list = deviation_result

        return total_dev_cost, full_stop_sequence, full_stops_wait_time, full_stop_req_pair, rv_list


    def unallocate(self, current_start_time, current_stop_index, passengers_in_bus, prev_passengers, bus_location,
                   stops_sequence, stops_wait_time, stop_request_pairing, request_capacities, requests_pickup_times, request_index,
                   cost_type, planning=False):

        local_stops_sequence = copy.deepcopy(stops_sequence[current_stop_index:])
        local_stops_wait_time = copy.deepcopy(stops_wait_time[current_stop_index:])
        local_stop_request_pairing = copy.deepcopy(stop_request_pairing[current_stop_index:])

        node_origin = None
        node_destination = None
        picked_up = False

        pos = 0
        while pos < len(local_stop_request_pairing):
            stop = local_stop_request_pairing[pos]
            if request_index in stop['pickup']:
                stop['pickup'].remove(request_index)
                node_origin = local_stops_sequence[pos]
                picked_up = True
                if stop['pickup'] == [] and stop['dropoff'] == [-1]:
                    local_stops_sequence.pop(pos)
                    local_stop_request_pairing.pop(pos)
                    local_stops_wait_time.pop(pos)
                elif stop['pickup'] == []:
                    stop['pickup'] = [-1]
                    pos += 1
                else:
                    pos +=1 
            elif request_index in stop['dropoff'] and picked_up:
                stop['dropoff'].remove(request_index)
                node_destination = local_stops_sequence[pos]
                if stop['pickup'] == [-1] and stop['dropoff'] == []:
                    local_stops_sequence.pop(pos)
                    local_stop_request_pairing.pop(pos)
                    local_stops_wait_time.pop(pos)
                elif stop['dropoff'] == []:
                    stop['dropoff'] = [-1]
                    pos += 1
                else:
                    pos += 1
            else:
                pos += 1

        local_stops_wait_time = self._update_stop_wait_times_unallocate(local_travel_time=current_start_time,
                                                                       stops_sequence=local_stops_sequence,
                                                                       stop_request_pairings=local_stop_request_pairing,
                                                                       stops_wait_time=local_stops_wait_time,
                                                                       requests_pickup_times=requests_pickup_times,
                                                                       current_bus_location=bus_location)
        
        original_cost = self._calculate_cost_of_route(stops_sequence = stops_sequence[current_stop_index:], 
                                                stops_wait_time = stops_wait_time[current_stop_index:], 
                                                stops_request_pair = stop_request_pairing[current_stop_index:], 
                                                request_capacities = request_capacities, 
                                                passengers_in_bus = passengers_in_bus, 
                                                current_start_time = current_start_time, 
                                                prev_passengers = prev_passengers, 
                                                bus_location = bus_location, 
                                                requests_pickup_times = requests_pickup_times,
                                                cost_type=cost_type,
                                                planning=planning)
        
        new_cost = self._calculate_cost_of_route(stops_sequence = local_stops_sequence, 
                                                stops_wait_time = local_stops_wait_time, 
                                                stops_request_pair = local_stop_request_pairing, 
                                                request_capacities = request_capacities, 
                                                passengers_in_bus = passengers_in_bus, 
                                                current_start_time = current_start_time, 
                                                prev_passengers = prev_passengers, 
                                                bus_location = bus_location, 
                                                requests_pickup_times = requests_pickup_times,
                                                cost_type=cost_type,
                                                planning=planning)
        if (node_origin is None) or (node_destination is None) or (new_cost is None) or (original_cost is None):
            return None
        else:
            total_dev_cost = new_cost - original_cost
            return total_dev_cost, node_origin, node_destination, local_stops_sequence, local_stops_wait_time, local_stop_request_pairing
            

    def _update_stop_wait_times_unallocate(self, local_travel_time, stops_sequence, 
                                        stop_request_pairings, stops_wait_time, requests_pickup_times,
                                        current_bus_location: int,
                            default_stop_wait_time=60):
    
        new_stops_wait_time = copy.deepcopy(stops_wait_time)
        for i in range(len(new_stops_wait_time)-1):
            if current_bus_location != stops_sequence[0] and i == 0:
                next_location = stops_sequence[i+1]
                current_edge_cost = self.map_graph.obtain_shortest_paths_time(current_bus_location, next_location)
                local_travel_time += current_edge_cost
            else:
                current_request_index_dict = stop_request_pairings[i]
                pickup_requests_list = current_request_index_dict["pickup"]
                for list_index, current_request_index in enumerate(pickup_requests_list):
                    if current_request_index == -1:
                        continue
                    else:
                        if list_index == 0:
                            new_stops_wait_time[i] = default_stop_wait_time
                        current_request_pickup_time = requests_pickup_times[current_request_index]
                        new_stops_wait_time[i] = max(new_stops_wait_time[i], (current_request_pickup_time-local_travel_time)+default_stop_wait_time)
                current_location = stops_sequence[i]
                wait_time = new_stops_wait_time[i]
                next_location = stops_sequence[i+1]
                current_edge_cost = self.map_graph.obtain_shortest_paths_time(current_location, next_location)
                local_travel_time += (wait_time + current_edge_cost)
        
        current_request_index_dict = stop_request_pairings[len(new_stops_wait_time)-1]
        pickup_requests_list = current_request_index_dict["pickup"]
        for list_index, current_request_index in enumerate(pickup_requests_list):
            if current_request_index == -1:
                continue
            else:
                if list_index == 0:
                    new_stops_wait_time[len(new_stops_wait_time)-1] = default_stop_wait_time
                current_request_pickup_time = requests_pickup_times[current_request_index]
                new_stops_wait_time[len(new_stops_wait_time)-1] = max(new_stops_wait_time[len(new_stops_wait_time)-1], 
                                                                      (current_request_pickup_time-local_travel_time)+default_stop_wait_time)
        
        return new_stops_wait_time