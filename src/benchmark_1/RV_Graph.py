from typing import Generator

from Data_structures import Config_flags, Routing_plan
from MCTS_state import MCTS_state
from benchmark_1.new_DS import SimRequest

class RVGraph:

    def __init__(self, 
                 mcts_state_object: MCTS_state, 
                 request: SimRequest, 
                 assignment_policy,
                 config_flags: Config_flags,
                 planning: bool) -> None:
        
        self.config_flags = config_flags
        self.request = request
        self.E_RV: list[list[Routing_plan]] = [[] for _ in range(self.config_flags.num_buses)]

        all_possible_rv_paths: list[tuple[int, Routing_plan]]
        if not hasattr(request, 'index'):
            print('no index')
        _, _, _, all_possible_rv_paths = assignment_policy(state_object=mcts_state_object, 
                                                        current_request_index=request.index, 
                                                        current_request_row={"Origin Node": request.origin,
                                                                           "Destination Node": request.destination},
                                                        planning=planning)
        for (bus_index, possible_routing_plan) in all_possible_rv_paths:
            self.E_RV[bus_index].append(possible_routing_plan)

    def edge_iterator(self) -> Generator[tuple[int, Routing_plan], None, None]:
        if self.E_RV:
            for vehicle_id, edges in enumerate(self.E_RV):
                for route in edges:
                    yield vehicle_id, route
        else:
            raise ValueError('RVGraph is empty')
        
    def get_min_PTT_edge(self) -> tuple[int, Routing_plan] | None:
        mins = [(bus_num, min(self.E_RV[bus_num], key=lambda x: x.newest_assignment_cost)) 
                for bus_num in range(self.config_flags.num_buses) 
                if self.E_RV[bus_num]]
        if not mins:
            return None
        else:
            min_bus, min_route = min(mins, key=lambda x: x[1].newest_assignment_cost)
            return min_bus, min_route

    def  __repr__(self) -> str:
        out = '-------RV GRAPH------\n'
        for bus_num, routes in enumerate(self.E_RV):
            out += f'BUS {bus_num} \n'
            for route in routes:
                out += f'cost {route.assignment_cost} with routing plan {hash(route)}\n'
        return out