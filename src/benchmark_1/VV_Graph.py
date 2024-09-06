from dataclasses import dataclass
from Data_structures import Routing_plan
from MCTS_state import MCTS_state

@dataclass
class VVEdge:
    bus_m_route: Routing_plan
    bus_n_route: Routing_plan
    swap_utility: int

    def __repr__(self) -> str:
        return f'VVEdge(swp_$:{hash(self.swap_utility)}, m:{hash(self.bus_m_route)}, n:{hash(self.bus_n_route)})'

class AdjacencyList:
    def __init__(self, config_flags) -> None:
        self.config_flags = config_flags
        self.adjacency_list: dict[int, dict[int, VVEdge | None]] = {i: {j: None for j in range(self.config_flags.num_buses) if j != i} 
                                                             for i in range(self.config_flags.num_buses)} 

    def insert_edge_min(self, m: int, n: int, edge: VVEdge) -> None:
        self.adjacency_list[m][n] = edge

    def delete_vertex(self, *vehicles: int) -> None:
        for vehicle in vehicles:
            self.adjacency_list.pop(vehicle)
            for vehicle_key in self.adjacency_list: 
                self.adjacency_list[vehicle_key].pop(vehicle)

    def min_global_edge(self) -> tuple[int, int, VVEdge]:
        return min(((m, n, value) for m, sub_dict in self.adjacency_list.items()
                                  for n, value in sub_dict.items()
                                  if value is not None),
                                  key=lambda x: x[2].swap_utility)

    def __repr__(self) -> str:
        out = '-----VV GRAPH-----\n'
        for m, sub_dict in self.adjacency_list.items():
            for n, edge in sub_dict.items():
                out += f'm={m} -> n={n}: {edge}\n'
        return out


class VVGraph:

    def __init__(self, 
                 mcts_state_object: MCTS_state,
                 unallocate_method,
                 config_flags,
                 planning: bool) -> None:
        
        self.config_flags = config_flags
        self.E_VV = AdjacencyList(config_flags)
        bus_gen = ((m, n) for m in range(self.config_flags.num_buses)
                          for n in range(self.config_flags.num_buses)
                          if m != n)

        for m, n in bus_gen:
            vv_edge = unallocate_method(m=m,
                                        n=n,
                                        state_object=mcts_state_object,
                                        planning=planning)
            if vv_edge:
                self.E_VV.insert_edge_min(m, n, vv_edge)
        
    def has_values(self) -> bool:
        return any(val for inner_dict in self.E_VV.adjacency_list.values() 
                       for val in inner_dict.values())

    def arg_min(self) -> tuple[int, int, VVEdge]:
        bus_m, bus_n, edge = self.E_VV.min_global_edge()
        return  bus_m, bus_n, edge

    def delete_vertex(self, *vehicles: int) -> None:
        self.E_VV.delete_vertex(*vehicles)
    
    def __repr__(self) -> str:
        return repr(self.E_VV)