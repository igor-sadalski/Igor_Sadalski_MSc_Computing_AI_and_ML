import copy

from MCTS_state import MCTS_state
from benchmark_1.RV_Graph import RVGraph
from benchmark_1.VV_Graph import VVGraph
from benchmark_1.new_DS import SimRequest
from Data_structures import Config_flags, Routing_plan

class Actions:
    def __init__(self, 
                 mcts_state_object: MCTS_state,
                 unallocate_method_vv,
                 greedy_assignment_rv,
                 slicing_method,
                 request: SimRequest,
                 config_flags: Config_flags,
                 planning: bool) -> None:
        
        self.config_flags = config_flags

        self.rv_graph = RVGraph(mcts_state_object=mcts_state_object,
                                request=request, 
                                assignment_policy=greedy_assignment_rv,
                                config_flags=self.config_flags,
                                planning=planning)
        
        self.vv_graph = VVGraph(mcts_state_object=mcts_state_object,
                                unallocate_method=unallocate_method_vv,
                                config_flags=self.config_flags,
                                planning=planning)
        self.request = request
        self.theta = slicing_method(current_state_object=mcts_state_object,
                                    next_state_object=mcts_state_object)
        self.heap: list[tuple[int, list[Routing_plan]]] = []
        self.promising = self._algo1()

    def _algo1(self) -> list[list[Routing_plan]] | None:
        if self.rv_is_empty():
            print("No insertion possible")
            return None
        for vehicle_id, er_ij_path in self.rv_graph.edge_iterator():
            updated_theta_ij = copy.deepcopy(self.theta)
            updated_theta_ij[vehicle_id] = er_ij_path
            u_x = sum(routing_plan.assignment_cost for routing_plan in updated_theta_ij)
            self.heap.append((u_x, copy.deepcopy(updated_theta_ij)))
            vv_copy = copy.deepcopy(self.vv_graph)
            vv_copy.delete_vertex(vehicle_id)
            while vv_copy.has_values():
                m, n, vv_edge = vv_copy.arg_min()
                u_x += vv_edge.swap_utility
                updated_theta_ij[m] = vv_edge.bus_m_route
                updated_theta_ij[n] = vv_edge.bus_n_route
                self.heap.append((u_x, copy.deepcopy(updated_theta_ij)))
                vv_copy.delete_vertex(m, n)
        self.heap.sort(key=lambda x: x[0])
        smallest_tuples = self.heap[:min(self.config_flags.K_MAX, len(self.heap))]
        k_smallest = [tupl[1] for tupl in smallest_tuples]
        return k_smallest

    def rv_is_empty(self) -> bool:
        return all(not sublist for sublist in self.rv_graph.E_RV)