import copy
import math
import random
import numpy as np
import pandas as pd
import multiprocessing
from typing import Optional
from collections import deque
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed

from MCTS_state import MCTS_state
from Request_handler import Request_handler
from benchmark_1.new_DS import SimRequest, SimRequestChain
from benchmark_1.RV_Graph import RVGraph
from benchmark_1.algo1 import Actions
from Data_structures import Routing_plan, Date_operational_range, Config_flags
import random

NUM_GENERATED_CSV = 199

@dataclass
class MCNode:
    request: SimRequest | None
    mcts_state_object: MCTS_state
    config_flags: Config_flags
    parent: 'Optional[MCNode]' = None
    children: list['MCNode'] = field(default_factory=list)
    visits: int = 1
    reward: list[float] = field(default_factory=list)

    def compute_ucb(self) -> float:
        '''update visits, avg_value and avg in this order'''
        parent = self.parent.visits if self.parent else 1
        current = self.visits
        freq_term = (math.sqrt(math.log(parent) / current))
        min_val = min(self.reward)
        return min_val - self.config_flags.MCTS_TUNING_PARAM * freq_term

    def update_visits(self) -> None:
        self.visits += 1

    def select_best_child(self) -> 'MCNode':
        if self.children:
            return min(self.children, key=lambda x: x.compute_ucb())
        else:
            raise ValueError('There are no children in this list')

    def append_child(self, node: 'MCNode') -> None:
        self.children.append(node)
    
    def __repr__(self) -> str:
        return f'''rew={self.compute_ucb()}, all_rews={min(self.reward), self.reward}, visits={self.visits}, parent={self.parent.request.index if self.parent else None}, children={[child.request.index for child in self.children] if self.children else None}, req={self.request.index}, bus_path={[path.stops_request_pairing for path in self.mcts_state_object.bus_fleet.routing_plans]}'''

class MCTree:
    def __init__(self,
                 initial_request: SimRequest,
                 unallocate_method_vv,
                 greedy_assignment_rv,
                 create_routes_method,
                 slicing_method,
                 mcts_state_object: MCTS_state,
                 config_flags: Config_flags) -> None:

        self.config_flags = config_flags
        self.unallocate_method_vv = unallocate_method_vv
        self.greedy_assignment_rv = greedy_assignment_rv
        self.create_routes_method = create_routes_method
        self.slicing_method = slicing_method
        self.root: MCNode = MCNode(request=initial_request, 
                                   mcts_state_object=copy.deepcopy(mcts_state_object),
                                   config_flags=self.config_flags)

    def select(self, node: MCNode | None = None, depth=0) -> tuple[int, MCNode]:
        curr_node = node if node else self.root
        curr_node.visits += 1
        if curr_node.children and depth <= self.config_flags.MCTS_DEPTH:
            return self.select(curr_node.select_best_child(), depth+1)
        else:
            return depth, curr_node
    
    def _move_time_forward(self, mcts_state_object: MCTS_state, current_request: SimRequest):
        while mcts_state_object.state_num < current_request.pickup_time - 120:
            mcts_state_object.next_state()
    
    def _update_mcts_state(self, bus_index: int, mcts_state_object: MCTS_state, new_routing_plan: Routing_plan):
        current_step_index = mcts_state_object.step_index[bus_index]
        current_stop_wait_time = mcts_state_object.bus_fleet.routing_plans[bus_index].route_stop_wait_time[current_step_index]
        if mcts_state_object.wait_time_at_the_station[bus_index] < current_stop_wait_time:
            current_bus_location=mcts_state_object.bus_fleet.routing_plans[bus_index].route[mcts_state_object.step_index[bus_index]]
        else:
            current_bus_location=mcts_state_object.bus_fleet.routing_plans[bus_index].route[mcts_state_object.step_index[bus_index]+1]

        route, bus_route_edge_time, routes_stop_wait_time = self.create_routes_method(current_bus_location=current_bus_location, 
                                                                                        bus_stops=new_routing_plan.bus_stops, 
                                                                                        stops_wait_times=new_routing_plan.stops_wait_times)
        
        new_routing_plan.update_routes(route=route,
                                    route_edge_times=bus_route_edge_time,
                                    route_stop_wait_time=routes_stop_wait_time)
        
        mcts_state_object.update_state(bus_index=bus_index, 
                                       request_index=0,
                                       new_routing_plan=new_routing_plan)


    def expand(self, current_node: MCNode, future_generated_request: SimRequest | None,
               planning: bool) -> None:
        new_mcts_state_object = copy.deepcopy(current_node.mcts_state_object)
        if planning:
            self._move_time_forward(mcts_state_object = new_mcts_state_object,
                                    current_request = current_node.request)
        best_actions = Actions(mcts_state_object = new_mcts_state_object,
                               unallocate_method_vv = self.unallocate_method_vv,
                               greedy_assignment_rv = self.greedy_assignment_rv,
                               slicing_method = self.slicing_method,
                               request = current_node.request,
                               config_flags = self.config_flags,
                               planning=planning)
        
        if best_actions.promising:
            for promising_routing_plans in best_actions.promising:
                promising_mcts_state_object = copy.deepcopy(new_mcts_state_object)
                for bus_index in range(self.config_flags.num_buses):
                    self._update_mcts_state(bus_index=bus_index,
                                            mcts_state_object=promising_mcts_state_object,
                                            new_routing_plan=promising_routing_plans[bus_index])
                    
                new_node = MCNode(request = future_generated_request,
                                mcts_state_object=promising_mcts_state_object,
                                config_flags = self.config_flags,
                                parent = current_node,
                                visits = 1,
                                reward = [0])
                current_node.append_child(new_node)
        else:
            new_node = MCNode(request = future_generated_request,
                              mcts_state_object=new_mcts_state_object,
                              parent = current_node,
                              visits = 1,
                              reward = [self.config_flags.MAX_ROUTE_TIME],
                              config_flags = self.config_flags)
            current_node.append_child(new_node)
            
    def rollout(self, mcts_state_object: MCTS_state, requests_chain: SimRequestChain | None,
                planning: bool) -> int:
        if (requests_chain is None) or requests_chain.reached_end():
            total = sum(routing_plan.assignment_cost for routing_plan in mcts_state_object.bus_fleet.routing_plans)
            return total
        else:
            self._move_time_forward(mcts_state_object=mcts_state_object,
                                    current_request=requests_chain.chain[0])
            out_rv = RVGraph(mcts_state_object=mcts_state_object,
                             request=requests_chain.chain[0],
                             assignment_policy=self.greedy_assignment_rv,
                             config_flags = self.config_flags,
                             planning=planning).get_min_PTT_edge()
            if not out_rv:
                final_cost = self.rollout(mcts_state_object=mcts_state_object,
                                          requests_chain=SimRequestChain(requests_chain.chain[1:]),
                                          planning=planning)
                return final_cost + self.config_flags.MAX_ROUTE_TIME
            else:
                min_bus_index, min_routing_plan = out_rv
                self._update_mcts_state(bus_index=min_bus_index,
                                        mcts_state_object=mcts_state_object,
                                        new_routing_plan=min_routing_plan)
                
                final_cost = self.rollout(mcts_state_object=mcts_state_object,
                                          requests_chain=SimRequestChain(requests_chain.chain[1:]),
                                          planning=planning)
                return final_cost

    def backpropagate(self, value: float, node: MCNode) -> None:
        if node:
            node.reward.append(value)
            if node.parent:
                self.backpropagate(value, node.parent)

    def __repr__(self) -> str:
        level = deque(self.root.children)
        empty = '(-, -, -)'
        out = '->0lvl | ' + '      ' * int(2**(self.config_flags.MCTS_DEPTH - 1))
        out += f'({self.root.request.index}, {self.root.reward}, {int(self.root.visits):d}), {hash(tuple(self.root.buses_paths))}' + '\n'
        lvl_ind = 1
        while any(val != empty for val in level):
            out += f'->{lvl_ind}lvl | ' 
            next_level = deque([])
            while level:
                child = level.popleft()
                out += '      ' * int(2**(self.config_flags.MCTS_DEPTH - lvl_ind - 1))
                if child != empty:
                    out += f'({child.reward}, {int(child.visits):d}, {hash(tuple(child.buses_paths))}) '

                    if child.children:
                        next_level.extend(child.children)
                    next_level.extend([empty for _ in range(self.config_flags.K_MAX - len(child.children))])
                else:
                    out += empty
                    next_level.extend([empty for _ in range(self.config_flags.K_MAX)])
                out += '      ' * int(2**(self.config_flags.MCTS_DEPTH - lvl_ind - 1)-1)
            out += '\n'
            lvl_ind += 1 
            level = next_level
        return out


class MCForest:

    def __init__(self,
                 initial_request: SimRequest,
                 unallocate_method_vv,
                 greedy_assignment_rv,
                 create_routes_method,
                 slicing_method,
                 config_flags, 
                 data_folders, 
                 mcts_state_object: MCTS_state) -> None:

        self.initial_request = initial_request
        self.unallocate_method_vv = unallocate_method_vv
        self.greedy_assignment_rv = greedy_assignment_rv
        self.create_routes_method = create_routes_method
        self.slicing_method = slicing_method
        self.mcts_state_object = mcts_state_object
        self.config_flags = config_flags 
        self.data_folders = data_folders 
        self.date_operational_range = mcts_state_object.date_operational_range

    def _evaluate_tree(self) -> MCTree:
        mcts_sampled_requests = self.prepare_requests()
        requests = SimRequestChain(mcts_sampled_requests)
        root = dummy = MCTree(initial_request=self.initial_request,
                              unallocate_method_vv=self.unallocate_method_vv,
                              greedy_assignment_rv=self.greedy_assignment_rv,
                              create_routes_method=self.create_routes_method,
                              slicing_method=self.slicing_method,
                              mcts_state_object=copy.deepcopy(self.mcts_state_object),
                              config_flags=self.config_flags)
        time_start = datetime.now()
        for _ in range(self.config_flags.SINGLE_MCTREE_ITERATIONS):
            current_time = datetime.now()
            time_difference = (current_time - time_start).total_seconds()
            if time_difference <= self.config_flags.TIMEOUT:
                dummy = root
                depth, selected_node = dummy.select()
                if selected_node.request:
                    if depth == self.config_flags.MCTS_DEPTH + 1:
                        one_node_up = selected_node.parent 
                        if one_node_up:
                            dummy.backpropagate(selected_node.reward[0], one_node_up)
                    else:
                        if depth == 0:
                            planning = False
                        else:
                            planning = True
                        next_requests = requests.from_depth(depth)
                        dummy.expand(selected_node, next_requests, planning=planning)
                        if self.unable_to_allocate_first_request_in_expand(depth, selected_node):
                            return None
                        for child_node in selected_node.children:
                            if child_node.request is None:
                                rollout_cost = dummy.rollout(copy.deepcopy(child_node.mcts_state_object),
                                                            None,
                                                            planning=planning)
                            else:
                                rollout_cost = dummy.rollout(copy.deepcopy(child_node.mcts_state_object),
                                                            SimRequestChain(requests.chain[depth:]),
                                                            planning=planning)
                            child_node.reward[0] += rollout_cost
                            dummy.backpropagate(rollout_cost, selected_node)
                else:
                    one_node_up = selected_node.parent 
                    if one_node_up:
                        dummy.backpropagate(selected_node.reward[0], one_node_up)
            else:
                break
        return root

    def parallel_evaluate_trees(self):
        num_cores = multiprocessing.cpu_count()
        workers = min(self.config_flags.N_CHAINS, num_cores)
        results = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_tree = {executor.submit(self._evaluate_tree): i for i in range(self.config_flags.N_CHAINS)}
            for future in as_completed(future_to_tree):
                tree = future.result()
                if tree is not None:
                    results.append(tree)
        return results

    def unable_to_allocate_first_request_in_expand(self, depth, selected_node):
        return (depth == 0 
                and len(selected_node.children) == 1 
                and selected_node.children[0].reward[0] == self.config_flags.MAX_ROUTE_TIME)

    def get_best_action(self) -> list[Routing_plan]:

        actions: list[tuple[list[Routing_plan], float, int]] = [[None for i in range(3)] for i in range(self.config_flags.K_MAX)]  # tuple[routing_plan, cost, visits]
        if self.config_flags.N_CHAINS > 1 and self.config_flags.IMPERFECT_GENERATIVE_MODEL:
            results = self.parallel_evaluate_trees()
            if results:
                for evaluated_tree in results:
                    for index, mc_node in enumerate(evaluated_tree.root.children):
                        old_cost = actions[index][1] or 0
                        old_visits = actions[index][2] or 0
                        new_visits = old_visits + 1
                        min_val = min(mc_node.reward)
                        new_cost = (old_cost * old_visits + min_val) / new_visits
                        actions[index] = (self.slicing_method(current_state_object=evaluated_tree.root.mcts_state_object,
                                                          next_state_object=mc_node.mcts_state_object), new_cost, new_visits)
            else:
                return None
        else:
            evaluated_tree = self._evaluate_tree()
            if evaluated_tree:
                for index, mc_node in enumerate(evaluated_tree.root.children):
                    old_cost = actions[index][1] or 0
                    old_visits = actions[index][2] or 0
                    new_visits = old_visits + 1
                    min_val = min(mc_node.reward)
                    new_cost = (old_cost * old_visits + min_val) / new_visits
                    actions[index] = (self.slicing_method(current_state_object=evaluated_tree.root.mcts_state_object,
                                                          next_state_object=mc_node.mcts_state_object), new_cost, new_visits)
            else:
                return None 
        best_action = min((x for x in actions if x[0]), key=lambda x: x[1])[0]
        return best_action

    def prepare_requests(self):
        request_handler = Request_handler(data_folders=self.data_folders, 
                                          process_requests=self.config_flags.process_requests)
        testing_dates = self.get_testing_days(request_handler)
        if self.config_flags.IMPERFECT_GENERATIVE_MODEL:
            if self.config_flags.BETTER_GEN_MODEL:
                #here load on the generated chains 
                sampled_requests = self.get_predicted_requests_gen_model(request_handler, self.config_flags)
                sampled_requests['time'] = sampled_requests['Requests Pickup Times Timestamped']
                sampled_requests = sampled_requests[sampled_requests['time'] > self.initial_request.pickup_time]
            else:
                sampled_requests = self.generate_sampled_chains(request_handler, testing_dates)
                sampled_requests['time'] = sampled_requests['Requests Pickup Times Timestamped']
                sampled_requests = sampled_requests[sampled_requests['time'] > self.initial_request.pickup_time]
        else:
            sampled_requests = self.generate_MCTS_perfect_predictions(self.mcts_state_object, request_handler)
            if self.initial_request.index in sampled_requests.index:
                position = sampled_requests.index.get_loc(self.initial_request.index)
                sampled_requests = sampled_requests.iloc[position + 1:]
            else:
                sampled_requests = sampled_requests
                
        self.mcts_state_object.request_capacities |= {req['row_hash'] : req['Number of Passengers'] 
                                                 for _,req in sampled_requests.iterrows()}            
        self.mcts_state_object.requests_pickup_times |= {req['row_hash'] : req['Requests Pickup Times Timestamped'] 
                                                    for _,req in sampled_requests.iterrows()} 
        
        chain_lenght = self.config_flags.MCTS_DEPTH
        
        if sampled_requests.empty:
            mcts_sampled_requests = []
        else:
            next_pd_rows = sampled_requests.head(min(chain_lenght, len(sampled_requests)))        
            mcts_sampled_requests = []
            for _, future_requests in next_pd_rows.iterrows():
                mcts_sampled_requests.append(SimRequest(origin=future_requests['Origin Node'], 
                                                destination=future_requests['Destination Node'], 
                                                index=future_requests['row_hash'],  
                                                pickup_time=future_requests['Requests Pickup Times Timestamped']))
                                                 
        return mcts_sampled_requests
            
    def get_testing_days(self, request_handler):
        random.seed(42)
        np.random.seed(42)
        unique_date_elements = list(request_handler.online_requests_df["Request Creation Date"].unique())
        date_series = pd.to_datetime(pd.Series(unique_date_elements))
        grouped = date_series.groupby([date_series.dt.year, date_series.dt.month])
        def select_two_random(group):
            if len(group) < 2:
                return group
            return group.sample(n=2, random_state=42)
        selected_dates = grouped.apply(select_two_random)
        selected_dates = selected_dates.reset_index(drop=True)
        if self.config_flags.SAVE_TEST_DAYS:
            selected_dates.to_csv("test_days.csv", index=False)
        return selected_dates.tolist()

    def generate_MCTS_perfect_predictions(self, state_object: MCTS_state, request_handler: Request_handler):
        all_requests = request_handler.get_requests_for_given_date_and_hour_range(state_object.date_operational_range)[1] 
        all_requests['Requests Pickup Times Timestamped'] = all_requests['Requested Pickup Time'].apply(lambda x: self.generate_pickup_time(x))
        all_requests = all_requests[['Number of Passengers', 'Origin Node', 
                                     'Destination Node', 'Requested Pickup Time',
                                     'Requests Pickup Times Timestamped']]
        all_requests['row_hash'] = all_requests.index
        return all_requests
    
    def generate_sampled_chains(self, request_handler: Request_handler, testing_dates):
        all_requests = request_handler.online_requests_df[~request_handler.online_requests_df["Request Creation Date"].isin(testing_dates)]
        if self.config_flags.RECOMPUTE_MEAN_STD:
            dates_to_sample = all_requests["Request Creation Date"].unique()
            out = []
            for date in dates_to_sample:
                date_range = Date_operational_range(year = date.year,
                                       month = date.month,
                                       day = date.day,
                                       start_hour = self.date_operational_range.start_hour,
                                       end_hour = self.date_operational_range.end_hour)
                out.append(len(request_handler.get_requests_for_given_date_and_hour_range(date_range)[1]))
            mean = np.mean(out)
            std = np.std(out)
        else:
            mean = 125.73
            std = 71.22
        all_requests = all_requests[all_requests['Requested Pickup Time'].apply(lambda x: x.hour >= self.date_operational_range.start_hour or x.hour < self.date_operational_range.end_hour)]
        num_requests_to_sample = int(np.random.normal(mean, std, 1)[0])
        num_requests_to_sample = max(0, num_requests_to_sample)
        sampled_requests = all_requests.sample(n=num_requests_to_sample, replace=True, random_state=42)
        sampled_requests['Requests Pickup Times Timestamped'] = sampled_requests['Requested Pickup Time'].apply(lambda x: self.generate_pickup_time(x))
        sampled_requests = sampled_requests.sort_values(by=['Requests Pickup Times Timestamped'])
        sampled_requests = sampled_requests[['Number of Passengers', 'Origin Node', 
                                                'Destination Node', 'Requested Pickup Time',
                                                'Requests Pickup Times Timestamped']]
        sampled_requests = sampled_requests.reset_index(drop=True)
        sampled_requests['row_hash'] = sampled_requests.apply(lambda x: hash(tuple(x)), axis=1)
        return sampled_requests

    def get_predicted_requests_gen_model(self, request_handler: Request_handler, config_flags: Config_flags):
        #read csv
        csv_date = f'{config_flags.year}-{config_flags.month:02d}-{config_flags.day:02d}'
        choose_chain = random.randint(0, NUM_GENERATED_CSV)
        predicted_gen_requests = pd.read_csv(f'data/igor_generative_model_data/output_gen_model_{csv_date}_{choose_chain}.csv')
        predicted_gen_requests['Requested Pickup Time'] = pd.to_datetime(predicted_gen_requests['Requested Pickup Time'])
        origin_nodes, destination_nodes = request_handler._get_nodes_from_coordinates(dataframe=predicted_gen_requests)
        predicted_gen_requests["Origin Node"] = origin_nodes
        predicted_gen_requests["Destination Node"] = destination_nodes
        predicted_gen_requests['Requests Pickup Times Timestamped'] = predicted_gen_requests['Requested Pickup Time'].apply(lambda x: self.generate_pickup_time(x))
        predicted_gen_requests = predicted_gen_requests[['Requested Pickup Time', 'Number of Passengers',  'Origin Node',  'Destination Node',  'Requests Pickup Times Timestamped']]
        predicted_gen_requests = predicted_gen_requests.reset_index(drop=True)
        predicted_gen_requests['row_hash'] = predicted_gen_requests.apply(lambda x: hash(tuple(x)), axis=1)
        return predicted_gen_requests
        
    
    def generate_pickup_time(self, pickup_datetime: datetime):
        if pickup_datetime.hour < self.mcts_state_object.date_operational_range.start_hour:
            new_pickup_time = (((((pickup_datetime.hour + 24) - self.mcts_state_object.date_operational_range.start_hour) * 60) \
                                    + pickup_datetime.minute) * 60) + pickup_datetime.second
        else:
            new_pickup_time = ((((pickup_datetime.hour - self.mcts_state_object.date_operational_range.start_hour) * 60) \
                                    + pickup_datetime.minute) * 60) + pickup_datetime.second
        return new_pickup_time