from dataclasses import dataclass
from typing import NamedTuple
from datetime import datetime

@dataclass
class SimRequest:
    origin: int
    destination: int
    index: int
    pickup_time: datetime | None = None

    def __repr__(self) -> str:
        return f'req={self.index}'

    def __hash__(self):
        return hash((self.origin, self.destination, self.index))

class SimRequestChain(NamedTuple):
    chain: list[SimRequest]

    def reached_end(self) -> bool:
        return self.chain == []
    
    def from_depth(self, depth: int) -> SimRequest | None:
        if len(self.chain) > depth:
            return self.chain[depth]
        else:
            return None

    def __getitem__(self, index) -> 'SimRequestChain | SimRequest':
        if isinstance(index, slice):
            return SimRequestChain(self.chain[index])
        else:
            return self.chain[index]
    
    def __hash__(self):
        return hash(tuple(self.chain))
    
    def append_left(self, new_request: SimRequest) -> None:
        return self.chain.insert(0, new_request)

    def __repr__(self) -> str:
        return 'SimReqChain' + str(self.chain)
