import pandas as pd

class LoggerDict:
    def __init__(self):
        self.dict = {}

    def __getitem__(self, key):
        return self.dict.get(key)

    def __setitem__(self, key, value):
        self.dict[key] = value

    def get(self, key, default=None):
        return self.dict.get(key, default)
    
    #add support gfor .item() on dict
    def items(self):
        return self.dict.items()
    
    # add support for .values() on dict
    def values(self):
        return self.dict.values()

logger_dict = LoggerDict()