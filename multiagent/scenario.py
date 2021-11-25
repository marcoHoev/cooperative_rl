import json

import numpy as np


# defines scenario upon which the world is built
class BaseScenario(object):

    def __init__(self, kwargs=None):
        if kwargs is not None:
            for key in kwargs:
                setattr(self, key, kwargs[key])

    # create elements of the world
    def make_world(self):
        raise NotImplementedError()

    # create initial conditions of the world
    def reset_world(self, world):
        raise NotImplementedError()

    def to_json(self):
        return json.dumps(self.__dict__)
