import random
import itertools

    
class MapObj:
    colors = None
    states = None

    def __init__(self, map_coloring,are_adjacent_function=None):
        self.map_coloring = map_coloring
        self.are_adjacent_function = are_adjacent_function
    
    def get_coloring(self):
        return self.map_coloring
    
    def set_color(self,state,color):
        self.map_coloring[state] = color
            
    def get_color(self,state):
        return self.map_coloring[state]
        
    def get_states(self):
        return self.states
        
    def get_colors(self):
        return self.colors

    def is_valid_coloring(self):
        for state_a, state_b in itertools.combinations(MapObj.states, 2):
            color_a = self.map_coloring[state_a]
            color_b = self.map_coloring[state_b]
            if MapObj.are_adjacent_function(state_a, state_b):
                if color_a == color_b:
                    return False
        return True
    
    def are_adjacent(self,current_state,state):
        return self.are_adjacent_function(current_state,state)
    
def get_initial_map(colors, states, are_adjacent_function):
    map_coloring = {state: '' for state in states}
    MapObj.colors = colors
    MapObj.states = states
    #MapObj.are_adjacent_function = are_adjacent_function
    return MapObj(map_coloring,are_adjacent_function)
    
def copy_map(init_map):
    new_map = get_initial_map(init_map.colors, init_map.states, init_map.are_adjacent_function)
    for state in new_map.states:
        new_map.set_color(state,init_map.get_color(state))
    return new_map