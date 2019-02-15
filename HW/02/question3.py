# Question 3

#import mapColoringCode
from mapColoringCode import *

def find_coloring(init_map):
    return r_find_coloring([], init_map)

# Given map, finds state that needs coloring
def select_unassigned(m):
    for state in m.get_states():
        if m.get_color(state) == '' :
            return state

# Given map, current state, and potential color
# Check if color at state is allowed via constraint 
def valid_assignment(m,s,c):
    for other in m.get_states():
        if m.are_adjacent(s,other):
            if m.get_color(other) == c:
                return False
    return True

def r_find_coloring(states_colored, m):
    #Catch when we are done
    if set(states_colored) == set(m.get_states()):
        return m
    
    # Grab a state that isnt colored
    s = select_unassigned(m)
    
    for color in m.get_colors():
        if valid_assignment(m,s,color):
            m_new = copy_map(m)
            m_new.set_color(s,color)

            # Keep track of what states we already seen
            states_colored.append(s)
            
            result = r_find_coloring(states_colored,m_new)
            
            if result != '':
                return result
            
            states_colored.remove(s)

    # If fail 
    return ''
