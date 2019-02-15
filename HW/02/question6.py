# Question 6

#import mapColoringCode
from mapColoringCode import *

def find_coloring(init_map):
    return r_find_coloring([], init_map, 0)

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

def r_find_coloring(states_colored, m, dead_end):
    #Catch when we are done
    if set(states_colored) == set(m.get_states()):
        return m,dead_end
    
    # Grab a state that isnt colored
    # New method returns a Minimum Remaining Values (MRV) 
    # ordering of the states that are still not colored. 
    s = m.get_not_colored()[0]
    
    for color in m.get_colors():
        if valid_assignment(m,s,color):
            m_new = copy_map(m)
            m_new.set_color(s,color)

            # Keep track of what states we already seen
            states_colored.append(s)
            
            result,dead_end = r_find_coloring(states_colored,m_new,dead_end)
            
            if result != '':
                return result,dead_end
            
            states_colored.remove(s)

    # If fail 
    dead_end += 1
    return '',dead_end
