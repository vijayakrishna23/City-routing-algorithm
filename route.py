#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 23:24:50 2018

@author: Vijaya Krishna (vgopala)
"""
"""
1. The routing search problem was formulated as any typical search problem with defining the start city, goal
   city, successor cities of a city, state space and cost function. 
   
   Initial state: Start city given by the user
   
   Goal state: Destination city given by the user
   
   Successors function: A function that lists all the cities and junctions that have a direct route with (from/to) a particular city 
                      are the successors states to the given city
   
   State space: All the cities and junctions in the road-segments data set.
   
   Cost function: Cost functions are distance, #segments, time between two cities
   
   Heuristic function :
   After a lot of consideration we came to the conclusion that the shortest geometric distance between the start 
   city and the goal city would be the best heuristic function. For calculating the distance between two GPS coordinates
   we used the HAVERSINE FORMULA, the haversine formula gives the shortest distance between any two points on earth
   using their respective latitude and longitude values. It is similar to Euclidean distance of a plane. Practically, 
   it is impossible for a road connecting two cities to be straight without any bends or curves, so our heuristic is
   admissible and even if there exists a road where the road distance is the shortest distance between the two cities
   even then our heuristic will be admissible as it does not overestimate the distance.

2. We used five different search algorithms in our program which are explained below:

   BFS: Breadth first search uses queue data structure which appends a city at one end
        of a list and gets another city from the opposite end. So, BFS will search all the successors
        of a particular city before moving to the next layer.
   DFS: Depth first search algorithm is similar to standard BFS, in that it is a blind search algorithm, with the only difference being stack 
        used as its data structure. So, the algorithm takes one of the cities from the successor
        cities and, if it is not the goal, the algorithm moves to the successor of that city instead of 
        exploring the sibling cities. For this reason, even searching between cities 
        with shorter distances like Bloomington,Indiana to Indianapolis,Indiana might take a lot of time.
   IDS: In IDS we initialy set the depth value as 1, so the algorithm function first searches the first layer 
        of successors for the goal and if the goal is not found we increment the depth by one, We set a limit
        on the depth so that the algorithm does not continuously proceed in one path before moving to a sibling as in DFS.
   A star: In A-star search, instead of blindly searching through the successor cities for the goal city we use a
          heuristic function and a priority queue. The heuristic function we use here is the shortest distance between two cities 
          (latitude, longitude coordinates) calculated using Haversine formula and the cost function being distance or segments or time.
   Uniform search: A variant of BFS implemented using priority queue, where the nodes are expanded based on edge cost.
        Here we used distance or segments or time for the cost function
               
       
3. Difficulty we faced:
   Initially we tried to reduce the number of successors by dividing the map into four quadrants using the 
   direction of the goal city from the start city and then filtering the successors based on the quadrants but the
   idea has its own loop holes which needed to be fixed so we discarded the idea considering the shortage of time. 
   
   Assumptions:
   For the routes where the distance and speed limits are missing we assumed them to be 25 and 50 respectively,
   considering the fact that the mean values of distance and speedlimits in the data set are 25.25 and 49.14. 
   As we do not have the GPS coordinates of the junctions we took the mean of the GPS coordinates of the cities 
   that were directly connected to the respective junction.

"""
import sys
import pandas as pd
import numpy as np
import timeit as t
from math import radians, cos, sin, asin, sqrt
import copy
from collections import deque
import heapq


city_gps_df = pd.read_csv("city-gps.txt", delimiter = " ", dtype = None)

road_segments_df = pd.read_csv("road-segments.txt", delimiter = " ", dtype = None)

city_gps_df.columns = ['City','Latitude','Longitude']
road_segments_df.columns = ['Origin','Destination','Distance','Speedlimit','Highway']

road_segments_df['Speedlimit'] = road_segments_df['Speedlimit'].replace(to_replace = 0, value = 50)
road_segments_df['Speedlimit'] = road_segments_df['Speedlimit'].replace(to_replace = np.NaN, value = 50)

# This function gives the latitude difference between two cities
def lat_diff(start_city,goal_city):
    start_lat = (city_gps_df.loc[city_gps_df['City'] == start_city])['Latitude']
    goal_lat = (city_gps_df.loc[city_gps_df['City'] == goal_city])['Latitude']
    lat_diff = float(goal_lat) - float(start_lat)
    return lat_diff

# This function gives the longitude difference between two cities
def lon_diff(start_city, goal_city):
    start_lon = (city_gps_df.loc[city_gps_df['City'] == start_city])['Longitude']
    goal_lon = (city_gps_df.loc[city_gps_df['City'] == goal_city])['Longitude']
    lon_diff = float(goal_lon) - float(start_lon)
    return lon_diff

# This function gives the directional quadrant in which the destination city is lying from the start city
def get_dest_quadrant(start_city, goal_city):
    if (city_gps_df.loc[city_gps_df['City'] == start_city]).size <= 0 or (city_gps_df.loc[city_gps_df['City'] == goal_city]).size <= 0:
        return 0
    
    else:
    
        lat1 = float((city_gps_df.loc[city_gps_df['City'] == start_city])['Latitude'])
        lat2 = float((city_gps_df.loc[city_gps_df['City'] == goal_city])['Latitude'])
       
        lon1 = float((city_gps_df.loc[city_gps_df['City'] == start_city])['Longitude'])
        lon2 = float((city_gps_df.loc[city_gps_df['City'] == goal_city])['Longitude'])
    
        quadrant = 0
    
        if lat_diff(start_city,goal_city) >= 0 and lon_diff(start_city,goal_city) >= 0:
#        print("Inside Q1")
            finallist = city_gps_df.loc[(city_gps_df['Latitude'] >= lat1) & (city_gps_df['Latitude'] <= lat2) \
                                    & (city_gps_df['Longitude'] >= lon1) & (city_gps_df['Longitude'] <= lon2)]
            quadrant = 1
    
        elif lat_diff(start_city,goal_city) >= 0 and lon_diff(start_city,goal_city) <= 0:
#        print("Inside Q2")
            finallist = city_gps_df.loc[(city_gps_df['Latitude'] >= lat1) & (city_gps_df['Latitude'] <= lat2) \
                                    & (city_gps_df['Longitude'] <= lon1) & (city_gps_df['Longitude'] >= lon2)]
            quadrant = 2

        elif lat_diff(start_city,goal_city) <= 0 and lon_diff(start_city,goal_city) <= 0:
#        print("Inside Q3")
            finallist = city_gps_df.loc[(city_gps_df['Latitude'] <= lat1) & (city_gps_df['Latitude'] >= lat2) \
                                    & (city_gps_df['Longitude'] <= lon1) & (city_gps_df['Longitude'] >= lon2)]
            quadrant = 3

        elif lat_diff(start_city,goal_city) <= 0 and lon_diff(start_city,goal_city) >= 0:
#        print("Inside Q4")
            finallist = city_gps_df.loc[(city_gps_df['Latitude'] <= lat1) & (city_gps_df['Latitude'] >= lat2) \
                                    & (city_gps_df['Longitude'] >= lon1) & (city_gps_df['Longitude'] <= lon2)]
            quadrant = 4
        
        return quadrant




# This function returns the successor states(cities connected) to the current state
def successors(start_city, goal_city):
    fromstart_df = copy.deepcopy(road_segments_df.loc[road_segments_df['Origin'] == start_city])
    tostart_df = copy.deepcopy(road_segments_df.loc[road_segments_df['Destination'] == start_city])
    tostart_df['Destination'] = tostart_df['Origin']
    tostart_df['Origin'] = start_city
    succ= pd.concat([fromstart_df, tostart_df])
    succ = succ.drop_duplicates(subset = ['Origin', 'Destination'])
    succ_cities = np.array(succ['Destination'])
    succ= succ_cities
    return succ

# This function check for goal state
def is_goal(state):
    return state == GOAL_CITY

# This function is the solve function for BFS and DFS
def solve_bfs_dfs(start_city, routing_algorithm, cost_function):

    fringe = deque([ (start_city, start_city) ]) if routing_algorithm == "bfs" else [ (start_city, start_city) ]

    visited = dict({})
    while len(fringe) > 0:
        (state, route_so_far) = fringe.popleft() if routing_algorithm == "bfs" else fringe.pop()
        visited[state] = route_so_far
        for succ in successors( state, goal_city):
            if is_goal(succ):
                return ( route_so_far + " " + succ )
            if succ not in visited:
                infringe = 0
                for f in range(len(fringe)):
                    if fringe[f][0] == succ:
                        f_len = len(fringe[f][1].split())
                        path =  route_so_far + " " + succ 
                        s_len = len(path.split())
                        if f_len > s_len:
                            del fringe[f]
                            fringe.append((succ, path))
                        infringe = 1
                if infringe == 0:
                    fringe.append((succ, route_so_far + " " + succ))
    return False

# This function is the solve function for Iterative Deepening Search
def solve_idfs(start_city, routing_algorithm, cost_function):

    depth = 1
    while depth > 0:
        fringe = [ (start_city, start_city) ]
        while len(fringe) > 0:
            (state, route_so_far) = fringe.pop()
            if len(route_so_far.split()) <= depth:
                for succ in successors( state, goal_city):
                    if is_goal(succ):
                        return ( route_so_far + " " + succ )
                    path =  route_so_far + " " + succ
                    if len(path.split()) <= depth:
                        fringe.append((succ, path))
        depth += 1
    return False

# This function gives the cost between first city and second city
def uniform_cost(first_city, second_city, cost_function):
    route_array = []
    row1 = road_segments_df.loc[(road_segments_df['Origin'] == first_city) & (road_segments_df['Destination'] == second_city)].values
    row2 = road_segments_df.loc[(road_segments_df['Origin'] == second_city) & (road_segments_df['Destination'] == first_city)].values 
    if len(row1) > 0:
        route_array.append(row1)
    if len(row2) > 0:
        route_array.append(row2)
    
    cost = 0
    if cost_function == "distance":
        cost = float(route_array[0][0, 2]) if float(route_array[0][0, 2]) > 0 else 25
    
    elif cost_function == "time":
        dist = float(route_array[0][0, 2]) if float(route_array[0][0, 2]) > 0 else 25
        speed = float(route_array[0][0, 3]) if float(route_array[0][0, 3]) > 0 else 50
        cost = round(dist/speed, 2)
        
    elif cost_function == "segments":
        cost = 1
        
    else :
        print("Please enter the valid cost")
        sys.exit()
    return cost
        
        
        
# This function is the solve function for Uniform Search using priority queue
def solve_uniform(start_city, routing_algorithm, cost_function):
    fringe = []
    heapq.heapify(fringe)
    heapq.heappush(fringe, [1, tuple([start_city, start_city]) ])
    visited = dict({})
    while len(fringe) > 0:
        popped = heapq.heappop(fringe)
        (state, route_so_far) = popped[1]
        visited[state] = route_so_far        
        
        for succ in successors( state, goal_city):
            if is_goal(succ):
                return ( route_so_far + " " + succ )
            
            path =  route_so_far + " " + succ 
            path_array = path.split()
            
            cost_so_far = 0
            if routing_algorithm == "uniform":
                cost_so_far = sum([uniform_cost(path_array[i], path_array[i+1], cost_function) for i in range(len(path_array) - 1)])
            elif routing_algorithm == "bfs" or routing_algorithm == "dfs":
                cost_so_far = uniform_cost(state, succ, cost_function)
            cost = cost_so_far
            #cost = uniform_cost(state, succ, cost_function)

            if succ not in visited:
                infringe = 0
                for f in range(len(fringe)):
                    if fringe[f][1][0] == succ:
                        f_array = fringe[f][1][1].split()
                        f_len = sum([uniform_cost(f_array[i], f_array[i+1], cost_function) for i in range(len(f_array) - 1)])
                        s_len = cost_so_far
                        if f_len > s_len:
                            del fringe[f]
                            heapq.heappush(fringe, [cost, tuple([succ, path])])
                        infringe = 1
                if infringe == 0:
                    if succ not in (route_so_far.split()):
                        heapq.heappush(fringe, [cost, tuple([succ, route_so_far + " " + succ])])
    return False

# Code Taken from Stack Overflow begins 
# https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points    
# This function calculates the Haversine Distance between two pairs of latitudes and longitudes
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 3956 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

# Code Taken from Stack Overflow ends
 
# This function finds the Latitude and Longitude coordinate of a place based on the places it has direct connectivity with
def find_loc(place):
    fromstart_df = copy.deepcopy(road_segments_df.loc[road_segments_df['Origin'] == place])
    tostart_df = copy.deepcopy(road_segments_df.loc[road_segments_df['Destination'] == place])
    tostart_df['Destination'] = tostart_df['Origin']
    tostart_df['Origin'] = place
    cities_df= pd.concat([fromstart_df, tostart_df])
    cities_df = cities_df.drop_duplicates(subset = ['Origin', 'Destination'])
    
    cities_around = np.array(cities_df['Destination'])
    lat_list = []
    lon_list = []
    for city in cities_around:
        if city in np.array(city_gps_df['City']):
            lat_list.append(city_gps_df.loc[city_gps_df['City'] == city].values[0][1])
            lon_list.append(city_gps_df.loc[city_gps_df['City'] == city].values[0][2])
   
    lat = np.mean(lat_list)
    lon = np.mean(lon_list)
    return [lat, lon]

# This function estimates the heuristic cost between two places based on haversine's formula
def heuristic_cost(first_city, second_city, cost_function):

    lat1 = 0 
    lon1 = 0
    lat2 = 0
    lon2 = 0

    if (city_gps_df.loc[city_gps_df['City'] == first_city]).size <= 0:
        lat1 = find_loc(first_city)[0]
        lon1 = find_loc(first_city)[1]
    
    else:
        lat1 = float((city_gps_df.loc[city_gps_df['City'] == first_city])['Latitude'])
        lon1 = float((city_gps_df.loc[city_gps_df['City'] == first_city])['Longitude'])
    
    if (city_gps_df.loc[city_gps_df['City'] == second_city]).size <= 0:
        lat2 = find_loc(second_city)[0]
        lon2 = find_loc(second_city)[1]
    
    else:
        lat2 = float((city_gps_df.loc[city_gps_df['City'] == second_city])['Latitude'])
        lon2 = float((city_gps_df.loc[city_gps_df['City'] == second_city])['Longitude'])
     
    cost = haversine(lon1, lat1, lon2, lat2)
    return cost
    
# This function is the solve function for the A star search using priority queue
def solve_heuristic(start_city, routing_algorithm, cost_function):
    fringe = []
    heapq.heapify(fringe)
    heapq.heappush(fringe, [0, tuple([start_city, start_city]) ])
    visited = dict({})
    while len(fringe) > 0:
        popped = heapq.heappop(fringe)
        (state, route_so_far) = popped[1]
        visited[state] = route_so_far
        if is_goal(state):
            return ( route_so_far )
        
        for succ in successors( state, goal_city):
 
            path = route_so_far + " " + succ
            path_array = path.split()
            cost_so_far = sum([uniform_cost(path_array[i], path_array[i+1], cost_function) for i in range(len(path_array) - 1)])
            cost = cost_so_far + heuristic_cost(succ, goal_city, cost_function)
            
            if succ not in visited:
                infringe = 0
                for f in range(len(fringe)):
                    if fringe[f][1][0] == succ:
                        f_array = fringe[f][1][1].split()
                        f_len = sum([uniform_cost(f_array[i], f_array[i+1], cost_function) for i in range(len(f_array) - 1)])
                        s_len = cost_so_far
                        if f_len > s_len:
                            del fringe[f]
                            heapq.heappush(fringe, [cost, tuple([succ, path])])
                        infringe = 1
                if infringe == 0:
                    if succ not in (route_so_far.split()):
                        heapq.heappush(fringe, [cost, tuple([succ, route_so_far + " " + succ])])
    return False

# This function formats the output
def format_output(path):
    if path == False:
        print("Route Not Found")
        sys.exit()
        
    path_array = path.split()
    pairs = [(path_array[i], path_array[i+1]) for i in range(len(path_array)-1)]
    
    #Calculating Bidirectional Routes
    route_array = []
    for pair in pairs:
        row1 = road_segments_df.loc[(road_segments_df['Origin'] == pair[0]) & (road_segments_df['Destination'] == pair[1])].values
        row2 = road_segments_df.loc[(road_segments_df['Origin'] == pair[1]) & (road_segments_df['Destination'] == pair[0])].values 
        if len(row1) > 0:
            route_array.append(row1)
        if len(row2) > 0:
            route_array.append(row2)

    tot_dist = 0
    tot_time = 0
    tot_highway = ""

    for route in route_array:
        dist = float(route[0, 2]) if float(route[0, 2]) > 0 else 25
        speed = float(route[0, 3]) if float(route[0, 3]) > 0 else 50
        tot_dist += dist
        tot_time += dist/speed
        tot_highway += " " + route[0, 4]
    
    optimal = "no"
    if routing_algorithm == "bfs":
        if cost_function == "segments":
            optimal = "yes"
        else:
            optimal = "no"
    elif routing_algorithm == "ids" or routing_algorithm == "uniform" or routing_algorithm == "astar":
        optimal = "yes"
    else:
        optimal = "no"
    #print("OUTPUT")
    print(optimal + " " + str(round(tot_dist, 2)) + " " + str(round(tot_time, 2)) + " " + path)


START_CITY = sys.argv[1]
GOAL_CITY = sys.argv[2]
routing_algorithm = sys.argv[3].lower()
cost_function = sys.argv[4].lower()

#START_CITY = 'Bloomington,_Indiana'
#GOAL_CITY = 'Indianapolis,_Indiana'
#routing_algorithm = "astar".lower()
#cost_function = "distance".lower()

start_city = copy.deepcopy(START_CITY)
goal_city = copy.deepcopy(GOAL_CITY)

t1 = t.default_timer()

#successors(start_city, goal_city)
if routing_algorithm == "bfs" or routing_algorithm == "dfs":
    if cost_function == "segments":
        path = solve_bfs_dfs(start_city, routing_algorithm, cost_function)
    else:
        path = solve_uniform(start_city, routing_algorithm, cost_function)
    #print(path)
    format_output(path)
elif routing_algorithm == "ids":
    path = solve_idfs(start_city, routing_algorithm, cost_function)
    #print(path)
    format_output(path)
elif routing_algorithm == "uniform":
    path = solve_uniform(start_city, routing_algorithm, cost_function)
    #print(path)
    format_output(path)
elif routing_algorithm == "astar":
    path = solve_heuristic(start_city, routing_algorithm, cost_function)
    #print(path)
    format_output(path)
else:
    print("Please enter the valid routing algorithm")

t2 = t.default_timer()
#print("Time Taken %f seconds" % (round(t2-t1, 4)))