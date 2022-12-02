#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools



class TSPSolver:
    def __init__( self, gui_view ):
        self._scenario = None

    def setupWithScenario( self, scenario ):
        self._scenario = scenario


    ''' <summary>
        This is the entry point for the default solver
        which just finds a valid random tour.  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of solution, 
        time spent to find solution, number of permutations tried during search, the 
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''
    
    def defaultRandomTour( self, time_allowance=60.0 ):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time()-start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation( ncities )
            route = []
            # Now build the route using the random permutation
            for i in range( ncities ):
                route.append( cities[ perm[i] ] )
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results


    ''' <summary>
        This is the entry point for the greedy solver, which you must implement for 
        the group project (but it is probably a good idea to just do it for the branch-and
        bound project as a way to get your feet wet).  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found, the best
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''
   #does the greedy algorithm using each node as a starting point. 
    #finds the shortest distance between each node not yet visited
    #returns the shortest path found
    def greedy( self,time_allowance=60.0 ):
        results = {}
        min_dist = math.inf
        min_path = []
        valid_paths = []
        count = 0
        city_indice = 0

        start_time = time.time()

        while city_indice < len(self._scenario.getCities()) and time.time() - start_time < time_allowance:
            #creates a copy of the cities that it can go to
            cities = self._scenario.getCities().copy()
            start_city = cities[city_indice]
            curr_city = start_city
            dist = 0
            path = [cities.pop(city_indice)]

            while True:
                
                min_index = 0
                
                #tries to find the shortest distance to the next node from the current one, that hasn't been visited yet
                for i, val in enumerate(cities):
                    if curr_city.costTo(val) < curr_city.costTo(cities[min_index]):
                        min_index = i

                #when all the cities have been visited, 
                #make sure that there is a path back to the first node
                #check if the bssf is any better and keep the path if it is

                if len(cities) == 0:
                    if curr_city.costTo(start_city) != math.inf:
                        new_path = []
                        for city in path:
                            new_path.append(city._index)

                        valid_paths.append(new_path)

                        count += 1
                    city_indice += 1
                    break

                #returns if there are no available paths that this node can take
                if curr_city.costTo(cities[min_index]) == math.inf:
                    city_indice += 1
                    break

                #each city that had been visited by this iteration is removed from the city queue
                dist += curr_city.costTo(cities[min_index])
                curr_city = cities[min_index]
                path.append(cities.pop(min_index))

        cost = math.inf
        if len(min_path) > 0:
            solution = TSPSolution(min_path)
            cost = solution.cost
        else:
            count = 0

        end_time = time.time()
        print(valid_paths)

        
        return valid_paths
    
    
    
    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints: 
        max queue size, total number of states created, and number of pruned states.</returns> 
    '''
        
    def branchAndBound( self, time_allowance=60.0 ):
        pass



    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found during search, the 
        best solution found.  You may use the other three field however you like.
        algorithm</returns> 
    '''
        
    def getNewPath( self ):
        #greedy, default... generate a bunch of paths
        pass

    def checkFitness(self, path):
        fitness = 0
        cities = self._scenario.getCities()
        
        for city_index in range(len(path) - 1):
            fitness += cities[city_index].costTo(cities[city_index])

        return fitness

    def selectWhichToCombine(self, allPaths):
        #keep top 3, match 50% of others together
        pass

    def crossOver(self, path1, path2):
        # Mix two paths together
        best_path = []
        sub_path1 = []
        sub_path2 = []

        a = int(random.random() * len(path1))
        b = int(random.random() * len(path2))

        start = min(a, b)
        end = max(a, b)

        for i in range(start, end):
            sub_path1.append(path1[i])

        sub_path2 = [index for index in path2 if index not in sub_path1]

        best_path = sub_path1 + sub_path2

        return best_path


def mutate(self, path):
		#do some random swaps on the path
		index1 = random.randrange(len(path))
		index2 = random.randrange(len(path))

		temp = path[index1]
		path[index1] = path[index2]
		path[index2] = temp
		return



        
    def fancy( self,time_allowance=60.0 ):
        paths = self.greedy(time_allowance)
        for p in paths:
            print(self.checkFitness(p))
        pass
        



