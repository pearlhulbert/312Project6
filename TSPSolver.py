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
import random
from copy import deepcopy



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
                        if dist < min_dist:
                            min_dist = dist
                            min_path = path.copy()

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
        end_time = time.time() - start_time
        solution = None
        cost = math.inf
        if len(min_path) > 0:
            solution = TSPSolution(min_path)
            cost = solution.cost
        else:
            count = 0

        
        print("greedy bssf: ", cost)
        print("greedy time: ", end_time)
        results = {}
        results['paths'] = valid_paths
        results['city_path'] = solution

        return results
    
    
        
    def branchAndBound( self, time_allowance=60.0 ):
        
        #initializes all the variables that we need to keep track of
        total_pruned = 0
        total_children = 0
        max_children_size = 0
        min_dist = math.inf
        min_path = []
        count_leaf = 0

        cities = self._scenario.getCities()
        start_matrix = []

        start_time = time.time()
        greedy_result = self.greedy(time_allowance)
        bssf = greedy_result['city_path'].cost

        #creaing the starting matrix
        for _, val1 in enumerate(cities):
            row = []
            for _, val2 in enumerate(cities):
                row.append(val1.costTo(val2))
            start_matrix.append(row)
        
        #creating starting nodes matrix and reducing it
        matrix = Matrix(start_matrix)
        matrix.reduce_matrix()

        #start of intelligent search
        queue = []
        head = Node(deepcopy(matrix), None, 0, [cities[0]])
        start_node = head
        
        #loop that counts for time
        while time.time() - start_time < time_allowance:
            #creating children and adding them to the queue
            #only iterates over the cities that aren't the start city
            for i in range(1, len(cities)):
                #doesn't try to compare the distance of a node to itself
                if i == start_node.city_index:
                    continue
                
                #creates child node, prunes the matrix, updates the bssf
                child = Node(deepcopy(start_node.matrix), start_node.city_index, i, deepcopy(start_node.path))
                child.path.append(cities[i])


                #adds child to the queue if less than the bssf, prunes otherwise
                if child.bssf <= bssf:
                    heapq.heappush(queue, (child.bssf, child.city_index, child))
                else:
                    total_pruned += 1

                total_children += 1
            
            #checking for an empty queue
            if start_node is None or len(queue) == 0:
                total_pruned += len(queue)
                break
            
            #checks if all the cities have been visited up to the current node
            #then checks if there is a path to the starting node
            #if there is, then update the bssf and copy the path to get there
            if len(start_node.path) >= len(cities):
                if start_node.bssf != math.inf and start_node.bssf < min_dist:
                    min_dist = start_node.bssf
                    min_path = deepcopy(start_node.path)
                    count_leaf += 1
                    bssf = min_dist

            max_children_size = max(max_children_size, len(queue))
            heapq.heapify(queue)

            #gets the next node out of the queue, breaks if there are no nodes that are better than the bssf
            start_node = heapq.heappop(queue)[2]
            if start_node.bssf > bssf:
                total_pruned += len(queue)
                break
        
        #gets time and creates results data structure
        end_time = time.time()

        solution = None
        if len(min_path) > 0:
            solution = TSPSolution(min_path)

        results = {}
        results['cost'] = bssf if solution is not None else 0
        results['time'] = end_time - start_time
        results['count'] = count_leaf
        results['soln'] = solution
        results['max'] = max_children_size
        results['total'] = total_children
        results['pruned'] = total_pruned
        return results



    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found during search, the 
        best solution found.  You may use the other three field however you like.
        algorithm</returns> 
    '''
        
    def generateStartingPopulation( self, population_size, time_allowance ):
        #greedy, default... generate a bunch of paths
        num_cities = len(self._scenario.getCities())
        starting_population = self.greedy(time_allowance)['paths']
        num_population_to_create = population_size - len(starting_population)
        
        remaining_population = [[i for i in range(num_cities)] for _ in range(num_population_to_create)]

        for path in remaining_population:
            random.shuffle(path)

        starting_population += remaining_population
        return starting_population

    def pickBestPath(self, best_path, paths):
        best_fitness = self.checkFitness(best_path)
        #cities = self._scenario.getCities()

        for path in paths:
            #print(path)
            new_fitness = self.checkFitness(path)
            if new_fitness < best_fitness:
                best_path = path
                best_fitness = new_fitness

        return deepcopy(best_path)

    def checkFitness(self, path):
        fitness = 0
        cities = self._scenario.getCities()
        
        for city_index in range(len(path) - 1):
            fitness += cities[path[city_index]].costTo(cities[path[city_index + 1]])

        return fitness

    #Returns an array of paths, 0+1 should be combined, 2+3, 4+5 etc.
    def selectWhichToCombine(self, allPaths):
        #keep top 1, and any that are within 50% of it
        min = np.inf
        max = 0
        pathsToCombine = []
        pathsToKeep = []
        for path in allPaths:
            dist = self.checkFitness(path)
            if dist > max:
                max = dist
            if dist < min:
                min = dist
        for path in allPaths:
            dist = self.checkFitness(path)
            if dist < max - ((max-min)/2):
                #Keep all that are 'good enough'
                pathsToCombine.append(path)
            if dist == max:
                #Keep the best one 2 times, so that it will not be destroyed
                pathsToKeep.append(path)    
                pathsToKeep.append(path)
        random.shuffle(pathsToCombine)
        for path in pathsToCombine:
            pathsToKeep.append(path)
        return pathsToKeep

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

    def crossPopulation(self, population, population_size):
        new_population = []
        new_population.append(deepcopy(population[0]))
        

        for path_index in range(0, len(population) - 2, 2):
            if len(new_population) >= population_size:
                break
            new_population.append(self.crossOver(population[path_index], population[path_index + 1]))
            new_population.append(self.crossOver(population[path_index + 1], population[path_index]))

        return new_population[:population_size]


    def mutate(self, path):
        #do some random swaps on the path
        index1 = random.randrange(len(path))
        index2 = random.randrange(len(path))

        temp = path[index1]
        path[index1] = path[index2]
        path[index2] = temp
        return path

    def mutatePopulation(self, population, generation):
        for path in population:
            path = self.mutate(path)
        return population

    def fancy( self, time_allowance=60.0 ):
        population_size = 100
        cities = self._scenario.getCities()
        start_time = time.time()
        population = self.generateStartingPopulation(population_size, time_allowance)
        best_path = deepcopy(population[0])
        generation = 0

        while time.time() - start_time < time_allowance:
            best_path = self.pickBestPath(best_path, population)
            combine_population = self.selectWhichToCombine(population)
            population = self.crossPopulation(combine_population, population_size)
            population = self.mutatePopulation(population, generation)

            generation += 1
            if generation >= 10000:
                break
            

        end_time = time.time()

        city_path = []
        for index in best_path:
            city_path.append(cities[index])

        solution = TSPSolution(city_path)
        results = {}
        results['time'] = end_time - start_time
        results['soln'] = solution
        results['cost'] = solution.cost
        results['count'] = 0
        return results
        



class Matrix:
    def __init__(self, matrix):
        self.bssf = 0
        self.matrix = []
        self.matrix = matrix

    #finds the smallest number of each row and subtracts the each number from it
    def reduce_matrix(self):
        for i in range(len(self.matrix)):
            #finds the smallest integer from the row
            row_min = min(self.matrix[i])
            if row_min == math.inf:
                continue
            self.bssf += row_min
            #subtracts each item in that row by the minimum
            self.matrix[i] = list(map(lambda x: x - row_min, self.matrix[i]))

        self.reduce_columns()

    def reduce_columns(self):
        for i in range(len(self.matrix)):
            #finds the smallest integer in each column
            column_min = math.inf
            for j in range(len(self.matrix[i])):
                column_min = min(column_min, self.matrix[j][i])

            #if there are no zeros in a column or the smallest integer is infinity,
            #then subtract each item in the column by the smallest number
            if column_min > 0 and column_min != math.inf:
                for j in range(len(self.matrix[i])):
                    self.matrix[j][i] -= column_min
                self.bssf += column_min
    
    def mark_matrix(self, parent, child):
        self.bssf += self.matrix[parent][child]

        for i in range(len(self.matrix)):
            self.matrix[parent][i] = math.inf

        for i in range(len(self.matrix[0])):
            self.matrix[i][child] = math.inf

        self.matrix[child][parent] = math.inf
        self.reduce_matrix()

    def __str__(self):
        out = ""
        for s in self.matrix:
            out += str(s) + "\n"
        return out


class Node:
    def __init__(self, matrix, parent_index, city_index, path):
        self.matrix = matrix
        self.path = path
        self.city_index = city_index
        self.bssf = self._set_bssf(parent_index)

    def _set_bssf(self, parent_index):
        if parent_index is not None:
            self.matrix.mark_matrix(parent_index, self.city_index)

        return self.matrix.bssf

    def __lt__(self, other):
        return self.city_index < other.city_index

