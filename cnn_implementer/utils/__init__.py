from DAG import DAG
from Storable import Storable
from Logger import Logger
from RandomNumberGenerator import RandomNumberGenerator
from packing import Packing
from math import ceil, log
from ParetoSet import ParetoSet, CNNParetoSet

#next power of 'base' given x
def next_pow(x, base=2):
    return pow(int(ceil(log(x,base))),base)


#Yield successive n-sized chunks from l.
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


#function to find pareto points
#input: list of 2D-tuples, where the first entry is a list/tuple with costs, and the second is ignored (typically a config)
def pareto_cost_config(points):
    def dom_min_config(row_cfg, candidateRow_cfg):
        #strip config and perform comparision
        row, _=row_cfg
        candidateRow, _ = candidateRow_cfg
        return sum([row[x] <= candidateRow[x] for x in range(len(row))]) == len(row)

    # Empty list does nothing
    if len(points)==0:
        return [], []

    pareto, dominated = simple_cull(points, dominates=dom_min_config)
    return list(pareto), list(dominated)

def dominates_minimize(row, candidateRow):
    return sum([row[x] <= candidateRow[x] for x in range(len(row))]) == len(row)

def simple_cull(inputPoints, dominates=dominates_minimize):
    paretoPoints = set()
    candidateRowNr = 0
    dominatedPoints = set()
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            if dominates(candidateRow, row):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow):
                nonDominated = False
                dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0:
            break
    return paretoPoints, dominatedPoints
