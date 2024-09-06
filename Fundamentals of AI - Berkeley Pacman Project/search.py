# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import Stack, Queue, PriorityQueue, PriorityQueueWithFunction
from typing import List, Tuple, Any
from game import Directions
from game import Agent
from game import Actions
import time
import search
import pacman


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # Define start state
    startState = problem.getStartState()
    # Define and initialize frontier state
    frontier = Stack()
    frontier.push([startState, []])
    # Initialize a list for loop detection
    loopList = []

    # If frontier state isn't empty
    # aka if there is still a possibility for solution
    while (not frontier.isEmpty()):
        # Pop the last injected in the stack (LIFO) and get its parts
        currentState, actions = frontier.pop()

        # Check if currentState is the goal
        # and if yes return the actions taken
        if problem.isGoalState(currentState):
            return actions

        # check if the currentState is in the loopList...
        if currentState not in loopList:
            # ...and if not append it...
            loopList.append(currentState)
            # and get all successors and actions and append them to the frontier
            for successor, action, _ in problem.getSuccessors(currentState):    
                succActions = actions + [action]
                frontier.push([successor, succActions])
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Exactly as above but with the FIFO data structure
    # Define start state
    startState = problem.getStartState()
    # Define and initialize frontier state
    frontier = Queue()
    frontier.push([startState, []])
    # Initialize a list for loop detection
    loopList = []

    # If frontier state isn't empty
    # aka if there is still a possibility for solution
    while (not frontier.isEmpty()):
        # Pop the first injected in the stack (FIFO) and get its parts
        currentState, actions = frontier.pop()

        # Check if currentState is the goal
        # and if yes return the actions taken
        if problem.isGoalState(currentState):
            return actions

        # check if the currentState is in the loopList...
        if currentState not in loopList:
            # ...and if not append it...
            loopList.append(currentState)
            # and get all successors and actions and append them to the frontier
            for successor, action, _ in problem.getSuccessors(currentState):    
                succActions = actions + [action]
                frontier.push([successor, succActions])
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Define start state
    startState = problem.getStartState()
    # Define and initialize frontier state
    frontier = PriorityQueue()
    # Prio is equal to cost
    frontier.push([startState, [], 0], 0)
    # Initialize a dictionary for loop detection
    loopDict = {startState: 0}

    # If frontier state isn't empty
    # aka if there is still a possibility for solution
    while (not frontier.isEmpty()):
        # Pop the least cost injected in the stack and get its parts
        currentState, actions, currentCost = frontier.pop()

        # Check if currentState is the goal
        # and if yes return the actions taken
        if problem.isGoalState(currentState):
            return actions
        
        # Add to dict to keep track
        if currentState not in loopDict.keys() or loopDict[currentState] > currentCost:
            loopDict[currentState] = currentCost

        # Get successor of current state, action and cost 
        for successor, action, cost in problem.getSuccessors(currentState):
            # Add to get path cost
            succCost = currentCost + cost
            # Check to see if successor already is there and if yes, check if we found a lesser cost path
            if successor not in loopDict.keys() or loopDict[successor] > succCost:
                # If yes, update dict
                loopDict[successor] = succCost
                # Update actions
                succActions = actions + [action]
                # Update object to change priorities and make sure
                # to pop the correct node next expansion
                frontier.update([successor, succActions, succCost], succCost)
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Define start state
    startState = problem.getStartState()
    # Define and initialize frontier state
    frontier = PriorityQueue()
    # Prio is cost + heuristic
    frontier.push([startState, [], 0], 0 + heuristic(startState, problem))
    # Initialize a list for loop detection
    # We dont need the dict since we know that pathing is optimal
    loopList = []

    # If frontier state isn't empty
    # aka if there is still a possibility for solution
    while (not frontier.isEmpty()):
        # Pop the least prio injected in the stack and get its parts
        currentState, actions, currentCost = frontier.pop()
        # Check if currentState is the goal
        if problem.isGoalState(currentState):
            return actions
        
        # Check if it exists in list
        # and if not apppend to list
        if currentState not in loopList:
            loopList.append(currentState)
            # Get successor of current state, action and cost 
            for successor, action, cost in problem.getSuccessors(currentState):
                # Check that you havent gone there yet
                if successor not in loopList:
                    # Add to get path cost
                    succCost = currentCost + cost
                    # Update actions
                    succActions = actions + [action]
                    # Update object to change priorities and make sure
                    # to pop the correct node next expansion
                    frontier.update([successor, succActions, succCost], succCost + heuristic(successor, problem))
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
