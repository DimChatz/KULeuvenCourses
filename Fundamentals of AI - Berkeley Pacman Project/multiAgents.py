# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # I will create a linear combination of any factors I am able to think of
        # and assign different weights to each of them

        # Get closest food in order to create reward for it
        foodList = newFood.asList()
        # Initialize big 
        minFoodDist = float("inf")
        # Find minimum food distance
        for food in foodList:
            distance = manhattanDistance(newPos, food)
            if distance < minFoodDist:
                minFoodDist = distance
        
        # If there is no food set it to 0
        if not foodList:
            minFoodDist = 0

        # Get closest ghost to create punishment for it
        # and maximum to create a scaled punishment for that
        minGhostDistance = float("inf")
        maxGhostDistance = 0
        # Get ghost positions
        for ghost in newGhostStates:
            ghostPos = ghost.getPosition()
            ghostManhattan = manhattanDistance(newPos, ghostPos)
            # Find closest ghost
            if ghostManhattan < minGhostDistance:
                minGhostDistance = ghostManhattan
            # Find furthest ghost
            if ghostManhattan > maxGhostDistance:
                maxGhostDistance = ghostManhattan

        # Penalize standing still
        standingPenalty = 0
        if action == 'Stop':
            standingPenalty = 1

        # I combine all the above
        # I scale the max ghost distance
        # I also penalize by the number of food left
        return successorGameState.getScore() - minFoodDist*3 + minGhostDistance + maxGhostDistance*(maxGhostDistance-minGhostDistance) - standingPenalty*100 - len(foodList)*100

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # Initialize best action and score
        bestAction = None
        bestScore = float("-inf")
        
        # get legal actions
        legalActions = gameState.getLegalActions(0)
        # and find the best of them 
        # by recursively calling the minimax
        # which has a prolog like logic of recursion
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            successorScore = self.minimax(successor, 1, 0)
            # get best action
            if successorScore > bestScore:
                bestScore = successorScore
                bestAction = action
        return bestAction    
        util.raiseNotDefined()


    ########################
    ### MINIMAX FUNCTION ###
    ########################
    def minimax(self, state, agent, depth):
        # Ending conditions
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)
        # Pacman's turn - MAX
        if agent == 0:
            highScore = float("-inf")
            actions = state.getLegalActions(agent)  
            for action in actions:
                successor = state.generateSuccessor(agent, action)
                # Recursive call of next agent
                tempScore = self.minimax(successor, agent + 1, depth)
                # Save best score
                if highScore < tempScore:
                    highScore = tempScore
            return highScore
        # Ghost's turn - MIN
        else:
            lowScore = float("inf")
            actions = state.getLegalActions(agent)
            for action in actions:
                successor = state.generateSuccessor(agent, action)
                # If it is the last ghost
                if agent == state.getNumAgents() - 1:
                    # Increase depth and call Pacman's turn
                    tempScore = self.minimax(successor, 0, depth + 1)
                else:
                    # Else increase agent by one
                    tempScore = self.minimax(successor, agent + 1, depth)
                # Save best score
                if lowScore > tempScore:
                    lowScore = tempScore
            return lowScore

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Set alpha and beta values and best action
        alpha = float("-inf")
        beta = float("inf")
        bestAction = None

        # Main loop - Same as minimax
        legalActions = gameState.getLegalActions(0)
        for action in legalActions:
            successor = gameState.generateSuccessor(0,action)
            value = self.alphaBeta(successor, 1, 0, alpha, beta)
            if value > alpha:
                alpha = value
                bestAction = action
        return bestAction
        util.raiseNotDefined()

    ##########################
    ### ALPHABETA FUNCTION ###
    ##########################
    def alphaBeta(self, state, agent, depth, alpha, beta):
        # End conistions
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)

        # Pacman's turn - alpha pruning
        if agent == 0:
            # Set value to compare
            value = float("-inf")
            legalActions = state.getLegalActions(agent)
            for action in legalActions:
                successor = state.generateSuccessor(agent, action)
                # Get max between value and recursive alphabeta
                value = max(value, self.alphaBeta(successor, 1, depth, alpha, beta))
                # Implement the pruning
                if value > beta:
                    return value
                alpha = max(alpha, value)
            return value
        # Ghost's turn - beta pruning
        else:
            # Set value to compare
            value = float("inf")
            legalActions = state.getLegalActions(agent)
            for action in legalActions:
                successor = state.generateSuccessor(agent, action)
                # Check for diffs in agents
                if agent == state.getNumAgents() - 1:
                    # Get min between value and recursive alphabeta
                    value = min(value, self.alphaBeta(successor, 0, depth + 1, alpha, beta))
                else:
                    value = min(value, self.alphaBeta(successor, agent + 1, depth, alpha, beta))
                # Implement the pruning
                if value < alpha:
                    return value
                beta = min(beta, value)
            return value
        
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Set Score and action like previous
        bestScore = float("-inf")
        bestAction = None
        legalActions = gameState.getLegalActions(0)
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            score = self.expectiMax(successor, 1, 0)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction
        util.raiseNotDefined()

    ##################
    ### EXPECTIMAX ###
    ##################
    def expectiMax(self, state, agent, depth):
        # Set end conditions
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)

        # Pacman's turn
        if agent == 0:
            # Set value to compare
            value = float("-inf")
            legalActions = state.getLegalActions(agent)
            for action in legalActions:
                successor = state.generateSuccessor(agent, action)
                # Call recursive expectimax and update value if needed
                tempValue = self.expectiMax(successor, 1, depth)
                # Implement the pruning
                if value < tempValue:
                    value = tempValue
            return value
        # Ghost's turn
        else: 
            value = 0
            legalActions = state.getLegalActions(agent)
            for action in legalActions:
                successor = state.generateSuccessor(agent, action)
                # In case of each agent call expectimax
                # and add the value to the sum
                if agent == state.getNumAgents() - 1:
                    value += self.expectiMax(successor, 0, depth + 1)
                else:
                    value += self.expectiMax(successor, agent + 1, depth)
            # Return the average of all values
            return value / len(legalActions)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    score = currentGameState.getScore()

    # Get closest food in order to create reward for it
    # Like previous evaluationFunction
    foodList = newFood.asList()
    minFoodDist = float("inf")
    for food in foodList:
        distance = manhattanDistance(newPos, food)
        if distance < minFoodDist:
            minFoodDist = distance
    
    if not foodList:
        minFoodDist = 0
    
    # Update score
    score -= 3 * minFoodDist

    # Get closest ghost to create punishment for it
    # and furthest ghost
    minGhostDistance = float("inf")
    maxGhostDistance = 0
    ghostHunting = 0
    ghostHunted = 0
    for count, ghost in enumerate(newGhostStates):
        ghostPos = ghost.getPosition()
        ghostManhattan = manhattanDistance(newPos, ghostPos)
        if ghostManhattan < minGhostDistance:
            minGhostDistance = ghostManhattan
            score += minGhostDistance
        if ghostManhattan > maxGhostDistance:
            maxGhostDistance = ghostManhattan
            score += 0.5 * maxGhostDistance
        # Get reward and for ghost hunting
        if newScaredTimes[count] > 0:
            ghostHunting = 1
            score += 50 * ghostHunting
        # Get punishment for being hunted
        else:
            ghostHunted = -1
            score -= ghostHunted

    # Penalize leaving food
    score -= 100 * len(foodList)
    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
