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



##############################################################
#               Partners for this problem set:               #
#                   1. Vivek Jayaram                         #
#                   2. Saroj Kandel                          #
##############################################################


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Find positions of scared and unscared ghosts
        new_ghost_pos = [ghost.getPosition() for ghost in newGhostStates if not ghost.scaredTimer]
        new_scared_pos = [ghost.getPosition() for ghost in newGhostStates if ghost.scaredTimer]

        # Find closest active ghost
        if new_ghost_pos:
          closest_active_ghost = min(map(lambda ghostPos: util.manhattanDistance(newPos, ghostPos), new_ghost_pos))
        else:
          closest_active_ghost = 100000

        # Find closest scared ghost
        if new_scared_pos:
          closest_scared_ghost = min(map(lambda ghostPos: util.manhattanDistance(newPos, ghostPos), new_scared_pos))
        else:
          closest_scared_ghost = 0

        # Find closest food
        if not newFood:
          closest_food = -100
        else:
          closest_food = min(map(lambda foodPos: util.manhattanDistance(newPos, foodPos), newFood))

        # Total food left
        food_left = len(newFood)

        # Total capsules left
        capsules_left = len(currentGameState.getCapsules())

        print "closest_active_ghost " + str(closest_active_ghost)
        print "closest_scared_ghost " + str(closest_scared_ghost)
        print "food left " + str(food_left)
        if closest_active_ghost < 2:
          return -100000000
        return float(1.0/closest_active_ghost) * -12 + (-1.5 * closest_food) + \
              (-20 * food_left) + (-4 * capsules_left) + scoreEvaluationFunction(successorGameState)

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        """
        # Rather than directly calling minimax on pacman, call on each of his moves to determine the best move
        temp = float("-inf")
        bestAction = None
        for action in gameState.getLegalActions(0):
            best_value = self.minimaxHelper(gameState.generateSuccessor(0, action), 1, self.depth)
            # If the minimax result was better on this action then keep that in temp
            if best_value > temp or not bestAction:
                bestAction = action
                temp = best_value
        return bestAction

    def minimaxHelper(self, state, agentIndex, depth):
        """ A recursive minimax helper function
            ARGS 
            state: the gameState we are inspecting
            agentIndex: the current agent Index, 0 is pacman, ghosts are >=1
            depth: the depth left that we have to explore
        """
        # Terminal states
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        # If we are pacman 
        if agentIndex == 0:
            bestValue = float("-inf")
            for action in state.getLegalActions(agentIndex):
                bestValue = max(bestValue, self.minimaxHelper(state.generateSuccessor(agentIndex, action), 
                    agentIndex + 1, depth))

        # If we are the last ghost (need to decrease depth here)
        elif agentIndex == (state.getNumAgents() - 1):
            bestValue = float("inf")
            for action in state.getLegalActions(agentIndex):
                bestValue = min(bestValue, self.minimaxHelper(state.generateSuccessor(agentIndex, action),
                    0, depth-1))

        # If we are a ghost but not the last one
        else:
            bestValue = float("inf")
            for action in state.getLegalActions(agentIndex):
                bestValue = min(bestValue, self.minimaxHelper(state.generateSuccessor(agentIndex, action),
                    agentIndex + 1, depth))  
        
        return bestValue      
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = float("-inf")
        bestValue = float("-inf")
        bestAction = None
        # Run alpha beta pruning on each possible action, update alpha but beta stays the same
        for action in gameState.getLegalActions(0):
            currentValue = self.alphaBetaHelper(gameState.generateSuccessor(0, action), 1, self.depth, alpha, float("inf"))
            # If the minimax result was better on this action then keep that in temp
            if currentValue > bestValue or not bestAction:
                bestAction = action
                bestValue = currentValue
                # It's important that the next action we try one has the best alpha, ie the best move so far, but beta is still inf
                #if v > beta:
                 #   return bestAction
                alpha = max(alpha, currentValue)
        return bestAction

    def alphaBetaHelper(self, gameState, agentIndex, depth, alpha, beta):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # If it's the maximizing player
        if agentIndex == 0:
            v = float("-inf")
            for action in gameState.getLegalActions(agentIndex):
                v = max(v, self.alphaBetaHelper(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v
        
        elif agentIndex == (gameState.getNumAgents() - 1):
            v = float("inf")
            for action in gameState.getLegalActions(agentIndex):
                v = min(v, self.alphaBetaHelper(gameState.generateSuccessor(agentIndex, action), 0, depth - 1, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        else:
            v = float("inf")
            for action in gameState.getLegalActions(agentIndex):
                v = min(v, self.alphaBetaHelper(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        maxVal = float('-Inf')

        bestAction = 0
        for action in actions:
            currentDepth = 0
            currState = gameState.generateSuccessor(0, action)
            currMax = self.helperExpectedVal (currState, currentDepth, 1)
            
            if currMax > maxVal:
                maxVal = currMax
                bestAction = action
        return bestAction
    
    def helperMaxVal (self, state, depth):
        depth += 1
        #if the state is the terminal state, just return the eval func value
        if (depth == self.depth) or (state.isWin()) or (state.isLose()):
            return self.evaluationFunction(state)
        
        else:
            maxVal = float('-Inf')
            # get the maximum value out of all actions
            for action in state.getLegalActions(0):
                maxVal = max(maxVal, self.helperExpectedVal (state.generateSuccessor(0, action), depth, 1))
            return maxVal

    def helperExpectedVal (self, state, depth, numGhosts):
        # if the state is the terminal state, just return the eval func value
        if (depth == self.depth) or (state.isWin()) or (state.isLose()):
            return self.evaluationFunction(state)
        
        else:
            expectedVal = 0
            for action in state.getLegalActions(numGhosts):
                # if all ghosts remaining
                if numGhosts == state.getNumAgents() - 1:
                    maxValState = state.generateSuccessor(numGhosts,action)
                    numGhostActions = len(state.getLegalActions(numGhosts))
                    maxVal = (self.helperMaxVal(maxValState, depth))/numGhostActions
                    expectedVal += maxVal

                else:
                    eValState = state.generateSuccessor(numGhosts,action)
                    numGhostActions = len(state.getLegalActions(numGhosts))
                    eVal = self.helperExpectedVal(eValState, depth, numGhosts + 1)/ numGhostActions
                    expectedVal += eVal
                    
            return expectedVal   

"""def betterEvaluationFunction(currentGameState)Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>*** YOUR CODE HERE ***"
    util.raiseNotDefined()"""

def betterEvaluationFunction(currentGameState):
     """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
    
      DESCRIPTION: <write something here so we know what you did>
       I built from the previous evaluation function that was given to us
       I first initialize our heuristic to zero, and add the number of moves
       that each ghost will remain scared for. I then calculate manhattan distance
       from pacman's position to each of the ghost's position. After that, I take the minimum
       of all the distances and subtracted 2 times that from my heuristic function
       Then I add the current score to the heuristic.
     """
     "*** YOUR CODE HERE ***"
     # useful numbers from the previous evaluation function
     newPos = currentGameState.getPacmanPosition()
     newGhostStates = currentGameState.getGhostStates()
     newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
     heuristic = 0
     
     distancesToGhost = []
     for item in newGhostStates:
         distancesToGhost += [manhattanDistance(item.getPosition(),newPos)]
     
     for item in newScaredTimes:
         heuristic += item
         
     heuristic -= min(distancesToGhost) * 2
     heuristic += currentGameState.getScore()
     return heuristic


# Abbreviation
better = betterEvaluationFunction

