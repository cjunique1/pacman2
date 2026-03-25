from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    Compact representation of the game state used as the key
    in the Q-table.

    The full GameState object contains lots of information, but
    for learning we only keep features that distinguish meaningful
    situations:
    - Pacman's position
    - ghost positions
    - remaining food locations

    Including the food layout helps differentiate states that look
    similar spatially but have different goals remaining.
    """
    def __init__(self, state: GameState):
        self.pacman_pos = state.getPacmanPosition()
        self.food = state.getFood()
        # Convert food grid to a hashable form so it can be used
        # in dictionary keys.
        self.food_key = tuple(sorted(state.getFood().asList()))
        self.ghosts = tuple(state.getGhostPositions())
        # Legal actions available to Pacman (excluding STOP)
        self.legal = state.getLegalPacmanActions()
        if Directions.STOP in self.legal:
            self.legal.remove(Directions.STOP)

    def __hash__(self):
        return hash((self.pacman_pos, self.ghosts, self.food_key))

    def __eq__(self, other):
        return (self.pacman_pos, self.ghosts, self.food_key) == (other.pacman_pos, other.ghosts, other.food_key)

class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        Q-learning agent for Pacman.

        alpha: learning rate
        epsilon: probability of random exploration
        gamma: discount factor for future rewards
        maxAttempts: threshold for count-based exploration
        numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        self.episodesSoFar = 0
        # Q-table storing Q(state, action)
        self.qValues = {}   
        # Counts of how often each (state, action) pair was taken
        self.counts = {}    
        # Counts of how often each (state, action) pair was taken
        self.lastState = None 
        self.lastAction = None

    # Episode counter helpers
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Parameter accessors
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Reward = change in score between states, with extra
        bonus/penalty for terminal outcomes.
        """
        reward = endState.getScore() - startState.getScore()

        if endState.isWin():
            reward += 500
        if endState.isLose():
            reward -= 500

        return reward
        
    def getQValue(self, state: GameStateFeatures, action: Directions) -> float:
        """
        Return Q(s, a). Unseen pairs default to 0.
        """
        return self.qValues.get((state, action), 0.0)

    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Maximum Q-value over legal actions in a state.
        Returns 0 if no actions are available.
        """
        legal = state.legal
        if not legal:
            return 0.0
        return max(self.getQValue(state, a) for a in legal)

    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Standard Q-learning update:

            Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]
        """
        old_q = self.getQValue(state, action)
        future = self.maxQValue(nextState)
        new_q = old_q + self.alpha * (reward + self.gamma * future - old_q)
        self.qValues[(state, action)] = new_q

    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Increment visit count for (state, action).
        """
        self.counts[(state, action)] = self.getCount(state, action) + 1

    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Return how many times this action was taken in this state.
        """
        return self.counts.get((state, action), 0)

    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Count-based exploration:
        Actions tried fewer than maxAttempts times are forced
        to be attractive by returning a very large value.
        """
        if counts < self.maxAttempts:
            return float("inf")
        return utility 

    def chooseAction(self, stateFeatures:GameStateFeatures) -> Directions:
        """
        Choose an action using epsilon-greedy exploration plus
        count-based scoring.
        """
        legal = stateFeatures.legal
        if not legal:
            return Directions.STOP

        # Random exploration
        if util.flipCoin(self.epsilon):
            return random.choice(legal)

        # Score each action
        scored = []
        for a in legal:
            q = self.getQValue(stateFeatures, a)
            n = self.getCount(stateFeatures, a)
            score = self.explorationFn(q, n)
            scored.append((score, a))

        # Choose among best-scoring actions
        best_score = max(score for score, _ in scored)
        best_actions = [action for score, action in scored if score == best_score]
        return random.choice(best_actions)
        
    def getAction(self, state: GameState) -> Directions:
        """
        Select the next action and update the Q-table using
        the transition from the previous step.
        """
        currentFeatures = GameStateFeatures(state)
        if not currentFeatures.legal:
            return Directions.STOP

        # Learn from previous transition
        if self.lastState is not None and self.lastAction is not None:
            reward = self.computeReward(self.lastState, state)
            self.learn(
                GameStateFeatures(self.lastState),
                self.lastAction,
                reward,
                currentFeatures,
            )
        action = self.chooseAction(currentFeatures)
        # Update visit count
        if action != Directions.STOP:
            self.updateCount(currentFeatures, action)
        # Store transition information
        self.lastState = state
        self.lastAction = action
        return action

    def final(self, state: GameState):
        """
        Called at the end of each episode.

        Handles the final update into a terminal state and
        disables learning after training is complete.
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")
        if self.lastState is not None and self.lastAction is not None:
            reward = self.computeReward(self.lastState, state)
            self.learn(
                GameStateFeatures(self.lastState),
                self.lastAction,
                reward,
                GameStateFeatures(state),
            )
        # Reset stored transition
        self.lastState = None
        self.lastAction = None
        self.incrementEpisodesSoFar()
        # Turn off learning after training phase
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)