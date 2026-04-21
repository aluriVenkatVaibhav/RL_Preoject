import random
from game import Agent
from .q_agent import QAgent

class RLGhostWrapper(Agent):
    """
    Wrapper for RL agents to be used as Ghosts.
    Ghosts need to be able to enable/disable learning for Alternating Training (Phase C).
    """
    def __init__(self, index, learner_class="QAgent", **kwargs):
        super(RLGhostWrapper, self).__init__(index)
        
        # Instantiate the inner learner
        if learner_class == "QAgent":
            self.learner = QAgent(**kwargs)
        else:
            raise NotImplementedError(f"RLGhostWrapper currently only fully supports QAgent. {learner_class} requested.")
            
        self.learner.index = index
        self.is_learning = False # Controlled from the train.py loop
        
        self.last_state = None
        self.last_action = None

    def getAction(self, state):
        action = self.learner.getAction(state)
        self.last_state = state
        self.last_action = action
        return action

    def update(self, state, action, nextState, reward, done=False):
        if self.is_learning:
            self.learner.update(state, action, nextState, reward)

    def final(self, state):
        if hasattr(self.learner, "final"):
            self.learner.final(state)
            
    def set_learning(self, is_learning):
        self.is_learning = is_learning
        if hasattr(self.learner, "is_eval"):
            self.learner.is_eval = not is_learning
