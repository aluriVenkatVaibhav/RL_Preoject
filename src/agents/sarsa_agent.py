from agents.q_agent import QAgent

class SARSAAgent(QAgent):
    """
    Tabular SARSA agent. Inherits from QAgent.
    """
    def __init__(self, alpha=0.1, epsilon=0.05, gamma=0.99, numTraining=10, **kwargs):
        super(SARSAAgent, self).__init__(alpha, epsilon, gamma, numTraining, **kwargs)

    def update(self, state, action, nextState, reward, done=False, **kwargs):
        """
        SARSA Update: Q(s,a) <- Q(s,a) + alpha * (R + gamma * Q(s',a') - Q(s,a))
        If nextAction is None (terminal), target assumes Q(s',a') = 0
        """
        nextAction = kwargs.get('nextAction', None)
        state_tuple = self.state_parser.get_flat_feature_vector(state)
        
        current_q = self.getQValue(state_tuple, action)
        
        if done or nextAction is None: # terminal state
            target = reward
        else:
            next_state_tuple = self.state_parser.get_flat_feature_vector(nextState)
            next_q = self.getQValue(next_state_tuple, nextAction)
            target = reward + self.gamma * next_q
            
        self.q_values[(state_tuple, action)] = current_q + self.alpha * (target - current_q)
