class Memory:
    def __init__(self): 
        self.clear()

  # Resets/restarts the memory buffer
    def clear(self): 
        self.observations = []
        self.actions = []
        self.rewards = []

  # Add observations, actions, rewards to memory
    def add_to_memory(self, new_observation, new_action, new_reward): 
        self.observations.append(new_observation)
        '''TODO: update the list of actions with new action'''
        self.actions.append(new_action) # TODO
        # ['''TODO''']
        '''TODO: update the list of rewards with new reward'''
        self.rewards.append(new_reward) # TODO
        # ['''TODO''']

    def __len__(self):
        return len(self.actions)

