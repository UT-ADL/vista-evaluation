# Vista's file
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
        self.actions.append(new_action)
        self.rewards.append(new_reward)

    def __len__(self):
        return len(self.actions)

