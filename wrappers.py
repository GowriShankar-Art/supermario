import gym

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip
        self.current_score = 0
        self.x = 0

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += self.RewardUpdate(reward, info)
            if done:
                break
        return obs, total_reward, done, info

    def RewardUpdate(self,reward, info):

        # Removing the dependency on velocity
        reward = reward - (info['x_pos'] - self.x)
        # Constant value of Reward for moving right
        if(info['x_pos'] > self.x):
            reward += 0.1
        # Reward to increase depending on the score
        reward += (info['score'] - self.current_score)/20
        
        self.current_score = info['score']
        self.x = info['x_pos']
        return reward
