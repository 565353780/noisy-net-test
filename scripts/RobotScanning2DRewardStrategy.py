import numpy as np

class RobotScanning2DRewardStrategy(object):

    def __init__(self, global_map):
        self.global_map = global_map
        self.previous_observation = np.zeros(global_map.shape)
        self.information_gain_weight = 0.6
        self.step_weight = 0.1
        self.obstacle_weight = 1.0

    def compute_reward(self, current_pose, current_observation):

        return self.step_weight * self.compute_step_reward() + \
               self.obstacle_weight * self.compute_obstacle_reward(current_pose) + \
               self.information_gain_weight * self.compute_information_gain_reward(current_observation) 
    
    def compute_step_reward(self):
        return -1.0

    def compute_obstacle_reward(self, current_pose):
        if int(current_pose[0]) < 0 or int(current_pose[0]) >= self.global_map.shape[0] or int(current_pose[1]) < 0 or int(current_pose[1]) >= self.global_map.shape[1]:
            return -1.0
        if self.global_map[int(current_pose[0])][int(current_pose[1])][0] == 0:
            # print("Hit an obstacle!")
            return -1.0
        else:
            return 0.0

    def compute_information_gain_reward(self, current_observation):
        if self.previous_observation is None:
            reward = float(np.sum(current_observation != 0)) / 3.0
        else:
            reward = float(np.sum(current_observation != 0) - np.sum(self.previous_observation != 0)) / 3.0

        self.previous_observation = current_observation.copy()

        return reward