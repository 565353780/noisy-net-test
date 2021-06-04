import numpy as np
import time

class RobotScanning2DDoneStrategy(object):

    def __init__(self, global_map):
        self.global_map = global_map
        self.completion_rate = 0.8
        # self.last_finish_time = int(time.time())
        self.global_free_space_size = None

        self.compute_global_free_space_size()

    def is_done(self, current_pose, current_observation):
        # if (int(time.time()) - self.last_finish_time) % 10 == 0:
        #     self.completion_rate = self.completion_rate - 0.0001 * (int(time.time()) - self.last_finish_time)/10
        #     self.last_finish_time = int(time.time())
        #     print(self.compute_observed_free_space_size(current_observation) / self.global_free_space_size, ' / ', self.completion_rate)
        #     self.last_finish_time = int(time.time())
        if self.compute_observed_free_space_size(current_observation) / self.global_free_space_size > self.completion_rate:
        #     if self.completion_rate < 0.95:
        #         self.completion_rate = self.completion_rate + 0.01
            print('Up to : ', self.completion_rate)
            # self.last_finish_time = int(time.time())
            return True
        if int(current_pose[0]) < 0 or int(current_pose[0]) >= self.global_map.shape[0] or int(current_pose[1]) < 0 or int(current_pose[1]) >= self.global_map.shape[1]:
            return True
        elif self.global_map[int(current_pose[0])][int(current_pose[1])][0] == 0:
            return True
        else:
            return False

    def compute_observed_free_space_size(self, observation):
        return float(np.sum(observation == 255)) / 3.0

    def compute_global_free_space_size(self):
        if self.global_free_space_size is None:
            self.global_free_space_size = float(np.sum(self.global_map == 255)) / 3.0