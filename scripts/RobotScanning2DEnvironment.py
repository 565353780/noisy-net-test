from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, ceil
# from gym import spaces
from cv2 import namedWindow, imshow, WINDOW_AUTOSIZE, waitKey
from random import randint

from RobotScanning2DRewardStrategy import RobotScanning2DRewardStrategy as RewardStrategy
from RobotScanning2DDoneStrategy import RobotScanning2DDoneStrategy as DoneStrategy

# label of environment[i][j]:
#     [0]:
#         0:can not go to
#         1:can go to
#     [1]:
#         0:do not know the message
#         1:got the messages

class RobotScanning2DEnvironment(object):
    def __init__(self, map_file, angle=50.0, distance=4, delta_angle=0.01):
        self.raw_map_file = map_file
        self.observation = None
        self.global_map = None
        self.output_image = None

        self.sight_angle = angle
        self.sight_distance = distance
        self.delta_ray_angle = delta_angle
        self.init_pose = np.array([1.0, 1.0, 0.0], dtype=np.float64)
        self.current_pose = self.init_pose.copy()

        self.action_space = None
        self.action_spaces = None

        self.max_w = 200
        self.max_h = 200

        self.load_map()
        self.init_action_space()

        self.reward_strategy = RewardStrategy(self.global_map)
        self.done_strategy = DoneStrategy(self.global_map)

        self.path = [self.current_pose.copy()]

        self.SHOW_IMAGE = False
        self.namedWindow = None

        self.hit_obstacle_time = 0
        self.max_hit_obstacle_time = 10
        self.total_move_num = 0

        self.test_idx = 0
        self.init_pose_fixed = False

    def init_action_space(self):
        # self.action_space = spaces.Discrete(8)
        self.action_spaces = np.empty((8, 3), dtype=np.float64)
        self.action_spaces[0] = [1.0, 0.0, 90.0]
        self.action_spaces[1] = [-1.0, 0.0, -90.0]
        self.action_spaces[2] = [0.0, -1.0, 180.0]
        self.action_spaces[3] = [0.0, 1.0, 0.0]
        self.action_spaces[4] = [1.0, -1.0, 135.0]
        self.action_spaces[5] = [1.0, 1.0, 45.0]
        self.action_spaces[6] = [-1.0, -1.0, -135.0]
        self.action_spaces[7] = [-1.0, 1.0, -45.0]
        # self.action_spaces[8] = [2.0, 0.0, 90.0]
        # self.action_spaces[9] = [-2.0, 0.0, -90.0]
        # self.action_spaces[10] = [0.0, -2.0, 180.0]
        # self.action_spaces[11] = [0.0, 2.0, 0.0]

    def get_num_actions(self):
        return len(self.action_spaces)
        # return self.action_space.n

    def step(self, action):
        if action not in range(len(self.action_spaces)):
        # if action not in range(self.action_space.n):
            print("Error action!")
            return (None, None, None, None)
        else:
            self.total_move_num += 1
            target_pose = self.current_pose.copy()
            target_pose[:2] += self.action_spaces[action][:2]

            if not 0 <= int(target_pose[0]) < self.global_map.shape[0] or not 0 <= int(target_pose[1]) < self.global_map.shape[1]:

                self.hit_obstacle_time = 0

                return (self.observation, -100.0, True, None)

            if self.global_map[int(target_pose[0])][int(target_pose[1])][0] == 0 and self.hit_obstacle_time < self.max_hit_obstacle_time:

                self.hit_obstacle_time += 1

                return (self.observation, -10.0, False, None)

            target_pose[2] = self.action_spaces[action][2]

            self.update_observation(target_pose)

            self.current_pose = target_pose.copy()

            reward = self.reward_strategy.compute_reward(self.current_pose, self.observation)
            done = self.done_strategy.is_done(self.current_pose, self.observation)

            last_path = self.path[len(self.path) - 1]

            self.path.append(self.current_pose.copy())

            self.observation[int(last_path[0])][int(last_path[1])] = [255, 255, 255]
            self.observation[int(self.current_pose[0])][int(self.current_pose[1])] = [255, 0, 0]

            if done:

                self.hit_obstacle_time = 0

                print(self.total_move_num)
                self.total_move_num = 0

            return (self.observation, reward, done, None)

    def reset(self):
        self.observation = np.zeros(self.global_map.shape, dtype=np.int32)
        # self.update_observation(self.init_pose)

        if self.init_pose_fixed:
            test_pose = [[int(0.45 * self.max_h), int(0.1 * self.max_w), 0], [int(0.55 * self.max_h), int(0.1 * self.max_w), 0]]

            random_pose = test_pose[self.test_idx]
            self.test_idx = (self.test_idx + 1) % len(test_pose)

        else:
            rand_w_l = int(0.1 * self.max_w)
            rand_w_r = int(0.9 * self.max_w)
            rand_h_l = int(0.1 * self.max_h)
            rand_h_r = int(0.9 * self.max_h)
            random_pose = np.array([randint(rand_h_l, rand_h_r), randint(rand_w_l, rand_w_r), 0], dtype=np.float64)
            while self.global_map[int(random_pose[0])][int(random_pose[1])][0] == 0:
                random_pose[0] = randint(rand_h_l, rand_h_r)
                random_pose[1] = randint(rand_w_l, rand_w_r)

        self.init_pose = random_pose.copy()

        self.current_pose = self.init_pose.copy()

        self.path = []
        self.path.append(self.current_pose.copy())

        return self.observation

    def load_map(self):

        img = Image.open(self.raw_map_file)
        img = img.convert("RGB")
        max_scale = self.max_w / img.size[0]
        if self.max_h / img.size[1] < max_scale:
            max_scale = self.max_h / img.size[1]

        if max_scale < 1:
            img = img.resize((int(img.size[0]*max_scale), int(img.size[1]*max_scale)), resample=Image.LANCZOS)

        self.max_w = img.size[0]
        self.max_h = img.size[1]

        self.global_map = self.transform_between_image_coordinate_and_map_coordinate(np.array(img))

        for i in range(self.global_map.shape[0]):
            for j in range(self.global_map.shape[1]):
                if self.global_map[i][j][0] < 200 or self.global_map[i][j][1] < 200 or self.global_map[i][j][2] < 200:
                    self.global_map[i][j] = [0, 0, 0]
                else:
                    self.global_map[i][j] = [255, 255, 255]

        self.add_boundary()

        # imshow('test', self.global_map)
        # waitKey()

    def add_boundary(self):
        self.global_map[0] = 0
        self.global_map[-1] = 0
        self.global_map[:, 0] = 0
        self.global_map[:, -1] = 0

    def transform_between_image_coordinate_and_map_coordinate(self, input_array):
        output_array = np.empty(input_array.shape, dtype=np.uint8)
        height = input_array.shape[0]
        width = input_array.shape[1]
        for i in range(height):
            for j in range(width):
                output_array[i][j] = np.uint8(input_array[height - 1 - i][j])
        
        return output_array

    def add_new_end_point(self, end_points, center_position, ray_angle):
        new_end_point_x = int(center_position[0] + self.sight_distance * sin(ray_angle * pi / 180.0))
        new_end_point_y = int(center_position[1] + self.sight_distance * cos(ray_angle * pi / 180.0))
        if end_points is None or len(end_points) == 0:
            return np.array([[new_end_point_x, new_end_point_y]], dtype=np.int32)
        else:
            if new_end_point_x != end_points[-1, 0] or new_end_point_y != end_points[-1, 1]:
                return np.append(end_points, [[new_end_point_x, new_end_point_y]], axis=0)
            else:
                return end_points

    def compute_ray_end_points(self, target_pose):
        ray_angle_right = target_pose[2] - self.sight_angle / 2.0
        ray_angle_left = target_pose[2] + self.sight_angle / 2.0
        # ray_num = ceil((ray_angle_left - ray_angle_right) / self.delta_ray_angle) + 1

        # end_points = self.add_new_end_point(None, target_pose[:2], ray_angle_right)
        end_points = None

        #for i in range(1, ray_num - 1):
        for ray_angle in np.arange(ray_angle_right, ray_angle_left, self.delta_ray_angle):
            end_points = self.add_new_end_point(end_points, target_pose[:2], ray_angle)

        end_points = self.add_new_end_point(end_points, target_pose[:2], ray_angle_left)

        return end_points

    def end_points_based_ray_cast(self, target_pose):
        end_points = self.compute_ray_end_points(target_pose)

        for i in range(len(end_points)):
            ray = end_points[i] - target_pose[:2]
            self.single_ray_cast(target_pose[:2], ray)

    def single_ray_cast(self, start_point, ray):
        long_axis_length = np.max(np.abs(ray))
        moving_unit = ray / long_axis_length

        for j in range(int(long_axis_length) + 1):
            if self.global_map[int(start_point[0] + moving_unit[0] * j)][int(start_point[1] + moving_unit[1] * j)][0] == 0:
                self.observation[int(start_point[0] + moving_unit[0] * j)][int(start_point[1] + moving_unit[1] * j)][:] = 128
                break
            else:
                self.observation[int(start_point[0] + moving_unit[0] * j)][int(start_point[1] + moving_unit[1] * j)][:] = 255

    def uniform_angle_based_ray_cast(self, target_pose):
        ray_angle_right = target_pose[2] - self.sight_angle / 2.0
        ray_angle_left = target_pose[2] + self.sight_angle / 2.0

        for ray_angle in np.arange(ray_angle_right, ray_angle_left, self.delta_ray_angle):
            ray = [self.sight_distance * sin(ray_angle * pi / 180.0), self.sight_distance * cos(ray_angle * pi / 180.0)]
            self.single_ray_cast(target_pose[:2], ray)

        ray = [self.sight_distance * sin(ray_angle_left * pi / 180.0), self.sight_distance * cos(ray_angle_left * pi / 180.0)]
        self.single_ray_cast(target_pose[:2], ray)

    def update_observation(self, target_pose):
        if not 0 <= target_pose[0] < self.global_map.shape[0] or not 0 <= target_pose[1] < self.global_map.shape[1]:
            # print("Target pose out of range!")
            return

        self.end_points_based_ray_cast(target_pose)
        # self.uniform_angle_based_ray_cast(target_pose)

        self.current_pose = target_pose.copy()

    def paint_color(self):
        self.output_image = self.global_map.copy()
        # self.output_image[np.where(self.observation == 1)] = 255
        for i in range(self.output_image.shape[0]):
            for j in range(self.output_image.shape[1]):
                if self.observation[i][j][0] == 255:
                    self.output_image[i][j] = [0, 0, 255]

        for pose in self.path:
            self.output_image[int(pose[0])][int(pose[1])] = [0, 255, 0]

        self.output_image = self.transform_between_image_coordinate_and_map_coordinate(self.output_image)

    def render(self):
        self.paint_color()

        if not self.SHOW_IMAGE:
            self.namedWindow = namedWindow('test_environment', WINDOW_AUTOSIZE)
            self.SHOW_IMAGE = True

        imshow('test_environment', self.output_image)
        waitKey(1)

        # plt.imshow(self.output_image)
        # plt.axis('off')
        # plt.show()

    def test(self, position, forward_direction, show_image=False):

        if self.global_map is None:

            self.reset()

        target_pose = [position[0], position[1], forward_direction]
        self.update_observation(target_pose)

        if show_image:
            # self.show_output_image()
            self.render()

if __name__ == "__main__":
    env = RobotScanning2DEnvironment(map_file="./test.png", angle=120.0, distance=35, delta_angle=0.2)
    env.reset()

    env.test(position=[70, 10], forward_direction=0, show_image=True)
    env.reset()
    # env.test(position=[7, 7], forward_direction=0, show_image=True)
    # env.test(position=[40, 7], forward_direction=30, show_image=True)