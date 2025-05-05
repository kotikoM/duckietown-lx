import itertools
import random
from typing import List
import numpy as np
# from helper import Axis, get_max_min_interval, check_interval_overlap

from aido_schemas import Context, FriendlyPose
from dt_protocols import (
    Circle,
    CollisionCheckQuery,
    CollisionCheckResult,
    MapDefinition,
    PlacedPrimitive,
    Rectangle,
)

__all__ = ["CollisionChecker"]


class CollisionChecker:
    params: MapDefinition

    def init(self, context: Context):
        context.info("init()")

    def on_received_set_params(self, context: Context, data: MapDefinition):
        context.info("initialized")
        self.params = data

    def on_received_query(self, context: Context, data: CollisionCheckQuery):
        collided = check_collision(
            environment=self.params.environment, robot_body=self.params.body, robot_pose=data.pose
        )
        result = CollisionCheckResult(collided)
        context.write("response", result)


def check_collision(
    environment: List[PlacedPrimitive], robot_body: List[PlacedPrimitive], robot_pose: FriendlyPose
) -> bool:
    # This is just some code to get you started, but you don't have to follow it exactly

    # TODO you can start by rototranslating the robot_body by the robot_pose
    rototranslated_robot: List[PlacedPrimitive] = []
    
    # apply the robot_pose to each primitive in robot_body
    for primitive in robot_body:
        # rototranslate the primitive
        x_w = primitive.pose.x + robot_pose.x
        y_w = primitive.pose.y + robot_pose.y
        theta_deg_w = primitive.pose.theta_deg + robot_pose.theta_deg # in degrees
        pose_w = FriendlyPose(x=x_w, y=y_w, theta_deg=theta_deg_w)
        
        # new premitive
        rototranslated_primitive = PlacedPrimitive(primitive=primitive.primitive, pose=pose_w)
        
        rototranslated_robot.append(rototranslated_primitive)

    # Then, call check_collision_list to see if the robot collides with the environment
    collided = check_collision_list(rototranslated_robot, environment)

    # TODO return the status of the collision
    # for now let's return a random guess
    # return random.uniform(0, 1) > 0.5
    return collided


def check_collision_list(
    rototranslated_robot: List[PlacedPrimitive], environment: List[PlacedPrimitive]
) -> bool:
    # This is just some code to get you started, but you don't have to follow it exactly
    for robot, envObject in itertools.product(rototranslated_robot, environment):
        if check_collision_shape(robot, envObject):
            return True

    return False


def check_collision_shape(a: PlacedPrimitive, b: PlacedPrimitive) -> bool:
    # This is just some code to get you started, but you don't have to follow it exactly

    # TODO check if the two primitives are colliding
    if isinstance(a.primitive, Circle) and isinstance(b.primitive, Circle):
        center_dist = ((a.pose.x - b.pose.x) ** 2 + (a.pose.y - b.pose.y) ** 2) ** 0.5
        return center_dist < a.primitive.radius + b.primitive.radius
    if isinstance(a.primitive, Rectangle) and isinstance(b.primitive, Circle):
        # # get the rectangle's corners
        # xmin, ymin, xmax, ymax = a.primitive.xmin, a.primitive.ymin, a.primitive.xmax, a.primitive.ymax
        # get the circle's center
        x_c, y_c = b.pose.x, b.pose.y
        radius = b.primitive.radius
        
        # get the world coordinates of the rectangle's corners 
        theta_a_rad = np.deg2rad(a.pose.theta_deg)
        x1 = a.pose.x + a.primitive.xmin * np.cos(theta_a_rad) - a.primitive.ymin * np.sin(theta_a_rad)
        y1 = a.pose.y + a.primitive.xmin * np.sin(theta_a_rad) + a.primitive.ymin * np.cos(theta_a_rad)
        x2 = a.pose.x + a.primitive.xmax * np.cos(theta_a_rad) - a.primitive.ymin * np.sin(theta_a_rad)
        y2 = a.pose.y + a.primitive.xmax * np.sin(theta_a_rad) + a.primitive.ymin * np.cos(theta_a_rad)
        x3 = a.pose.x + a.primitive.xmax * np.cos(theta_a_rad) - a.primitive.ymax * np.sin(theta_a_rad)
        y3 = a.pose.y + a.primitive.xmax * np.sin(theta_a_rad) + a.primitive.ymax * np.cos(theta_a_rad)
        x4 = a.pose.x + a.primitive.xmin * np.cos(theta_a_rad) - a.primitive.ymax * np.sin(theta_a_rad)
        y4 = a.pose.y + a.primitive.xmin * np.sin(theta_a_rad) + a.primitive.ymax * np.cos(theta_a_rad)
        points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        
        x_rect = (x1 + x2 + x3 + x4) / 4
        y_rect = (y1 + y2 + y3 + y4) / 4
        
        x_c_in_rect = x_c - x_rect
        y_c_in_rect = y_c - y_rect
        
        # check if the circle's center is inside the extended rectangle
        # if xmin <= x_c_in_rect <= xmax and ymin <= y_c_in_rect <= ymax:
            # return True
        # using separating axis theorem for further check
        n_sample_circle = 10
        # uniformly sample n_sample_circle points on the circle
        thetas = np.linspace(0, 2*np.pi, n_sample_circle)
        x_circ_sample = radius * np.cos(thetas) + x_c
        y_circ_sample = radius * np.sin(thetas) + y_c
        points_circ = list(zip(x_circ_sample, y_circ_sample))
        # for each point create an axis
        axes = []
        for i in range(n_sample_circle):
            axis = Axis(x_circ_sample[i], y_circ_sample[i], x_c, y_c)
            axes.append(axis)
        # for each axis get the max and min interval of the projection of the rectangle's corners
        intervals_rect = []
        for axis in axes:
            intervals_rect.append(get_max_min_interval(axis, points))
        # for each axis get the max and min interval of the projection of the circle's points
        intervals_circ = []
        for axis in axes:
            intervals_circ.append((-radius, radius))
        # check if the intervals overlap
        results = []
        for i in range(n_sample_circle):
            results.append(check_interval_overlap(intervals_rect[i], intervals_circ[i]))
        if all(results):
            return True
        else:
            return False

    if isinstance(a.primitive, Rectangle) and isinstance(b.primitive, Rectangle):
        # get the world coordinates of the rectangle's corners 
        theta_a_rad = np.deg2rad(a.pose.theta_deg)
        x1_a = a.pose.x + a.primitive.xmin * np.cos(theta_a_rad) - a.primitive.ymin * np.sin(theta_a_rad)
        y1_a = a.pose.y + a.primitive.xmin * np.sin(theta_a_rad) + a.primitive.ymin * np.cos(theta_a_rad)
        x2_a = a.pose.x + a.primitive.xmax * np.cos(theta_a_rad) - a.primitive.ymin * np.sin(theta_a_rad)
        y2_a = a.pose.y + a.primitive.xmax * np.sin(theta_a_rad) + a.primitive.ymin * np.cos(theta_a_rad)
        x3_a = a.pose.x + a.primitive.xmax * np.cos(theta_a_rad) - a.primitive.ymax * np.sin(theta_a_rad)
        y3_a = a.pose.y + a.primitive.xmax * np.sin(theta_a_rad) + a.primitive.ymax * np.cos(theta_a_rad)
        x4_a = a.pose.x + a.primitive.xmin * np.cos(theta_a_rad) - a.primitive.ymax * np.sin(theta_a_rad)
        y4_a = a.pose.y + a.primitive.xmin * np.sin(theta_a_rad) + a.primitive.ymax * np.cos(theta_a_rad)
        points_a = [(x1_a, y1_a), (x2_a, y2_a), (x3_a, y3_a), (x4_a, y4_a)]
        
        x_a = (x1_a + x2_a + x3_a + x4_a) / 4
        y_a = (y1_a + y2_a + y3_a + y4_a) / 4
        
        # get the world coordinates of the rectangle's corners
        theta_b_rad = np.deg2rad(b.pose.theta_deg)
        x1_b = b.pose.x + b.primitive.xmin * np.cos(theta_b_rad) - b.primitive.ymin * np.sin(theta_b_rad)
        y1_b = b.pose.y + b.primitive.xmin * np.sin(theta_b_rad) + b.primitive.ymin * np.cos(theta_b_rad)
        x2_b = b.pose.x + b.primitive.xmax * np.cos(theta_b_rad) - b.primitive.ymin * np.sin(theta_b_rad)
        y2_b = b.pose.y + b.primitive.xmax * np.sin(theta_b_rad) + b.primitive.ymin * np.cos(theta_b_rad)
        x3_b = b.pose.x + b.primitive.xmax * np.cos(theta_b_rad) - b.primitive.ymax * np.sin(theta_b_rad)
        y3_b = b.pose.y + b.primitive.xmax * np.sin(theta_b_rad) + b.primitive.ymax * np.cos(theta_b_rad)
        x4_b = b.pose.x + b.primitive.xmin * np.cos(theta_b_rad) - b.primitive.ymax * np.sin(theta_b_rad)
        y4_b = b.pose.y + b.primitive.xmin * np.sin(theta_b_rad) + b.primitive.ymax * np.cos(theta_b_rad)
        points_b = [(x1_b, y1_b), (x2_b, y2_b), (x3_b, y3_b), (x4_b, y4_b)]
        
        # get the center point of the edges
        x1_e_1 = (x1_a + x2_a) / 2
        y1_e_1 = (y1_a + y2_a) / 2
        x2_e_2 = (x2_a + x3_a) / 2
        y2_e_2 = (y2_a + y3_a) / 2
        
        x_b = (x1_b + x2_b + x3_b + x4_b) / 4
        y_b = (y1_b + y2_b + y3_b + y4_b) / 4
        
        # use the center of the first rectangle as the origin
        # use center to center vector as the projection axis
        axis_1 = Axis(x1_a, y1_a, x_a, y_a)
        axis_2 = Axis(x2_a, y2_a, x_a, y_a)
        axis_3 = Axis(x1_e_1, y1_e_1, x_a, y_a)
        axis_4 = Axis(x2_e_2, y2_e_2, x_a, y_a)
        
        axes = [axis_1, axis_2, axis_3, axis_4]
        results = []
        for i in range(len(axes)):
            interval_a = get_max_min_interval(axes[i], points_a)
            interval_b = get_max_min_interval(axes[i], points_b)
            results.append(check_interval_overlap(interval_a, interval_b))
        # if any of the axes does not overlap, then the rectangles do not overlap
        if all(results):
            return True
        else:
            return False

    # TODO return the status of the collision
    # for now let's return a random guess
    # return random.uniform(0, 1) > 0.5


class Axis:
    def __init__(self, x_dir, y_dir, xo=0, yo=0,):
        # origin
        self.xo= xo
        self.yo= yo
        # direction
        x_dir= x_dir - xo
        y_dir= y_dir - yo
        # normalize
        self.x_dir= x_dir/np.linalg.norm([x_dir, y_dir])
        self.y_dir= y_dir/np.linalg.norm([x_dir, y_dir])
        
        self.dir_vec = np.array([self.x_dir, self.y_dir])
        # print(self.dir_vec)
    
    def project_of(self, x, y):
        return np.dot([x-self.xo, y-self.yo], self.dir_vec)
    
def get_max_min_interval(axis, points):
    '''
    get the max and min value of the projection of points on the axis
    '''
    min_val= np.inf
    max_val= -np.inf
    for point in points:
        # print(axis.project_of(*point))
        min_val= min(min_val, axis.project_of(*point))
        max_val= max(max_val, axis.project_of(*point))
    return (min_val, max_val)

def check_interval_overlap(interval1, interval2):
    '''
    check if two intervals overlap
    '''
    return interval1[1] >= interval2[0] and interval2[1] >= interval1[0]