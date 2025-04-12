import os
import sys
import scipy
import numpy as np
import glob
import json
import cv2

sys.path.append('../')


IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
VIEW_FOV = 100

BB_COLOR = (248, 64, 24)  # the color for drawing bounding boxes


class Agent(object):
    def __init__(self, measure):
        super(Agent, self).__init__()
        self.attr = {}
        for k, v in measure.items():
            if isinstance(v, dict):
                self.attr[k] = Agent(v)
            else:
                self.attr[k] = v

    def __getattr__(self, item):
        return self.attr[str(item)]

    def get_transform(self):
        if 'transform' in self.attr:
            return self.attr['transform']
        else:
            raise Exception

    def __str__(self):
        return self.attr.__str__()


# ==============================================================================
# -- BoundingBoxesTransform ---------------------------------------------------
# ==============================================================================


class BoundingBoxesTransform(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    @staticmethod
    def get_bounding_boxes(nonplayers: list, camera, player):
        """
        Creates 3D bounding boxes based on nonplayers list and camera.
        :param nonplayers: the list of non-player-agent objects
        :param camera: camera object
        :param player: player object, i.e. the ego car
        :return: the list of coordinates of bounding boxes (each has 8 vertexes), the format is like:
            [matrix([[x0,y0],[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5],[x6,y6],[x7,y7]]), matrix(...), ...]
        """
        bounding_boxes = [BoundingBoxesTransform.get_bounding_box(nonplayer, camera, player) for nonplayer in nonplayers]
        return bounding_boxes

    @staticmethod
    def draw_3D_bounding_boxes(image: np.matrix, bounding_boxes: list):
        """
        Draws 3D bounding boxes on the input image.
        Do not modify this function! Adjust the format of your bounding boxes to fit this function.
        :param image: image matrix
        :param bounding_boxes: a list of bounding box coordinates
        :return: image array: np.matrix
        """
        for bbox in bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            # draw lines
            # base
            cv2.line(image, points[0], points[1], color=BB_COLOR, thickness=2)
            cv2.line(image, points[1], points[2], color=BB_COLOR, thickness=2)
            cv2.line(image, points[2], points[3], color=BB_COLOR, thickness=2)
            cv2.line(image, points[3], points[0], color=BB_COLOR, thickness=2)
            # top
            cv2.line(image, points[4], points[5], color=BB_COLOR, thickness=2)
            cv2.line(image, points[5], points[6], color=BB_COLOR, thickness=2)
            cv2.line(image, points[6], points[7], color=BB_COLOR, thickness=2)
            cv2.line(image, points[7], points[4], color=BB_COLOR, thickness=2)
            # base-top
            cv2.line(image, points[0], points[4], color=BB_COLOR, thickness=2)
            cv2.line(image, points[1], points[5], color=BB_COLOR, thickness=2)
            cv2.line(image, points[2], points[6], color=BB_COLOR, thickness=2)
            cv2.line(image, points[3], points[7], color=BB_COLOR, thickness=2)
        return image
    def rotate_nonplayer(nonplayer: Agent):
        bb_cords = BoundingBoxesTransform._create_bb_points(nonplayer)
        transform = BoundingBoxesTransform._complete_transform(nonplayer.get_transform())
        transform_box = nonplayer.boundingBox.transform
        transform_box = BoundingBoxesTransform._complete_transform(transform_box)
        yaw_nonplayer = transform.rotation.yaw
        pitch_nonplayer = transform.rotation.pitch
        roll_nonplayer = transform.rotation.roll
        for i in  range(8):
            bb_cords[i, 0] += transform_box.location.x
            bb_cords[i, 1] += transform_box.location.y
            bb_cords[i, 2] += transform_box.location.z
        yaw_nonplayer_matrix = np.array([
            [np.cos(np.deg2rad(yaw_nonplayer)), -np.sin(np.deg2rad(yaw_nonplayer)), 0],
            [np.sin(np.deg2rad(yaw_nonplayer)), np.cos(np.deg2rad(yaw_nonplayer)), 0],
            [0, 0, 1]
        ])
        pitch_nonplayer_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(np.deg2rad(pitch_nonplayer)), -np.sin(np.deg2rad(pitch_nonplayer))],
            [0, np.sin(np.deg2rad(pitch_nonplayer)), np.cos(np.deg2rad(pitch_nonplayer))]
        ])
        roll_nonplayer_matrix = np.array([
            [np.cos(np.deg2rad(roll_nonplayer)), -np.sin(np.deg2rad(roll_nonplayer)), 0],
            [np.sin(np.deg2rad(roll_nonplayer)), np.cos(np.deg2rad(roll_nonplayer)), 0],
            [0, 0, 1]
        ])
        # Combine rotations 
        R_nonplayer = np.dot(np.dot(roll_nonplayer_matrix, pitch_nonplayer_matrix), yaw_nonplayer_matrix)
        # Apply rotation
        for i in range(8):
            bb_cords[i,:3]= np.dot(R_nonplayer, bb_cords[i, :3])
            # Apply translation
            bb_cords[i, 3] = 1.0
        return bb_cords
    @staticmethod
    def get_bounding_box(nonplayer: Agent, camera: Agent, player: Agent):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        :param nonplayer: the non-player object
        :param camera: the camera object
        :param player: the player object, i.e. the ego vehicle
        :return: the 2D coordinates of the bounding box vertexes
        """
        # Get 3D coordinates of bounding box
        bb_cords = BoundingBoxesTransform.rotate_nonplayer(nonplayer)
            
        # Transform the object coordinates to the camera's view
        # Get transforms
        player_transform = BoundingBoxesTransform._complete_transform(player.get_transform())
        nonplayer_transform = BoundingBoxesTransform._complete_transform(nonplayer.get_transform())
        camera_transform = BoundingBoxesTransform._complete_transform(camera.get_transform())
        
        # Use the calibration matrix to project points
        K = camera.calibration
        
        # Simple projection using camera calibration matrix
        screen_coords = np.zeros((8, 2))
        rel_z = nonplayer_transform.location.x - player_transform.location.x
        rel_y = nonplayer_transform.location.z - player_transform.location.z
        rel_x = nonplayer_transform.location.y - player_transform.location.y
        for i in range(8):
            point_3d = np.array([bb_cords[i, 1], -bb_cords[i, 2], bb_cords[i, 0]])
            point_3d[0] += rel_x
            point_3d[1] -= rel_y
            point_3d[2] += rel_z
            point_homogeneous = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
            yaw_matrix = np.array([
                [np.cos(np.deg2rad(player_transform.rotation.yaw)),0 , -np.sin(np.deg2rad(player_transform.rotation.yaw))],
                [0, 1, 0],
                [+np.sin(np.deg2rad(player_transform.rotation.yaw)), 0, np.cos(np.deg2rad(player_transform.rotation.yaw))]
            ])
            pitch_matrix_player = np.array([
                [1, 0, 0],
                [0, np.cos(np.deg2rad(player_transform.rotation.pitch)), -np.sin(np.deg2rad(player_transform.rotation.pitch))],
                [0, +np.sin(np.deg2rad(player_transform.rotation.pitch)), np.cos(np.deg2rad(player_transform.rotation.pitch))]
            ])
            roll_matrix_player = np.array([
                [np.cos(np.deg2rad(player_transform.rotation.roll)), +np.sin(np.deg2rad(player_transform.rotation.roll)), 0],
                [-np.sin(np.deg2rad(player_transform.rotation.roll)), np.cos(np.deg2rad(player_transform.rotation.roll)), 0],
                [0, 0, 1]
            ])
            new_point = np.dot(np.dot(roll_matrix_player, pitch_matrix_player),np.dot(yaw_matrix, point_homogeneous[:3]))
            point_3d[0] = new_point[0]
            point_3d[1] = new_point[1]
            point_3d[2] = new_point[2]
            camera_player_transform = {'transform': {'location': {'x': 2.0, 'y': 0.0, 'z': 1.4},
                                                 'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                                                 'rotation': {"pitch": -15, "roll": 0.0, "yaw": 0.0}}}
            point_3d[2] -= camera_player_transform['transform']['location']['x']
            point_3d[1] += camera_player_transform['transform']['location']['z']
            angle = camera_player_transform['transform']['rotation']['pitch']
            pitch_matrix = np.array([   
                [1, 0, 0],
                [0, np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))],
                [0, -np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]
            ])
            point_3d = np.dot(pitch_matrix, point_3d)
            # Simple perspective division
            depth = max(0.001, point_3d[2])  # avoid division by zero
            # Apply calibration matrix
            px = K[0, 0] * (point_3d[0] / depth) + K[0, 2]
            py = K[1, 1] * (point_3d[1]/ depth) + K[1, 2] 
            
            screen_coords[i, 0] = px
            screen_coords[i, 1] = py
            
        return screen_coords
        non_player_ground_bb_points = BoundingBoxesTransform.non_player_ground_bb_points(nonplayer)
        #non_player_player_bb_points = BoundingBoxesTransform.ground_bb2player(non_player_ground_bb_points, player)
        non_player_camera_bb_points = BoundingBoxesTransform.player2camera(non_player_ground_bb_points, camera)
        screen_coords = BoundingBoxesTransform.player2picture(non_player_camera_bb_points)
        return screen_coords

    @staticmethod
    def _create_bb_points(nonplayer):
        """
        Returns 3D bounding box for a non-player-agent, relative to the vehicle coordinate system.
        """

        cords = np.zeros((8, 4))
        extent = nonplayer.boundingBox.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords
    def non_player_ground_bb_points(nonplayer):
        transform = BoundingBoxesTransform._complete_transform(nonplayer.get_transform())
        real_cords = np.zeros((8, 4))
        yaw = transform.rotation.yaw
        pitch = transform.rotation.pitch
        roll = transform.rotation.roll
        # Convert yaw, pitch, roll to radians
        yaw = np.deg2rad(yaw)
        pitch = np.deg2rad(pitch)
        roll = np.deg2rad(roll)
        # Create rotation matrices
        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
    
        R_pitch = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        final_transform = np.dot(np.dot(R_yaw, R_pitch), R_roll)
        prev_cords = BoundingBoxesTransform._create_bb_points(nonplayer)
        for i in range(8):
            # Apply rotation
            rotated_cord = np.dot(final_transform, prev_cords[i, :3])
            # Apply translation
            real_cords[i, :3] = rotated_cord + np.array([transform.location.x, transform.location.y, transform.location.z])
            real_cords[i, 3] = 1.0
        return real_cords
    def ground_bb2player(real_cords, player):
        """
        Returns the 3D bounding box for a non-player-agent, relative to the player vehicle coordinate system.
        """
        player_transform = BoundingBoxesTransform._complete_transform(player.get_transform())
        cords = np.zeros((8, 4))
        rel_x = real_cords[0, 0] - player_transform.location.x
        rel_y = real_cords[0, 1] - player_transform.location.y
        rel_z = real_cords[0, 2] - player_transform.location.z
        for i in range(8):
            cords[i, 0] = real_cords[i, 0] + rel_x
            cords[i, 1] = real_cords[i, 1] + rel_y
            cords[i, 2] = real_cords[i, 2] + rel_z
            cords[i, 3] = real_cords[i, 3]
        transform = BoundingBoxesTransform._complete_transform(player.get_transform())
        yaw = transform.rotation.yaw
        pitch = transform.rotation.pitch
        roll = transform.rotation.roll
        yaw = np.deg2rad(yaw)
        pitch = np.deg2rad(pitch)
        roll = np.deg2rad(roll)
        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
    
        R_pitch = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        final_transform = np.dot(np.dot(R_yaw, R_pitch), R_roll)
        inverse_transform = np.linalg.inv(final_transform)
        for i in range(8):
            rotated_cord = np.dot(inverse_transform, cords[i, :3])
            cords[i, :3] = rotated_cord 
            cords[i, 3] = 1.0
        return cords
    def player2camera(cords, camera):
        """
        Returns the 3D bounding box for a non-player-agent, relative to the camera coordinate system.
        """
        # Create a copy to avoid modifying the original coordinates
        result_cords = cords.copy()
        
        camera_transform = BoundingBoxesTransform._complete_transform(camera.get_transform())
        
        # Apply translation (subtract camera position, not add)
        for i in range(8):
            result_cords[i, 0] -= camera_transform.location.x
            result_cords[i, 1] -= camera_transform.location.y
            result_cords[i, 2] -= camera_transform.location.z
        
        # Get rotation parameters
        yaw = np.deg2rad(camera_transform.rotation.yaw)
        pitch = np.deg2rad(camera_transform.rotation.pitch)
        roll = np.deg2rad(camera_transform.rotation.roll)
        
        # Create rotation matrices
        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        R_pitch = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # Combine rotations
        R = np.dot(np.dot(R_yaw, R_pitch), R_roll)
        
        # Apply inverse rotation
        R_inv = np.linalg.inv(R)
        for i in range(8):
            result_cords[i, :3] = np.dot(R_inv, result_cords[i, :3])
        
        return result_cords
    @staticmethod
    def player2picture(cords):
        calibration = np.array([
            [335.6, 0,   400],  # fx, 0, cx
            [0,   335.6, 300],  # 0, fy, cy 
            [0,     0,     1]   # 0, 0,  1
        ])
        
        screen_coords = np.zeros((8, 2))
        for i in range(8):
            point_3d = np.array([cords[i, 0], cords[i, 1], cords[i, 2]])
            
            # For depth perception in the new coordinate system 
            # X is forward, so use that for depth
            depth = max(0.1, point_3d[0])  
            
            # Z axis (side direction) maps to screen X
            px = calibration[0, 0] * (point_3d[2] / depth) + calibration[0, 2]
            
            # -Y axis (negative up direction) maps to screen Y (because screen Y increases downward)
            py = calibration[1, 1] * (-point_3d[1] / depth) + calibration[1, 2]
            
            screen_coords[i, 0] = px
            screen_coords[i, 1] = py
        
        return screen_coords
        
    @staticmethod
    def _complete_transform(transform):
        """
        Complete the missing items in transform so avoid raising errors. Maybe useful for you.
        """
        if 'x' not in transform.location.attr:
            transform.location.x = 0.0
        if 'y' not in transform.location.attr:
            transform.location.y = 0.0
        if 'z' not in transform.location.attr:
            transform.location.z = 0.0
        if 'yaw' not in transform.rotation.attr:
            transform.rotation.yaw = 0.0
        if 'roll' not in transform.rotation.attr:
            transform.rotation.roll = 0.0
        if 'pitch' not in transform.rotation.attr:
            transform.rotation.pitch = 0.0
        return transform
    

def set_calibration(camera):
    """
    get the camera calibration matrix
    :param camera: the camera agent object
    :return: camera
    """
    calibration = np.array([
        [335.6, 0,   400],  # fx, 0, cx
        [0,   335.6, 300],  # 0, fy, cy 
        [0,     0,     1]   # 0, 0,  1
    ])
    """
    complete the camera calibration matrix here
    """
    camera.calibration = calibration
    return camera


def filte_out_near_nonplayer(nonPlayerAgents, playerAgent, threshold=50):
    player_location = np.array([playerAgent.transform.location.x, playerAgent.transform.location.y, playerAgent.transform.location.z])
    near_nonPlayerAgents = []
    for nonplayer in nonPlayerAgents:
        nonplayer_location = np.array([nonplayer.transform.location.x, nonplayer.transform.location.y, nonplayer.transform.location.z])
        dis = np.linalg.norm(player_location - nonplayer_location)
        if dis <= threshold:
            near_nonPlayerAgents.append(nonplayer)
            print("nonPlayerAgent: ", nonplayer.transform.rotation)
    print ("--------------------------------------------")
    return near_nonPlayerAgents


def main():
    img_path_list = glob.glob('./data/image_*.png')
    img_path_list.sort()
    measurement_path_list = glob.glob('./data/measurements_*.json')
    measurement_path_list.sort()

    for img_path, measurement_path in zip(img_path_list, measurement_path_list):
        idx = img_path.split('_')[1].split('.')[0]
        image = cv2.imread(img_path)

        with open(measurement_path, 'r') as f:
            measurement = json.load(f)

        nonPlayerAgents = []
        for item in measurement['nonPlayerAgents']:
            if 'vehicle' in item:
                nonPlayerAgents.append(Agent(item['vehicle']))
            elif 'pedestrian' in item:
                nonPlayerAgents.append(Agent(item['pedestrian']))

        playerAgent = Agent(measurement['playerMeasurements'])

        nonPlayerAgents = filte_out_near_nonplayer(nonPlayerAgents, playerAgent)

        camera_player_transform = {'transform': {'location': {'x': 2.0, 'y': 0.0, 'z': 1.4},
                                                 'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                                                 'rotation': {"pitch": -15.0, "roll": 0.0, "yaw": 0.0}}}
        camera_agent = Agent(camera_player_transform)
        camera_agent = set_calibration(camera_agent)

        bounding_boxes = BoundingBoxesTransform.get_bounding_boxes(nonPlayerAgents, camera_agent, playerAgent)
        result = BoundingBoxesTransform.draw_3D_bounding_boxes(image, bounding_boxes)

        cv2.imwrite('test_{}.png'.format(idx), result)


if __name__ == '__main__':
    main()
