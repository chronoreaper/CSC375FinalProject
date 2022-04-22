from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
import random
import os
from gym import spaces
import time
import pybullet as p
import kukaclaw
import numpy as np
import pybullet_data
import pdb
import distutils.dir_util
import glob
from pkg_resources import parse_version
import gym

class ClawEnv(KukaGymEnv):
  """Class for Kuka environment with diverse objects.

  In each episode some objects are chosen from a set of 1000 diverse objects.
  These 1000 objects are split 90/10 into a train and test set.
  """

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=80,
               isEnableSelfCollision=True,
               renders=False,
               isDiscrete=False,
               maxSteps=8,
               dv=0.06,
               removeHeightHack=False,
               blockRandom=0.3,
               cameraRandom=0,
               width=48,
               height=48,
               numObjects=1,
               isTest=False):
    """Initializes the ClawEnv.

    Args:
      urdfRoot: The diretory from which to load environment URDF's.
      actionRepeat: The number of simulation steps to apply for each action.
      isEnableSelfCollision: If true, enable self-collision.
      renders: If true, render the bullet GUI.
      isDiscrete: If true, the action space is discrete. If False, the
        action space is continuous.
      maxSteps: The maximum number of actions per episode.
      dv: The velocity along each dimension for each action.
      removeHeightHack: If false, there is a "height hack" where the gripper
        automatically moves down for each action. If true, the environment is
        harder and the policy chooses the height displacement.
      blockRandom: A float between 0 and 1 indicated block randomness. 0 is
        deterministic.
      cameraRandom: A float between 0 and 1 indicating camera placement
        randomness. 0 is deterministic.
      width: The image width.
      height: The observation image height.
      numObjects: The number of objects in the bin.
      isTest: If true, use the test set of objects. If false, use the train
        set of objects.
    """

    self._isDiscrete = isDiscrete
    self._timeStep = 1. / 240.
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._envStepCounter = 0
    self._renders = renders
    self._maxSteps = maxSteps
    self.terminated = 0
    self._cam_dist = 1.3
    self._cam_yaw = 180
    self._cam_pitch = -40
    self._dv = dv
    self._p = p
    self._removeHeightHack = removeHeightHack
    self._blockRandom = blockRandom
    self._cameraRandom = cameraRandom
    self._width = width
    self._height = height
    self._numObjects = numObjects
    self._isTest = isTest

    if self._renders:
      self.cid = p.connect(p.SHARED_MEMORY)
      if (self.cid < 0):
        self.cid = p.connect(p.GUI)
      p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
    else:
      self.cid = p.connect(p.DIRECT)
    self.seed()

    if (self._isDiscrete):
      if self._removeHeightHack:
        self.action_space = spaces.Discrete(9)
      else:
        self.action_space = spaces.Discrete(7)
    else:
      self.action_space = spaces.Box(low=-1, high=1, shape=(3,))  # dx, dy, da
      if self._removeHeightHack:
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,))  # dx, dy, dz, da
    self.observation_space = spaces.Box(low=0, high=255, shape=(self._height,
                                                                self._width,
                                                                3))
    self.viewer = None

  def reset(self):
    """Environment reset called at the beginning of an episode.
    """
    # Set the camera settings.
    look = [0.23, 0.2, 0.54]
    distance = 1.
    pitch = -56 + self._cameraRandom * np.random.uniform(-3, 3)
    yaw = 245 + self._cameraRandom * np.random.uniform(-3, 3)
    roll = 0
    self._view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
    fov = 20. + self._cameraRandom * np.random.uniform(-2, 2)
    aspect = self._width / self._height
    near = 0.01
    far = 10
    self._proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    self._attempted_grasp = False
    self._env_step = 0
    self.terminated = 0

    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])

    p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.820000,
               0.000000, 0.000000, 0.0, 1.0)

    p.setGravity(0, 0, -10)

    self._kuka = kukaclaw.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    self._envStepCounter = 0
    p.stepSimulation()

    # Choose the objects in the bin.
    urdfList = self._get_random_object(self._numObjects, self._isTest)
    self._objectUids = self._randomly_place_objects(urdfList)
    self._observation = self._get_observation()
    return np.array(self._observation)

  def _randomly_place_objects(self, urdfList):
    """Randomly places the objects in the bin.

    Args:
      urdfList: The list of urdf files to place in the bin.

    Returns:
      The list of object unique ID's.
    """

    # Randomize positions of each object urdf.
    objectUids = []
    for urdf_name in urdfList:
      xpos = 0.4 + self._blockRandom * random.random()
      ypos = self._blockRandom * (random.random() - .5)
      angle = np.pi / 2 + self._blockRandom * np.pi * random.random()
      orn = p.getQuaternionFromEuler([0, 0, angle])
      urdf_path = os.path.join(self._urdfRoot, urdf_name)
      uid = p.loadURDF(urdf_path, [xpos, ypos, .15], [orn[0], orn[1], orn[2], orn[3]])
      objectUids.append(uid)
      # Let each object fall to the tray individual, to prevent object
      # intersection.
      for _ in range(500):
        p.stepSimulation()
    return objectUids

  def _get_observation(self):
    """Return the observation as an image.
    """
    img_arr = p.getCameraImage(width=self._width,
                               height=self._height,
                               viewMatrix=self._view_matrix,
                               projectionMatrix=self._proj_matrix)
    rgb = img_arr[2]
    np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
    return np_img_arr[:, :, :3]

  def step(self, action):
    """Environment step.

    Args:
      action: 5-vector parameterizing XYZ offset, vertical angle offset
      (radians), and grasp angle (radians).
    Returns:
      observation: Next observation.
      reward: Float of the per-step reward as a result of taking the action.
      done: Bool of whether or not the episode has ended.
      debug: Dictionary of extra information provided by environment.
    """
    dv = self._dv  # velocity per physics step.
    if self._isDiscrete:
      # Static type assertion for integers.
      assert isinstance(action, int)
      if self._removeHeightHack:
        dx = [0, -dv, dv, 0, 0, 0, 0, 0, 0][action]
        dy = [0, 0, 0, -dv, dv, 0, 0, 0, 0][action]
        dz = [0, 0, 0, 0, 0, -dv, dv, 0, 0][action]
        da = [0, 0, 0, 0, 0, 0, 0, -0.25, 0.25][action]
      else:
        dx = [0, -dv, dv, 0, 0, 0, 0][action]
        dy = [0, 0, 0, -dv, dv, 0, 0][action]
        dz = [-dv, 0, 0, 0, 0, 0, 0][action]
        da = [0, 0, 0, 0, 0, -0.25, 0.25][action]
    else:
      dx = dv * action[0]
      dy = dv * action[1]
      if self._removeHeightHack:
        dz = dv * action[2]
        da = 0.25 * action[3]
      else:
        dz = -dv
        da = 0.25 * action[2]

    return self._step_continuous([dx, dy, dz, da, 0.3])

  def _step_continuous(self, action):
    """Applies a continuous velocity-control action.

    Args:
      action: 5-vector parameterizing XYZ offset, vertical angle offset
      (radians), and grasp angle (radians).
    Returns:
      observation: Next observation.
      reward: Float of the per-step reward as a result of taking the action.
      done: Bool of whether or not the episode has ended.
      debug: Dictionary of extra information provided by environment.
    """
    # Get the current block's position
    block_pos = self._get_object_position()
    
    
    self._env_step += 1
    # move in each direction sequentially
    self._apply_action([action[0], 0, 0, 0, action[4]]) # translate only x
    self._apply_action([0, action[1], 0, 0, action[4]]) # translate only y
    self._apply_action([0, 0, 0, action[3], action[4]]) # translate only a
    
    # perform grasp motion when dz is set
    if action[2] != 0:
      # If we are close to the bin, attempt grasp.
      state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
      end_effector_pos = state[0]
      # if end_effector_pos[2] <= 0.1:
      position = [end_effector_pos[0], end_effector_pos[1], 0.2, 0, action[4]]
      self._move_to_position(position)
      self._grasp()
      print('Going to goal')
      self._move_to_goal()
      self._release()
      self._move_to_position([0, 0, 0.2, 0, action[4]])

    observation = self._get_observation()
    done = self._termination()
    reward = self._reward(block_pos)

    debug = {'grasp_success': self._graspSuccess}
    return observation, reward, done, debug

  def _release(self):
    print('Release start')
    finger_angle = 0
    for _ in range(100):
      grasp_action = [0, 0, 0, 0, finger_angle]
      self._kuka.applyAction(grasp_action)
      p.stepSimulation()
      if self._renders:
        time.sleep(self._timeStep)
      finger_angle += 0.3 / 100.
      if finger_angle > 0.3:
        finger_angle = 0.3

  def _grasp(self):
    print('Grasp start')
    finger_angle = 0.3
    for _ in range(100):
      grasp_action = [0, 0, 0, 0, finger_angle]
      self._kuka.applyAction(grasp_action)
      p.stepSimulation()
      if self._renders:
        time.sleep(self._timeStep)
      finger_angle -= 0.3 / 100.
      if finger_angle < 0:
        finger_angle = 0
    print('Grasp 2 start')
    for _ in range(100):
      grasp_action = [0, 0, 0.002, 0, finger_angle]
      self._kuka.applyAction(grasp_action)
      p.stepSimulation()
      if self._renders:
        time.sleep(self._timeStep)
      finger_angle -= 0.3 / 100.
      if finger_angle < 0:
        finger_angle = 0
    self._attempted_grasp = True

  def _apply_action(self, action):
    if any([i != 0 for i in action[:4]]): # only take action when there is some non-zero command
      print(f'Applying action: {action}')
      self._kuka.applyAction(action)
      for _ in range(self._actionRepeat):
        p.stepSimulation()
        if self._renders:
          time.sleep(self._timeStep)
        # if self._termination():
        #   break

  def _move_to_position(self, pos):
    """ Moves the Kuka arm to a position
    Args: position, (x,y,z,a,r) coordinates, a is the end defector angle, r is the claw angle

    """
    print(f'Moving to {[round(i, 2) for i in pos]}')
    pos = self._kuka.clamp_positions(pos)
    self._kuka.move_to_pos(pos)
    for _ in range(500):
      # self._kuka.applyAction(self._calculate_force(pos))
      p.stepSimulation()
      if self._renders:
        time.sleep(self._timeStep)
      end_effector_pos = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)[0]
      if abs(end_effector_pos[0] - pos[0]) < 0.1 and abs(end_effector_pos[1] - pos[1]) < 0.1 and abs(end_effector_pos[2] - pos[2]) < 0.1:
        break

  def _move_to_goal(self):
    goal = [1, 0, 0.25, 0, 0]
    self._move_to_position(goal)
    print("Went to Goal!")


  def _get_object_position(self):
    """
    Returns all the block's position in the form of a dictionary
    """
    dict = {}
    for uid in self._objectUids:
      pos, _ = p.getBasePositionAndOrientation(uid)
      dict[uid] = pos
    return dict

  def _reward(self, pre_pos):
    """Calculates the reward for the episode.

    The reward is 1 if one of the objects is above height .2 at the end of the
    episode.
    """
    reward = 0
    furthest_block = 0
    self._graspSuccess = 0
    for uid in self._objectUids:
      pos, _ = p.getBasePositionAndOrientation(uid)
      # If any block is in the hole, provide reward.
      if pos[0] > 0.8 and pos[0] < 0.9 and pos[2] < 0.2: # If the object is in the wedge
        self._graspSuccess += 1
        reward = 1
        break
      elif pos[0] - pre_pos[uid][0] > furthest_block: # save the block that moved the furthest
        furthest_block = pos[0] - pre_pos[uid][0]

      if self._graspSuccess == 0:
        reward = furthest_block # Reward the robot slightly

    return reward

  def _termination(self):
      """Terminates the episode if the block is in the tray or if we are above
      maxSteps steps.
      """
      in_goal = []
      for uid in self._objectUids:
        block_pos = self._get_object_position()[uid]
        in_goal.append(0.85 <= block_pos[0] <= 0.9 and 
                      -0.2 <= block_pos[1] <= 0.15 and 
                              block_pos[2] <= -0.1)
      terimination = any(in_goal) or self._env_step >= self._maxSteps
      return terimination


  def _get_random_object(self, num_objects, test):
    """Randomly choose an object urdf from the random_urdfs directory.

    Args:
      num_objects:
        Number of graspable objects.

    Returns:
      A list of urdf filenames.
    """
    if test:
      urdf_pattern = os.path.join(self._urdfRoot, 'random_urdfs/*0/*.urdf')
    else:
      urdf_pattern = os.path.join(self._urdfRoot, 'random_urdfs/*[1-9]/*.urdf')
    found_object_directories = glob.glob(urdf_pattern)
    total_num_objects = len(found_object_directories)
    selected_objects = np.random.choice(np.arange(total_num_objects), num_objects)
    selected_objects_filenames = []
    for object_index in selected_objects:
      selected_objects_filenames += [found_object_directories[object_index]]
    return selected_objects_filenames

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _reset = reset
    _step = step
