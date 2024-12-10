# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import asyncio
import numpy as np
import carb
import time
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid, GroundPlane
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils import distance_metrics
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats, quats_to_rot_matrices
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.motion_generation import ArticulationMotionPolicy, RmpFlow
from omni.isaac.motion_generation.interface_config_loader import load_supported_motion_policy_config
from omni.isaac.nucleus import get_assets_root_path

from omni.isaac.motion_generation.interface_config_loader import get_supported_robot_policy_pairs

from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats

from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver

class SO100RmpFlowScript:
    def __init__(self):
        self._rmpflow = None
        self._articulation_rmpflow = None

        self._articulation = None
        self._target = None

        self._script_generator = None
        self._dbg_mode = False

    def load_example_assets(self):
        """Load assets onto the stage and return them so they can be registered with the
        core.World.

        This function is called from ui_builder._setup_scene()

        The position in which things are loaded is also the position to which
        they will be returned on reset.
        """
        ################# use SO-Arm-100
        robot_prim_path = "/World"
        path_to_robot_usd = "/home/hph/Documents/so100_follower/so100.usd"
        ################# use SO-Arm-100

        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self._articulation = Articulation(robot_prim_path, position=np.array([0,0,0]))

        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        self._target = XFormPrim(
            "/World/target",
            scale=[0.04, 0.04, 0.04],
            position=np.array([0.13, -0.1, 0.16]),
            orientation=euler_angles_to_quats([np.pi,  0,np.pi]),
        )

        self._goal_block = DynamicCuboid(
            name="Cube",
            # position=np.array([0.4, 0, 0.025]),
            position=np.array([0.22, 0, 0.08]),
            prim_path="/World/pick_cube",
            size=0.05,
            color=np.array([1, 0, 0]),
        )
        
        self._obstacles = [
            self._goal_block
            # FixedCuboid(
            #     name="ob1",
            #     prim_path="/World/obstacle_1",
            #     scale=np.array([0.03, 1.0, 0.05]),
            #     position=np.array([0.15, 0.25, 0.1]),
            #     color=np.array([1.0, 1.0, 0.0]),
            # ),
            # FixedCuboid(
            #     name="ob2",
            #     prim_path="/World/obstacle_2",
            #     scale=np.array([0.5, 0.03, 0.05]),
            #     position=np.array([0.45, 0.25, 0.1]),
            #     color=np.array([1.0, 1.0, 0.0]),
            # ),
        ]

        
        self._ground_plane = GroundPlane("/World/Ground")

        # Return assets that were added to the stage so that they can be registered with the core.World
        return self._articulation, self._target, *self._obstacles, self._ground_plane #, self._goal_block

    def setup(self):
        """
        This function is called after assets have been loaded from ui_builder._setup_scenario().
        """
        
        # Set a camera view that looks good
        set_camera_view(eye=[2, 0.8, 1], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")

        print(get_supported_robot_policy_pairs())
        # Loading RMPflow can be done quickly for supported robots
        rmp_config = load_supported_motion_policy_config("SO100", "RMPflow", "/home/hph/Documents/so100_follower/so100_ext/config/motion_policy_configs")
        
        print(rmp_config)
        # rmp_config = load_supported_motion_policy_config("Cobotta_Pro_900", "RMPflow")

        # Initialize an RmpFlow object
        self._rmpflow = RmpFlow(**rmp_config)

        for obstacle in self._obstacles:
            self._rmpflow.add_obstacle(obstacle)

        # Use the ArticulationMotionPolicy wrapper object to connect rmpflow to the Franka robot articulation.
        self._articulation_rmpflow = ArticulationMotionPolicy(self._articulation, self._rmpflow)

        # Create a script generator to execute my_script().
        self._script_generator = self.my_script()
        
        if self._dbg_mode:
            self._rmpflow.set_ignore_state_updates(True)
            self._rmpflow.visualize_collision_spheres()

            # Set the robot gains to be deliberately poor
            bad_proportional_gains = self._articulation.get_articulation_controller().get_gains()[0]/50
            self._articulation.get_articulation_controller().set_gains(kps = bad_proportional_gains)

    def reset(self):
        """
        This function is called when the reset button is pressed.
        In this example the core.World takes care of all necessary resetting
        by putting everything back in the position it was in when loaded.

        In more complicated scripts, e.g. scripts that modify or create USD properties
        or attributes at runtime, the user will need to implement necessary resetting
        behavior to ensure their script runs deterministically.
        """
        # Start the script over by recreating the generator.
        self._script_generator = self.my_script()
        
        if self._dbg_mode:
            # RMPflow was set to roll out robot state internally, assuming that all returned joint targets were hit exactly.
            self._rmpflow.reset()
            self._rmpflow.visualize_collision_spheres()

    """
    The following two functions demonstrate the mechanics of running code in a script-like way
    from a UI-based extension.  This takes advantage of Python's yield/generator framework.  

    The update() function is tied to a physics subscription, which means that it will be called
    one time on every physics step (usually 60 frames per second).  Each time it is called, it
    queries the script generator using next().  This makes the script generator execute until it hits
    a yield().  In this case, no value need be yielded.  This behavior can be nested into subroutines
    using the "yield from" keywords.
    """

    def update(self, step: float):
        
        try:
            _ = next(self._script_generator)
        except StopIteration:
            return True

    def my_script(self):
        while True:  # 持续追踪目标
            # 获取当前目标的位置和方向
            translation_target, orientation_target = self._target.get_world_pose()
            
            # 获取机械臂基座的位置和方向
            robot_base_translation, robot_base_orientation = self._articulation.get_world_pose()
            
            # 更新RMPFlow的基座姿态
            self._rmpflow.set_robot_base_pose(robot_base_translation, robot_base_orientation)
            
            # 调用goto_position()函数去追踪当前目标位置
            success = yield from self.goto_position(
                translation_target, orientation_target, self._articulation, self._rmpflow, timeout=50
            )
            
            # 检查是否成功到达目标
            if not success:
                print("Could not reach target position, retrying...")
                continue  # 如果失败，继续循环重新获取目标位置

            # # 可选：添加条件结束追踪（例如：目标达到某个状态、时间超时等）
            # if self.should_stop_tracking():  # 自定义的结束条件函数
            #     print("Tracking stopped.")
            #     break
            
            # 小延迟避免过高频率更新，实际应用中可根据目标移动速度调整
            # asyncio.sleep(delay=33)
            yield()
        # translation_target, orientation_target = self._target.get_world_pose()
        
        # robot_base_translation,robot_base_orientation = self._articulation.get_world_pose()
        # self._rmpflow.set_robot_base_pose(robot_base_translation,robot_base_orientation)
        
        # yield from self.close_gripper_so100(self._articulation) #TODO:

        # # Notice that subroutines can still use return statements to exit.  goto_position() returns a boolean to indicate success.
        # success = yield from self.goto_position(
        #     translation_target, orientation_target, self._articulation, self._rmpflow, timeout=200
        # )

        # if not success:
        #     print("Could not reach target position")
        #     return

        # yield from self.open_gripper_so100(self._articulation) #TODO:

        # # Visualize the new target.
        # lower_translation_target = np.array([0.22, 0.1, 0.04])
        # self._target.set_world_pose(lower_translation_target, orientation_target)

        # success = yield from self.goto_position(
        #     lower_translation_target, orientation_target, self._articulation, self._rmpflow, timeout=250
        # )

        # yield from self.close_gripper_so100(self._articulation, close_position=np.array([0.02]), atol=0.006) #TODO:

        # high_translation_target = np.array([0.4, 0, 0.4])
        # self._target.set_world_pose(high_translation_target, orientation_target)

        # success = yield from self.goto_position(
        #     high_translation_target, orientation_target, self._articulation, self._rmpflow, timeout=200
        # )

        # next_translation_target = np.array([0.4, 0.4, 0.4])
        # self._target.set_world_pose(next_translation_target, orientation_target)

        # success = yield from self.goto_position(
        #     next_translation_target, orientation_target, self._articulation, self._rmpflow, timeout=200
        # )

        # next_translation_target = np.array([0.4, 0.4, 0.25])
        # self._target.set_world_pose(next_translation_target, orientation_target)

        # success = yield from self.goto_position(
        #     next_translation_target, orientation_target, self._articulation, self._rmpflow, timeout=200
        # )

        # yield from self.open_gripper_so100(self._articulation) #TODO:

    ################################### Functions

    def goto_position(
        self,
        translation_target,
        orientation_target,
        articulation,
        rmpflow,
        translation_thresh=0.01,
        orientation_thresh=0.1,
        timeout=500,
    ):
        """
        Use RMPflow to move a robot Articulation to a desired task-space position.
        Exit upon timeout or when end effector comes within the provided threshholds of the target pose.
        """

        articulation_motion_policy = ArticulationMotionPolicy(articulation, rmpflow, 1 / 60)
        rmpflow.set_end_effector_target(translation_target, orientation_target)
        
        base_trans, base_rot = articulation.get_world_pose()
        base_rot_matrix = quats_to_rot_matrices(base_rot)  # 基座旋转矩阵
        
        for i in range(timeout):
            ee_trans, ee_rot = rmpflow.get_end_effector_pose(
                articulation_motion_policy.get_active_joints_subset().get_joint_positions()
            )

            # ee_trans = base_rot_matrix @ ee_trans + base_trans  # 平移变换
            # ee_rot = base_rot_matrix @ ee_rot  # 旋转变换
            
            trans_dist = distance_metrics.weighted_translational_distance(ee_trans, translation_target)
            rotation_target = quats_to_rot_matrices(orientation_target)
            rot_dist = distance_metrics.rotational_distance_angle(ee_rot, rotation_target)
            
            done = trans_dist < translation_thresh#  and rot_dist < orientation_thresh
            # print("trans_dist==",trans_dist)
            # print("rot_dist==",rot_dist)
            if done:
                return True

            rmpflow.update_world()
            action = articulation_motion_policy.get_next_articulation_action(1 / 60)
            articulation.apply_action(action)

            # If not done on this frame, yield() to pause execution of this function until
            # the next frame.
            yield ()

        return False

    def open_gripper_so100(self, articulation):
        print('open_gripper_so100')
        
        p_open = 0.7
        open_gripper_action = ArticulationAction(np.array([p_open]), joint_indices=np.array([5]))
        articulation.apply_action(open_gripper_action)

        # Check in once a frame until the gripper has been successfully opened.
        while not np.allclose(articulation.get_joint_positions()[5:], np.array([p_open]), atol=0.001):
            yield ()

        print('open_gripper_so100 done')
        return True

    def close_gripper_so100(self, articulation, close_position=np.array([0]), atol=0.001):
        # To close around the cube, different values are passed in for close_position and atol
        open_gripper_action = ArticulationAction(np.array(close_position), joint_indices=np.array([5]))
        articulation.apply_action(open_gripper_action)

        # Check in once a frame until the gripper has been successfully closed.
        while not np.allclose(articulation.get_joint_positions()[5:], np.array(close_position), atol=atol):
            yield ()

        return True

class SO100FollowScript:
    def __init__(self):

        self._articulation = None
        self._target = None

        self._script_generator = None

    def load_example_assets(self):
        """Load assets onto the stage and return them so they can be registered with the
        core.World.

        This function is called from ui_builder._setup_scene()

        The position in which things are loaded is also the position to which
        they will be returned on reset.
        """
        ################# use SO-Arm-100
        robot_prim_path = "/World"
        path_to_robot_usd = "/home/hph/Documents/so100_follower/so100.usd"
        ################# use SO-Arm-100

        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self._articulation = Articulation(robot_prim_path, position=np.array([0,0,0]))

        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        self._target = XFormPrim(
            "/World/target",
            scale=[0.04, 0.04, 0.04],
            position=np.array([0.0036570001584201785, -0.17443716520711405, 0.20550990329078275]),
            orientation=euler_angles_to_quats([0, np.pi, 0]),
        )

        self._ground_plane = GroundPlane("/World/Ground")

        # Return assets that were added to the stage so that they can be registered with the core.World
        return self._articulation, self._target, self._ground_plane

    def setup(self):
        """
        This function is called after assets have been loaded from ui_builder._setup_scenario().
        """
        
        kinematics_config_dir = "/home/hph/Documents/so100_follower/so100_ext/config/motion_policy_configs"

        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path = kinematics_config_dir + "/SO100/rmpflow/robot_descriptor_follow.yaml",
            urdf_path = kinematics_config_dir + "/SO100/urdf/SO_5DOF_ARM100_8j_URDF.SLDASM.urdf"
        )

        print("Valid frame names at which to compute kinematics:", self._kinematics_solver.get_all_frame_names())

        end_effector_name = "Moving_Jaw"
        self._articulation_kinematics_solver = ArticulationKinematicsSolver(self._articulation,self._kinematics_solver, end_effector_name)
        
    def reset(self):
        """
        This function is called when the reset button is pressed.
        In this example the core.World takes care of all necessary resetting
        by putting everything back in the position it was in when loaded.

        In more complicated scripts, e.g. scripts that modify or create USD properties
        or attributes at runtime, the user will need to implement necessary resetting
        behavior to ensure their script runs deterministically.
        """
        # Start the script over by recreating the generator.
        # self._script_generator = self.my_script()
        

    """
    The following two functions demonstrate the mechanics of running code in a script-like way
    from a UI-based extension.  This takes advantage of Python's yield/generator framework.  

    The update() function is tied to a physics subscription, which means that it will be called
    one time on every physics step (usually 60 frames per second).  Each time it is called, it
    queries the script generator using next().  This makes the script generator execute until it hits
    a yield().  In this case, no value need be yielded.  This behavior can be nested into subroutines
    using the "yield from" keywords.
    """

    def update(self, step: float):
        target_position, target_orientation = self._target.get_world_pose()

        #Track any movements of the robot base
        robot_base_translation,robot_base_orientation = self._articulation.get_world_pose()
        self._kinematics_solver.set_robot_base_pose(robot_base_translation,robot_base_orientation)

        action, success = self._articulation_kinematics_solver.compute_inverse_kinematics(target_position, target_orientation) # target_orientation)

        if success:
            self._articulation.apply_action(action)
        else:
            carb.log_warn("IK did not converge to a solution.  No action is being taken")