import math,os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import xml.etree.ElementTree as ET

# Convert quaternion to Euler angle 
def quaternion_to_euler_angle(w, x, y, z):
	ysqr = y * y
	
	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + ysqr)
	X = math.degrees(math.atan2(t0, t1))
	
	t2 = +2.0 * (w * y - z * x)
	t2 = +1.0 if t2 > +1.0 else t2
	t2 = -1.0 if t2 < -1.0 else t2
	Y = math.degrees(math.asin(t2))
	
	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (ysqr + z * z)
	Z = math.degrees(math.atan2(t3, t4))
	
	return X, Y, Z

# Snapbot (6 legs) Environment
class Snapbot6EnvClass(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                VERBOSE     = True,
                name        = 'Snapbot with 6 legs',
                xml_path    = 'envs/snapbot/xml/snapbot_6/robot_6.xml',
                frame_skip  = 5,
                condition   = None, 
                ctrl_coef   = 0,
                body_coef   = 0,
                jump_coef   = 0,
                vel_coef    = 0,
                head_coef   = 0,
                render_mode = 'human',
                render_w    = 1500,
                render_h    = 1000,
                render_res  = 200
                ):
        """
            Initialize
        """
        self.VERBOSE    = VERBOSE
        self.name       = name
        self.xml_path   = os.path.join(os.path.dirname(__file__), xml_path)
        self.frame_skip = frame_skip
        self.condition  = condition
        self.ctrl_coef  = ctrl_coef
        self.body_coef  = body_coef
        self.jump_coef  = jump_coef
        self.vel_coef   = vel_coef
        self.head_coef  = head_coef
        self.rand_mass  = None
        self.k_p = 0.2,
        self.k_i = 0.001,
        self.k_d = 0.01,
        self.joint_pos_deg_min = -np.array([43,40,43,40,43,40,43,40,43,40,43,40])
        self.joint_pos_deg_max = np.array([43,40,43,40,43,40,43,40,43,40,43,40])

        # Open xml
        self.xml = open(xml_path, 'rt', encoding='UTF8')
        try:
            mujoco_env.MujocoEnv.__init__(
            self,
            model_path      = self.xml_path,
            frame_skip      = self.frame_skip,
            mujoco_bindings = 'mujoco_py'
            )
        except:
            mujoco_env.MujocoEnv.__init__(
            self,
            model_path      = self.xml_path,
            frame_skip      = self.frame_skip
            )
        utils.EzPickle.__init__(self)

        # Observation and action dimension
        self.odim = self.observation_space.shape[0]
        self.adim = self.action_space.shape[0]

        if self.VERBOSE:
            print("Snapbot(6legs) Environment")   
            print("Obs Dim: [{}] Act Dim: [{}] dt:[{}] Condition:[{}]".format(self.odim, self.adim, self.dt, condition))
            print("ctrl_coef:[{}] body_coef:[{}] jump_coef:[{}] vel_coef:[{}] head_coef:[{}]".format(
                self.ctrl_coef, self.body_coef, self.jump_coef, self.vel_coef, self.head_coef))

        # Timing
        self.hz = int(1/self.dt)
        # Reset
        self.reset()
        # Viewer setup
        if render_mode is not None:
            self.viewer_custom_setup(
                render_mode = render_mode,
                render_w    = render_w,
                render_h    = render_h,
                render_res  = render_res
                )

    def step(self, a):
        """
            Step forward
        """        
        # Before run
        x_pos_before      = self.get_body_com("torso")[0]
        y_pos_before      = self.get_body_com("torso")[1]
        heading_before    = self.get_heading()
        self.prev_state   = np.concatenate([self.sim.data.qpos.flat[2:],self.sim.data.qvel.flat])
        self.prev_torque  = a
        self.contact_data = np.array(self.sim.data.sensordata[6:])

        # Run sim
        self.do_simulation(a, self.frame_skip)
        x_pos_after   = self.get_body_com("torso")[0]
        y_pos_after   = self.get_body_com("torso")[1]
        heading_after = self.get_heading()
        heading_diff  = heading_after - heading_before
        z_pos = self.get_body_com("torso")[2]
        velocity_list = abs(np.array([self.sim.data.qvel[8], self.sim.data.qvel[12], self.sim.data.qvel[16], self.sim.data.qvel[20], self.sim.data.qvel[24], self.sim.data.qvel[28]]))

        reward_forward = (x_pos_after - x_pos_before) / self.dt
        cost_control   = self.ctrl_coef * np.square(a).sum()
        cost_contact   = self.body_coef * np.square(self.contact_data).sum()
        cost_heading   = self.head_coef * (heading_after**2+y_pos_after**2)
        if max(velocity_list) > 10:
            cost_velocity = self.vel_coef * max(velocity_list)
        else:
            cost_velocity = 0
            
        # Accumulate
        self.a    = a
        self.o    = self._get_obs()
        self.r    = reward_forward - (cost_control+cost_contact+cost_heading+cost_velocity)
        self.info = dict()
        
        # Done condition
        state   = self.state_vector()
        r, p, y = quaternion_to_euler_angle(state[3], state[4], state[5], state[6])
        notdone = np.isfinite(state).all and abs(r) < 170
        self.d  = not notdone
        
        return self.o, self.r, self.d, self.info

    def _get_obs(self):
        self.index = np.array(self.get_max_leg()).reshape(1)
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            self.prev_state,
            self.prev_torque,
            self.index
        ])

    def reset_model(self):
        """
            Reset
        """
        o = np.zeros(self.odim)
        return o

    def get_joint_pos_deg(self):
        """
            Get joint position in [Deg]
        """
        q = self.sim.data.qpos.flat
        return np.asarray(
            [q[9],q[10],q[13],q[14],q[17],q[18],q[21],q[22],q[25],q[26],q[29],q[30]]
            )*180.0/np.pi

    def get_heading(self):
        """
            Get z-axis rotation angle in [Deg]
        """
        q = self.data.get_body_xquat('torso')
        _, _, z_deg = quaternion_to_euler_angle(q[0], q[1], q[2], q[3])
        return z_deg

    def get_seonsor_data(self):
        l1 = self.sim.data.get_sensor('touchsensor_1')
        l2 = self.sim.data.get_sensor('touchsensor_2')
        l3 = self.sim.data.get_sensor('touchsensor_3')
        l4 = self.sim.data.get_sensor('touchsensor_4')
        l5 = self.sim.data.get_sensor('touchsensor_5')
        l6 = self.sim.data.get_sensor('touchsensor_6')
        ls = [l1,l2,l3,l4,l5,l6]
        return ls

    def get_max_leg(self):  
        lst   = self.get_seonsor_data()
        score = 0
        index = 0
        for i,j in enumerate(lst):
            if j > score : 
                score = j
                index = i+1
        return index

    def get_time(self):
        """
            Get time in [Sec]
        """
        return self.sim.data.time
    
    def viewer_custom_setup(
        self,
        render_mode = 'human',
        render_w    = 1500,
        render_h    = 1000,
        render_res  = 200
        ):
        """
            View setup
        """
        self.render_mode = render_mode
        self.render_w    = render_w
        self.render_h    = render_h
        self.render_res  = render_res
        
        frame = self.render(
            mode   = self.render_mode,
            width  = self.render_w,
            height = self.render_h)
        self.viewer.cam.distance  = 8.0 # distance to plane (1.5)
        self.viewer.cam.elevation = -45 # elevation angle (-30)
        self.viewer.cam.lookat[0] = 0.0 # x-axis (let this follow the robot)
        self.viewer.cam.lookat[1] = 0.0
        self.viewer.cam.lookat[2] = 0.0
        
    def render_center(self):
        """
            Render with torso-centered
        """
        for d_idx in range(3): # follow the robot torso
            self.viewer.cam.lookat[d_idx] = self.get_body_com("torso")[d_idx]
        frame = self.render(
            mode   = self.render_mode,
            width  = self.render_w,
            height = self.render_h)
        return frame

# Snapbot (5 legs) Environment
class Snapbot5EnvClass(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                VERBOSE     = True,
                name        = 'Snapbot with 5 legs',
                leg_idx     = 12345,
                xml_path    = 'envs/snapbot/xml/snapbot_6/robot_6.xml',
                frame_skip  = 5,
                condition   = None, 
                ctrl_coef   = 0,
                body_coef   = 0,
                jump_coef   = 0,
                vel_coef    = 0,
                head_coef   = 0,
                render_mode = 'human',
                render_w    = 1500,
                render_h    = 1000,
                render_res  = 200
                ):
        """
            Initialize
        """
        self.VERBOSE     = VERBOSE
        self.name        = name
        self.leg_idx     = str(leg_idx)
        self.xml_path    = os.path.abspath('snapbot_env/xml/snapbot_5/robot_5_{}.xml'.format(self.leg_idx))
        self.frame_skip  = frame_skip
        self.condition   = condition
        self.ctrl_coef   = ctrl_coef
        self.body_coef   = body_coef
        self.jump_coef   = jump_coef
        self.vel_coef    = vel_coef
        self.head_coef   = head_coef
        self.rand_mass   = None
        self.k_p = 0.2,
        self.k_i = 0.001,
        self.k_d = 0.01,
        self.joint_pos_deg_min = -np.array([43,40,43,40,43,40,43,40,43,40])
        self.joint_pos_deg_max = np.array([43,40,43,40,43,40,43,40,43,40])

        # Open xml
        self.xml = open(xml_path, 'rt', encoding='UTF8')
        try:
            mujoco_env.MujocoEnv.__init__(
            self,
            model_path      = self.xml_path,
            frame_skip      = self.frame_skip,
            mujoco_bindings = 'mujoco_py'
            )
        except:
            mujoco_env.MujocoEnv.__init__(
            self,
            model_path      = self.xml_path,
            frame_skip      = self.frame_skip
            )
        utils.EzPickle.__init__(self)

        # Observation and action dimension
        self.odim = self.observation_space.shape[0]
        self.adim = self.action_space.shape[0]

        if self.VERBOSE:
            print("Snapbot(5legs: {}) Environment".format(self.leg_idx))   
            print("Obs Dim: [{}] Act Dim: [{}] dt:[{}] Condition:[{}]".format(self.odim, self.adim, self.dt, condition))
            print("ctrl_coef:[{}] body_coef:[{}] jump_coef:[{}] vel_coef:[{}] head_coef:[{}]".format(
                self.ctrl_coef, self.body_coef, self.jump_coef, self.vel_coef, self.head_coef))

        # Timing
        self.hz = int(1/self.dt)
        # Reset
        self.reset()
        # Viewer setup
        if render_mode is not None:
            self.viewer_custom_setup(
                render_mode = render_mode,
                render_w    = render_w,
                render_h    = render_h,
                render_res  = render_res
                )

    def step(self, a):
        """
            Step forward
        """        
        # Before run
        x_pos_before      = self.get_body_com("torso")[0]
        y_pos_before      = self.get_body_com("torso")[1]
        heading_before    = self.get_heading()
        self.prev_state   = np.concatenate([self.sim.data.qpos.flat[2:],self.sim.data.qvel.flat])
        self.prev_torque  = a
        self.contact_data = np.array(self.sim.data.sensordata[6:])

        # Run sim
        self.do_simulation(a, self.frame_skip)
        x_pos_after   = self.get_body_com("torso")[0]
        y_pos_after   = self.get_body_com("torso")[1]
        heading_after = self.get_heading()
        heading_diff  = heading_after - heading_before
        z_pos = self.get_body_com("torso")[2]

        # Accumulate
        self.a    = a
        self.o    = self._get_obs()
        self.r    = (x_pos_after - x_pos_before) / self.dt
        self.info = dict()
        
        # Done condition
        state   = self.state_vector()
        r, p, y = quaternion_to_euler_angle(state[3], state[4], state[5], state[6])
        notdone = np.isfinite(state).all and abs(r) < 170
        self.d  = not notdone
        
        return self.o, self.r, self.d, self.info

    def _get_obs(self):
        self.index = np.array(self.get_max_leg()).reshape(1)
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            self.prev_state,
            self.prev_torque,
            self.index
        ])

    def reset_model(self):
        """
            Reset
        """
        o = np.zeros(self.odim)
        return o

    def get_joint_pos_deg(self):
        """
            Get joint position in [Deg]
        """
        q = self.sim.data.qpos.flat
        return np.asarray(
            [q[9],q[10],q[13],q[14],q[17],q[18],q[21],q[22],q[25],q[26]]
            )*180.0/np.pi

    def get_heading(self):
        """
            Get z-axis rotation angle in [Deg]
        """
        q = self.data.get_body_xquat('torso')
        _, _, z_deg = quaternion_to_euler_angle(q[0], q[1], q[2], q[3])
        return z_deg

    def get_seonsor_data(self):
        l1 = self.sim.data.get_sensor('touchsensor_{}'.format(self.leg_idx[0]))
        l2 = self.sim.data.get_sensor('touchsensor_{}'.format(self.leg_idx[1]))
        l3 = self.sim.data.get_sensor('touchsensor_{}'.format(self.leg_idx[2]))
        l4 = self.sim.data.get_sensor('touchsensor_{}'.format(self.leg_idx[3]))
        l5 = self.sim.data.get_sensor('touchsensor_{}'.format(self.leg_idx[4]))
        ls = [l1,l2,l3,l4,l5]
        return ls

    def get_max_leg(self):  
        lst   = self.get_seonsor_data()
        score = 0
        index = 0
        for i,j in enumerate(lst):
            if j > score : 
                score = j
                index = i+1
        return index

    def get_time(self):
        """
            Get time in [Sec]
        """
        return self.sim.data.time
    
    def viewer_custom_setup(
        self,
        render_mode = 'human',
        render_w    = 1500,
        render_h    = 1000,
        render_res  = 200
        ):
        """
            View setup
        """
        self.render_mode = render_mode
        self.render_w    = render_w
        self.render_h    = render_h
        self.render_res  = render_res
        
        frame = self.render(
            mode   = self.render_mode,
            width  = self.render_w,
            height = self.render_h)
        self.viewer.cam.distance  = 8.0 # distance to plane (1.5)
        self.viewer.cam.elevation = -45 # elevation angle (-30)
        self.viewer.cam.lookat[0] = 0.0 # x-axis (let this follow the robot)
        self.viewer.cam.lookat[1] = 0.0
        self.viewer.cam.lookat[2] = 0.0
        
    def render_center(self):
        """
            Render with torso-centered
        """
        for d_idx in range(3): # follow the robot torso
            self.viewer.cam.lookat[d_idx] = self.get_body_com("torso")[d_idx]
        frame = self.render(
            mode   = self.render_mode,
            width  = self.render_w,
            height = self.render_h)
        return frame

# Snapbot (4 legs) Environment
class Snapbot4EnvClass(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                VERBOSE     = True,
                name        = 'Snapbot with 4 legs',
                xml_path    = 'xml/snapbot_4/robot_4_1245.xml',
                frame_skip  = 5,
                condition   = None, 
                ctrl_coef   = 0,
                body_coef   = 0,
                jump_coef   = 0,
                vel_coef    = 0,
                head_coef   = 0,
                render_mode = 'human',
                render_w    = 1500,
                render_h    = 1000,
                render_res  = 200
                ):
        """
            Initialize
        """
        self.VERBOSE    = VERBOSE
        self.name       = name
        self.xml_path   = os.path.join(os.path.dirname(__file__), xml_path)
        self.frame_skip = frame_skip
        self.condition  = condition
        self.ctrl_coef  = ctrl_coef
        self.body_coef  = body_coef
        self.jump_coef  = jump_coef
        self.vel_coef   = vel_coef
        self.head_coef  = head_coef
        self.rand_mass  = None
        self.k_p = 0.2,
        self.k_i = 0.001,
        self.k_d = 0.01,
        self.joint_pos_deg_min = -np.array([43,40,43,40,43,40,43,40])
        self.joint_pos_deg_max = np.array([43,40,43,40,43,40,43,40])

        # Open xml
        self.xml = open(self.xml_path, 'rt', encoding='UTF8')
        try:
            mujoco_env.MujocoEnv.__init__(
            self,
            model_path      = self.xml_path,
            frame_skip      = self.frame_skip,
            mujoco_bindings = 'mujoco_py'
            )
        except:
            mujoco_env.MujocoEnv.__init__(
            self,
            model_path      = self.xml_path,
            frame_skip      = self.frame_skip
            )
        utils.EzPickle.__init__(self)

        # Observation and action dimension
        self.odim = self.observation_space.shape[0]
        self.adim = self.action_space.shape[0]

        if self.VERBOSE:
            print("Snapbot(4legs) Environment")   
            print("Obs Dim: [{}] Act Dim: [{}] dt:[{}] Condition:[{}]".format(self.odim, self.adim, self.dt, condition))
            print("ctrl_coef:[{}] body_coef:[{}] jump_coef:[{}] vel_coef:[{}] head_coef:[{}]".format(
                self.ctrl_coef, self.body_coef, self.jump_coef, self.vel_coef, self.head_coef))

        # Timing
        self.hz = int(1/self.dt)
        # Reset
        self.reset()
        # Viewer setup
        if render_mode is not None:
            self.viewer_custom_setup(
                render_mode = render_mode,
                render_w    = render_w,
                render_h    = render_h,
                render_res  = render_res
                )

    def step(self, a):
        """
            Step forward
        """        
        # Before run
        x_pos_before      = self.get_body_com("torso")[0]
        y_pos_before      = self.get_body_com("torso")[1]
        heading_before    = self.get_heading()
        self.prev_state   = np.concatenate([self.sim.data.qpos.flat[2:], self.sim.data.qvel.flat])
        self.prev_torque  = a
        self.contact_data = np.array(self.sim.data.sensordata[4:])

        # Run sim
        self.do_simulation(a, self.frame_skip)
        x_pos_after   = self.get_body_com("torso")[0]
        y_pos_after   = self.get_body_com("torso")[1]
        heading_after = self.get_heading()
        heading_diff  = heading_after - heading_before
        z_pos         = self.get_body_com("torso")[2]
        velocity_list = abs(np.array([self.sim.data.qvel[8], self.sim.data.qvel[12], self.sim.data.qvel[16], self.sim.data.qvel[20]]))

        reward_forward = (x_pos_after - x_pos_before) / self.dt
        cost_control   = self.ctrl_coef * np.square(a).sum()
        cost_contact   = self.body_coef * np.square(self.contact_data).sum()
        cost_heading   = self.head_coef * (heading_after**2+y_pos_after**2)
        if max(velocity_list) > 10:
            cost_velocity = self.vel_coef * max(velocity_list)
        else:
            cost_velocity = 0

        # Accumulate
        self.a    = a
        self.o    = self._get_obs()
        self.r    = reward_forward - (cost_control+cost_contact+cost_heading+cost_velocity)
        self.info = dict()

        # Done condition
        state   = self.state_vector()
        r, p, y = quaternion_to_euler_angle(state[3], state[4], state[5], state[6])
        notdone = np.isfinite(state).all and abs(r) < 170
        self.d  = not notdone
        
        return self.o, self.r, self.d, self.info

    def set_torso_mass(self, mass, idx):
        xml_path = os.path.join(os.path.dirname(__file__), 'xml/snapbot_4/snapbot_4_1245_{}.xml'.format(idx))
        target_xml = open(xml_path, 'rt', encoding='UTF8')
        tree = ET.parse(target_xml)
        root = tree.getroot()
        tag_inertial=root.find('body').find('inertial')
        tag_inertial.attrib["mass"] = "{}".format(mass)
        tree.write(xml_path)

        xml_path = os.path.join(os.path.dirname(__file__), 'xml/snapbot_4/robot_4_1245_{}.xml'.format(idx))
        try:
            mujoco_env.MujocoEnv.__init__(
            self,
            model_path      = xml_path,
            frame_skip      = self.frame_skip,
            mujoco_bindings = 'mujoco_py'
            )
        except:
            mujoco_env.MujocoEnv.__init__(
            self,
            model_path      = xml_path,
            frame_skip      = self.frame_skip
            )
        utils.EzPickle.__init__(self)
        # Observation and action dimension
        self.odim = self.observation_space.shape[0]
        self.adim = self.action_space.shape[0]

    
    def _get_obs(self):
        """
            Get observation
        """
        self.index = np.array(self.get_max_leg()).reshape(1)
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            self.prev_state,
            self.prev_torque,
            self.index
        ])
    
    def reset_model(self):
        """
            Reset
        """
        o = np.zeros(self.odim)
        return o

    def get_joint_pos_deg(self):
        """
            Get joint position in [Deg]
        """
        q = self.sim.data.qpos.flat
        return np.asarray(
            [q[9],q[10],q[13],q[14],q[17],q[18],q[21],q[22]]
            )*180.0/np.pi

    def get_heading(self):
        """
            Get z-axis rotation angle in [Deg]
        """
        q = self.data.get_body_xquat('torso')
        _, _, z_deg = quaternion_to_euler_angle(q[0], q[1], q[2], q[3])
        return z_deg
    
    def get_seonsor_data(self):
        """
            Get sensor data from touchsensors
        """
        l1 = self.sim.data.get_sensor('touchsensor_1')
        l2 = self.sim.data.get_sensor('touchsensor_2')
        l3 = self.sim.data.get_sensor('touchsensor_4')
        l4 = self.sim.data.get_sensor('touchsensor_5')
        ls = [l1, l2, l3, l4]
        return ls

    def get_max_leg(self):
        lst   = self.get_seonsor_data()
        score = 0
        index = 0
        for i,j in enumerate(lst):
            if j > score : 
                score = j
                index = i+1
        return index

    def get_time(self):
        """
            Get time in [Sec]
        """
        return self.sim.data.time
    
    def viewer_custom_setup(
        self,
        render_mode = 'human',
        render_w    = 1500,
        render_h    = 1000,
        render_res  = 200
        ):
        """
            View setup
        """
        self.render_mode = render_mode
        self.render_w    = render_w
        self.render_h    = render_h
        self.render_res  = render_res
        
        frame = self.render(
            mode   = self.render_mode,
            width  = self.render_w,
            height = self.render_h)
        self.viewer.cam.distance  = 3.3 # distance to plane (1.5)
        self.viewer.cam.elevation = -45 # elevation angle (-30)
        self.viewer.cam.lookat[0] = 0.0 # x-axis (let this follow the robot)
        self.viewer.cam.lookat[1] = 0.0
        self.viewer.cam.lookat[2] = 0.0
        
    def render_center(self):
        """
            Render with torso-centered
        """
        for d_idx in range(3):
            self.viewer.cam.lookat[d_idx] = self.get_body_com("torso")[d_idx] # follow the robot torso
            # self.viewer.cam.lookat[d_idx] = 0 # fix at zero
        frame = self.render(
            mode   = self.render_mode,
            width  = self.render_w,
            height = self.render_h)
        return frame

# SnapbotEnvClass
class Snapbot3EnvClass(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                VERBOSE     = True,
                name        = 'Snapbot with 3 legs',
                xml_path    = 'snapbot_env/xml/snapbot_3/robot_3_245.xml',
                frame_skip  = 5,
                condition   = None, 
                ctrl_coef   = 0,
                body_coef   = 0,
                jump_coef   = 0,
                vel_coef    = 0,
                head_coef   = 0,
                render_mode = 'human',
                render_w    = 1500,
                render_h    = 1000,
                render_res  = 200
                ):
        """
            Initialize
        """
        self.VERBOSE    = VERBOSE
        self.name       = name
        self.xml_path   = os.path.abspath(xml_path)
        self.frame_skip = frame_skip
        self.condition  = condition
        self.ctrl_coef  = ctrl_coef
        self.body_coef  = body_coef
        self.jump_coef  = jump_coef
        self.vel_coef   = vel_coef
        self.head_coef  = head_coef
        self.rand_mass  = None
        self.k_p = 0.2,
        self.k_i = 0.001,
        self.k_d = 0.01,
        self.joint_pos_deg_min = -np.array([43,40,43,40,43,40])
        self.joint_pos_deg_max = np.array([43,40,43,40,43,40])

        # Open xml
        self.xml = open(xml_path, 'rt', encoding='UTF8')
        try:
            mujoco_env.MujocoEnv.__init__(
            self,
            model_path      = self.xml_path,
            frame_skip      = self.frame_skip,
            mujoco_bindings = 'mujoco_py'
            )
        except:
            mujoco_env.MujocoEnv.__init__(
            self,
            model_path      = self.xml_path,
            frame_skip      = self.frame_skip
            )
        utils.EzPickle.__init__(self)

        # Observation and action dimension
        self.odim = self.observation_space.shape[0]
        self.adim = self.action_space.shape[0]

        if self.VERBOSE:
            print("Snapbot(3legs) Environment")   
            print("Obs Dim: [{}] Act Dim: [{}] dt:[{}] Condition:[{}]".format(self.odim, self.adim, self.dt, condition))
            print("ctrl_coef:[{}] body_coef:[{}] jump_coef:[{}] vel_coef:[{}] head_coef:[{}]".format(
                self.ctrl_coef, self.body_coef, self.jump_coef, self.vel_coef, self.head_coef))

        # Timing
        self.hz = int(1/self.dt)
        # Reset
        self.reset()
        # Viewer setup
        if render_mode is not None:
            self.viewer_custom_setup(
                render_mode = render_mode,
                render_w    = render_w,
                render_h    = render_h,
                render_res  = render_res
                )

    def step(self, a):
        """
            Step forward
        """        
        # Before run
        x_pos_before      = self.get_body_com("torso")[0]
        y_pos_before      = self.get_body_com("torso")[1]
        heading_before    = self.get_heading()
        self.prev_state   = np.concatenate([self.sim.data.qpos.flat[2:], self.sim.data.qvel.flat])
        self.prev_torque  = a
        self.contact_data = np.array(self.sim.data.sensordata[3:])

        # Run sim
        self.do_simulation(a, self.frame_skip)
        x_pos_after   = self.get_body_com("torso")[0]
        y_pos_after   = self.get_body_com("torso")[1]
        heading_after = self.get_heading()
        heading_diff  = heading_after - heading_before
        z_pos = self.get_body_com("torso")[2]

        # Accumulate
        self.a    = a
        self.o    = self._get_obs()
        self.r    = (x_pos_after - x_pos_before) / self.dt
        self.info = dict()

        # Done condition
        state   = self.state_vector()
        r, p, y = quaternion_to_euler_angle(state[3], state[4], state[5], state[6])
        notdone = np.isfinite(state).all and abs(r) < 170
        self.d  = not notdone
        
        return self.o, self.r, self.d, self.info
    
    def _get_obs(self):
        """
            Get observation
        """
        self.index = np.array(self.get_max_leg()).reshape(1)
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            self.prev_state,
            self.prev_torque,
            self.index
        ])
    
    def reset_model(self):
        """
            Reset
        """
        o = np.zeros(self.odim)
        return o

    def get_joint_pos_deg(self):
        """
            Get joint position in [Deg]
        """
        q = self.sim.data.qpos.flat
        return np.asarray(
            [q[9],q[10],q[13],q[14],q[17],q[18]]
            )*180.0/np.pi

    def get_heading(self):
        """
            Get z-axis rotation angle in [Deg]
        """
        q = self.data.get_body_xquat('torso')
        _, _, z_deg = quaternion_to_euler_angle(q[0], q[1], q[2], q[3])
        return z_deg
    
    def get_seonsor_data(self):
        """
            Get sensor data from touchsensors
        """
        l2 = self.sim.data.get_sensor('touchsensor_2')
        l3 = self.sim.data.get_sensor('touchsensor_4')
        l4 = self.sim.data.get_sensor('touchsensor_5')
        ls = [l2, l3, l4]
        return ls

    def get_max_leg(self):
        lst   = self.get_seonsor_data()
        score = 0
        index = 0
        for i,j in enumerate(lst):
            if j > score : 
                score = j
                index = i+1
        return index

    def get_time(self):
        """
            Get time in [Sec]
        """
        return self.sim.data.time
    
    def viewer_custom_setup(
        self,
        render_mode = 'human',
        render_w    = 1500,
        render_h    = 1000,
        render_res  = 200
        ):
        """
            View setup
        """
        self.render_mode = render_mode
        self.render_w    = render_w
        self.render_h    = render_h
        self.render_res  = render_res
        
        frame = self.render(
            mode   = self.render_mode,
            width  = self.render_w,
            height = self.render_h)
        self.viewer.cam.distance  = 3.3 # distance to plane (1.5)
        self.viewer.cam.elevation = -45 # elevation angle (-30)
        self.viewer.cam.lookat[0] = 0.0 # x-axis (let this follow the robot)
        self.viewer.cam.lookat[1] = 0.0
        self.viewer.cam.lookat[2] = 0.0
        
    def render_center(self):
        """
            Render with torso-centered
        """
        for d_idx in range(3):
            self.viewer.cam.lookat[d_idx] = self.get_body_com("torso")[d_idx] # follow the robot torso
            # self.viewer.cam.lookat[d_idx] = 0 # fix at zero
        frame = self.render(
            mode   = self.render_mode,
            width  = self.render_w,
            height = self.render_h)
        return frame
    
if __name__ == "__main__":
    # env = Snapbot4EnvClass(render_mode=None)
    env2 = Snapbot4EnvClass(render_mode=None,xml_path='envs/snapbot/xml/snapbot_4/robot_4_1245_heavy.xml')
    # env = Snapbot3EnvClass(render_mode=None)
    for i in range(1000):
        # env.render()
        env2.render()
        # print(env.get_joint_pos_deg())
        # env.step(np.random.standard_normal(8)*1)
        env2.step(np.random.standard_normal(8)*1)

