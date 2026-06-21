from common.path_config import PROJECT_ROOT

from FSM.FSMState import FSMStateName, FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput
import numpy as np
import yaml
from common.utils import FSMCommand
import os

class DefaultPose(FSMState):
    def __init__(self, state_cmd:StateAndCmd, policy_output:PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.DEFAULTPOSE
        self.name_str = "default_pose"
        self.alpha = 0.
        self.cur_step = 0
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "DefaultPose.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.kds = np.array(config["kds"], dtype=np.float32)
            self.kps = np.array(config["kps"], dtype=np.float32)
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)
            self.joint2motor_idx = np.array(config["joint2motor_idx"], dtype=np.int32)
            self.control_dt = config["control_dt"]
    
    def enter(self):
        print("Moving to default pos(configuration A).")
        self.total_time = 2.0
        self.num_step = int(self.total_time / self.control_dt)
        self.dof_size = len(self.joint2motor_idx)
        self.init_dof_pos = np.zeros(self.dof_size, dtype=np.float32)
        self.alpha = 0.
        self.cur_step = 0
        for i in range(self.dof_size):
            self.init_dof_pos[i] = self.state_cmd.q[self.joint2motor_idx[i]]
        
        
    def run(self):
        self.cur_step += 1
        self.alpha = min(self.cur_step / self.num_step, 1.0)
        for j in range(self.dof_size):
            motor_idx = self.joint2motor_idx[j]
            target_pos = self.default_angles[j]
            self.policy_output.actions[motor_idx] = self.init_dof_pos[j] * (1 - self.alpha) + target_pos * self.alpha
            self.policy_output.kps[motor_idx] = self.kps[j]
            self.policy_output.kds[motor_idx] = self.kds[j]
    
    def exit(self):
        for j in range(self.dof_size):
            motor_idx = self.joint2motor_idx[j]
            self.policy_output.actions[motor_idx] = self.default_angles[j]
            self.policy_output.kps[motor_idx] = self.kps[j]
            self.policy_output.kds[motor_idx] = self.kds[j]
    
    def checkChange(self):
        if(self.state_cmd.skill_cmd == FSMCommand.LOCO):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.LOCOMODE
        elif(self.state_cmd.skill_cmd == FSMCommand.PASSIVE):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.PASSIVE
        else:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.DEFAULTPOSE
