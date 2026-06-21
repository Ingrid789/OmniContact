from common.path_config import PROJECT_ROOT

from policy.passive.PassiveMode import PassiveMode
from policy.defaultpose.DefaultPose import DefaultPose
from policy.loco_mode.LocoMode import LocoMode
from policy.skill_cooldown.SkillCooldown import SkillCooldown
from policy.omnicontact.OmniContact import OmniContact
from FSM.FSMState import *
import time
from common.ctrlcomp import *
from enum import Enum, unique

@unique
class FSMMode(Enum):
    CHANGE = 1
    NORMAL = 2

class FSM:
    def __init__(self, state_cmd:StateAndCmd, policy_output:PolicyOutput):
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.cur_policy : FSMState
        self.next_policy : FSMState
        
        self.FSMmode = FSMMode.NORMAL
        
        self.passive_mode = PassiveMode(state_cmd, policy_output)
        self.default_pose = DefaultPose(state_cmd, policy_output)
        self.loco_policy = LocoMode(state_cmd, policy_output)
        self.skill_cooldown_policy = SkillCooldown(state_cmd, policy_output)
        self.omnicontact = OmniContact(state_cmd, policy_output)

        print("initalized all policies!!!")
        
        self.cur_policy = self.passive_mode
        print("current policy is ", self.cur_policy.name_str)
        
        
        
    def run(self):
        self.cur_policy.run()
        nextPolicyName = self.cur_policy.checkChange()
        if(nextPolicyName != self.cur_policy.name):
            # change policy
            self.cur_policy.exit()
            self.get_next_policy(nextPolicyName)
            print("Switched to ", self.cur_policy.name_str)
            self.cur_policy.enter()
            self.cur_policy.run()

    def absoluteWait(self, control_dt, start_time):
        end_time = time.time()
        delta_time = end_time - start_time
        if(delta_time < control_dt):
            time.sleep(control_dt - delta_time)
        else:
            print("inference time beyond control horzion!!!")
            
            
    def get_next_policy(self, policy_name:FSMStateName):
        if(policy_name == FSMStateName.PASSIVE):
            self.cur_policy = self.passive_mode
        elif((policy_name == FSMStateName.DEFAULTPOSE)):
            self.cur_policy = self.default_pose
        elif((policy_name == FSMStateName.LOCOMODE)):
            self.cur_policy = self.loco_policy
        elif((policy_name == FSMStateName.SKILL_COOLDOWN)):
            self.cur_policy = self.skill_cooldown_policy
        elif((policy_name == FSMStateName.SKILL_OmniContact)):
            self.cur_policy = self.omnicontact
        else:
            pass
