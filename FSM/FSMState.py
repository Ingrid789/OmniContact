from common.path_config import PROJECT_ROOT

from common.utils import FSMStateName    

class FSMState:
    def __init__(self):
        self.name = FSMStateName.INVALID
        self.name_str = "invalid"
        self.control_dt = 0.02
    def enter(self):
        raise NotImplementedError("enter() function must be implement!")
    
    def run(self):
        raise NotImplementedError("run() function must be implement!")
    
    def exit(self):
        raise NotImplementedError("exit() function must be implement!")
    
    def checkChange(self):
        # joystick callback
        raise NotImplementedError("checkChange() function must be implement!")
        