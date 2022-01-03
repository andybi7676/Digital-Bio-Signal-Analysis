from Ax12 import Ax12
import time


class RobotController:
    def __init__(self, port='COM3', baud=9600, num_motors=4):
        self.AX12 = Ax12
        self.AX12.DEVICENAME = port
        self.AX12.BAUDRATE = baud
        self.AX12.connect()
        self.dxl_motors = []
        self.num_motors = num_motors
        for i in range(self.num_motors):
            self.dxl_motors.append(self.AX12(i))

    def init_pos(self, motor_pos):
        """Initialize positions for each motor"""
        for i in range(self.num_motors):
            self.dxl_motors[i].set_moving_speed(200)    # change the speed as you want
            self.dxl_motors[i].set_goal_position(motor_pos[i])  # initialize position of Joint ith
            time.sleep(0.5)     # set time as you want

    def bow(self, motor_pos):
        """
        Define the method name with your action by yourself. This method is just an example
        @param motor_pos: a list of position of motors
        """
        for i in range(self.num_motors):
            self.dxl_motors[i].set_moving_speed(200)
            self.dxl_motors[i].set_goal_position(motor_pos[i])  # set the goal position
            time.sleep(0.5)    # set time as you want

    def shake(self, motor_pos):
        """
        Define the method name with your action by yourself. This method is just an example
        @param motor_pos: a list of position of motors
        """
        for i in range(self.num_motors):
            self.dxl_motors[i].set_moving_speed(200)
            self.dxl_motors[i].set_goal_position(motor_pos[i])  # set the goal position
            time.sleep(0.5)    # set time as you want

    def up(self, motor_pos):
        """
        Define the method name with your action by yourself. This method is just an example
        @param motor_pos: a list of position of motors
        """
        for i in range(self.num_motors):
            self.dxl_motors[i].set_moving_speed(200)
            self.dxl_motors[i].set_goal_position(motor_pos[i])  # set the goal position
            time.sleep(0.5)    # set time as you want

    def disconnect(self):
        """Disconnect the robot"""
        self.dxl_motors[0].set_torque_enable(0)
        self.AX12.disconnect()


if __name__ == '__main__':
    port = '/dev/ttyUSB0'   # change the port as your setting
    baud = 1000000          # change the baudrate as your setting
    num_motors = 4  # define number of motors you use
    controller = RobotController(port='/dev/ttyUSB0', baud=1000000, num_motors=4)
    controller.init_pos([220, 512, 300, 512])   # define positions for each motor
    try:
        while True:
            input_pose = int(input("Choose the pose u want to do:  1:Bow, 2:Shake, 3:Disconnect\n"))
            if input_pose == 1:
                controller.bow([220, 300, 200, 512])    # define position of each motor for your action
            elif input_pose == 2:
                controller.shake([0, 512, 500, 0])      # define position of each motor for your action
            elif input_pose == 3:
                break
    except KeyboardInterrupt:
        print('Press Ctrl-C to terminate while statement')
    controller.disconnect()
