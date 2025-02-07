import argparse
import threading
import time
import pickle as pk

from rsa import sign
import serial
from filter import BandpassFilter1D, NotchFilter1D
from processing import MeanShift1D, Detrend1D, Resample1D, Normalize1D
from feature_extraction import *
from Ax12 import Ax12
import torch
from model import ResNetClassifier

# process signal of each channel
def process_signal1d(x, raw_fs=1000, low_fs=1, high_fs=120, notch_fs=60, Q=20, window_size=250, step_size=50, target_fs=512):
    """
    @param x: signal of a single channel
    @param raw_fs: original sampling rate
    @param low_fs: low cutoff frequency
    @param high_fs: high cutoff frequency
    @param notch_fs: notch cutoff frequency
    @param Q: Q factor
    @param window_size: windows size for detrending
    @param step_size: step size for detrending
    @param target_fs: target sampling rate for resampling step
    """
    # mean-correct signal
    x_processed = MeanShift1D.apply(x)
    # filtering noise
    x_processed = BandpassFilter1D.apply(x_processed, low_fs, high_fs, order=4, fs=raw_fs)
    x_processed = NotchFilter1D.apply(x_processed, notch_fs, Q=Q, fs=raw_fs)
    # detrend
    x_processed = Detrend1D.apply(x_processed, detrend_type='locreg', window_size=window_size, step_size=step_size)
    # resample
    x_processed = Resample1D.apply(x_processed, raw_fs, target_fs)
    # rectify
    x_processed = abs(x_processed)
    # normalize
    x_processed = Normalize1D.apply(x_processed, norm_type='min_max')
    return x_processed


# process multi-channel signal
def process_signalnd(x, raw_fs=1000, low_fs=1, high_fs=120, notch_fs=60, Q=20, window_size=250, step_size=50, target_fs=512):
    """
    @param x: signal of a single channel
    @param raw_fs: original sampling rate
    @param low_fs: low cutoff frequency
    @param high_fs: high cutoff frequency
    @param notch_fs: notch cutoff frequency
    @param Q: Q factor
    @param window_size: windows size for detrending
    @param step_size: step size for detrending
    @param target_fs: target sampling rate for resampling step
    """
    num_channels = x.shape[1]
    x_processed = np.array([])
    for i in range(num_channels):
        # process each channel
        channel_processed = process_signal1d(x[:, i], raw_fs, low_fs, high_fs, notch_fs, Q, window_size, step_size, target_fs)
        channel_processed = np.expand_dims(channel_processed, axis=1)
        if i == 0:
            x_processed = channel_processed
            continue
        x_processed = np.hstack((x_processed, channel_processed))
    return x_processed


# The class to connect the electrodes through serial
class SerialPort:
    def __init__(self, port='COM1', baud=115200, cls=None, controller=None, num_channels=4, interval=1000, timeout=0.1, device='cpu'):
        super(SerialPort, self).__init__()
        self.port = serial.Serial(port, baud)
        self.signal = None
        self.interval = interval
        self.cls = cls
        self.num_channels = num_channels
        self.timeout = timeout
        self.feature_window_size = 50   # Please modify as your setting
        self.concat = True   # Please change as your setting
        self.avg_pool = True # Please change as your setting
        self.controller = controller
        self.device = device

    def serial_open(self):
        if not self.port.isOpen():
            self.port.open()

    def serial_close(self):
        self.port.close()

    def serial_send(self):
        print('Send action...')
        time.sleep(self.timeout)
        if self.action == '0':
            # define your '0' action
            self.controller.bow(motor_pos=[220, 300, 200, 512])   # set motor positions as your setting
        elif self.action == '1':
            # define your '1' action
            self.controller.shake(motor_pos=[0, 512, 500, 0])  # set motor positions as your setting
        elif self.action == '2':
            # define your '3' action
            self.controller.up(motor_pos=[0, 200, 512, 128])    # set motor positions as your setting

    def serial_read(self):
        print('Receiving signal...')
        self.action = '0'

        while True:
            values = []
            # read signal from serial
            for i in range(self.interval):
                string = self.port.readline().decode('utf-8').rstrip()  # Read and decode a byte string
                values.extend([float(value) for value in string.split(' ')])
            # reshape signal
            signal = np.reshape(np.array(values), (self.interval, self.num_channels), order='C')
            # process signal
            # please change parameters as your settings
            signal_processed = process_signalnd(signal, raw_fs=1000, low_fs=10, high_fs=120, notch_fs=60, Q=20, window_size=512, step_size=50, target_fs=512)
            # extract, transpose and flatten feature vectors
            # change your feature as your setting
            # peak = MaxPeak.apply(signal_processed, self.feature_window_size).T.flatten()
            # mean = Mean.apply(signal_processed, self.feature_window_size).T.flatten()
            # var = Variance.apply(signal_processed, self.feature_window_size).T.flatten()
            # std = StandardDeviation.apply(signal_processed, self.feature_window_size).T.flatten()
            # skew = Skewness.apply(signal_processed, self.feature_window_size).T.flatten()
            # kurt = Kurtosis.apply(signal_processed, self.feature_window_size).T.flatten()
            # rms = RootMeanSquare.apply(signal_processed, self.feature_window_size).T.flatten()
            # if self.concat:
            #     feature = np.hstack([peak, mean, var, std, skew, kurt, rms])
            #     feature = np.expand_dims(feature, axis=0)
            # else:
            #     feature = np.vstack([peak, mean, var, std, skew, kurt, rms])
            #     if self.avg_pool:
            #         # average pooling
            #         feature = feature.mean(axis=0)
            #     else:
            #         # max pooling
            #         feature = feature.max(axis=0)
            #     feature = np.expand_dims(feature, axis=0)
            # if self.pca:
            #     feature = self.pca.transform(feature)
            signal_processed = np.expand_dims(signal_processed.astype(np.float32), axis=0)
            # print(signal_processed.shape) # check shape = (1, 512, 4) ? if you got an error~
            feature = torch.FloatTensor(signal_processed).to(self.device)
            with torch.no_grad():
                outs = self.cls(feature)
                y_preds = outs.argmax(dim=-1).cpu().numpy()[0]
                # y_preds = self.cls.predict(signal_processed)
                self.action = str(y_preds)


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
            time.sleep(0.1)     # set time as you want

    def bow(self, motor_pos):
        """
        Define the method name with your action by yourself. This method is just an example
        @param motor_pos: a list of position of motors
        """
        for i in range(self.num_motors):
            self.dxl_motors[i].set_moving_speed(200)
            self.dxl_motors[i].set_goal_position(motor_pos[i])  # set the goal position
            time.sleep(0.1)    # set time as you want

    def shake(self, motor_pos):
        """
        Define the method name with your action by yourself. This method is just an example
        @param motor_pos: a list of position of motors
        """
        for i in range(self.num_motors):
            self.dxl_motors[i].set_moving_speed(200)
            self.dxl_motors[i].set_goal_position(motor_pos[i])  # set the goal position
            time.sleep(0.1)    # set time as you want

    def up(self, motor_pos):
        """
        Define the method name with your action by yourself. This method is just an example
        @param motor_pos: a list of position of motors
        """
        for i in range(self.num_motors):
            self.dxl_motors[i].set_moving_speed(200)
            self.dxl_motors[i].set_goal_position(motor_pos[i])  # set the goal position
            time.sleep(0.1)    # set time as you want

    def disconnect(self):
        """Disconnect the robot"""
        self.dxl_motors[0].set_torque_enable(0)
        self.AX12.disconnect()


if '__name__' == '__main__':
    # Set command line arguments
    parser = argparse.ArgumentParser(description='Real-time robot-arm controlling')
    parser.add_argument('--arduport', type=str, default='COM1', help='COM port for arduino')
    parser.add_argument('--ardubaud', type=int, default=115200, help='Baud rate for arduino')
    parser.add_argument('--axport', type=str, default='COM3', help='COM port for dynamix 12')
    parser.add_argument('--axbaud', type=int, default=9600, help='Baud rate for dynamix 12')
    parser.add_argument('--num-motors', type=int, default=4, help='Number of motors')
    parser.add_argument('--channels', type=int, default=4, help='The number of channels')
    parser.add_argument('--segment', type=int, default=1000, help='Segmentation interval')
    parser.add_argument('--timeout', type=float, default=1, help='Time out')
    args = parser.parse_args()
    # define reboot
    controller = RobotController(port=args.axport, baud=args.axbaud, num_motors=args.num_motors)
    controller.init_pos([220, 512, 300, 512])
    # define classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetClassifier(
        label_num=3,
        in_channels=[4, 16, 16],
        out_channels=[16, 16, 8],
        downsample_scales=[1, 1, 1],
        kernel_size=5,
        z_channels=8,
        dilation=True,
        leaky_relu=True,
        dropout=0.0,
        stack_kernel_size=3,
        stack_layers=2,
        nin_layers=0,
        stacks=[3, 3, 3],
    ).to(device)
    model.load_state_dict(torch.load("./DL/resnet_best.ckpt", map_location=device))
    model.eval()
    # cls = pk.load(open('svc.pkl', 'rb'))
    # define pre-processing pca
    # pca = pk.load(open('pca.pkl', 'rb'))
    # Setup serial line
    mserial = SerialPort(args.port, args.baud_rate, model, controller, args.channels, args.segment, args.timeout)
    t1 = threading.Thread(target=mserial.serial_read)
    t1.start()
    try:
        while True:
            mserial.serial_send()
    except KeyboardInterrupt:
        print('Press Ctrl-C to terminate while statement')
    mserial.serial_close()
    controller.disconnect()
