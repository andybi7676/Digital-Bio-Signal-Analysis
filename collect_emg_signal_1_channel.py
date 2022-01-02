import serial
import csv

if __name__ == '__main__':
    fn = 'action_1_1.csv'
    f = open(fn, 'w')
    cwriter = csv.writer(f)
    cwriter.writerow('CH1')
    port = 'COM1'   # change the port name as your settings
    baud = 9600     # change the port name as your settings
    s = serial.Serial(port='COM1', baudrate=baud)
    try:
        while True:
            val = s.readline()              # read byte value
            val = val.decode('utf-8').rstrip()     # Decode byte value to string value and remove \n and \r
            val = float(val)            # convert to float
            cwriter.writerow(val)
    except KeyboardInterrupt:
        # Press Ctrl-C to stop collect data
        pass
    # close csv file
    f.close()


