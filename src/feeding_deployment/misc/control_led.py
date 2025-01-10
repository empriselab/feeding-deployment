"""
Example for Adafruit USB tower light w/alarm
don't forge to `pip install pyserial` or `pip3 install pyserial`
"""

import serial
import time

serialPort = '/dev/ttyUSB0'  # Change to the serial/COM port of the tower light
#serialPort = '/dev/USBserial0'  # on mac/linux, it will be a /dev path
baudRate = 9600

RED_ON = 0x11
RED_OFF = 0x21
RED_BLINK = 0x41

YELLOW_ON= 0x12
YELLOW_OFF = 0x22
YELLOW_BLINK = 0x42

GREEN_ON = 0x14
GREEN_OFF = 0x24
GREEN_BLINK = 0x44

BLUE_ON = 0x15
BLUE_OFF = 0x25
BLUE_BLINK = 0x45

WHITE_ON = 0x16
WHITE_OFF = 0x26
WHITE_BLINK = 0x46

def sendCommand(serialport, cmd):
    serialport.write(bytes([cmd]))

if __name__ == '__main__':
    mSerial = serial.Serial(serialPort, baudRate)

    input("Press Enter to turn off all lights")
    sendCommand(mSerial, RED_OFF)
    sendCommand(mSerial, YELLOW_OFF)
    sendCommand(mSerial, GREEN_OFF)

    # iterate through all possible codes
    for i in range(0, 255):
        input(f"Press Enter to send command {i}")
        sendCommand(mSerial, i)
        time.sleep(1)

    # Please be kind, re-wind!
    input("Press Enter to turn off all lights")
    sendCommand(mSerial, RED_OFF)
    sendCommand(mSerial, YELLOW_OFF)
    sendCommand(mSerial, GREEN_OFF)
    mSerial.close()