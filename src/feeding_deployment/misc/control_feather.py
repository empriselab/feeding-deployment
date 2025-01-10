import serial
import time

# Replace with your Feather board's serial port
SERIAL_PORT = "/dev/ttyACM0"  # Adjust as needed
BAUD_RATE = 115200

try:
    # Open the serial connection
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
        time.sleep(2)  # Wait for Feather board to initialize

        # Function to send a single command
        def send_command(command):
            ser.reset_input_buffer()  # Clear input buffer
            ser.reset_output_buffer()  # Clear output buffer
            ser.write(f"{command}\r\n".encode())  # Send the command
            print(f"Sent: {command}")
            time.sleep(2)  # Wait for 2 seconds to allow Feather to process

        # Send commands one by one
        send_command("ON")    # Turn all LEDs ON
        send_command("RED")   # Set LEDs to RED
        send_command("GREEN") # Set LEDs to GREEN
        send_command("BLUE")  # Set LEDs to BLUE
        send_command("OFF")   # Turn all LEDs OFF
except serial.SerialException as e:
    print(f"Serial error: {e}")
