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

        send_command("ON")             # Turn all LEDs ON
        send_command("BRIGHTNESS 0.9") # Set brightness to 90%
        send_command("BRIGHTNESS 0.8") # Set brightness to 80%
        send_command("BRIGHTNESS 0.7")
        send_command("BRIGHTNESS 0.6")
        send_command("BRIGHTNESS 0.5") # Set brightness to 50%
        send_command("BRIGHTNESS 0.4")
        send_command("BRIGHTNESS 0.3")
        send_command("BRIGHTNESS 0.2")
        send_command("BRIGHTNESS 0.1")  # Set brightness to 10%
        send_command("BRIGHTNESS 1.0")
        send_command("OFF")            # Turn all LEDs OFF
except serial.SerialException as e:
    print(f"Serial error: {e}")
