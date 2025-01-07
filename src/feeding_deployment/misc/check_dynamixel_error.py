from dynamixel_sdk import *  # Dynamixel SDK library

# Constants
BAUDRATE = 1000000
PORT_NAME = '/dev/ttyUSB0'
PROTOCOL_VERSION = 2.0

ADDR_HARDWARE_ERROR_STATUS = 70  # Hardware Error Status address
JT1_ID = 100  # Dynamixel ID for the problematic motor

# Initialize PortHandler and PacketHandler
port_handler = PortHandler(PORT_NAME)
packet_handler = PacketHandler(PROTOCOL_VERSION)

def open_port_and_set_baudrate():
    """Opens the port and sets the baudrate."""
    if port_handler.openPort():
        print("Port opened successfully.")
    else:
        print("Failed to open port.")
        return False

    if port_handler.setBaudRate(BAUDRATE):
        print(f"Baudrate set to {BAUDRATE}.")
        return True
    else:
        print("Failed to set baudrate.")
        return False

def read_hardware_error_status(dxl_id):
    """Reads and prints the Hardware Error Status of the given Dynamixel ID."""
    dxl_error_status, dxl_comm_result, dxl_error = packet_handler.read1ByteTxRx(port_handler, dxl_id, ADDR_HARDWARE_ERROR_STATUS)
    
    if dxl_comm_result != COMM_SUCCESS:
        print(f"[ERROR] Communication: {packet_handler.getTxRxResult(dxl_comm_result)}")
    elif dxl_error != 0:
        print(f"[ERROR] Packet: {packet_handler.getRxPacketError(dxl_error)}")
    else:
        print(f"Motor ID {dxl_id}: Hardware Error Status = {dxl_error_status}")
        return dxl_error_status

    return None

def close_port():
    """Closes the port."""
    port_handler.closePort()
    print("Port closed.")

if __name__ == "__main__":
    if open_port_and_set_baudrate():
        try:
            # Read hardware error status for JT1
            print("Checking Hardware Error Status...")
            error_status = read_hardware_error_status(JT1_ID)
            if error_status is not None:
                if error_status == 0:
                    print(f"Motor ID {JT1_ID}: No hardware errors detected.")
                else:
                    print(f"Motor ID {JT1_ID}: Hardware Error Detected! Status = {error_status}")
                    # Decode specific errors based on the bit flags
                    if error_status & 0x01:
                        print("  - Input voltage error.")
                    if error_status & 0x02:
                        print("  - Overheating error.")
                    if error_status & 0x04:
                        print("  - Motor encoder error.")
                    if error_status & 0x08:
                        print("  - Electrical shock error.")
                    if error_status & 0x10:
                        print("  - Overload error.")
        finally:
            close_port()
