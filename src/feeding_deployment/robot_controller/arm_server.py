import numpy as np
import signal
import sys
from feeding_deployment.robot_controller.kinova import KinovaArm
from feeding_deployment.robot_controller.arm_interface import ArmInterface, ArmManager, NUC_HOSTNAME, ARM_RPC_PORT, RPC_AUTHKEY

y = input("Is bulldog ready to be run again? (y/n): ")
if y != 'y':
    sys.exit(0)

# Create a single instance of KinovaArm and ArmInterface
kinova_arm_instance = KinovaArm()
arm_interface_instance = ArmInterface(kinova_arm_instance)

# Register ArmInterface but return the existing instance
ArmManager.register("ArmInterface", lambda: arm_interface_instance)

# Flag to check if the signal handler has been triggered
signal_triggered = False

# Signal handler function to call arm_interface_instance.stop() on Ctrl-C or Ctrl-\
def signal_handler(sig, frame):
    global signal_triggered
    if not signal_triggered:
        signal_triggered = True
        print(f"Signal {sig} received, stopping the arm.")
        arm_interface_instance.close()
        sys.exit(0)
    else:
        print(f"Signal {sig} received, but handler is already processing.")

# Register signal handler for Ctrl-C (SIGINT) and Ctrl-\ (SIGQUIT)
signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl-C
signal.signal(signal.SIGQUIT, signal_handler)  # Handle Ctrl-\

if __name__ == "__main__":
    manager = ArmManager(address=(NUC_HOSTNAME, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    server = manager.get_server()
    print(f"Arm manager server started at {NUC_HOSTNAME}:{ARM_RPC_PORT}")
    server.serve_forever()
