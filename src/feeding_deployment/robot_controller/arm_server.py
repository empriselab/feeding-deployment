import numpy as np
import signal
import sys
try:
    from feeding_deployment.robot_controller.kinova import KinovaArm
except ImportError:
    print(
        "KinovaArm import failed, continuing without executing arm commands on real robot"
    )
from feeding_deployment.robot_controller.arm_interface import ArmInterface, ArmManager, NUC_HOSTNAME, ARM_RPC_PORT, RPC_AUTHKEY

# Create a single instance of KinovaArm and ArmInterface
kinova_arm_instance = KinovaArm()
arm_interface_instance = ArmInterface(kinova_arm_instance)

# Register ArmInterface but return the existing instance
ArmManager.register("ArmInterface", lambda: arm_interface_instance)

# Signal handler function to call arm_interface_instance.stop() on Ctrl-C or Ctrl-\
def signal_handler(sig, frame):
    print(f"Signal {sig} received, stopping the arm.")
    arm_interface_instance.stop()
    sys.exit(0)

# Register signal handler for Ctrl-C (SIGINT) and Ctrl-\ (SIGQUIT)
signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl-C
signal.signal(signal.SIGQUIT, signal_handler)  # Handle Ctrl-\

if __name__ == "__main__":
    manager = ArmManager(address=(NUC_HOSTNAME, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    server = manager.get_server()
    print(f"Arm manager server started at {NUC_HOSTNAME}:{ARM_RPC_PORT}")
    server.serve_forever()
