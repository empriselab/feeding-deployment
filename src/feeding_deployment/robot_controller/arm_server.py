import numpy as np
from feeding_deployment.robot_controller.arm_interface import ArmManager, NUC_HOSTNAME, ARM_RPC_PORT, RPC_AUTHKEY

if __name__ == "__main__":
    manager = ArmManager(address=(NUC_HOSTNAME, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    server = manager.get_server()
    print(f"Arm manager server started at {NUC_HOSTNAME}:{ARM_RPC_PORT}")
    server.serve_forever()
