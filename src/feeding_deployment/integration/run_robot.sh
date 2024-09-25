#!/bin/bash

# Function to clean up background processes
cleanup() {
    echo "Stopping background processes..."
    if kill -0 $joint_states_publisher_pid 2>/dev/null; then
        kill $joint_states_publisher_pid
    fi
    if kill -0 $collision_monitor_pid 2>/dev/null; then
        kill $collision_monitor_pid
    fi
    if kill -0 $transfer_button_pid 2>/dev/null; then
        kill $transfer_button_pid
    fi
    if kill -0 $watchdog_pid 2>/dev/null; then
        kill $watchdog_pid
    fi
}

# Trap Ctrl+C and call cleanup
trap cleanup SIGINT

# Start joint states publisher
cd /home/isacc/deployment_ws/src/feeding-deployment/src/feeding_deployment/robot_controller
python joint_states_publisher.py &
joint_states_publisher_pid=$!  # Store the PID of joint_states_publisher

# move to safety directory
cd /home/isacc/deployment_ws/src/feeding-deployment/src/feeding_deployment/safety

# Start transfer button
python transfer_button_listener.py --button_id -1 &
transfer_button_pid=$!  # Store the PID of transfer_button_listener

# Start collision monitor
python collision_monitor.py &
collision_monitor_pid=$!  # Store the PID of collision_monitor

# Wait for 2 second to make sure collision monitor is running
sleep 2

# Run watchdog
python watchdog.py 

cleanup  # Ensure cleanup is called when bulldog finishes
wait $joint_states_publisher_pid
wait $collision_monitor_pid
wait $transfer_button_pid
