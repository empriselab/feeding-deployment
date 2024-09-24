#!/bin/bash

# Function to clean up background processes
cleanup() {
    echo "Stopping background processes..."
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

# Start collision monitor
python collision_monitor.py &
collision_monitor_pid=$!  # Store the PID of estops_publisher

# Start transfer button
python transfer_button_listener.py --button_id -1 &
transfer_button_pid=$!  # Store the PID of roscore

# Wait for 2 second to make sure collision monitor is running
sleep 2

# Run watchdog
python watchdog.py 

cleanup  # Ensure cleanup is called when bulldog finishes
wait $collision_monitor_pid
wait $transfer_button_pid
