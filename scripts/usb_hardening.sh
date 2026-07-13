#!/usr/bin/env bash
#
# USB hardening for the sensor USB paths: arm RealSense + its OWC TB4 hub
# chain, and the lidars + their Realtek tree on the PCH.
#
# Why (two incidents):
#  - Jul 3: the D435i's depth-endpoint stall recovery (CLEAR_HALT) escalated to
#    a controller-wide reset that dropped the two lidars sharing the PCH xHCI.
#  - Jul 13 (post TB4-hub rewire): starting the realsense2 node with the camera
#    runtime-suspended breaks stream start behind the hub chain — random subsets
#    of streams come up at 0 Hz (endpoint-130 watchdog, "HW not ready", EAGAIN
#    control transfers); launched awake, all streams start clean at 30 Hz.
# This applies librealsense's recommended USB settings:
#   1. usbfs_memory_mb -> 128   (kernel default 16 is too small for the dual stream)
#   2. USB autosuspend off for the RealSense, both OWC TB4 hubs, CP210x lidars,
#      and Realtek hubs
# It also installs the persistent udev rule for (2).
#
# Re-run this script after any RealSense hardware reset (initial_reset or storm
# recovery): the re-enumerated device instance can come back at power/control=auto.
#
# The ZED is on Hub 1 and its autosuspend is disabled via 99-slabs.rules; it is
# intentionally NOT touched by this script.
#
# Usage:  sudo scripts/usb_hardening.sh
#
# usbfs_memory_mb here is applied live only; persistence across reboots is a
# one-time GRUB change (printed at the end).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RULE_SRC="${REPO_ROOT}/config/udev/99-usb-hardening.rules"
RULE_DST="/etc/udev/rules.d/99-usb-hardening.rules"
USBFS_MB=128

# Autosuspend-off targets: idVendor:idProduct of every device in the sensor
# USB paths (RealSense, lidars, Realtek lidar-tree hubs, OWC TB4 hub functions).
TARGETS=("8086:0b3a" "10c4:ea60" "0bda:5411" "0bda:0411" "8087:0b40" "1d5c:5801")

if [[ ${EUID} -ne 0 ]]; then
  echo "This script needs root. Re-run: sudo $0" >&2
  exit 1
fi

echo "[1/3] usbfs_memory_mb: $(cat /sys/module/usbcore/parameters/usbfs_memory_mb) -> ${USBFS_MB} (live)"
echo "${USBFS_MB}" > /sys/module/usbcore/parameters/usbfs_memory_mb

echo "[2/3] Disabling USB autosuspend on connected target devices (live)"
for dev in /sys/bus/usb/devices/*; do
  [[ -r "${dev}/idVendor" && -r "${dev}/idProduct" ]] || continue
  id="$(cat "${dev}/idVendor"):$(cat "${dev}/idProduct")"
  for t in "${TARGETS[@]}"; do
    if [[ "${id}" == "${t}" && -w "${dev}/power/control" ]]; then
      echo "on" > "${dev}/power/control"
      echo "    ${dev##*/}  ${id}  power/control=on"
    fi
  done
done

echo "[3/3] Installing udev rule -> ${RULE_DST}"
install -m 0644 "${RULE_SRC}" "${RULE_DST}"
udevadm control --reload-rules
udevadm trigger --subsystem-match=usb --action=add

cat <<'EOF'

Done (live settings + persistent udev rule installed).

To persist usbfs_memory_mb across reboots, add the kernel arg to GRUB once
(check it is not already present first):

  grep -q 'usbcore.usbfs_memory_mb' /etc/default/grub \
    || sudo sed -i 's/\(GRUB_CMDLINE_LINUX_DEFAULT="[^"]*\)"/\1 usbcore.usbfs_memory_mb=128"/' /etc/default/grub
  sudo update-grub
  # takes effect on next reboot

Recommended (robot is on AC power): also disable USB autosuspend globally at
the kernel level, which closes every udev-race / reset-revert gap for good:

  grep -q 'usbcore.autosuspend' /etc/default/grub \
    || sudo sed -i 's/\(GRUB_CMDLINE_LINUX_DEFAULT="[^"]*\)"/\1 usbcore.autosuspend=-1"/' /etc/default/grub
  sudo update-grub
  # takes effect on next reboot

Verify:
  cat /sys/module/usbcore/parameters/usbfs_memory_mb        # -> 128
  for d in /sys/bus/usb/devices/*; do \
    [ -r "$d/idProduct" ] && [ "$(cat $d/idProduct)" = 0b3a ] && \
    echo "RealSense $d power/control=$(cat $d/power/control)"; done   # -> on
EOF
