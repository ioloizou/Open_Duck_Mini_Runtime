#!/usr/bin/env bash
set -u

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)

echo "Logging to logs/kernel_usb_${TS}.log"
echo "Press Ctrl+C to stop."

{
  echo "=== START $(date -Is) ==="
  echo
  echo "=== Initial tty devices ==="
  ls -l /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || true
  echo
  echo "=== dmesg follow ==="
} | tee "logs/kernel_usb_${TS}.log"

sudo dmesg -wT | tee -a "logs/kernel_usb_${TS}.log"