#!/bin/bash
# Rebuild and push the site every 15 minutes (replays rendered with nice). Stop: kill the PID in publish_loop.pid.
cd "$(dirname "$0")"
echo $$ > publish_loop.pid
while true; do
  nice -n 15 .venv/bin/python publish.py --push >> publish.log 2>&1
  sleep 900
done
