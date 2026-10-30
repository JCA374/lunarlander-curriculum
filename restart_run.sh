#!/bin/bash
# Restart a supervised run cleanly (it resumes from its checkpoint):  ./restart_run.sh K curricula/K.json [train args...]
# Patterns are anchored to the start of the command line, so they never match the shell running this script
# (an unanchored 'pgrep -f "supervise.sh K"' matched itself and waited forever, 2026-10-30).
cd "$(dirname "$0")"
RUN=$1; CUR=$2; shift 2
main=$(pgrep -f "^.venv/bin/python train.py --curriculum $CUR " | head -1)
[ -n "$main" ] && kill -TERM "$main"
while pgrep -f "^.venv/bin/python train.py --curriculum $CUR " > /dev/null; do sleep 2; done
while pgrep -f "^/bin/bash ./supervise.sh $RUN " > /dev/null; do sleep 1; done
setsid nohup ./supervise.sh "$RUN" "$CUR" "$@" > "logs/supervise_$RUN.log" 2>&1 < /dev/null &
sleep 2
echo "supervisors: $(pgrep -f "^/bin/bash ./supervise.sh $RUN " | wc -l), trainer processes: $(pgrep -f "^.venv/bin/python train.py --curriculum $CUR " | wc -l)"
