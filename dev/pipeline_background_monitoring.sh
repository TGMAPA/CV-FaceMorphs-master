
# Create new session with a name
tmux new -s <session_name>

# Close session without killing the process (ctrl + b) then press D

# Reconnect to session 
tmux attach -t <session_name>

# Inside the tmux session or in terminal
python -u main.py 2>&1 | tee -a outputs/pipeline.log

# See the last log lines live
tail -f outputs/pipeline.log

