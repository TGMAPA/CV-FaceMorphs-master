# ==== Start session ====
# Create new session with a name
tmux new -s <session_name>

# Close session without killing the process (ctrl + b) then press D

# List sessions
tmux ls

# Reconnect to session 
tmux attach -t <session_name>

# Kill session
tmux kill-session -t <session_name>


# ==== Run pipeline ====
# Inside the tmux session or in terminal
TF_USE_LEGACY_KERAS=1 LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib python -u main.py 2>&1 | tee -a outputs/pipeline.log

# See the last log lines live
tail -f outputs/pipeline.log

