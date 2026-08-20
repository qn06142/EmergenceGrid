#!/bin/bash
# Watcher: wait for group g64c step200 ckpt, render baseline GIF, kill old run,
# relaunch on the FIXED (walled oasis) env.
cd /d/rl-emergence
. .venv/Scripts/activate
CKPT="D:/rl-emergence/ckpts/eg16_g64c/eg16_g64c_policy_step200.pt"
echo "watcher: waiting for $CKPT"
for i in $(seq 1 240); do  # up to ~120 min
  if [ -f "$CKPT" ]; then
    echo "watcher: checkpoint found"
    break
  fi
  sleep 30
done
if [ ! -f "$CKPT" ]; then echo "watcher: timeout, no ckpt"; exit 1; fi

# 1) render baseline GIF from the OLD (open-oasis) policy
echo "watcher: rendering baseline GIF (old env)"
python src/render.py --ckpt "$CKPT" --n 16 --grid 64 --steps 250 --out "D:/rl-emergence/gifs/group_g64c_baseline.gif" --seed 7 --greedy --fps 10

# 2) kill the OLD (open-oasis) group run by command signature, so GPU is free
echo "watcher: killing old open-oasis group run"
pkill -f "exp eg16_g64c" 2>/dev/null
sleep 3

# 3) relaunch training on the FIXED (walled oasis) env under new exp name
echo "watcher: relaunching on fixed env (eg16_g64d)"
nohup python src/train.py --n 16 --grid 64 --k 8 --nstep 64 --nupd 600 --log_every 25 --save_every 200 --ckpt_dir /d/rl-emergence/ckpts/eg16_g64d --exp eg16_g64d > /d/rl-emergence/train_g64d_group.log 2>&1 &
echo "watcher: relaunched pid $!"
