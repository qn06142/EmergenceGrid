#!/bin/bash
cd /d/rl-emergence
. .venv/Scripts/activate
seen=""
for i in $(seq 1 900); do
  gck=$(ls -1 D:/rl-emergence/ckpts/eg16_g64e/eg16_g64e_policy_step*.pt 2>/dev/null | grep -v step0 | sort | tail -1)
  base=$(basename "$gck" 2>/dev/null)
  if [ -n "$base" ] && [ "$base" != "$seen" ]; then
    seen="$base"
    step=$(echo "$base" | grep -oE 'step[0-9]+' | grep -oE '[0-9]+')
    sck="D:/rl-emergence/ckpts/solo_g64e/solo_g64e_policy_step${step}.pt"
    echo "watch_g64e: new group ckpt step${step}"
    python src/render.py --ckpt "$gck" --n 16 --grid 64 --steps 250 --out "D:/rl-emergence/gifs/group_g64e_step${step}.gif" --seed 7 --fps 10
    if [ -f "$sck" ]; then
      echo "watch_g64e: computing ECI at step${step}"
      python src/eval_eci.py --group_ckpt "$gck" --solo_ckpt "$sck" --grid 64 --n 16 --episodes 12 --device cuda 2>&1 | tail -15
    else
      echo "watch_g64e: solo step${step} not ready yet, skipping ECI"
    fi
  fi
  sleep 30
done
echo "watch_g64e: DONE"
