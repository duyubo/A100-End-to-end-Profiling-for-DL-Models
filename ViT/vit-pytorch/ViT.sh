watch -n 0.2 'nvidia-smi | grep "250W" >> VIT_power.log'&
p=$!
python ViT.py >> VIT_profile.txt
kill -9 $p
