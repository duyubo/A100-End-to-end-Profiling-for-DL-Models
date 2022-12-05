watch -n 0.2 'nvidia-smi | grep "250W" >> MLP_power.log'&
p=$!
python MLP.py >> MLP_profile.txt
kill -9 $p
