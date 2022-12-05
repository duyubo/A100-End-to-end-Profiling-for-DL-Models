watch -n 0.2 'nvidia-smi | grep "250W" >> BERT_power.log'&
p=$!
python test_bert.py >> BERT_profile.txt
kill -9 $p
