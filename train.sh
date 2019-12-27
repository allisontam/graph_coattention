for fold in `seq 1 4`;
do
	python train.py DECAGON ./data/decagon/ --n_attention_head 8 --fold $fold/10 --n_epochs 30
	echo "DONE TRAINING $fold"
done
