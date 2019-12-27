for fold in `seq 1 4`;
do
	echo "START TESTING FOR $fold" >> preds.log
  python test.py DECAGON --settings default-cv_$(fold)_10.npy --memo default-cv_$(fold)_10.pth >> preds.log
	#python train.py DECAGON ./data/decagon/ --n_attention_head 8 --fold $fold/10 --n_epochs 30
	echo "DONE TESTING $fold"
done
