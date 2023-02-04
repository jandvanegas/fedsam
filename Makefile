training:
	# run_fedavg 
	# run_fedavg_with_swa 
	# run_fedsam 
	# run_fedsam_with_swa 
	# run_fedasam 
	# run_fedasam_with_swa 
	./paper_experiments/cifar10.sh $(method) $(alpha)

all_at_once:
	make training method=run_fedavg
	make training method=run_fedavg_with_swa
	make training method=run_fedsam
	make training method=run_fedsam_with_swa
	make training method=run_fedasam
	make training method=run_fedasam_with_swa

train_fedmd:
	cd ./models; python main.py --num-rounds 10000 --eval-every 100 --batch-size 64 --num-epochs 1 --clients-per-round 5 -model destillation -lr 0.01 --weight-decay 0.0004 -device cuda:0 -algorithm fedmd --server-lr 1 --server-opt sgd --num-workers 0 --where-loading init -dataset cifar10 --publicdataset cifar100 -alpha $(alpha)
