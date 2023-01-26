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
