for encoder in 'bow' 'cnn' 'gru' 'lstm'
do 
    for mode in 'QueueDataset' 'InMemoryDataset'
    do
	echo "encoder: " $encoder
	echo "dataset mode: " $mode
	python local_train_benchmark.py --text_encoder $encoder --dataset_mode $mode --model_output_dir $encoder" "$mode" model" --num_passes 5
    done
done
