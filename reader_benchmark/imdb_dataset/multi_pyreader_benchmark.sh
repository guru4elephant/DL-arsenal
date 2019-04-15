for b in 128
do
    for i in {1..10}
    do
	export FLAGS_print_sub_graph_dir=pyreader_graph_$i
	echo "batch size= " $b, " thread num= " $i
	python multi_process_pyreader_throughputs.py $b $i 2> multi.pyreader.elog.$b.$i
    done
done
