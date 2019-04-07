for b in 128
do
	 for t in {1..10}
	 do
	     rm elog.$b.$t
	     echo "batch: " $b, "thread num: " $t >> elog.$b.$t
	     python dataset_throughputs.py $b $t 2>> elog.$b.$t
	 done
done
