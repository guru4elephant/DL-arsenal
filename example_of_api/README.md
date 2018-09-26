partition the program into several parts by hand designed checkpoints
get the max number of vars for each partition ( ignore all weight and weight parameter)
prepare a memory pool for forward pass in each partition
for each partition in forward:
    replace forward variable with vars in memory pool
    do not replace the checkpoint and weight 

prepare a memory pool for backward pass in each partition
for each partition in backward:
    prepare forward partition computation for backward by looking up checkpoints
    replace backward variable with vars in backward memory pool

