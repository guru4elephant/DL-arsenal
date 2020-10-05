
``` python
import paddle
import paddle.distributed.fleet as fleet
import model

paddle.enable_static()
mnist_model = model.Mnist()

fleet.init()
strategy = fleet.DistributedStrategy()
strategy.a_sync = True

optimizer = paddle.optimizer.Adam(learning_rate=0.001)
optimizer = fleet.distributed_optimizer(optimizer, strategy)
optimizer.minimize(mnist_model.cost)

if fleet.is_server():
   fleet.init_server()
   fleet.run_server()
else:
   # your training code on workers
   fleet.stop_worker()
```



``` python
import paddle
import paddle.distributed.fleet as fleet
import model

paddle.enable_static()
mnist_model = model.Mnist()

fleet.init(is_collective=True)
strategy = fleet.DistributedStrategy()
strategy.amp = True
strategy.lamb = True

optimizer = paddle.optimizer.Adam(learning_rate=0.001)
optimizer = fleet.distributed_optimizer(optimizer, strategy)
optimizer.minimize(mnist_model.cost)

# your training code on each GPU




```
