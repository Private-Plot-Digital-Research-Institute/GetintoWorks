
消息队列是一种在无服务器和微服务架构中使用的异步服务到服务通信的形式。它使应用程序能够通过将消息发送到彼此来相互通信。生产者创建消息并将其传递到消息队列中，而消费者连接到队列并检索消息以进行处理。消息队列有助于在应用程序和服务之间传输消息，促进诸如图像优化、数据处理和负载平衡等任务。

[Modelling distributed locking in TLA+ | Medium](https://medium.com/@polyglot_factotum/modelling-distributed-locking-in-tla-8a75dc441c5a)

### Zookeeper

一种实现分布式共识的方法

![[zookeeper.pdf]]

ZooKeeper seems to be Chubby without the lock methods, open, and close.
为什么选择无锁设计 -错误的、连接速度慢的用户会干扰速度快的用户
![[Pasted image 20230917201346.png]]## 设计目的

1.最终一致性：client 不论连接到哪个 Server，展示给它都是同一个视图，这是 zookeeper 最重要的性能。

2 .可靠性：具有简单、健壮、良好的性能，如果消息 m 被到一台服务器接受，那么它 将被所有的服务器接受。

3 .实时性：Zookeeper 保证客户端将在一个时间间隔范围内获得服务器的更新信息，或 者服务器失效的信息。但由于网络延时等原因，Zookeeper 不能保证两个客户端能同时得到 刚更新的数据，如果需要最新数据，应该在读数据之前调用 sync()接口。

4 .等待无关（wait-free）：慢的或者失效的 client 不得干预快速的 client 的请求，使得每 个 client 都能有效的等待。

5.原子性：更新只能成功或者失败，没有中间状态。

6 .顺序性：包括全局有序和偏序两种：全局有序是指如果在一台服务器上消息 a 在消息 b 前发布，则在所有 Server 上消息 a 都将在消息 b 前被发布；偏序是指如果一个消息 b 在消 息 a 后被同一个发送者发布，a 必将排在 b 前面。

[ZooKeeper 详解 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/72902467)