0518分享

[google-research/tuning_playbook: A playbook for systematically maximizing the performance of deep learning models. (github.com)](https://github.com/google-research/tuning_playbook)

### Dataset

![[Pasted image 20231013110331.png]]

self.data用列表会发生内存泄漏（子进程复制父进程的内存），要用Pandas, Numpy, PyArrow等

![[Pasted image 20231013110546.png]]
![[Pasted image 20231013110603.png]]

![[Pasted image 20231013110632.png]]

### 内存

![[Pasted image 20231013110705.png]]

![[Pasted image 20231013110723.png]]
1. 最理想的batch_size是最大的batch_size, 不应作为超参数
2. 大部分超参数的理想值对batch_size敏感
3. 

![[Pasted image 20231013111412.png]]

![[Pasted image 20231013111556.png]]

![[Pasted image 20231013111838.png]]

![[Pasted image 20231013111847.png]]

![[Pasted image 20231013111858.png]]

![[Pasted image 20231013111910.png]]

![[Pasted image 20231013111929.png]]

![[Pasted image 20231013111950.png]]

![[Pasted image 20231013112020.png]]
![[Pasted image 20231013114300.png]]

AutoHPO: [**Automatic Hyper-Parameter Optimization**](https://arxiv.org/abs/2003.01751)
![[Pasted image 20231013114313.png]]

![[Pasted image 20231013114509.png]]
![[Pasted image 20231013114603.png]]
![[Pasted image 20231013114641.png]]
![[Pasted image 20231013114658.png]]
![[Pasted image 20231013114719.png]]
![[Pasted image 20231013114753.png]]

前向
1. 把不同batch塞进GPU里
2. 对每个GPU复制一份模型
3. 并行运行
4. 在GPU-1处合并
反向
1. GPU-1计算所有梯度
2. 把不同的梯度发到其他GPU
3. 反向
4. 将所有梯度归约到GPU-1
![[Pasted image 20231013115550.png]]
![[Pasted image 20231013115609.png]]
![[Pasted image 20231013115621.png]]
![[Pasted image 20231013115634.png]]
![[Pasted image 20231013115702.png]]