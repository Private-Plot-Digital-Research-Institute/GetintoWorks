# FIO

https://fio.readthedocs.io/en/latest/fio_doc.html#i-o-engine

## 测试参数

### direct=bool

Linux下打开O_DIRECT，Windows不起效果（实际上就是使用/禁用page cache）

### blocksize=int[,int][,int], bs=int[,int][,int]¶

一次IO读取多少字节

`bssplit`可以混合不同的blocksize进行io

## 控制参数

### IOEngine

### 任务设置

```
name=str // 任务名
loops=int // 自动循环次数
startdelay=irange(time) // 开始任务前设置的时延
```

## Shell传参

传参方法：fio文件定义参数，然后输入命令的时候传入

```

$ SIZE=64m NUMJOBS=4 fio jobfile.fio
```

```

; -- start job file --
[random-writers]
rw=randwrite
size=${SIZE}
numjobs=${NUMJOBS}
; -- end job file --
```

This will expand to the following equivalent job file at runtime:

```
; -- start job file --
[random-writers]
rw=randwrite
size=64m
numjobs=4
; -- end job file --
```