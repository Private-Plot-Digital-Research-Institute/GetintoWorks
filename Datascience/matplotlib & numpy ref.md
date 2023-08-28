



## numpy 随机数生成

### numpy.random

- numpy.random.randint(low, high=None, size=None, dtype='l')
函数的作用是，返回一个随机整型数，范围从低（包括）到高（不包括），即[low, high)。
如果没有写参数high的值，则返回[0,low)的值。

- np.random.randn(d0, d1, ..., dn)
返回服从标准正态分布的N维张量

**randn 才是标准正态分布**

```python

>>> np.random.randn(2,2)
array([[0.10811758, 0.45357553],
       [0.35510466, 0.41470196]])
```

**如果要返回指定正态分布的张量，可以利用x = (x`-u)/σ**


#numpy.random.choice(a, size=None, replace=True, p=None)
#从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
#replace:True表示可以取相同数字，False表示不可以取相同数字
#数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。

### sklearn.utils.shuffle

打乱列表，如果有多个列表则按照同一步调打乱


b=np.argmax(a)#取出a中元素最大值所对应的索引，此时最大值位6，其对应的位置索引值为4，（索引值默认从0开始）


### ZIP打包元组


实例
以下实例展示了 zip 的使用方法：

```python
>>>a = [1,2,3]
>>> b = [4,5,6]
>>> c = [4,5,6,7,8]
>>> zipped = zip(a,b)     # 打包为元组的列表
[(1, 4), (2, 5), (3, 6)]
>>> zip(a,c)              # 元素个数与最短的列表一致
[(1, 4), (2, 5), (3, 6)]
>>> zip(*zipped)          # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
[(1, 2, 3), (4, 5, 6)]
```


注意zip不属于list类型，数组操作需要用其他方法.
zip返回的是迭代器，不能用索引访问

```python
>>> a = [3, 9, 2, 24, 1, 6]
>>> b = ['a', 'b', 'c', 'd', 'e']
>>> sorted(zip(a, b))
[(1, 'e'), (2, 'c'), (3, 'a'), (9, 'b'), (24, 'd')]
>>> sorted(zip(a, b), key=lambda x: x[1])
[(3, 'a'), (9, 'b'), (2, 'c'), (24, 'd'), (1, 'e')]
```


zip可以直接转list


np.arange(a) 返回0~a-1的，长度为a的数组

np.dot 就是矩阵乘法


## numpy 列表转换

### numpy list

<https://blog.csdn.net/Liuseii/article/details/80023733>


import numpy as np

List转numpy.array:

temp = np.array(list)

numpy.array转List:

arr = temp.tolist()

### python list

```python

#基于seed产生随机数
rdm = np.random.RandomState(SEED)
#随机数返回32行2列的矩阵 表示32组 体积和重量 作为输入数据集
X = rdm.rand(32,2)
#从X这个32行2列的矩阵中 取出一行 判断如果和小于1 给Y赋值1 如果和不小于1 给Y赋值0 
#作为输入数据集的标签（正确答案） 
Y_ = [[int(x0 + x1 < 1)] for (x0, x1) in X]

```

### numpy reshape

x.reshape(xxx) 返回值为长度xxx的ndarray，原变量值不变

x.flatten() 返回值为展平后的ndarray,原变量值不变

总个数用x.size

在numpy中，进行1*120 矩阵与 120*3 矩阵乘算，得到的结果是1*3大小而不是3大小

对于 x.shape = (10,2,3), 使用

```python
x = [xi.reshape(6) for xi in x]
```

效果与

```python
x = np.reshape(10,6)
```

几乎完全一致，不同之处在于前者是一个List

### numpy max

2.np.maximum(X, Y, out=None)

 X和Y逐位进行比较,选择最大值.
最少接受两个参数

ex:

>> np.maximum([-3, -2, 0, 1, 2], 0)
array([0, 0, 0, 1, 2])

### numpy flatten square

```python

>>> import numpy as np
>>> a = np.random.rand(2,2)
>>> print(a)
[[0.9575825  0.37387049]
 [0.56442419 0.17732237]]
>>> b = a.flatten()
>>> print(a)
[[0.9575825  0.37387049]
 [0.56442419 0.17732237]]
>>> print(b)
[0.9575825  0.37387049 0.56442419 0.17732237]
>>> print(b ** 2)
[0.91696424 0.13977914 0.31857467 0.03144322]
>>> print(np.sum(b))
2.0731995504985683
>>> print(np.sum(b**2))
1.4067612727582957

```

### numpy连接

arr.shape = (1024,1)
ins = np.concatenate((arr,arr))
注意两层括号
ins.shape = (1024,2)



### np.sum

axis的参数是哪个轴，就将哪个轴压缩
可以传一个元组进去，同时压缩几个轴

```python
>>> x
array([[[1, 2],
        [3, 4]],

       [[5, 6],
        [7, 8]]])
>>> x.shape
(2, 2, 2)
>>> x[0]
array([[1, 2],
       [3, 4]])
>>> np.sum(x,axis=0)
array([[ 6,  8],
       [10, 12]])
```

```python
>>> y = [[[1,2,3,4],[5,6,7,8]]]
>>> y = np.asarray(y)
>>> y.shape
(1, 2, 4)
>>> np.sum(y,0)
array([[1, 2, 3, 4],
       [5, 6, 7, 8]])
>>> np.sum(y,1)
array([[ 6,  8, 10, 12]])
>>> np.sum(y,2)
array([[10, 26]])
```

### np.transpose

对（4，）之类的数组没用

## Word Embedding

### np.add.at

### indexing



## Matplotlib 2D坐标可视化


### numpy linspace
api

```python
numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
```

用法:

```python
>>> np.linspace(-2.0,2.0,10)
array([-2.        , -1.55555556, -1.11111111, -0.66666667, -0.22222222,
        0.22222222,  0.66666667,  1.11111111,  1.55555556,  2.        ])
```

<https://blog.csdn.net/Asher117/article/details/87855493>

### numpy meshgrid

快速生成坐标矩阵

```python
>>> X = [1,2,3,4,5]
>>> Y = [1,2,3,4,5]
>>> X,Y = np.meshgrid(X,Y)
>>> print(X)[[1 2 3 4 5]
 [1 2 3 4 5]
 [1 2 3 4 5]
 [1 2 3 4 5]
 [1 2 3 4 5]]
>>> print(Y)
[[1 1 1 1 1]
 [2 2 2 2 2]
 [3 3 3 3 3]
 [4 4 4 4 4]
 [5 5 5 5 5]]
 ````

### numpy ismember

```python
np.in1d(array1, array2)
```
### plt.imshow

可以直接把np.array变成heatmap