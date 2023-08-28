## 指针

  

### 智能指针

```cpp

std::unique_ptr<T> foo(new xxx);

  

std::shared_ptr<T> bar;

```

  

1. 智能指针离开本身的作用域就会自动释放，如果引入了其他内存管理机制则很容易出现重复释放

  

2. unique_ptr可以用get方法获取一份指针在别处使用, 但是unique_ptr对象本身不能传入函数中作为参数

  

3. 可以先创建对象再创建unique_ptr并传入对象指针，此时unique_ptr会管理该对象内存释放

  

### const指针

  

常量指针

```c

const int* a;

```

  

不能通过指针改变变量的值

  

```c

int* const a;

```

  

不能改变指向

  

const在星号左边就是不能改值，在右边就是不能改指向

  

```c

const int** a;

```

  

虽然不能通过指针改变变量的值，但可以解引用后再赋值

例如如下操作

  

```cpp

*a = new int(10);

```

可以将*a正确地赋值为堆上存储int(10)的地址

  
  

### 性能

  

1. string +=

  

2. const string&

  

3. emplace_back绝大多数情况下都可以替换push_back ，唯一改动是该函数会在vector->back()处原地构造一个对象.

  

不能用的场景：

auto &it = ivec.back();

ivec.emplace_back(it)

vector重新分配内存，导致it失效，传入了一个无法获取的值

  

3. 使用初始化列表可以少调用一次默认构造函数

  

例如，如果B拥有A的对象作为成员变量，用带有A1的参数作为构造函数时，会先调用一次A的默认构造函数，初始化A对象，执行完毕后再去执行B的构造函数，定义初始化列表就不需要先在B中初始化A对象

  

## 类型转换

  

1. const_cast<> 可以转常量 依据的原理是C++的指针可以任意转换

非常量转常量只用加一个const
  

2. 一般使用static_cast<>

  

3. dynamic_cast<> 可以将对象转为不同类

  

4. std::to_string()可以做到将任意基本类型转化为字符串

  

5. `constexpr` 指定编译器将变量或者函数或者成员变量优化为常数（无法优化为常数的则不优化）

  
  

6. `volatile` 阻止编译器将不会改动的值优化为常数，与constexpr相反

  
  

#### std::bind

  

std::bind 可以将成员非静态函数绑定到函数指针上，也可以指定函数参数。std::bind可以嵌套

  

成员非静态函数默认第一个参数为类的对象。可以使用std::placeholder::_1 , _2 填充参数

  

例如

``` cpp

auto foo = std::bind(std::max, std::placeholder::_1, 10)

```

就可以使用foo(1)来比较1与10的大小

  

### std::move

  

1. 将左值引用转化为右值

  

例如左值引用 `int &&i`与右值引用`int k`

  

有`i = std::move(k)`

  

2. 转移使用对象所有权

例如修改`std::unque_ptr`以转移形参所有权到实参中

```cpp
 void fun(std::unique_ptr u2)

{

}


unique_ptr<cls> u1;

fun(std::move(u1));
```


  
```cpp
int i;

int&& j = i++;

int&& k = ++i;

int& m = i++;

int& l = ++i;

  

move.cpp: In function ‘int main()’:

move.cpp:72:14: error: cannot bind ‘int’ lvalue to ‘int&&’

  int&& k = ++i;

              ^

move.cpp:73:15: error: invalid initialization of non-const reference of type ‘int&’ from an rvalue of type ‘int’

     int& m = i++;

```

  
  

## Lambda表达式

  

一共5个部分

 1   2   3      4          5  6

[=] () mutable throw() -> int {}

  

1. 传进lambda的外部参数，可以用=或者&，或者具体参数

  

2. lambda参数，可选

  

3. mutable specification Optional.

  

4. c++异常类参数

  

5. 返回值，可选

  

6. 函数体

  
  
  

## 格式化输出

  

%8s 固定输出行宽8的字符串


cout使用std::setw(8)

  
  

## 宏

  

#### 字符串

  

1. 将宏参数替换为字符串值: 在宏定义内部的变量中加#

  

```c

  

#define WARN_IF(EXP) \

do { if (EXP) \

        fprintf (stderr, "Warning: " #EXP "\n"); } \

while (0)

WARN_IF (x == 0);

     → do { if (x == 0)

           fprintf (stderr, "Warning: " "x == 0" "\n"); } while (0);

```

  

2. 字符串拼接： 将宏参数与宏定义中的字符串拼接

  

```c

struct command

{

  char *name;

  void (*function) (void);

};

  

struct command commands[] =

{

  { "quit", quit_command },

  { "help", help_command },

  …

};

  

```

等价于

  

```c

#define COMMAND(NAME)  { #NAME, NAME ## _command }

  

struct command commands[] =

{

  COMMAND (quit),

  COMMAND (help),

  …

};

```

  

  
  

  

