

### protobuf

  

#### 传递格式

  

如果使用DebugString() 输出， 一定要使用Textformat而不能使用ParseFromString(),否则解析不出

  

Message序列化-反序列化使用Message::SerializeToString - Message::ParseFromString()

  
  

### rocksdb

  

基本操作：

  

DB::Open DB::Read DB::Put

  
  
  

#### column family handle

  

https://github.com/facebook/rocksdb/blob/master/examples/column_families_example.cc

  

```cpp

  Status s = DB::Open(options, kDBPath, &db);

  assert(s.ok());

  

  // create column family

  ColumnFamilyHandle* cf;

  s = db->CreateColumnFamily(ColumnFamilyOptions(), "new_cf", &cf);

  assert(s.ok());

```

  
  

#### 写入顺序

  

rocksdb的数据大致遵循如下序列

  

MemTable -> WAL -> SSTFile

  

#### Write Stall

  

### gtest

  

TEST(Testclass, Testname)

  

初始化使用

  

```cpp

：：testing：：InitGoogleTest(&argc, argv);

```

注意如果要使用可执行文件，运行路径要设置在.runfile下

  

#### gmock

  

### glog

  

InitGoogleLogging();

  

LOG(LEVEL)

  

LEVEL可以是 INFO WARNING ERROR FATAL

  

VLOG(NUM)

  

指定LOG级别

  

--logtostderr

  

### gflags

  

google::ParseCommandLineFlags(&argc, &argv, true);

  

### git

  

git stash

  

git patch

  

### ClickHouse

  

### pulsar

  

### jemalloc/tcmalloc

  

通过直接链接就可以替换掉默认的malloc
