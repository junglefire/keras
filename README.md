### Deep Learning with Python代码示例
通过指定`KERAS_HOME`来设置存储数据集的目录。每个例子的开头都有一行：
```python
## 设置KERAS根目录，所有的数据集都从这个目录加载
log.info("set ENV `KERAS_HOME` ...")
os.environ["KERAS_HOME"] = "./keras"
```

原始仓库参考：https://github.com/fchollet/deep-learning-with-python-notebooks


例子：
1. imdb.py     : 3.4节`电影评论分类：二分类问题`

2. reuters.py  : 3.5节`新闻分类：多分类问题`

3. boston.py   : 3.6节`波士顿房价预测`

4. mnist.py    : 5.1节`使用CNN训练模型识别MNIST`

5. image_v1.py : 5.2.1~5.2.4节，小规模图片训练
