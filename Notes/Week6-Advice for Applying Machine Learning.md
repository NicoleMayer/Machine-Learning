---
title: Week6-Advice for Applying Machine Learning
date: 2020-07-23 23:26
---

# Evaluating a Learning Algorithm
Debugging a learning algorithm 可能尝试的方向：
![](./_image/2020-07/2020-07-23-23-27-24.png?r=59)

下面这个问题就概括了Andrew想说的话：
![](./_image/2020-07/2020-07-23-23-32-27.png?r=61)

* split up the data into two sets: a**training set**and a**test set**
* Typically, the training set consists of 70 % of your data and the test set is the remaining 30%.

下面是DL过程的新方法：
![](./_image/2020-07/2020-07-23-23-52-40.png)

前面说的 test set error 该怎么算呢？
![](./_image/2020-07/2020-07-23-23-51-23.png?r=72)

- - - - - 
模型选择问题：
![](./_image/2020-07/2020-07-24-00-06-39.png?r=67)

假如通过这10个结果选择了d=5，这其实还不是最优的，存在一些问题（因为testing data还是我们拿来做拟合的数据）这时候就引入了交叉验证。

交叉验证数据集怎么分？
![](./_image/2020-07/2020-07-24-00-33-07.png)

Error表示基本一致：
![](./_image/2020-07/2020-07-24-00-11-10.png?r=63)

采用交叉验证的 Model Selection：
![](./_image/2020-07/2020-07-24-00-31-32.png?r=65)

不选择testing中error最小的，而是选择cross validation中error最小的！！！
⬇️
老师讲了一大串，其实是为了说明这个：
![](./_image/2020-07/2020-07-24-00-28-37.png?r=72)

# Bias vs. Variance

如何判断是bias还是variance？
![](./_image/2020-07/2020-07-24-06-54-48.png?r=65)

如果加了正则化呢？上面的图会发生怎么样的变化？
（注意这里图中J的公式累加上面应该改为n！！！）
![](./_image/2020-07/2020-07-24-07-08-49.png?r=70)
![](./_image/2020-07/2020-07-24-07-04-58.png?r=67)

具体操作步骤：
![](./_image/2020-07/2020-07-24-07-07-16.png?r=78)

画Learning Curves是一个很好的理解训练状况的方式
![](./_image/2020-07/2020-07-24-08-05-09.png?r=70)

下面来分析分析两种case：
![](./_image/2020-07/2020-07-24-08-08-26.png?r=72)
![](./_image/2020-07/2020-07-24-08-11-11.png?r=74)

经过我们的分析，回到之前的一些debug方法上，我们现在有更好的决断了。
![](./_image/2020-07/2020-07-24-08-18-24.png?r=75)

Neural networks and overfitting
![](./_image/2020-07/2020-07-24-08-20-18.png?r=77)
接上图：
![](./_image/2020-07/2020-07-24-08-25-59.png?r=79)
一次都没选对！请记住：差太多一定是variance！！
![](./_image/2020-07/2020-07-24-08-22-55.png?r=68)

- - - - - 
画风突变！另一个主题！

# Building a Spam Classifier
一些想法
![](./_image/2020-07/2020-07-24-10-04-44.png?r=67)
根据这个想法的问题
![](./_image/2020-07/2020-07-24-10-03-37.png?r=73)

非常真实的场景（尽管AI很玄学，但我们还是得理性分析！！找到最合适的part！！）
![](./_image/2020-07/2020-07-24-10-05-49.png)

下面引出了错误分析这一方法：
![](./_image/2020-07/2020-07-24-10-08-36.png?r=78)

下面是一个error analysis的例子：
![](./_image/2020-07/2020-07-24-10-15-53.png?r=72)

如果只用error analysis是不够的，我们需要numerical evaluation
![](./_image/2020-07/2020-07-24-10-20-11.png?r=76)

太真实了！
![](./_image/2020-07/2020-07-24-10-21-57.png)


# Handling Skewed Data
![](./_image/2020-07/2020-07-24-10-30-35.png?r=80)

通常把少的那部分当作positive！

![](./_image/2020-07/2020-07-24-10-42-39.png?r=80)

![](./_image/2020-07/2020-07-24-10-41-28.png)

加油加油！！
![](./_image/2020-07/2020-07-24-10-46-10.png?r=75)

# Using Large Data Sets

哈哈哈，我们常听到的数据为王，这就是现实。
![](./_image/2020-07/2020-07-24-10-50-10.png?r=71)
很简单，如果人类专家通过给定的input，也无法解决问题；那么再多的数据也无济于事！
enough information很重要，不仅仅是数据的量，还有数据的维度！

单元检测
![](./_image/2020-07/2020-07-24-11-05-41.png?r=65)


