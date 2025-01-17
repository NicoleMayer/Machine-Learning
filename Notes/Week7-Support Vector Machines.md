---
title: Week7-Support Vector Machines
date: 2020-08-02 09:48
---

# Large Margin Classification
![](./_image/2020-08/2020-08-02-09-59-21.png)

吴恩达讲得真的好。之前学习SVM从来没和逻辑回归的公式比较过。这种比较归纳的思维看问题会理解的更深入！
![](./_image/2020-08/2020-08-02-10-03-39.png)

![](./_image/2020-08/2020-08-02-10-05-20.png)

- - - - - 
可见SVM要求更高，不仅仅以0为分割线，而是要有两条分割线
![](./_image/2020-08/2020-08-02-10-09-17.png)

这一小章节的标题叫 Large Margin Classification，它是神马意思呢？大间距分类器！

举个例子，假如C无穷大，为了最小化这一方程，sum那部分就会为0，然后我们得到右下角的优化方程。s.t.部分是如果想要sum部分为0需要有什么条件。
![](./_image/2020-08/2020-08-02-10-35-56.png)
根据上图的优化方程，我们能得出下面这个图：
![](./_image/2020-08/2020-08-02-10-32-12.png?r=56)

如果C很大，那么会拟合的比较好，像粉红色的分割线；如果没那么大，就像黑色的线。
![](./_image/2020-08/2020-08-02-10-42-27.png?r=67)

- - - - - 
下面的内容偏数学公式推导

先讲了讲 vector inner product，感觉也很简单。。都是些知道的东西，还是为了说明为什么是大间距分类器！
![](./_image/2020-08/2020-08-02-10-58-49.png)

# Kernels
![](./_image/2020-08/2020-08-02-11-25-04.png)
![](./_image/2020-08/2020-08-02-11-26-56.png)
![](./_image/2020-08/2020-08-02-11-28-24.png)
![](./_image/2020-08/2020-08-02-11-31-38.png)
如何应用？使用kernel去预测label：
![](./_image/2020-08/2020-08-02-11-33-47.png)
- - - - - 
如何选择landmarks呢？可以根据dataset选择：
![](./_image/2020-08/2020-08-02-11-35-44.png)
![](./_image/2020-08/2020-08-02-11-38-09.png)
![](./_image/2020-08/2020-08-02-11-41-17.png)
上图中下面那堆奇怪的东西是想说implementation需要修正的地方，theta被正则化，M表示间距的大小，下面是具体的翻译：
![](./_image/2020-08/2020-08-02-11-49-11.png?r=75)

参数如何选择？
![](./_image/2020-08/2020-08-02-11-42-41.png)

# SVMs in Practice
![](./_image/2020-08/2020-08-02-12-01-27.png)
![](./_image/2020-08/2020-08-02-12-06-06.png)
![](./_image/2020-08/2020-08-02-12-08-32.png)

差点就选D了。。
![](./_image/2020-08/2020-08-02-12-09-07.png)

![](./_image/2020-08/2020-08-02-12-10-17.png)
如何选择？
![](./_image/2020-08/2020-08-02-12-12-26.png)





