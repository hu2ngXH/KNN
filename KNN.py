#coding=utf-8
from numpy import *
import operator
#科学计算包
#运算符模块

def creatDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    #array函数的参数只有一个所以要用[]先括起来
    labels = ['A','A','B','B']
    return group,labels

#使用欧氏距离计算距离 需要在各个维度上相减然后求平方和最后开方
def classify0(inX,dataSet,labels,k):
    '''
        用于分类的输入向量：inX
        输入的训练样本集：dataSet
        标签向量：labels
        参数K用于选择最近邻居的数目
    '''
    dataSetSize = dataSet.shape[0] #此例中dataSetSize等于4
    #shape函数返回矩阵的【行数，列数】 shape[0]就代表行数
    #距离计算
    diffMat = tile(inX,(dataSetSize,1)) - dataSet #tile为矩阵扩展函数
    #tile(inX,(dataSetSize,1))这句话的意思就是复制为4个列表 每个列表中inX只复制一次
    #然后依次与原来训练集相减
    sqDiffMat = diffMat**2
    #平方
    sqDistanaces = sqDiffMat.sum(axis = 1)
    #axis=1代表行相加 axis=0代表列相加 每一个列表就像相当于是矩阵的一行
    distances = sqDistanaces**0.5
    #开方
    #距离计算结束 现在得到一个列向量 列向量记载着目标点与各个训练数据的点的距离 欧氏距离表示
    sortedDistIndicies = distances.argsort()
    #排序 根据数组中的值的排序 返回其索引值数组 其数组的值为下标
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        #将K范围内的邻居的类别都记录下来
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        #将字典中的类别数加一
    sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse = True)
    #sorted返回的是一个新的list 第一个参数是排序体，因为classCount是一个字典所以需要返回其列表 iteritem是一个迭代器
    #key是用来进行比较的元素，operator.itemgetter(1)代表去第一个域即第二个值来比较，就是该字典建对应的value
    #反转标志，false代表不反转从小到大排排序，true代表反转从大到小
    return sortedClassCount[0][0]
    #返回该类别而不是类别的数目

#书上的做法
'''
group,labels = creatDataSet()
resualt = classify0([0,0],group,labels,3)
print resualt

#手动输入数据
Datatraining = [0,0]
d1 = raw_input('enter the first featrue:')
d2 = raw_input('enter the second featrue:')
Datatraining[0] = int(d1)
Datatraining[1] = int(d2)
group,labels = creatDataSet()
resualt = classify0(Datatraining,group,labels,3)
print resualt
'''