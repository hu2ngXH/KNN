#coding=utf-8
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
import KNN

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOFLines = len(arrayOLines) #得到文件行数
    returnMat = zeros((numberOFLines,3))#得到numberOFLines这么多个3维的数组的元组 创建返回的numpy矩阵
    ClassLabelVector = [] #可以这样定义一个数组
    index = 0
    for line in arrayOLines :
        line = line.strip() #用于移除字符串头尾指定的字符（默认为空格） 我们取回来的数是字符串形式的
        listFormLine = line.split('\t') #因为原来的数据读行之后一行的数据是连接在一起的以'\t'为分隔，需要将其分开
        returnMat[index,:] = listFormLine[0:3] #解析文件数据到列表到这一步就结束了
        ClassLabelVector.append(int(listFormLine[-1])) #存储其标签
        index += 1
    return returnMat,ClassLabelVector

datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')

'''
#画图像
fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
这个是没有分类的图像

ax.scatter(datingDataMat[:,0],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
plt.show()
'''

#因为数据的来源不同 所以使用原本数据难免会发生某一特征占很大比重的情况，因此我们需要归一化处理
#多维数组每一维都做了同样的操作
def autoNorm(dataSet):
    minvals = dataSet.min(0) #取数组中最小的值
    maxvals = dataSet.max(0) #取数组中最大的值
    ranges = maxvals - minvals #求出范围
    #minvals maxvals ranges都为三维列表
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minvals,(m,1))
    normDataSet = normDataSet / tile(ranges,(m,1))
    return normDataSet,ranges,minvals

normMat,ranges,minVals = autoNorm(datingDataMat)
print normMat
print ranges
print minVals

#测试模块 学习如何进行测试
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt') #解析文件数据
    normMat, ranges, minVals = autoNorm(datingDataMat)#数据归一化处理
    m = normMat.shape[0] #得到数组的长度
    numTestVecs = int(m * hoRatio) #取其十分之一的数据来测试是否可行
    errorCount = 0.0#初始化错误次数
    for i in range(numTestVecs):#
        classifierResualt = KNN.classify0(normMat[i,:] , normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with :%d , the real answer is : %d " %(classifierResualt,datingLabels[i])
        if(classifierResualt != datingLabels[i]): errorCount += 1.0
    print "the total error rate is : %f" %(errorCount / float(numTestVecs))
    return numTestVecs


#测试模块效果不错 可以用于约会网站预测
def classifyPerson():
    resualtList = ['not at all','in some doses','in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffmiles = float(raw_input("frequent flier miles earned per year?"))
    icecream = float(raw_input("liters  of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # 解析文件数据
    normMat, ranges, minVals = autoNorm(datingDataMat)  # 数据归一化处理
    inArr = array([ffmiles,percentTats,icecream])
    classifyresult = KNN.classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print "You will probably like this person:",resualtList[classifyresult - 1]




