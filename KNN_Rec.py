#coding=utf-8
#手写数字识别
from numpy import *
import os,sys
import KNN
#我们将32*32的二进制矩阵转换为1*1024的向量
#二维矩阵存储为一维矩阵的方式
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

#手写数字识别系统的测试代码
def handwritingClassTest():
    hwLabels = [] #s手写数字的标签
    trainingFileList = os.listdir('trainingDigits') #文件夹中的文件名 获取目录的内容
    m = len(trainingFileList)#统计一共有多少个训练
    trainingMat = zeros((m,1024))
    #从文件名解析分类数字 开始
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0] #用'.'号分隔然后取第一个元素
        classNumStr = int(fileStr.split('_')[0]) #用'_'分隔然后取第一个元素
        hwLabels.append(classNumStr) #将数字标签存入数组
        trainingMat[i,:] = img2vector('trainingDigits/%s' %fileNameStr)
    testFileList = os.listdir('testDigits')#获得测试数据
    errorCount = 0.0
    mTest = len(testFileList)#一共有多少测试数据
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' %fileNameStr)
        classifierResualt = KNN.classify0(vectorUnderTest,trainingMat,hwLabels,30)
        print "the classsifier came back with :%d , the real answer is %d"%(classifierResualt,classNumStr)
        if (classifierResualt != classNumStr): errorCount += 1.0
    print "\nthe totle number of error is : %d" %errorCount
    print "\nthe totle error rate is : %d" %(errorCount/float(mTest))