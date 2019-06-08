from math import log
import operator

import matplotlib.pyplot as plt

# --------建树部分--------

def createDataSet():
    """
    创建数据集
    有9个贷款申请的样本，每个样本分为四个特征（年龄、有工作、有房子、信贷情况）和一个类（银行是否贷款给他）
    返回数据集和四个特征的标题
    """
    dataSet = [
        ['Young',      'False', 'False', 'NotBad',    'Rejected'],
        ['Young',      'False', 'False', 'Good',      'Rejected'],
        ['Young',      'True',  'False', 'Good',      'Accepted'],
        ['MiddleAged', 'False', 'False', 'NotBad',    'Rejected'],
        ['MiddleAged', 'False', 'False', 'Good',      'Rejected'],
        ['MiddleAged', 'False', 'True',  'Excellent', 'Accepted'],
        ['Old',        'False', 'True',  'Excellent', 'Accepted'],
        ['Old',        'True',  'False', 'Excellent', 'Accepted'],
        ['Old',        'False', 'False', 'NotBad',    'Rejected'],
    ]
    labels = ['Age', 'In work', 'Property', 'Credit situation']
    return dataSet, labels

def calcShannonEnt(dataSet, axis = -1):
    """
    计算给定数据集的熵
    axis: 给定特征的索引，用于计算取每个值的概率。默认为-1（样本最后一项，即分类）
    """
    numEntries = len(dataSet)
    labelCounts = {}  # 对数据集中每个类别进行计数
    for featVec in dataSet:
        currentLabel = featVec[axis]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # 计算以2为底的对数
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    """
    按照给定特征划分数据集
    axis: 给定特征的索引
    value: 该特征的给定取值
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])  # 将axis对应的特征去掉
            retDataSet.append(reducedFeatVec)  # 加入新列表
    return retDataSet

def chooseBestFeatureToSplitUsingID3(dataSet):
    """
    通过计算信息增益选择最好的数据集划分方式（用于ID3算法），返回最优特征索引
    """
    numFeatures = len(dataSet[0]) - 1  # 特征个数
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0  # 存储最高的信息增益
    bestFeature = -1  # 存储最优的特征索引
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  # 获取数据集中第i个特征的所有取值
        uniqueVals = set(featList)  # 去重
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)  # 将数据集中第i个特征值为value的所有样本划分出来
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  # 计算用第i个特征划分的信息增益
        if infoGain > bestInfoGain:  # 记录最好的信息增益
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def chooseBestFeatureToSplitUsingC4Dot5(dataSet):
    """
    通过计算信息增益选择最好的数据集划分方式（用于C4.5算法），返回最优特征索引
    """
    numFeatures = len(dataSet[0]) - 1  # 特征个数
    baseEntropy = calcShannonEnt(dataSet)
    infoGains = []
    infoGainRatios = []
    aveInfoGain = 0.0  # 存储平均信息增益
    bestInfoGainRatio = 0.0  # 存储最高的信息增益比
    bestFeature = -1  # 存储最优的特征索引
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  # 获取数据集中第i个特征的所有取值
        uniqueVals = set(featList)  # 去重
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)  # 将数据集中第i个特征值为value的所有样本划分出来
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  # 计算用第i个特征划分的信息增益
        infoGains.append(infoGain)
        aveInfoGain += infoGain
        infoGainRatio = 0.0
        hA = calcShannonEnt(dataSet, i)
        if hA != 0:
            infoGainRatio = infoGain / hA  # 计算信息增益比
        infoGainRatios.append(infoGainRatio)
    aveInfoGain /= numFeatures
    for i in range(numFeatures):
        if infoGains[i] >= aveInfoGain and infoGainRatios[i] > bestInfoGainRatio:  # 在信息增益大于平均值的样本中记录最好的信息增益比
            bestInfoGainRatio = infoGainRatios[i]
            bestFeature = i
    return bestFeature

def createTreeUsingID3(dataSet, labels):
    """
    用ID3算法创建决策树
    dataSet: 数据集
    labels: 特征标题的列表
    """
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):  # 如果数据集只剩下一种类别则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:  # 如果特征已被划分殆尽，但数据集中还有多种类别，则输出出现次数最多的
        classCount = {}  # 对数据集中每个类别进行计数
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.iteritems(), \
        key = operator.itemgetter(1), reverse = True)  # 根据出现次数，对类别排倒序
        return sortedClassCount[0][0]
    # 特殊情况都已排除
    bestFeat = chooseBestFeatureToSplitUsingID3(dataSet)  # 通过信息增益计算最优特征索引
    bestFeatLabel = labels[bestFeat]  # 最优特征的标题
    myTree = {bestFeatLabel:{}}  # 定义树结构
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]  # 获取数据集中最优特征的所有取值
    uniqueVals = set(featValues)  # 去重
    for value in uniqueVals:  # 根据每个取值来递归建树
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = \
        createTreeUsingID3(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

def createTreeUsingC4Dot5(dataSet, labels):
    """
    用C4.5算法创建决策树
    dataSet: 数据集
    labels: 特征标题的列表
    """
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):  # 如果数据集只剩下一种类别则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:  # 如果特征已被划分殆尽，但数据集中还有多种类别，则输出出现次数最多的
        classCount = {}  # 对数据集中每个类别进行计数
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.iteritems(), \
        key = operator.itemgetter(1), reverse = True)  # 根据出现次数，对类别排倒序
        return sortedClassCount[0][0]
    # 特殊情况都已排除
    bestFeat = chooseBestFeatureToSplitUsingC4Dot5(dataSet)  # 通过信息增益计算最优特征索引
    bestFeatLabel = labels[bestFeat]  # 最优特征的标题
    myTree = {bestFeatLabel:{}}  # 定义树结构
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]  # 获取数据集中最优特征的所有取值
    uniqueVals = set(featValues)  # 去重
    for value in uniqueVals:  # 根据每个取值来递归建树
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = \
        createTreeUsingC4Dot5(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

# --------绘图部分--------

# 定义节点和箭头格式的常量
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def getNumLeafs(myTree):
    """
    获取叶结点的数目
    """
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':  # 测试结点的数据类型是否为字典
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    """
    获取树的深度
    """
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':  # 测试结点的数据类型是否为字典
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

def plotMidTest(cntrPt, parentPt,txtString):
    """
    在父子结点间填充文本信息
    """
    xMid = (parentPt[0] + cntrPt[0])/2.0
    yMid = (parentPt[1] + cntrPt[1])/2.0
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    """
    绘制树形图
    若当前子节点不是叶子节点则递归，若当子节点为叶子节点，则绘制该节点
    """
    numLeafs = getNumLeafs(myTree)  # 计算宽度
    # depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xoff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yoff)
    plotMidTest(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yoff = plotTree.yoff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xoff = plotTree.xoff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xoff, plotTree.yoff), cntrPt, leafNode)
            plotMidTest((plotTree.xoff, plotTree.yoff), cntrPt, str(key))
    plotTree.yoff = plotTree.yoff + 1.0/plotTree.totalD

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    绘制结点的模板
    """
    createPlot.ax1.annotate(nodeTxt,  # 注释的文字，（一个字符串）
                            xy=parentPt,  # 被注释的地方（一个坐标）
                            xycoords='axes fraction',  # xy所用的坐标系
                            xytext=centerPt,  # 插入文本的地方（一个坐标）
                            textcoords='axes fraction',  # xytext所用的坐标系
                            va="center",
                            ha="center",
                            bbox=nodeType,  # 注释文字用的框的格式
                            arrowprops=arrow_args)  # 箭头属性

def createPlot(inTree):
    """
    绘制图形
    """
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111,frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xoff = -0.5/plotTree.totalW
    plotTree.yoff = 1.0

    plotTree(inTree, (0.5, 1.0),'')  # 树的引用作为父节点，但不画出来，所以用''
    plt.show()

dataSet, labels = createDataSet()
tree = createTreeUsingID3(dataSet, labels)  # 通过ID3算法建树
# tree = createTreeUsingC4Dot5(dataSet, labels)  # 通过C4.5算法建树
print(tree)
createPlot(tree)