# -*- coding: utf-8 -*-
#Level 1（每次放回SRS中没被选中的元素） HTRSS抽样均值估计以及方差估计
from __future__ import division
import matplotlib.pyplot as plt
from scipy.special import comb
import numpy as np   
import random 
from scipy.stats import *
import os

#from numpy import mean, ptp, var, std, median

#a=numpy.random.normal(60,20,N) #随机生成均值600,标准差20,正态分布的N个元素为总体a
#a=np.array([])
#蒙特卡洛法生成给定分布，峰度的数据



def I(i,r,m,N):
  t=comb(i-1,r-1)*comb(N-i,m-r)/comb(N,m) 
  return t

def U(j,k,i,m1,N):   #MRSS(m)下前k个被选中测量的元中包含j个比i小的元但不含i本身的概率
  if k==0:
    if j==0:  
      return 1
    else:
      return 0  
  else:
    if m1%2!=0:
      r=(m1+1)/2  #MRSS(m)下每次规模为m的SRS中取第r位元，r的计算方法
    else:
      if k%2!=0:
        r=m1/2
      else:
        r=m1/2+1 #MRSS(m)下若m为奇数，取r=(m+1)/2，若m位偶数，交替取第m/2和m/2+1位元  
    if j==0:  
      t1=0  
      for s in range(i+1,N-k+2):
        t1+=I(s,r,m1,N-k+1)
      t1=t1*U(0,k-1,i,m1,N)
      return t1
    else:
      t2,t3=0,0
      for s in range(i+1-j,N-k+2):
        t2+=I(s,r,m1,N-k+1)
      for s in range(1,i-j+1):
        t3+=I(s,r,m1,N-k+1)  
      t2=t2*U(j,k-1,i,m1,N)
      t3=t3*U(j-1,k-1,i,m1,N)
      t=t2+t3
      return t  
  
def MRSSInclusion1(m1,n,N):
  y=[0]*N  
  for i in range(1,N+1):  
    y1=0
    for j in range(0,i):
      y1+=U(j,n,i,m1,N)
    y1=1-y1 
    y[i-1]=y1   #分别计算MRSS（m）下元素1,...,N的包含概率，并存入数组y
  return y
  
def V(j,k,i,m,N):  #RSS(m)下前k个被选中测量的元中包含j个比i小的元但不含i本身的概率
  if k==0:
    if j==0:  
      return 1
    else:
      return 0  
  else:
    r=(k+2)%m+1   #第k次SRS样本选取第r小的元素
    if j==0:  
      t1=0  
      for s in range(i+1,N-k+2):
        t1+=I(s,r,m,N-k+1)
      t1=t1*V(0,k-1,i,m,N)
      return t1
    else:
      t2,t3=0,0
      for s in range(i+1-j,N-k+2):
        t2+=I(s,r,m,N-k+1)
      for s in range(1,i-j+1):
        t3+=I(s,r,m,N-k+1)  
      t2=t2*V(j,k-1,i,m,N)
      t3=t3*V(j-1,k-1,i,m,N)
      t=t2+t3
      return t   
  
def RSSInclusion1(m,n,N):
  z=[0]*N  
  for i in range(1,N+1):  
    z1=0
    for j in range(0,i):
      z1+=V(j,n,i,m,N)
    z1=1-z1
    z[i-1]=z1   #分别计算RSS下元素1,...,N的包含概率，并存入数组z
  return z  

def RSS(m,n,z,iterations,N,a):
  d3,est=[0]*n,[]
  for k in range(iterations):
    a1,d1=a,[0]*n #初始化RSS和HTMRSS(m)样本#copy总体a,在此总体上分别执行RSS和HTMRSS(m)        
    for i in range(n):   #循环
        #print 'a1',a1  
        d= random.sample(a1, m)  #在总体a1中随机取m元素,list格式，for RSS
        d.sort()                       #抽样矩阵每行元素按从小到大排序    
        k1=i%m     #第i次SRS样本选取样本中秩为k1的元素     
        d1[i]=d[k1]               #d1为最终RSS样本,list格式      
        index1 = np.argwhere(a1==d1[i]) #将数值为d1[i]的a（ndarray格式）中索引赋值到index
        a1 = np.delete(a1, index1)  #从a1中删去这些元素（未被选中的元素返回总体）
        #print 'a1=',a1
    #print 'd1=',d1 
    indexd1=np.argwhere(a==d1[0])[0] #返回RSS第一个元素在总体中的秩，ndarray格式
    d3[0]=d1[0]/z[indexd1[0]]   #RSS第一个元素除以它的包含概率，list格式
    for i in range(1,n):
      t1=np.argwhere(a==d1[i])[0] 
      indexd1= np.append(indexd1, t1)  #继续添加RSS样本在在总体中的秩，ndarray格式
      d3[i]=d1[i]/z[indexd1[i]] #RSS其他元素除以它的包含概率，list格式
    d6=float(sum(d3))/N  #1次HTRSS估计值  
    est.append(d6)
    #vd4+=(d6-c)**2   #1万次HTRSS方差叠加
  #vd4=vd4/iterations #1万次HTRSS方差平均  
  d4=np.mean(est)   #1万次HTRSS估计值平均    
  vd4=np.var(est) #1万次HTRSS方差
  return (d4,vd4)
  
def HT2MRSS(m1,n,y,iterations,N,a): #a已经排好序，如果没有，加上一句a.sort()
  i2=int((m1+1)/2)-1  
  d10,size=0,n-2*i2
  for i3 in range(i2):
    d10+=a[i3]+a[N-1-i3]
  d8,est1,vd1=[0]*(size),[],0 #初始化    
  for i in range(iterations):
    a2,f1=a,[0]*(size)  #初始化  
    for j in range(size):   #抽n-2次和组成n-2个最终HTMRSS样本
      d7= random.sample(a2, m1)  #在总体a2中随机取m1个元素,list格式，for HTMRSS
      d7.sort()                       #抽样矩阵每行元素按从小到大排序 
      if m1%2!=0:
        r=int((m1+1)/2-1)  #HTMRSS(m)下每次规模为m的SRS中取d7的第r位元，r的计算方法
        #由于列表d7的第1位元是d7[0]，所以这里减一
      else:
        if j%2==0:   #执行SRS的次数，第j次,j=0,1,...,n-3.总共n-2次  
          r=int(m1/2-1)   #由于列表d7的第1位元是d7[0]，所以这里减一
        else:
          r=int(m1/2)  #HTMRSS(m)下若m为奇数，取r=(m+1)/2，若m位偶数，交替取第m/2和m/2+1位元
      f1[j]=d7[r]               #f1为最终HTMRSS样本,list格式
      index2 = np.argwhere(a2==f1[j]) #将数值为d7[i]的a（ndarray格式）中索引赋值到index
      a2 = np.delete(a2, index2)  #从a2中删去这些元素（未被选中的元素返回总体）
      #print 'a2=',a2
    #print 'f1=',f1      
    indexd2=np.argwhere(a==f1[0])[0] #返回HTMRSS第一个元素在总体中的秩，ndarray格式
    d8[0]=f1[0]/y[indexd2[0]]   #HTMRSS(m)第一个元素除以它的包含概率，list格式  
    if size>1:  #如果抽的个数size至少是两个
      for k in range(1,size):
        t2=np.argwhere(a==f1[k])[0] 
        indexd2= np.append(indexd2, t2)  #继续添加HTMRSS样本在在总体中的秩，ndarray格式
        d8[k]=f1[k]/y[indexd2[k]] #HTMRSS其他元素除以它的包含概率，list格式                
    d9=float(sum(d8)+d10)/N  #1次HTMRSS估计值
    vd1+=(d9-realmean)**2
    est1.append(d9) #1万次HTMRSS估计值添加到列表
    #d3=float(sum(f1)+d10)/n  #1次MRSS估计值
    #vd2+=(d3-realmean)**2
    #est2.append(d3) #1万次HTMRSS估计值添加到列表
  #d2=np.mean(est2) #1万次MRSS(m)估计值平均
  #vd2=vd2/iterations #1万次MRSS(m)方差平均
  d1=np.mean(est1) #1万次HTMRSS(m)估计值平均
  vd1=vd1/iterations  #MSE
  return (d1,vd1) #抽极端值下HT估计的均值和方差

def HT1MRSS(m1,n,y,iterations,N,a):
  d8,est1,vd1=[0]*n,[],0 #初始化  
  for i in range(iterations):
    a2,f1=a,[0]*(n)  #初始化  
    for j in range(n):   #抽n次和组成n个最终HTMRSS样本
      d7= random.sample(a2, m1)  #在总体a2中随机取m1个元素,list格式，for HTMRSS
      d7.sort()                       #抽样矩阵每行元素按从小到大排序 
      if m1%2!=0:
        r=int((m1+1)/2-1)  #HTMRSS(m)下每次规模为m的SRS中取d7的第r位元，r的计算方法
        #由于列表d7的第1位元是d7[0]，所以这里减一
      else:
        if j%2==0:   #执行SRS的次数，第j次,j=0,1,...,n-3.总共n-2次  
          r=int(m1/2-1)   #由于列表d7的第1位元是d7[0]，所以这里减一
        else:
          r=int(m1/2)  #HTMRSS(m)下若m为奇数，取r=(m+1)/2，若m位偶数，交替取第m/2和m/2+1位元
      f1[j]=d7[r]               #f1为最终HTMRSS样本,list格式
      index2 = np.argwhere(a2==f1[j]) #将数值为d7[i]的a（ndarray格式）中索引赋值到index
      a2 = np.delete(a2, index2)  #从a2中删去这些元素（未被选中的元素返回总体）
      #print 'a2=',a2
    #print 'f1=',f1      
    indexd2=np.argwhere(a==f1[0])[0][0] #返回HTMRSS第一个元素在总体中的秩，ndarray格式
    d8[0]=f1[0]/y[indexd2]   #HTMRSS(m)第一个元素除以它的包含概率，list格式  
    if n>3:
      for i1 in range(1,n):
        t2=np.argwhere(a==f1[i1])[0] 
        indexd2= np.append(indexd2, t2)  #继续添加HTMRSS样本在在总体中的秩，ndarray格式
        d8[i1]=f1[i1]/y[indexd2[i1]] #HTMRSS其他元素除以它的包含概率，list格式   
    d9=float(sum(d8))/N  #1次HTMRSS估计值
    vd1+=(d9-realmean)**2
    est1.append(d9) #1万次HTMRSS估计值添加到列表
    #d3=float(sum(f1))/n  #1次MRSS样本均值估计值
    #vd2+=(d3-realmean)**2
    #est2.append(d3) #1万次HTMRSS估计值添加到列表
  #d2=np.mean(est2) #1万次MRSS估计值平均
  d1=np.mean(est1) #1万次HTMRSS估计值平均
  vd1=vd1/iterations  #MSE
  #vd2=vd2/iterations  #MSE
  return (d1,vd1) #不抽极端值下HT估计的均值和MSE

def run(N,n,a):  
  a.sort() #将总体a中元素从小到大排序
  m2=n
  z2=np.loadtxt("D:\\MRSS\\N_n_m=%d_%d_%d HT1MRSS Inclu_Prob.txt"%(N,n,2)) #SRS规模2，不抽极端值
  z3=np.loadtxt("D:\\MRSS\\N_n_m=%d_%d_%d HT1MRSS Inclu_Prob.txt"%(N,n,3)) #SRS规模3，不抽极端值
  zm=np.loadtxt("D:\\MRSS\\N_n_m=%d_%d_%d HT1MRSS Inclu_Prob.txt"%(N,n,m2)) #SRS规模n，不抽极端值
  iterations=1000
  MRSSresultz2=HT1MRSS(2,n,z2,iterations,N,a) #SRS规模2，不抽极端值时的HT估计
  MRSSresultz3=HT1MRSS(3,n,z3,iterations,N,a) #SRS规模3，不抽极端值时的HT估计
  MRSSresultzm=HT1MRSS(m2,n,zm,iterations,N,a) #SRS规模n，不抽极端值时的HT估计
  #RSSresult=RSS(n,n,z,iterations,N,a)  
  return (MRSSresultzm[0],MRSSresultzm[1],MRSSresultz2[0],MRSSresultz2[1],MRSSresultz3[0],MRSSresultz3[1])
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#



Para_dgamma=[2.3, 0.734, 0.448, 0.324] #dgamma分布的参数，分别对应峰度值0，5，10,15
Para_dweibull=[1.439, 0.891, 0.755, 0.686 ]
Para_genpareto=[-0.350, -0.023, 0.061,0.101]
Para_lognorm=[0.001, 0.472, 0.596, 0.670]
var1,var2,var3=[],[],[]
MSE1,MSE2,MSE3=[],[],[]
kurt=[0,5,10,15]
N=50 #修改总体规模
n=int(N/4)
name='lognorm'   #修改分布名称1 ,共三处 
for i1 in range(4): 
  t=Para_lognorm[i1]  #修改分布名称2,共三处 
  k1=0
  for j1 in range(10000):
    if k1<10:  
      a =lognorm.rvs(t, size=N) #修改分布名称3,共三处 
      if 5*i1<kurtosis(a)<(5*i1+0.1):   
        print 'kurt=',kurtosis(a) 
        realmean=np.mean(a)
        res=run(N,n,a)        
        var1.append(res[1])
        var2.append(res[3])
        var3.append(res[5])
        k1+=1
  MSE1.append(np.mean(var1))
  MSE2.append(np.mean(var2))
  MSE3.append(np.mean(var3))

np.savetxt("D:\\MRSS\\MSE_N_n=%d_%d %s HT1MRSS.txt"%(N,n,name), MSE1) 
np.savetxt("D:\\MRSS\\MSE_N_n=%d_%d %s HT1MRSS(2).txt"%(N,n,name), MSE2) 
np.savetxt("D:\\MRSS\\MSE_N_n=%d_%d %s HT1MRSS(3).txt"%(N,n,name), MSE3)
###############################################################################
plt.figure(1)  

name_list = ['kurt=0','kurt=5','kurt=10','kurt=15']  
MSE1 =np.loadtxt("D:\\MRSS\\MSE_N_n=%d_%d %s HT1MRSS.txt"%(N,n,name)) 
MSE2 =np.loadtxt("D:\\MRSS\\MSE_N_n=%d_%d %s HT1MRSS(2).txt"%(N,n,name))   
MSE3 =np.loadtxt("D:\\MRSS\\MSE_N_n=%d_%d %s HT1MRSS(3).txt"%(N,n,name)) 

x =list(range(len(MSE1)))  
total_width, n1 = 0.8, 5  
width = total_width / n1  
plt.bar(x, MSE2, width=width, label='HTMRSS(2)',fc = 'b')  

for i in range(len(x)):  
    x[i] = x[i] + width  
plt.bar(x, MSE3, width=width, label='HTMRSS(3)',tick_label = name_list,fc = 'g')    

for i in range(len(x)):  
    x[i] = x[i] + width 
plt.bar(x, MSE1, width=width, label='HTMRSS',fc = 'r')  

plt.title('N,n=%d,%d; Distribution: %s'%(N,n,name))  
plt.legend()  
plt.savefig("D:\\MRSS\\fig\\MSE_N_n=%d_%d %s.jpg"%(N,n,name)) #保存柱状图
plt.show()











#print 'HT1MRSS(n) average MSE=',np.mean(var1) , 'average bias**2= ' ,(np.mean(bias1) )
#print 'HT1MRSS(2) average MSE=',np.mean(var2), 'average bias**2= ', (np.mean(bias2))
#print 'HT1MRSS(3) average MSE=',np.mean(var3), 'average bias**2= ', (np.mean(bias3))
#print 'HT2MRSS(n) average var=',np.mean(var4) , 'average bias**2=',(np.mean(bias4))    
#print 'HT2MRSS(3) average var=',np.mean(var5) , 'average bia**2s=',(np.mean(bias5) )
#print 'N=',N ,'n=',n    
         
  

