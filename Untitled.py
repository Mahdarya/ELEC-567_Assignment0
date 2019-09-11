#!/usr/bin/env python
# coding: utf-8

# In[153]:


from numpy import *
import scipy.linalg
a=array([[1,2,3],[3,4,5]])
print(a)


# In[78]:


ndim(a)


# In[79]:


size(a)


# In[80]:


shape(a)


# In[82]:


a[1,2]


# In[83]:


array([[1.,2,3],[4,5,6]])


# In[87]:


a1=array([1,2])
b1=array([3,4])
c1=array([5,6])
d1=array([7,8])
block([[a1,b1],[c1,d1]])


# In[89]:


a1[-1]


# In[90]:


a[0,1]


# In[91]:


a[1]


# In[92]:


a[1,:]


# In[94]:


a[0:2]


# In[95]:


a[:2]


# In[96]:


a[0:2,:]


# In[102]:


a[-2,:]


# In[103]:


a[-2:]


# In[106]:


a[0:1,:][:,1:3]


# In[108]:


a[ix_([0,1],[1,2])]


# In[111]:


a[1,0:3:2]


# In[112]:


a[1,::2]


# In[113]:


a[::-1,:]


# In[115]:


a[r_[:len(a),]]


# In[116]:


a.transpose()


# In[117]:


a.T


# In[118]:


a.conj().T


# In[123]:


(a1.T)@b1


# In[125]:


a1@b1.T


# In[126]:


a1*b1


# In[127]:


a1/b1


# In[131]:


a1**3


# In[132]:


a>1


# In[135]:


print(a)
nonzero(a>1)


# In[137]:


a[:,nonzero(a[1]>1)[0]]


# In[138]:


a[:,a[1].T>1]


# In[154]:


a[a<2]=0
print(a)


# In[155]:


a=a*(a>3)
print(a)


# In[157]:


a[:]=3
print(a)


# In[162]:


x=array([[1,2],[3,4],[5,6]])
y=x.copy()
print(y)


# In[163]:


y=x[1,:].copy()
print(y)


# In[165]:


y=x.flatten()
print(y)


# In[174]:


arange(1.,11.)


# In[175]:


r_[1.:11.]


# In[170]:


r_[1:10:10j]


# In[171]:


arange(10.)


# In[172]:


r_[:10.]


# In[173]:


r_[:9:10j]


# In[176]:


arange(1.,11.)[:,newaxis]


# In[177]:


zeros((3,4))


# In[178]:


ones((3,4))


# In[182]:


e=eye(3)


# In[183]:


diag(e)


# In[187]:


diag(e,0)


# In[188]:


random.rand(3,4)


# In[189]:


random.random_sample((3,4))


# In[190]:


linspace(1,3,4)


# In[191]:


mgrid[0:9.,0:6.]


# In[194]:


meshgrid(r_[0:9.],r_[0:6.])


# In[195]:


ogrid[0:9.,0:6.]


# In[196]:


ix_(r_[0:9.],r_[0:6.])


# In[197]:


meshgrid([1,2,4],[2,4,5])


# In[198]:


ix_([1,2,4],[2,4,5])


# In[200]:


tile(a1,(2,3))


# In[209]:


concatenate((a1,b1))


# In[206]:


hstack((a1,b1))


# In[212]:


r_[a1,b1]


# In[215]:


concatenate((a1.T,b1.T))


# In[220]:


c_[a1,b1]


# In[221]:


vstack((a1,b1))


# In[223]:


a1.max()


# In[224]:


a1.max(0)


# In[225]:


maximum(a1,b1)


# In[226]:


sqrt(a1@a1)


# In[227]:


np.linalg.norm(a1)


# In[230]:


logical_and(a1,b1)


# In[233]:


print(a1)
print(b1)
a1&b1


# In[234]:


a1|b1


# In[237]:


linalg.inv(e)


# In[253]:


linalg.pinv(e)


# In[238]:


linalg.matrix_rank(e)


# In[249]:


a=array([[1,5],[2,2]])
b=array([[3],[5]])
linalg.solve(a,b)


# In[257]:


U, S, Vh = linalg.svd(a)
V=Vh.T
print(U,S,Vh,V)


# In[272]:


a=array([[1,1],[1,2]])
linalg.cholesky(a)


# In[289]:


D,V = linalg.eig(a)
print(D)
print(V)


# In[290]:


b=array([[1,1],[3,5]])
D,V=scipy.linalg.eig(a,b)
print(D)
print(V)


# In[291]:


Q, R= scipy.linalg.qr(a)
print(Q)
print(R)


# In[292]:


LU,P=scipy.linalg.lu_factor(a)
print(LU)
print(P)


# In[294]:


#from scipy.sparse.linalg import cg
scipy.sparse.linalg.cg


# In[305]:


#from numpy import *
fft.fft(a)


# In[307]:


fft.ifft(a)


# In[308]:


sort(a)


# In[315]:


I = argsort(a[:,1])
b=a[I,:]
print(I)
print(b)


# In[327]:


X=array([[3,4],[1,1]])
y=array([1,5])
scipy.linalg.lstsq(X,y)


# In[340]:


from scipy import signal
x=array([1,2,3,10])
scipy.signal.resample(x,int(len(x)/1))


# In[341]:


unique(a)


# In[342]:


a.squeeze()


# In[343]:


import matplotlib.pyplot as plt
plt.plot([1,2,3,4], [1,2,7,14])
plt.axis([0, 6, 0, 20])
plt.show()


# In[344]:


import matplotlib.pyplot as plt
plt.plot([1,2,3,4], [1,2,7,14])
plt.axis([0, 6, 0, 20])
plt.show()


# In[348]:


x= linspace(0,5,20)
plt.plot(x, sqrt(x))
plt.axis([0,4,0,2])
plt.show()


# In[ ]:




