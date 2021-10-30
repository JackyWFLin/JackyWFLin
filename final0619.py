#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import os


# In[3]:


def nothing(x):
    pass


# In[4]:


def blur(img,size,times):
    
    for i in range(times):
        img = cv2.blur(img, (size, size))
    #cv2.GaussianBlur(img,(size,size),times)
    return img


# In[ ]:





# In[13]:


Save_path = 'Judge\\0802\\'
path = '.\\0802\\'
i = 1

file = os.listdir(path)
for f in file:
    print(path + f)
    img_ori = cv2.imread(path + f)
    img_rgb = cv2.imread(path + f)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    gray_blur = cv2.blur(gray, (3, 3))
    gray_blur = cv2.blur(gray_blur, (3, 3))
    ret1, th_binary = cv2.threshold(gray_blur, 46, 255, cv2.THRESH_BINARY)
    
    
    #gray_blur = blur(gray.copy(),5,5)
    
    
    #ret1, th_binary = cv2.threshold(gray_blur, 130, 255, cv2.THRESH_BINARY)
    #ret2, th_binary = cv2.threshold(gray_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th_binary = cv2.medianBlur(th_binary,5)
    
    contours, hierarchy = cv2.findContours(th_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    print("---")
    for c in contours:
        if(cv2.contourArea(c)>18) and (cv2.contourArea(c)<70):
            x,y,w,h = cv2.boundingRect(c)
            if(abs(w-h)<4):
                count+=1
                cv2.drawContours(img_rgb,[c],-1,(0,0,255),2)
                cv2.putText(img_rgb,str(int(cv2.contourArea(c))),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
                #cv2.putText(img_rgb,str(abs(w-h)),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)

    cv2.putText(img_rgb,"Count:" + str(count),(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),1,cv2.LINE_AA)
    
    cv2.imshow('My Image', img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(Save_path + str(i) + '.jpg' , img_ori)
    cv2.imwrite(Save_path + str(i) + '_judge.jpg' , img_rgb)
    i+=1


# In[14]:


path = '.\\0802\\'
file = os.listdir(path)
print(path + file[0])
gray = cv2.imread(path +'00.jpg',0)
gray_blur = gray
gray_blur = blur(gray,3, 2)
ret1, th_binary = cv2.threshold(gray_blur, 46, 255, cv2.THRESH_BINARY)
th_binary = cv2.medianBlur(th_binary,5)

contours, hierarchy = cv2.findContours(th_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
count = 0
print("---")
for c in contours:
    '''
    if(cv2.contourArea(c)>100) and (cv2.contourArea(c)<450):
        x,y,w,h = cv2.boundingRect(c)
        if(abs(w-h)<8):
            count+=1
            #print(cv2.contourArea(c))
            #print(x,y,w,h)
            cv2.drawContours(img_rgb,[c],-1,(0,0,255),2)
            cv2.putText(img_rgb,str(int(cv2.contourArea(c))),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
            '''
    count+=1
    #print(cv2.contourArea(c))
    #print(x,y,w,h)
    cv2.drawContours(img_rgb,[c],-1,(0,0,255),2)
    cv2.putText(img_rgb,str(int(cv2.contourArea(c))),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)

cv2.putText(img_rgb,"Count:" + str(count),(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),1,cv2.LINE_AA)


cv2.imshow('My Image1', gray)
cv2.imshow('My Image', th_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
    


# In[ ]:


Save_path = 'Judge\\0802\\'
path = '.\\0802\\'
i = 1

file = os.listdir(path)
for f in file:
    print(path + f)
    img_ori = cv2.imread(path + f)
    img_rgb = cv2.imread(path + f)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    gray_blur = cv2.blur(gray, (3, 3))
    gray_blur = cv2.blur(gray_blur, (3, 3))
    ret1, th_binary = cv2.threshold(gray_blur, 45, 255, cv2.THRESH_BINARY)
    
    
    #gray_blur = blur(gray.copy(),5,5)
    
    
    #ret1, th_binary = cv2.threshold(gray_blur, 130, 255, cv2.THRESH_BINARY)
    #ret2, th_binary = cv2.threshold(gray_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th_binary = cv2.medianBlur(th_binary,5)
    
    contours, hierarchy = cv2.findContours(th_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    print("---")
    for c in contours:
        
        if(cv2.contourArea(c)>100) and (cv2.contourArea(c)<450):
            x,y,w,h = cv2.boundingRect(c)
            if(abs(w-h)<8):
                count+=1
                #print(cv2.contourArea(c))
                #print(x,y,w,h)
                cv2.drawContours(img_rgb,[c],-1,(0,0,255),2)
                cv2.putText(img_rgb,str(int(cv2.contourArea(c))),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)

    cv2.putText(img_rgb,"Count:" + str(count),(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),1,cv2.LINE_AA)
    
    cv2.imshow('My Image', img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(Save_path + str(i) + '.jpg' , img_ori)
    cv2.imwrite(Save_path + str(i) + '_judge.jpg' , img_rgb)
    i+=1


# In[18]:


gray


# In[5]:


path = '.\\img\\'
file = os.listdir(path)
print(path + file[0])
gray = cv2.imread(path +'2.jpg',0)
gray_blur = gray
gray_blur = cv2.blur(gray, (11, 11))
gray_blur = cv2.blur(gray_blur, (11, 11))
#gray_blur = cv2.equalizeHist(gray_blur)

winName = 'canny tool'
cv2.namedWindow(winName)

cv2.createTrackbar('LowerbH',winName,0,255,nothing)

cv2.createTrackbar('UpperbH',winName,0,255,nothing)

cv2.createTrackbar('binary',winName,0,255,nothing)

cv2.createTrackbar('c1',winName,1,255,nothing)
cv2.createTrackbar('c2',winName,0,255,nothing)

while(1):


    lowerbH=cv2.getTrackbarPos('LowerbH',winName)

    upperbH=cv2.getTrackbarPos('UpperbH',winName)

    binary=cv2.getTrackbarPos('binary',winName)
    
    c1=cv2.getTrackbarPos('c1',winName)
    c1 = c1*2 + 1
    
    c2=cv2.getTrackbarPos('c2',winName)

    ret1, th_binary = cv2.threshold(gray_blur, binary, 255, cv2.THRESH_BINARY)
    #th_binary = cv2.adaptiveThreshold(gray_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C  ,cv2.THRESH_BINARY,c1,c2)

    #ret2,th_binary = cv2.threshold(gray_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    
    cv2.imshow('result3', cv2.resize(th_binary, (int(gray.shape[1] / 4),int(gray.shape[0] / 4)), interpolation=cv2.INTER_AREA))
    cv2.imshow('source',  cv2.resize(gray_blur, (int(gray.shape[1] / 4),int(gray.shape[0] / 4)), interpolation=cv2.INTER_AREA))

    if cv2.waitKey(1)==ord('q'):

        break


cv2.destroyAllWindows()


# In[6]:





# In[ ]:




