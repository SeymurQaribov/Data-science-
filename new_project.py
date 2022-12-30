#!/usr/bin/env python
# coding: utf-8

# In[1]:


Array = [80, 85, 90, 95, 100, 105, 110, 115, 120, 125]
print(Array)#Bu Datanı sutrukturlaşdırmaq və təqdim etmənin ən yaxçı yollarından biridi.


# In[2]:


Array = max(80, 85, 90, 95, 100, 105, 110, 115, 120, 125)
print(Array)#Masimum dəyərin tapılması


# In[3]:


Array = min(80, 85, 90, 95, 100, 105, 110, 115, 120, 125)
print(Array)#Minimum dəyərin hesablanması


# In[4]:


import numpy as np 
Array = [80, 85, 90, 95, 100, 105, 110, 115, 120, 125]
Mean_Array = np.mean(Array)
print(Mean_Array)#Orta dəyərin hesablanması


# In[5]:


import pandas as pd

d={'col1':[1,2,3,4,5],'col2':[4,5,2,7,8],'col3':[9,3,2,4,5]}
df = pd.DataFrame(data=d)
print(df)#Verilmiş datanın cədvəl halından təqdim etmək üçün Pythonnın pandas kitabxanasından istifadə edə bilərik.


# In[6]:


count_column = df.shape[1]#Cədvəldəki sütunların hesablanması
print(count_column)


# In[7]:


count_row = df.shape[0]#Sətirlərin hesablanması
print(count_row)


# In[8]:


d2 =  {'Müddət':[30,30,45,45,45,60,60,60,75,75],
        'Orta_Ürəkdöyüntüləri':[80,85,90,95,100,105,110,115,120,125],
        'Max_Ürəkdöyüntüləri':[120,120,130,130,140,140,145,145,150,150],
        'Kalori_yanma':[240,250,260,270,280,290,300,310,320,330],
        'İs_müddəti':[10,10,8,8,0,7,7,8,0,8],
        'Yatmaq_müddəti':[7,7,7,7,7,8,8,8,8,8]}#Bu məlumatlardan verilmiş dəyişənlər arasında əqlaqənin mövcud olub olmadığını
# və birinin dəyişməsi digərlərinə necə təsir göstərdiyini başa düşmək üçün istifadə edəceəyik.
df2=pd.DataFrame(data=d2)
print(df2)


# In[9]:


df2.dropna(axis = 0,inplace = True)
print(df2)#Əgər cədvəldə NaN dəyərlər varsa onları  silmək üçün dropna() funksiyasından istifadə edə bilərik.


# In[10]:


print(df2.info())#Cədvəl haqqında məumat əldə etmək üçün funksiya


# In[11]:


df2["Müddət"]=df2["Müddət"].astype(float)
df2["Orta_Ürəkdöyüntüləri"]=df2["Orta_Ürəkdöyüntüləri"].astype(float)
df2["Max_Ürəkdöyüntüləri"]=df2["Max_Ürəkdöyüntüləri"].astype(float)
df2["Kalori_yanma"]=df2["Kalori_yanma"].astype(float)
df2["İs_müddəti"]=df2["İs_müddəti"].astype(float)
df2["Yatmaq_müddəti"]=df2["Yatmaq_müddəti"].astype(float)#Data tipini dəyiştirkmək üçün funksiya


# In[12]:


print(df2.info())


# In[13]:


print(df2.describe())#Məlumatları ümumiləşdirmək üçün Python-da describe() funksiyasından istifadə edə bilərik


# In[14]:


import matplotlib.pyplot as plt#İndi matplotlib kitabxanasından istifadə edərək ilk olaraq
#Orta_Ürəkdöyüntülərinin dəyərlərini Kalori_yanma arasındakı əlaqəni əks etdirə bilərik.

df2.plot(x = 'Orta_Ürəkdöyüntüləri',y = 'Kalori_yanma',kind = 'line'),
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.show()


# In[15]:


std = np.std(df2)
print(std)#Dəyişənin standart kənarlaşmasını tapmaq üçün Numpy-dan std() funksiyasından istifadə edə bilərik


# In[16]:


cv = np.std(df2)/np.mean(df2)
print(cv)#The coefficient of variation standart kənarlaşmanın nə qədər 
#böyük olduğu barədə fikir əldə etmək üçün istifadə olunur(Coefficient of Variation = Standard Deviation / Mean).


# In[17]:


var = np.var(df2)#Variance dəyərlərin necə yayıldığını göstərən başqa bir rəqəmdir.
print(var)


# In[18]:


#Korrelyasiya iki dəyişən arasındakı əlaqəni ölçür.
# Əgər korrelyasiya = 1 olarsa dəyişənlər arasında mükəmməl xətti əlaqə var (məsələn, Kalori_Yanma qarşı Orta_Ürəkdöyüntüləri)
df2.plot(x= 'Orta_Ürəkdöyüntüləri',y='Kalori_yanma',kind = 'scatter')
plt.show()


# In[19]:


# Əgər korrelyasiya = 0 olarsa dəyişənlər arasında xətti əlaqə yoxdur
df2.plot(x='Müddət',y='Max_Ürəkdöyüntüləri',kind='scatter')
plt.show()


# In[20]:


#Əgər korrelyasiya = -1 olarsa dəyişənlər arasında mükəmməl mənfi xətti əlaqə var
menfi_corr = {'İs_müddəti': [10,9,8,7,6,5,4,3,2,1],
'Kalori_yanma': [220,240,260,280,300,320,340,360,380,400]}
menfi_corr = pd.DataFrame(data=menfi_corr)

menfi_corr.plot(x ='İs_müddəti', y='Kalori_yanma', kind='scatter')
plt.show()


# In[21]:


#Korrelyasiya matrisi dəyişənlər arasında korrelyasiya əmsallarını göstərən cədvəldir.
Corr_matrix = round(df2.corr(),2)
print(Corr_matrix)


# In[22]:


#Dəyişənlər arasındakı əlaqəni vizuallaşdırmaq üçün Heatmap'dən istifadə edə bilərik
import seaborn as sns
corr_df2 = df2.corr()
axis_corr = sns.heatmap(
corr_df2,
vmin = -1,vmax = 1,center = 0,
cmap = sns.diverging_palette(50,500,n = 500),
square = True)
plt.show()


# In[23]:


#Bu nümunədə Xətti Reqressiyadan istifadə edərək Kalori_Yanmanı Orta_Ürəkdöyüntüləri proqnozlaşdırmağa çalışacağıq.
from scipy import stats
x = df2["Orta_Ürəkdöyüntüləri"]
y = df2["Kalori_yanma"]
slope,intercept,r,p,std_err = stats.linregress(x,y)
def my_func(x):
    return x*slope + intercept
mymodel = list(map(my_func,x))
plt.scatter(x,y)
plt.plot(x,mymodel)
plt.ylim(ymin = 0,ymax = 2000)
plt.xlim(xmin = 0,xmax = 200)
plt.xlabel("Orta_Ürəkdöyüntüləri")
plt.ylabel("Kalori_yanma")

plt.show()


# In[24]:


#Python-da xətti reqressiya cədvəlini necə yaratmaq olar
import statsmodels.formula.api as smf
model = smf.ols('Kalori_yanma~Orta_Ürəkdöyüntüləri',data = df2)
netice = model.fit()
print(netice.summary())# Bu cedvelde P deyerinin 0-ra beraber olamsi bu iki deyisen arasinda elaqenin oldugunu gosterir.
#Eger P>t 1-e yaxin deyer alsaydi ve ya 0.5-den boyuk olsaydi onlar arasinda elaqe olmazdi(Ancaq bu teqdim etdiyim datalar 
# realliga uygun deyil)
# R-squared 1-e bearberdi yeni data cox sepelenmir.


# In[25]:


def Predict_Kalori_yanma(Orta_Ürəkdöyüntüləri):
 return(2.0000*Orta_Ürəkdöyüntüləri + 80.0000)# Burada 80.000 ve 2.000 yuxaridan goturduyumuz intercept ve slope'du.

print(Predict_Kalori_yanma(140))
print(Predict_Kalori_yanma(120))
print(Predict_Kalori_yanma(190))
#Proqnozları yerinə yetirmək üçün Python-da xətti reqressiya funksiyasını təyin edek.

#Orta_Ürəkdöyüntüləri 140, 120, 190 olarsa, Kalori_Yanım nədir?


# In[26]:


import statsmodels.formula.api as smf
model = smf.ols('Kalori_yanma~Orta_Ürəkdöyüntüləri + Müddət ',data = df2)
netice = model.fit()
print(netice.summary())


# In[30]:


def Predict_Kalori_yanma(Orta_Ürəkdöyüntüləri,Müddət):
 return(2.0000*Orta_Ürəkdöyüntüləri + Müddət*(-1.11e-15) + 80.0000)

print(Predict_Kalori_yanma(140,60))
print(Predict_Kalori_yanma(120,23))
print(Predict_Kalori_yanma(190,40))


# In[ ]:




