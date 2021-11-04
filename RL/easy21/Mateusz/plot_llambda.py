import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

with open('mse_sl_V0.0.pkl','rb') as f:
    l00 = pkl.load(f)  
with open('mse_sl_V0.1.pkl','rb') as f:
    l01 = pkl.load(f)  
with open('mse_sl_V0.2.pkl','rb') as f:
    l02 = pkl.load(f)  
with open('mse_sl_V0.3.pkl','rb') as f:
    l03 = pkl.load(f)  
with open('mse_sl_V0.4.pkl','rb') as f:
    l04 = pkl.load(f) 
with open('mse_sl_V0.5.pkl','rb') as f:
    l05 = pkl.load(f)  
with open('mse_sl_V0.6.pkl','rb') as f:
    l06 = pkl.load(f)  
with open('mse_sl_V0.7.pkl','rb') as f:
    l07 = pkl.load(f) 
with open('mse_sl_V0.8.pkl','rb') as f:
    l08 = pkl.load(f) 
with open('mse_sl_V0.9.pkl','rb') as f:
    l09 = pkl.load(f)  
with open('mse_sl_V1.0.pkl','rb') as f:
    l10 = pkl.load(f) 
 

fig,ax = plt.subplots()



eps = 400
ax.plot(l00[:eps], label='lamda = 0.0')
ax.plot(l01[:eps], label='lamda = 0.1')
ax.plot(l02[:eps], label='lamda = 0.2')
ax.plot(l03[:eps], label='lamda = 0.3')
ax.plot(l04[:eps], label='lamda = 0.4')
ax.plot(l05[:eps], label='lamda = 0.5')
ax.plot(l06[:eps], label='lamda = 0.6')
ax.plot(l07[:eps], label='lamda = 0.7')
ax.plot(l08[:eps], label='lamda = 0.8')
ax.plot(l09[:eps], label='lamda = 0.9')
ax.plot(l10[:eps], label='lamda = 1.0')


ax.set_ylabel('MSE')
ax.set_xlabel('kilo-Episodes')

plt.show()

