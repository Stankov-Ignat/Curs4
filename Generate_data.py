#%%
import warnings
warnings.filterwarnings("ignore")

import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

import seaborn as sns
sns.set_style('whitegrid')

def h(t):
    if isinstance(t, pd.Series):
        return t.apply(lambda x: 1. if x >= 0 else 0.)
    else:
        return 1. if t >= 0 else 0.


#%%

param_data = np.array([[69, np.NAN, np.NAN],
                      [10.88, 1.088, 0.1],
                      [9.009, 9.009, 1],
                      [5.388, 53.88, 10],
                      [5.975, 597.5, 1.e2],
                      [10.4, 10400, 1.e3]])

pm = pd.DataFrame(param_data, columns=['E', 'nu', 'z'])

N = 5                                                                          # число членов в ряде прони
time_steps = 1000;    time_end = 100

time_mesh = pd.Series(np.linspace(0, time_end, time_steps))

#               0        1       2       3       4       5       6       7      8
# count_frame = [10_000,  10_000,  10_000,  10_000,  10_000,  10_000,  3_000,  3_000, 3_000]

count_frame = [30000 for i in range(9)]

#%%           0  ЛИНЕЙНАЯ ДЕФОРМАЦИЯ 

def strain_0(k, t):
    return k*t

def stress_0(k, t):
    return pm.iloc[0,0]*strain_0(k, t) + sum( k*pm.iloc[i,1] * (1-np.exp(-t/pm.iloc[i,2])) for i in range(1,N+1) )

start_0 = time.time()

np.random.seed(12)
data_k = np.random.uniform(1, 20, count_frame[0])*1.e-5
 
df_0_strain = pd.DataFrame(strain_0(data_k[0], time_mesh)).T
df_0_stress = pd.DataFrame(stress_0(data_k[0], time_mesh)).T

for k in data_k[1:]:
    df_0_strain.loc[len(df_0_strain)] = strain_0(k, time_mesh)
    df_0_stress.loc[len(df_0_stress)] = stress_0(k, time_mesh)

print(df_0_strain.shape)

finish_0 = time.time()
print(f'time_0 = {finish_0 - start_0:.2f} sec   max_strain = {100 * df_0_strain.max().max():.2f} %')

#%%           1  ЛИНЕЙНАЯ РАМП ДЕФОРМАЦИЯ 

def strain_1(k, t0, t):
    return k*t*(h(t) - h(t-t0)) + k*t0*h(t-t0)

def stress_1(k, t0, t):
    return pm.iloc[0,0]*strain_1(k,t0,t) + k * sum(pm.iloc[i,1] * (np.exp(-(t-t0)/pm.iloc[i,2]) - np.exp(-t/pm.iloc[i,2]))*h(t-t0) +
                                                   pm.iloc[i,1] * (1 - np.exp(-t/pm.iloc[i,2]))*h(t0-t) for i in range(1,N+1) )
start_1 = time.time()

np.random.seed(12)
data_k_1 = np.random.uniform(1, 7, count_frame[1])*1.e-4 / 2

np.random.seed(2)
data_t0 = np.random.uniform(0, 70, count_frame[1])

df_1_strain = pd.DataFrame(strain_1(data_k_1[0], data_t0[0], time_mesh)).T
df_1_stress = pd.DataFrame(stress_1(data_k_1[0], data_t0[0], time_mesh)).T

for k,t0 in zip(data_k_1[1:], data_t0[1:]):
    df_1_strain.loc[len(df_1_strain)] = strain_1(k,t0,time_mesh)
    df_1_stress.loc[len(df_1_stress)] = stress_1(k,t0,time_mesh)

print('count_nan = ',df_1_strain.shape,df_1_stress.isna().sum().sum())

finish_1 = time.time()
print(f'time_1 = {finish_1 - start_1:.2f} sec   max_strain = {100 * df_1_strain.max().max():.2f} %')

#%%           2  КВАДРАТИЧНАЯ РАМП ДЕФОРМАЦИЯ 

def fun_2(i,t0,t):
    return ( 
             h(t-t0)*( np.exp(-t/pm.iloc[i,2]) + (t0/pm.iloc[i,2] - 1)*np.exp(-(t-t0)/pm.iloc[i,2]) ) +
             h(t0-t)*(t/pm.iloc[i,2] - 1 + np.exp(-t/pm.iloc[i,2]))
           )     
            
def strain_2(k, t0, t):
    return k*t*t*(h(t) - h(t-t0)) + k*t0*t0*h(t-t0)

def stress_2(k, t0, t):
    return  pm.iloc[0,0]*strain_2(k,t0,t) + 2*k * sum( pm.iloc[i,1]*pm.iloc[i,2] * fun_2(i,t0,t) for i in range(1, N+1) )
           
start_2 = time.time()

np.random.seed(12)
data_k_2 = np.random.uniform(1, 20, count_frame[2])*1.e-6 / 5

np.random.seed(2)
data_t0_1 = np.random.uniform(0, 70, count_frame[2])

df_2_strain = pd.DataFrame(strain_2(data_k_2[0], data_t0_1[0], time_mesh)).T
df_2_stress = pd.DataFrame(stress_2(data_k_2[0], data_t0_1[0], time_mesh)).T

for k,t0 in zip(data_k_2[1:], data_t0_1[1:]):
    df_2_strain.loc[len(df_2_strain)] = strain_2(k,t0,time_mesh)
    df_2_stress.loc[len(df_2_stress)] = stress_2(k,t0,time_mesh)

print('count_nan = ',df_2_strain.shape,df_2_stress.isna().sum().sum())

finish_2 = time.time()
print(f'time_2 = {finish_2 - start_2:.2f} sec   max_strain = {100 * df_2_strain.max().max():.2f} %')

#%%           3  КВАДРАТИЧНАЯ ДЕФОРМАЦИЯ

def fun_3(i,t):
    return pm.iloc[i,0] * pm.iloc[i,2]**2 * ( t/pm.iloc[i,2] - 1 + np.exp(-t/pm.iloc[i,2]) )

def strain_3(al, t):
    return al*t*t

def stress_3(al, t):
    return pm.iloc[0,0]*strain_3(al,t) + 2*al * sum( fun_3(i,t) for i in range(1,N+1) )


np.random.seed(3)
data_al_0 = np.random.uniform(1, 20, count_frame[3])*1.e-7

start_3 = time.time()
    
df_3_strain = pd.DataFrame(strain_3(data_al_0[0], time_mesh)).T
df_3_stress = pd.DataFrame(stress_3(data_al_0[0], time_mesh)).T

for al in data_al_0[1:]:
    df_3_strain.loc[len(df_3_strain)] = strain_3(al,time_mesh)
    df_3_stress.loc[len(df_3_stress)] = stress_3(al,time_mesh)

print(df_3_strain.shape)

finish_3 = time.time()
print(f'time_3 = {finish_3 - start_3:.2f} sec   max_strain = {100 * df_3_strain.max().max():.2f} %')

#%%           4  КУБИЧЕСКАЯ ДЕФОРМАЦИЯ

def fun_4(i,t):
    return pm.iloc[i,0] * pm.iloc[i,2]**3 * ( t/pm.iloc[i,2] * (t/pm.iloc[i,2] - 2) - 2*np.exp(-t/pm.iloc[i,2]) + 2 )

def strain_4(al, t):
    return al * t**3

def stress_4(al, t):
    return pm.iloc[0,0]*strain_4(al,t) + 3*al * sum( fun_4(i,t) for i in range(1,N+1) )

np.random.seed(6)
data_al_1 = np.random.uniform(1, 20, count_frame[4])*1.e-9 

start_4 = time.time()

df_4_strain = pd.DataFrame(strain_4(data_al_1[0], time_mesh)).T
df_4_stress = pd.DataFrame(stress_4(data_al_1[0], time_mesh)).T

for al in data_al_1[1:]:
    df_4_strain.loc[len(df_4_strain)] = strain_4(al,time_mesh)
    df_4_stress.loc[len(df_4_stress)] = stress_4(al,time_mesh)

print(df_4_strain.shape)

finish_4 = time.time()
print(f'time_4 = {finish_4 - start_4:.2f} sec   max_strain = {100 * df_4_strain.max().max():.2f} %')

#%%           5  S1_S2 ДЕФОРМАЦИЯ

def fun_5(i,s1,s2,t0,t):
    return s1*pm.iloc[i,0]*np.exp(-t/pm.iloc[i,2]) * h(t) + (s2-s1)*pm.iloc[i,0]*np.exp(-(t-t0)/pm.iloc[i,2]) * h(t-t0)

def strain_5(s1,s2,t0,t):
    return s1*h(t) + (s2-s1)*h(t-t0)

def stress_5(s1,s2,t0,t):
    return pm.iloc[0,0]*strain_5(s1,s2,t0,t) + sum( fun_5(i,s1,s2,t0,t) for i in range(1,N+1) )


np.random.seed(2)
data_s1 = np.random.uniform(1, 20, count_frame[5])*1.e-3

np.random.seed(6)
data_s2 = np.random.uniform(1, 20, count_frame[5])*1.e-3

np.random.seed(2)
data_t0_0 = np.random.uniform(30, 70, count_frame[5])

start_5 = time.time()

df_5_strain = pd.DataFrame(strain_5(data_s1[0],data_s2[0],data_t0_0[0],time_mesh)).T
df_5_stress = pd.DataFrame(stress_5(data_s1[0],data_s2[0],data_t0_0[0],time_mesh)).T

for s1,s2,t0 in zip(data_s1[1:],data_s2[1:],data_t0_0[1:]):
    df_5_strain.loc[len(df_5_strain)] = strain_5(s1,s2,t0,time_mesh)
    df_5_stress.loc[len(df_5_stress)] = stress_5(s1,s2,t0,time_mesh)

print(f'{df_5_strain.shape = }')

finish_5 = time.time()
print(f'time_5 = {finish_5 - start_5:.2f} sec   max_strain = {100 * df_5_strain.max().max():.2f} %')


#%%             6 СИНУС
def find_first_local_max_index(series):
    for i in range(1, len(series) - 1):
        if series[i] > series[i - 1] and series[i] > series[i + 1]:
            return i
    return len(series) - 1

def GG(strain, stress, b):
    G1 = np.array([]);      G2 = np.array([]);      delta = np.array([])
    for i in range(len(b)):
        
        i1 = find_first_local_max_index(strain.iloc[i]) 
        i2 = find_first_local_max_index(stress.iloc[i]) 
        
        delta = np.append(delta, abs(time_mesh[i1] - time_mesh[i2]) * b.iloc[i] / (2*np.pi)) 
    
        sigma = stress.iloc[i].max()
        epsilon = strain.iloc[i].max()
        G1 = np.append(G1, sigma/epsilon * np.cos(delta[-1]))
        G2 = np.append(G2, sigma/epsilon * np.sin(delta[-1]))
    
    G1_G2_delta = pd.DataFrame(np.column_stack((G1, G2, delta)))
    return G1_G2_delta

def GG1(strain, stress, b):
    G1 = np.array([]);      G2 = np.array([]);      delta = np.array([])
    for i in range(len(b)):
        
        i1 = find_first_local_max_index(strain.iloc[i]) 
        i2 = find_first_local_max_index(stress.iloc[i]) 
        
        strain_centered = strain.iloc[i] - np.mean(strain)
        stress_centered = stress.iloc[i] - np.mean(stress.iloc[i])
        
        sample_rate = 100  # частота дискретизации данных
        freq = np.fft.fftfreq(len(strain_centered), d=1/sample_rate)
        strain_fft = np.fft.fft(strain_centered)
        stress_fft = np.fft.fft(stress_centered)
        
        index_max = np.argmax(np.abs(strain_fft))
        main_freq = freq[index_max]
        
        cross_corr = np.correlate(strain_centered, stress_centered, mode='full')
        delay = np.argmax(cross_corr) - (len(strain) - 1)
        
        phase_shift = (2 * np.pi * main_freq * delay) / sample_rate
        
        delta_ = np.abs(phase_shift) 
        delta = np.append(delta, delta_)
        
        sigma = stress.iloc[i].max()
        epsilon = strain.iloc[i].max()
        G1 = np.append(G1, sigma/epsilon * np.cos(delta[-1]))
        G2 = np.append(G2, sigma/epsilon * np.sin(delta[-1]))
    
    G1_G2_delta = pd.DataFrame(np.column_stack((G1, G2, delta)))
    return G1_G2_delta

#%%

def const_6(i,b):
    return pm.iloc[i,1] / (b**2 * pm.iloc[i,2]**2 + 1)

def fun_6(i,a,b,t):
    return np.cos(b*t) + b* pm.iloc[i,2] *np.sin(b*t) - np.exp(-t/pm.iloc[i,2])

def strain_6(a,b,t):
    return a * np.sin(b*t)

def stress_6(a,b,t):
    return pm.iloc[0,0]*strain_6(a,b,t) + a*b * sum( const_6(i,b) * fun_6(i,a,b,t) for i in range(1,N+1) )


# data_a_0 = np.random.uniform(5, 20, count_frame[6])*1.e-3
# data_b_0 = np.random.uniform(1, 20, count_frame[6])*1.e-2 * 11/4

np.random.seed(3)
#data_a_0_G = np.linspace(5*1.e-3, 20*1.e-3, 1000)
data_a_0_G = np.full(10000, 1e-2)
np.random.seed(7)
data_b_0_G = np.linspace(11/4 * 1.e-2, 11/4 * 20*1.e-2, 10000)

df_6_strain = pd.DataFrame(strain_6(data_a_0_G[0],data_b_0_G[0],time_mesh)).T
df_6_stress = pd.DataFrame(stress_6(data_a_0_G[0],data_b_0_G[0],time_mesh)).T

defer = 0.
for a,b in zip(data_a_0_G[1:],data_b_0_G[1:]):
    defer = max(defer, abs(a*b))
    df_6_strain.loc[len(df_6_strain)] = strain_6(a,b,time_mesh)
    df_6_stress.loc[len(df_6_stress)] = stress_6(a,b,time_mesh)

#%%

def l1_rel(true_values, estimated_values, count_no=0):
    np.array(true_values).shape = (-1,1);  
    np.array(estimated_values).shape = (-1,1)
    return np.sum(np.abs(true_values[count_no:] - estimated_values[count_no:])) / np.sum(np.abs(true_values))

def strain_deferential(a,b,t):
    return a*b * np.cos(b*t)

def find_closest_position(num, array):
    array = np.array(array)
    differences = np.abs(array - num)
    closest_index = np.argmin(differences)
    return closest_index

data_b_0_G = pd.DataFrame(data_b_0_G)
t = GG1(df_6_strain, df_6_stress, data_b_0_G)
b_G1_G2_delta = pd.concat([data_b_0_G, t], axis=1)
b_G1_G2_delta.columns = ['b', 'G1_МПа', 'G2_МПа', 'delta_радиан']

b_G1_G2_delta.to_excel("C:\\Users\\751\\Documents\\Методические пособия\\Курсовая\\vxod_data_100_30k\\b_G1_G2_delta_4.xlsx",index=False)
# b_G1_G2_delta = pd.read_excel("C:\\Users\\751\\Documents\\Методические пособия\\Курсовая\\vxod_data_100_30k\\b_G1_G2_delta_1.xlsx")

i=100

b = b_G1_G2_delta.iloc[i,0];  G1 = b_G1_G2_delta.iloc[i,1];  G2 = b_G1_G2_delta.iloc[i,2] 
tt = G1*df_6_strain.iloc[i] + G2/b* strain_deferential(data_a_0_G[i], b, time_mesh)
 
print(l1_rel(df_6_stress.iloc[i], tt))

#%%           7  ПИЛА ДЕФОРМАЦИЯ

def H(t,t1,t2):
    return h(t-t1) - h(t-t2)

def I(t_,T_):
    return h(t_ - T_)

def fun_7_1(k,i,n,T,T_,t):
    q_1 = lambda t: 0. if H(t, T*k, T*k + T_) == 0       else (1 - np.exp(-(t-T*k)/pm.iloc[i, 2]))
    q_2 = lambda t: 0. if h(t - T*k - T_) == 0           else (np.exp(-(t-T*k-T_)/pm.iloc[i, 2]) - np.exp(-(t-T*k)/pm.iloc[i, 2]))
    q_3 = lambda t: 0. if H(t, T*k + T_, T*(k+1)) == 0   else (1 - np.exp(-(t-T*k-T_)/pm.iloc[i, 2]))
    q_4 = lambda t: 0. if h(t-T*(k+1)) == 0              else (np.exp(-(t-T*(k+1))/pm.iloc[i, 2]) - np.exp(-(t-T*k-T_)/pm.iloc[i, 2]))
    return t.apply(q_1) + t.apply(q_2) - t.apply(q_3) - t.apply(q_4)


def fun_7_2(i,t,T,T_,t_,n):
    q_1 = lambda t: 0. if H(t,T*n,T*n + T_) == 0         else ( 1 - np.exp(-(t-T*n)/pm.iloc[i,2]) )
    q_2 = lambda t: 0. if h(t- T*n -T_) == 0             else I(t_,T_) * ( np.exp(-(t-T*n-T_)/pm.iloc[i,2]) - np.exp(-(t - T*n)/pm.iloc[i,2]) )
    q_3 = lambda t: 0. if H(t,T*n + T_, T*(n+1)) == 0    else I(t_,T_) * ( 1 - np.exp(-(t-T*n-T_)/pm.iloc[i,2]) )
    return t.apply(q_1) + t.apply(q_2) - t.apply(q_3)


def fun_7(i,n,T,T_,b,t):
    return sum(fun_7_1(k,i,n,T,T_,t) for k in range(int(n)))

def strain_7(a,b,t):
    return a * ((np.arcsin(np.sin(b * t - np.pi / 2)) + np.pi / 2))

def stress_7(a,b,t):
    T = 2 * np.pi / b
    n = time_end // T; 
    t_ = time_end - n*T;                                                  # остаток
    T_ = np.pi / b
    return pm.iloc[0,0]*strain_7(a,b,t) + a*b * sum( pm.iloc[i,1] * (fun_7(i,n,T,T_,b,t) + fun_7_2(i,t,T,T_,t_,n)) for i in range(1,N+1) )


np.random.seed(2)
data_a_1 = np.random.uniform(10, 20, count_frame[7])*(1.e-3 / np.pi)

np.random.seed(6)
data_b_1 = np.random.uniform(10, 100, count_frame[7])*(1.e-2 * np.pi/2)

if 0:
    n = 150
    c = [1,3,5,10]
    fig, ax = plt.subplots(2,2)
    ax[0][0].plot(np.linspace(0,100,H), strain_7(data_a_1[c[0]],data_b_1[c[0]],np.linspace(0,100,n)), label='strain', c='purple')
    ax[0][1].plot(np.linspace(0,100,H), strain_7(data_a_1[c[1]],data_b_1[c[1]],np.linspace(0,100,n)), label='strain', c='purple')
    ax[1][0].plot(np.linspace(0,100,H), strain_7(data_a_1[c[2]],data_b_1[c[2]],np.linspace(0,100,n)), label='strain', c='purple')
    ax[1][1].plot(np.linspace(0,100,H), strain_7(data_a_1[c[3]],data_b_1[c[3]],np.linspace(0,100,n)), label='strain', c='purple')


df_7_strain = pd.DataFrame(strain_7(data_a_1[0],data_b_1[0],time_mesh)).T
df_7_stress = pd.DataFrame(stress_7(data_a_1[0],data_b_1[0],time_mesh)).T

defer = 0.
for a,b in zip(data_a_1[1:],data_b_1[1:]):
    defer = max(defer, abs(a*b))
    df_7_strain.loc[len(df_7_strain)] = strain_7(a,b,time_mesh)
    df_7_stress.loc[len(df_7_stress)] = stress_7(a,b,time_mesh)


print(f'{df_7_strain.shape = }')
print(f'max_strain = {100 * df_7_strain.max().max():.2f} %  max_defer = {100 * defer:.2f} %')


if 0:
    c = 1
    fig, ax = plt.subplots(1,2)
    ax[0].plot(time_mesh, df_7_strain.loc[c], label='strain', c='purple')
    ax[1].plot(time_mesh, df_7_stress.loc[c], label='stress')
    ax[0].legend();     ax[1].legend();      ax[1].set_xlim(xmin=0)
    print(f'a = {data_a_1[c]}   b = {data_b_1[c]}')

#%%           8  РЕЛАКСАЦИЯ

def strain_8(e, t):
    return e * h(t)

def stress_8(e, t):
    return pm.iloc[0,0]*strain_8(e,t) + e * sum( pm.iloc[i,0] * np.exp(-t/pm.iloc[i,2]) for i in range(1,N+1) )


np.random.seed(6)
data_e = np.random.uniform(1, 20, count_frame[8])*1.e-3

df_8_strain = pd.DataFrame(strain_8(data_e[0], time_mesh)).T
df_8_stress = pd.DataFrame(stress_8(data_e[0], time_mesh)).T

for e in data_e[1:]:
    df_8_strain.loc[len(df_8_strain)] = strain_8(e,time_mesh)
    df_8_stress.loc[len(df_8_stress)] = stress_8(e,time_mesh)

print(f'{df_8_strain.shape = }')
print(f'max_strain_relax = {100 * df_8_strain.max().max():.2f} %')

#%%                  ЗАГРУЗКА ДАННЫХ В EXCEL

path = "C:\\Users\\751\\Documents\\Методические пособия\\Курсовая\\vxod_data_100_30k\\"

# df_0_strain.to_excel(path + 'strain_0.xlsx', index=False)
# df_1_strain.to_excel(path + 'strain_1.xlsx', index=False)
# df_2_strain.to_excel(path + 'strain_2.xlsx', index=False)
# df_3_strain.to_excel(path + 'strain_3.xlsx', index=False)
# df_4_strain.to_excel(path + 'strain_4.xlsx', index=False)
# df_5_strain.to_excel(path + 'strain_5.xlsx', index=False)
# df_6_strain.to_excel(path + 'strain_6.xlsx', index=False)
# df_7_strain.to_excel(path + 'strain_7.xlsx', index=False)
# df_8_strain.to_excel(path + 'strain_8.xlsx', index=False)

# df_0_stress.to_excel(path + 'stress_0.xlsx', index=False)
# df_1_stress.to_excel(path + 'stress_1.xlsx', index=False)
# df_2_stress.to_excel(path + 'stress_2.xlsx', index=False)
# df_3_stress.to_excel(path + 'stress_3.xlsx', index=False)
# df_4_stress.to_excel(path + 'stress_4.xlsx', index=False)
# df_5_stress.to_excel(path + 'stress_5.xlsx', index=False)
# df_6_stress.to_excel(path + 'stress_6.xlsx', index=False)
# df_7_stress.to_excel(path + 'stress_7.xlsx', index=False)
# df_8_stress.to_excel(path + 'stress_8.xlsx', index=False)