import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras


param_data = np.array([[69, np.NAN, np.NAN],
                      [10.88, 1.088, 0.1],
                      [9.009, 9.009, 1],
                      [5.388, 53.88, 10],
                      [5.975, 597.5, 1.e2],
                      [10.4, 10400, 1.e3]])

pm = pd.DataFrame(param_data, columns=['E', 'nu', 'z'])

N = 5                                                                          # число членов в ряде прони
time_steps = 100;    time_end = 100

time_mesh = pd.Series(np.linspace(0, time_end, time_steps))

#====================================================================================================================

def h(t):
    if isinstance(t, pd.Series):
        return t.apply(lambda x: 1. if x >= 0 else 0.)
    else:
        return 1. if t >= 0 else 0.

def strain_0(k, t):
    return k*t

def stress_0(k, t):
    return pm.iloc[0,0]*strain_0(k, t) + sum( k*pm.iloc[i,1] * (1-np.exp(-t/pm.iloc[i,2])) for i in range(1,N+1) )


def strain_1(k, t0, t):
    return k*t*(h(t) - h(t-t0)) + k*t0*h(t-t0)

def stress_1(k, t0, t):
    return pm.iloc[0,0]*strain_1(k,t0,t) + k * sum(pm.iloc[i,1] * (np.exp(-(t-t0)/pm.iloc[i,2]) - np.exp(-t/pm.iloc[i,2]))*h(t-t0) +
                                                   pm.iloc[i,1] * (1 - np.exp(-t/pm.iloc[i,2]))*h(t0-t) for i in range(1,N+1) )

def fun_2(i,t0,t):
    return ( 
             h(t-t0)*( np.exp(-t/pm.iloc[i,2]) + (t0/pm.iloc[i,2] - 1)*np.exp(-(t-t0)/pm.iloc[i,2]) ) +
             h(t0-t)*(t/pm.iloc[i,2] - 1 + np.exp(-t/pm.iloc[i,2]))
           )     
            
def strain_2(k, t0, t):
    return k*t*t*(h(t) - h(t-t0)) + k*t0*t0*h(t-t0)

def stress_2(k, t0, t):
    return  pm.iloc[0,0]*strain_2(k,t0,t) + 2*k * sum( pm.iloc[i,1]*pm.iloc[i,2] * fun_2(i,t0,t) for i in range(1, N+1) )

def fun_3(i,t):
    return pm.iloc[i,0] * pm.iloc[i,2]**2 * ( t/pm.iloc[i,2] - 1 + np.exp(-t/pm.iloc[i,2]) )

def strain_3(al, t):
    return al*t*t

def stress_3(al, t):
    return pm.iloc[0,0]*strain_3(al,t) + 2*al * sum( fun_3(i,t) for i in range(1,N+1) )

def fun_4(i,t):
    return pm.iloc[i,0] * pm.iloc[i,2]**3 * ( t/pm.iloc[i,2] * (t/pm.iloc[i,2] - 2) - 2*np.exp(-t/pm.iloc[i,2]) + 2 )

def strain_4(al, t):
    return al * t**3

def stress_4(al, t):
    return pm.iloc[0,0]*strain_4(al,t) + 3*al * sum( fun_4(i,t) for i in range(1,N+1) )

def fun_5(i,s1,s2,t0,t):
    return s1*pm.iloc[i,0]*np.exp(-t/pm.iloc[i,2]) * h(t) + (s2-s1)*pm.iloc[i,0]*np.exp(-(t-t0)/pm.iloc[i,2]) * h(t-t0)

def strain_5(s1,s2,t0,t):
    return s1*h(t) + (s2-s1)*h(t-t0)

def stress_5(s1,s2,t0,t):
    return pm.iloc[0,0]*strain_5(s1,s2,t0,t) + sum( fun_5(i,s1,s2,t0,t) for i in range(1,N+1) )

def const_6(i,b):
    return pm.iloc[i,1] / (b**2 * pm.iloc[i,2]**2 + 1)

def fun_6(i,a,b,t):
    return np.cos(b*t) + b* pm.iloc[i,2] *np.sin(b*t) - np.exp(-t/pm.iloc[i,2])

def strain_6(a,b,t):
    return a * np.sin(b*t)

def stress_6(a,b,t):
    return pm.iloc[0,0]*strain_6(a,b,t) + a*b * sum( const_6(i,b) * fun_6(i,a,b,t) for i in range(1,N+1) )

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

def strain_8(e, t):
    return e * h(t)

def stress_8(e, t):
    return pm.iloc[0,0]*strain_8(e,t) + e * sum( pm.iloc[i,0] * np.exp(-t/pm.iloc[i,2]) for i in range(1,N+1) )

def exponential_smoothing(df, alpha):
    smoothed_df = pd.DataFrame(index=df.index, columns=df.columns)
    
    for idx in df.index:
        smoothed_df.loc[idx] = df.loc[idx].ewm(alpha=alpha).mean()
    
    return smoothed_df

def non_zero_item(count_frame):
    non_zero_elements = [0]        
    for key, value in count_frame.items():
        if value != 0:  # Проверяем, является ли значение ненулевым
            non_zero_elements.append(value + non_zero_elements[-1]) 
    return non_zero_elements         

def shaffle_df(df_strain, df_stress, count_frame, mult=2):
    
    shape_list = non_zero_item(count_frame)                       # начиная с нуля индексы в итоговом df 
    shape_data = df_strain.shape[0]

    num_new_rows = int((mult-1) * shape_data)                     # Количество новых строк, которые нужно добавить
    new_rows_strain = [];    new_rows_stress = []

    for k in range(num_new_rows):

        random_indices = [int(np.random.randint(shape_list[i], shape_list[i+1], 1)) for i in range(len(shape_list)-1)]
            
        random_rows_strain = df_strain.loc[random_indices]
        random_rows_stress = df_stress.loc[random_indices]
        
        new_rows_strain.append(random_rows_strain.sum() / (len(shape_list)-1))
        new_rows_stress.append(random_rows_stress.sum() / (len(shape_list)-1))

    new_df_strain = pd.DataFrame(new_rows_strain)
    new_df_stress = pd.DataFrame(new_rows_stress)
    
    
    epsilon_max_row = 0
    for row in new_df_strain.itertuples(index=False):
        epsilon_max_row = max(np.array(row).max(), epsilon_max_row)
    
    if epsilon_max_row >0.02:   
        print(f'ERROR  {epsilon_max_row = :.3f}')
  
    epsilon_derivative_max_row = 0
    for row in new_df_strain.itertuples(index=False): 
        epsilon_derivative_max_row = max(epsilon_derivative_max_row, max(np.abs(row[i-1] - 2*row[i] + row[i+1]) for i in range(1, len(row)-1)))
    
    if epsilon_derivative_max_row > 0.01:
        print(f'ERROR  {epsilon_derivative_max_row = }')
        
    df_strain = pd.concat([df_strain, new_df_strain], ignore_index=True)
    df_stress = pd.concat([df_stress, new_df_stress], ignore_index=True)
    
    return df_strain, df_stress

def df_is_equal(df1, df2):
    comparison = df1 == df2
    equal_rows = comparison.all(axis=1)
    
    equal_list = []
    for i, is_equal in enumerate(equal_rows):
        if is_equal:
            equal_list.append(i)
    if len(equal_list) > 0:
        print('ERROR_SAME_ROWS')
        return equal_list        
    return 0

def epsilon_max(df):
    epsilon_max_row = 0
    for row in df.itertuples(index=False):
        epsilon_max_row = max(np.array(row).max(), epsilon_max_row)
    return epsilon_max_row
