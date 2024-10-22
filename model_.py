#%%           DEFAULT
#     %load_ext autotime          %unload_ext autotime
#     print(f"Количество доступных ядер процессора: {os.cpu_count()}")

import autotime
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.utils import Sequence

import keras
import keras_tuner
from keras import layers
from sklearn.model_selection import train_test_split

import seaborn as sns
sns.set_style('whitegrid')

import functions as FU

def print_history(history, start):
    plt.plot(history.history['loss'][start:], label="loss")
    plt.plot(history.history['val_loss'][start:], label='val_loss')
    plt.grid();     plt.legend()

def l1_rel(true_values, estimated_values, count_no=0):
    np.array(true_values).shape = (-1,1);  
    np.array(estimated_values).shape = (-1,1)
    return np.sum(np.abs(true_values[count_no:] - estimated_values[count_no:])) / np.sum(np.abs(true_values))

def l2_rel(true_values, estimated_values, count_no=0):
    true_values.shape = (-1,1);       estimated_values.shape = (-1,1)
    return np.sqrt(np.sum(np.power(true_values[count_no:] - estimated_values[count_no:], 2))) / np.sqrt(np.sum(np.power(true_values, 2)))

def linf_rel(true_values, estimated_values, count_no=0):
    true_values.shape = (-1,1);       estimated_values.shape = (-1,1)
    return np.max(np.abs(true_values[count_no:] - estimated_values[count_no:])) / np.max(np.abs(true_values))


def metrics_err(df_true,df_pred, l=1, mode_out=0, mode_bios=1):
    l1_l2_li = np.empty((0, 3))
    df_pred = pd.DataFrame(df_pred)
    df_true = df_true.reset_index(drop=True)
    
    for (ind_1,true),(ind_2,pred) in zip(df_true.iterrows(), df_pred.iterrows()):
        l1 = l1_rel(np.array(true),np.array(pred),0)
        l2 = l2_rel(np.array(true),np.array(pred),0)
        li = linf_rel(np.array(true),np.array(pred),0)
        
        l1_l2_li = np.vstack([  l1_l2_li, np.array([l1,l2,li])  ])

    if df_true.shape[0] == 1:
       return l1_l2_li

    str_l = {0:'L1_rel',1:'L2_rel',2:'Linf_rel'}
    
    if l==0:
        sorted_l = np.sort(l1_l2_li[:,2])
        cdf = np.arange(1, len(sorted_l) + 1) / len(sorted_l)
        plt.plot(sorted_l, cdf,label='cdf')
        plt.xlim(0,np.median(l1_l2_li,axis=0)[2] * 3)
        plt.title(f'Функция распределения {str_l[2]}')
        
    bios = 2 if mode_bios == 1 else 0    
            
    if l==1:                                                                   # функция распределния и плотность l1_rel
        for i in range(1):
            sorted_l = np.sort(l1_l2_li[:,i])
            cdf = np.arange(1, len(sorted_l) + 1) / len(sorted_l)
            plt.subplot(1 + bios, 2, i + 1 + 2*bios) 
            plt.plot(sorted_l, cdf,label='cdf')
            plt.xlim(0,np.median(l1_l2_li,axis=0)[i] * 3)
            plt.title(f'Функция распределения {str_l[i]}')
        
        plt.subplot(1 + bios, 2, 1 + 1 + 2*bios)
        sns.distplot(l1_l2_li[:,0], hist=True, kde=True, bins=40, hist_kws={'alpha': 1.0})
        plt.xlim(xmin=0, xmax=min(2,np.max(l1_l2_li[:,0])))
        
    if l == 2:
        # plt.figure(figsize=(18, 6))
        plt.subplot(1, 2, 2)
        sns.distplot(l1_l2_li[:,0], hist=True, kde=True, bins=  30, hist_kws={'alpha': 1.0})
        plt.xlabel(r"$\ell_1$", fontsize=16, family='Arial')  # Подпись горизонтальной оси
        plt.ylabel("Плотность", fontsize=16, family='Arial')  # Подпись вертикальной оси
        plt.xlim(xmin=0, xmax=min(2,np.max(l1_l2_li[:,0])))
        plt.tick_params(axis='both', labelsize=14)  # Настройка шрифта делений на осях
        
    if mode_out == 0:
        print('l1_l2_li = ', end='')
        print('  '.join(f'{num:.4f}' for num in l1_l2_li.mean(axis=0)))
        # print(l1_l2_li.mean(axis=0))
        return l1_l2_li.mean(axis=0)
    if mode_out != 0:
        print(l1_l2_li)
        return l1_l2_li

def print_4_graf(y_test,x_test,model,alpha=1):
    plt.figure(figsize=(10, 10)) 
    for num in range(1,5):
        i = np.random.randint(1,y_test.shape[0])
        print(i, end=' ')
        res_real = y_test.iloc[i]
        res_tmp = pd.DataFrame(model.predict(x_test.iloc[i].to_frame().T,verbose=0)).T.ewm(alpha=alpha, adjust=False).mean().T
        res = pd.DataFrame(model.predict(x_test.iloc[i].to_frame().T,verbose=0).reshape(-1,1))
        plt.subplot(2, 2, num)
        
        # plt.plot(time_mesh,res,label='pred')
        plt.plot(time_mesh,res_tmp.to_numpy().reshape(-1,1),label='pred')
        plt.plot(time_mesh,res_real,label='real')
        plt.ylim(ymin=0)
        plt.xlabel("Время (с)", fontsize=12, family='Arial')  # Подпись горизонтальной оси
        plt.ylabel("Напряжение (MPa)", fontsize=12, family='Arial')  # Подпись вертикальной оси
        plt.legend();   
    print()    
      
    
def print_4_graf_castom(y_test,x_test,model,alpha=1):
    np.random.seed(None)
    plt.figure(figsize=(10, 10)) 
    for num in range(1,5):
        i = np.random.randint(1,y_test.shape[0])
        print(i, end=' ')
        res_real = y_test.iloc[i]
        res_tmp = pd.DataFrame(model.predict(x_test.iloc[i].to_frame().T,verbose=0)).T.ewm(alpha=alpha, adjust=False).mean().T
        
        # res = pd.DataFrame(model.predict(x_test.iloc[i].to_frame().T,verbose=0).reshape(-1,1))
        plt.subplot(3, 2, num)
        
        # plt.plot(time_mesh,res,label='pred')
        plt.plot(time_mesh,res_tmp.to_numpy().reshape(-1,1),label='pred')
        plt.plot(time_mesh,res_real,label='real')
        plt.ylim(ymin=0)
        plt.xlabel("Время (с)", fontsize=12, family='Arial')  # Подпись горизонтальной оси
        plt.ylabel("Напряжение (MPa)", fontsize=12, family='Arial')  # Подпись вертикальной оси

        plt.legend();
    print()    
    
def print_graf_castom(y_test,x_test,model,alpha=1):
    plt.figure(figsize=(20, 8)) 
    i = np.random.randint(1,y_test.shape[0])
    print(f'{i = }')
    res_real = y_test.iloc[i]
    res_tmp = pd.DataFrame(model.predict(x_test.iloc[i].to_frame().T, verbose=0)).T.ewm(alpha=alpha, adjust=False).mean().T
    
    # res = pd.DataFrame(model.predict(x_test.iloc[i].to_frame().T,verbose=0).reshape(-1,1))
    plt.subplot(1, 2, 1)
    
    # plt.plot(time_mesh,res,label='pred')
    # plt.text(90, max(res_real) * 0.9, f'alpha = {alpha:.1f}', fontsize=16, 
             # family='Arial', color='black', ha='right', va='top')
    plt.plot(time_mesh,res_tmp.to_numpy().reshape(-1,1),label='pred')
    plt.plot(time_mesh,res_real,label='real')
    plt.xlim(xmin=0,xmax=100)
    # plt.ylim(ymin=0)
    plt.xlabel("Время (с)", fontsize=16, family='Arial')  # Подпись горизонтальной оси
    plt.ylabel("Напряжение (MPa)", fontsize=16, family='Arial')  # Подпись вертикальной оси
    plt.tick_params(axis='both', labelsize=14)  # Настройка шрифта делений на осях
    plt.legend();        
        
def other_test(num, model, saving_name, alpha=1, from_=5000):

    df_strain_test = pd.read_excel(path + ph_strain[num]).loc[from_:]
    df_stress_test = pd.read_excel(path + ph_stress[num]).loc[from_:]

    print(f'{alpha = :.1f}', end='    ')

    # print(f'{df_strain_test.shape = }')
    # print(f'{df_stress_test.shape = }')

    # print_4_graf(df_stress_test, df_strain_test, model, alpha);    plt.show()
    
    print_graf_castom(df_stress_test, df_strain_test, model, alpha)

    t = pd.DataFrame(model.predict(df_strain_test,verbose=0)).T.ewm(alpha=alpha, adjust=False).mean().T

    metrics_err(df_stress_test, t,l=2)

    plt.savefig(path_to_save_grafs + saving_name + '.png', dpi=200, bbox_inches='tight')

def save_results(df_stress, df_strain, model, saving_name,  alpha=1):
    print_graf_castom(df_stress, df_strain, model, alpha)
    t = pd.DataFrame(model.predict(df_strain, verbose=0)).T.ewm(alpha=alpha, adjust=False).mean().T
    tt = metrics_err(df_stress, t, l=2)
    plt.savefig(path_to_save_grafs + saving_name + '.png', dpi=200, bbox_inches='tight')
    return tt


param_data = np.array([[69, np.NAN, np.NAN],
                      [10.88, 1.088, 0.1],
                      [9.009, 9.009, 1],
                      [5.388, 53.88, 10],
                      [5.975, 597.5, 1.e2],
                      [10.4, 10400, 1.e3]])

pm = pd.DataFrame(param_data, columns=['E', 'nu', 'z'])
path = 'C:\\Users\\751\\Documents\\Методические пособия\\Курсовая\\vxod_data_100_30k\\'
N = 100

#%%                     ЗАГРУЗКА ДАННЫХ  

N = 100
time_mesh = pd.Series(np.linspace(0, 100, N))

path = 'C:\\Users\\751\\Documents\\Методические пособия\\Курсовая\\vxod_data_100_30k\\'
path_to_save_grafs = 'C:\\Users\\751\\Documents\\Методические пособия\\Курсовая\\Графики\\res\\'

ph_strain = {0: 'strain_0.xlsx', 1: 'strain_1.xlsx', 2: 'strain_2.xlsx', 3: 'strain_3.xlsx', 4: 'strain_4.xlsx', 5: 'strain_5.xlsx', 6: 'strain_6.xlsx', 7:'strain_7.xlsx', 8:'strain_8.xlsx'}
ph_stress = {0: 'stress_0.xlsx', 1: 'stress_1.xlsx', 2: 'stress_2.xlsx', 3: 'stress_3.xlsx', 4: 'stress_4.xlsx', 5: 'stress_5.xlsx', 6: 'stress_6.xlsx', 7:'stress_7.xlsx', 8:'stress_8.xlsx'}

data_exp = {0:'лин', 1:'ремп лин', 2:'ремп квад', 3:'квад', 4:'куб', 5:'ступень', 6:'синус', 7:'пила', 8:'релакс'}

# count_frame = {0:10_000, 1:10_000,  2:10_000,  3:10,  4:10,  5:0,  6:0,  7:0, 8:0}
count_frame = {0:0, 1:0,  2:0,  3:0,  4:0,  5:0,  6:10000,  7:0, 8:0}

df_list_0 = [pd.read_excel(path + ph_strain[i], nrows=count_frame[i], dtype='float32') for i in count_frame]
df_list_1 = [pd.read_excel(path + ph_stress[i], nrows=count_frame[i], dtype='float32') for i in count_frame]

df_strain = pd.concat(df_list_0, ignore_index=True)
df_stress = pd.concat(df_list_1, ignore_index=True)

#                    SHAFFLE DATA
mult = 1
df_strain, df_stress = FU.shaffle_df(df_strain, df_stress, count_frame, mult=mult)
equal_list = FU.df_is_equal(df_strain, df_stress)

#%%                     РАЗБИЕНИЕ ДАННЫХ (70,10,20)%

x_train, x_val, y_train, y_val = train_test_split(df_strain, df_stress, test_size=0.15, random_state=1)

#%%                     KERAS TUNER

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1.e-3, patience=2, verbose=0,
    start_from_epoch=2, restore_best_weights=True)
]

def build_model(hp):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=(N,)))    
    
    for i in range(hp.Int('layers', 4, 8)):
        model.add(layers.Dense(
            units=hp.Int('units_' + str(i), 32, 512, step=60),
            activation=hp.Choice('activ_' + str(i), ['relu', 'tanh', 'elu', 'sigmoid'])
        ))
        
    model.add(layers.Dense(N, activation='linear'))
    
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = hp.Float('lr', min_value=1e-4, max_value=1e-3, sampling='log')),
                  loss = 'mse', metrics=['mae', 'mape'])    
    return model

tuner = keras_tuner.RandomSearch(build_model, objective='val_loss', max_trials=300,
                                 project_name='my_models_1',executions_per_trial=1)

# tuner.reload()
# tuner.search_space_summary()

tuner.search(x_train, y_train, batch_size = 96, epochs = 80, validation_data = (x_val, y_val),
                callbacks = callbacks,verbose=1)


count_best_models = 3
tuner.results_summary(num_trials = count_best_models);     # print()
num_completed_trials = len(tuner.oracle.trials)
print(f'\n\nКоличество завершенных испытаний: {num_completed_trials}')


#%%                     МОДЕЛЬ KERAS
"""
Hyperparameters: 100
layers: 4
units_0: 392
activ_0: tanh
units_1: 512
activ_1: tanh
units_2: 392
activ_2: relu
units_3: 332
activ_3: relu
learning_rate: 0.0008648556710911036
units_4: 152
activ_4: tanh
units_5: 392
activ_5: elu
units_6: 332
activ_6: sigmoid
units_7: 392
activ_7: sigmoid
Score: 0.004949990194290876

layers: 4
units_0: 96
activ_0: tanh
units_1: 32
activ_1: relu
learning_rate: 0.0008733327303608299
units_2: 128
activ_2: elu
units_3: 96
activ_3: elu
units_4: 128
activ_4: tanh
units_5: 16
activ_5: elu
Score: 105.776 val_mape
"""

def build_model_2():                                                           # 100 points
    model = keras.Sequential(
        [
            layers.Dense(392, activation='tanh', input_shape = (N,), name="input",kernel_initializer='glorot_uniform'),
            layers.Dense(512, 'tanh', name="hidden_layer_1",      kernel_initializer='glorot_uniform'),
            layers.Dense(392, 'relu', name="hidden_layer_2",      kernel_initializer='he_normal'),
            layers.Dense(332, "relu", name="hidden_layer_3",      kernel_initializer='he_normal'),     
            layers.Dense(152, "tanh", name="hidden_leaf_4",       kernel_initializer='glorot_uniform'),     
            layers.Dense(392, "elu", name="hidden_layer_5",       kernel_initializer='he_normal'),     
            layers.Dense(332, "sigmoid", name="hidden_layer_6",   kernel_initializer='glorot_uniform'),     
            layers.Dense(392, "sigmoid", name="hidden_layer_7",   kernel_initializer='glorot_uniform'),     
            
            layers.Dense(N, activation="linear", name="output"),
        ]
    )
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.00086485), loss = 'mse', metrics = ['mae', 'mape'])
    return model

def build_model_1():
    model = keras.Sequential(
        [
            layers.Dense(96, activation='tanh', input_shape = (N,), name="input"),
            layers.Dense(64, 'relu', name="hiden_layer_1"),
            layers.Dense(128, 'elu', name="hiden_layer_2"),
            layers.Dense(96, "elu", name="hiden_layer3"),     
            layers.Dense(128, "tanh", name="hiden_layer4"),     
            layers.Dense(32, "elu", name="hiden_layer5"),     
            
            layers.Dense(N, activation="linear", name="output"),
        ]
    )
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.00087), loss = 'mse', metrics = ['mae', 'mape'])
    return model

class DataGenerator(Sequence):
    def __init__(self, data, labels, batch_size=128, shuffle=False):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch_data = self.data.iloc[index * self.batch_size:(index + 1) * self.batch_size].values
        batch_labels = self.labels.iloc[index * self.batch_size:(index + 1) * self.batch_size].values
        return batch_data, batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            indices = np.arange(len(self.data))
            np.random.shuffle(indices)
            self.data = self.data.iloc[indices].reset_index(drop=True)
            self.labels = self.labels.iloc[indices].reset_index(drop=True)

# train_generator = DataGenerator(x_train, y_train, batch_size=64)
# val_generator = DataGenerator(x_val, y_val, batch_size=16)

#%%                     FIT MODEL

model = build_model_2()

# history = model.fit(train_generator, epochs = 150, validation_data = val_generator, workers=16, use_multiprocessing=False, verbose=0)

history = model.fit(x_train, y_train, batch_size = 256, epochs = 150, validation_data = (x_val, y_val), verbose=0)

#%%                     L1_accuracy ИНИЦИАЛИЗАЦИЯ

L1_accuracy = pd.DataFrame([mult], columns=['mult'])

#%%                     ГРАФИКИ ДЛЯ SAME ВЫБОРКИ 

# print_4_graf_castom(y_test, x_test, model)
# print(f'{x_test.shape = }')
# print('metric_same ==>',end=' ');        metrics_err(y_test, model.predict(x_test,verbose=0));  
# plt.savefig(path_to_save_grafs + 'sssssss.png', dpi=200, bbox_inches='tight')

#%%                     ГРАФИКИ ДЛЯ OTHER  

# other_test(3, model, '0-', from_=27000, alpha=1)
# other_test(3, model, 'ewm 0.6 0-', from_=27000, alpha=0.6)


other_test(4, model, '1-', from_=27_000, alpha=1)
other_test(4, model, 'ewm 0.6 1-', from_=27_000, alpha=0.6)


#%%                     comparison function

def strain_proiz(a,b,t):
    return a*b * np.cos(b*t)

def find_a_b(row, t):
    y = np.array(row)
    if y.ndim != 1:
        print(y.ndim)
    fft_result = np.fft.fft(y)
    frequencies = np.fft.fftfreq(len(y), t[1] - t[0])
    peak_index = np.argmax(np.abs(fft_result))
    dominant_frequency = np.abs(frequencies[peak_index])
    b = 2 * np.pi * dominant_frequency
    a = y.max()
    return a,b

def find_closest_position(num, array):
    array = np.array(array)
    differences = np.abs(array - num)
    closest_index = np.argmin(differences)
    return closest_index

def stress_g(strain, time_mesh, b_G1_G2_delta):
    def calculate_stress(row):
        a, b = find_a_b(row, time_mesh)
        if b==0:
            b_min = b_G1_G2_delta.iloc[:,0].min();
            b += b_min
        i = find_closest_position(b, b_G1_G2_delta.iloc[:, 0])
        G1 = b_G1_G2_delta.iloc[i, 1]
        G2 = b_G1_G2_delta.iloc[i, 2]
        return G1 * row + G2 / b * strain_proiz(a, b, time_mesh)
    
    if isinstance(strain, pd.DataFrame):
        return strain.apply(calculate_stress, axis=1)
    
    return calculate_stress(strain)

#%%                     LOADING df_test

df_strain_test = pd.read_excel(path + ph_strain[6], dtype='float32').loc[27000:]
df_strain_test = df_strain_test.reset_index(drop=True)

df_stress_test = pd.read_excel(path + ph_stress[6], dtype='float32').loc[27000:]
df_stress_test = df_stress_test.reset_index(drop=True)

#%%

b_G1_G2_delta = pd.read_excel("C:\\Users\\751\\Documents\\Методические пособия\\Курсовая\\vxod_data_100_30k\\b_G1_G2_delta_3.xlsx")

#%%                     попытка с помощью b_G1_G2_delta улучшить предсказаняи

def plot_comparison(df1, df2):
    if df1.shape != df2.shape:
        raise ValueError("Оба датафрейма должны быть одинакового размера")

    time_mesh = pd.Series(np.linspace(0, 100, 100))
    
    if df1.shape[0] < 4:
        raise ValueError("Датафреймы должны иметь как минимум 4 строки")
        
    random_indices = np.random.choice(df1.index, size=4, replace=False)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    
    for i, idx in enumerate(random_indices):
        y1 = df1.iloc[idx]
        y2 = df2.iloc[idx]

        axs[i].plot(time_mesh, y1, label=f'DataFrame 1 - Row {idx}', color='blue')
        axs[i].plot(time_mesh, y2, label=f'DataFrame 2 - Row {idx}', color='orange')
        
        axs[i].set_xlabel('Time')
        axs[i].legend()
        axs[i].grid(True)
    
    plt.tight_layout()
    plt.show()

# b_min = b_G1_G2_delta.iloc[:,0].min();      

def print_graf_castom_(y_test,pred,alpha=1):
    plt.figure(figsize=(20, 8)) 
    i = np.random.randint(1,y_test.shape[0])
    print(f'{i = }')
    i = 1387
    res_real = y_test.iloc[i]
    plt.subplot(1, 2, 1)
    
    # plt.plot(time_mesh,res,label='pred')
    # plt.text(90, max(res_real) * 0.9, f'alpha = {alpha:.1f}', fontsize=16, 
             # family='Arial', color='black', ha='right', va='top')
    plt.plot(time_mesh,pred.iloc[i],label='pred')
    plt.plot(time_mesh,res_real,label='real')
    plt.xlim(xmin=0,xmax=100)
    # plt.ylim(ymin=0)
    plt.xlabel("Время (с)", fontsize=16, family='Arial')  # Подпись горизонтальной оси
    plt.ylabel("Напряжение (MPa)", fontsize=16, family='Arial')  # Подпись вертикальной оси
    plt.tick_params(axis='both', labelsize=14)  # Настройка шрифта делений на осях
    plt.legend();      

pred = pd.DataFrame(model.predict(df_strain_test, verbose=0))
df_stress_g = stress_g(df_strain_test, time_mesh, b_G1_G2_delta)
df_stress_g = df_stress_g.reset_index(drop=True)

total_nan = df_stress_g.isna().sum().sum();    

print_graf_castom_(df_stress_test, df_stress_g)
metrics_err(df_stress_test, df_stress_g,l=2);   print('b_G\n')

# plt.savefig(path_to_save_grafs + 'Dma.png', dpi=200, bbox_inches='tight')
# plot_comparison(df_stress_test, df_stress_g)

# metrics_err(df_stress_test, (df_stress_g + pred)/2, l=2);   print('semisum\n')
# plt.show(block=False)
# plot_comparison(df_stress_test, (df_stress_g + pred)/2)

print_graf_castom(df_stress_test, df_strain_test, model)
metrics_err(df_stress_test, pd.DataFrame(model.predict(df_strain_test)), l=2);  print('pred')
plt.savefig(path_to_save_grafs + 'Dma_nn.png', dpi=200, bbox_inches='tight')
# plt.show(block=False)
# plot_comparison(df_stress_test, pd.DataFrame(model.predict(df_strain_test)))
                 

#%%                     ЛИНЕЙНАЯ + КВАДРАТИЧНАЯ

alpha_0 = 0.6
count_0 = 3000
np.random.seed(14);     data_k = np.random.uniform(1, 20, count_0) * 1.e-5 / np.sqrt(2)
np.random.seed(5);      data_al_0 = np.random.uniform(1, 20, count_0)*1.e-7 / np.sqrt(2)

df_0_strain = pd.DataFrame([FU.strain_0(k, time_mesh) + FU.strain_3(al, time_mesh) for k, al in zip(data_k, data_al_0)])
df_0_stress = pd.DataFrame([FU.stress_0(k, time_mesh) + FU.stress_3(al, time_mesh) for k, al in zip(data_k, data_al_0)])

t = save_results(df_0_stress, df_0_strain, model, 'lin+kvad 1', alpha=alpha_0)
L1_accuracy['лин+ква'] = t[0]

print_4_graf_castom(df_0_stress, df_0_strain, model)


#%%                     ЛИНЕЙНАЯ + КУБИЧЕСКАЯ

alpha_1 = 0.6
count_0 = 3000
np.random.seed(1);      data_k = np.random.uniform(1, 20, count_0) * 1.e-5 / np.sqrt(2)
np.random.seed(7);      data_al_1 = np.random.uniform(1, 20, count_0)*1.e-9 / np.sqrt(2)

df_1_strain = pd.DataFrame([FU.strain_0(k, time_mesh) + FU.strain_4(al, time_mesh) for k, al in zip(data_k, data_al_1)])
df_1_stress = pd.DataFrame([FU.stress_0(k, time_mesh) + FU.stress_4(al, time_mesh) for k, al in zip(data_k, data_al_1)])

t = save_results(df_1_stress, df_1_strain, model, 'lin+kub 1', alpha=alpha_1)
L1_accuracy['лин+куб'] = t[0]

print_4_graf_castom(df_1_stress, df_1_strain, model)

#%%                     ЛИНЕЙНАЯ + КВАДРАТИЧНАЯ + КУБИЧЕСКАЯ

alpha_2 = 0.6
count_0 = 3000
np.random.seed(22);       data_k = np.random.uniform(1, 20, count_0) * 1.e-5 
np.random.seed(91);       data_al_2 = np.random.uniform(1, 20, count_0)*1.e-7 
np.random.seed(712);      data_al_3 = np.random.uniform(1, 20, count_0)*1.e-9 

df_2_strain = pd.DataFrame([(FU.strain_0(k, time_mesh) + FU.strain_3(al_1, time_mesh) + FU.strain_4(al_2, time_mesh))/3.0 for k, al_1, al_2 in zip(data_k, data_al_2, data_al_3)])
df_2_stress = pd.DataFrame([(FU.stress_0(k, time_mesh) + FU.stress_3(al_1, time_mesh) + FU.stress_4(al_2, time_mesh))/3.0 for k, al_1, al_2 in zip(data_k, data_al_2, data_al_3)])

t = save_results(df_2_stress, df_2_strain, model, 'lin+kvad+kub 1', alpha=alpha_2)

L1_accuracy['лин+ква+куб'] = t[0]

print_4_graf_castom(df_2_stress, df_2_strain, model)

#%%                     КВАДРАТИЧНАЯ + КУБИЧЕСКАЯ

alpha_3 = 0.6
count_0 = 3000
np.random.seed(5);      data_al_4 = np.random.uniform(1, 20, count_0)*1.e-7 / np.sqrt(2)
np.random.seed(9);      data_al_5 = np.random.uniform(1, 20, count_0)*1.e-9 / np.sqrt(2)

df_3_strain = pd.DataFrame([FU.strain_3(al_1, time_mesh) + FU.strain_4(al_2, time_mesh) for al_1, al_2 in zip(data_al_4, data_al_5)])
df_3_stress = pd.DataFrame([FU.stress_3(al_1, time_mesh) + FU.stress_4(al_2, time_mesh) for al_1, al_2 in zip(data_al_4, data_al_5)])

t = save_results(df_3_stress, df_3_strain, model, 'kvad+kub 1', alpha=alpha_3)
L1_accuracy['ква+куб'] = t[0]

print_4_graf_castom(df_3_stress, df_3_strain, model)

#%%                     ЛИНЕЙНАЯ + SINUS

alpha_0 = 1
count_0 = 3000
np.random.seed(14);     data_k = np.random.uniform(1, 20, count_0) * 1.e-5 
np.random.seed(2);      data_a_0 = np.random.uniform(5, 20, count_0)*1.e-3 
np.random.seed(57);     data_b_0 = np.random.uniform(1, 20, count_0)*1.e-2 * 11/4

df_4_strain = pd.DataFrame([(FU.strain_0(k, time_mesh) + FU.strain_6(a, b, time_mesh))/2 for k, a, b in zip(data_k, data_a_0, data_b_0)])
df_4_stress = pd.DataFrame([(FU.stress_0(k, time_mesh) + FU.stress_6(a, b, time_mesh))/2 for k, a, b in zip(data_k, data_a_0, data_b_0)])
print(f'{FU.epsilon_max(df_4_strain) = }')

for i in [1, 0.8, 0.6, 0.3]:
    print(f'{i:.1f}', end='    ')
    t = save_results(df_4_stress, df_4_strain, model, 'lin+sin 0', alpha=i)


# L1_accuracy['лин+sin'] = t[0]

# print_4_graf_castom(df_4_stress, df_4_strain, model)

#%%                     PRINT L1_accuracy

print(L1_accuracy.to_string(index=False, header=False))
