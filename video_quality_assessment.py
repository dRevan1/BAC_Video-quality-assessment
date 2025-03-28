import csv
import tensorflow as tf
import keras as keras
from keras import Model, layers, callbacks, Input
import numpy as num
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plotter

num.set_printoptions(suppress=True)
scene_list = []
resolution_list = [] # HD, FHD, UHD - 720p, 1080p, 2160p
bitrate_list = [] #[-1] je posledný prvok
codec_list = [] #a.insert(0, 5) je 5 na index 0
packet_loss_list = []
ssim_list = [] # SSIM je jedna z objektívnych metrík - Structural similarity index measure
vmaf_list = [] # VMAF je jedna z objektívnych metrík - Video Multimethod Assessment Fusion
labels_list = []


def scene_switch(scene_name):
    switch = {
        "Campfire": 1,
        "Construction": 2,
        "Runners": 3,
        "Rush": 4,
        "Tall": 5,
        "Wood": 6
    }
    return switch[scene_name]

def resolution_switch(res_name):
    switch = {
        "HD": 720,
        "FHD": 1080,
        "FullHD": 1080,
        "UHD": 2160
    }
    return switch[res_name]

def get_packet_loss(split_line):
    loss_number = 1.0

    if len(split_line) == 1:
        return 0.0

    if len(split_line) != 2:
        loss_number = float(split_line[2])
        while loss_number >= 1.0:
            loss_number /= 10

    return loss_number

def get_resolution(video_name):
    resolutions = ["FullHD", "FHD", "UHD", "HD"]
    d = dict()
    d["delimiter"] = "error_res"
    d["res"] = "error_res"

    for res in resolutions:
        if res in video_name:
            d["delimiter"] = res
            d["res"] = resolution_switch(res)
            break
    
    return d

def train_network_configuration_test(neurons_list, activation_function, x_train, y_train, x_test, y_test, training_results):
    input_layer = Input(shape=(4,))
    layer_list = []

    layer_list.append(layers.Dense(neurons_list[0], activation=activation_function)(input_layer))
    for i in range(1, len(neurons_list)):
        layer_list.append(layers.Dense(neurons_list[i], activation=activation_function)(layer_list[-1]))
    output_layer = layers.Dense(1, activation='linear')(layer_list[-1])

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test), callbacks=[early_stop])
    
    test_loss = model.evaluate(x_test, y_test)
    training_results.append([neurons_list, test_loss])
    


def get_network_configurations_result(x_train, y_train, x_test, y_test):    
    network_configs = [
        [256, 128, 64, 32, 16],
        [128, 96, 64, 32, 16],
        [128, 64, 32, 16, 8],
        [100, 90, 80, 70, 60],
        [100, 95, 90, 70, 25],
        [100, 80, 50, 20, 10],
        [90, 80, 50, 20, 10],
        [80, 70, 65, 60, 25],
        [70, 65, 65, 45, 20],
        [70, 60, 50, 40, 30],
        [70, 50, 40, 30, 20],
        [256, 128, 64, 32],
        [128, 96, 64, 32],
        [128, 64, 32, 16],
        [100, 90, 80, 70],
        [100, 80, 50, 20],
        [100, 80, 50, 10],
        [100, 80, 40, 20],
        [100, 80, 30, 20],
        [100, 80, 20, 10],
        [100, 70, 60, 50],
        [100, 70, 60, 40],
        [100, 70, 50, 40],
        [100, 70, 50, 30],
        [100, 70, 40, 20],
        [100, 70, 30, 10],
        [95, 85, 80, 75],
        [90, 75, 75, 32],
        [90, 75, 70, 60],
        [90, 75, 64, 32],
        [256, 128, 64],
        [158, 80, 64],
        [158, 80, 32],
        [158, 64, 32],
        [158, 64, 16],
        [128, 96, 64],
        [128, 64, 32],
        [100, 90, 20],
        [80, 70, 60],
        [80, 70, 50],
        [80, 70, 40],
        [80, 60, 40],
        [80, 60, 30],
        [80, 50, 20],
        [80, 40, 20],
        [80, 30, 20],
        [80, 20, 10],
        [70, 60, 50],
        [70, 60, 40],
        [70, 50, 40],
        [70, 50, 30],
        [70, 40, 20],
        [70, 30, 10],
    ]
    results = []
    experiment_results = []
    
    for i in range(15):
        for config in network_configs:
            train_network_configuration_test(config, 'relu', x_train, y_train, x_test, y_test, results)
        
        results.sort(key=lambda x: x[1])
        experiment_results.append(results[0])    
        results.clear()
    
    for result in experiment_results:
        print(f"Neurons: {result[0]}")
        print(f"\nFinal test loss: {result[1]}")
        print("-----------------------------------------------------")
    
    
    
def print_training_result(model, x_train, y_train, x_test, y_test):
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)    
    history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test), callbacks=[early_stop])

    print("Training history:")
    for epoch in range(len(history.history['loss'])):
        print(f"Epoch {epoch+1}:")
        print(f"  Training loss: {history.history['loss'][epoch]}")
        print(f"  Validation loss: {history.history['val_loss'][epoch]}")
        print(f"  Training MSE: {history.history['mse'][epoch]}")
        print(f"  Validation MSE: {history.history['val_mse'][epoch]}")
    
    test_loss, test_mse = model.evaluate(x_test, y_test)
    print(f"\nFinal test loss: {test_loss}")
    print(f"Final test MSE: {test_mse}")

    plotter.plot(history.history['loss'], label='Train Loss')
    plotter.plot(history.history['val_loss'], label='Val Loss')
    plotter.title('Model Loss (MSE)')
    plotter.xlabel('Epochs')
    plotter.ylabel('Loss')
    plotter.legend(loc='upper left')
    plotter.show()
    

with open("first_session.csv", mode = "r") as file:  #with closene rovno file = nice
    first_session = csv.reader(file, delimiter = ";")
    i = 0 #celkovo je 502 záznamov vo first_session
    codec = 0 #bude 0/1 podľa kodeku (h264, h265 v poradí)
    parsed_line = []

    for line in first_session:   #chceme index 0 - slovo sa rozdelý na inputy do ML modelu, index 64 - priemerné ohodnotenie videa (1-5)
        parameters = line[0]

        if "Mbit" in parameters:
            parsed_line = parameters.split('/')
            codec = 0 if parsed_line[1].split(' ')[0] == "H264" else 1
            parsed_line = parameters.split(' ')
        else:
            if i > 1:
                label = float(line[64].strip())      #na indexe 64 je MOS, 63 VMAF a 62 SSIM výsledky
                vmaf = float(line[63].strip())
                ssim = float(line[62].strip())
                labels_list.append(label)
                vmaf_list.append(vmaf)
                ssim_list.append(ssim)

                scene_list.append(scene_switch(parsed_line[0]))
                resolution_list.append(resolution_switch(parsed_line[1]))
                bitrate_list.append(int(parsed_line[2]))
                codec_list.append(codec)
                packet_loss_list.append(get_packet_loss(parameters.split('_')))
                
                #print("{} {} {} {} {} {} {} {}".format(scene_list[-1], resolution_list[-1], bitrate_list[-1], codec_list[-1], ssim, vmaf, label, packet_loss_list[-1]))
        i += 1


#načítanie second session
with open("second_session.csv", mode = "r") as file:  #with closene rovno file = nice
    second_session = csv.reader(file, delimiter = ";")
    i = 0 #celkovo je 502 záznamov vo first_session
    codec = 0 #bude 0/1 podľa kodeku (h264, h265 v poradí)
    parsed_line = []

    for line in second_session:   #chceme index 0 - slovo sa rozdelý na inputy do ML modelu, index 64 - priemerné ohodnotenie videa (1-5)
        parameters = line[0]

        if "Subjective" in parameters:
            break

        if "HEVC" in parameters:
            codec = 1
        elif len(line[1]) == 0:
            codec = 0
        else:
            if i > 1:
                label = float(line[47].strip().replace(',', '.'))
                vmaf = float(line[52].strip())
                ssim = float(line[51].strip())
                labels_list.append(label)
                vmaf_list.append(vmaf)
                ssim_list.append(ssim)
                
                resolution = get_resolution(parameters)
        
                if codec == 1:
                    resolution["delimiter"] = "{}{}".format("X", resolution["delimiter"])
                    
                parsed_line = parameters.split(resolution["delimiter"])
                scene = parsed_line[0][:-2] if (parsed_line[0][-1] == 'X') else parsed_line[0][:-1]
                bitrate_PL = parsed_line[1].split('_')
                scene_list.append(scene_switch(scene))
                resolution_list.append(resolution["res"])
                bitrate_list.append(bitrate_PL[0])
                codec_list.append(codec)
                packet_loss_list.append(get_packet_loss(bitrate_PL))
                
                #print("{} {} {} {} {} {} {} {}".format(scene_list[-1], resolution_list[-1], bitrate_list[-1], codec_list[-1], ssim, vmaf, label, packet_loss_list[-1]))

        i += 1


#tu začína PCA
#príprava dát do 2 kommponentov (X, Y)
x_list = []
features = ["Scene", "Resolution", "Bitrate", "Codec", "Packet loss", "SSIM", "VMAF"]
for i in range(len(labels_list)):
    x_list.append([scene_list[i], resolution_list[i], bitrate_list[i], codec_list[i], packet_loss_list[i], ssim_list[i], vmaf_list[i]])

X = num.array(x_list)
Y = num.array(labels_list)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train = X_train.astype(num.float64)

Scaler = StandardScaler()
X_train = Scaler.fit_transform(X_train)
X_test = Scaler.transform(X_test)
pca = PCA(n_components=4)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

temp = []
activation_functions = ['relu', 'elu', 'selu', 'sigmoid', 'tanh', 'swish', 'softplus']
for activation in activation_functions:
    train_network_configuration_test([256, 128, 64], activation, X_train, Y_train, X_test, Y_test, temp)
    
for i in range(len(activation_functions)):
    print(f"Activation function: {activation_functions[i]}")
    print(f"Final test loss: {temp[i][1]}")
    print("-----------------------------------------------------")