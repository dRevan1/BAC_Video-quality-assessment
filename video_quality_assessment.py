import csv
import numpy as num
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#import matplotlib.pyplot as plotter

num.set_printoptions(suppress=True)
scene_list = []
resolution_list = [] # HD, FHD, UHD - 720p, 1080p, 2160p
bitrate_list = [] #[-1] je posledný prvok
codec_list = [] #a.insert(0, 5) je 5 na index 0
packet_loss_list = []
expected_result_list = []


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
                expected_result = round(float(line[64].strip()))
                int(expected_result)
                expected_result_list.append(expected_result)

                scene_list.append(scene_switch(parsed_line[0]))
                resolution_list.append(resolution_switch(parsed_line[1]))
                bitrate_list.append(int(parsed_line[2]))
                codec_list.append(codec)
                packet_loss_list.append(get_packet_loss(parameters.split('_')))

        i += 1


#načítanie second session
with open("second_session.csv", mode = "r") as file:  #with closene rovno file = nice
    second_session = csv.reader(file, delimiter = ";")
    i = 0 #celkovo je 502 záznamov vo first_session
    codec = 0 #bude 0/1 podľa kodeku (h264, h265 v poradí)
    first_session_flag = False
    parsed_line = []

    for line in second_session:   #chceme index 0 - slovo sa rozdelý na inputy do ML modelu, index 64 - priemerné ohodnotenie videa (1-5)
        parameters = line[0]

        if "Subjective" in parameters:
            first_session_flag = True
            codec = 0

        if "HEVC" in parameters:
            codec = 1
        elif len(line[1]) == 0:
            codec = 0
        else:
            if i > 1:
                expected_result = round(float(line[47].strip().replace(',', '.')))
                int(expected_result)
                expected_result_list.append(expected_result)
                resolution = get_resolution(parameters)
        
                if codec == 1 and not first_session_flag:
                    resolution["delimiter"] = "{}{}".format("X", resolution["delimiter"])
                    
                parsed_line = parameters.split(resolution["delimiter"])

                if first_session_flag:
                    codec = 1 if parsed_line[0][-1] == 'X' else 0

                scene = parsed_line[0][:-1] if not first_session_flag or (parsed_line[0][-1] == 'X') else parsed_line[0]
                bitrate_PL = parsed_line[1].split('_')
                scene_list.append(scene_switch(scene))
                resolution_list.append(resolution["res"])
                bitrate_list.append(bitrate_PL[0])
                codec_list.append(codec)
                packet_loss_list.append(get_packet_loss(bitrate_PL))

        i += 1

#tu začína PCA
#príprava dát do 2 kommponentov (X, Y)
x_list = []
genes = ["Scene", "Resolution", "Bitrate", "Codec", "Packet loss"]
for i in range(len(expected_result_list)):
    x_list.append([scene_list[i], resolution_list[i], bitrate_list[i], codec_list[i], packet_loss_list[i]])

X = num.array(x_list)
Y = num.array(expected_result_list)

Scaler = StandardScaler()
standard_data = Scaler.fit_transform(X)  #štandardizácia dát
pca = PCA(n_components=2)  #PCA na vytvorenie 2 komponentov
