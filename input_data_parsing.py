import csv
import numpy as num
import network_training as nt

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
    return switch.get(scene_name, "error_scene")

def resolution_switch(res_name):
    switch = {
        "HD": 720,
        "FHD": 1080,
        "FullHD": 1080,
        "UHD": 2160
    }
    return switch.get(res_name, "error_res")

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

def load_first_file():
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
            i += 1
    
def load_second_file():
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

            i += 1
            
def train_model():
    load_first_file()
    load_second_file()
    x_train, x_test, y_train, y_test = nt.preprocess_data(scene_list, codec_list, resolution_list, bitrate_list, packet_loss_list, ssim_list, vmaf_list, labels_list)
    model = nt.train_network_configuration_test([256, 128, 64, 32], "relu", x_train, y_train, x_test, y_test, [])
    return model
    