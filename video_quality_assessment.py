import csv


scene_list = []
resolution_list = [] # HD, FHD, UHD - 720p, 1080p, 2160p
bitrate_list = [] #[-1] je posledný prvok
codec_list = [] #a.insert(0, 5) je 5 na index 0
packet_loss_list = []
expected_result_list = []


def get_scene(scene_name):
    switch = {
        "Campfire": 1,
        "Construction": 2,
        "Runners": 3,
        "Wood": 4
    }
    return switch[scene_name]

def get_resolution(res_name):
    switch = {
        "HD": 720,
        "FHD": 1080,
        "UHD": 2160
    }
    return switch[res_name]

def get_packet_loss(split_line):
    loss_number = 1.0

    if len(split_line) != 2:
        loss_number = float(split_line[2])
        while loss_number >= 1.0:
            loss_number /= 10

    return loss_number



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

                scene_list.append(get_scene(parsed_line[0]))
                resolution_list.append(get_resolution(parsed_line[1]))
                bitrate_list.append(int(parsed_line[2]))
                codec_list.append(codec)
                packet_loss_list.append(get_packet_loss(parameters.split('_')))

                print("{} {} {} {} {} {}".format(scene_list[-1], resolution_list[-1], bitrate_list[-1], codec_list[-1], expected_result, packet_loss_list[-1]))
        i += 1

    print(i)    
