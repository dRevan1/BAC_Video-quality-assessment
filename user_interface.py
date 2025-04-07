import csv
import json
import joblib
import pandas as pd
import numpy as num
import keras
import input_data_parsing as idp
import matplotlib.pyplot as plotter
from openpyxl import Workbook
from PySide6 import QtWidgets
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import Qt, QLocale
from PySide6.QtGui import QIntValidator, QDoubleValidator

validator_double = QDoubleValidator()
validator_double.setLocale(QLocale('C'))
loader = QUiLoader()
app = QtWidgets.QApplication([])
Scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
model = keras.saving.load_model("model110b64d005.keras")
   
window = loader.load("VQA_ui_new.ui")
window.setWindowTitle("Video Quality Assessment")

def submit_for_prediction():
    if check_inputs():
        scene = window.scene_combo.currentIndex() + 1
        res_text = window.resolution_combo.currentText().split(" ")[0]
        resolution = idp.resolution_switch(res_text)
        codec = window.codec_combo.currentIndex()
        packet_loss = float(window.loss_input.text().replace(',', '.'))
        bitrate = int(window.bitrate_input.text())
        ssim = float(window.ssim_input.text().replace(',', '.'))
        vmaf = float(window.vmaf_input.text().replace(',', '.'))
        prediction = predict_video_quality(scene, codec, resolution, bitrate, packet_loss, ssim, vmaf)
        window.result_label.setText(str(prediction))

def refresh_ui():
    window.result_label.setText("")
    window.bitrate_input.clear()
    window.ssim_input.clear()
    window.vmaf_input.clear()
    window.loss_input.clear()

def save_to_csv(selected_file, header):
    with open(selected_file, mode="w", newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=";")
        csv_writer.writerow(header)
        for row in range(window.predictions_table.rowCount()):
            row_data = []
            for column in range(window.predictions_table.columnCount()):
                item = window.predictions_table.item(row, column)
                row_data.append(item.text() if item is not None else "")
            csv_writer.writerow(row_data)

def save_to_excel(selected_file, header):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(header)
    for row in range(window.predictions_table.rowCount()):
        row_data = []
        for column in range(window.predictions_table.columnCount()):
            item = window.predictions_table.item(row, column)
            row_data.append(item.text() if item is not None else "")
        sheet.append(row_data)
    workbook.save(selected_file)

def save_to_json(selected_file, header):
    data = []
    for row in range(window.predictions_table.rowCount()):
        row_data = {}
        for column in range(window.predictions_table.columnCount()):
            item = window.predictions_table.item(row, column)
            row_data[header[column]] = item.text() if item is not None else ""
        data.append(row_data)
    with open(selected_file, mode="w") as json_file:
        json.dump(data, json_file, indent=4)

def save_file_dialog():
    file_dialog = QtWidgets.QFileDialog()
    file_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
    file_dialog.setNameFilters(["CSV files (*.csv)", "Excel files (*.xlsx)", "JSON files (*.json)"])
    file_dialog.setDefaultSuffix("csv")
    file_dialog.setViewMode(QtWidgets.QFileDialog.List)
    header = ["Scene", "Codec", "Resolution", "Bitrate", "Packet loss", "SSIM", "VMAF", "MOS"]

    if file_dialog.exec():
        selected_file = file_dialog.selectedFiles()[0]
        if selected_file.split(".")[-1] == "xlsx":
            save_to_excel(selected_file, header)
        elif selected_file.split(".")[-1] == "json":
            save_to_json(selected_file, header)
        elif selected_file.split(".")[-1] == "csv":
            save_to_csv(selected_file, header)

def load_from_csv(selected_file):
    with open(selected_file, mode="r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=";")
        window.predictions_table.setRowCount(0)
        window.predictions_table.setSortingEnabled(False)
        window.predictions_table.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        data = [row for row in csv_reader if any(cell.strip() for cell in row)]
        return data

def load_from_json(selected_file):
    with open(selected_file, mode="r") as json_file:
        header = ["Scene", "Codec", "Resolution", "Bitrate", "Packet loss", "SSIM", "VMAF", "MOS"]
        json_data = json.load(json_file)
        json_data = [[item["Scene"], item["Codec"], item["Resolution"], item["Bitrate"], item["Packet loss"], item["SSIM"], item["VMAF"], item["MOS"]] for item in json_data]
        data = [header] + json_data
        return data
         
def get_file_data():
    file_dialog = QtWidgets.QFileDialog()
    file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
    file_dialog.setNameFilters(["CSV files (*.csv)", "Excel files (*.xlsx)", "JSON files (*.json)"])
    file_dialog.setViewMode(QtWidgets.QFileDialog.List)
    data = None
    

    if file_dialog.exec():
        selected_file = file_dialog.selectedFiles()[0]
        if selected_file.split(".")[-1] == "xlsx":
            data = pd.read_excel(selected_file, header=None).values.tolist()
        elif selected_file.split(".")[-1] == "json":
            data = load_from_json(selected_file)
        elif selected_file.split(".")[-1] == "csv":
            data = load_from_csv(selected_file) 
                 
    return data
        
def open_file_dialog():
    table_data = get_file_data()
    if table_data:
        for row in table_data:
            window.predictions_table.insertRow(window.predictions_table.rowCount())
            for column, value in enumerate(row):
                window.predictions_table.setItem(window.predictions_table.rowCount() - 1, column, QtWidgets.QTableWidgetItem(value))
            if row[1] == "H.264":
                codec = 0
            elif row[1] == "H.265":
                codec = 1
            else:
                codec = "error_codec"
            try:
                predicted_value = predict_video_quality(idp.scene_switch(row[0]), codec, idp.resolution_switch(row[2]), int(row[3]), float(row[4]), float(row[5]), float(row[6]))
            except:
                predicted_value = "ERROR"
            window.predictions_table.setItem(window.predictions_table.rowCount() - 1, 7, QtWidgets.QTableWidgetItem(str(predicted_value)))
        window.predictions_table.setSortingEnabled(True)
        
def open_results_file_dialog():
    table_data = get_file_data()
    if table_data:
        if table_data[0][:8] == ["Scene", "Codec", "Resolution", "Bitrate", "Packet loss", "SSIM", "VMAF", "MOS"]:
            for i in range(1, len(table_data)):
                window.predictions_table.insertRow(window.predictions_table.rowCount())
                for column, value in enumerate(table_data[i]):
                    window.predictions_table.setItem(window.predictions_table.rowCount() - 1, column, QtWidgets.QTableWidgetItem(value))
            window.predictions_table.setSortingEnabled(True)              

def predict_video_quality(scene, codec, resolution, bitrate, packet_loss, ssim, vmaf):
    try:
        input_data = num.array([[scene, codec, resolution, bitrate, packet_loss, ssim, vmaf]])
        input_data = Scaler.transform(input_data)
        input_data = pca.transform(input_data)
        prediction = model.predict(input_data, verbose=0)
        prediction= prediction[0][0]
    except:
        prediction = "ERROR"
        
    return prediction

def check_inputs():
    error_message = ""
    if window.bitrate_input.text() == "" or int(window.bitrate_input.text()) < 1 or int(window.bitrate_input.text()) > 15:
        window.bitrate_input.clear()
        error_message += "Bitrate must be between 1 and 15.\n"
    if window.loss_input.text() == "" or float(window.loss_input.text()) < 0.0 or float(window.loss_input.text()) > 1.0:
        window.loss_input.clear()
        error_message += "Packet loss must be between 0.0 and 1.0.\n"
    if window.ssim_input.text() == "" or float(window.ssim_input.text()) < 0.0 or float(window.ssim_input.text()) > 1.0:
        window.ssim_input.clear()
        error_message += "SSIM must be between 0.0 and 1.0.\n"
    if window.vmaf_input.text() == "" or float(window.vmaf_input.text()) < 0.0 or float(window.vmaf_input.text()) > 100.0:
        window.vmaf_input.clear()
        error_message += "VMAF must be between 0.0 and 100.0.\n"
        
    if error_message != "":
        QtWidgets.QMessageBox.warning(window, "Input Error", error_message)
        return False
    
    return True

def run_ui():
    window.submit_button.setFocus()

    window.resolution_combo.addItems(["HD (720p)", "FHD (1080p)", "UHD (2160p)"])
    window.codec_combo.addItems(["H.264", "H.265"])
    window.scene_combo.addItems(["Campfire", "Construction", "Runners", "Rush", "Tall", "Wood"])
    window.predictions_table.setSortingEnabled(True)
    window.predictions_table.model().rowsInserted.connect(lambda: window.save_button.setEnabled(window.predictions_table.rowCount() > 0))
    window.predictions_table.model().rowsRemoved.connect(lambda: window.save_button.setEnabled(window.predictions_table.rowCount() > 0))

    window.submit_button.clicked.connect(submit_for_prediction)
    window.refresh_button.clicked.connect(refresh_ui)
    window.file_screen_button.clicked.connect(lambda: window.stackedWidget.setCurrentWidget(window.page_file))
    window.home_button.clicked.connect(lambda: window.stackedWidget.setCurrentWidget(window.page_main))
    window.load_file_button.clicked.connect(open_file_dialog)
    window.load_results_button.clicked.connect(open_results_file_dialog)
    window.save_button.clicked.connect(save_file_dialog)

    window.bitrate_input.setValidator(QIntValidator())
    window.loss_input.setValidator(validator_double)
    window.ssim_input.setValidator(validator_double)
    window.vmaf_input.setValidator(validator_double)

    window.show()
    app.exec()

def show_model_graphs():
    bitrate = 10
    packet_loss = 0.1
    ssim = 0.971
    vmaf = 97.2
    mos_results = []
    packet_loss_list = []
    ssim_list = []
    vmaf_list = []
    while packet_loss <= 1.0:
        result = predict_video_quality(1, 0, 1080, bitrate, packet_loss, ssim, vmaf)
        mos_results.append(result)
        packet_loss_list.append(packet_loss)
        packet_loss += 0.01
    packet_loss = 0.1
    ssim = 0.3
    plotter.plot(packet_loss_list, mos_results)
    plotter.title("Packet loss vs MOS")
    plotter.xlabel("Packet loss")
    plotter.ylabel("MOS")
    plotter.show()
    packet_loss_list.clear()
    mos_results.clear()
    while ssim <= 1.0:
        result = predict_video_quality(1, 0, 1080, bitrate, packet_loss, ssim, vmaf)
        mos_results.append(result)
        ssim_list.append(ssim)
        ssim += 0.001
    ssim = 0.971
    vmaf = 10.0
    plotter.plot(ssim_list, mos_results)
    plotter.title("SSIM vs MOS")
    plotter.xlabel("SSIM")
    plotter.ylabel("MOS")
    plotter.show()
    ssim_list.clear()
    mos_results.clear()
    while vmaf <= 100.0:
        result = predict_video_quality(1, 0, 1080, bitrate, packet_loss, ssim, vmaf)
        mos_results.append(result)
        vmaf_list.append(vmaf)
        vmaf += 0.01
    plotter.plot(vmaf_list, mos_results)
    plotter.title("VMAF vs MOS")
    plotter.xlabel("VMAF")
    plotter.ylabel("MOS")
    plotter.show()
        
    

def main():
    run_ui()

main()

#if __name__ == "__main__":
    #main()