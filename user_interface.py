# toto je "main" súbor a spúšťa sa z neho aplikácia, ktorá načítava model, pca a scaler vytvorené funkciami v súbore network_training.py
# a údaje boli načítané funkciami v súbore input_data_parsing.py, teraz už to nie je spojené a aplikácia sa spúšťa tu, v starších verziách
# na git hub je celý skript v jednom súbore
import csv
import json
import joblib
import pandas as pd
import numpy as num
import seaborn as sns
import keras
import input_data_parsing as idp
import matplotlib.pyplot as plotter
from openpyxl import Workbook
from PySide6 import QtWidgets
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import Qt, QLocale
from PySide6.QtGui import QIntValidator, QDoubleValidator
from scipy.stats import pearsonr

# tu sa načítava model, scaler, pca a nastavuje sa okno, do ktorého je načítané grafické rozhranie
validator_double = QDoubleValidator()
validator_double.setLocale(QLocale('C'))
loader = QUiLoader()
app = QtWidgets.QApplication([])
Scaler_ssim = joblib.load("scaler_ssim.pkl")
Scaler_vmaf = joblib.load("scaler_vmaf.pkl")
pca_ssim = joblib.load("pca_ssim.pkl")
pca_vmaf = joblib.load("pca_vmaf.pkl")
model_ssim = keras.saving.load_model("ssim_model.keras")
model_vmaf = keras.saving.load_model("vmaf_model.keras")
   
window = loader.load("VQA_ui_new.ui")
window.setWindowTitle("Video Quality Assessment")

# potvrdenie vstupov na predikciu v prvej časti aplikácie
def submit_for_prediction():
    if check_inputs():
        scene = window.scene_combo.currentIndex() + 1
        res_text = window.resolution_combo.currentText().split(" ")[0]
        resolution = idp.resolution_switch(res_text)
        codec = window.codec_combo.currentIndex()
        packet_loss = float(window.loss_input.text().replace(',', '.'))
        bitrate = int(window.bitrate_input.text())
        objective_metric = float(window.objective_metric_input.text())
        model, scaler, pca = model_ssim, Scaler_ssim, pca_ssim if window.objective_metric_combo.currentText() == "SSIM" else (model_vmaf, Scaler_vmaf, pca_vmaf)
        prediction = predict_video_quality(scene, codec, resolution, bitrate, packet_loss, objective_metric, model, pca, scaler)
        window.result_label.setText(str(prediction))

# obnovenie vstupov v prvej časti apikácie
def refresh_ui():
    window.result_label.setText("")
    window.bitrate_input.clear()
    window.objective_metric_input.clear()
    window.loss_input.clear()

# ukladanie výsledkov do .csv, pod tým do .xlsx (excle) súboru a poton .json
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

# otvorenie okna na uloženie do súboru
def save_file_dialog():
    file_dialog = QtWidgets.QFileDialog()
    file_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
    file_dialog.setNameFilters(["CSV files (*.csv)", "Excel files (*.xlsx)", "JSON files (*.json)"])
    file_dialog.setDefaultSuffix("csv")
    file_dialog.setViewMode(QtWidgets.QFileDialog.List)
    header = ["Scene", "Codec", "Resolution", "Bitrate", "Packet loss", "SSIM", "VMAF", "MOS (SSIM)", "MOS (VMAF)"]

    if file_dialog.exec():
        selected_file = file_dialog.selectedFiles()[0]
        if selected_file.split(".")[-1] == "xlsx":
            save_to_excel(selected_file, header)
        elif selected_file.split(".")[-1] == "json":
            save_to_json(selected_file, header)
        elif selected_file.split(".")[-1] == "csv":
            save_to_csv(selected_file, header)

# načítanie údajov zo .csv súboru a ich vrátenie, pod tým funkcia na to isté pre .json
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
        header = ["Scene", "Codec", "Resolution", "Bitrate", "Packet loss", "SSIM", "VMAF", "MOS (SSIM)", "MOS (VMAF)"]
        json_data = json.load(json_file)
        json_data = [[item["Scene"], item["Codec"], item["Resolution"], item["Bitrate"], item["Packet loss"], item["SSIM"], item["VMAF"], item["MOS (SSIM)"], item["MOS (VMAF)"]] for item in json_data]
        data = [header] + json_data
        return data
    
# táto funkcia zaobaľuje jednotlivé funkcie načítavania údajov, teda podľa typu zvoleného súboru zavolá príslušnú funkciu
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

# získa údaje zo vstupného súboru a vloží ich do tabuľky aj s vypočítaným výsledkom predikcie        
def open_file_dialog():
    table_data = get_file_data()
    if table_data:
        for i in range(1, len(table_data)):
            window.predictions_table.insertRow(window.predictions_table.rowCount())
            for column, value in enumerate(table_data[i]):
                window.predictions_table.setItem(window.predictions_table.rowCount() - 1, column, QtWidgets.QTableWidgetItem(value))
            if table_data[i][1] == "H.264":
                codec = 0
            elif table_data[i][1] == "H.265":
                codec = 1
            else:
                codec = "error_codec"
            try:
                ssim_pred = predict_video_quality(idp.scene_switch(table_data[i][0]), codec, idp.resolution_switch(table_data[i][2]), int(table_data[i][3]), float(table_data[i][4]), float(table_data[i][5]), model_ssim, pca_ssim, Scaler_ssim)
            except:
                ssim_pred = "ERROR"
            try:
                vmaf_pred = predict_video_quality(idp.scene_switch(table_data[i][0]), codec, idp.resolution_switch(table_data[i][2]), int(table_data[i][3]), float(table_data[i][4]), float(table_data[i][6]), model_vmaf, pca_vmaf, Scaler_vmaf)
            except:
                vmaf_pred = "ERROR"
            window.predictions_table.setItem(window.predictions_table.rowCount() - 1, 7, QtWidgets.QTableWidgetItem(str(ssim_pred)))
            window.predictions_table.setItem(window.predictions_table.rowCount() - 1, 8, QtWidgets.QTableWidgetItem(str(vmaf_pred)))
        window.predictions_table.setSortingEnabled(True)

# otvorí okno na výber súbora na načítanie s výsledkami, takže okrem vstupov má aj MOS, iba sa naplní tabuľka, teda otvára súbory
# uložené touto aplikáciou        
def open_results_file_dialog():
    table_data = get_file_data()
    if table_data:
        if table_data[0][:9] == ["Scene", "Codec", "Resolution", "Bitrate", "Packet loss", "SSIM", "VMAF", "MOS (SSIM)", "MOS (VMAF)"]:
            for i in range(1, len(table_data)):
                window.predictions_table.insertRow(window.predictions_table.rowCount())
                for column, value in enumerate(table_data[i]):
                    window.predictions_table.setItem(window.predictions_table.rowCount() - 1, column, QtWidgets.QTableWidgetItem(value))
            window.predictions_table.setSortingEnabled(True)              

# použije model na predikciu, ak sú vstupy zlé a predikcia zlyhá, vráti "ERROR" miesto výsledku
def predict_video_quality(scene, codec, resolution, bitrate, packet_loss, objective_metric, model, pca, scaler):
    try:
        input_data = num.array([[scene, codec, resolution, bitrate, packet_loss, objective_metric]])
        input_data = scaler.transform(input_data)
        input_data = pca.transform(input_data)
        prediction = model.predict(input_data, verbose=0)
        prediction= prediction[0][0]
    except:
        prediction = "ERROR"
        
    return prediction

# skontrolovanie vstupov v prvej časti aplikácie, kontroluje sa hodnota, typ je ošetrený na samotných vstupných poliach, 
# kde je obmedzený ich typ, ak by sa to obišlo tak predikcia zlyhá a jednoducho vráti ako výsledok "ERROR" miesto čísla
def check_inputs():
    error_message = ""
    if window.bitrate_input.text() == "" or int(window.bitrate_input.text()) < 1 or int(window.bitrate_input.text()) > 15:
        window.bitrate_input.clear()
        error_message += "Bitrate must be between 1 and 15.\n"
    if window.loss_input.text() == "" or float(window.loss_input.text()) < 0.0 or float(window.loss_input.text()) > 1.0:
        window.loss_input.clear()
        error_message += "Packet loss must be between 0.0 and 1.0.\n"
    if window.objective_metric_combo.currentText() == "SSIM" and (window.objective_metric_input.text() == "" or float(window.objective_metric_input.text()) < 0.0 or float(window.objective_metric_input.text()) > 1.0):
        window.objective_metric_input.clear()
        error_message += "SSIM must be between 0.0 and 1.0.\n"
    if window.objective_metric_combo.currentText() == "VMAF" and (window.objective_metric_input.text() == "" or float(window.objective_metric_input.text()) < 0.0 or float(window.objective_metric_input.text()) > 100.0):
        window.objective_metric_input.clear()
        error_message += "VMAF must be between 0.0 and 100.0.\n"
        
    if error_message != "":
        QtWidgets.QMessageBox.warning(window, "Input Error", error_message)
        return False
    
    return True

# nastavenie prvkov desktop aplikácie, priradenie funkcií tlačidlám a iným prvkom, vložené hodnoty pre combo boxy
# na konci sa zobrazí okno a spustí aplikácia
def run_ui():
    window.submit_button.setFocus()

    window.resolution_combo.addItems(["HD (720p)", "FHD (1080p)", "UHD (2160p)"])
    window.codec_combo.addItems(["H.264", "H.265"])
    window.scene_combo.addItems(["Campfire", "Construction", "Runners", "Rush", "Tall", "Wood"])
    window.objective_metric_combo.addItems(["SSIM", "VMAF"])
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
    window.objective_metric_input.setValidator(validator_double)

    window.show()
    app.exec()

# funkcia na zobrazenie grafov MOS oproti hodnote stratovosti (packet_loss), SSIM a VMAF
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
        result = predict_video_quality(1, 0, 1080, bitrate, packet_loss, ssim, model_ssim, pca_ssim, Scaler_ssim)
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
        result = predict_video_quality(1, 0, 1080, bitrate, packet_loss, ssim, model_ssim, pca_ssim, Scaler_ssim)
        mos_results.append(result)
        ssim_list.append(ssim)
        ssim += 0.001
    ssim = 0.971
    vmaf = 20.0
    plotter.plot(ssim_list, mos_results)
    plotter.title("SSIM vs MOS")
    plotter.xlabel("SSIM")
    plotter.ylabel("MOS")
    plotter.show()
    ssim_list.clear()
    mos_results.clear()
    while vmaf <= 100.0:
        result = predict_video_quality(1, 0, 1080, bitrate, packet_loss, vmaf, model_vmaf, pca_vmaf, Scaler_vmaf)
        mos_results.append(result)
        vmaf_list.append(vmaf)
        vmaf += 0.01
    plotter.plot(vmaf_list, mos_results)
    plotter.title("VMAF vs MOS")
    plotter.xlabel("VMAF")
    plotter.ylabel("MOS")
    plotter.show()
        
    
def get_pearsons_correlation():
    scene_list, codec_list, resolution_list, bitrate_list, packet_loss_list, ssim_list, vmaf_list, labels_list = idp.get_input_data()
    mos_true, mos_ssim_pred, mos_vmaf_pred = [], [], []
    for i in range(len(labels_list)):
        ssim_pred = predict_video_quality(scene_list[i], codec_list[i], resolution_list[i], bitrate_list[i], packet_loss_list[i], ssim_list[i], model_ssim, pca_ssim, Scaler_ssim)
        vmaf_pred = predict_video_quality(scene_list[i], codec_list[i], resolution_list[i], bitrate_list[i], packet_loss_list[i], vmaf_list[i], model_vmaf, pca_vmaf, Scaler_vmaf)
        mos_true.append(labels_list[i])
        mos_ssim_pred.append(ssim_pred)
        mos_vmaf_pred.append(vmaf_pred)
        
    pearson_ssim, ssim_p_value = pearsonr(mos_true, mos_ssim_pred)
    pearson_vmaf, vmaf_p_value = pearsonr(mos_true, mos_vmaf_pred)
    
    plotter.figure(figsize=(6, 6))
    sns.set(style="whitegrid")

    sns.regplot(x=mos_true, y=mos_ssim_pred, scatter_kws={"s": 40, "alpha": 0.7}, line_kws={"color": "red"})
    plotter.xlabel("Referenčná MOS")
    plotter.ylabel("Predikovaná MOS")
    plotter.title(f"MOS Korelácia pre SSIM model: {pearson_ssim:.4f}")

    plotter.tight_layout()
    plotter.show()
    
    plotter.figure(figsize=(6, 6))
    sns.set(style="whitegrid")

    sns.regplot(x=mos_true, y=mos_vmaf_pred, scatter_kws={"s": 40, "alpha": 0.7}, line_kws={"color": "red"})
    plotter.xlabel("Referenčná MOS")
    plotter.ylabel("Predikovaná MOS")
    plotter.title(f"MOS Korelácia pre VMAF model: {pearson_vmaf:.4f}")

    plotter.tight_layout()
    plotter.show()
    
    print("-------------------------")
    print(f"SSIM Pearson correlation coefficient: {pearson_ssim:.4f}")
    print(f"SSIM P-value: {ssim_p_value:}")
    print("-------------------------")
    print(f"VMAF Pearson correlation coefficient: {pearson_vmaf:.4f}")
    print(f"VMAF P-value: {vmaf_p_value:}")
    print("-------------------------")
        

def main():
    #get_pearsons_correlation()
    #show_model_graphs()
    run_ui()

main()

#if __name__ == "__main__":
    #main()