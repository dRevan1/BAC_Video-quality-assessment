from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as num
from sklearn.model_selection import train_test_split
from keras import Model, layers, Input

def preprocess_data(scene_list, codec_list, resolution_list, bitrate_list, packet_loss_list, ssim_list, vmaf_list, labels_list):
    #tu začína PCA
    #príprava dát do 2 kommponentov (X, Y)
    # scaler a pca boli odtiaľto uložené do súborov scaler.pkl a pca.pkl
    x_list = []
    for i in range(len(labels_list)):
        x_list.append([scene_list[i], codec_list[i], resolution_list[i], bitrate_list[i], packet_loss_list[i], ssim_list[i], vmaf_list[i]])

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
    return X_train, X_test, Y_train, Y_test

# zostavenie a trénovanie siete, vracia výsledný model, môže ukladať výsledky do zoznamu training_results, ale
# pri volaní sa môže poslať aj prázdny zoznam, ak ich nepotrebujeme    
def train_network_configuration_test(neurons_list, activation_function, x_train, y_train, x_test, y_test, training_results):
    input_layer = Input(shape=(4,))
    layer_list = []

    layer_list.append(layers.Dense(neurons_list[0], activation=activation_function)(input_layer))
    for i in range(1, len(neurons_list)):
        layer_list.append(layers.Dense(neurons_list[i], activation=activation_function)(layer_list[-1]))
        layer_list.append(layers.Dropout(0.05)(layer_list[-1]))
    output_layer = layers.Dense(1, activation='linear')(layer_list[-1])

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    model.fit(x_train, y_train, epochs=110, batch_size=64, validation_data=(x_test, y_test))
    
    test_loss = model.evaluate(x_test, y_test)
    training_results.append([neurons_list, test_loss])
    
    return model