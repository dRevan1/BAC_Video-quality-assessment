from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as num
import pandas
from sklearn.model_selection import train_test_split
from keras import Model, layers, Input

def preprocess_data(scene_list, codec_list, resolution_list, bitrate_list, packet_loss_list, objective_metric_list, labels_list):
    #tu začína PCA
    #príprava dát do 5 kommponentov (X, Y)
    # scaler a pca boli odtiaľto uložené do súborov scaler.pkl a pca.pkl
    x_list = []
    for i in range(len(labels_list)):
        x_list.append([scene_list[i], codec_list[i], resolution_list[i], bitrate_list[i], packet_loss_list[i], objective_metric_list[i]])
    feature_names = ['scene', 'codec', 'resolution', 'bitrate', 'packet_loss', 'objective_metric']
    X = num.array(x_list)
    Y = num.array(labels_list)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train = X_train.astype(num.float64)

    Scaler = StandardScaler()
    X_train = Scaler.fit_transform(X_train)
    X_test = Scaler.transform(X_test)
    pca = PCA(n_components=5)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    explained_ratios = pca.explained_variance_ratio_
    print("Principal components: \n")
    for i, ratio in enumerate(explained_ratios):
        print(f"{i+1}. - {ratio:.4f}")
        
    print(f"\nSum 1-5: {num.sum(explained_ratios[:5]):.4f}")
    loadings = pandas.DataFrame(pca.components_.T, 
                        columns=[f'PC{i+1}' for i in range(5)], 
                        index=feature_names)

    print("\nPCA Components / Loadings Table:")
    print(loadings.round(4))
    return X_train, X_test, Y_train, Y_test

# zostavenie a trénovanie siete, vracia výsledný model, môže ukladať výsledky do zoznamu training_results, ale
# pri volaní sa môže poslať aj prázdny zoznam, ak ich nepotrebujeme    
def train_network_configuration_test(neurons_list, activation_function, x_train, y_train, x_test, y_test, training_results):
    input_layer = Input(shape=(5,))
    layer_list = []

    layer_list.append(layers.Dense(neurons_list[0], activation=activation_function)(input_layer))
    for i in range(1, len(neurons_list)):
        layer_list.append(layers.Dense(neurons_list[i], activation=activation_function)(layer_list[-1]))
        layer_list.append(layers.Dropout(0.05)(layer_list[-1]))
    output_layer = layers.Dense(1, activation='linear')(layer_list[-1])

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    model.fit(x_train, y_train, epochs=300, batch_size=32, validation_data=(x_test, y_test))
    #model.save("vmaf_model.keras")
    
    test_loss = model.evaluate(x_test, y_test)
    training_results.append([neurons_list, test_loss])
    
    return model