from keras import callbacks
import matplotlib.pyplot as plotter
import network_training as nt

# funkcia, ktorá skúša rôzne topológie zo zoznamu nižšie, 10-krát prejde všetky a vyberie najlepšiu podľa kvadratickej chyby a vypíše ju
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
    
    for i in range(10):
        for config in network_configs:
            nt.train_network_configuration_test(config, 'relu', x_train, y_train, x_test, y_test, results)
        
        results.sort(key=lambda x: x[1])
        experiment_results.append(results[0])    
        results.clear()
    
    for result in experiment_results:
        print(f"Neurons: {result[0]}")
        print(f"\nFinal test loss: {result[1]}")
        print("-----------------------------------------------------")

# tu sa skúšajú rôzne aktivačné funkcie, pri testovaní bola pustená viac krát po sebe
def try_activation_functions(x_train, y_train, x_test, y_test):
    results = []
    activation_functions = ['relu', 'elu', 'selu', 'sigmoid', 'tanh', 'swish', 'softplus']
    for i in range(15): 
        for activation in activation_functions:
            nt.train_network_configuration_test([256, 128, 64, 32, 16], activation, x_train, y_train, x_test, y_test, results)
            
        print(f"Experiment {i+1}:")
        for j in range(len(activation_functions)):
            print(f"Activation function: {activation_functions[j]}")
            print(f"Final test loss: {results[j][1]}")
            print("-----------------------------------------------------")
            
        results.clear()
     

 # tu sa trénuje neurónová sieť, pričom priebeh sa zaznamená a zobrazí na grafe, toto je funkcia primárne na zobrazenie
 # kvadratickej chyby do grafu, inak sa použila tá v súbore network_training.py - train_network_configuration_test()
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
    
    