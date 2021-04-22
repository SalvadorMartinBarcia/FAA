from Datos import Datos

if __name__ == '__main__':
    dataset = Datos('tic-tac-toe.data')
    dataset2 = Datos('german.data')
    print(dataset2.diccionario)

    clases = dataset.diccionario['Class']
    print(list(clases.values()))
    for key in dataset.diccionario.items():

        
        print(list(key[-1].values()))