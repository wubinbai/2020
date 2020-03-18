import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import Model
from keras.layers import Input
 
# The input and output, i.e. truth table, of a NAND gate
#x_train = np.array([[0,0],[0,1],[1,0],[1,1]], "uint8")
x_train = np.random.rand(1000,2)
#y_train = np.array([[0],[1],[1],[1]], "uint8")
y_train = x_train.sum(axis=1)

using = input('Using Seq. or Functional API? 1 or 2: ')
if using == '1':
    # Create neural networks model
    model = Sequential()
    # Add layers to the model
    model.add(Dense(output_dim=3, activation='relu', input_dim=2))      # first hidden layer
    model.add(Dense(output_dim=3, activation='relu'))                   # second hidden layer
    model.add(Dense(output_dim=1, activation='sigmoid'))                # output layer
     
    # Compile the neural networks model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    # Train the neural networks model
    model.fit(x_train, y_train, nb_epoch=500)
     
    # Test the output of the trained neural networks based NAND gate
    y_predict = model.predict(x_train)
    print(y_predict)
     
    # save model as h5 file
    model.save("nand.h5")

if using == '2':
    # Create neural networks model
    x_in = x = Input(shape=(2,))
    #x = Dense(3, activation='relu')(x)
    #x = Dense(3, activation='relu')(x)
    y = Dense(1, activation='linear')(x)
    model = Model(inputs=x_in,outputs=y)

    #model = Sequential()
    # Add layers to the model
    #model.add(Dense(output_dim=3, activation='relu', input_dim=2))      # first hidden layer
    #model.add(Dense(output_dim=3, activation='relu'))                   # second hidden layer
    #model.add(Dense(output_dim=1, activation='sigmoid'))                # output layer
     
    # Compile the neural networks model
    opt = keras.optimizers.sgd(1e-5)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    # Train the neural networks model
    #model.fit(x_train, y_train, nb_epoch=50)
    x_data = x_train
    y_data = y_train
    #for step in range(3001):
    #    cost = model.train_on_batch(x_data, y_data)
    #    if step%500 == 0:
    #        print('cost: ',cost)
    model.fit(x_train, y_train, nb_epoch=500*20,batch_size=1)

    # Test the output of the trained neural networks based NAND gate
    y_predict = model.predict(x_train)
    print(y_predict)
     
    # save model as h5 file
    #model.save("nand.h5")

