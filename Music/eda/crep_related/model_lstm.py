from keras.models import Model
from keras.layers import *

def get_model():
    model_input = Input((700,1))
    #mask = Masking(mask_value=0.,input_shape=(5000,3))
    #lstm_layer1 = CuDNNLSTM(512,return_sequences=True,input_shape=(5000,1))
    lstm_layer1 = CuDNNLSTM(512,input_shape=(700,1))
    #lstm_layer1 = CuDNNLSTM(512,return_sequences=True)
    #lstm_layer2 = CuDNNLSTM(512,return_sequences=True)
    #lstm_layer3 = CuDNNLSTM(512)
    dense = Dense(6,activation='softmax')
    
    #x = mask(model_input)
    x = lstm_layer1(model_input)
    #x = lstm_layer2(x)
    #x = lstm_layer3(x)
    x = dense(x)


    model = Model(inputs=model_input,outputs=x)
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    return model
