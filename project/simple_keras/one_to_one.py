from keras import Input, Model
from keras.layers import Dense

X_train = np.array([i/1000 for i in list(range(1000))])
X_train -= 0.5

#X_train = np.array([-1000 for i in list(range(1000))])
#X_train[500:] = 1000
y_train = np.array([0 for i in list(range(1000))])
y_train[500:] = 1
print(X_train.shape,y_train.shape)


x = Input(shape=(1,))
hidden0 = Dense(2)(x)
#x = Dense(2)(x)
#hidden2 = Dense()
y = Dense(1,activation='sigmoid')(hidden0)
#y = Dense(1)(hidden)

model = Model(inputs=x, outputs=y)
print(model.summary())

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train,y_train,epochs=10)


