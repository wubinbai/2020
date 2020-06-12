from keras.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout 

def headmodel(baseModel):

    # 在baseModel基础上添加新的层

    headModel = baseModel.output

    

    # 新的FC层

    headModel = Flatten(name="flatten")(headModel)

    headModel = Dense(2048, activation="relu")(headModel)

    headModel = Dropout(0.5)(headModel)

    headModel = Dense(2048, activation="relu")(headModel)

    headModel = Dropout(0.5)(headModel)


    # 由于是2分类，只有1个神经元输出

    headModel = Dense(1, activation="sigmoid")(headModel)

    return headModel


baseModel =  VGG16(weights="imagenet",include_top=False,input_tensor=Input(shape=(224,224,3)))

headModel = headmodel(baseModel)
model = Model(baseModel.input,headModel)
