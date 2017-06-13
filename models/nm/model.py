from keras.models import Sequential, Model
from keras.layers import Input, concatenate
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

def model(weights_path=None):
    input = Input((8, 128, 128))

    # ------------------------------------------------------
    # ------------------ Conv Block 1 ----------------------
    # --------------------- 128x128 ------------------------
    conv11 = Conv2D(64, (3, 3), padding='same')(input)
    bn11 = BatchNormalization(axis=1)(conv11)
    act11 = Activation('relu')(bn11)

    concat11 = concatenate([input, act11], axis=1)

    conv12 = Conv2D(64, (3, 3), padding='same')(concat11)
    bn12 = BatchNormalization(axis=1)(conv12)
    act12 = Activation('relu')(bn12)

    concat12 = concatenate([input, act11, act12], axis=1)

    conv13 = Conv2D(64, (3, 3), padding='same')(concat12)
    bn13 = BatchNormalization(axis=1)(conv13)
    act13 = Activation('relu')(bn13)

    concat12 = concatenate([input, act11, act12, act13], axis=1)

    conv14 = Conv2D(64, (3, 3), padding='same')(concat12)
    bn14 = BatchNormalization(axis=1)(conv14)
    act14 = Activation('relu')(bn14)

    concat13 = concatenate([input, act11, act12, act13, act14], axis=1)

    conv15 = Conv2D(64, (3, 3), padding='same')(concat13)
    bn15 = BatchNormalization(axis=1)(conv15)
    act15 = Activation('relu')(bn15)

    concat14 = concatenate([input, act11, act12, act13, act14, act15], axis=1)

    conv16 = Conv2D(64, (3, 3), padding='same')(concat14)
    bn16 = BatchNormalization(axis=1)(conv16)
    act16 = Activation('relu')(bn16)

    pool11 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(act16)

    # ------------------------------------------------------
    # ------------------ Conv Block 2 ----------------------
    # ---------------------- 64x64 -------------------------
    conv21 = Conv2D(76, (3, 3), padding='same')(pool11)
    bn21 = BatchNormalization(axis=1)(conv21)
    act21 = Activation('relu')(bn21)

    concat21 = concatenate([pool11, act21], axis=1)

    conv22 = Conv2D(76, (3, 3), padding='same')(concat21)
    bn22 = BatchNormalization(axis=1)(conv22)
    act22 = Activation('relu')(bn22)

    concat22 = concatenate([pool11, act21, act22], axis=1)

    conv23 = Conv2D(76, (3, 3), padding='same')(concat22)
    bn23 = BatchNormalization(axis=1)(conv23)
    act23 = Activation('relu')(bn23)

    concat23 = concatenate([pool11, act21, act22, act23], axis=1)

    conv24 = Conv2D(76, (3, 3), padding='same')(concat23)
    bn24 = BatchNormalization(axis=1)(conv24)
    act24 = Activation('relu')(bn24)

    concat24 = concatenate([pool11, act21, act22, act23, act24], axis=1)

    conv25 = Conv2D(76, (3, 3), padding='same')(concat24)
    bn25 = BatchNormalization(axis=1)(conv25)
    act25 = Activation('relu')(bn25)

    concat25 = concatenate([pool11, act21, act22, act23, act24, act25], axis=1)

    conv26 = Conv2D(76, (3, 3), padding='same')(concat25)
    bn26 = BatchNormalization(axis=1)(conv26)
    act26 = Activation('relu')(bn26)

    pool21 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(act26)

    # ------------------------------------------------------
    # ------------------ Conv Block 3 ----------------------
    # ---------------------- 32x32 -------------------------
    conv31 = Conv2D(88, (3, 3), padding='same')(pool21)
    bn31 = BatchNormalization(axis=1)(conv31)
    act31 = Activation('relu')(bn31)

    concat31 = concatenate([pool21, act31], axis=1)

    conv32 = Conv2D(88, (3, 3), padding='same')(concat31)
    bn32 = BatchNormalization(axis=1)(conv32)
    act32 = Activation('relu')(bn32)

    concat32 = concatenate([pool21, act31, act32], axis=1)

    conv33 = Conv2D(88, (3, 3), padding='same')(concat32)
    bn33 = BatchNormalization(axis=1)(conv33)
    act33 = Activation('relu')(bn33)

    concat33 = concatenate([pool21, act31, act32, act33], axis=1)

    conv34 = Conv2D(88, (3, 3), padding='same')(concat33)
    bn34 = BatchNormalization(axis=1)(conv34)
    act34 = Activation('relu')(bn34)

    concat34 = concatenate([pool21, act31, act32, act33, act34], axis=1)

    conv35 = Conv2D(88, (3, 3), padding='same')(concat34)
    bn35 = BatchNormalization(axis=1)(conv35)
    act35 = Activation('relu')(bn35)

    concat35 = concatenate([pool21, act31, act32, act33, act34, act35], axis=1)

    conv36 = Conv2D(88, (3, 3), padding='same')(concat35)
    bn36 = BatchNormalization(axis=1)(conv36)
    act36 = Activation('relu')(bn36)

    pool31 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(act36)

    # ------------------------------------------------------
    # ------------------ Conv Block 4 ----------------------
    # ---------------------- 16x16 -------------------------
    conv41 = Conv2D(100, (3, 3), padding='same')(pool31)
    bn41 = BatchNormalization(axis=1)(conv41)
    act41 = Activation('relu')(bn41)

    concat41 = concatenate([pool31, act41], axis=1)

    conv42 = Conv2D(100, (3, 3), padding='same')(concat41)
    bn42 = BatchNormalization(axis=1)(conv42)
    act42 = Activation('relu')(bn42)

    concat42 = concatenate([pool31, act41, act42], axis=1)

    conv43 = Conv2D(100, (3, 3), padding='same')(concat42)
    bn43 = BatchNormalization(axis=1)(conv43)
    act43 = Activation('relu')(bn43)

    concat43 = concatenate([pool31, act41, act42, act43], axis=1)

    conv44 = Conv2D(100, (3, 3), padding='same')(concat43)
    bn44 = BatchNormalization(axis=1)(conv44)
    act44 = Activation('relu')(bn44)

    concat44 = concatenate([pool31, act41, act42, act43, act44], axis=1)

    conv45 = Conv2D(100, (3, 3), padding='same')(concat44)
    bn45 = BatchNormalization(axis=1)(conv45)
    act45 = Activation('relu')(bn45)

    concat45 = concatenate([pool31, act41, act42, act43, act44, act45], axis=1)

    conv46 = Conv2D(100, (3, 3), padding='same')(concat45)
    bn46 = BatchNormalization(axis=1)(conv46)
    act46 = Activation('relu')(bn46)

    pool41 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(act46)

    # ------------------------------------------------------
    # ------------------ Conv Block 5 ----------------------
    # ---------------------- 8x8 -------------------------
    conv51 = Conv2D(112, (3, 3), padding='same')(pool41)
    bn51 = BatchNormalization(axis=1)(conv51)
    act51 = Activation('relu')(bn51)

    concat51 = concatenate([pool41, act51], axis=1)

    conv52 = Conv2D(112, (3, 3), padding='same')(concat51)
    bn52 = BatchNormalization(axis=1)(conv52)
    act52 = Activation('relu')(bn52)

    concat52 = concatenate([pool41, act51, act52], axis=1)

    conv53 = Conv2D(112, (3, 3), padding='same')(concat52)
    bn53 = BatchNormalization(axis=1)(conv53)
    act53 = Activation('relu')(bn53)

    concat53 = concatenate([pool41, act51, act52, act53], axis=1)

    conv54 = Conv2D(112, (3, 3), padding='same')(concat53)
    bn54 = BatchNormalization(axis=1)(conv54)
    act54 = Activation('relu')(bn54)

    concat54 = concatenate([pool41, act51, act52, act53, act54], axis=1)

    conv55 = Conv2D(112, (3, 3), padding='same')(concat54)
    bn55 = BatchNormalization(axis=1)(conv55)
    act55 = Activation('relu')(bn55)

    concat55 = concatenate([pool41, act51, act52, act53, act54, act55], axis=1)

    conv56 = Conv2D(112, (3, 3), padding='same')(concat55)
    bn56 = BatchNormalization(axis=1)(conv56)
    act56 = Activation('relu')(bn56)

    pool51 = AveragePooling2D(pool_size=(8, 8), strides=(1, 1))(act56)

    # Dense layers
    flt = Flatten()(pool51)
    dense1 = Dense(256, kernel_regularizer=l2(1e-5))(flt)
    dnact1 = Activation('relu')(dense1)
    dndrop1 = Dropout(0.1)(dnact1)

    dense2 = Dense(128, kernel_regularizer=l2(1e-5))(dndrop1)
    dnact2 = Activation('relu')(dense2)
    dndrop2 = Dropout(0.1)(dnact2)

    output = Dense(17, activation='sigmoid')(dndrop2)

    _model = Model(inputs=input, outputs=output)

    if weights_path:
        _model.load_weights(weights_path)

    return [_model, 'models/nm/structures/']
