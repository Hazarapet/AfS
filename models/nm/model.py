from keras.models import Model
from keras.layers import Input, concatenate
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D
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

    # -----------------------------------------------------
    # --------------------- Bridge 1 ----------------------
    bridge_conv11 = Conv2D(64, (3, 3), padding='same')(act16)
    bridge_bn11 = BatchNormalization(axis=1)(bridge_conv11)
    bridge_act11 = Activation('relu')(bridge_bn11)

    bridge_pool11 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(bridge_act11)

    # ------------------------------------------------------
    # ------------------ Conv Block 2 ----------------------
    # ---------------------- 64x64 -------------------------
    input_bridge1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input)
    concat_bridge1 = concatenate([input_bridge1, bridge_pool11], axis=1)

    conv21 = Conv2D(76, (3, 3), padding='same')(concat_bridge1)
    bn21 = BatchNormalization(axis=1)(conv21)
    act21 = Activation('relu')(bn21)

    concat21 = concatenate([concat_bridge1, act21], axis=1)

    conv22 = Conv2D(76, (3, 3), padding='same')(concat21)
    bn22 = BatchNormalization(axis=1)(conv22)
    act22 = Activation('relu')(bn22)

    concat22 = concatenate([concat_bridge1, act21, act22], axis=1)

    conv23 = Conv2D(76, (3, 3), padding='same')(concat22)
    bn23 = BatchNormalization(axis=1)(conv23)
    act23 = Activation('relu')(bn23)

    concat23 = concatenate([concat_bridge1, act21, act22, act23], axis=1)

    conv24 = Conv2D(76, (3, 3), padding='same')(concat23)
    bn24 = BatchNormalization(axis=1)(conv24)
    act24 = Activation('relu')(bn24)

    concat24 = concatenate([concat_bridge1, act21, act22, act23, act24], axis=1)

    conv25 = Conv2D(76, (3, 3), padding='same')(concat24)
    bn25 = BatchNormalization(axis=1)(conv25)
    act25 = Activation('relu')(bn25)

    concat25 = concatenate([concat_bridge1, act21, act22, act23, act24, act25], axis=1)

    conv26 = Conv2D(76, (3, 3), padding='same')(concat25)
    bn26 = BatchNormalization(axis=1)(conv26)
    act26 = Activation('relu')(bn26)

    # -----------------------------------------------------
    # --------------------- Bridge 2 ----------------------
    bridge_conv21 = Conv2D(128, (3, 3), padding='same')(act26)
    bridge_bn21 = BatchNormalization(axis=1)(bridge_conv21)
    bridge_act21 = Activation('relu')(bridge_bn21)

    bridge_pool21 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(bridge_act21)

    # ------------------------------------------------------
    # ------------------ Conv Block 3 ----------------------
    # ---------------------- 32x32 -------------------------
    input_bridge2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act21)
    concat_bridge2 = concatenate([input_bridge2, bridge_pool21], axis=1)

    conv31 = Conv2D(88, (3, 3), padding='same')(concat_bridge2)
    bn31 = BatchNormalization(axis=1)(conv31)
    act31 = Activation('relu')(bn31)

    concat31 = concatenate([concat_bridge2, act31], axis=1)

    conv32 = Conv2D(88, (3, 3), padding='same')(concat31)
    bn32 = BatchNormalization(axis=1)(conv32)
    act32 = Activation('relu')(bn32)

    concat32 = concatenate([concat_bridge2, act31, act32], axis=1)

    conv33 = Conv2D(88, (3, 3), padding='same')(concat32)
    bn33 = BatchNormalization(axis=1)(conv33)
    act33 = Activation('relu')(bn33)

    concat33 = concatenate([concat_bridge2, act31, act32, act33], axis=1)

    conv34 = Conv2D(88, (3, 3), padding='same')(concat33)
    bn34 = BatchNormalization(axis=1)(conv34)
    act34 = Activation('relu')(bn34)

    concat34 = concatenate([concat_bridge2, act31, act32, act33, act34], axis=1)

    conv35 = Conv2D(88, (3, 3), padding='same')(concat34)
    bn35 = BatchNormalization(axis=1)(conv35)
    act35 = Activation('relu')(bn35)

    concat35 = concatenate([concat_bridge2, act31, act32, act33, act34, act35], axis=1)

    conv36 = Conv2D(88, (3, 3), padding='same')(concat35)
    bn36 = BatchNormalization(axis=1)(conv36)
    act36 = Activation('relu')(bn36)

    # -----------------------------------------------------
    # --------------------- Bridge 3 ----------------------
    bridge_conv31 = Conv2D(256, (3, 3), padding='same')(act36)
    bridge_bn31 = BatchNormalization(axis=1)(bridge_conv31)
    bridge_act31 = Activation('relu')(bridge_bn31)

    bridge_pool31 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(bridge_act31)

    # ------------------------------------------------------
    # ------------------ Conv Block 4 ----------------------
    # ---------------------- 16x16 -------------------------
    input_bridge3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act31)
    concat_bridge3 = concatenate([input_bridge3, bridge_pool31], axis=1)

    conv41 = Conv2D(100, (3, 3), padding='same')(concat_bridge3)
    bn41 = BatchNormalization(axis=1)(conv41)
    act41 = Activation('relu')(bn41)

    concat41 = concatenate([concat_bridge3, act41], axis=1)

    conv42 = Conv2D(100, (3, 3), padding='same')(concat41)
    bn42 = BatchNormalization(axis=1)(conv42)
    act42 = Activation('relu')(bn42)

    concat42 = concatenate([concat_bridge3, act41, act42], axis=1)

    conv43 = Conv2D(100, (3, 3), padding='same')(concat42)
    bn43 = BatchNormalization(axis=1)(conv43)
    act43 = Activation('relu')(bn43)

    concat43 = concatenate([concat_bridge3, act41, act42, act43], axis=1)

    conv44 = Conv2D(100, (3, 3), padding='same')(concat43)
    bn44 = BatchNormalization(axis=1)(conv44)
    act44 = Activation('relu')(bn44)

    concat44 = concatenate([concat_bridge3, act41, act42, act43, act44], axis=1)

    conv45 = Conv2D(100, (3, 3), padding='same')(concat44)
    bn45 = BatchNormalization(axis=1)(conv45)
    act45 = Activation('relu')(bn45)

    concat45 = concatenate([concat_bridge3, act41, act42, act43, act44, act45], axis=1)

    conv46 = Conv2D(100, (3, 3), padding='same')(concat45)
    bn46 = BatchNormalization(axis=1)(conv46)
    act46 = Activation('relu')(bn46)

    # -----------------------------------------------------
    # --------------------- Bridge 4 ----------------------
    bridge_conv41 = Conv2D(256, (3, 3), padding='same')(act46)
    bridge_bn41 = BatchNormalization(axis=1)(bridge_conv41)
    bridge_act41 = Activation('relu')(bridge_bn41)

    bridge_pool41 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(bridge_act41)

    # ------------------------------------------------------
    # ------------------ Conv Block 5 ----------------------
    # ----------------------- 8x8 --------------------------
    input_bridge4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act41)
    concat_bridge4 = concatenate([input_bridge4, bridge_pool41], axis=1)

    conv51 = Conv2D(112, (3, 3), padding='same')(concat_bridge4)
    bn51 = BatchNormalization(axis=1)(conv51)
    act51 = Activation('relu')(bn51)

    concat51 = concatenate([concat_bridge4, act51], axis=1)

    conv52 = Conv2D(112, (3, 3), padding='same')(concat51)
    bn52 = BatchNormalization(axis=1)(conv52)
    act52 = Activation('relu')(bn52)

    concat52 = concatenate([concat_bridge4, act51, act52], axis=1)

    conv53 = Conv2D(112, (3, 3), padding='same')(concat52)
    bn53 = BatchNormalization(axis=1)(conv53)
    act53 = Activation('relu')(bn53)

    concat53 = concatenate([concat_bridge4, act51, act52, act53], axis=1)

    conv54 = Conv2D(112, (3, 3), padding='same')(concat53)
    bn54 = BatchNormalization(axis=1)(conv54)
    act54 = Activation('relu')(bn54)

    concat54 = concatenate([concat_bridge4, act51, act52, act53, act54], axis=1)

    conv55 = Conv2D(112, (3, 3), padding='same')(concat54)
    bn55 = BatchNormalization(axis=1)(conv55)
    act55 = Activation('relu')(bn55)

    concat55 = concatenate([concat_bridge4, act51, act52, act53, act54, act55], axis=1)

    conv56 = Conv2D(112, (3, 3), padding='same')(concat55)
    bn56 = BatchNormalization(axis=1)(conv56)
    act56 = Activation('relu')(bn56)

    # -----------------------------------------------------
    # --------------------- Bridge 5 ----------------------
    bridge_conv51 = Conv2D(512, (3, 3), padding='same')(act56)
    bridge_bn51 = BatchNormalization(axis=1)(bridge_conv51)
    bridge_act51 = Activation('relu')(bridge_bn51)

    bridge_pool51 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(bridge_act51)

    # ------------------------------------------------------
    # ------------------ Conv Block 6 ----------------------
    # ----------------------- 4x4 --------------------------
    input_bridge4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act41)
    concat_bridge4 = concatenate([input_bridge4, bridge_pool51], axis=1)

    conv61 = Conv2D(124, (3, 3), padding='same')(concat_bridge4)
    bn61 = BatchNormalization(axis=1)(conv61)
    act61 = Activation('relu')(bn61)

    concat61 = concatenate([concat_bridge4, act61], axis=1)

    conv62 = Conv2D(124, (3, 3), padding='same')(concat61)
    bn62 = BatchNormalization(axis=1)(conv62)
    act62 = Activation('relu')(bn62)

    concat62 = concatenate([concat_bridge4, act61, act62], axis=1)

    conv63 = Conv2D(124, (3, 3), padding='same')(concat62)
    bn63 = BatchNormalization(axis=1)(conv63)
    act63 = Activation('relu')(bn63)

    concat63 = concatenate([concat_bridge4, act61, act62, act63], axis=1)

    conv64 = Conv2D(124, (3, 3), padding='same')(concat63)
    bn64 = BatchNormalization(axis=1)(conv64)
    act64 = Activation('relu')(bn64)

    concat64 = concatenate([concat_bridge4, act61, act62, act63, act64], axis=1)

    conv65 = Conv2D(124, (3, 3), padding='same')(concat64)
    bn65 = BatchNormalization(axis=1)(conv65)
    act65 = Activation('relu')(bn65)

    concat65 = concatenate([concat_bridge4, act61, act62, act63, act64, act65], axis=1)

    conv66 = Conv2D(124, (3, 3), padding='same')(concat65)
    bn66 = BatchNormalization(axis=1)(conv66)
    act66 = Activation('relu')(bn66)

    # -----------------------------------------------------
    # --------------------- Bridge 6 ----------------------
    bridge_pool61 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act66)

    # Dense layers
    flt = Flatten()(bridge_pool61)

    dense1 = Dense(512, kernel_regularizer=l2(1e-5))(flt)
    dnbn1 = BatchNormalization(axis=1)(dense1)
    dnact1 = Activation('relu')(dnbn1)
    dndrop1 = Dropout(0.1)(dnact1)

    dense2 = Dense(512, kernel_regularizer=l2(1e-5))(dndrop1)
    dnbn2 = BatchNormalization(axis=1)(dense2)
    dnact2 = Activation('relu')(dnbn2)
    dndrop2 = Dropout(0.1)(dnact2)

    output = Dense(17, activation='sigmoid')(dndrop2)

    _model = Model(inputs=input, outputs=output)

    if weights_path:
        _model.load_weights(weights_path)

    return [_model, 'models/nm/structures/']
