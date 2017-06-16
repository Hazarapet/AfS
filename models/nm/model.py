from keras.models import Model
from keras.layers import Input, concatenate
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


def model(weights_path=None):
    input = Input((3, 128, 128))

    # ------------------------------------------------------
    # ------------------ Conv Block 1 ----------------------
    # --------------------- 128x128 ------------------------
    conv11 = Conv2D(64, (3, 3), padding='same')(input)
    bn11 = BatchNormalization(axis=1)(conv11)
    act11 = Activation('relu')(bn11)

    concat11 = concatenate([input, act11], axis=1)

    conv12 = Conv2D(76, (3, 3), padding='same')(concat11)
    bn12 = BatchNormalization(axis=1)(conv12)
    act12 = Activation('relu')(bn12)

    concat12 = concatenate([concat11, act12], axis=1)

    conv13 = Conv2D(88, (3, 3), padding='same')(concat12)
    bn13 = BatchNormalization(axis=1)(conv13)
    act13 = Activation('relu')(bn13)

    concat13 = concatenate([concat12, act13], axis=1)

    conv14 = Conv2D(100, (3, 3), padding='same')(concat13)
    bn14 = BatchNormalization(axis=1)(conv14)
    act14 = Activation('relu')(bn14)

    concat14 = concatenate([concat13, act14], axis=1)

    conv15 = Conv2D(112, (3, 3), padding='same')(concat14)
    bn15 = BatchNormalization(axis=1)(conv15)
    act15 = Activation('relu')(bn15)

    concat15 = concatenate([concat14, act15], axis=1)

    conv16 = Conv2D(124, (3, 3), padding='same')(concat15)
    bn16 = BatchNormalization(axis=1)(conv16)
    act16 = Activation('relu')(bn16)

    # -----------------------------------------------------
    # --------------------- Bridge 1 ----------------------
    bridge_conv11 = Conv2D(124, (3, 3), padding='same')(act16)
    bridge_bn11 = BatchNormalization(axis=1)(bridge_conv11)
    bridge_act11 = Activation('relu')(bridge_bn11)

    bridge_pool11 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(bridge_act11)

    # ------------------------------------------------------
    # ------------------ Conv Block 2 ----------------------
    # ---------------------- 64x64 -------------------------
    conv21 = Conv2D(136, (3, 3), padding='same')(bridge_pool11)
    bn21 = BatchNormalization(axis=1)(conv21)
    act21 = Activation('relu')(bn21)

    concat21 = concatenate([bridge_pool11, act21], axis=1)

    conv22 = Conv2D(148, (3, 3), padding='same')(concat21)
    bn22 = BatchNormalization(axis=1)(conv22)
    act22 = Activation('relu')(bn22)

    concat22 = concatenate([concat21, act22], axis=1)

    conv23 = Conv2D(160, (3, 3), padding='same')(concat22)
    bn23 = BatchNormalization(axis=1)(conv23)
    act23 = Activation('relu')(bn23)

    concat23 = concatenate([concat22, act23], axis=1)

    conv24 = Conv2D(172, (3, 3), padding='same')(concat23)
    bn24 = BatchNormalization(axis=1)(conv24)
    act24 = Activation('relu')(bn24)

    concat24 = concatenate([concat23, act24], axis=1)

    conv25 = Conv2D(184, (3, 3), padding='same')(concat24)
    bn25 = BatchNormalization(axis=1)(conv25)
    act25 = Activation('relu')(bn25)

    concat25 = concatenate([concat24, act25], axis=1)

    conv26 = Conv2D(196, (3, 3), padding='same')(concat25)
    bn26 = BatchNormalization(axis=1)(conv26)
    act26 = Activation('relu')(bn26)

    concat26 = concatenate([concat25, act26], axis=1)

    conv27 = Conv2D(208, (3, 3), padding='same')(concat26)
    bn27 = BatchNormalization(axis=1)(conv27)
    act27 = Activation('relu')(bn27)

    concat27 = concatenate([concat26, act27], axis=1)

    conv28 = Conv2D(220, (3, 3), padding='same')(concat27)
    bn28 = BatchNormalization(axis=1)(conv28)
    act28 = Activation('relu')(bn28)

    concat28 = concatenate([concat27, act28], axis=1)

    conv29 = Conv2D(232, (3, 3), padding='same')(concat28)
    bn29 = BatchNormalization(axis=1)(conv29)
    act29 = Activation('relu')(bn29)

    concat29 = concatenate([concat28, act29], axis=1)

    conv210 = Conv2D(244, (3, 3), padding='same')(concat29)
    bn210 = BatchNormalization(axis=1)(conv210)
    act210 = Activation('relu')(bn210)

    # -----------------------------------------------------
    # --------------------- Bridge 2 ----------------------
    bridge_conv21 = Conv2D(244, (3, 3), padding='same')(act210)
    bridge_bn21 = BatchNormalization(axis=1)(bridge_conv21)
    bridge_act21 = Activation('relu')(bridge_bn21)

    bridge_pool21 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(bridge_act21)

    # ------------------------------------------------------
    # ------------------ Conv Block 3 ----------------------
    # ---------------------- 32x32 -------------------------
    conv31 = Conv2D(256, (3, 3), padding='same')(bridge_pool21)
    bn31 = BatchNormalization(axis=1)(conv31)
    act31 = Activation('relu')(bn31)

    concat31 = concatenate([bridge_pool21, act31], axis=1)

    conv32 = Conv2D(268, (3, 3), padding='same')(concat31)
    bn32 = BatchNormalization(axis=1)(conv32)
    act32 = Activation('relu')(bn32)

    concat32 = concatenate([concat31, act32], axis=1)

    conv33 = Conv2D(280, (3, 3), padding='same')(concat32)
    bn33 = BatchNormalization(axis=1)(conv33)
    act33 = Activation('relu')(bn33)

    concat33 = concatenate([concat32, act33], axis=1)

    conv34 = Conv2D(292, (3, 3), padding='same')(concat33)
    bn34 = BatchNormalization(axis=1)(conv34)
    act34 = Activation('relu')(bn34)

    concat34 = concatenate([concat33, act34], axis=1)

    conv35 = Conv2D(304, (3, 3), padding='same')(concat34)
    bn35 = BatchNormalization(axis=1)(conv35)
    act35 = Activation('relu')(bn35)

    concat35 = concatenate([concat34, act35], axis=1)

    conv36 = Conv2D(316, (3, 3), padding='same')(concat35)
    bn36 = BatchNormalization(axis=1)(conv36)
    act36 = Activation('relu')(bn36)

    concat36 = concatenate([concat35, act36], axis=1)

    conv37 = Conv2D(328, (3, 3), padding='same')(concat36)
    bn37 = BatchNormalization(axis=1)(conv37)
    act37 = Activation('relu')(bn37)

    concat37 = concatenate([concat36, act37], axis=1)

    conv38 = Conv2D(340, (3, 3), padding='same')(concat37)
    bn38 = BatchNormalization(axis=1)(conv38)
    act38 = Activation('relu')(bn38)

    concat38 = concatenate([concat37, act38], axis=1)

    conv39 = Conv2D(352, (3, 3), padding='same')(concat38)
    bn39 = BatchNormalization(axis=1)(conv39)
    act39 = Activation('relu')(bn39)

    concat39 = concatenate([concat38, act39], axis=1)

    conv310 = Conv2D(364, (3, 3), padding='same')(concat39)
    bn310 = BatchNormalization(axis=1)(conv310)
    act310 = Activation('relu')(bn310)

    # -----------------------------------------------------
    # --------------------- Bridge 3 ----------------------
    bridge_conv31 = Conv2D(364, (3, 3), padding='same')(act310)
    bridge_bn31 = BatchNormalization(axis=1)(bridge_conv31)
    bridge_act31 = Activation('relu')(bridge_bn31)

    bridge_pool31 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(bridge_act31)

    # ------------------------------------------------------
    # ------------------ Conv Block 4 ----------------------
    # ---------------------- 16x16 -------------------------
    conv41 = Conv2D(376, (3, 3), padding='same')(bridge_pool31)
    bn41 = BatchNormalization(axis=1)(conv41)
    act41 = Activation('relu')(bn41)

    concat41 = concatenate([bridge_pool31, act41], axis=1)

    conv42 = Conv2D(388, (3, 3), padding='same')(concat41)
    bn42 = BatchNormalization(axis=1)(conv42)
    act42 = Activation('relu')(bn42)

    concat42 = concatenate([concat41, act42], axis=1)

    conv43 = Conv2D(400, (3, 3), padding='same')(concat42)
    bn43 = BatchNormalization(axis=1)(conv43)
    act43 = Activation('relu')(bn43)

    concat43 = concatenate([concat42, act43], axis=1)

    conv44 = Conv2D(412, (3, 3), padding='same')(concat43)
    bn44 = BatchNormalization(axis=1)(conv44)
    act44 = Activation('relu')(bn44)

    concat44 = concatenate([concat43, act44], axis=1)

    conv45 = Conv2D(424, (3, 3), padding='same')(concat44)
    bn45 = BatchNormalization(axis=1)(conv45)
    act45 = Activation('relu')(bn45)

    concat45 = concatenate([concat44, act45], axis=1)

    conv46 = Conv2D(436, (3, 3), padding='same')(concat45)
    bn46 = BatchNormalization(axis=1)(conv46)
    act46 = Activation('relu')(bn46)

    concat46 = concatenate([concat45, act46], axis=1)

    conv47 = Conv2D(448, (3, 3), padding='same')(concat46)
    bn47 = BatchNormalization(axis=1)(conv47)
    act47 = Activation('relu')(bn47)

    concat47 = concatenate([concat46, act47], axis=1)

    conv48 = Conv2D(460, (3, 3), padding='same')(concat47)
    bn48 = BatchNormalization(axis=1)(conv48)
    act48 = Activation('relu')(bn48)

    concat48 = concatenate([concat47, act48], axis=1)

    conv49 = Conv2D(472, (3, 3), padding='same')(concat48)
    bn49 = BatchNormalization(axis=1)(conv49)
    act49 = Activation('relu')(bn49)

    concat49 = concatenate([concat48, act49], axis=1)

    conv410 = Conv2D(484, (3, 3), padding='same')(concat49)
    bn410 = BatchNormalization(axis=1)(conv410)
    act410 = Activation('relu')(bn410)

    # -----------------------------------------------------
    # --------------------- Bridge 4 ----------------------
    bridge_conv41 = Conv2D(484, (3, 3), padding='same')(act410)
    bridge_bn41 = BatchNormalization(axis=1)(bridge_conv41)
    bridge_act41 = Activation('relu')(bridge_bn41)

    bridge_pool41 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(bridge_act41)

    # Dense layers
    flt = Flatten()(bridge_pool41)

    dense1 = Dense(512, kernel_regularizer=l2(1e-5))(flt)
    # dnbn1 = BatchNormalization(axis=1)(dense1)
    dnact1 = Activation('relu')(dense1)
    dndrop1 = Dropout(0.2)(dnact1)

    output = Dense(17, activation='sigmoid')(dndrop1)

    _model = Model(inputs=input, outputs=output)

    if weights_path:
        _model.load_weights(weights_path)

    return [_model, 'models/nm/structures/']
