import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import numpy as np
from pathlib import Path
import random

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(176, 200, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(4, activation='softmax'))

learning_rate = 0.0001
opt = keras.optimizers.adam(lr=learning_rate, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

train_data_dir = Path('train_data')
model_data_dir = Path('model')
model_data_dir.mkdir(parents=True, exist_ok=True)
hm_epochs = 200


def check_data():
    choices = {"no_attacks": no_attacks,
               "attack_closest_to_nexus": attack_closest_to_nexus,
               "attack_enemy_structures": attack_enemy_structures,
               "attack_enemy_start": attack_enemy_start}

    total_data = 0

    lengths = []
    for choice in choices:
        print("Length of {} is: {}".format(choice, len(choices[choice])))
        total_data += len(choices[choice])
        lengths.append(len(choices[choice]))

    print("Total data length now is:", total_data)
    return lengths


all_files = list(train_data_dir.glob("*.npy"))

no_attacks = []
attack_closest_to_nexus = []
attack_enemy_structures = []
attack_enemy_start = []
for file in all_files:
    data = list(np.load(file.absolute()))
    for d in data:
        choice = np.argmax(d[0])
        if choice == 0:
            no_attacks.append(d)
        elif choice == 1:
            attack_closest_to_nexus.append(d)
        elif choice == 2:
            attack_enemy_structures.append(d)
        elif choice == 3:
            attack_enemy_start.append(d)

lengths = check_data()
lowest_data = min(lengths)

no_attacks = no_attacks[:lowest_data]
attack_closest_to_nexus = attack_closest_to_nexus[:lowest_data]
attack_enemy_structures = attack_enemy_structures[:lowest_data]
attack_enemy_start = attack_enemy_start[:lowest_data]

train_data = no_attacks + attack_closest_to_nexus + attack_enemy_structures + attack_enemy_start
test_size = len(train_data) // 10
batch_size = 250

x_train = np.array([i[1] for i in train_data[:-test_size]]).reshape(-1, 176, 200, 3)
y_train = np.array([i[0] for i in train_data[:-test_size]])

x_test = np.array([i[1] for i in train_data[-test_size:]]).reshape(-1, 176, 200, 3)
y_test = np.array([i[0] for i in train_data[-test_size:]])

model.fit(x_train, y_train,
          epochs=hm_epochs,
          batch_size=batch_size,
          validation_data=(x_test, y_test),
          shuffle=True,
          verbose=1,
          callbacks=[TensorBoard(log_dir=f"logs/stage1-e{hm_epochs}-lr{learning_rate}")])

model.save(model_data_dir / f"BasicCNN-model-{hm_epochs}-epochs-{learning_rate}-LR-STAGE1.h5")
model.save_weights(model_data_dir / f"BasicCNN-weights-{hm_epochs}-epochs-{learning_rate}-LR-STAGE1.h5")
