import cv2, os, glob
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Conv2D, Input, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from skimage.transform import pyramid_expand

# 아래 두 모듈은 별도 파일에 저장되어 있는 클래스 입니다.
from Subpixel import Subpixel
from DataGenerator import DataGenerator

base_path = 'C:/AI/SuperResolution/processed'
#x_train_list = sorted(glob.glob(os.path.join(base_path, 'x_train', '*.npy')))
#x_val_list = sorted(glob.glob(os.path.join(base_path, 'x_val', '*.npy')))
x_train_list_1 = sorted(glob.glob('C:/AI/SuperResolution/processed/x_train/*.npy'))
x_val_list_1 = sorted(glob.glob('C:/AI/SuperResolution/processed/x_val/*.npy'))


# 학습셋 162770 검증셋 19867 ==> 너무 커서 오래 걸리네요... 줄이겠습니다.. 정말 살인적이네요 ㅋㅋㅋ
x_train_list = x_train_list_1[:1000]
x_val_list = x_val_list_1[:200]
print(len(x_train_list), len(x_val_list))
print(x_train_list[0])

x1 = np.load(x_train_list[1])
x2 = np.load(x_val_list[1])

print(x1.shape, x2.shape)
plt.subplot(1, 2, 1)
plt.imshow(x1)
plt.subplot(1, 2, 2)
plt.imshow(x2)

train_gen = DataGenerator(list_IDs=x_train_list, labels=None, batch_size=16, dim=(44,44), n_channels=3, n_classes=None, shuffle=True)
val_gen = DataGenerator(list_IDs=x_val_list, labels=None, batch_size=16, dim=(44,44), n_channels=3, n_classes=None, shuffle=False)

upscale_factor = 4
inputs = Input(shape=(44, 44, 3))

net = Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')(inputs)
net = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(net)
net = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(net)
net = Conv2D(filters=upscale_factor**2, kernel_size=3, strides=1, padding='same', activation='relu')(net)
net = Subpixel(filters=3, kernel_size=3, r=upscale_factor, padding='same')(net)
outputs = Activation('relu')(net)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(train_gen, epochs=10, verbose=1, shuffle=True, callbacks=[
ModelCheckpoint('C:/AI/SuperResolution/models/model.h5', monitor='val_loss', verbose=1, save_best_only=True)])

x_test_list = sorted(glob.glob(os.path.join(base_path, 'x_test', '*.npy')))
y_test_list = sorted(glob.glob(os.path.join(base_path, 'y_test', '*.npy')))
print(len(x_test_list), len(y_test_list))
print(x_test_list[0])

test_idx = 21
x1_test = np.load(x_test_list[test_idx])
# 저해상도 이미지를 확대시킨 이미지
x1_test_resized = pyramid_expand(x1_test, 4, multichannel=True) # 색깔채널 조건 추가.
y1_test = np.load(y_test_list[test_idx])
# 모델이 예측한 이미지(output)
y_pred = model.predict(x1_test.reshape((1, 44, 44, 3)))
x1_test = (x1_test * 255).astype(np.uint8)
x1_test_resized = (x1_test_resized * 255).astype(np.uint8)
y1_test = (y1_test * 255).astype(np.uint8)
y_pred = np.clip(y_pred.reshape((176, 176, 3)), 0, 1)
x1_test = cv2.cvtColor(x1_test, cv2.COLOR_BGR2RGB)
x1_test_resized = cv2.cvtColor(x1_test_resized, cv2.COLOR_BGR2RGB)
y1_test = cv2.cvtColor(y1_test, cv2.COLOR_BGR2RGB)
y_pred = cv2.cvtColor(y_pred, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(15, 10))
plt.subplot(1, 4, 1)
plt.title('input')
plt.imshow(x1_test)
plt.subplot(1, 4, 2)
plt.title('resized')
plt.imshow(x1_test_resized)
plt.subplot(1, 4, 3)
plt.title('output')
plt.imshow(y_pred)
plt.subplot(1, 4, 4)
plt.title('groundtruth')
plt.imshow(y1_test)