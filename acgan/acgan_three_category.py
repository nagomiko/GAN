from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, ReLU
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

from data_loader import DataLoader

import numpy as np


class ACGAN():
    def __init__(self):
        # TODO したのができ次第 g d の初期化処理書き直し
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.hair_type_classes = 13
        self.hair_color_classes = 14
        self.eye_color_classes = 12
        self.num_classes = self.hair_type_classes + self.hair_color_classes + self.eye_color_classes

        self.latent_dim = 3900

        self.data_loader = DataLoader(img_res=(self.img_rows, self.img_cols))

        optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy',
                  'categorical_crossentropy']

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses,
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.num_classes,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, hair_type_label, hair_color_label, eye_color_label = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], [valid, hair_type_label, hair_color_label, eye_color_label])
        self.combined.compile(loss=losses,
                              optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(512 * 4 * 4, input_dim=self.latent_dim))
        model.add(BatchNormalization(momentum=0.8))
        model.add(ReLU())
        model.add(Reshape((4, 4, 512)))

        model.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(ReLU())

        model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(ReLU())

        model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(ReLU())

        model.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding='same'))

        model.add(Activation("tanh"))

        model.summary()
        # TODO inputのしかたを考えろ
        # TODO labelの形式 [0,1,0,0,0,1,0,0,0,1,0]のまとめるか[0,1,0][1,0,0][0,0,1]区切るか
        # TODO その場合のnoiseとの合わせ方考えろカス
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.num_classes,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, 100)(label))
        # label_embedding = Flatten()(label)
        # label_embedding = Dense(100,)(label)
        # label_embedding = Embedding(self.num_classes, 100)(label)

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=4, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        # model.add(Dropout(0.2))

        model.add(Conv2D(64, kernel_size=4, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(ZeroPadding2D(padding=((0, 1), (1, 0))))

        # model.add(Dropout(0.2))

        model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        # model.add(Dropout(0.2))

        model.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        # model.add(Dropout(0.2))

        model.add(Flatten())
        model.summary()

        img = Input(shape=self.img_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        # TODO おっけーのはず
        validity = Dense(1, activation="sigmoid")(features)
        label_hair_type = Dense(self.hair_type_classes + 1, activation="softmax")(features)
        label_hair_color = Dense(self.hair_color_classes + 1, activation="softmax")(features)
        label_eye_color = Dense(self.eye_color_classes + 1, activation="softmax")(features)
        # label = Dense(self.num_classes + 1, activation="sigmoid")(features)
        return Model(img, [validity, label_hair_type, label_hair_color, label_eye_color])

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        # (X_train, y_train), (_, _) = mnist.load_data()

        # Configure inputs
        # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # X_train = np.expand_dims(X_train, axis=3)
        # y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            imgs, hair_type, hair_color, eye_color = self.data_loader.load_data(batch_size)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            # idx = np.random.randint(0, X_train.shape[0], batch_size)
            # imgs = X_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # The labels of the digits that the generator tries to create an
            # image representation of
            # TODO generatorにいれるラベル作成
            # TODO ここ終わんないと下のConbined model作れない
            # 0~各classまでの数字を生成[[][]]
            sampled_hair_type = np.random.randint(0, self.hair_type_classes, (batch_size, 1))
            sampled_hair_color = np.random.randint(0, self.hair_color_classes, (batch_size, 1))
            sampled_eye_color = np.random.randint(0, self.eye_color_classes, (batch_size, 1))

            # Onehot化

            sampled_hair_type = np.array([self.label2onehot(i, 'hair_type') for i in sampled_hair_type], dtype='int32')
            sampled_hair_color = np.array([self.label2onehot(i, 'hair_color') for i in sampled_hair_color],
                                          dtype='int32')
            sampled_eye_color = np.array([self.label2onehot(i, 'eye_color') for i in sampled_eye_color], dtype='int32')

            # サンプルラベルを結合

            sampled_concat = np.concatenate([sampled_hair_type, sampled_hair_color, sampled_eye_color], axis=1)

            # Generate a half batch of new images

            gen_imgs = self.generator.predict([noise, sampled_concat])

            # Image labels. 0-9 if image is valid or 10 if it is generated (fake)
            # img_labels = y_train[idx]
            # リアル画像用のラベル作成　ラベルデータ＋後ろに0
            # TODO ここはいいはず
            fake_labels = np.array(np.zeros(shape=(batch_size, 1), dtype='int32'))
            hair_type = np.append(hair_type, fake_labels, axis=1)
            hair_color = np.append(hair_color, fake_labels, axis=1)
            eye_color = np.append(eye_color, fake_labels, axis=1)

            # 生成画像用のラベル作成 hair_type[0,~,1] hair_color[0,~,1] eye_color[0,~,1]

            # fake_labels = np.array([self.label2onehot_1(i) for i in fake_labels], dtype='int32')
            # TODO ここもいい
            fake_hair_type = np.array([self.label2onehot_fake('hair_type') for _ in range(batch_size)], dtype='int32')
            fake_hair_color = np.array([self.label2onehot_fake('hair_color') for _ in range(batch_size)], dtype='int32')
            fake_eye_color = np.array([self.label2onehot_fake('eye_color') for _ in range(batch_size)], dtype='int32')

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, hair_type, hair_color, eye_color])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs,
                                                            [fake, fake_hair_type, fake_hair_color, fake_eye_color])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # ラベルのケツに０いれて要素数を合わせる
            sampled_hair_type_0 = np.insert(sampled_hair_type, self.hair_type_classes, 0, axis=1)
            sampled_hair_color_0 = np.insert(sampled_hair_color, self.hair_color_classes, 0, axis=1)
            sampled_eye_color_0 = np.insert(sampled_eye_color, self.eye_color_classes, 0, axis=1)

            # Train the generator
            # TODO sampled_labelsをさくせい
            # TODO train_on_batchを作成
            #
            g_loss = self.combined.train_on_batch([noise, sampled_concat],
                                                  [valid, sampled_hair_type_0, sampled_hair_color_0,
                                                   sampled_eye_color_0])

            # Plot the progress
            # print("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (
            #     epoch, d_loss[0], 100 * d_loss[3], 100 * d_loss[4], g_loss[0]))

            print('epoch: %d' % epoch)
            print(d_loss)
            print(g_loss)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.save_model()
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        # TODO 上のやつができ次第作成

        # 0~各classまでの数字を生成[[][]]
        sampled_hair_type = np.random.randint(0, self.hair_type_classes, (r * c, 1))
        sampled_hair_color = np.random.randint(0, self.hair_color_classes, (r * c, 1))
        sampled_eye_color = np.random.randint(0, self.eye_color_classes, (r * c, 1))

        # Onehot化

        sampled_hair_type = np.array([self.label2onehot(i, 'hair_type') for i in sampled_hair_type], dtype='int32')
        sampled_hair_color = np.array([self.label2onehot(i, 'hair_color') for i in sampled_hair_color],
                                      dtype='int32')
        sampled_eye_color = np.array([self.label2onehot(i, 'eye_color') for i in sampled_eye_color], dtype='int32')

        # サンプルラベルを結合

        sampled_concat = np.concatenate([sampled_hair_type, sampled_hair_color, sampled_eye_color], axis=1)

        gen_imgs = self.generator.predict([noise, sampled_concat])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images1/%d.png" % epoch)
        plt.close()

    def save_model(self):

        self.generator.save_weights('./saved_model1/generator_weights_sig.h5')
        self.generator.save('./saved_model1/generator_models_sig.h5')
        # self.discriminator.save_weights(self.path + 'discriminator.h5')

    def label2onehot(self, label, label_type):
        if label_type == 'hair_type':
            onehot = np.zeros(self.hair_type_classes)
            onehot[label] = 1
            return onehot
        elif label_type == 'hair_color':
            onehot = np.zeros(self.hair_color_classes)
            onehot[label] = 1
            return onehot
        elif label_type == 'eye_color':
            onehot = np.zeros(self.eye_color_classes)
            onehot[label] = 1
            return onehot

    def label2onehot_fake(self, label_type):
        if label_type == 'hair_type':
            onehot = np.zeros(self.hair_type_classes + 1)
            onehot[self.hair_type_classes] = 1
            return onehot
        elif label_type == 'hair_color':
            onehot = np.zeros(self.hair_color_classes + 1)
            onehot[self.hair_color_classes] = 1
            return onehot
        elif label_type == 'eye_color':
            onehot = np.zeros(self.eye_color_classes + 1)
            onehot[self.eye_color_classes] = 1
            return onehot


if __name__ == '__main__':
    acgan = ACGAN()
    acgan.train(epochs=14000, batch_size=32, sample_interval=200)
