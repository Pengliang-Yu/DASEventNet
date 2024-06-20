#!/usr/bin/env python
# coding: utf-8

import os
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
from datetime import datetime, timedelta
from obspy import UTCDateTime
import scipy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization,
                                     Activation, AveragePooling2D, GlobalAveragePooling2D, Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
import pickle
import tensorflow as tf

def load_data(file_path, event_files, noise_files):
    def load_pickle_data(file_list):
        all_list = []
        for file in file_list:
            with open(file_path + file, 'rb') as fp:
                data_list = pickle.load(fp)
                all_list.extend([data[:,:2000] for data in data_list if data.ndim > 1 and data.shape[1] == 2000])
        return all_list

    event_all_list = load_pickle_data(event_files)
    noise_all_list1 = load_pickle_data(noise_files[0])
    noise_all_list2 = load_pickle_data(noise_files[1])
    noise_all_list3 = load_pickle_data(noise_files[2])
    
    return event_all_list, noise_all_list1, noise_all_list2, noise_all_list3

def remove_invalid_samples(all_list):
    return [data for data in all_list if data.shape[1] == 2000]

def shuffle_data(data):
    np.random.seed(20)
    idx = np.random.permutation(len(data))
    return np.array(data)[idx]

def prepare_datasets(event_list, noise_list1, noise_list2, noise_list3):
    f0, f1 = int(len(event_list) * 0.75), int(len(event_list) * 0.9)
    n1f0, n2f0, n3f0 = int(len(noise_list1) * 0.75), int(len(noise_list2) * 0.75), int(len(noise_list3) * 0.75)
    n1f1, n2f1, n3f1 = int(len(noise_list1) * 0.9), int(len(noise_list2) * 0.9), int(len(noise_list3) * 0.9)

    event_train, event_test, event_eval = event_list[:f0], event_list[f0:f1], event_list[f1:]
    X_train = np.concatenate([event_train, noise_list1[:n1f0], noise_list2[:n2f0], noise_list3[:len(event_train) - n1f0 - n2f0]], axis=0)
    Y_train = np.array([1] * len(event_train) + [0] * len(event_train))

    s0 = len(event_train) - n1f0 - n2f0
    s1 = len(event_test) + len(event_train) - n1f0 - n2f0 - (n1f1 - n1f0) - (n2f1 - n2f0)
    X_test = np.concatenate([event_test, noise_list1[n1f0:n1f1], noise_list2[n2f0:n2f1], noise_list3[s0:s1]], axis=0)
    Y_test = np.array([1] * (f1 - f0) + [0] * (f1 - f0))

    X_eval = np.concatenate([event_eval, noise_list1[n1f1:], noise_list2[n2f1:], noise_list3[s1:s1 + len(event_eval) - (len(noise_list1) - n1f1) - (len(noise_list2) - n2f1)]], axis=0)
    Y_eval = np.array([1] * (len(event_list) - f1) + [0] * (len(event_list) - f1))

    return shuffle_data(X_train), shuffle_data(Y_train), shuffle_data(X_test), shuffle_data(Y_test), shuffle_data(X_eval), shuffle_data(Y_eval)

def build_model(input_shape):
    base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def lr_schedule(epoch):
    lr = 1e-4
    if epoch > 20:
        lr *= 0.1
    elif epoch > 10:
        lr *= 0.5
    print('Learning rate:', lr)
    return lr

def compile_and_train_model(model, X_train, Y_train, X_test, Y_test, save_path):
    optimizer = Adam(learning_rate=lr_schedule(0))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    lr_scheduler = LearningRateScheduler(lr_schedule)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, min_delta=0.0001)
    checkpoint = ModelCheckpoint(save_path + '/model_norm2_expand_shuffle_LR_DP_noise.h5', save_best_only=True)

    history = model.fit(X_train, Y_train, batch_size=16, epochs=100, validation_data=(X_test, Y_test), callbacks=[lr_scheduler, early_stopping, checkpoint])
    return history

def plot_metrics(history, save_path):
    metrics = ['loss', 'val_loss', 'accuracy', 'val_accuracy']
    labels = ['Train Loss', 'Validation Loss', 'Train Accuracy', 'Validation Accuracy']

    for metric, label in zip(metrics, labels):
        plt.plot(history.history[metric], label=label)
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel(label)
        plt.savefig(f"{save_path}/{label.replace(' ', '_').lower()}_jupyter9_norm2_expand_shuffle.png", dpi=200, format='png')
        plt.close()

def evaluate_model(model, X, Y):
    loss, accuracy = model.evaluate(X, Y)
    print(f"Loss: {loss}, Accuracy: {accuracy}")
    return loss, accuracy

def main():
    file_path = r'/storage/group/cjm38/default/Pengliang/'
    eventnorm_file_list = ['event0_180_norm.pkl', 'event180_360_norm.pkl', 'event360_540_norm.pkl', 'event540_720_norm.pkl', 'event720_900_norm.pkl', 'event900_1080_norm.pkl', 'event1080_1309_norm.pkl']
    noise_norm_file_list1 = ['noise180_norm.pkl']
    noise_norm_file_list2 = ['noise180_norm_more.pkl']
    noise_norm_file_list3 = ['noise180_norm_Bshear.pkl', 'noise360_norm_Bshear.pkl', 'noise540_norm_Bshear.pkl', 'noise720_norm_Bshear.pkl', 'noise900_norm_Bshear.pkl', 'noise1080_norm_Bshear.pkl']
    
    event_all_list, noise_all_list1, noise_all_list2, noise_all_list3 = load_data(file_path, eventnorm_file_list, [noise_norm_file_list1, noise_norm_file_list2, noise_norm_file_list3])
    
    event_all_list = remove_invalid_samples(event_all_list)
    noise_all_list1 = remove_invalid_samples(noise_all_list1)
    noise_all_list2 = remove_invalid_samples(noise_all_list2)
    noise_all_list3 = remove_invalid_samples(noise_all_list3)
    
    X_train, Y_train, X_test, Y_test, X_eval, Y_eval = prepare_datasets(event_all_list, noise_all_list1, noise_all_list2, noise_all_list3)
    
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape)
    
    save_path = r'/storage/group/cjm38/default/Pengliang/noise8_global2'
    os.makedirs(save_path, exist_ok=True)
    
    history = compile_and_train_model(model, X_train, Y_train, X_test, Y_test, save_path)
    
    plot_metrics(history, save_path)
    
    model = load_model(save_path + '/model_norm2_expand_shuffle_LR_DP_noise.h5')
    
    print('Evaluating model on training data:')
    evaluate_model(model, X_train, Y_train)
    
    print('Evaluating model on test data:')
    evaluate_model(model, X_test, Y_test)
    
    print('Evaluating model on evaluation data:')
    evaluate_model(model, X_eval, Y_eval)
    
    print(model.summary())

if __name__ == "__main__":
    main()
