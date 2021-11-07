import os

os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import Configs.config as config
from Siamese_Model.siameseModel import Siamese
from Siamese_Model.Layers.layers import EmbeddingBlock, CNNBlock, L1Dist
from Utils.preprocess import *
import shutil
import glob
import random
import cv2
import numpy as np


def predict(image_path, model, detection_threshold, verification_threshold):
	'''
	image -> path to image that want to predict
	model -> model
	detection_threshold -> [0,1] Metric above which a prediction is considered positive
	verification_threshold -> [0,1] Propotion of positive predictions needed for a positive label
	'''
	results = []

	input_img = preprocess(image_path)
	input_img = np.expand_dims(input_img, axis=0)

	for image in random.sample(glob.glob(f'{config.POSITIVE_LABEL_PATH}*.jpg'), config.ANCHOR_NR_IMAGES_FOR_PREDICTION):
		anchor_img = preprocess(image)
		anchor_img = np.expand_dims(anchor_img, axis=0)

		# Make prediction
		result = model(input_img, anchor_img)
		results.append(result)

	detection = np.sum(np.array(results) > detection_threshold)

	verified = (detection/config.ANCHOR_NR_IMAGES_FOR_PREDICTION) > verification_threshold

	return results, (detection/config.ANCHOR_NR_IMAGES_FOR_PREDICTION), verified


def main():
	cap = cv2.VideoCapture(0)

	model = tf.keras.models.load_model(
		f'./saved-model/siamese-model-1636298745', 
		custom_objects={
			'Siamese_Model':Siamese,
			'EmbeddingBlock':EmbeddingBlock,
			'CNNBlock':CNNBlock,
			'L1Dist':L1Dist
		}
	)

	while cap.isOpened():
		ret, frame = cap.read()
		center_frame = frame[120:120+250, 120:120+250, :]

		# Predict Image
		if cv2.waitKey(1) & 0XFF == ord('p'):
			print('Start prediction...')
			# It is saving than load the image again inside the "predict" method... TODO better 
			cv2.imwrite('./predict.jpg', center_frame)

			results, aux, verified = predict('./predict.jpg', model, config.DETECTION_THRESHOLD, config.VERIFICATION_THRESHOLD)

			print(aux)

			if verified:
				predicted_frame = cv2.rectangle(center_frame, (0,0), center_frame.shape[:2], (0,255,0), 2)
			else:
				predicted_frame = cv2.rectangle(center_frame, (0,0), center_frame.shape[:2], (0,0,255), 2)
			cv2.imshow('Prediction Results', predicted_frame)

		cv2.imshow('Center Image', center_frame)

		if cv2.waitKey(1) & 0XFF == ord('q'):
			break

	os.remove('./predict.jpg')
	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
