import cv2
import os
import uuid

POSITIVE_LABEL_PATH = './data/positive/'
ANCHOR_LABEL_PATH = './data/anchor/'
IMAGE_SIZE = (250, 250)


def main():
	cap = cv2.VideoCapture(0)

	while cap.isOpened():
		ret, frame = cap.read()
		center_frame = frame[120:120+IMAGE_SIZE[0], 120:120+IMAGE_SIZE[1], :]

		# Collect anchor label images
		if cv2.waitKey(1) & 0XFF == ord('a'):
			img = os.path.join(ANCHOR_LABEL_PATH, f'{uuid.uuid1()}.jpg')
			cv2.imwrite(img, center_frame)

		# Collect positive label images
		if cv2.waitKey(1) & 0XFF == ord('p'):
			img = os.path.join(POSITIVE_LABEL_PATH, f'{uuid.uuid1()}.jpg')
			cv2.imwrite(img, center_frame)

		cv2.imshow('Center Image', center_frame)

		if cv2.waitKey(1) & 0XFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
