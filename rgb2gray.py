import imutils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from PIL import Image
import argparse
import dlib
import cv2, glob
variables = ["female", "male"]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)
def get_files(variable):
	files = glob.glob("./train_gender/%s/*" %variable)
	return files
i=1
for variable in variables:
	invariable = get_files(variable)
	for item in invariable:
		image = cv2.imread(item)
		#image = imutils.resize(image,width=256)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		clahe_image = clahe.apply(gray)
		#rects = detector(gray, 2)
		
		#for rect in rects:
		a = variable
			#(x, y, w, h) = rect_to_bb(rect)
			#faceAligned = fa.align(clahe_image, gray, rect)
		cv2.imwrite('./train_gender_gray/'+a+'/ali11gn1_%i.jpg' %i, clahe_image)
		i+=1
