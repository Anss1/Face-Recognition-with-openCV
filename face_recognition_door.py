import numpy as np
import cv2
import face_recognition as fr
from glob import glob
import pickle
import utility
import define_constants as const

import RPi.GPIO as GPIO
import time

print('-----------------------------------------------------\n')

# Load data from pickle file (n_people)
with open('assets/pickles/n_people.pk', 'rb') as pickle_file:
	n_people_in_pickle = pickle.load(pickle_file)
print(f"Number of files that should be in '{const.PEOPLE_DIR}' directory : {n_people_in_pickle}")

# Read all images
##people = glob(const.PEOPLE_DIR + '/*.*')
people = glob(const.PEOPLE_DIR + '/*')
print(f"Number of files in '{const.PEOPLE_DIR}' directory : {len(people)}")

num_images = utility.get_num_of_images(people)

# Relay
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# Relay 1
RELAY = 17

GPIO.setup(RELAY, GPIO.OUT)
# Lock door at start
GPIO.output(RELAY,GPIO.LOW)

# Check if number of files in PEOPLE_DIR is same as value in pickle file
if n_people_in_pickle == len(people):
	# Get names
	names = list(map(utility.get_names, people))
	
	# Get encodings
	face_encode = np.load('assets/face_encodings/data.npy')
	
	# Constants for eye blink detection
	eye_blink_counter = 0
	eye_blink_total = 0
	random_blink_number = 3
	frame_current_name = None
	try:
		while True:
##			print(utility.distance())
			time.sleep(1)
			if utility.distance() <= 80:
				# Initiate Webcam
				print("\nInitiating camera...\n")
				cap = cv2.VideoCapture(const.n_camera)
				#============================================== START ========================================================
				t_end = time.time() + 60 * 2
##				print(t_end)
				while cap.isOpened() and time.time() < t_end:
##					print(time.time())
					# Read Frames
					ret, frame = cap.read()
					# Get Face locations, landmarks and encodings
					frame_face_loc = fr.face_locations(frame)
					frame_face_landmarks = fr.face_landmarks(frame, frame_face_loc)
					frame_face_encode = fr.face_encodings(frame, frame_face_loc)

					# Iterate through locations, landmarks and encodings
					for index, (loc, encode, landmark) in enumerate(zip(frame_face_loc, frame_face_encode, frame_face_landmarks)):

						# Find index match
						score = fr.face_distance(face_encode, encode)
						index_match = np.argmin(score)

						# Check if min(score) is < face_recognition_threshold
						if np.min(score) < const.face_recognition_threshold:
							
							# Store name temporarily to check if frame_current_name matches with temp_name
							temp_name = frame_current_name
							
							# Store new name
			##				frame_current_name = names[index_match]
							frame_current_name = names[int(index_match / num_images)] #sub by num of images for every person
						else:
							frame_current_name = "Unknown"

						# If frame_current_name is known
						if not frame_current_name == "Unknown":
							
							# Eye blink detection
							left_eye_points = np.array(landmark['left_eye'], dtype=np.int32)
							right_eye_points = np.array(landmark['right_eye'], dtype=np.int32)
							#Estimate EAR_ratio (eye aspect ratio)
							EAR_avg = ( utility.get_EAR_ratio(left_eye_points) + utility.get_EAR_ratio(right_eye_points) ) / 2
			##				print(EAR_avg)
							# Check if EAR ratio is less than threshold, eye is closed
							if EAR_avg < const.EAR_ratio_threshold:
								eye_blink_counter += 1
							else:
								# Check if counter is greater than min_frames_eyes_closed threshold
								if eye_blink_counter >= const.min_frames_eyes_closed:
									eye_blink_total += 1

								# Reset eye blink counter
								eye_blink_counter = 0

							# If temp_name doesn't matches with frame_current_name, reset eye_blink_total
							if temp_name != frame_current_name:
								eye_blink_total = 0

							# Set messages and face box color
							blink_message = f"Blinks:{eye_blink_total}"
							face_box_color = const.default_face_box_color

							# If random_blink_number and total blink number matches
							if random_blink_number == eye_blink_total:

								# open the door only if score is atmost 0.6 to check if he the same person or not
								if np.min(score) < const.face_recognition_threshold:
									# Unlock the door
									GPIO.output(RELAY, GPIO.HIGH)
									print('Relay 1 ON')
									time.sleep(5)
			##						utility.text_to_speech(f"Welcome to home,{frame_current_name}")# welcoming message
									face_box_color = const.success_face_box_color # Set face box color to green for one frame
									
									# Reset eye blink constants
									eye_blink_total = 0
									eye_blink_counter = 0

							# Draw Eye points and display blink_message
							cv2.polylines(frame, [left_eye_points], True, const.eye_color , 1)
							cv2.polylines(frame, [right_eye_points], True, const.eye_color , 1)
							cv2.putText(frame,blink_message,(10,50),cv2.FONT_HERSHEY_PLAIN,1.5,const.text_in_frame_color,2)

						# If frame_current_name is Unknown
						else:
							# Set face_box_color for unknown face
							face_box_color = const.unknown_face_box_color
							
							#save photo of unknown person
							cv2.imwrite(f"{const.Unknown_people}/Unknown_person.jpg",frame)
							
							#send an emergency E-mail to the home owner
			##                utility.send_emergency_Email()

						# Draw Reactangle around faces with their names
						cv2.rectangle(frame,(loc[3],loc[0]),(loc[1],loc[2]),face_box_color,2) # top-right, bottom-left
						cv2.putText(frame,frame_current_name,(loc[3],loc[0]-3),cv2.FONT_HERSHEY_PLAIN,2,const.text_in_frame_color,2)
					# Lock the door
					GPIO.output(RELAY, GPIO.LOW)
					print('Relay 1 OFF')
			##		time.sleep(5)
					# Display frame
					cv2.imshow("Webcam: Press \"q\" to quit", frame)
					if (cv2.waitKey(1) & 0xFF == ord('q')):
						break	
				#Cleanup
				cap.release()
				cv2.destroyAllWindows()
	# Reset by pressing CTRL + C
	except KeyboardInterrupt:
		print("stopped by User")
		GPIO.cleanup()
				
else:
	print(f"Retrain model to encode all faces in '{const.PEOPLE_DIR}' directory...")
