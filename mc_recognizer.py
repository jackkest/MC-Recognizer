import face_recognition as fr
import cv2
import os
import re
import time
import argparse
import json
import ffmpeg
import multiprocessing as mp
from tqdm import tqdm
from shutil import rmtree
from itertools import repeat
import matplotlib.pyplot as plt

def image_files_in_folder(folder):
	return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]

def scan_known_people(known_people_folder):
	known_names = []
	known_face_encodings = []

	for file in image_files_in_folder(known_people_folder):
		basename = os.path.splitext(os.path.basename(file))[0]
		img = fr.load_image_file(file)
		encodings = fr.face_encodings(img)

		if len(encodings) > 1:
			click.echo("WARNING: More than one face found in {}. Only considering the first face.".format(file))
		if len(encodings) == 0:
			click.echo("WARNING: No faces found in {}. Ignoring file.".format(file))
		else:
			known_names.append(basename)
			known_face_encodings.append(encodings[0])

	return known_names, known_face_encodings

def scan_image(image_to_check, known_names, known_face_encodings, tolerance=0.5):
	unknown_encoding = fr.face_encodings(image_to_check)[0]
	distances = fr.face_distance(known_face_encodings, unknown_encoding)
	return list(distances <= tolerance)

def draw_graph(count_dict, analyzed):
	count_dict['Analyzed'] = analyzed
	characters = list(count_dict.keys())
	fig, ax = plt.subplots()

	sorted_count_list = list(count_dict.items())
	sorted_count_list.sort(key=lambda x: x[1],reverse=True)
	x, y = zip(*sorted_count_list)

	ax.bar(x, y, label='Characters', color='blue')
	ax.set_ylabel('Number of Closeups')
	ax.set_title('Character Closeups')

	plt.show()



def main(videopath, closeup_percentage, scale_factor, check_every, group_number):
	returned = list(scan_known_people("Face_Images/"))

	if group_number == 0:
		rmtree("Matches/")
		os.mkdir("Matches/")
		print("\nDetecting Closeups...\n")

	if os.path.exists(videopath):

		vidcap = cv2.VideoCapture(videopath)
		frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
		frame_jump_unit =  frame_count // mp.cpu_count()
		vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_jump_unit * group_number)

		fps = vidcap.get(cv2.CAP_PROP_FPS)
		duration = frame_count/fps
		area = (vidcap.get(cv2.CAP_PROP_FRAME_WIDTH) * float(scale_factor)) * (vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) * float(scale_factor))
		vid_scale = area / (2073600 * float(scale_factor) * float(scale_factor))
		closeup_ratio = (int(closeup_percentage) / 100)

		success, image = vidcap.read()
		count = 0
		closeups_analyzed = 0
		closeup_count = {}
		for i in range(len(returned[0])):
			closeup_count[returned[0][i]] = 0
		col = "green"

		for i in tqdm(range(frame_jump_unit), colour=col, position=group_number, ncols=100, mininterval=0.5, leave=False, desc=(str(group_number))):
			if success:
				if count % int(check_every) == 0:
					face = cv2.resize(image, (0, 0), fx=float(scale_factor), fy=float(scale_factor))
					bound_box = fr.face_locations(face)

					if len(bound_box) == 1:
						face_area = (bound_box[0][1] - bound_box[0][3]) * (bound_box[0][1] - bound_box[0][3])
						ratio = face_area / area

						if ratio > closeup_ratio:
							closeups_analyzed += 1
							character = scan_image(face, returned[0], returned[1])
							for i in range(len(character)):
								if character[i]:
									name = returned[0][i]
									closeup_count[name] += 1
									break
									'''
									if count % 2 == 0:
										img = face
										cv2.rectangle(img, (bound_box[0][1], bound_box[0][0]), (bound_box[0][3], bound_box[0][2]), (0, 255, 0), 3)
										font = cv2.FONT_HERSHEY_DUPLEX
										if vid_scale < 1:
											cv2.putText(img, name, (bound_box[0][3] + 6, bound_box[0][2] - int((50 * ratio * vid_scale))), font, 2.0 * ratio, (255, 255, 255), 1)
										else:
											cv2.putText(img, name, (bound_box[0][3] + 6, bound_box[0][2] - int((50 * ratio * vid_scale))), font, 4.0 * ratio * (vid_scale), (255, 255, 255), 1)
										cv2.imwrite("Matches/" + str(chr(int(65 + group_number))) + str(count) + ".jpg", img)
									'''
					#vidcap.set(cv2.CAP_PROP_POS_FRAMES, (frame_jump_unit + int(check_every))) -> slows ~10x
					#frame_jump_unit += int(check_every)
					success, image = vidcap.read()
				count += 1
		
		vidcap.release()
		with open('Results/Thread' + str(group_number) + '_out.json', 'w') as file:
			file.write(json.dumps(closeup_count))

		return closeups_analyzed


if __name__ == "__main__":
	rmtree("Results/")
	os.mkdir("Results/")
	num_processes = mp.cpu_count()
	parser = argparse.ArgumentParser()
	parser.add_argument('--videopath')
	parser.add_argument('--closeup_percentage')
	parser.add_argument('--scale_factor')
	parser.add_argument('--check_every')
	args = parser.parse_args()
	os.system("clear")

	start_time = time.time()
	context = mp.Pool(num_processes)
	function_parameters = zip(repeat(args.videopath, num_processes),
							 repeat(args.closeup_percentage, num_processes),
							 repeat(args.scale_factor, num_processes), 
							 repeat(args.check_every, num_processes), 
							 range(num_processes))
	ret = context.starmap(main, function_parameters)
	closeups_analyzed = sum(ret)
	end_time = time.time()
	os.system("clear")
	total_processing_time = end_time - start_time
	out = {}
	names, encodings = scan_known_people("Face_Images/")

	for i in range(len(names)):
		out[names[i]] = 0
	for i in range(num_processes):
		f = open('Results/Thread' + str(i) + '_out.json')
		data = json.load(f)
		for j, (k, v) in enumerate(data.items()):
			out[k] += v
	
	cap = cv2.VideoCapture(args.videopath)
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	fps = cap.get(cv2.CAP_PROP_FPS)
	duration = frame_count/fps
	cap.release()

	graph_dict = {}

	print("\nTime taken: {}".format(total_processing_time))
	print("FPS : {}".format((frame_count / int(args.check_every))/total_processing_time))
	print("\nThis video is...\n")
	for i, (name, count) in enumerate(out.items()):
		percentage = round(((count / (frame_count / int(args.check_every)))), 3)
		character_time = round(percentage * duration, 2)
		if percentage != 0.000:
			graph_dict[name] = count
			print(str(round(percentage * 100, 2)) + "% " + name + " Closeups.")
			print("... Which amounts to about " + str(character_time) + " Seconds")
			print("... Or " + str(round(character_time / 60, 2))  + " Minutes\n")

	print(graph_dict)

	width = int((cap.get(cv2.CAP_PROP_FRAME_WIDTH) * float(args.scale_factor)))
	height = int((cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * float(args.scale_factor)))
	size = (width, height)
	draw_graph(graph_dict, closeups_analyzed)


	#out = (ffmpeg.input('Matches/*.jpg', pattern_type='glob', framerate=30).output('Output/out.mp4').run())