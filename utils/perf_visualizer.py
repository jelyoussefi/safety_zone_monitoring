import os
import cv2
import numpy as np


def draw_perf(image:np.ndarray, model_name, device, fps, infer_fps, cpu_load, data_type, async_mode):
	frame_size = image.shape[:-1]
	fontFace =  cv2.FONT_HERSHEY_SIMPLEX
	fontScale = 0.4
	thickness = 1
	margin = 15
	bcolor = (0,255,0)
	fcolor = (0,0,255)
	
	def circle(text, radius, pos, left=True, bcolor = (0,255,0), fcolor = (0,0,255), legend=""):
		textsize = cv2.getTextSize(text, fontFace, fontScale, thickness)[0]

		if left:
			x = margin + 2*(radius+5)*pos + radius/2
		else :
			x = frame_size[1] - margin - 2*(radius+5)*pos - radius/2

		center = (int(x), int(margin + radius / 2))
		cv2.circle(image, center, radius, bcolor, 1, cv2.LINE_AA)
		textPos = (int(center[0] - textsize[0]/2), int(center[1] + textsize[1]/2))
		cv2.putText(image, text, textPos, fontFace, fontScale, fcolor, thickness, cv2.LINE_AA)

		textsize = cv2.getTextSize(legend, fontFace, fontScale, thickness)[0]
		center = (int(x), int(margin + radius*2))
		textPos = (int(center[0] - textsize[0]/2), int(center[1] + textsize[1]/2))
		cv2.putText(image, legend, textPos, fontFace, 0.4, (255,255,255), thickness, cv2.LINE_AA)

	# device name & infer fps
	infer_fps = f"{int(infer_fps)}"
	circle(device, 18, 0)
	if not async_mode:
		circle(infer_fps, 18, 1, legend="inf. fps", fcolor=(255,0,0))

	# fps
	fps = f"{int(fps)}"
	circle(fps, 18, 1, False, legend="fps")

	#cpu load
	cpu_load = f"{int(cpu_load)}"
	circle(cpu_load, 18, 0, False, legend="%cpu", fcolor=(255,0,0))

	
	# model name
	info = f'{model_name} : {data_type} {" : Async" if async_mode else "" }'
	textsize = cv2.getTextSize(info, fontFace, fontScale, thickness)[0]
	center = (int( frame_size[1]/2), int(margin + textsize[1] / 2))
	cv2.rectangle(image, (int(center[0] - 8 - textsize[0]/2), margin), (int(center[0] + 8 + textsize[0]/2), margin + textsize[1] + 2*8), bcolor, 1, cv2.LINE_AA)
	textPos = (int(center[0] - textsize[0]/2), int(center[1] + textsize[1]/2 + 8))
	cv2.putText(image, info, textPos, fontFace, fontScale, fcolor, thickness, cv2.LINE_AA)


