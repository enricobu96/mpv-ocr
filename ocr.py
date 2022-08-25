import cv2
import numpy as np
import pytesseract
import time
import spacy
# Params
RECOG_RATE = 1 # Recognition rate in seconds (1 second = 1 frame per second to perform ocr on)
nlp = spacy.load('en_core_web_sm')
SIMILARITY_TRESHOLD = .7


def get_fps(vid):
    return int(vid.get(cv2.CAP_PROP_FPS))

def check_similarity(t1, t2, nlp):
    t1 = nlp(t1)
    t2 = nlp(t2)
    return t1.similarity(t2) < SIMILARITY_TRESHOLD

# Read video file
vid = cv2.VideoCapture('./ex_videos/vid1.mp4')
success, image = vid.read()

# Get fps for frame skip
fps = get_fps(vid)
frame_counter = 0

# Ocr
prev_text = ''
while success:
    if frame_counter % fps*RECOG_RATE == 0:
        image = np.array(image)
        text = pytesseract.image_to_string(image)
        if check_similarity(prev_text, text, nlp):
            with open('./ex_videos/result.txt', 'a') as file:
                file.writelines(str(frame_counter/fps) + '\n')
                file.writelines(line+'\n' for line in text.split('\n'))
            prev_text = text

    success, image = vid.read()
    frame_counter += 1
    if frame_counter == 100:
        # print(text)
        break
