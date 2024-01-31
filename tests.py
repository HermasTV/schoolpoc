# import cv2
# vid1 = cv2.VideoCapture('rtsp://admin:HLM@1234@192.168.1.101')
# vid2 = cv2.VideoCapture("rtsp://admin:HLM@1234@192.168.1.102")
# vid3 = cv2.VideoCapture("rtsp://admin:HLM@1234@192.168.1.103")
# vid4 = cv2.VideoCapture("rtsp://admin:HLM@1234@192.168.1.104")

# # print(vid1.read())
# while True:
#     ret1, frame1 = vid1.read()
#     ret2, frame2 = vid2.read()
#     ret3, frame3 = vid3.read()
#     ret4, frame4 = vid4.read()
#     print(frame1.shape)
#     cv2.imshow("frame1", frame1)
#     cv2.imshow("frame2", frame2)
#     cv2.imshow("frame3", frame3)
#     cv2.imshow("frame4", frame4)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
# vid1.release()
# vid2.release()
# vid3.release()
# vid4.release()



from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import time 
import cv2

def process_video(vid_path):
    vid = cv2.VideoCapture(vid_path)
    while True:
        ret, frame = vid.read()
        if not ret:
            break

    vid.release()
    

if __name__ == "__main__":
    start = time.time()
    with ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(process_video, ["./streams/Rand-3.mp4"]*4)
    
    print(f"took {time.time()-start} seconds")