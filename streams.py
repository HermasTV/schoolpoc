import cv2
import time


class CameraStream:
    # init the video camera stream / video file
    def __init__(self, src=0, name="CameraStream",location = "entrance",visualize=False):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.name = name
        self.location = location
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.frame_count = 0
        self.time_stamp = time.time()
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)

        self.presence = {}

        if not self.grabbed:
            print("Stream not initialized", self.name, self.location)
            self.stop()
            return
        else :
            print("Stream initialized", self.name, self.location, self.fps)

        if visualize:
            self.outstream = cv2.VideoWriter('logs/'+self.name+'_out.avi', cv2.VideoWriter_fourcc(*'MP4V'), self.fps, (self.frame.shape[1], self.frame.shape[0]))

    def update_frame(self):
        # if the thread indicator variable is set, stop the thread
        if self.stopped:
            return
        # otherwise, read the next frame from the stream
        self.grabbed, self.frame = self.stream.read()
        if not self.grabbed:
            self.stop()
            return
        self.frame_count += 1
        self.time_stamp = time.time()

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.stream.release()

    def is_stopped(self):
        return self.stopped

    def write_frame(self):
        self.outstream.write(self.frame)

    def release_stream(self):
        self.outstream.release()

if __name__ == '__main__':
    stream_vids = ["streams/entrance1.mp4", "streams/exit1.mp4"]
    videos = [CameraStream(src=stream_vids[0], name="Entrance", location="entrance"),
                CameraStream(src=stream_vids[1], name="Exit", location="exit")]
    while True:
        for video in videos:
            video.get_frame()
            # cv2.imshow(video.name, video.frame)

        key = cv2.waitKey(1) & 0xFF
        if any([video.is_stopped() for video in videos]):
            break