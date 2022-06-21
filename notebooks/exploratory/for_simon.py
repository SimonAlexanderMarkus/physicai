import cv2
import warnings
import glob

# ignore warnings
warnings.filterwarnings('ignore')

#---------------------------------------------
# initialize video frame capture using OpenCV2
#---------------------------------------------

# path where the movie is stored
path = glob.glob("./videos/yoga_videos_Sha/*")

cap = cv2.VideoCapture(path[3])

# Obtain frame size information using get() method
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outvideo.avi' file:
# ----------------------------------------------------------------------------------------
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# fourcc = cv2.VideoWriter_fourcc('X','2','6','4')
# fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter('./output_video/outvideo'+'.avi',fourcc, 25, (frame_width,frame_height))

capture_frames = 0 # if you want to save each frame as an image, set this to 1

num=0

while cap.isOpened():
    
    # read frame
    ret, frame = cap.read()
        
    if ret == True:

        disp_image = frame.copy()

        # make sure frame has the correct color
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Write the frame into the file 'output.avi'
        cv2.imshow('MoveNet singlepose', frame)

        out.write(frame)

        if capture_frames == 1:
            num += 1 
            cv2.imwrite('./frames/Frame'+format(str(num).zfill(3))+'.jpg', frame)           
            
        # define brake-out: if we hit "q" on the keyboard
        # frame capure is stopped
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break # break out of the while loop

    else:
        break

cap.release() # release the camera
out.release() # release the video writer

cv2.destroyAllWindows() # close all open windows