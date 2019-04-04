import cv2
import numpy as np
import traceback

# Instantiate cascade classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Camera instance
cam = cv2.VideoCapture(0)

# Check if instantiation was successful
if not cam.isOpened():
    raise Exception("Could not open camera")
    


## USE OPENCV DOCUMENTATION TO FIND OUT HOW CERTAIN FUNCTIONS WORK.
## Your task is to implement real-time face point tracking. 
## A few tips:
##  You should start by implementing the detection part first. 
##  Try drawing the trackable points in the detection part without saving them 
##  to p0 so you're able to see if the point coordinates are correct.
##  When finding the good points in the tracking part, use isFound as an index
##  (you may have to convert this to a boolean array first).
##  If you want to draw points (i.e filled circles) with cv2.circle, you can
##  specify the thickness to be -1.



gray_prev = None  # previous frame
p0 = []  # previous points

frames = 0
while True:
    try:            
        # Get a single frame
        ret_val, img = cam.read()
        if not ret_val:
            break
        else:
            frames = frames+1
            # Mirror
            img = cv2.flip(img, 1)
            
            # Grayscale copy
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if frames % 100 == 0:
                p0 = []
                print(frames)
                frames = 0
                print(frames)
                print(p0)

            if len(p0) <= 5 :
                # Detection
                img = cv2.putText(img, 'Detection', (0,20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0,255,255))
                
                
                # Detect faces
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                
                # Take the first face and get trackable points.
                if len(faces) != 0:
                    # Extract ROI (face) from the grayscale frame
                    # Detections are in the form
                    # (x_upperleft, y_upperleft, width, height)
                    # You can also crop this ROI even more to make sure only 
                    # the face area is considered in the tracking.
                    
                    ##-your-code-starts-here-##
                    for x_upperleft, y_upperleft, width, height in faces:
                        roi_gray = gray[y_upperleft:y_upperleft + height, x_upperleft:x_upperleft + width]
                    ##-your-code-ends-here-##
                    
                    # Get trackable points
                    p0 = cv2.goodFeaturesToTrack(roi_gray, 
                                                        maxCorners=25, 
                                                        qualityLevel=0.01, 
                                                        minDistance=10)
                    
                    # Convert points to form (point_id, coordinates)
                    p0 = p0[:, 0, :]
                    # Convert from ROI to image coordinates
                    ##-your-code-starts-here-##
                    p0[:, 0] = p0[:, 0] + x_upperleft
                    p0[:, 1] = p0[:, 1] + y_upperleft
                    ##-your-code-ends-here-##

                # Save grayscale copy for next iteration
                gray_prev = gray.copy()
                
            else:
                # Tracking
                img = cv2.putText(img, 'Tracking', (0,20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0,255,255))
                
                # Calculate optical flow using calcOpticalFlowPyrLK
                p1, isFound, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray, p0, 
                                                       None, winSize = (21,21), 
                                                       maxLevel = 4, 
                                                       criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                
                # Select good points. Use isFound to select valid found points 
                # from p1.
                ##-your-code-starts-here-##
                p1 = [p1[i] for i in range(len(isFound)) if isFound[i] == 1]
                p1 = np.array(p1)
                ##-your-code-ends-here-##
                
                # Draw points using e.g. cv2.circle
                ##-your-code-starts-here-##
                for p in p1:
                    img = cv2.circle(img, (p[0], p[1]), radius=5, color=(0, 255, 255))
                ##-your-code-ends-here-##
                
                # Update p0 (which points should be kept?) and gray_prev for 
                # next iteration
                ##-your-code-starts-here-##
                p0 = p1
                gray_prev = gray.copy()
                ##-your-code-ends-here-##
    
            # Quit text
            img = cv2.putText(img, 'Press q to quit', (440, 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0,0,255))
            cv2.imshow('Video feed', img)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break # esc to quit
        
    # Catch exceptions in order to close camera and video feed window properly
    except:
        traceback.print_exc()  # display for user
        break
            
# Close camera and video feed window
cam.release()   
cv2.destroyAllWindows()

