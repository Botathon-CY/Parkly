import time
import cv2
import numpy as np
import mrcnn.utils
from mrcnn.model import MaskRCNN
from compute import COCO_MODEL_PATH, MODEL_DIR, MaskRCNNConfig, get_car_boxes, space_Violation
from parking_state import Settings


def worker4():

    # Video file or camera to process - set this to 0 to use your webcam instead of a video file
    VIDEO_SOURCE = "http://109.236.111.203:90/mjpg/video.mjpg"

    # Physical capacity of area

    TOTAL_PARKING_CAPACITY = 70

    # Create a Mask-RCNN model in inference mode
    model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

    # Load pre-trained model
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # Location of parking spaces
    parked_car_boxes = None

    # Load the video file we want to run detection on
    video_capture = cv2.VideoCapture(VIDEO_SOURCE)
    cv2.waitKey(33)

    frame_counter = 1
    has_space = False
    # Loop over each frame of video

    while video_capture.isOpened():

        success, frame = video_capture.read()
        if not success:
            break

        parking_areas = np.array([[304, 556, 359, 650]])

        overlaps = parking_areas

        if(frame_counter % 100 == 0):

            print("----------------------------------------------")
            print("Start detecting cars ....")

            # Capture frame-by-frame
            start_time = time.time()

            # Convert the image from BGR color (which OpenCV uses) to RGB color
            rgb_image = frame[:, :, ::-1]

            # Run the image through the Mask R-CNN model to get results.
            results = model.detect([rgb_image], verbose=0)

            # Mask R-CNN assumes we are running detection on multiple images.
            # We only passed in one image to detect, so only grab the first result.
            r = results[0]

            # The r variable will now have the results of detection:
            # - r['rois'] are the bounding box of each detected object
            # - r['class_ids'] are the class id (type) of each detected object
            # - r['scores'] are the confidence scores for each detection
            # - r['masks'] are the object masks for each detected object (which gives you the object outline)

            # Filter the results to only grab the car / truck bounding boxes
            car_boxes = get_car_boxes(r['rois'], r['class_ids'])
            print("car_boxes")
            # print(car_boxes)
            print("Cars found in frame of video:")

            # Draw each box on the frame
            i = 1
            for box in car_boxes:
                print("Car ", i, ":", box)

                y1, x1, y2, x2 = box

                # Draw the box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                cv2.putText(frame, "%s" % str(i), (x1, y1), cv2.LINE_AA, 1, (0, 0, 255))
                cv2.circle(frame, (x1, y1), 5, (0, 0, 255), -1)

                i = i + 1

            # See how much cars overlap with the known parking spaces
            print("parking_areas")
            print(parking_areas)

            overlaps = mrcnn.utils.compute_overlaps(car_boxes, parking_areas) # parking_areas

            print(len(overlaps.tolist()))

            print("Checking overlaps .... frame %d" % frame_counter)
            print(overlaps)
            # print(overlaps)
            print(overlaps < 0.5)
            result = space_Violation(overlaps)


            if result < 2:
                print("Free Parking Spaces")
                has_space =  True
                cv2.putText(frame, "Parking Spaces Available : %s" % str(TOTAL_PARKING_CAPACITY - result),  (10, 50), cv2.LINE_AA, 1, (100, 255, 0))
                # Add Time Stamp
                time_stamp = time.strftime("%Y/%m/%d %H:%M:%S %Z", time.localtime())
                cv2.putText(frame, time_stamp, (950, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0, 0), 1, cv2.LINE_AA)
            else:
                has_space = False
                cv2.putText(frame,"Don't Have Parking Spaces", (10, 50), cv2.LINE_AA, 1, (0, 0, 255))

                # Add Time Stamp
                time_stamp = time.strftime("%Y/%m/%d %H:%M:%S %Z", time.localtime())
                cv2.putText(frame, time_stamp, (950, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0, 0), 1, cv2.LINE_AA)

            #cv2.imwrite("analyze/output/frame%d.jpg" % frame_counter, frame), for debug

            cv2.imwrite("analyze/output/frame-analyzed-4.jpg", frame)

            # Show the frame of video on the screen, for debug
            # cv2.imshow('Video', frame)
            #add a sleep for demo
            # time.sleep(5)

            feed2 = Settings()
            feed2.device.state.update({'SOUTH': TOTAL_PARKING_CAPACITY - result})

            print("-----STATE--------")
            print(feed2.device.state)

        if has_space:
            # print("Free Parking Spaces")
            # TODO - Push to DB
            cv2.putText(frame, "Free Parking Spaces", (10, 50), cv2.LINE_AA, 1, (0,255,0))
        else:
            # TODO - Push to DB
            cv2.putText(frame, "Don't Have Free Parking Spaces", (10, 50), cv2.LINE_AA, 1, (0,0,255))

        # Show the frame of video on the screen, for debug
        # cv2.imshow('Video', frame)

        frame_counter = frame_counter + 1

        #Hit 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Clean up everything when finished
    video_capture.release()
    cv2.destroyAllWindows()