import sys
import logging
import os
import cv2
from utils import write_image, init_cam
from resnet50predictor import ResNet50Predictor
from cutlerypredictor import CutleryPredictor



if __name__ == "__main__":

    # folder to write images to
    out_folder = 'images' #sys.argv[1]

    # maybe you need this ???
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    logging.getLogger().setLevel(logging.INFO)

    # default no predictor
    predictor = None
   
    # also try out this resolution: 640 x 360
    webcam = init_cam(640, 480)
    key = None
    logging.info('q=quit, s=save_img')
    logging.info('1=disable predictions')
    logging.info('2=start CutleryPredictor (bases on MobileNet_V2)')
    logging.info('3=start ImageNetPredictor (ResNet50)')

    try:
        # q key not pressed 
        while key != 113: # q button

            # Capture frame-by-frame
            ret, frame = webcam.read()
            # fliping the image 
            frame = cv2.flip(frame, 1)
   
            # draw a [224x224] rectangle into the frame, leave some space for the black border 
            offset = 2
            width = 224
            x = 160
            y = 120
            cv2.rectangle(img=frame, 
                          pt1=(x-offset,y-offset), 
                          pt2=(x+width+offset, y+width+offset), 
                          color=(0, 0, 0), 
                          thickness=2
            )
            
            # get key event https://www.ascii-code.com/
            key = cv2.waitKey(5)

            if key == 49: # 1 key
                predictor = None
            
            if key == 50: # 2 key
                if not predictor:
                    predictor = CutleryPredictor()

            if key == 51: # 3 key
                if not predictor:
                    predictor = ResNet50Predictor()

            if key == 115: # s key
                # write the image without overlay
                # extract the [224x224] rectangle out of it
                image = frame[y:y+width, x:x+width, :]
                write_image(out_folder, image) 

            #if key == 32 or key == 112: # space bar, p key
            if predictor:
                image = frame[y:y+width, x:x+width, :]
                pname, pperc = predictor.predict(image)
                ptext = f'{pperc} %  {pname}'
                cv2.rectangle(frame, (120,360), (420, 400), (204, 204, 204), -1)
                cv2.putText(frame, ptext, (125, 390), 1, 2, (0, 0, 0), 2)

            # disable ugly toolbar
            cv2.namedWindow('frame', flags=cv2.WINDOW_GUI_NORMAL)
            
            # display the resulting frame
            cv2.imshow('frame', frame)
            
    finally:
        # when everything done, release the capture
        logging.info('quit webcam')
        webcam.release()
        cv2.destroyAllWindows()
