import cv2
import winsound
 
scale_factor = 1.2
min_neighbors = 3
min_size = (50, 50)
webcam=True 
 
def detect(path):
 
    cascade = cv2.CascadeClassifier(path)
    #if webcam:
        #video_cap = cv2.VideoCapture(0) 
    #else:
    video_cap = cv2.VideoCapture("http://192.168.43.3:8000/stream.mjpg")
    while True:
        
        ret, img = video_cap.read()
 
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
        rects = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
                                         minSize=min_size)
        
        if len(rects) >= 0:
            
            for (x, y, w, h) in rects:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                
                Facefilename= "unknown\\face_" + str(y) + ".jpg"
                cv2.imwrite(Facefilename,img)
                winsound.Beep(3000, 10)
 
            
            cv2.imshow('Face Detection on Video', img)
            
            if cv2.waitKey(1) & 0xFF == ord('c'):
                break
    video_cap.release()
 
def main():
    cascadeFilePath="haarcascade_frontalface_alt.xml"
    detect(cascadeFilePath)
    cv2.destroyAllWindows()
 
 
main()