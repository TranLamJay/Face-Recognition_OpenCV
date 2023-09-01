import cv2
import face_recognition
from training import CheckImage

image = "images/lamjay.jpg"

def faceRecognition(image):
  # Upload file ảnh 
  imgRoot = face_recognition.load_image_file(image)
  imgRoot = cv2.cvtColor(imgRoot,cv2.COLOR_BGR2RGB) # chuyển đổi BGR -> RGB

  # Xác đinh vị trí khuôn mặt cần nhận dạng
  faceLocation = face_recognition.face_locations(imgRoot)[0]

  print(faceLocation) #(y1,x2,y2,x1)
  # encodeRoot = face_recognition.face_encodings(imgRoot)[0]
  # print(encodeRoot)

  result = CheckImage(image)
  print(result)

  cv2.rectangle(imgRoot,(faceLocation[3],faceLocation[0]),(faceLocation[1],faceLocation[2]),(0,255,0),2)
  cv2.putText(imgRoot,f"{result[0]}",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

  cv2.imshow(result[0],imgRoot)  # view thử ảnh để kiểm tra
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
faceRecognition(image)