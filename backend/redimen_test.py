import cv2
import os
def redimension( image):

    height, width = int(image.shape[0]), int(image.shape[1])
    new_width = int((255 / height) * width)
    resized_image = cv2.resize(image, (new_width, 255), interpolation=cv2.INTER_AREA)

    return  resized_image

def readImage(filename,imagespath):
    # OpenCV uses BGR channels
    img = cv2.imread(os.path.join(imagespath,filename))
    return img

directory = r'C:\Users\Ruben\Desktop\sistema-logo'


#os.chdir(directory)
imagespath = os.path.join(directory,"models","tmp", "train", "images") 

print(imagespath)

for filename in os.listdir(imagespath):

    image = readImage(filename,imagespath)
    resized = redimension(image)

    image_path = os.path.join(directory,"models","tmp",  "prueba",filename)

        # Guarda la imagen en la ruta especificada
    cv2.imwrite(image_path, resized)


