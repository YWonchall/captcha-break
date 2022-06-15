from recognize import Recognize
import flowvision.transforms as transforms
import cv2 as cv

img = cv.imread('images/CP07.jpg')
img = cv.resize(img,(192, 64), interpolation=cv.INTER_AREA)
transf = transforms.ToTensor()
img_tensor = transf(img)/255  # 模型输入为0-1

reg = Recognize()
print(reg.recognize(img_tensor))