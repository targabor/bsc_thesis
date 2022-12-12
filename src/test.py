import cv2
from random import randint

# Load the image
image = cv2.imread("../images/szabivan.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Add impulse noise to the image
img_size = image.shape[1] * image.shape[0]
percentage = .05
for i in range(round(img_size*percentage)):
    x = randint(0, image.shape[1] - 1)
    y = randint(0, image.shape[0] - 1)
    image[y,x] = randint(0,255)

# Save the result
cv2.imshow("output.jpg", image)
cv2.waitKey()
cv2.destroyAllWindows()
