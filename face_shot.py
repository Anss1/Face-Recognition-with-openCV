import cv2

cam = cv2.VideoCapture(0)

path = "people/anas"
img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("press space to take a photo and ESC to close", frame)

    k = cv2.waitKey(1)
    if k%256 == 27: #ASCII for ESC
        # ESC pressed
        print("Escape hit, closing…")
        break
    elif k%256 == 32: #ASCII for SPACE
        # SPACE pressed
        img_name =f"{path}/image_{img_counter}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"{img_name} written!")
        img_counter += 1


cam.release()
cv2.destroyAllWindows()
