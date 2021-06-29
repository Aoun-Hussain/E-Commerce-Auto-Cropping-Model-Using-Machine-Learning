# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def version1():
    '''
    Sources:
    http://pythonpath.wordpress.com/2012/05/08/pil-to-opencv-image/
    http://www.lucaamore.com/?p=638
    '''

    # Python 2.7.2
    # Opencv 2.4.2
    # PIL 1.1.7

    import cv2 as cv
    import image
    from PIL import Image


    def DetectFace(image, faceCascade):
        # modified from: http://www.lucaamore.com/?p=638

        min_size = (20, 20)
        image_scale = 1
        haar_scale = 1.1
        min_neighbors = 3
        haar_flags = 0

        # Allocate the temporary images
        smallImage = cv.CreateImage(
            (
                cv.Round(image.width / image_scale),
                cv.Round(image.height / image_scale)
            ), 8, 1)

        # Scale input image for faster processing
        cv.Resize(image, smallImage, cv.CV_INTER_LINEAR)

        # Equalize the histogram
        cv.EqualizeHist(smallImage, smallImage)

        # Detect the faces
        faces = cv.HaarDetectObjects(
            smallImage, faceCascade, cv.CreateMemStorage(0),
            haar_scale, min_neighbors, haar_flags, min_size
        )

        # If faces are found
        if faces:
            for ((x, y, w, h), n) in faces:
                # the input to cv.HaarDetectObjects was resized, so scale the
                # bounding box of each face and convert it to two CvPoints
                pt1 = (int(x * image_scale), int(y * image_scale))
                pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
                cv.Rectangle(image, pt1, pt2, cv.RGB(255, 0, 0), 5, 8, 0)

        return image

    def pil2cvGrey(pil_im):
        # from: http://pythonpath.wordpress.com/2012/05/08/pil-to-opencv-image/
        pil_im = pil_im.convert('L')
        cv.face.LBPHFaceRecognizer_create()
        cv_im = cv.CreateImageHeader(pil_im.size, cv.IPL_DEPTH_8U, 1)
        cv.SetData(cv_im, pil_im.tostring(), pil_im.size[0])
        return cv_im

    def cv2pil(cv_im):
        return Image.fromstring("L", cv.GetSize(cv_im), cv_im.tostring())

    pil_im = Image.open('Test-Pic/Google-faces.jpg')
    cv_im = pil2cvGrey(pil_im)
    # the haarcascade files tells opencv what to look for.
    faceCascade = cv.Load('C:/Python27/Lib/site-packages/opencv/haarcascade_frontalface_default.xml')
    face = DetectFace(cv_im, faceCascade)
    img = cv2pil(face)
    img.show()

def version2():
    import cv2

    def detect(path):
        img = cv2.imread(path)
        cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        rects = cascade.detectMultiScale(img, 1.3, 4, cv2.CASCADE_SCALE_IMAGE, (20, 20))

        if len(rects) == 0:
            return [], img
        rects[:, 2:] += rects[:, :2]
        return rects, img

    def box(rects, img):
        for x1, y1, x2, y2 in rects:
            cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)
        cv2.imwrite('detected-test1.jpg', img);

    rects, img = detect("Test-Pic/t1.jpeg")
    box(rects, img)

def version3():
    '''
    Sources:
    http://opencv.willowgarage.com/documentation/python/cookbook.html
    http://www.lucaamore.com/?p=638
    '''

    # Python 2.7.2
    # Opencv 2.4.2
    # PIL 1.1.7

    import cv2 as cv  # Opencv
    from PIL import Image  # Image from PIL
    import glob
    import os
    import numpy

    def DetectFace(image, faceCascade, returnImage=False):
        # This function takes a grey scale cv image and finds
        # the patterns defined in the haarcascade function
        # modified from: http://www.lucaamore.com/?p=638

        # variables
        min_size = (20, 20)
        haar_scale = 1.1
        min_neighbors = 3
        haar_flags = 0

        # Equalize the histogram
        #cv.EqualizeHist(image, image)

        # Detect the faces
        faces = cv.HaarDetectObjects(
            image, faceCascade, cv.CreateMemStorage(0),
            haar_scale, min_neighbors, haar_flags, min_size
        )

        # If faces are found
        if faces and returnImage:
            for ((x, y, w, h), n) in faces:
                # Convert bounding box to two CvPoints
                pt1 = (int(x), int(y))
                pt2 = (int(x + w), int(y + h))
                cv.Rectangle(image, pt1, pt2, cv.RGB(255, 0, 0), 5, 8, 0)

        if returnImage:
            return image
        else:
            return faces

    def pil2cvGrey(pil_im):
        # Convert a PIL image to a greyscale cv image
        # from: http://pythonpath.wordpress.com/2012/05/08/pil-to-opencv-image/
        pil_im = pil_im.convert('L')
        cv_im = numpy.array(pil_im)
        #cv.SetData(cv_im, pil_im.tostring(), pil_im.size[0])
        return cv_im

    def cv2pil(cv_im):
        # Convert the cv image to a PIL image
        return Image.fromstring("L", cv.GetSize(cv_im), cv_im.tostring())

    def imgCrop(image, cropBox, boxScale=1):
        # Crop a PIL image with the provided box [x(left), y(upper), w(width), h(height)]

        # Calculate scale factors
        xDelta = max(cropBox[2] * (boxScale - 1), 0)
        yDelta = max(cropBox[3] * (boxScale - 1), 0)

        # Convert cv box to PIL box [left, upper, right, lower]
        PIL_box = [cropBox[0] - xDelta, cropBox[1] - yDelta, cropBox[0] + cropBox[2] + xDelta,
                   cropBox[1] + cropBox[3] + yDelta]

        return image.crop(PIL_box)

    def faceCrop(imagePattern, boxScale=1):
        # Select one of the haarcascade files:
        #   haarcascade_frontalface_alt.xml  <-- Best one?
        #   haarcascade_frontalface_alt2.xml
        #   haarcascade_frontalface_alt_tree.xml
        #   haarcascade_frontalface_default.xml
        #   haarcascade_profileface.xml
        faceCascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

        imgList = glob.glob(imagePattern)
        if len(imgList) <= 0:
            print
            'No Images Found'
            return

        for img in imgList:
            pil_im = Image.open(img)
            cv_im = pil2cvGrey(pil_im)
            faces = DetectFace(cv_im, faceCascade)
            if faces:
                n = 1
                for face in faces:
                    croppedImage = imgCrop(pil_im, face[0], boxScale=boxScale)
                    fname, ext = os.path.splitext(img)
                    croppedImage.save(fname + '_crop' + str(n) + ext)
                    n += 1
            else:
                print
                'No faces found:', img

    def test(imageFilePath):
        pil_im = Image.open(imageFilePath)
        cv_im = pil2cvGrey(pil_im)
        # Select one of the haarcascade files:
        #   haarcascade_frontalface_alt.xml  <-- Best one?
        #   haarcascade_frontalface_alt2.xml
        #   haarcascade_frontalface_alt_tree.xml
        #   haarcascade_frontalface_default.xml
        #   haarcascade_profileface.xml
        faceCascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
        face_im = DetectFace(cv_im, faceCascade, returnImage=True)
        img = cv2pil(face_im)
        img.show()
        img.save('cropped.png')

    # Test the algorithm on an image
    # test('testPics/faces.jpg')

    # Crop all jpegs in a folder. Note: the code uses glob which follows unix shell rules.
    # Use the boxScale to scale the cropping area. 1=opencv box, 2=2x the width and height
    faceCrop('Test-Pic/Google-faces.jpg', boxScale=1)

def version4():
    import cv2

    # Read the input image
    img = cv2.imread("Test-Pic/t1.jpeg")

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print(gray)
    # Load the cascade
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    print(len(faces))
    # Draw rectangle around the faces and crop the faces
    i = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        facee = img[y:y + h, x:x + w]
        #cv2.imshow("face", faces)
        cv2.imwrite('face'+str(i)+'.jpg', facee)
        i +=1

    # Display the output
    cv2.imwrite('detected-test.jpg', img)
    #cv2.imshow('img', img)
    #cv2.waitKey()

def version1_lowerbody():
    import cv2

    # Read the input image
    img = cv2.imread("Test-Pic/t6.jpg")
    print('Original Dimensions : ', img.shape)
    resized = 0
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray)
    # Load the cascade
    body_cascade = cv2.CascadeClassifier("haarcascade_lowerbody.xml")

    # Detect faces
    bodies = body_cascade.detectMultiScale(gray, 1.1, 4)
    print(len(bodies))
    # Draw rectangle around the faces and crop the faces
    i = 0
    for (x, y, w, h) in bodies:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        bodyy = img[y:y + h, x:x + w]
        # cv2.imshow("face", faces)
        img1 = img[y+h+int(img.shape[0]/10):,0:]  #removes body from the image
        # resize image
        scale_percent = (((y+h+int(img.shape[0]/10))/(img.shape[0])) * 100) + 100  # percent of original size width to be increased
        print("Scale percent: ", scale_percent)
        width = int(img.shape[1] * scale_percent / 100)
        #height = int(img.shape[0] * scale_percent / 100)
        height = int(img.shape[0])
        dim = (width, height)
        #resized = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
        #print('Resized Dimensions : ', resized.shape)
        cv2.imwrite('body' + str(i) + '.jpg', bodyy)
        i += 1

    # Display the output
    #cv2.imwrite('resized-cropped-lowerbody.jpg', resized)
    cv2.imwrite('detected-test-lowerbody.jpg', img)
    # cv2.imshow('img', img)
    # cv2.waitKey()


def version2_lowerbody():
    import cv2

    # Read the input image
    img = cv2.imread("Test-Pic/Test-Fullbody/t1.jpg")
    print('Original Width : ', img.shape[1])
    print('Original Height : ', img.shape[0])
    if (img.shape[1] > img.shape[0]) or (img.shape[0] < 400) or (img.shape[1] < 400) or (img.shape[0]-img.shape[1] < 250) or ((img.shape[0]-img.shape[1] > 250)):
        img = cv2.resize(img, (400, 700), interpolation=cv2.INTER_AREA)
        print('Adjusted Width : ', img.shape[1])
        print('Adjusted Height : ', img.shape[0])
    resized = 0
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray)
    # Load the cascade
    body_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")

    # Detect faces
    bodies = body_cascade.detectMultiScale(gray, 1.1, 4)
    print(len(bodies))
    # Draw rectangle around the faces and crop the faces
    i = 0
    for (x, y, w, h) in bodies:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        bodyy = img[y:y + h, x:x + w]
        # cv2.imshow("face", faces)
        img1 = img[y+h+int(img.shape[0]/10):,0:]  #removes body from the image
        # resize image
        scale_percent = (((y+h+int(img.shape[0]/10))/(img.shape[0])) * 100) + 100  # percent of original size width to be increased
        print("Scale percent: ", scale_percent)
        width = int(img.shape[1] * scale_percent / 100)
        #height = int(img.shape[0] * scale_percent / 100)
        height = int(img.shape[0])
        dim = (width, height)
        #resized = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
        #print('Resized Dimensions : ', resized.shape)
        cv2.imwrite('body' + str(i) + '.jpg', bodyy)
        i += 1

    # Display the output
    #cv2.imwrite('resized-cropped-lowerbody.jpg', resized)
    cv2.imwrite('detected-test-lowerbody.jpg', img)
    # cv2.imshow('img', img)
    # cv2.waitKey()

def version5_face(path = "Test-Pic/t3.jpg"):
    import cv2

    # Read the input image
    img = cv2.imread(path)
    print('Original Dimensions : ', img.shape)

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray)
    # Load the cascade
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    print(len(faces))
    # Draw rectangle around the faces and crop the faces
    i = 0
    img1 = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        facee = img[y:y + h, x:x + w]
        # cv2.imshow("face", faces)
        img1 = img[y+h+int(img.shape[0]/20):,0:]  #removes face from the image, the height of neck needs to be adjusted
        cv2.imwrite('face' + str(i) + '.jpg', facee)
        i += 1

    #Save the output
    cv2.imwrite('cropped-face.jpg', img1)
    print('Cropped Dimensions : ', img1.shape)
    cv2.imwrite('detected-face.jpg', img)
    return img1

def version3_lowerbody(path="Test-Pic/Test-Fullbody/t8.jpg"):
    import cv2

    (x_b, y_b, w_b, h_b) = (0, 0, 0, 0)  #coordinates for bodies
    (x_f, y_f, w_f, h_f) = (0, 0, 0, 0)  #coordinates for faces

    # Read the input image
    img = cv2.imread(path)
    img_copy = cv2.imread(path)
    img_main = cv2.imread(path)
    print('Original Width : ', img.shape[1])
    print('Original Height : ', img.shape[0])

    #calibration of image for best results on the model
    if (img.shape[1] > img.shape[0]) or (img.shape[0] < 400) or (img.shape[1] < 400) or (img.shape[0]-img.shape[1] < 250) or ((img.shape[0]-img.shape[1] > 250)):
        img = cv2.resize(img, (400, 700), interpolation=cv2.INTER_AREA)
        img_copy = cv2.resize(img_copy, (400, 700), interpolation=cv2.INTER_AREA)
        img_main = cv2.resize(img_copy, (400, 700), interpolation=cv2.INTER_AREA)
        print('Adjusted Width : ', img.shape[1])
        print('Adjusted Height : ', img.shape[0])
    else:
        print("No adjustemnts needed")

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray)

    # Load the face and body cascade
    body_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

    #Detect bodies and faces
    bodies = body_cascade.detectMultiScale(gray, 1.1, 4)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    print("Bodies Detected:", len(bodies))
    print("Faces Detected:", len(faces))

    w_old = 0 #coordinates for comparison
    h_old = 0
    x_old = img.shape[1] * 99999
    y_old = img.shape[0] * 99999

    #Save the best body
    if (len(bodies) > 1):
        for (x, y, w, h) in bodies:
            if w > w_old:
                w_old = w
            if h > h_old:
                h_old = h
            if x < x_old:
                x_old = x
            if y < y_old:
                y_old = y
        (x_b, y_b, w_b, h_b) = (x_old, y_old, w_old, h_old)
    else:
        for (x, y, w, h) in bodies:
            (x_b, y_b, w_b, h_b) = (x, y, w, h)
            break

    #Drawing rectangle on the body
    cv2.rectangle(img, (x_b, y_b), (x_b + w_b, y_b + h_b), (0, 0, 255), 2)
    body = img[y_b:y_b + h_b, x_b:x_b + w_b]

    w_old = 0 #coordinates for comparison
    h_old = 0
    x_old = img.shape[1] * 99999
    y_old = img.shape[0] * 99999

    #Save the best face
    if (len(faces) > 1):
        for (x, y, w, h) in faces:
            if w > w_old:
                w_old = w
            if h > h_old:
                h_old = h
            if x < x_old:
                x_old = x
            if y < y_old:
                y_old = y
        (x_f, y_f, w_f, h_f) = (x_old, y_old, w_old, h_old)
    else:
        for (x, y, w, h) in faces:
            (x_f, y_f, w_f, h_f) = (x, y, w, h)
            break

    #drawing rectangle on the face
    cv2.rectangle(img, (x_f, y_f), (x_f + w_f, y_f + h_f), (0, 0, 255), 2)
    face = img[y_f:y_f + h_f, x_f:x_f + w_f]

    #draw rectangle on the full_body
    cv2.rectangle(img_copy, (x_b + 30, y_f - 30), (x_b - 30 + w_b, y_f - 10 + h_f + h_b), (0, 0, 255), 2)
    full_body = img_main[y_f - 30:y_f -10 + h_f + h_b, x_b + 30:x_b -30 + w_b]

    lower_body = img_main[int(0.4 * (y_f - 10 + h_f + h_b)) + (y_f - 30):(y_f -10 + h_f + h_b) - int(0*(y_f -10 + h_f + h_b)), x_b + 30:x_b -30 + w_b]

    #Save the output
    cv2.imwrite('body' + '.jpg', body)
    cv2.imwrite('face' + '.jpg', face)
    cv2.imwrite('full_body' + '.jpg', full_body)
    cv2.imwrite('detected-face-body.jpg', img)
    cv2.imwrite('detected-full-body.jpg', img_copy)
    cv2.imwrite('lower_body.jpg', lower_body)
    return lower_body

def final(path = "Test-Pic/Test-Fullbody/t5.jpg", input = " "):
    import cv2

    (x_b, y_b, w_b, h_b) = (0, 0, 0, 0)  #coordinates for bodies
    (x_f, y_f, w_f, h_f) = (0, 0, 0, 0)  #coordinates for faces

    # Read the input image
    img = cv2.imread(path)
    cv2.imwrite('Original.jpg', img)
    img_copy = cv2.imread(path)
    img_main = cv2.imread(path)
    print('Original Width : ', img.shape[1])
    print('Original Height : ', img.shape[0])

    #calibration of image for best results on the model
    if (img.shape[1] > img.shape[0]) or (img.shape[0] < 400) or (img.shape[1] < 400) or (img.shape[0]-img.shape[1] < 250) or ((img.shape[0]-img.shape[1] > 250)):
        img = cv2.resize(img, (400, 700), interpolation=cv2.INTER_AREA)
        img_copy = cv2.resize(img_copy, (400, 700), interpolation=cv2.INTER_AREA)
        img_main = cv2.resize(img_copy, (400, 700), interpolation=cv2.INTER_AREA)
        print('Adjusted Width : ', img.shape[1])
        print('Adjusted Height : ', img.shape[0])
    else:
        print("No adjustemnts needed")

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray)

    # Load the face and body cascade
    body_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

    #Detect bodies and faces
    bodies = body_cascade.detectMultiScale(gray, 1.1, 4)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    print("Bodies Detected:", len(bodies))
    print("Faces Detected:", len(faces))

    w_old = 0 #coordinates for comparison
    h_old = 0
    x_old = img.shape[1] * 99999
    y_old = img.shape[0] * 99999

    #Save the best body
    if (len(bodies) > 1):
        for (x, y, w, h) in bodies:
            if w > w_old:
                w_old = w
            if h > h_old:
                h_old = h
            if x < x_old:
                x_old = x
            if y < y_old:
                y_old = y
        (x_b, y_b, w_b, h_b) = (x_old, y_old, w_old, h_old)
    else:
        for (x, y, w, h) in bodies:
            (x_b, y_b, w_b, h_b) = (x, y, w, h)
            break

    #Drawing rectangle on the body
    cv2.rectangle(img, (x_b, y_b), (x_b + w_b, y_b + h_b), (0, 0, 255), 2)
    body = img[y_b:y_b + h_b, x_b:x_b + w_b]

    w_old = 0 #coordinates for comparison
    h_old = 0
    x_old = img.shape[1] * 99999
    y_old = img.shape[0] * 99999

    #Save the best face
    if (len(faces) > 1):
        for (x, y, w, h) in faces:
            if w > w_old:
                w_old = w
            if h > h_old:
                h_old = h
            if x < x_old:
                x_old = x
            if y < y_old:
                y_old = y
        (x_f, y_f, w_f, h_f) = (x_old, y_old, w_old, h_old)
    else:
        for (x, y, w, h) in faces:
            (x_f, y_f, w_f, h_f) = (x, y, w, h)
            break

    #drawing rectangle on the face
    cv2.rectangle(img, (x_f, y_f), (x_f + w_f, y_f + h_f), (0, 0, 255), 2)
    face = img[y_f:y_f + h_f, x_f:x_f + w_f]

    #drawing rectangle on the full_body
    cv2.rectangle(img_copy, (x_b + 30, y_f - 30), (x_b - 30 + w_b, y_f - 10 + h_f + h_b), (0, 0, 255), 2)
    full_body = img_main[y_f - 30:y_f -10 + h_f + h_b, x_b + 30:x_b -30 + w_b]

    face_body = img_main[y_f + h_f + int(img.shape[0] / 20):, x_b + 30:x_b -30 + w_b]  # removes face from the image, the height of neck needs to be adjusted
    lower_body = img_main[int(0.4 * (y_f - 10 + h_f + h_b)) + (y_f - 30):(y_f -10 + h_f + h_b), x_b + 30:x_b -30 + w_b]
    upper_body = img_main[y_f + h_f + int(img.shape[0] / 20):(y_f -10 + h_f + h_b) - int(0.4 * (y_f - 10 + h_f + h_b)), x_b + 30:x_b -30 + w_b]

    #Save the output
    '''
    cv2.imwrite('body' + '.jpg', body)
    cv2.imwrite('face' + '.jpg', face)
    cv2.imwrite('detected-face-body.jpg', img)
    cv2.imwrite('detected-full-body.jpg', img_copy)
    cv2.imwrite('full_body' + '.jpg', full_body)
    '''

    if input == "face":
        cv2.imwrite('face_body.jpg', face_body)
        return face_body
    elif input == "lower":
        cv2.imwrite('lower_body.jpg', lower_body)
        return lower_body
    elif input == "upper":
        cv2.imwrite('upper_body.jpg', upper_body)
        return upper_body
    else:
        cv2.imwrite('face_body.jpg', face_body)
        cv2.imwrite('lower_body.jpg', lower_body)
        cv2.imwrite('upper_body.jpg', upper_body)

def bg_white(path = "Test-Pic/t1.jpg"):
    import numpy as np
    import cv2

    # Load the Image
    imgo = cv2.imread(path)
    height, width = imgo.shape[:2]

    # Create a mask holder
    mask = np.zeros(imgo.shape[:2], np.uint8)

    # Grab Cut the object
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Hard Coding the Rect… The object must lie within this rect.
    rect = (10, 10, width, height)
    cv2.grabCut(imgo, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img1 = imgo * mask[:, :, np.newaxis]

    # Get the background
    background = imgo - img1

    # Change all pixels in the background that are not black to white
    background[np.where((background > [0, 0, 0]).all(axis=2))] = [255, 255, 255]

    # Add the background and the image
    final = background + img1

    # To be done – Smoothening the edges….

    cv2.imwrite('bg_final.jpg', final)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #version1()
    #version2()
    #version3()
    #version4()
    ##version5_face() #Final working version for face

    #version1_lowerbody()
    #version2_lowerbody()
    ##version3_lowerbody() #Final working version for lower body

    ##final() #Final working version for all three parts

    bg_white() #function to turn the image background to white

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
