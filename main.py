import cv2
import numpy as np
import matplotlib.pyplot as plt
from local_utils import detect_lp
from os.path import splitext, basename
from keras.models import model_from_json
import glob
import matplotlib.gridspec as gridspec
import pytesseract
from sklearn.preprocessing import LabelEncoder
#truyền file tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)


wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)


def preprocess_image(img, resize=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

# Create a list of image paths
# image_paths = glob.glob("Plate_examples/*.jpg")
# print("Found %i images..."%(len(image_paths)))

# Visualize data in subplot
# fig = plt.figure(figsize=(12,8))
# cols = 5
# rows = 4
# fig_list = []
# for i in range(cols*rows):
#     fig_list.append(fig.add_subplot(rows,cols,i+1))
#     title = splitext(basename(image_paths[i]))[0]
#     fig_list[-1].set_title(title)
#     img = preprocess_image(image_paths[i],True)
#     plt.axis(False)
#     plt.imshow(img)
#
# plt.tight_layout(True)
# plt.show()

def get_plate(img, Dmax=608, Dmin=256):
    vehicle = preprocess_image(img)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    try:
        _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.4)
        return LpImg, cor
    except:
        return None, None

def sort_contours(cnts, reverse=False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

def predict_from_model(image, model, labels):
    image = cv2.resize(image, (80, 80))
    image = np.stack((image,) * 3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis, :]))])
    return prediction
cap = cv2.VideoCapture('license_video.mp4')
count = 0
# if (cap.isOpened()== False):
#     print("Error opening video stream or file")
while True:
# Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('Frame', frame)
    count += 30 # i.e. at 30 fps, this advances one second
    cap.set(cv2.CAP_PROP_POS_FRAMES, count)
    # Press Q on keyboard to  exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Obtain plate image and its coordinates from an image
    test_image = frame
    LpImg, cor = get_plate(test_image)
    if LpImg is None or cor is None:
        continue
    # print("Detect %i plate(s) in" % len(LpImg), splitext(basename(test_image))[0])
    print("Coordinate of plate(s) in image: \n", cor)
    # Visualize our result
    # plt.figure(figsize=(10,5))
    # plt.subplot(1,2,1)
    # plt.axis(False)
    # plt.imshow(preprocess_image(test_image))
    # plt.subplot(1,2,2)
    # plt.axis(False)
    # plt.imshow(LpImg[0])
    # # print(test_image)
    # # print("hiii")
    # print(LpImg[0].shape)
    # cv2.imwrite("file.jpg", LpImg[0]*255)
    #tesseract
    # plt.subplot(1, 1, 1)
    # # plt.imshow(LpImg[0])
    # plt.axis(False)
    # plt.savefig("file1.jpg")
    # # plt.show()
    # license_plate = cv2.imread("file1.jpg")
    # print("\n")
    # text = pytesseract.image_to_string(license_plate)
    # print(text)
    # def draw_box(image_path, cor, thickness=3):
    #     pts = []
    #     x_coordinates = cor[0][0]
    #     y_coordinates = cor[0][1]
    #     # store the top-left, top-right, bottom-left, bottom-right
    #     # of the plate license respectively
    #     for i in range(4):
    #         pts.append([int(x_coordinates[i]), int(y_coordinates[i])])
    #
    #     pts = np.array(pts, np.int32)
    #     pts = pts.reshape((-1, 1, 2))
    #     vehicle_image = preprocess_image(image_path)
    #
    #     cv2.polylines(vehicle_image, [pts], True, (0, 255, 0), thickness)
    #     return vehicle_image
    #
    #
    # plt.figure(figsize=(8,8))
    # plt.axis(False)
    # plt.imshow(draw_box(test_image,cor))
    # plt.show()
    if (len(LpImg)):  # check if there is at least one license image
        # Scales, calculates absolute values, and converts the result to 8-bit.
        plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
        # convert to grayscale and blur the image
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        # Applied inversed thresh_binary
        binary = cv2.threshold(blur, 180, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        ## Applied dilation
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
        # plt.imshow(thre_mor)
        # plt.show()

    cont, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # creat a copy version "test_roi" of plat_image to draw bounding box
    test_roi = plate_image.copy()
    # Initialize a list which will be used to append charater image
    crop_characters = []
    # define standard width and height of character
    digit_w, digit_h = 30, 60
    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h / w
        if 1 <= ratio <= 3.5:  # Only select contour with defined ratio
            if h / plate_image.shape[0] >= 0.5:  # Select contour which has the height larger than 50% of the plate
                # Draw bounding box arroung digit number
                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Sperate number and gibe prediction
                curr_num = thre_mor[y:y + h, x:x + w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_characters.append(curr_num)
    print("Detect {} letters...".format(len(crop_characters)))
    # fig = plt.figure(figsize=(14, 4))
    # grid = gridspec.GridSpec(ncols=len(crop_characters), nrows=1, figure=fig)
    # for i in range(len(crop_characters)):
    #     fig.add_subplot(grid[i])
    #     plt.axis(False)
    #     plt.imshow(crop_characters[i], cmap="gray")
    # plt.show()
    # Load model architecture, weight and labels
    json_file = open('MobileNets_character_recognition.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("License_character_recognition_weight.h5")
    print("[INFO] Model loaded successfully...")
    labels = LabelEncoder()
    labels.classes_ = np.load('license_character_classes.npy')
    print("[INFO] Labels loaded successfully...")
    # pre-processing input images and pedict with model

    # fig = plt.figure(figsize=(15, 3))
    # cols = len(crop_characters)
    # grid = gridspec.GridSpec(ncols=cols, nrows=1, figure=fig)
    final_string = ''
    for i, character in enumerate(crop_characters):
        # fig.add_subplot(grid[i])
        title = np.array2string(predict_from_model(character, model, labels))
        # plt.title('{}'.format(title.strip("'[]"), fontsize=20))
        final_string += title.strip("'[]")
        # plt.axis(False)
        # plt.imshow(character, cmap='gray')
    print(final_string)
    # plt.savefig('final_result.png', dpi=300)

