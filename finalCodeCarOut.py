import cv2
import numpy as np
from local_utils import detect_lp
from os.path import splitext
from keras.models import model_from_json
import pytesseract
from sklearn.preprocessing import LabelEncoder
import time
from datetime import datetime
import uuid
import requests
from Google import Create_Service
from googleapiclient.http import MediaFileUpload
# Open port
import serial
from requests.packages.urllib3.exceptions import InsecureRequestWarning


CAR_COME_IN_MESSAGE = "1"
BARRIER_IN_OPEN_MESSAGE = "2"
CAR_COME_OUT_MESSAGE = "3"
BARRIER_OUT_OPEN_MESSAGE = "4"


#truyền file tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# config google drive API
CLIENT_SECRET_FILE = 'client_secret_GoogleCloudDemo.json'
API_NAME = 'drive'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/drive']

service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)
#config serial port
arduino = serial.Serial(port='COM4', baudrate=115200, timeout=1)

def handshake_arduino(
    arduino, sleep_time=1, print_handshake_message=False, handshake_code=0
):
    """Make sure connection is established by sending
    and receiving bytes."""
    # Close and reopen
    arduino.close()
    arduino.open()

    # Chill out while everything gets set
    time.sleep(sleep_time)

    # Set a long timeout to complete handshake
    timeout = arduino.timeout
    arduino.timeout = 1

    # Read and discard everything that may be in the input buffer
    _ = arduino.read_all()

    # Send request to Arduino
    arduino.write(bytes([handshake_code]))

    # Read in what Arduino sent
    handshake_message = arduino.read_until()

    # Send and receive request again
    arduino.write(bytes([handshake_code]))
    handshake_message = arduino.read_until()

    # Print the handshake message, if desired
    if print_handshake_message:
        print("Handshake message: " + handshake_message.decode())

    # Reset the timeout
    arduino.timeout = timeout
    return handshake_message.decode()
def write_read(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.05)
    data = arduino.readline()
    return data

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

def CallAPIUploadData(FrontImageLink, BackImageLink, LicensePlateNumber, ParkingAreaId):
    # FrontImageLink = "https://drive.google.com/uc?export=view&id=1ojh3ABPB6LEh1tCQTyYt_kfjkn4otWa2"
    # BackImageLink = "https://drive.google.com/uc?export=view&id=1ojh3ABPB6LEh1tCQTyYt_kfjkn4otWa2"
    # LicensePlateNumber = "60A-999.99"
    # ParkingAreaId = "633b4557-7cad-4c18-94cd-e939dd0285b6"
    # TimeIn = int(time.mktime(datetime.utcnow().timetuple()))
    # TimeOut = 0
    # Status = 1
    data = '{{"FrontImageLink":"{FrontImageLink}", "BackImageLink":"{BackImageLink}", "LicensePlateNumber":"{LicensePlateNumber}", "ParkingAreaId":"{ParkingAreaId}", "TimeIn":{TimeIn}, "TimeOut":{TimeOut}}}'
    data = data.format(FrontImageLink=FrontImageLink, BackImageLink=BackImageLink,
                       LicensePlateNumber=LicensePlateNumber, ParkingAreaId=ParkingAreaId)
    print(data)
    # Create a new resource
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    response = requests.post('https://localhost:44307/api/Car/out', data=data, verify=False, headers=headers)
    print(response.status_code)

def CallAPIPayment(licensePlate):
    licensePlate = '29A08129'
    url = 'http://smartparking.local:5555/api/User/payment?licensePlate={LicensePlate}'.format(
        LicensePlate=licensePlate)
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    response = requests.get(url, verify=False, headers=headers)
    print(response.status_code)
    print(response.json())
    response_dict = response.json()
    print(response_dict['message'])
    return response_dict['code']

def UploadImageToGoogleDrive(file_names):
    # upload file
    folder_id = '1b0xl3bovUpvFZOsfuDVxKGkEjeEcS2Pb'

    mime_types = ['image/jpeg']

    for file_name, mime_type in zip(file_names, mime_types):
        file_metadata = {
            'name': file_name,
            'parents': [folder_id]
        }
        media = MediaFileUpload('./MyFile/{0}'.format(file_name), mimetype=mime_type, resumable=True)
        result = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        print(result['id'])

    # share link url

    file_id = result['id']  # = result.id

    request_body = {
        'role': 'reader',
        'type': 'anyone'
    }

    response_permission = service.permissions().create(
        fileId=file_id,
        body=request_body,
    ).execute()

    print(response_permission)

    response_share_link = service.files().get(
        fileId=file_id,
        fields='webViewLink'
    ).execute()

    print(response_share_link)
    show_link = "https://drive.google.com/uc?export=view&id=" + file_id

    print(show_link)
    return show_link




lock = False
while True:
    if lock == True:
        continue
    # Call the handshake function
    if handshake_arduino(arduino, print_handshake_message=True) == CAR_COME_OUT_MESSAGE:
        lock = True
        read_image_count = 0
        cap = cv2.VideoCapture('vietnamlicenseplate.mp4')
        count = 0
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
            print("Coordinate of plate(s) in image: \n", cor)
            # Visualize our result

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

            final_string = ''
            for i, character in enumerate(crop_characters):
                title = np.array2string(predict_from_model(character, model, labels))
                final_string += title.strip("'[]")
            print(final_string)
            read_image_count +=1
            if read_image_count> 10:
                print("Call Police")
                break;
            if len(final_string)>=7:
                image_name = "%s.jpg" % uuid.uuid4()
                image_location = "MyFile/%s" % image_name
                cv2.imwrite(image_location, frame)
                file_names = [image_name]
                front_image_link = UploadImageToGoogleDrive(file_names)
                start_time = time.time()
                while True:
                    # thanh toán cho đến khi thành công sẽ có timeout ở đây

                    respone = CallAPIPayment(final_string)
                    if respone == 0:
                        break
                    time.sleep(5)
                    current_time = datetime.now().time()
                    if (start_time- time.time() > 300):
                        print("call police")
                CallAPIUploadData(front_image_link, front_image_link, final_string, "633b4557-7cad-4c18-94cd-e939dd0285b6")
                # open barrier
                value = write_read(BARRIER_OUT_OPEN_MESSAGE)
                print(value)

                # open lock
                lock = False
                break;

