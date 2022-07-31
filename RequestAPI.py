import time
from datetime import datetime
import uuid

# importing the module
import json
import uuid
# response = requests.get("http://smartparking.local:5555/api/parkingspace")
#
# print(response.json()) # This method is convenient when the API returns JSON
# print(response)
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# ID = uuid.uuid4()
# FrontImageLink = "https://drive.google.com/uc?export=view&id=1ojh3ABPB6LEh1tCQTyYt_kfjkn4otWa2"
# BackImageLink ="https://drive.google.com/uc?export=view&id=1ojh3ABPB6LEh1tCQTyYt_kfjkn4otWa2"
# LicensePlateNumber = "60A-999.99"
# ParkingAreaId = "633b4557-7cad-4c18-94cd-e939dd0285b6"
# TimeIn = int(time.mktime(datetime.utcnow().timetuple()))
# TimeOut = 0
# Status = 1
# data = '{{ "Id":"{ID}", "FrontImageLink":"{FrontImageLink}", "BackImageLink":"{BackImageLink}", "LicensePlateNumber":"{LicensePlateNumber}", "ParkingAreaId":"{ParkingAreaId}", "TimeIn":{TimeIn}, "TimeOut":{TimeOut}, "Status":{Status}}}'
# data = data.format(ID=ID,FrontImageLink=FrontImageLink,BackImageLink=BackImageLink,LicensePlateNumber=LicensePlateNumber,ParkingAreaId=ParkingAreaId,TimeIn=TimeIn,TimeOut=TimeOut,Status=Status)
# print(data)
# # Create a new resource
# headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
# response = requests.post('https://localhost:44307/api/Car/in', data=data, verify=False, headers=headers)
# print(response.status_code)

# https://localhost:44307/api/User/payment?licensePlate=29A08129

licensePlate = '29A08129'
url = 'http://smartparking.local:5555/api/User/payment?licensePlate={LicensePlate}'.format(LicensePlate=licensePlate)
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
response = requests.get(url, verify=False, headers=headers)
print(response.status_code)
print(response.json())
response_dict = response.json()

print(response_dict['code'])
print(response_dict['message'])
