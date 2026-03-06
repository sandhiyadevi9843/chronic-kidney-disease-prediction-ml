import requests
import json

url = 'http://127.0.0.1:5000/predict'
data = {
    'age': '48',
    'bp': '80',
    'sg': '1.020',
    'al': '1',
    'su': '0',
    'rbc': 'normal',
    'pc': 'normal',
    'pcc': 'notpresent',
    'ba': 'notpresent',
    'bgr': '121',
    'bu': '36',
    'sc': '1.2',
    'sod': '136',
    'pot': '4.7',
    'hemo': '15.4',
    'pcv': '44',
    'wc': '7800',
    'rc': '5.2',
    'htn': 'yes',
    'dm': 'yes',
    'cad': 'no',
    'appet': 'good',
    'pe': 'no',
    'ane': 'no'
}

response = requests.post(url, data=data)
print(f"Status Code: {response.status_code}")
print("Response Text:")
print(response.text)
