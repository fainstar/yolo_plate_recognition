import requests
import base64

# 讀取並編碼圖片
with open("Data/data03.jpeg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# 發送請求
response = requests.post('http://localhost:5000/infer', 
                        json={"image": encoded_string})

# 如果請求成功
if response.status_code == 200:
    # 解碼並儲存回傳的圖片
    img_data = base64.b64decode(response.json()['image'])
    with open("OutPut/result.jpg", "wb") as f:
        f.write(img_data)