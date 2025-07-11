import requests
import os
login_url = "https://go.genadata.com/auth/login"
login_payload = {
    "email": "domixidantoctay@gmail.com",       
    "password": "Domixi123@"     
}
login_headers = {
    "accept": "application/json, text/plain, */*",
    "content-type": "application/json",
    "origin": "https://genadata.com",
    "referer": "https://genadata.com/",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
}

login_response = requests.post(login_url, json=login_payload, headers=login_headers)

if login_response.status_code == 200:
    login_data = login_response.json()
    token = login_data.get("token") or login_data.get("access_token")
    
    if token:
        print("✅ Đăng nhập thành công. Token:")
        print(token)
    else:
        print("❌ Không tìm thấy token trong response:")
        print(login_data)
else:
    print("❌ Đăng nhập thất bại:", login_response.status_code)
    print(login_response.text)

access_token='eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJ1c2VyX2NyZWRlbnRpYWxzIiwiZW1haWwiOiJkb21peGlkYW50b2N0YXlAZ21haWwuY29tIiwiZXhwIjoxNzUxMjY2NTUzLCJpYXQiOjE3NDg2NzQ1NTMsImlzcyI6IkdlbmlGYXN0LVNlYXJjaF9HbyIsIm5iZiI6MTc0ODY3NDU1MywidXNlcl9pZCI6IjI1NSJ9.dRtM-0v1k6jWVpI0FhT-tjMU2qhKSy4MoKJI7Jj2T-qZjRhhHB2NXendeJyU5QKD01MPl8hwSPok3iKUxgpPnYOsdYLru_TmuADGohB47bEF417A8c_NbAusg1Pq-Ne3UGN9bCCNgLbjl3DY1CTNOzVARH6FxJnRKYzI5BJOAxshRdw6R6UenbUmG6clYQuyE3LrFtRf-Reom7pMSCkDb8Jp5SrnxgEL8aVTjE9SIdYxB4u3VSyeoLp_f6El0C7lE8djKm1S735Lt-Wkb7BUnjP4vfhMAwov1agBBbq0rOSW7WtejpYWyB6tznbHkoJquXIIoP4SW_isKg0BKTbBkkWzbDV8_KnC1Eq_rmwW1OpvHXYjxksb5VZUu53MGlh3GQvxcJFqKVEKv07WCVWMCdm3uFRi28YuoOXQkq_ARtIJJ-wuJug-oZh8ATeiRKbImgaG94xOeiJuk8YSv_p8Cgm74DlPS-8xqOhejk7RAeL5CKIVWyUuYiVj4suZNoD7PtsZ2KmV5g7pO9eai53NPfgDAV65w2Bvk43UZdZenSBQ-RV3GW0hmcYYQcmRTgE1_DD3qqI8tX387w4yQ6_aasox13UPrPlLW2wHg6yOCZk6PvfFoTsAT98J-9EEUmDwCekhb6G_t1wR1OohHFLo0m-OOqGdNRZW3pyI7WVh3h0'
share_url="https://java.genadata.com/share"
headers = {
    "Authorization": f"Bearer {access_token}",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://genadata.com/",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
}
response=requests.get(share_url,headers=headers)
if response.status_code == 200:
    data = response.json()
    print("✅ Dữ liệu được chia sẻ:")
else:
    print("❌ Lỗi khi gọi API:", response.status_code)
    print(response.text)
print(os.getcwd()) 
base_dir=os.path.dirname(os.path.dirname(__file__))
destination_dir=os.path.join(base_dir,"data")
os.makedirs(destination_dir,exist_ok=True)
if 'data' in locals():
    for doc in data:
        title=doc.get("title","untitled.txt").strip()
        content=doc.get("content","")
        content_clean=content.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n").strip()
        filename="".join(c if c.isalnum() or c in " ._-" else "_" for c in title)
        filepath=os.path.join(destination_dir,filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content_clean)
            print(f"📁 Đã lưu file: {filepath}")
else:
    print("❌ Không có dữ liệu để xử lý do lỗi khi gọi API.")
