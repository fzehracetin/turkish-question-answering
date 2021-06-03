import requests
import json

data = {
    "question": "Doğal olarak bulunan altı soy gaz nelerdir?",
    "context": "Soy gaz veya asal gaz, standart şartlar altında her biri, diğer elementlere kıyasla daha düşük "
               "kimyasal reaktifliğe sahip, kokusuz, renksiz, tek atomlu gaz olan kimyasal element grubudur. "
               "Helyum (He), neon (Ne), argon (Ar), kripton (Kr), ksenon (Xe) ve radon (Rn) doğal olarak bulunan "
               "altı soy gazdır ve tamamı ametaldir. Her biri periyodik tablonun sırasıyla ilk altı periyodunda, "
               "18. grubunda (8A) yer alır. Grupta yer alan oganesson (Og) için ise önceleri soy gaz olabileceği "
               "ihtimali üzerinde durulsa da günümüzde metalik görünümlü reaktif bir katı olduğu öngörülmektedir.",
    "model": "bert",
}

data = json.dumps(data)
headers = {"content-type": "application/json"}
json_response = requests.post("http://localhost:8090/predict", data=data, headers=headers)

response = json.loads(json_response.text)

print(response)

