
import base64
import requests
import cv2

class GPT4VFoodIdentification:
    def __init__(self, api_key, prompt_dir):

        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
            }
        self.prompt_dir = prompt_dir
        
        with open("%s/prompt.txt"%self.prompt_dir, 'r') as f:
            self.prompt_text = f.read()
        
        self.detection_prompt_img1 = cv2.imread("%s/11.jpg"%self.prompt_dir)
        self.detection_prompt_img2 = cv2.imread("%s/12.jpg"%self.prompt_dir)
        self.detection_prompt_img3 = cv2.imread("%s/13.jpg"%self.prompt_dir)

        self.detection_prompt_img1 = self.encode_image(self.detection_prompt_img1)
        self.detection_prompt_img2 = self.encode_image(self.detection_prompt_img2)
        self.detection_prompt_img3 = self.encode_image(self.detection_prompt_img3)

        self.mode = 'ours' # ['ours', 'preference', 'efficiency']

    def encode_image(self, openCV_image):
        retval, buffer = cv2.imencode('.jpg', openCV_image)
        return base64.b64encode(buffer).decode('utf-8')

    def prompt_zero_shot(self, image, prompt):
        # Getting the base64 string
        base64_image = self.encode_image(image)

        payload = {
        "model": "gpt-4o",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": prompt
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
        response_text =  response.json()['choices'][0]["message"]["content"]
        return response_text
        
    def prompt(self, image):
        
        # Getting the base64 string
        base64_image = self.encode_image(image)

        payload = {
        "model": "gpt-4o",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": self.prompt_text
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.detection_prompt_img1}"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.detection_prompt_img2}"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.detection_prompt_img3}"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
        response_text =  response.json()['choices'][0]["message"]["content"]

        return response_text