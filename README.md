# Be:대면조
Change Your Hairstyle (with StyleCLIP)

# Introduction
- 대부분의 사람들은 나이가 들어감에 따라 헤어스타일에 변화주고자 한다. 다만, 대부분 헤어스타일에 변화를 주는데 두려움을 가지고 있다. 헤어스타일을 바꿨다 하더라도 실패할 경우 머리를 다시 길러야 하거나, 재염색을 해야 하는 등 원래 상태로 돌아가기는 힘들기 때문이다. 따라서, 우리는 이런 점들을 고려하여 딥러닝을 활용한 헤어스타일 변환 서비스를 제공하고자 한다.
- 현재 '헤어핏' 이라는 헤어스타일 가상체험 앱이 존재한다. 그러나 사진에 스티커를 붙인 것처럼 부자연스러운 부분이 존재한다. 이를 극복하고자, GAN이라는 딥러닝 생성 모델을 사용하여 보다 더 자연스러운 결과를 보이는 것이 본 프로젝트의 강점이다.

# Member
![Member](/img/member.JPG)  

# Arichitecture

![Architecture](/img/architecture.JPG)  

# Stack
![Stack](/img/stack.JPG)  

# Instance 환경
- Ubuntu: 18.04
- GPU: Geforce RTX 2080
- CUDA: 11.1

# Directory architecture

```
├── kb
│   ├── FaceSwqp
│   ├── StyleCLIP
│   └── encoder4editing
├── myapp
│    ├── admin.py
│    └── apps.py
│    ├── models.py
│    └── views.py
├── myproject
│    ├── settings.py
│    └── urls.py
├── static
│    ├── assets
│    ├── forms
│    └── img
├── template
│    └── index.html
├── media
│    └── result
├── latentVector
│    └── *.pt
├── e4e.py
├── faceswap.py
├── styleclip.py
├── manage.py
├── db.sqlite3
├── requirements.txt
├── envRequirements.txt
├── LICENSE
└── README.md    	
```