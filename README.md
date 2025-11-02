AI 쓰레기 분류 (Garbage Classification)
이 프로젝트는 딥러닝 모델을 처음부터 끝까지 구축하여, 이미지를 보고 6가지 다른 종류의 쓰레기를 분류하는 과제입니다. 모델은 전이 학습(Transfer Learning) 기술을 사용하여 훈련되었으며, 상호작용이 가능한 웹 애플리케이션 형태로 Streamlit을 통해 배포되었습니다.
최종 모델은 테스트 데이터셋에서 약 83%의 정확도를 달성했습니다.

데모 인터페이스
(팁: Streamlit Community Cloud에 앱을 배포한 후, GIPHY Capture나 ScreenToGif 같은 도구를 사용하여 앱 작동 방식을 보여주는 짧은 GIF를 녹화하여 여기에 삽입하세요. 가장 인상적인 방법입니다!)

사용된 기술
언어 : Python 3.10
딥러닝 라이브러리 : TensorFlow, Keras
웹 인터페이스 : Streamlit
이미지 처리 : OpenCV, Pillow
훈련 환경 : Google Colab

버전 관리 : Git & GitHub (with Git LFS)

프로젝트 구조 
Garbage-Classification-AI/
 --.devcontainer/         # GitHub Codespaces 자동 설정 파일
 --app/                   # Streamlit 앱 코드
 --dataset/               # 원본 이미지 데이터 (GitHub 비포함)
 --notebooks/             # 훈련용 Colab 노트북
 --saved_models/          # 훈련된 .h5 모델 및 class_indices.json
 --.gitattributes         # Git LFS 설정
 --README.md              # 프로젝트 소개 (현재 파일)
 --requirements.txt       # Python 라이브러리 목록
설치 및 앱 실행

1. 준비 사항
컴퓨터에 Git과 Git LFS가 설치되어 있는지 확인하세요.

2. 리포지토리 클론 (Clone Repository)
# 리포지토리를 로컬 컴퓨터로 클론합니다
git clone [https://github.com/YourUsername/Garbage-Classification-AI.git](https://github.com/YourUsername/Garbage-Classification-AI.git)
cd Garbage-Classification-AI

# Git LFS를 사용하여 대용량 모델 파일(.h5)을 가져옵니다 (매우 중요)
git lfs pull

3. 가상 환경 설정
# 가상 환경을 생성합니다
python3 -m venv venv

# 가상 환경을 활성화합니다
source venv/bin/activate

4. 라이브러리 설치 (Cài đặt các Thư viện)
# requirements.txt 파일에서 필요한 모든 패키지를 설치합니다
pip install -r requirements.txt

5. 애플리케이션 실행 (Chạy Ứng dụng)
streamlit run app/app.py
이후 브라우저에서 http://localhost:8501로 접속합니다.

방법 2: GitHub Codespaces에서 실행 (권장/가장 쉬운 방법)
이 방법을 사용하면 로컬 컴퓨터에 아무것도 설치할 필요 없이, 브라우저에서 바로 프로젝트를 실행하고 테스트할 수 있습니다.
1. Codespace 시작
이 GitHub 리포지토리 페이지의 상단으로 이동합니다.
녹색 <> Code 버튼을 클릭합니다.
Codespaces 탭을 선택합니다.
Create codespace on main 버튼을 클릭합니다.

2. 자동 설정 대기
약 1~2분 정도 기다리면, 브라우저에 VS Code와 유사한 환경이 나타납니다.
이 프로젝트에는 .devcontainer 설정 파일이 포함되어 있어, Codespaces가 시작될 때 자동으로 필요한 시스템 라이브러리(libgl1-mesa-glx)와 Python 라이브러리(requirements.txt)를 설치합니다.
하단의 TERMINAL 창에서 설치 과정이 완료될 때까지 기다립니다.

3. 애플리케이션 실행
모든 자동 설치가 완료되면, TERMINAL에 다음 명령어를 입력합니다:
streamlit run app/app.py


