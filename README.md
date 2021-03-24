# AI-DL-OpenCV2-SuperResolution<br/>
## A.I Deep Learning 기반, 저해상도 화질의 이미지를 고해상도 화질의 이미지로 만들기<br/>
### 논문을 기반으로, 구현한 딥러닝 알고리즘 입니다.<br/>
<br/>

## Data From<br/>
|Data From|URL|Explanation|
|:----:|:----:|:----:|
|From Kaggle|https://www.kaggle.com/jessicali9530/celeba-dataset|용량이 1GB가 넘어가므로, 이곳에 올리지 못해서 URL을 첨부한 점 양해 부탁드립니다.|
|Paper|https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf|Subpixel, DataGenerator는 학술 논문으로부터 구현한 알고리즘 입니다.|
<br/>

## Requirements
- Python 3+
- Keras
- OpenCV
- numpy
- scikit-image
- matplotlib (For visualization)
<br/>

## About Model(Algorithm Explanation)<br/>
일단, 위에 소개드린 주소를 통해 데이터셋을 다운받아주셔야 합니다.<br/>
상위에 소개된 데이터셋은, 유명인들의 얼굴 데이터가 방대한 범위로 세팅되어있다 보시면 됩니다.<br/>
첨부한 소스코드를 실행해주시고, 아시다시피 딥러닝이라는 것은 시간싸움입니다. 직접 저의 노트북으로 돌려보니 10시간하고 31분이 걸렸는데요.<br/>
이는 워낙 처리해야하는 사진의 양이 방대하다보니 그렇습니다, 머신러닝 학습을 위한 데이터 준비 과정은 어떻게보면 길고 매우 지루한 작업이지요.<br/>
이미지를 읽은 후, crop을 해서 pyramid_reduce 함수를 사용합니다.<br/>
그렇게 되면, 이미지를 흐릿하게 해주는데 그 후 copyMakeBorder 함수를 이용하여 이미지를 액자처럼 테두리를 씌워주게 됩니다.<br/>
그 다음, train과 test 그리고 val로 나누어 저장해주는 작업을 거칩니다.<br/>
<br/>
이미지 작업에서도 정규화가 필요한 경우가 있습니다.<br/>
특정 영역에 몰려있는 경우, 화질을 개선해주기도 하고 이미지간의 연산 시 서로 조건이 다른 경우, 같은 조건으로 만들어주기도 합니다.<br/>
OpenCV는 cv2.normalize()함수를 통해 정규화를 제공하고 있습니다.<br/>
다시말해, 학습을 위해서 0과 1사이의 값으로 정규화를 해줍니다.<br/>
유명인 데이터셋을 전처리하면 거의 80GB의 저장소가 필요합니다.<br/>
실행을 해주고나면, 데이터가 뙇! 하고 만들어져 있을텐데 이때 이 프로젝트 실행을 위해 'SubPixel'과 'DataGenerator'를 실행해주시면 됩니다.<br/>
DataGenerator 함수는 list_IDs, labels, batch_size, dim, n_channels, n_classes, shuffle 을 파라메터로 받아서 학습용 이미지 데이터를 생성합니다.<br/>
Subpixel 레이어는 Conv2D의 하위 클래스입니다.<br/>
인수 r은 Conv2D의 일반 출력에 적용되는 업샘플링 계수를 나타냅니다.<br/>
이 레이어의 출력은 표시된 필터 필드와 동일한 수의 채널을 가지므로 grayscale, color 또는 숨겨진 레이어로 동작합니다.<br/>
일반 Conv2D 테스트 출력에 적용되는 업스케일링 팩터가 포함, 이미지 데이터 세트에서 초해상도를 수행합니다.<br/>
glob 모듈의 glob 함수는 사용자가 제시한 조건에 맞는 파일명을 리스트 형식으로 반환합니다.<br/>

## Author<br/>
Ian / aoa8538@gmail.com
