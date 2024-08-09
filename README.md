# DMAP 생성

이 함수는 MVS 파일로부터 깊이 맵을 생성하고 DMAP 형식으로 저장하는 스크립트를 제공합니다.
깊이맵 생성을 위해 DPT_Large 모델을 사용합니다. 해당 모델은 대체 가능합니다. 

## 사용법 


코드 정상작동을 위해 다음과 같은 폴더 구조와 폴더명을 유지해야합니다. 

```
<path>
  - images
      - 00000.jpg
      - 00001.jpg  
      - ...
  - scene.mvs
```

MVS 파일로부터 DMAP 파일을 생성하려면 다음 명령어를 사용하세요:

```
python CreateDmap.py -mvs_path {path}/scene.mvs
```

해당 경로에 다음과 같은 파일이 생성된다면 성공입니다.
```
<path>
  - images
      - 00000.jpg
      - 00001.jpg  
      - ...
  - scene.mvs
  - depthmap0001.dmap
  - depthmap0002.dmap
  - ...
```

## 추가 정보

### 깊이 추정 알고리즘 수정

load_midas_model 함수를 수정하여, 더욱 좋은 깊이 추정 알고리즘을 사용하거나 GT depth영상을 넣을 수 있습니다


### DMAP 파일 형식

DMAP는 depthmap을 저장하는 일반적인 방식은 아닙니다. 해당 파일은 다음과 같은 설정을 포함합니다:

```python
dmap_config = {
  'has_normal': has_normal,           # 노멀 맵이 포함되어 있는지 여부 (기본값: False)
  'has_conf': has_conf,               # 신뢰도 맵이 포함되어 있는지 여부 (기본값: False)
  'has_views': has_views,             # 뷰 ID가 포함되어 있는지 여부 (기본값: False)
  'image_width': image_width,         # 입력 이미지의 너비 (원본 이미지의 너비의 절반이어야 함)
  'image_height': image_height,       # 입력 이미지의 높이 (원본 이미지의 높이의 절반이어야 함)
  'depth_width': depth_width,         # 깊이 맵의 너비
  'depth_height': depth_height,       # 깊이 맵의 높이
  'depth_min': depth_min,             # 최소 깊이 값
  'depth_max': depth_max,             # 최대 깊이 값
  'file_name': file_name,             # 이미지 파일 이름
  'reference_view_id': reference_view_id, # 참조 뷰의 ID
  'neighbor_view_ids': neighbor_view_ids, # 이웃 뷰 ID의 리스트
  'K': K,                             # 카메라 내부 매트릭스/2
  'R': R,                             # 카메라 회전 매트릭스
  'C': C                              # 카메라 이동 벡터
}
```


### 주의사항

has_normal, has_conf, has_views 값은 False로 설정되어 있습니다.
neighbor_view_ids는 별도로 설정되지 않았습니다.
