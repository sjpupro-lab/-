# Row-wise Residual VAE-like Image Generation — Full Experiment Archive

## 목적

이 프로젝트는 대화 중 실험한 내용을 풀 버전으로 정리한 아카이브입니다.

핵심 질문:

> 작은 latent / diff / row-wise residual 정보를 이용해서 이미지를 빠르게 다시 그리거나 새 캐릭터를 만들 수 있는가?

## 핵심 아이디어

```text
이미지 = low-resolution base + learned row-wise residual
```

즉 이미지를 픽셀 전체로 직접 저장하기보다:

```text
1. 작은 latent로 전체 구조 저장
2. base = latent를 키운 저주파 이미지
3. residual(diff) = 원본 - base
4. residual을 행(row) 단위 신호로 보고 학습
5. latent와 residual coefficient를 섞어 새 이미지 생성
```

## 왜 row 단위인가?

블록 단위가 아니라 행 단위로 보면:

```text
한 행 = 하나의 긴 신호
행마다 residual 패턴을 학습
행 단위 coefficient를 저장하거나 생성
```

이 구조는 오디오 신호 처리, VAE, super-resolution, residual compression의 중간 형태입니다.

## 실험 흐름

### 1. 단순 스케일 테스트
- 256 → 512
- 128 → 512
- 64 → 512

결과: 자연 이미지에서는 256→512가 거의 동일하게 보임.

관련 파일:
- `images/experiments/scale_tests/`

### 2. Tick / FFT / 채널 테스트
RGBA를 tick 채널로 나누고, FFT 기반 확장을 실험했습니다.

중요 결론:
- FFT를 생성 과정에 무리하게 넣으면 링잉이 생김
- tick은 정밀도 표현에는 유용하지만, delta/cumsum 방식은 불안정
- 안정적인 방향은 latent/base + learned residual

관련 파일:
- `images/experiments/fft_tick_tests/`
- `images/experiments/latent_tick_tests/`

### 3. Diff / residual 테스트
base와 residual을 분리했습니다.

```text
base = low-res latent upscale
residual = original - base
```

단순 residual 저장은 효과가 제한적이지만, residual을 row-wise로 학습하면 품질이 개선됨.

관련 파일:
- `images/experiments/residual_tests/`

### 4. Row-wise residual learning
행 단위 basis를 학습하여 residual을 복원했습니다.

결과:
```text
base 128→512: PSNR 약 33.82
row-wise residual learned: PSNR 약 36.20
```

관련 파일:
- `images/experiments/row_learning_tests/`

### 5. Mini RowVAE 캐릭터 생성
여러 간단한 캐릭터를 생성하고 학습했습니다.

학습:
```text
캐릭터 이미지
→ 32/64 latent
→ base
→ residual
→ row basis 학습
→ row coefficient 저장
```

생성:
```text
여러 latent base mix
+ 여러 residual coefficient mix
+ coefficient noise
→ 새 캐릭터 생성
```

결과:
- 학습 캐릭터와 다른 새 캐릭터 생성 성공
- diffusion step 없이 single-pass 생성

관련 파일:
- `images/characters/`

## 구현 뼈대

`code/rowvae_skeleton.py`

포함 함수:
- `train_row_residual_basis`
- `decode`
- `generate_mixed`
- `psnr`
- `mse`

## 프로젝트 구조

```text
rowvae_full_project/
├── README.md
├── code/
│   └── rowvae_skeleton.py
└── images/
    ├── source/
    ├── experiments/
    │   ├── scale_tests/
    │   ├── fft_tick_tests/
    │   ├── latent_tick_tests/
    │   ├── residual_tests/
    │   └── row_learning_tests/
    └── characters/
```

## 핵심 결론

이 방식은 완전한 diffusion 모델이 아니라:

```text
row-wise VAE
+ residual generator
+ signal compression
+ fast renderer
```

에 가깝습니다.

장점:
- diffusion보다 빠른 single-pass 복원 가능
- 작은 latent로 구조 보존
- residual coefficient mix로 창작 가능
- 행 단위라 구현이 단순하고 빠름

한계:
- 아직은 작은 실험
- 큰 데이터셋 학습 필요
- 의미 기반 생성보다는 신호 기반 생성
- row만 쓰면 세로/가로 구조의 상호작용이 약할 수 있음

## 다음 발전 방향

1. row + column 양방향 residual 학습
2. tick RGBA 정밀도 추가
3. coefficient generator를 MLP/Transformer로 교체
4. 대규모 캐릭터 데이터셋 학습
5. 512/1024 고해상도 생성 테스트
6. latent + residual + style vector 분리

## 포함 이미지

### 학습 캐릭터
![training](images/characters/mini_rowvae_train_chars.png)

### 원본 / base / 복원
![reconstruction](images/characters/mini_rowvae_recon_compare.png)

### 생성 캐릭터
![generated](images/characters/mini_rowvae_generated_grid.png)
