# EgoX 소스코드 독해 가이드

이 문서는 EgoX 코드를 처음 분석할 때 어떤 파일을 어떤 순서로 읽으면 좋은지 정리한 가이드다. 논문 개념과 코드 매핑은 `documents/paper_code_concept_mapping.md`에 더 자세히 정리되어 있으므로, 이 문서는 “실제로 코드를 읽는 순서”와 “각 파일이 맡는 일”에 집중한다.

## 1. 먼저 잡아야 할 큰 그림

EgoX 저장소는 크게 세 덩어리로 나뉜다.

| 영역 | 목적 | 먼저 볼 파일 |
| --- | --- | --- |
| 추론/테스트 | 준비된 exo video와 ego prior로 ego video 생성 | `scripts/infer_itw.sh`, `scripts/infer_ego4d.sh`, `infer.py`, `core/inference/wan.py` |
| 학습/파인튜닝 | Wan2.1 I2V에 LoRA를 붙여 EgoX 방식으로 학습 | `scripts/finetune.sh`, `finetune.py`, `core/finetune/models/wan_i2v/sft_trainer.py` |
| Ego prior 생성 | exo video를 3D로 올리고 ego pose에서 `ego_Prior.mp4` 렌더링 | `EgoX-EgoPriorRenderer/scripts/infer_vipe.sh`, `EgoX-EgoPriorRenderer/scripts/render_vipe_pointcloud.py` |

코드 전체를 한 번에 읽으려 하지 말고, 다음 순서를 추천한다.

1. `README.md`에서 실행 방식과 입력 데이터 구조를 본다.
2. `documents/paper_code_concept_mapping.md`에서 논문 개념과 코드 위치를 대략 연결한다.
3. 추론 경로를 먼저 읽는다. 학습보다 짧고 전체 입출력이 잘 보인다.
4. 그 다음 학습 경로를 읽는다. 특히 latent concat, mask, loss를 집중해서 본다.
5. 마지막으로 EgoPriorRenderer를 읽는다. 이 부분은 별도 프로젝트 성격이고 코드량이 크다.

## 2. 빠른 30분 독해 순서

시간이 적다면 아래 파일만 먼저 읽어도 전체 구조를 잡을 수 있다.

1. `scripts/infer_itw.sh`
   - in-the-wild 예제 추론을 어떻게 호출하는지 본다.
   - `--model_path`, `--lora_path`, `--meta_data_file`, `--use_GGA`, `--cos_sim_scaling_factor`가 실제로 어떻게 쓰이는지 확인한다.

2. `infer.py`
   - `meta.json`을 읽어 exo video, ego prior, prompt, camera/depth 정보를 모으는 진입점이다.
   - `--use_GGA` 블록이 길지만 중요하다. 이 블록은 depth와 camera pose로 GGA attention에 필요한 3D 방향 벡터를 만든다.

3. `core/inference/wan.py`
   - exo/ego prior 동영상을 로드하고 `WanWidthConcatImageToVideoPipeline`을 호출한다.
   - 모델 입력 canvas가 `[exo 784px | ego 448px]` 형태로 붙는다는 점을 확인한다.

4. `core/finetune/models/wan_i2v/sft_trainer.py`
   - 가장 중요한 파일이다.
   - `WanWidthConcatImageToVideoPipeline.prepare_latents()`에서 exo latent와 ego prior latent가 어떻게 조건으로 들어가는지 본다.
   - `compute_loss()`에서 exo 영역은 clean하게 유지하고 ego 영역에만 loss를 거는 부분을 본다.

5. `core/finetune/models/wan_i2v/custom_transformer.py`
   - GGA가 실제 attention bias로 들어가는 파일이다.
   - `WanAttnProcessor2_0.__call__()`와 `WanTransformer3DModel_GGA.forward()`만 먼저 읽으면 된다.

6. `EgoX-EgoPriorRenderer/scripts/render_vipe_pointcloud.py`
   - `ego_Prior.mp4`가 어떻게 만들어지는지 보는 파일이다.
   - 처음에는 전체를 다 읽지 말고 `main()`, `load_camera_params_from_meta()`, `project_points_to_image_sequential()`만 본다.

## 3. 테스트/추론 쪽 읽는 법

추론 경로는 “입력 meta 읽기 → GGA 준비 → Wan pipeline 호출 → MP4 저장” 순서로 읽으면 된다.

### 3.1 실행 스크립트

#### `scripts/infer_itw.sh`

- in-the-wild 예제용 추론 wrapper다.
- 보통 `example/in_the_wild/meta.json`을 입력으로 사용한다.
- `--use_GGA`가 켜져 있으면 `infer.py`가 depth map과 camera parameter로 GGA tensor를 만든다.
- `--cos_sim_scaling_factor`는 GGA geometry bias 강도에 해당한다.

#### `scripts/infer_ego4d.sh`

- EgoExo4D 예제용 추론 wrapper다.
- 구조는 `infer_itw.sh`와 거의 같다.
- EgoExo4D 예제는 GT ego camera pose와 depth map이 더 명시적으로 준비되어 있다고 보면 된다.

### 3.2 추론 진입점

#### `infer.py`

하는 일:

- `meta_data_file`을 읽고 `test_datasets` 안의 sample들을 순회한다.
- 각 sample에서 다음 값을 모은다.
  - `exo_path`: 입력 3인칭 비디오
  - `ego_prior_path`: point cloud rendering으로 만든 ego prior
  - `prompt`: text condition
  - `camera_intrinsics`, `camera_extrinsics`: exo camera 정보
  - `ego_intrinsics`, `ego_extrinsics`: target ego camera trajectory
- pretrained Wan transformer와 EgoX LoRA weight를 로드한다.
- `--use_GGA`가 켜진 경우 GGA용 geometry tensor를 만든다.
- `core/inference/wan.py`의 `generate_video()`를 호출한다.

읽을 때 집중할 부분:

1. `main(args)` 초반의 meta parsing
   - 논문에서 말하는 입력 `X`, prior `P`, camera pose `phi`가 코드에서 어떤 필드인지 확인한다.

2. model load 부분
   - `WanTransformer3DModel_GGA.from_pretrained()`
   - `WanWidthConcatImageToVideoPipeline.from_pretrained()`
   - `pipe.load_lora_weights()`, `pipe.fuse_lora()`

3. `if args.use_GGA:` 블록
   - depth map을 읽는다.
   - exo depth를 3D point map으로 역투영한다.
   - ego camera origin에서 exo point로 향하는 방향 벡터를 만든다.
   - ego camera ray를 만든다.
   - `attn_maps`, `attn_masks`, `point_vecs_per_frame`, `cam_rays`를 만든다.

주의할 점:

- `C, F, H, W = 16, 13, 56, 154`가 하드코딩되어 있다. 이는 Wan latent/channel/patch 크기와 연결된다.
- `ego_extrinsic[::4]`, `point_map[::4]`처럼 49 frame에서 latent frame 13개로 맞추는 downsampling이 있다.
- `--do_kv_cache`를 켜면 현재 transformer 구현상 GGA bias branch가 우회될 수 있다. 이 부분은 `custom_transformer.py` 주석도 같이 보자.

### 3.3 실제 Wan pipeline 호출

#### `core/inference/wan.py`

하는 일:

- exo video와 ego prior video를 `diffusers.utils.load_video()`로 읽는다.
- exo video를 `784x448`로 resize한다.
- ego prior video를 target ego view로 둔다.
- 최종 canvas width를 `exo_width + ego_width`로 만든다.
- `pipe(...)`를 호출해 generated video를 얻고 `imageio.mimsave()`로 저장한다.

읽을 때 집중할 부분:

- `width = exo_width + ego_width`
  - 이 코드가 논문의 width-wise exo concat과 연결된다.
- `attention_GGA`, `attention_mask_GGA`, `point_vecs_per_frame`, `cam_rays`
  - `infer.py`에서 만든 GGA tensor가 그대로 pipeline으로 전달된다.

## 4. 학습 쪽 읽는 법

학습 경로는 “CLI args → trainer 선택 → dataset cache → latent/mask/loss → LoRA 저장” 순서로 읽으면 된다.

### 4.1 실행 스크립트와 진입점

#### `scripts/finetune.sh`

하는 일:

- `accelerate launch finetune.py ...` 형태로 학습을 실행한다.
- 실제 논문 설정과 가까운 값이 들어 있다.
  - `--model_name wan-i2v`
  - `--training_type lora`
  - `--train_resolution 49x448x1232`
  - `--rank 256`
  - `--lora_alpha 256`
  - `--cos_sim_scaling_factor`

읽을 때 집중할 부분:

- `train_resolution`의 width `1232`는 `784 + 448`이다.
- 학습 타입이 `lora`라서 전체 모델을 다 학습하지 않는다.

#### `finetune.py`

하는 일:

- `Args.parse_args()`로 CLI 인자를 읽는다.
- `get_model_cls(args.model_name, args.training_type)`로 trainer class를 고른다.
- `trainer.fit()`을 호출한다.

이 파일은 짧다. “어디로 들어가는지 확인하는 진입점” 정도로 보면 된다.

### 4.2 CLI와 설정

#### `core/finetune/schemas/args.py`

하는 일:

- 학습에 필요한 인자를 정의한다.
- 중요한 인자:
  - `model_path`
  - `model_name`
  - `training_type`
  - `data_root`
  - `meta_data_file`
  - `train_resolution`
  - `rank`, `lora_alpha`, `target_modules`
  - `cos_sim_scaling_factor`

읽을 때 집중할 부분:

- `target_modules = ["to_q", "to_k", "to_v", "to_out.0"]`
  - LoRA가 attention projection 쪽에 붙는다는 의미다.
- `train_resolution`
  - frame, height, width 순서다.

### 4.3 Trainer 공통 로직

#### `core/finetune/trainer.py`

하는 일:

- 학습 전체 루프와 공통 유틸을 담당한다.
- dataset 준비, optimizer 준비, checkpoint 저장/로드, LoRA adapter 추가 등을 처리한다.

읽을 때 집중할 부분:

1. `prepare_trainable_parameters()`
   - `training_type == "lora"`이면 대부분 parameter를 freeze한다.
   - transformer에 `LoraConfig` adapter를 추가한다.

2. `prepare_optimizer()`
   - 실제 trainable parameter만 optimizer에 넘긴다.

3. checkpoint 관련 hook
   - LoRA weight 저장/로드가 어떻게 되는지 볼 수 있다.

이 파일은 길기 때문에 처음부터 끝까지 정독하지 말고, LoRA와 optimizer 부분만 먼저 보면 된다.

### 4.4 Model registry

#### `core/finetune/models/utils.py`

하는 일:

- `model_name`, `training_type`으로 trainer class를 찾는다.

#### `core/finetune/models/wan_i2v/lora_trainer.py`

하는 일:

- `WanI2VLoraTrainer`를 `wan-i2v`, `lora` 조합으로 등록한다.
- 실제 구현은 `WanI2VSftTrainer`를 거의 그대로 사용한다.

읽을 때 포인트:

- EgoX의 LoRA 학습은 별도 loss를 새로 정의한 trainer가 아니라, SFT trainer의 학습 로직을 LoRA trainable parameter 설정과 함께 쓰는 방식이다.

### 4.5 학습 dataset과 cache

#### `core/finetune/datasets/wan_dataset.py`

하는 일:

- 학습 meta를 읽고 exo video, ego GT video, ego prior video, prompt, camera/depth 정보를 sample로 만든다.
- prompt embedding, video latent, image embedding, GGA tensor를 cache에 저장한다.
- 학습 batch가 바로 transformer loss 계산에 들어갈 수 있는 형태가 되도록 한다.

읽을 때 집중할 부분:

1. `BaseWanDataset.__init__()`
   - 학습용 meta schema를 확인한다.
   - 현재 예제 추론용 `test_datasets[]` meta와 학습용 meta 구조가 다르므로 주의한다.

2. `getitem()`
   - prompt, exo video, ego GT, ego prior를 가져온다.
   - video를 preprocess하고 VAE latent로 encode한다.

3. latent concat 부분
   - `encoded_exo_ego_gt_video = torch.cat([encoded_exo_video, encoded_ego_gt_video], dim=-1)`
   - `encoded_exo_ego_prior_video = torch.cat([encoded_exo_video, encoded_ego_prior_video], dim=-1)`
   - `dim=-1`이 width 방향이다.

4. GGA cache 생성 부분
   - depth map을 읽는다.
   - camera pose로 exo point direction과 ego ray를 만든다.
   - `attn_maps`, `attn_masks`, `point_vecs_per_frame`, `cam_rays`를 저장한다.

5. return dict
   - `compute_loss()`에서 쓰는 key가 무엇인지 확인한다.

중요 return key:

| key | 의미 |
| --- | --- |
| `encoded_exo_ego_gt_video` | exo clean latent + ego GT latent |
| `encoded_exo_ego_prior_video` | exo clean latent + ego prior latent |
| `prompt_embedding` | text condition |
| `image_embedding` | Wan I2V image condition |
| `attention_GGA` | GGA direction map |
| `attention_mask_GGA` | exo/ego token 구분 |
| `point_vecs_per_frame` | ego origin에서 exo point로 향하는 vector |
| `cam_rays` | ego camera ray |

### 4.6 EgoX 핵심 학습/추론 pipeline

#### `core/finetune/models/wan_i2v/sft_trainer.py`

이 저장소의 핵심 파일이다. 크게 두 부분으로 나눠서 읽으면 된다.

#### A. `WanWidthConcatImageToVideoPipeline`

역할:

- diffusers `WanImageToVideoPipeline`을 EgoX용으로 확장한다.
- 추론 시 exo video와 ego prior video를 동시에 받아 `[exo | ego]` latent canvas를 구성한다.

중요 함수:

1. `_setup_ic_lora_latent()`
   - exo video를 VAE latent로 encode한다.
   - ego 영역은 random latent로 만든다.
   - `torch.cat([exo_latents, ego_latents], dim=-1)`로 width-wise canvas를 만든다.

2. `prepare_latents()`
   - ego prior video를 encode한다.
   - shared latent의 오른쪽 ego 영역을 ego prior latent로 교체한다.
   - mask와 latent condition을 channel-wise로 붙여 transformer condition을 만든다.

3. `__call__()`
   - exo/ego prior preprocessing
   - prompt/image embedding
   - denoising loop
   - transformer 호출
   - scheduler step

읽을 때 질문:

- exo latent는 어디서 clean하게 유지되는가?
- ego prior는 어떤 tensor의 어느 영역에 들어가는가?
- width-wise concat과 channel-wise condition이 각각 어느 줄에서 발생하는가?

#### B. `WanI2VSftTrainer.compute_loss()`

역할:

- 학습 batch를 받아 diffusion loss를 계산한다.

중요 흐름:

1. `latent = batch["encoded_exo_ego_gt_video"]`
   - target 역할이다. 왼쪽은 exo, 오른쪽은 ego GT다.

2. `latent_condition = batch["encoded_exo_ego_prior_video"]`
   - condition 역할이다. 왼쪽은 exo, 오른쪽은 ego prior다.

3. `mask_lat_size` 생성
   - exo/ego 영역과 temporal condition을 구분한다.

4. `condition = torch.concat([mask_lat_size, latent_condition], dim=1)`
   - channel-wise inpainting condition이다.

5. `noisy_latents[:, :, :, :, :-ego_latent_width] = latent[:, :, :, :, :-ego_latent_width]`
   - exo 영역을 clean latent로 되돌린다.

6. loss crop
   - `predicted_noise`와 `target` 모두 오른쪽 ego 영역만 잘라서 loss를 계산한다.

가장 중요한 이해 포인트:

- EgoX는 전체 canvas를 생성하는 것처럼 보이지만 실제 목표는 오른쪽 ego 영역이다.
- 왼쪽 exo 영역은 condition canvas이며, 논문의 clean exocentric latent `x_0`에 해당한다.

## 5. GGA를 읽는 법

GGA는 세 파일에 나뉘어 있다.

| 단계 | 파일 | 하는 일 |
| --- | --- | --- |
| 추론 precompute | `infer.py` | depth/camera에서 GGA vector 생성 |
| 학습 precompute/cache | `core/finetune/datasets/wan_dataset.py` | 같은 GGA vector를 학습 sample마다 생성/캐시 |
| transformer 적용 | `core/finetune/models/wan_i2v/custom_transformer.py` | cosine similarity를 attention bias로 적용 |

### 5.1 `custom_transformer.py`

처음 읽을 함수:

1. `WanTransformer3DModel_GGA.forward()`
   - `attention_GGA`, `point_vecs_per_frame`, `cam_rays`를 받는다.
   - vector들을 patch-token scale로 pooling한다.
   - cosine similarity matrix `cos_sim`을 만든다.
   - transformer block에 `cos_sim`을 넘긴다.

2. `WanAttnProcessor2_0.__call__()`
   - `cos_sim + 1` 후 `log()`를 취한다.
   - ego token attention에 bias로 넣는다.
   - exo token과 ego token을 나누어 attention을 계산한 뒤 다시 합친다.

주의할 점:

- GGA branch는 `if cos_sim is not None and not do_kv_cache:` 조건에서만 돈다.
- 현재 학습 `compute_loss()`에서는 transformer 호출에 `do_kv_cache=True`가 들어가 있다. 이 조건이 실제 GGA 적용 의도와 맞는지는 별도로 확인할 가치가 있다.
- `cos_sim_scaling_factor`는 논문에서 geometry prior의 강도 `lambda_g`에 해당한다고 보면 된다.

## 6. Ego prior renderer 읽는 법

Ego prior 생성 코드는 `EgoX-EgoPriorRenderer` 안에 있고, 사실상 별도 프로젝트다. 처음에는 핵심 path만 읽자.

### 6.1 ViPE inference

#### `EgoX-EgoPriorRenderer/scripts/infer_vipe.sh`

하는 일:

- `vipe infer`를 실행한다.
- exo video에서 depth, pose, mask 등 ViPE artifact를 만든다.
- `--assume_fixed_camera_pose`가 중요하다. EgoX는 fixed exo camera 입력을 가정한다.

#### `EgoX-EgoPriorRenderer/vipe/cli/main.py`

하는 일:

- `vipe infer`, `vipe visualize` 같은 CLI command를 정의한다.

#### `EgoX-EgoPriorRenderer/vipe/pipeline/default.py`

하는 일:

- ViPE annotation pipeline 본체다.
- intrinsics, mask, SLAM, depth, artifact 저장 흐름을 묶는다.

#### `EgoX-EgoPriorRenderer/vipe/pipeline/processors.py`

하는 일:

- intrinsics processor, mask processor, depth processor 등 개별 preprocessing module이 있다.
- `TrackAnythingProcessor`, `AdaptiveDepthProcessor`가 논문에서 말하는 dynamic object masking, depth processing과 연결된다.

### 6.2 Ego prior rendering

#### `EgoX-EgoPriorRenderer/scripts/render_vipe.sh`

하는 일:

- `render_vipe_pointcloud.py` 실행 wrapper다.
- `--input_dir`, `--meta_json_path`, `--out_dir`, `--point_size`, `--fish_eye_rendering`, `--use_mean_bg`를 넘긴다.

#### `EgoX-EgoPriorRenderer/scripts/render_vipe_pointcloud.py`

처음 읽을 함수:

1. `main()`
   - meta에서 camera parameter를 읽는다.
   - background point cloud를 만든다.
   - target ego pose마다 렌더링한다.
   - `ego_Prior.mp4`로 저장한다.

2. `load_camera_params_from_meta()`
   - `meta.json`에서 exo/ego intrinsics/extrinsics를 읽는다.
   - 논문의 `phi`가 여기서는 `ego_extrinsics`다.

3. `build_background_pointcloud()`
   - depth와 RGB를 exo camera에서 3D point cloud로 만든다.
   - instance mask가 있으면 background만 남긴다.

4. `build_mean_background_pointcloud()`
   - `--use_mean_bg`일 때 여러 frame의 static background를 평균화한다.

5. `project_points_to_image_sequential()`
   - 각 ego camera pose에서 point cloud를 렌더링한다.
   - 논문 Eq. (2)의 `P = render(X, D_f, phi)`에 해당한다.

6. `render_points_fisheye()` / `render_points_pytorch3d()`
   - 실제 PyTorch3D 렌더링 함수다.

### 6.3 Depth 변환

#### `EgoX-EgoPriorRenderer/scripts/convert_depth_zip_to_npy.py`

하는 일:

- ViPE depth artifact zip의 `.exr` depth map을 EgoX 본체가 읽는 `.npy` depth map으로 변환한다.
- `infer.py`와 `wan_dataset.py`는 이 `.npy` depth map을 읽어 GGA vector를 만든다.

## 7. 데이터와 파일 구조를 같이 보는 법

### 7.1 추론 예제 데이터

먼저 볼 파일:

- `example/in_the_wild/meta.json`
- `example/egoexo4D/meta.json`

중요 필드:

| 필드 | 의미 |
| --- | --- |
| `exo_path` | 입력 3인칭 동영상 |
| `ego_prior_path` | renderer가 만든 1인칭 prior |
| `ego_gt_path` | GT ego video, 예제/평가용 |
| `prompt` | Wan text condition |
| `camera_intrinsics` | exo camera 내부 파라미터 |
| `camera_extrinsics` | exo camera pose |
| `ego_intrinsics` | target ego camera 내부 파라미터 |
| `ego_extrinsics` | target ego trajectory |

### 7.2 학습 데이터

학습 dataset 코드는 추론 예제 meta와 다른 schema를 기대한다.

`core/finetune/datasets/wan_dataset.py` 기준으로 학습 meta에는 대략 다음 필드가 필요하다.

- `exo_video_path`
- `ego_video_path`
- `ego_prior_path`
- `prompt`
- `take_name`
- `vipe_results_path`
- `best_camera`
- `camera_intrinsics`
- `camera_extrinsics`
- `ego_intrinsics`
- `ego_extrinsics`

분석할 때 주의할 점:

- `example/*/meta.json`은 추론용이다.
- 학습용 meta는 preprocessing pipeline이 만든 별도 포맷일 가능성이 크다.
- 따라서 학습 dataset을 이해할 때는 `data_preprocess`와 `wan_dataset.py`를 같이 봐야 한다.

## 8. 보조 파일

### `caption.py`

하는 일:

- exo/ego prior/GT frame을 보고 GPT-4o로 text prompt를 만드는 전처리 유틸이다.
- 논문 supplementary의 prompt generation protocol과 관련 있다.

읽을 때:

- 모델 학습/추론 core는 아니므로 나중에 봐도 된다.
- `meta.json`의 `prompt`가 어떻게 만들어졌는지 궁금할 때 읽는다.

### `meta_init.py`

하는 일:

- dataset folder를 스캔해서 초기 meta JSON을 만든다.
- `videos/<take>/exo.mp4` 같은 구조를 기준으로 sample entry를 만든다.

읽을 때:

- custom dataset을 만들 때 필요하다.
- core model 이해에는 필수는 아니다.

### `core/tokenizer/wan.py`

하는 일:

- Wan VAE를 사용해 video frame을 latent로 encode/decode하는 보조 wrapper다.
- 현재 핵심 학습 경로에서는 `sft_trainer.py`의 VAE encode 함수와 함께 보면 된다.

### `core/finetune/schemas/components.py`, `state.py`

하는 일:

- trainer가 쓰는 component와 training state dataclass/pydantic schema다.
- 큰 흐름을 이해한 뒤 필요할 때만 보면 된다.

## 9. 파일별 역할 요약

| 파일 | 역할 | 독해 우선순위 |
| --- | --- | --- |
| `README.md` | 설치, 모델 weight, 실행법, 데이터 구조 | 높음 |
| `documents/paper_code_concept_mapping.md` | 논문 개념과 코드 위치 매핑 | 높음 |
| `scripts/infer_itw.sh` | in-the-wild 추론 실행 예시 | 높음 |
| `scripts/infer_ego4d.sh` | EgoExo4D 추론 실행 예시 | 높음 |
| `infer.py` | 추론 main, meta parsing, GGA precompute | 매우 높음 |
| `core/inference/wan.py` | video load, Wan pipeline 호출, output 저장 | 높음 |
| `scripts/finetune.sh` | 학습 실행 예시 | 높음 |
| `finetune.py` | 학습 entrypoint | 높음 |
| `core/finetune/schemas/args.py` | CLI argument 정의 | 중간 |
| `core/finetune/trainer.py` | 공통 training loop, LoRA setup | 높음 |
| `core/finetune/models/wan_i2v/lora_trainer.py` | LoRA trainer registry | 중간 |
| `core/finetune/models/wan_i2v/sft_trainer.py` | EgoX latent conditioning, denoising, loss | 매우 높음 |
| `core/finetune/models/wan_i2v/custom_transformer.py` | Wan transformer + GGA attention | 매우 높음 |
| `core/finetune/datasets/wan_dataset.py` | 학습 sample, latent/GGA cache | 매우 높음 |
| `core/finetune/datasets/utils.py` | image/video load, projection utility | 중간 |
| `caption.py` | prompt 생성 | 낮음 |
| `meta_init.py` | meta 초기화 | 낮음 |
| `EgoX-EgoPriorRenderer/scripts/infer_vipe.sh` | ViPE inference 실행 | 중간 |
| `EgoX-EgoPriorRenderer/scripts/render_vipe.sh` | ego prior rendering 실행 | 중간 |
| `EgoX-EgoPriorRenderer/scripts/render_vipe_pointcloud.py` | point cloud 기반 ego prior 생성 | 높음 |
| `EgoX-EgoPriorRenderer/vipe/pipeline/default.py` | ViPE pipeline 본체 | 중간 |
| `EgoX-EgoPriorRenderer/vipe/pipeline/processors.py` | depth/mask/intrinsics processor | 중간 |
| `EgoX-EgoPriorRenderer/vipe/utils/viser.py` | in-the-wild ego camera 수동 annotation viewer | 낮음 |

## 10. 추천 학습 루트

### 루트 A: “추론을 먼저 이해”

1. `scripts/infer_itw.sh`
2. `example/in_the_wild/meta.json`
3. `infer.py`
4. `core/inference/wan.py`
5. `core/finetune/models/wan_i2v/sft_trainer.py`의 pipeline 부분
6. `core/finetune/models/wan_i2v/custom_transformer.py`의 GGA 부분

이 루트는 결과 동영상이 어떻게 생성되는지 빠르게 이해하는 데 좋다.

### 루트 B: “학습을 먼저 이해”

1. `scripts/finetune.sh`
2. `finetune.py`
3. `core/finetune/schemas/args.py`
4. `core/finetune/trainer.py`
5. `core/finetune/datasets/wan_dataset.py`
6. `core/finetune/models/wan_i2v/sft_trainer.py`의 `compute_loss()`
7. `core/finetune/models/wan_i2v/custom_transformer.py`

이 루트는 loss, LoRA, dataset cache를 이해하는 데 좋다.

### 루트 C: “Ego prior 생성 이해”

1. `EgoX-EgoPriorRenderer/README.md`
2. `EgoX-EgoPriorRenderer/scripts/infer_vipe.sh`
3. `EgoX-EgoPriorRenderer/scripts/render_vipe.sh`
4. `EgoX-EgoPriorRenderer/scripts/render_vipe_pointcloud.py`
5. `EgoX-EgoPriorRenderer/vipe/pipeline/default.py`
6. `EgoX-EgoPriorRenderer/vipe/pipeline/processors.py`

이 루트는 `ego_Prior.mp4`와 depth map이 어디서 오는지 이해하는 데 좋다.

## 11. 이해 체크리스트

아래 질문에 답할 수 있으면 코드 구조를 상당히 이해한 것이다.

- `meta.json`의 `exo_path`, `ego_prior_path`, `ego_extrinsics`는 각각 어느 코드에서 쓰이는가?
- 왜 total width가 `1232`인가?
- `encoded_exo_ego_gt_video`와 `encoded_exo_ego_prior_video`의 차이는 무엇인가?
- exo latent는 왜 noise를 넣은 뒤 다시 clean latent로 덮어쓰는가?
- loss는 왜 ego width 영역에만 걸리는가?
- `attention_GGA`, `point_vecs_per_frame`, `cam_rays`는 각각 어떤 geometry를 담는가?
- `cos_sim_scaling_factor`는 GGA에서 어떤 역할을 하는가?
- `ego_Prior.mp4`는 EgoX 본체에서 만드는가, renderer 서브프로젝트에서 만드는가?
- `--use_GGA`와 `--do_kv_cache`를 같이 쓸 때 어떤 코드 경로가 실행되는가?
- 추론용 meta schema와 학습용 meta schema는 어떻게 다른가?

## 12. 다음에 보면 좋은 것

코드를 한 번 따라간 뒤에는 다음 순서로 더 깊게 보면 좋다.

1. `documents/paper_code_concept_mapping.md`
   - 논문 섹션별로 코드 주석과 다시 대조한다.

2. `git diff`로 이번에 추가된 주석만 보기
   - 연구 개념 주석이 붙은 지점이 곧 핵심 독해 지점이다.

3. 작은 sample 하나를 기준으로 tensor shape 추적
   - exo pixel: `448x784`
   - ego pixel: `448x448`
   - total pixel: `448x1232`
   - latent spatial scale: 대략 `1/8`
   - latent width: `98 + 56 = 154`

4. GGA on/off 비교
   - `--use_GGA`를 켠 경우와 끈 경우 어떤 tensor가 `None`이 되는지 따라가면 GGA의 의존성을 이해하기 쉽다.

