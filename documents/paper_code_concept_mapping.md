# EgoX 논문 개념과 소스코드 매핑

이 문서는 `documents/paper/Kang et al. - 2025 - EgoX Egocentric Video Generation from a Single Exocentric Video.pdf`의 주요 연구 개념이 현재 저장소의 어느 코드에 대응되는지 정리한다. 다음 단계에서 소스코드에 논문 개념 주석을 추가할 때 기준 문서로 쓰는 것을 목표로 한다.

## 1. 전체 구조 요약

논문의 EgoX는 단일 3인칭 동영상(`exocentric video`, `X`)을 입력으로 받아, 같은 장면을 1인칭 시점(`egocentric video`, `Y`)으로 생성하는 프레임워크다. 코드에서는 크게 두 영역으로 나뉜다.

| 논문 구성 | 코드 서브시스템 | 핵심 파일 |
| --- | --- | --- |
| Exo-to-Ego generation | Wan2.1 I2V 기반 학습/추론 본체 | `finetune.py`, `infer.py`, `core/finetune/models/wan_i2v/sft_trainer.py`, `core/inference/wan.py` |
| Egocentric prior rendering | ViPE 결과와 point cloud renderer로 `ego_Prior.mp4` 생성 | `EgoX-EgoPriorRenderer/scripts/render_vipe_pointcloud.py`, `EgoX-EgoPriorRenderer/vipe/pipeline/default.py` |
| Unified conditioning | exo latent와 ego prior latent를 결합해 diffusion transformer에 조건으로 투입 | `core/finetune/models/wan_i2v/sft_trainer.py`, `core/finetune/datasets/wan_dataset.py` |
| Clean latent representation | exo latent 영역을 clean condition으로 유지하고 ego 영역만 denoise/loss 계산 | `core/finetune/models/wan_i2v/sft_trainer.py` |
| Geometry-Guided Self-Attention, GGA | 3D 방향 벡터 cosine similarity를 self-attention bias로 사용 | `infer.py`, `core/finetune/datasets/wan_dataset.py`, `core/finetune/models/wan_i2v/custom_transformer.py` |
| LoRA adaptation | Wan transformer에 PEFT LoRA adapter 추가 | `core/finetune/trainer.py`, `core/finetune/models/wan_i2v/lora_trainer.py`, `scripts/finetune.sh` |
| Text prompt conditioning | exo/ego view 설명 prompt 생성 및 meta에 저장 | `caption.py`, `example/*/meta.json` |

전체 데이터 흐름은 다음과 같다.

1. `EgoX-EgoPriorRenderer`에서 exo video를 ViPE로 3D reconstruction/depth 처리한다.
2. `render_vipe_pointcloud.py`가 camera intrinsics/extrinsics와 ego trajectory를 사용해 `ego_Prior.mp4`를 생성한다.
3. `meta.json`에는 `exo_path`, `ego_prior_path`, `prompt`, exo/ego camera parameter가 저장된다.
4. 학습에서는 `core/finetune/datasets/wan_dataset.py`가 exo, ego GT, ego prior를 VAE latent로 encode하고 캐시한다.
5. 추론에서는 `infer.py`가 exo/ego prior와 camera/depth 정보를 읽어 GGA tensor를 만들고 `generate_video()`로 넘긴다.
6. `WanWidthConcatImageToVideoPipeline`과 `WanTransformer3DModel_GGA`가 conditioning, denoising, GGA attention을 수행한다.

## 2. 논문 개념별 상세 매핑

### 2.1 Exo-to-Ego View Generation 문제 정의

- 논문 위치: Sec. 1, Sec. 3, Fig. 1, Fig. 2
- 개념: 단일 exocentric video `X = {X_i}`와 egocentric camera pose `phi = {phi_i}`를 사용해 egocentric video `Y = {Y_i}`를 생성한다. 극단적 시점 변화, 작은 view overlap, unseen region synthesis, unrelated exo region suppression이 핵심 난점이다.
- 코드 매핑:
  - `infer.py`
    - `meta_data_file['test_datasets']`에서 `exo_path`, `ego_prior_path`, `prompt`, `camera_intrinsics`, `camera_extrinsics`, `ego_intrinsics`, `ego_extrinsics`를 읽는다.
    - 하나의 sample마다 exo video와 ego prior를 `generate_video()`에 전달한다.
  - `core/inference/wan.py`
    - `generate_video()`가 exo video와 ego prior video를 로드하고, exo는 `784x448`, ego prior는 ego view 크기로 전처리한 뒤 pipeline을 호출한다.
  - `example/in_the_wild/meta.json`, `example/egoexo4D/meta.json`
    - 논문 문제 정의에서 필요한 입력 묶음이 JSON으로 표현된다.
- 주석 후보:
  - `infer.py`에서 meta 필드를 읽는 부분에 “논문 Sec. 3의 `X`, `P`, camera pose 입력을 구성하는 단계”라고 설명한다.

### 2.2 EgoX 전체 파이프라인

- 논문 위치: Sec. 3, Fig. 3
- 개념: exo video를 3D point cloud로 lifting하고 ego pose에서 render해 ego prior `P`를 만든 뒤, exo video `X`와 ego prior `P`를 video diffusion model에 조건으로 넣는다.
- 코드 매핑:
  - Ego prior 생성:
    - `EgoX-EgoPriorRenderer/scripts/infer_vipe.sh`
    - `EgoX-EgoPriorRenderer/scripts/render_vipe.sh`
    - `EgoX-EgoPriorRenderer/scripts/render_vipe_pointcloud.py`
  - EgoX diffusion 본체:
    - `infer.py`
    - `core/inference/wan.py`
    - `core/finetune/models/wan_i2v/sft_trainer.py`
    - `core/finetune/models/wan_i2v/custom_transformer.py`
  - 학습:
    - `finetune.py`
    - `core/finetune/trainer.py`
    - `core/finetune/datasets/wan_dataset.py`
- 구현 설명:
  - 저장소 루트의 EgoX 본체는 `ego_Prior.mp4`가 이미 준비되어 있다는 전제로 학습/추론을 수행한다.
  - `EgoX-EgoPriorRenderer`는 ViPE inference, depth/mask/pose artifact, point cloud rendering을 담당하는 별도 서브프로젝트다.
- 주석 후보:
  - `core/inference/wan.py`의 `generate_video()` 시작부에 “Fig. 3의 diffusion generation stage”를 표시한다.
  - renderer의 `main()` 또는 frame rendering loop에 “Fig. 3의 egocentric prior rendering stage”를 표시한다.

### 2.3 Egocentric Point Cloud Rendering

- 논문 위치: Sec. 3.1, Eq. (2), Fig. 3, Fig. 9
- 개념: exo RGB video `X`, aligned depth `D_f`, ego camera pose `phi`로 ego prior `P = render(X, D_f, phi)`를 만든다. `P`는 target ego view와 pixel-wise로 가까운 RGB 단서와 camera trajectory 단서를 제공한다.
- 코드 매핑:
  - `EgoX-EgoPriorRenderer/scripts/render_vipe_pointcloud.py`
    - `load_camera_params_from_meta()`가 `meta.json`에서 exo/ego intrinsics/extrinsics를 읽는다.
    - `build_background_pointcloud()`가 depth artifact와 RGB를 point cloud로 역투영하고 world coordinate로 변환한다.
    - `build_mean_background_pointcloud()`가 `--use_mean_bg`일 때 static background를 nanmean 방식으로 구성한다.
    - `build_dynamic_points_for_frame()`가 instance mask를 사용해 dynamic object point cloud를 별도로 구성한다.
    - `render_points_pytorch3d()`와 `render_points_fisheye()`가 target ego camera view에서 point cloud를 렌더링한다.
    - main rendering loop가 ego extrinsics를 따라 frame별 image를 만들고 `ego_Prior.mp4`로 저장한다.
  - `EgoX-EgoPriorRenderer/scripts/render_vipe.sh`
    - `--input_dir`, `--meta_json_path`, `--point_size`, `--fish_eye_rendering`, `--use_mean_bg`를 지정하는 wrapper다.
- 구현 설명:
  - 논문의 `render(X, D_f, phi)`는 코드에서 “ViPE artifacts + meta camera parameters + PyTorch3D point rendering”으로 구현된다.
  - in-the-wild 예제에서는 `--fish_eye_rendering`이 Aria 계열 fisheye ego view를 흉내 내는 데 사용된다.
- 주석 후보:
  - `render_vipe_pointcloud.py`의 point cloud construction 함수들에 “논문 Sec. 3.1의 exo frame lifting 단계”를 붙인다.
  - frame별 ego rendering loop에 “Eq. (2)의 `P = render(X, D_f, phi)` 구현”을 붙인다.

### 2.4 Depth Alignment

- 논문 위치: Sec. 3.1, Eq. (1), Fig. 9
- 개념: monocular depth `D_m`과 video depth `D_v`의 장점을 결합해 temporally aligned depth `D_f`를 만든다. 논문은 affine parameter `alpha`, `beta`를 momentum update로 추정해 temporal inconsistency를 줄인다.
- 코드 매핑:
  - `EgoX-EgoPriorRenderer/vipe/pipeline/default.py`
    - ViPE annotation pipeline에서 intrinsics, mask, SLAM, depth processing, artifact 저장 흐름을 담당한다.
  - `EgoX-EgoPriorRenderer/vipe/pipeline/processors.py`
    - `AdaptiveDepthProcessor`가 depth estimation/alignment 계열 processor로 연결된다.
  - `EgoX-EgoPriorRenderer/configs/pipeline/lyra.yaml`, `lyra_no_vda.yaml`, `metric_vda.yaml`, `no_vda.yaml`
    - ViPE pipeline configuration으로 depth model 및 temporal depth consistency 관련 설정을 고른다.
  - `EgoX-EgoPriorRenderer/scripts/convert_depth_zip_to_npy.py`
    - ViPE depth artifact `.zip` 안의 `.exr` depth를 EgoX 본체가 읽는 `.npy` depth map으로 변환한다.
- 구현 설명:
  - 논문 Eq. (1)의 affine alignment 수식이 EgoX 루트 코드에 직접 명시되어 있지는 않다.
  - 현재 저장소에서는 ViPE pipeline과 depth processor/artifact를 통해 aligned depth가 생성되어 renderer와 GGA 계산에 쓰이는 구조로 보인다.
  - EgoX 본체의 `infer.py`와 `wan_dataset.py`는 이미 준비된 `.npy` depth map을 읽어 GGA용 3D vector를 계산한다.
- 주석 후보:
  - `convert_depth_zip_to_npy.py`에는 “ViPE depth artifact를 EgoX GGA/renderer 입력 형식으로 변환”이라고 표시한다.
  - `infer.py`의 depth map 로드 부분에는 “Sec. 3.1에서 얻은 aligned depth를 사용한다고 가정”이라고 표시한다.

### 2.5 Dynamic Object Masking

- 논문 위치: Sec. 3.1
- 개념: depth alignment와 point cloud rendering에서 dynamic object를 mask out하여 static background 중심으로 geometry를 안정화한다.
- 코드 매핑:
  - `EgoX-EgoPriorRenderer/scripts/render_vipe_pointcloud.py`
    - `build_background_pointcloud()`는 `instance_mask == 0`인 static/background point만 유지한다.
    - `build_mean_background_pointcloud()`는 static mask와 reliable depth mask를 결합해 평균 background point cloud를 만든다.
    - `build_dynamic_points_for_frame()`는 `instance_mask != 0`인 dynamic point를 별도로 만들 수 있다.
  - `EgoX-EgoPriorRenderer/vipe/pipeline/processors.py`
    - `TrackAnythingProcessor`가 instance/object mask 생성 흐름에 대응된다.
- 구현 설명:
  - renderer는 static background prior를 안정적으로 렌더링하기 위해 background-only 또는 mean background 옵션을 제공한다.
  - dynamic point는 별도 함수가 있지만, 논문 설명의 핵심은 dynamic object가 depth/geometry prior를 오염시키지 않도록 분리하거나 제거하는 것이다.
- 주석 후보:
  - background mask 적용 조건 옆에 “Sec. 3.1 dynamic object masking”을 표시한다.

### 2.6 Ego Camera Pose와 In-the-Wild 수동 annotation

- 논문 위치: Sec. 3, Supplementary Fig. 8, Conclusion/Future Work
- 개념: EgoExo4D에서는 GT ego camera pose를 사용하고, in-the-wild 예제에서는 3D viewer에서 ego camera extrinsic을 수동 지정한다.
- 코드 매핑:
  - `example/in_the_wild/meta.json`, `example/egoexo4D/meta.json`
    - `ego_intrinsics`, `ego_extrinsics`가 target ego camera trajectory다.
  - `EgoX-EgoPriorRenderer/vipe/utils/viser.py`
    - `ego_manual` mode, manual ego frustum, transform controls, extrinsic matrix 표시 기능이 있다.
  - `EgoX-EgoPriorRenderer/README.md`
    - `vipe visualize ... --ego_manual` workflow를 설명한다.
  - `infer.py`
    - `ego_extrinsics`를 읽어 GGA용 ego camera ray와 camera origin을 계산한다.
  - `render_vipe_pointcloud.py`
    - `ego_extrinsics`를 따라 ego prior frame을 렌더링한다.
- 구현 설명:
  - 논문의 `phi`는 코드에서 `ego_extrinsics`와 `ego_intrinsics`로 표현된다.
  - 자동 ego pose estimation은 현재 구현의 핵심 경로가 아니며, 논문에서도 future work로 언급된다.
- 주석 후보:
  - `infer.py`에서 `ego_extrinsic`을 4x4로 확장하는 부분에 “논문 target ego camera pose `phi`”를 표시한다.

### 2.7 Unified Conditioning Strategy

- 논문 위치: Sec. 3.2, Fig. 3, Supplementary conditioning ablation
- 개념:
  - exocentric latent `x_0`는 target ego latent와 pixel alignment가 없으므로 width-wise concatenation으로 붙인다.
  - egocentric prior latent `p_0`는 ego target과 같은 viewpoint이므로 channel-wise condition으로 제공한다.
  - mask `m`은 condition 영역과 synthesis 영역을 구분한다.
- 코드 매핑:
  - `core/finetune/models/wan_i2v/sft_trainer.py`
    - `WanWidthConcatImageToVideoPipeline.prepare_latents()`
      - latent width를 exo width와 ego width로 나눈다.
      - `_shared_latent_exo_randn`의 exo 영역은 exo latent로 유지하고, ego 영역은 `ego_prior_latents`로 교체해 condition tensor를 만든다.
      - `mask_lat_size`와 `latent_condition`을 channel dimension으로 concat한다.
    - `WanWidthConcatImageToVideoPipeline._setup_ic_lora_latent()`
      - exo video를 VAE encode하고 random ego latent와 width 방향으로 이어 붙인다.
    - `WanWidthConcatImageToVideoPipeline.__call__()`
      - exo video와 ego prior video를 각각 전처리하고 `prepare_latents()`로 전달한다.
  - `core/finetune/datasets/wan_dataset.py`
    - `encoded_exo_ego_gt_video = torch.cat([encoded_exo_video, encoded_ego_gt_video], dim=-1)`
    - `encoded_exo_ego_prior_video = torch.cat([encoded_exo_video, encoded_ego_prior_video], dim=-1)`
    - 학습 캐시에서 exo+ego GT와 exo+ego prior를 width 방향으로 저장한다.
- 구현 설명:
  - 코드상 exo와 ego 영역은 latent width dimension에서 좌우로 붙는다. 기본 pixel resolution은 exo `784x448`, ego `448x448`, total `1232x448`이다.
  - ego prior 자체도 width상 ego 영역에 들어가지만, transformer 입력에서는 `torch.cat([latents, condition], dim=1)`로 noisy latent와 condition이 channel dimension에서 결합된다. 따라서 논문의 “ego prior channel-wise conditioning”은 이 condition channel 결합과 연결된다.
- 주석 후보:
  - `prepare_latents()`의 exo/ego width 계산 부분에 “Fig. 3의 width-wise exo concat”을 표시한다.
  - `torch.concat([mask_lat_size, latent_condition], dim=1)`에 “mask와 clean condition latent를 channel-wise로 붙이는 inpainting condition”이라고 표시한다.

### 2.8 Clean Exocentric Latent Representation

- 논문 위치: Sec. 3.2, Eq. (3), Ablation
- 개념: denoising 중 exo latent `x_0`를 noisy 상태로 만들지 않고 clean condition으로 유지한다. ego 영역만 생성 대상이다.
- 코드 매핑:
  - `core/finetune/models/wan_i2v/sft_trainer.py`
    - 학습 `compute_loss()`:
      - `latent = encoded_exo_ego_gt_video`
      - `latent_condition = encoded_exo_ego_prior_video`
      - noise를 추가한 뒤 exo 영역은 원본 latent로 되돌린다.
      - loss는 `ego_predicted_noise`와 `ego_target`에 대해서만 계산한다.
    - 추론 denoising loop:
      - transformer output 이후 scheduler step에서 ego 영역만 생성 결과로 유지되도록 처리한다.
  - `core/finetune/datasets/wan_dataset.py`
    - exo latent가 VAE encode된 clean latent로 캐시된다.
- 구현 설명:
  - clean latent 개념은 “exo 영역에 noise를 넣지 않거나, 넣더라도 원본으로 overwrite하는 방식”으로 구현되어 있다.
  - 이는 논문 Eq. (3)의 `x_0`가 denoising function의 조건으로 계속 유지되는 것과 대응된다.
- 주석 후보:
  - `compute_loss()`에서 `noisy_latents`의 exo 영역을 원본으로 되돌리는 줄에 “clean exocentric latent”를 표시한다.
  - ego-only loss 계산 부분에 “sampling 후 exo region은 버리고 ego region만 사용한다는 논문 설명과 대응”을 표시한다.

### 2.9 Denoising Relation과 Mask

- 논문 위치: Sec. 3.2, Eq. (3)
- 개념: `f_theta(x_0, z_t | x_0, p_0 | m_1, m_0)` 형태로 clean exo latent, noisy target latent, ego prior, binary mask를 함께 사용한다.
- 코드 매핑:
  - `core/finetune/models/wan_i2v/sft_trainer.py`
    - `mask_lat_size`가 exo 영역과 ego 영역, temporal visible frame을 구분한다.
    - `condition = torch.concat([mask_lat_size, latent_condition], dim=1)`가 inpainting-style condition이다.
    - `latent_model_input = torch.cat([noisy_latents, condition], dim=1)`가 transformer 입력이다.
  - `core/finetune/models/wan_i2v/custom_transformer.py`
    - `WanTransformer3DModel_GGA.forward()`의 `hidden_states`는 noisy latent와 condition이 channel-wise로 붙은 tensor다.
- 구현 설명:
  - 코드의 mask는 spatial exo/ego 구분뿐 아니라 Wan VAE temporal downsampling에 맞춘 first-frame mask 처리를 포함한다.
  - `actual_num_frames`, `vae_scale_factor_temporal`, `mask_lat_size.view(...).transpose(...)`는 Wan VAE의 latent frame 구조에 맞추는 부분이다.
- 주석 후보:
  - mask 생성 블록에 “Eq. (3)의 `m_1`, `m_0`에 해당하는 condition mask”라고 표시한다.

### 2.10 Wan2.1 I2V Video Diffusion Model

- 논문 위치: Sec. 3.2, Sec. 4.1
- 개념: 대규모 pretrained video diffusion model의 spatio-temporal prior를 활용한다. 논문은 Wan2.1 I2V 14B inpainting variant를 기반으로 LoRA fine-tuning한다고 설명한다.
- 코드 매핑:
  - `core/finetune/models/wan_i2v/sft_trainer.py`
    - `WanWidthConcatImageToVideoPipeline`은 diffusers `WanImageToVideoPipeline`을 상속한다.
    - Wan VAE, text encoder, CLIP image encoder, transformer, scheduler를 사용한다.
  - `core/finetune/models/wan_i2v/custom_transformer.py`
    - diffusers Wan transformer 구조를 기반으로 GGA 인자를 추가한 `WanTransformer3DModel_GGA`를 정의한다.
  - `README.md`, `scripts/infer_*.sh`, `scripts/finetune.sh`
    - `Wan2.1-I2V-14B-480P-Diffusers` pretrained model 경로와 EgoX LoRA weight 사용법을 제공한다.
- 구현 설명:
  - pretrained model은 `model_path`로 주입된다.
  - EgoX의 custom transformer는 Wan checkpoint의 transformer subfolder에서 로드된다.
- 주석 후보:
  - `infer.py`에서 `WanTransformer3DModel_GGA.from_pretrained()`를 호출하는 부분에 “pretrained Wan transformer에 GGA-capable class를 씌우는 단계”라고 표시한다.

### 2.11 LoRA Adaptation

- 논문 위치: Abstract, Sec. 3.2, Sec. 4.1
- 개념: pretrained VDM을 크게 바꾸지 않고 lightweight LoRA adaptation으로 exo-to-ego task에 맞춘다.
- 코드 매핑:
  - `core/finetune/trainer.py`
    - `prepare_trainable_parameters()`에서 `training_type == "lora"`일 때 transformer를 freeze하고 `LoraConfig` adapter를 추가한다.
    - `target_modules` 기본값은 `to_q`, `to_k`, `to_v`, `to_out.0`이다.
    - 저장/로드 hook에서 LoRA weights만 저장/로드한다.
  - `core/finetune/models/wan_i2v/lora_trainer.py`
    - `WanI2VLoraTrainer`는 SFT trainer를 상속하고 registry에 `wan-i2v`, `lora`로 등록한다.
  - `core/finetune/schemas/args.py`
    - `rank`, `lora_alpha`, `target_modules`, `training_type` CLI를 정의한다.
  - `scripts/finetune.sh`
    - 실제 실행값으로 `rank=256`, `lora_alpha=256`, `train_resolution=49x448x1232` 등을 사용한다.
  - `infer.py`
    - `pipe.load_lora_weights()`와 `pipe.fuse_lora()`로 추론 시 LoRA를 fuse한다.
- 구현 설명:
  - LoRA가 붙는 대상은 transformer attention projection 계열이다.
  - 논문에서 rank 256을 사용했다고 보고한 설정은 `scripts/finetune.sh`에 반영되어 있다.
- 주석 후보:
  - `trainer.py`의 LoRA adapter 추가 블록에 “논문의 lightweight LoRA adaptation”을 표시한다.

### 2.12 Geometry-Guided Self-Attention, GGA

- 논문 위치: Sec. 3.3, Fig. 4, Eq. (4)-(7), Fig. 7, Fig. 10
- 개념: ego query token과 exo key token 사이에 3D geometric alignment bias를 추가한다. 방향 벡터 cosine similarity를 `log(cos + 1)` 형태의 attention bias로 사용해, view-relevant exo region에 더 집중하게 한다.
- 코드 매핑:
  - `infer.py`
    - `--use_GGA`일 때 depth map, exo camera intrinsics/extrinsics, ego intrinsics/extrinsics를 읽는다.
    - exo point map을 world coordinate로 변환한다.
    - ego camera origin에서 exo point로 향하는 `point_vecs`와 ego ray `cam_rays`를 만든다.
    - `attn_maps = torch.cat((point_vecs, cam_rays), dim=2)`와 `attn_masks`를 생성해 `generate_video()`에 전달한다.
  - `core/finetune/datasets/wan_dataset.py`
    - 학습 시에도 depth/camera parameter에서 `attn_maps`, `attn_masks`, `point_vecs_per_frame`, `cam_rays`를 만들고 safetensors cache로 저장한다.
  - `core/finetune/models/wan_i2v/custom_transformer.py`
    - `WanTransformer3DModel_GGA.forward()`가 `attention_GGA`, `attention_mask_GGA`, `point_vecs_per_frame`, `cam_rays`, `cos_sim_scaling_factor`를 받는다.
    - GGA vector를 patch scale로 pooling하고 flatten한다.
    - `cos_sim` matrix를 계산하고 scaling/clamp/mask를 적용한다.
    - `WanAttnProcessor2_0.__call__()`에서 `cos_sim`을 attention mask bias로 적용한다.
- 구현 설명:
  - 논문의 `hat q`, `hat k` 방향 벡터는 코드에서 `cam_rays`, `point_vecs`, `point_vecs_per_frame`, `attention_GGA`로 표현된다.
  - `cos_sim + 1`, `torch.log(cos_sim + 1e-6)`는 논문 Eq. (4)-(7)의 geometric prior를 attention logit에 더하는 구현이다.
  - `cos_sim_scaling_factor`는 논문의 geometric bias strength `lambda_g`에 해당하는 하이퍼파라미터로 볼 수 있다.
  - 구현상 `WanAttnProcessor2_0`의 GGA bias 경로는 `cos_sim is not None and not do_kv_cache`일 때만 실행된다. `do_kv_cache=True`이면 현재 코드에서는 일반 attention 경로로 우회된다.
- 주석 후보:
  - `infer.py`의 `attn_maps`, `attn_masks` 생성 부분에 “GGA precomputation” 주석.
  - `custom_transformer.py`의 `cos_sim` 생성 block에 “Eq. (4)-(7) geometry prior” 주석.
  - `WanAttnProcessor2_0.__call__()`의 `attn_mask_from_cos_sim`에 “attention logit bias로 주입” 주석.

### 2.13 Latent-Space GGA Downsampling

- 논문 위치: Supplementary F.1
- 개념: diffusion model은 latent token/patch 공간에서 동작하므로 pixel-level 3D direction vector를 VAE/patch scale로 downsample해 attention token과 맞춘다.
- 코드 매핑:
  - `core/finetune/models/wan_i2v/custom_transformer.py`
    - `F.avg_pool3d(..., kernel_size=(1, 2, 2), stride=(1, 2, 2))`로 GGA vector를 transformer patch resolution에 맞춘다.
    - `attention_GGA.flatten(2).transpose(1, 2)`로 token sequence와 같은 순서로 펼친다.
    - `point_vecs_per_frame.flatten(2).transpose(1,2)`도 같은 방식으로 tokenized vector가 된다.
- 구현 설명:
  - `patch_embedding`이 `Conv3d` patch stride로 hidden token을 만들기 때문에 GGA vector도 같은 spatial downsampling을 거친다.
  - 코드 주석에 적힌 shape는 `F, H, W, C`와 `B, C, F, H, W` 간 변환을 추적하는 데 중요하다.
- 주석 후보:
  - pooling 직전에 “pixel-space GGA vector를 Wan transformer token scale로 맞추는 단계”를 표시한다.

### 2.14 Text Prompt Generation

- 논문 위치: Supplementary prompt section, Tab. 6, Fig. 18
- 개념: pretrained diffusion model의 text conditioning을 위해 `[Exo view]`, `[Ego view]` 블록을 포함한 상세 prompt를 생성한다.
- 코드 매핑:
  - `caption.py`
    - exo/ego prior 또는 GT frame을 sampling하고 GPT-4o API를 통해 scene/action prompt를 생성한다.
    - `meta.json`의 `prompt` 필드를 채우는 흐름과 연결된다.
  - `core/finetune/datasets/wan_dataset.py`
    - `prompt`를 hash key로 삼아 text embedding을 `data_root/cache/prompt_embeddings`에 저장한다.
  - `core/finetune/models/wan_i2v/sft_trainer.py`
    - `encode_prompt()` 또는 `encode_text()` 계열로 Wan text encoder embedding을 만든다.
  - `example/*/meta.json`
    - 실제 prompt 예시가 저장되어 있다.
- 구현 설명:
  - 논문에서는 GPT-4o로 prompt를 만든다고 설명하며, 코드의 `caption.py`가 이 전처리 역할에 대응된다.
  - 추론 시에는 이미 meta에 저장된 prompt를 읽는 방식이다.
- 주석 후보:
  - `caption.py`의 prompt 생성 함수에 “논문 prompt generation protocol” 주석.
  - dataset의 prompt embedding cache 부분에는 “VDM text conditioning” 주석.

### 2.15 학습 Dataset과 Cache

- 논문 위치: Sec. 4.1 dataset setup
- 개념: Ego-Exo4D clips를 사용해 exo video, ego GT video, ego prior video, text prompt, camera/depth 정보를 학습 sample로 구성한다.
- 코드 매핑:
  - `core/finetune/datasets/wan_dataset.py`
    - `BaseWanDataset`가 `meta_data_file`을 읽고 `exo_video_path`, `ego_video_path`, `ego_prior_path`, `take_name`, `vipe_results_path`, `best_camera`, camera parameter를 참조한다.
    - video latent, prompt embedding, image embedding, attention map을 cache한다.
  - `meta_init.py`
    - `videos/<take>/exo.mp4`를 스캔해 초기 meta를 만든다.
  - `EgoX-EgoPriorRenderer/data_preprocess/scripts/infer_vipe_all_takes.sh`
    - Ego-Exo4D 여러 take/camera에 대해 ViPE inference/render를 batch 실행하고 best view를 선택하는 전처리 흐름이다.
- 구현 설명:
  - 추론 예제 meta는 `test_datasets[]` 배열 구조다.
  - 학습 dataset 코드는 `meta_data.items()` 형태와 `exo_video_path`, `ego_video_path` 키를 기대한다. 따라서 실제 학습용 meta 생성 경로는 추론 예제 meta와 형식이 다르다.
  - 이 차이는 문서/주석에서 명확히 남겨야 한다.
- 주석 후보:
  - dataset init에서 “training meta schema와 inference meta schema가 다르다”는 주의 주석.

### 2.16 Training Loss

- 논문 위치: Sec. 3.2, Sec. 4.1
- 개념: exo+ego target latent 중 ego 영역만 생성 대상으로 두고 diffusion noise prediction loss를 계산한다.
- 코드 매핑:
  - `core/finetune/models/wan_i2v/sft_trainer.py`
    - `compute_loss()`가 batch에서 `encoded_exo_ego_gt_video`를 target latent로, `encoded_exo_ego_prior_video`를 condition latent로 읽는다.
    - random timestep과 sigma를 샘플링한다.
    - `noise = torch.randn_like(latent)`와 flow matching target `target = noise - latent`를 만든다.
    - exo 영역은 clean latent로 유지한다.
    - `ego_predicted_noise = predicted_noise[..., -ego_latent_width:]`와 `ego_target = target[..., -ego_latent_width:]`로 ego 영역 loss만 계산한다.
- 구현 설명:
  - 논문에서는 exo latent가 conditioning 역할이고 최종 output은 ego view다. 코드의 loss crop이 이 구조를 직접 반영한다.
  - exo 영역에도 transformer output이 존재하지만, 학습 objective는 ego 영역에 집중된다.
- 주석 후보:
  - ego crop loss 라인에 “only synthesize/evaluate target ego latent region” 주석.

### 2.17 Inference Pipeline

- 논문 위치: Fig. 3 generation stage
- 개념: 준비된 exo video, ego prior, text prompt, GGA geometry를 사용해 denoising을 수행하고 ego view video를 저장한다.
- 코드 매핑:
  - `scripts/infer_itw.sh`, `scripts/infer_ego4d.sh`
    - `infer.py`를 호출하는 예제 wrapper다.
    - `--use_GGA`, `--cos_sim_scaling_factor`, `--lora_path`, `--model_path`, `--meta_data_file` 등을 설정한다.
  - `infer.py`
    - model/transformer/image encoder/pipeline을 로드한다.
    - LoRA weight가 있으면 load/fuse한다.
    - 각 meta sample에 대해 GGA tensor를 optional로 계산한다.
  - `core/inference/wan.py`
    - diffusers `load_video()`로 exo/ego prior를 로드하고 pipeline을 호출한다.
    - `imageio.mimsave()`로 결과 MP4를 저장한다.
- 구현 설명:
  - inference output은 `results` 또는 `args.out` 폴더에 sample take name으로 저장된다.
  - exo input은 `784x448`, ego prior는 `448x448`에 대응되고 total generation canvas width는 `1232`다.
- 주석 후보:
  - `infer.py` model load block에 “Wan pretrained + EgoX LoRA assembly” 주석.
  - `core/inference/wan.py` width 계산 부분에 “exo canvas + ego target canvas” 주석.

### 2.18 Ego Prior Renderer 전처리 CLI

- 논문 위치: Sec. 3.1, Supplementary in-the-wild camera annotation
- 개념: exo video에서 depth/camera/mask artifact를 만들고, ego trajectory에서 prior view를 렌더링한다.
- 코드 매핑:
  - `EgoX-EgoPriorRenderer/scripts/infer_vipe.sh`
    - `vipe infer`를 호출한다.
    - `--assume_fixed_camera_pose`는 EgoX 학습 조건인 fixed exo camera assumption과 연결된다.
    - `--pipeline lyra`는 README에서 EgoX에 사용했다고 설명된 pipeline이다.
  - `EgoX-EgoPriorRenderer/scripts/render_vipe.sh`
    - ViPE result와 meta를 받아 `ego_Prior.mp4`를 생성한다.
  - `EgoX-EgoPriorRenderer/scripts/convert_depth_zip_to_npy.py`
    - ViPE depth zip을 EgoX 본체의 `depth_maps/<take>/*.npy` 구조로 변환한다.
  - `EgoX-EgoPriorRenderer/data_preprocess/scripts/infer_vipe_all_takes.sh`
    - Ego-Exo4D 학습/평가용 batch preprocessing을 수행한다.
- 구현 설명:
  - `--assume_fixed_camera_pose`는 README의 제약과 논문 dataset assumption을 코드 실행 옵션으로 반영한다.
  - `--use_mean_bg`는 static scene prior를 더 안정적으로 렌더링하기 위한 옵션이다.
- 주석 후보:
  - shell script에는 길게 주석을 달기보다 README/문서 링크 수준이면 충분하다.

### 2.19 Evaluation Metrics와 Baseline

- 논문 위치: Sec. 4, Table 1, Table 2, Supplementary evaluation sections
- 개념: PSNR, SSIM, LPIPS, CLIP-I, SAM2+DINOv3 object metrics, FVD, VBench, user study, baseline 비교, ablation.
- 코드 매핑:
  - 현재 확인한 저장소 루트에는 논문 평가 metric을 실행하는 독립 평가 스크립트가 명확히 보이지 않는다.
  - `example/*`에는 qualitative inference용 sample data가 있다.
  - ablation 개념은 코드 옵션/구조로 일부 대응된다.
    - GGA on/off: `infer.py --use_GGA`
    - GGA strength: `--cos_sim_scaling_factor`
    - LoRA rank: `--rank`, `--lora_rank`
    - ego prior 입력: pipeline이 필수 입력으로 요구하므로 완전 제거 ablation은 별도 코드 수정이나 다른 branch가 필요하다.
- 구현 설명:
  - 논문 실험/평가 결과 자체는 코드 주석 대상이라기보다 연구 결과 설명에 해당한다.
  - 다음 단계 주석에서는 평가 metric보다 학습/추론/GGA 구현에 집중하는 것이 적절하다.
- 주석 후보:
  - `--use_GGA` 인자 설명에 “GGA ablation switch로도 사용 가능” 정도만 추가한다.

## 3. 핵심 데이터 구조 매핑

### 3.1 추론용 meta schema

주요 위치:

- `example/in_the_wild/meta.json`
- `example/egoexo4D/meta.json`
- `infer.py`
- `render_vipe_pointcloud.py`

주요 필드:

| 필드 | 논문 개념 | 사용 위치 |
| --- | --- | --- |
| `exo_path` | exocentric input video `X` | `infer.py`, `render_vipe_pointcloud.py` |
| `ego_prior_path` | egocentric prior video `P` | `infer.py`, `core/inference/wan.py` |
| `ego_gt_path` | target egocentric GT `Y` | 예제/평가/학습 준비 맥락 |
| `prompt` | VDM text conditioning | `infer.py`, `wan_dataset.py`, `sft_trainer.py` |
| `camera_intrinsics` | exo camera intrinsics | `infer.py`, `render_vipe_pointcloud.py` |
| `camera_extrinsics` | exo camera extrinsics | `infer.py`, `render_vipe_pointcloud.py` |
| `ego_intrinsics` | target ego camera intrinsics | `infer.py`, `render_vipe_pointcloud.py` |
| `ego_extrinsics` | target ego trajectory `phi` | `infer.py`, `render_vipe_pointcloud.py` |

### 3.2 학습용 dataset output

주요 위치: `core/finetune/datasets/wan_dataset.py`

| Dataset return key | 논문 개념 | 설명 |
| --- | --- | --- |
| `prompt_embedding` | text condition | Wan text encoder embedding cache |
| `encoded_exo_ego_gt_video` | `x_0` + target ego latent | exo clean latent와 ego GT latent를 width-wise concat |
| `encoded_exo_ego_prior_video` | `x_0` + `p_0` condition | exo clean latent와 ego prior latent를 width-wise concat |
| `image_embedding` | Wan I2V image condition | exo+ego GT first frame 또는 decoded latent 기반 CLIP image embedding |
| `attention_GGA` | GGA query/key direction map | geometry-guided attention에 들어갈 direction vector |
| `attention_mask_GGA` | exo/ego token mask | GGA 적용 대상 token 구분 |
| `point_vecs_per_frame` | exo point direction vectors | ego origin에서 exo point로 향하는 frame-pair vector |
| `cam_rays` | ego camera rays | ego query token 방향 vector |

### 3.3 Transformer 입력 tensor

주요 위치:

- `core/finetune/models/wan_i2v/sft_trainer.py`
- `core/finetune/models/wan_i2v/custom_transformer.py`

개념적 형태:

```text
latent_model_input = concat_channel(
    noisy_or_clean_latents,        # exo 영역은 clean, ego 영역은 denoising 대상
    mask_lat_size,                 # condition mask
    latent_condition               # exo clean + ego prior condition latent
)
```

공간 방향:

```text
width axis: [ exo latent region | ego latent region ]
```

중요한 하드코딩/가정:

- frame count: `49` input frames, Wan latent frame count는 보통 `13`.
- exo pixel size: `448x784`.
- ego pixel size: `448x448`.
- total canvas: `448x1232`.
- VAE spatial scale: 코드에서 주로 `1/8` 또는 latent width `784/8`, `448/8`로 계산된다.

## 4. 다음 단계 코드 주석 계획

다음 단계에서는 모든 파일에 과도하게 주석을 넣기보다, 논문 개념과 구현 대응이 모호한 지점에만 짧은 연구 개념 주석을 추가하는 것이 좋다.

| 우선순위 | 파일 | 주석 대상 | 주석 내용 |
| --- | --- | --- | --- |
| 1 | `core/finetune/models/wan_i2v/custom_transformer.py` | `cos_sim` 생성, `attn_mask_from_cos_sim`, GGA branch | Sec. 3.3 / Eq. (4)-(7)의 Geometry-Guided Self-Attention |
| 1 | `core/finetune/models/wan_i2v/sft_trainer.py` | `prepare_latents()`, `_setup_ic_lora_latent()`, `compute_loss()` | Sec. 3.2의 unified conditioning, clean latent, ego-only denoising loss |
| 1 | `infer.py` | `--use_GGA` geometry tensor 생성 block | 추론 시 GGA precomputation과 ego/exo direction vector 생성 |
| 2 | `core/finetune/datasets/wan_dataset.py` | latent cache와 attention map cache 생성 | 학습 시 exo/ego prior/ego GT latent 구성과 GGA cache |
| 2 | `EgoX-EgoPriorRenderer/scripts/render_vipe_pointcloud.py` | point cloud 생성과 ego rendering loop | Sec. 3.1의 ego prior `P = render(X, D_f, phi)` |
| 3 | `core/finetune/trainer.py` | LoRA adapter 추가 block | 논문의 lightweight LoRA adaptation |
| 3 | `caption.py` | prompt 생성 flow | supplementary prompt generation |

주석 작성 원칙:

- “무엇을 하는 코드인가”보다 “논문의 어떤 개념에 해당하는가”를 짧게 남긴다.
- shape가 어려운 부분에는 exo/ego width split 또는 token scale 변환을 함께 적는다.
- 이미 자명한 import, argparse, 단순 파일 I/O에는 주석을 추가하지 않는다.
- 현재 구현상 주의점인 `do_kv_cache=True`일 때 GGA bias branch가 우회되는 조건은 반드시 명시한다.

## 5. 코드에 직접 매핑되지 않거나 간접적인 논문 개념

| 논문 개념 | 현재 코드 상태 | 문서화 판단 |
| --- | --- | --- |
| PSNR/SSIM/LPIPS/CLIP-I/FVD/VBench 평가 | 명확한 독립 평가 스크립트 미확인 | 코드 주석 대상 아님. 논문 실험 결과로만 문서에 기록 |
| SAM2+DINOv3 object metric | 명확한 평가 구현 미확인 | 코드 주석 대상 아님 |
| User study | 코드 구현 대상 아님 | 문서에서 실험 절차로만 언급 |
| Baseline training/evaluation | 현재 저장소에는 EgoX 중심 코드만 확인 | 주석 대상 아님 |
| Automatic head pose estimation | 논문 future work 성격 | 코드 주석 대상 아님 |
| Eq. (1)의 affine depth alignment 수식 | ViPE pipeline 내부 처리로 간접 대응, EgoX 루트에 직접 수식 구현 미확인 | depth artifact 사용부에만 간접 주석 |

## 6. 빠른 참조: 논문 개념에서 코드로 찾기

- “exo video `X`를 어디서 읽는가?”
  - `infer.py`, `core/inference/wan.py`, `core/finetune/datasets/wan_dataset.py`
- “ego prior `P`는 어디서 만들어지는가?”
  - `EgoX-EgoPriorRenderer/scripts/render_vipe_pointcloud.py`
- “exo와 ego latent를 어디서 width-wise concat하는가?”
  - `core/finetune/datasets/wan_dataset.py`
  - `core/finetune/models/wan_i2v/sft_trainer.py`
- “ego prior를 diffusion condition으로 어디서 넣는가?”
  - `WanWidthConcatImageToVideoPipeline.prepare_latents()` in `core/finetune/models/wan_i2v/sft_trainer.py`
- “clean exocentric latent는 어디서 보장되는가?”
  - `compute_loss()` in `core/finetune/models/wan_i2v/sft_trainer.py`
- “GGA vector는 어디서 계산되는가?”
  - 추론: `infer.py`
  - 학습: `core/finetune/datasets/wan_dataset.py`
- “GGA attention bias는 어디서 들어가는가?”
  - `WanTransformer3DModel_GGA.forward()`와 `WanAttnProcessor2_0.__call__()` in `core/finetune/models/wan_i2v/custom_transformer.py`
- “LoRA는 어디서 붙는가?”
  - `core/finetune/trainer.py`
- “prompt는 어디서 만들어지고 쓰이는가?”
  - 생성: `caption.py`
  - 사용: `infer.py`, `core/finetune/datasets/wan_dataset.py`, `core/finetune/models/wan_i2v/sft_trainer.py`

