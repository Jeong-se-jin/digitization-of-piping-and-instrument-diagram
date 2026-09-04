# STA → 선 검출 → 그래프 브리지

P&ID 한 장을 **심볼·텍스트 검출 → 선 검출 → 그래프 구성 → 뷰어**까지 한 번에 돌립니다.
Azure 서비스는 쓰지 않습니다.

```
이미지 ──▶ STA (RF-DETR + PaddleOCR) ──▶ adapt ──▶ 선 검출 ──▶ 그래프 구성 ──▶ 뷰어
          심볼 + OCR                    스키마 변환   OpenCV      휴리스틱      HTML
```

---

## 1. 준비

### venv가 두 개인 이유

한 프로세스로 합칠 수 없습니다. STA는 pydantic 2(RF-DETR, PaddleOCR)를 요구하고,
이 레포는 pydantic 1(`BaseSettings`, v1 validator)을 요구합니다. 그래서 단계마다
맞는 인터프리터로 실행하고 파일로 결과를 넘깁니다.

| 단계 | 인터프리터 |
|---|---|
| STA (심볼 + OCR) | `/home/rx/project/STA-main/.venv-detect/bin/python` |
| 나머지 전부 | `./.venv-pid/bin/python` |

`run_all`이 단계마다 알아서 골라 씁니다.

### 이 레포에 들어 있는 것

```
tools/sta_bridge/
  PID_pipeline_.py     STA 파이프라인 (STA-main 에서 가져옴, 여기서 편집)
  dataset.yaml         심볼 클래스 32종
  model/               RF-DETR 체크포인트 (128 MB, git 제외)
```

`.venv-detect`만 아직 STA쪽에 있습니다.

---

## 2. 실행

```bash
cd /home/rx/project/digitization-of-piping-and-instrument-diagrams-main

.venv-pid/bin/python -m tools.sta_bridge.run_all \
  --image /home/rx/project/STA-main/samples/18.png \
  --name 18 \
  --ocr-cache cache/ocr_cache_18.json \
  --box-mask-inset 2 --hough-max-line-gap 2 --hough-min-line-length 4 \
  --drop-boxed-segments
```

캐시가 있으면 **약 60초**, 없으면 OCR을 돌리느라 **약 8분** 걸리고 그 경로에 캐시를 남깁니다.
같은 도면을 다시 돌릴 때는 캐시를 재사용합니다.

### STA를 다시 돌리지 않고 기존 결과 쓰기

```bash
.venv-pid/bin/python -m tools.sta_bridge.run_all \
  --image /home/rx/project/STA-main/samples/18.png --name 18 \
  --from-results /home/rx/project/STA-main/results/18 \
  --ocr-cache /home/rx/project/STA-main/results/ocr_cache_18.json
```

`associations.json`은 **태그를 받은 심볼만** 담고 있어서, 태그 없는 심볼이 빠집니다.
그 심볼들도 마스킹 대상이므로 정확도가 필요하면 `--from-results` 없이 돌리세요.

### 선 검출부터만 다시 (파라미터 실험용, 30초)

```bash
DEBUG=true .venv-pid/bin/python -m tools.sta_bridge.run_local \
  --text-detection out/18/text_detection.json \
  --image out/18/diagram.png \
  --output-dir out/실험이름 \
  --box-mask-inset 2 --hough-max-line-gap 2 --hough-min-line-length 4 \
  --drop-boxed-segments
```

---

## 3. 결과물 — `out/<이름>/`

| 파일 | 내용 |
|---|---|
| **`graph_viewer.html`** | 인터랙티브 뷰어. 자산을 클릭하면 그 연결만 격리해서 보여줌 |
| **`21_orphan_lines.png`** | 선분 상태. **파랑**=연결에 사용 / **빨강**=미사용 실선 / **보라**=미사용 점선 / **주황**=심볼·텍스트 박스 안에 갇힘 / 옅은 회색=검출 안 된 잉크 |
| **`20_connections.png`** | 자산 간 연결을 직선으로. 고립 자산은 빨간 박스 + 태그 |
| `graph_connectivity.json` | 최종 연결 데이터 (자산별 인접 리스트 + 경로 선분) |
| `line_detection.json` | 선분 전체. `line_type`(solid/dashed), `dash_px`, `inside_box` 포함 |
| `text_detection.json` | 심볼·텍스트 (정규화 좌표, 라벨 매핑 후) |
| `10_preprocessed.png` | **Hough 입력 이미지** — 마스킹·이진화 완료 |
| `11_before_thinning.png` | thinning 직전 |
| `12_line_segments.png` | 검출 선분 오버레이 |
| `13~15_*.png` | 레포 기본 그래프 디버그 이미지 |
| `02~04_*.png`, `associations.json`, `sta_export.json` | STA 자체 출력 |
| `diagram.png` | 좌표 기준 이미지 |

`graph_connectivity.json` 구조:

```json
{ "id": 73, "label": "...", "text_associated": "PUMP 1A",
  "bounding_box": { "topX": 0.048, ... },
  "connections": [
    { "id": 34, "text_associated": "TE 125A",
      "flow_direction": "unknown",
      "segments": [ {선분1}, {선분2}, ... ] } ] }
```

엣지는 **양쪽에 중복 저장**됩니다. 총 엣지 수를 셀 때는 2로 나누세요.
`segments`의 `topX/topY/bottomX/bottomY`는 사각형이 아니라 **선분의 양 끝점**입니다.

---

## 4. 자주 쓰는 옵션

| 옵션 | 뜻 |
|---|---|
| `--symbol-conf 0.3` | 심볼 검출 문턱. 기본은 `PID_pipeline_.py`의 상수(0.30) |
| `--symbol-scale auto` | 검출 입력 배율. `auto`가 이 도면에서 1.8을 추론 |
| `--cost-cap -500` | 심볼↔텍스트 연결 문턱 |
| `--symbol-mask-inset 3` | 심볼 마스크를 N px 안쪽으로 줄임 |
| `--box-mask-inset 2` | 패널 계기 박스 전용 여백 |
| `--text-mask-inset-x 4` / `-y 0` | 텍스트 마스크 여백. **가로/세로 독립** |
| `--hough-min-line-length 4` | 이보다 짧은 선분은 버림 |
| `--hough-max-line-gap 2` | 이 이하 틈은 이어서 한 선분으로 |
| `--drop-boxed-segments` | 심볼·텍스트 박스 안에 **완전히** 들어간 선분을 그래프에서 제외 |
| `--dedup-segments` | 같은 획을 중복 검출한 선분 제거 (기본 켜짐) |
| `--no-thinning` / `--thinning-iterations N` | thinning 끄기 / N회만 |
| `--thin-min-stroke-width W` | W px 이상 굵은 획만 thinning |
| `--classify-line-types` | 실선/점선 분류 (기본 켜짐) |
| `--all-symbols-as-assets` | 태그 없는 심볼도 그래프 노드로 (기본 켜짐) |
| `--exclude-dashed` | 점선을 그래프 구성에서 제외 |

---

## 5. 기본값 — `src/app/config.py`

플래그 없이 돌리면 이 값들이 적용됩니다.

```python
enable_thinning_preprocessing_line_detection = False   # thinning 끔
line_detection_deduplicate_segments          = True    # 중복 검출 제거
line_detection_symbol_mask_inset_pixels      = 3
line_detection_text_mask_inset_x_pixels      = 4
line_detection_text_mask_inset_y_pixels      = 0
classify_line_types                          = True
treat_all_symbols_as_assets                  = True
graph_ray_cast_to_symbol_pixels              = 60
```

**thinning 대신 중복 제거**를 쓰는 이유: thinning은 굵은 획이 양쪽 edge로 두 번
검출되는 걸 막지만, 동시에 대시의 **끝을 깎습니다**. 9px 대시가 7px이 되고 그 2px이
그대로 갭으로 들어가 점선 인식과 연결을 망칩니다. 검출 후에 겹치는 선분 중 긴 것만
남기면 대시는 온전하고 이중선도 사라집니다.

---

## 6. 알아둘 것

**좌표는 crop 여부에 종속됩니다.** `PID_pipeline_.crop_diagram()`은 7168×4562 시트에
맞춘 고정 슬라이스(`[160:-160, 290:-1500]`)라 더 작은 도면에서는 본체를 잘라냅니다.
`run_all`은 기본이 `--no-crop`이고, 이 도면에서는 그게 맞습니다.

**OCR 캐시는 crop 모드가 같아야 합니다.** 어긋나면 모든 박스가 엉뚱한 자리로 갑니다.
좌표 범위가 이미지를 벗어나면 즉시 에러를 냅니다.

**`box` 클래스는 주석 상자가 아니라 패널 계기입니다** (FICA 101B, TICA 121A 등).
`label_map.py`에서 `Instrument/Indicator/Panel box`로 매핑합니다. 이걸 주석으로 취급하면
46개 계기가 그래프에서 빠지고, 계기 버블과 잇는 점선이 붙을 상대를 잃습니다.

**`export.py`는 `PID_pipeline_.main()`을 복제합니다.** 파이프라인에 단계가 추가되면
여기도 같이 고쳐야 합니다. 과거에 `deduplicate_symbols`와 `infer_symbol_scale`을
빠뜨려 조용히 틀린 결과가 나온 적이 있습니다.

### 아직 안 되는 것

| | 이유 |
|---|---|
| 흐름 방향 (`flow_direction`이 대부분 `unknown`) | 전파의 출발점이 `Equipment/`나 `Piping/Endpoint/Pagination` 라벨인데, STA 32클래스에 장비도 off-page 커넥터도 없습니다 |
| 도면 밖으로 나가는 배관 | off-page 커넥터가 자산이 아니라 한쪽 끝이 비어 미사용으로 남습니다 |
| 긴 선분이 T-접합에서 밀림 | 선분마다 끝점 후보를 **하나만** 유지하고, 끝점끼리 붙은 후보가 이미 있으면 T-접합은 거리 비교 없이 차단됩니다 (`has_update_on_line_distance`). 선이 길수록 경쟁자가 많아 불리합니다 |
| 원 검출 | `HoughLinesP`는 직선만 찾습니다. 원자로 용기 같은 큰 원은 검출되지 않습니다 |

---

## 7. 모듈

| 파일 | 역할 |
|---|---|
| `run_all.py` | 전체 드라이버. 단계마다 venv를 골라 subprocess 실행 |
| `export.py` | STA 실행 → `diagram.png` + `sta_export.json` (+ STA 자체 오버레이) |
| `from_results.py` | STA 재실행 없이 기존 결과 폴더를 읽어 같은 형식으로 변환 |
| `adapt.py` | `sta_export.json` → 이 레포의 요청 스키마 (좌표 정규화 + 라벨 매핑) |
| `label_map.py` | STA 32클래스 → `Instrument/Valve/…` 계층 라벨 |
| `run_local.py` | 선 검출 + 그래프 구성 + 오버레이 생성 |
| `viewer.py` / `viewer_template.html` | 단일 HTML 뷰어 생성 (이미지·데이터 인라인) |
| `draw_connections.py` | `20_connections.png`, `21_orphan_lines.png` |
