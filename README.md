# FabricInspect

Few-shot, label-free fabric defect inspection for a seaming station.
The operator shows the camera a few seconds of **defect-free** fabric; FabricInspect learns
that fabric and then flags anything that does not look like it — no defect images, no
labelling, no GPU.

It is designed to sit next to the FabricScope ultrasonic seaming machine
(USB camera over the seam line) but works with any USB/UVC camera or image folder.

## Install

```
pip install -r requirements.txt
```

(Windows/Linux, Python 3.9+. CPU only. The first run of the deep detector downloads the
ImageNet ResNet-18 weights, ~45 MB.)

## Use

GUI (camera or image folder):

```
python -m fabricinspect gui
```

1. **Open camera** (index 0, 1, …) or **Image folder…**
2. Type the fabric name, choose the detector and the number of good shots (10 is a good default).
3. **Learn fabric** – slowly move defect-free fabric under the camera. The model and a
   label-free threshold (leave-one-out over the shots + 10 % margin) are computed and saved
   as a profile in `~/FabricInspect/profiles/<name>.npz`.
4. **Start inspection** – live OK / DEFECT verdict with heat-map; defect regions are outlined.
   Every frame is logged to `~/FabricInspect/logs/inspect_YYYYMMDD.csv`, defect frames are
   saved to `logs/defects/`.
5. Next time, pick the fabric from **Profile** and press **Load**.

Command line:

```
python -m fabricinspect learn   --name denim_blue --images good_frames/ [--detector gabor]
python -m fabricinspect inspect --name denim_blue --images to_check/ [--webhook http://...]
python -m fabricinspect profiles
```

Detectors

| key         | name            | what it is                                   | speed (CPU)  |
|-------------|-----------------|----------------------------------------------|--------------|
| `patchcore` | PatchCore-lite  | ResNet-18 patch features + nearest neighbour | see paper    |
| `gabor`     | Gabor-Gaussian  | Gabor energies + Lab colour, one Gaussian    | no deep libs |

### Machine integration (optional)

With `--webhook URL` (or the GUI field) every verdict is POSTed as JSON:

```json
{"fabric": "denim_blue", "defect": true, "score": 2.98, "threshold": 1.67, "defect_area_frac": 0.012}
```

The receiver (e.g. the seaming machine's ESP32 firmware) must accept this schema; the
FabricScope firmware's `/api/analysis_result` endpoint would need a matching handler.
Network errors never interrupt inspection. Off by default.

## IoT deployment (FabricInspect-IoT)

Three roles talk over MQTT (any broker; Mosquitto recommended):

```
mosquitto -p 1883                                   # or: python -m fabricinspect broker
python -m fabricinspect edge    --host <broker>     # plant edge server (PatchCore-lite)
python -m fabricinspect publish --host <broker> --name denim --images good_frames/
python -m fabricinspect station --host <broker> --name st01 --fabric denim --camera 0
```

`publish` learns both models from defect-free frames, stores the deep profile for the edge
server and publishes the ~7 kB station profile as a *retained* message
(`fi/profile/<fabric>/gabor`), so stations get it as soon as they subscribe. Each station
decides clear frames locally and offloads only frames whose score lies in
`[alpha*tau, beta*tau]` (default 0.6 / 1.5) to `fi/st/<id>/offload`; results come back on
`fi/st/<id>/result`, verdicts are published on `fi/st/<id>/verdict`.
`--policy local|edge` gives station-only / edge-only operation.

To measure the station model on a real device (Raspberry Pi, mini-PC, Termux):
`python experiments/device_bench.py` (needs only numpy, scipy, opencv).

## Reproducing the paper (IEEE IoT Journal manuscript in `paper/ieee/`)

Run on mains power with the performance power plan – timing results are invalid on battery.

```
python experiments/run_benchmark.py all     # detectors: few-shot, thresholds, ablation, robustness, latency
python experiments/run_iot.py cascade       # gate trade-off (offline)
python experiments/run_iot.py edgebench     # edge throughput mu(b), station cost
python experiments/run_iot.py live          # Mosquitto + edge + 1..32 stations, emulated links
python experiments/run_iot.py profile       # profile sizes and distribution time
python experiments/make_figures.py          # figures + LaTeX tables into paper/
python -m pytest tests                      # smoke tests (synthetic data, incl. IoT round trip)
```

Mosquitto is looked up in `tools/mosquitto/mosquitto.exe` (or `FI_MOSQUITTO`); without it the
bundled minimal MQTT 3.1.1 broker is used.

MVTec AD is CC BY-NC-SA 4.0 (non-commercial); download it from MVTec and place the
`carpet grid leather tile wood` folders in `data/mvtec/`.
