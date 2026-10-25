# local-gs-eval：大基线下局部前馈 GS 补全质量验证（实验 3）

本仓库用于回答一个问题：**在光流插视点失败的区域，前馈 3DGS（AnySplat）渲染的结果能不能比光流更好，并且好到值得做"光流为主 + 局部 GS 补全"的混合方案？**

采用留一法：从 9 路会聚相机中只取少数几台作为输入（例如 0/4/8 三台，对应之前 `北极星/2-3/魔方_9` 中魔方/头发出错的实验），被抽掉的真实相机作为真值，分别比较：

- **光流**：Practical-RIFE，用两侧最近的输入相机合成目标相机，逐时刻真实推理（不做 t=0.5 线性外推）；
- **局部 GS**：AnySplat 只用输入相机前馈生成高斯，在目标相机位姿处渲染；
- **混合**：光流结果为底，在指定区域内换成 GS，并羽化边界。

全部比较都在单应对齐后的图像域（即现有光流管线的输入/输出域）中进行。

> 数据说明：魔方场景的实际数据是 **9 路**相机（`北极星/0/魔方_9/grid_00~08` 原图、`北极星/1/魔方_9/aligned_grid_00~08` 对齐图），`北极星/0-3` 的三路就是第 0、4、8 号。相机数量和输入组合都在配置里设置。

---

## ★ 新主机（RTX 4090，Linux）从零部署并运行六路数据集实验

本节是完整的操作清单，可以交给自动执行的 AI 按顺序完成。每一步都有检查方法；不满足时停下来报告，不要跳过。

### 第 1 步：确认用户已经手动拷贝的数据（网上下载不到）

`DATA_ROOT` 默认是 `/home/han/linux-han`。如果数据在别处，先执行 `export LGS_DATA_ROOT=<实际路径>`，下面的 `$DATA_ROOT` 都指这个目录。

```bash
DATA_ROOT=${LGS_DATA_ROOT:-/home/han/linux-han}
for s in changjing/shot_0007 changjing/shot_0003 changjing1/shot_0012 changjing1/shot_0032 changjing2/shot_0000; do
  for v in 0 1 2 3 4 5; do test -f "$DATA_ROOT/datasets/$s/view_$v.png" || echo "缺少 $DATA_ROOT/datasets/$s/view_$v.png"; done
done
test -f "$DATA_ROOT/datasets_1080/calib/homographies_cam3_raw2304_to_crop1920x1088_halfpx.npy" || echo "缺少单应矩阵"
test -f "$DATA_ROOT/datasets_1080/calib/calib_1920x1080.json" || echo "缺少标定 json"
test -f "$DATA_ROOT/Practical-RIFE/train_log/flownet.pkl" || echo "缺少 RIFE 权重"
test -f "$DATA_ROOT/Practical-RIFE/train_log/IFNet_HDv3.py" || echo "缺少 RIFE 网络定义"
test -f "$DATA_ROOT/Practical-RIFE/model/warplayer.py" || echo "缺少 RIFE model/warplayer.py"
```

如果上面有输出，说明数据没拷全，需要用户从 Windows 主机拷贝（`D:\hb` 对应 `$DATA_ROOT`）：
- `datasets/` 下的 5 组场景原图（2304×1296，每组约 52 MB）；
- `datasets_1080/calib/`；
- `Practical-RIFE/train_log/` 和 `Practical-RIFE/model/`。

`train_log` 和 `flownet.pkl` 必须是同一版本，不要用网上下载的替换。

### 第 2 步：克隆代码

```bash
cd $DATA_ROOT
git clone https://github.com/luckyhan-mike/local-gs-eval.git
git clone https://github.com/InternRobotics/AnySplat.git      # 放在 $DATA_ROOT/AnySplat（配置 anysplat_dir）
```

### 第 3 步：Python 环境（按 AnySplat 官方方式，Python 3.10 + torch 2.2 + CUDA 12.1，支持 4090）

```bash
conda create -n gspl python=3.10 -y
conda activate gspl
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
cd $DATA_ROOT/AnySplat && pip install -r requirements.txt     # 含 gsplat 预编译包、lpips、e3nn、xformers 等
pip install pyyaml scipy scikit-image opencv-python
```

- `requirements.txt` 里的 pytorch3d 需要从源码编译。如果失败可以先跳过，本实验不调用 pytorch3d：删掉那一行后重新安装。
- 不要让 pip 把 torch 升级到别的版本。若被升级，重新执行上面安装 torch 的那一行。

检查：

```bash
python -c "import torch, gsplat, lpips; print(torch.cuda.get_device_name(0), torch.__version__, gsplat.__version__)"
```

应打印出 4090 和版本号。

### 第 4 步：单元测试

```bash
cd $DATA_ROOT/local-gs-eval
python -m unittest discover -s tests -v       # 17 项应全部通过
```

### 第 5 步：运行实验（5 组场景，每组依次跑 prepare → token → GS+TTO → 光流 → 评估 → 验证）

```bash
cd $DATA_ROOT/local-gs-eval
export PYTHONUNBUFFERED=1            # 第一次运行不要设置 HF_HUB_OFFLINE，让 AnySplat 权重自动下载
mkdir -p work
python scripts/run_scenes.py --config configs/dataset6_lean.yaml 2>&1 | tee work/dataset6_run.log
```

- 中途失败：看 `work/dataset6/<场景>_<shot>/logs/<步骤>.log`。修好后可以从失败的步骤接着跑，例如 `--shots changjing/shot_0007 --steps anysplat flow evaluate validate`。
- 只重新生成汇总：加 `--summary-only`。
- 如果 TTO 一步报错，或者显存不足，把日志交给用户，不要擅自改算法参数。

### 第 6 步：上传结果

结果汇总在 `work/dataset6/summary_all.md`。上传到结果仓库 https://github.com/hhan70701/local-gs-eval-results 的新分支 `experiment/dataset6-in035-<日期>`，放在目录 `results/dataset6-in035-<日期>/` 下。

要上传的内容：
- `summary_all.md`、`dataset6_run.log`；
- 每组场景目录下的 `config.yaml`、`timing.json`、`poses/`、`logs/`、`eval/`（包括 `eval/in035/cam*_montage.png`）、`prepare/prepare.json`、`tokens/tokens.json`；
- `gs/in035/*/meta.json`（含 TTO 耗时与监督框）；
- 本机环境信息：GPU、驱动、torch、gsplat 版本。

不用上传：`aligned/`、`flow/`、`gs/` 里的大图。

## 0. 六路新数据集上的局部 GS 验证（配置 `configs/dataset6_lean.yaml`，脚本 `scripts/run_scenes.py`）

方法和第十二轮的精简流程完全相同：token 变体 rife，`_al` 对齐域，直接贴回并附 oracle 上限，前馈 + TTO 50 / 100 / 200 步，逐视图裁剪监督。只把数据换成六路阵列数据集（2304×1296 原图 `datasets/`，标定在 `datasets_1080/calib/`，灰墙背景）。

**相机组合**：输入 0 / 3 / 5（间距约 359 / 233 mm），真值是 1、2、4 号。

**场景**：从宫格图里挑了 5 组，手、抱枕、小物件挡在人前，或人离相机很近，写在 `dataset6.shots` 里，可以增删：
- `changjing/shot_0007`：抱枕举在脸前
- `changjing/shot_0003`：双手挡在脸前
- `changjing1/shot_0012`：手拿小物件挡在脸前
- `changjing1/shot_0032`：书挡在脸前
- `changjing2/shot_0000`：人离相机很近

**和魔方数据的差别（都写在配置里）**：
- **对齐图**：和 video-tell 的几何一致（见 `datasets_1080/README.md` 第 6 节）。`prepare.py` 在 **2304×1296 原图**上以 cam3 为参考做单应对齐，再中心裁出 **1920×1088**，用的矩阵是 `homographies_cam3_raw2304_to_crop1920x1088_halfpx.npy`：6 路的裁剪窗口一起平移半个像素，让 cam3 也经过一次重采样。否则 cam3 只是整数平移、没有插值，会比其他几路更锐，人为抬高光流的失败率（见 `datasets_1080/README.md`）。
  - 不先缩小，像素尺度与原图相同；cam0、cam1 只有约 0.3% 的黑边。
  - cam3 是输入相机，`H_3` 只是平移裁剪，所以对齐域相机不会出现斜切。
  - 没有去畸变。
  - AnySplat 的输入是 2304 原图上的裁剪窗口。
- **前景**：用背景墙抠像代替绿幕。`chroma` 设为墙面的蓝灰色、低饱和区间，新增 `s_max`。
- **object 区域**：真值中最大的高饱和色块（抱枕、水果、书）膨胀 40 px；没有彩色物体的场景（只有手或人）该区域为空。
- **插值时刻 t**：由标定的相机光心计算，即目标光心在两侧输入光心连线上的投影位置，写入每组的 `poses/in035.json`。不用目标图，也不用 VGGT 联合估计。

**跑法**：需要的数据是 `data_root/datasets/<场景>/<shot>/view_0..5.png`（原图，可以只拷选中的 shot）和 `data_root/datasets_1080/calib/`，不需要第七轮的复用目录。每组场景依次运行 prepare → token 检测 → GS（含 TTO）→ 参考光流 → 评估 → 验证，各自的工作目录在 `work/dataset6/<场景>_<shot>/`。

```bash
cd local-gs-eval && git pull
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1      # 第一次运行、权重还没缓存时不要设 HF_HUB_OFFLINE
python -m unittest discover -s tests -v
python scripts/run_scenes.py --config configs/dataset6_lean.yaml 2>&1 | tee work/dataset6_run.log
```

- 只跑某几组：加 `--shots changjing1/shot_0012 changjing/shot_0007`。
- 某一步失败后重跑：从失败的步骤开始，例如 `--steps anysplat flow evaluate validate`。
- 只重新汇总：加 `--summary-only`。

**结果**：
- 汇总在 `work/dataset6/summary_all.md`：逐场景、全部场景合计两张表（纯光流 / 前馈 / TTO 各步数 × 直接 / oracle），以及耗时。
- 每组场景的完整 `eval/summary.md`、对比图和日志在各自的工作目录里。

上传结果时，把 `work/dataset6/` 下除 `aligned/`、`gs/`、`flow/` 大图以外的内容打包，即 `summary_all.md`，以及各组的 `eval/`、`logs/`、`timing.json`、`config.yaml`、`poses/`、`prepare/prepare.json`、`tokens/tokens.json`。另外附上每组 `eval/<setup>/cam*_montage.png`。

**判断标准**：
1. 光流在这组数据上是否真的失败：纯光流的 fg / object 指标，以及 oracle_fail 面积。如果光流几乎不失败，GS 也就没有用武之地。
2. 局部 GS 能否超过光流：直接贴回的 fg 两项指标和不差目标数，前馈与 TTO 分开看。
3. 和魔方数据的结论是否一致：前馈单独不够，TTO 50–100 步后超过光流。
4. 耗时：编码时间、监督框面积、各检查点的累计时间。

## 0.1 第十二轮：精简实验 + 逐视图裁剪 TTO（配置 `configs/mofang_9cam_lean.yaml`）

第十一轮结论：
- **质量**：TTO 在 100 步左右就基本到顶。cam3 只要 50 步（3.5 s）就 6/6 不差；cam5 要 200 步（24.5 s）才 3/4 不差。300 步的质量与第十轮持平，说明裁剪监督不掉质量。
- **速度**：裁剪框仍占视图的 96% / 91%，每步耗时没有下降，提速只来自少跑步数。框这么大，是因为可训练高斯取的是全部 5 个 token 变体（含很松的 r5）重建区的并集；另外所有视图共用同一个框尺寸。

**精简**：前十一轮已经有结论的对照分支和诊断不再跑，只保留部署方法和必要的对照。完整配置 `mofang_9cam.yaml` 及其代码不动，换新 token 等情况下仍可跑全套。

| 去掉的内容 | 依据 |
|---|---|
| token 变体 r5 / strict / rife_g / strict_g 的局部渲染与 TTO | rife 最好；gate 变体在有 TTO 后反而切掉增益；r5 太松（编码窗口仍按全部变体的并集，与前几轮相同） |
| `_alv` 位姿（用目标图估计） | 只是对照，部署时不可用 |
| 整窗口 GS 渲染、真值配准（+reg）、环带配准（ringreg） | 早期诊断，已有结论 |
| w20 目标侧复核 | 第八轮：与不复核几乎相同（oracle 上限保留） |
| 其余光流变体（s0.5、uniform、oracle t）的评估、风险阈值扫描 | 光流与 token 自第七轮起固定 |
| TTO 25 / 300 步 | 第十一轮：25 步不够，100 步后几乎不再提升 |

保留的内容：
- 两组相机 cam3 / cam5；
- 参考光流；
- 方法：rife、`_al`、直接贴回，阶段为前馈和 TTO 50 / 100 / 200 步，每个都附 oracle 上限；
- 指标：fg / object / rest 等区域的 PSNR 和 LPIPS、不差目标数、逐目标明细、耗时。

TTO 输入 PSNR 只在监督框内计算；SSIM 指标仍照常计算。

**提速**（`gs.tto.crop.per_view: true`）：
- 可训练高斯只来自 rife（`gs.local_variants: [rife]`）。
- 每个输入视图按自己的投影外接框（加 32 px 余量）单独光栅化和监督，不再共用一个框尺寸。
- 日志中 `supervised box` 打印的是最大框尺寸和各视图平均面积占比；各视图的框写在 `meta.json` 的 `tto.log[0].box.per_view`。

跑法（新主机的准备见下一节；token 等沿用第七轮）：

```bash
export LGS_WORK_DIR=work/round12 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p $LGS_WORK_DIR && cp -r work/round7/{prepare,flow,poses,tokens,compat_packages} $LGS_WORK_DIR/
python -m unittest discover -s tests -v
python scripts/run_anysplat.py --config configs/mofang_9cam_lean.yaml 2>&1 | tee $LGS_WORK_DIR/1_anysplat.log
python scripts/evaluate.py --config configs/mofang_9cam_lean.yaml 2>&1 | tee $LGS_WORK_DIR/3_evaluate.log
python scripts/validate_results.py configs/mofang_9cam_lean.yaml
```

先看 `summary.md` 的"测试时优化：步数–耗时–质量"表，再看"颜色来源对比"表（只有 rife，含 oracle）。

判断标准：
1. **提速**：监督框面积占比明显低于 96% / 91%；每步耗时 = 累计耗时 / 步数，与第十一轮的 70 / 123 ms 比。换了 4090，可用日志里的编码时间（3090 上 0.83 / 0.52 s）估计两台机器本身的速度差。
2. **质量不变**：rife 前馈与 TTO 100 / 200 步应与第十一轮接近（cam3 TTO200 27.33 / 0.0460；cam5 TTO200 29.15 / 0.0382）。
3. **最少步数**：cam3 和 cam5 最早在哪个检查点达到 6/6、3/4，以及对应的每帧耗时。
4. **流程总耗时**：从 GS 到评估完，应在十几分钟内。

## 0.a 新主机准备（魔方数据 / 第十二轮用；六路数据集见上面 ★ 一节）

需要从旧主机或网上拿到下面这些（`data_root` 默认 `/home/han/linux-han`，也可以用环境变量 `LGS_DATA_ROOT` 指定）：

1. **代码**：`git clone` 本仓库。
2. **数据集**：`北极星/0/魔方_9/grid_00~08.png`（原图）和 `北极星/1/魔方_9/aligned_grid_00~08.png`（对齐图），放在 `data_root` 下。
3. **AnySplat 源码**：`data_root/AnySplat/`。
4. **AnySplat 权重**：Hugging Face 缓存 `~/.cache/huggingface/hub/models--lhjiang--anysplat/`（使用 `HF_HUB_OFFLINE=1` 时必需）。新主机能联网的话，也可以不设 `HF_HUB_OFFLINE`，让它自动下载。
5. **LPIPS 用的 AlexNet 权重**：`~/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`（没有的话首次评估会自动下载）。
6. **第七轮的复用输入**：`local-gs-eval/work/round7/{prepare,flow,poses,tokens,compat_packages}`，放到新主机同一路径。其中 `compat_packages` 是 e3nn 0.5.6 兼容包。这样光流基线和 token 与前几轮逐位一致，验证脚本会核对。
7. **Practical-RIFE**：`data_root/Practical-RIFE/train_log/{IFNet_HDv3.py,flownet.pkl}`。只有重新跑光流或 token 时才需要，本轮用不到，建议一并拷贝。
8. **conda 环境**：建议在旧主机上用 `conda pack -n gspl -o gspl.tar.gz` 打包，在新主机解压到 `~/anaconda3/envs/gspl` 后执行 `bin/conda-unpack`。这样得到的是同一套 Python 3.9、torch 2.0.1+cu117、gsplat 1.5.3。
   - 4090 能直接运行 cu117 的 torch，但需要较新的 NVIDIA 驱动（≥ 525）。
   - gsplat 首次运行如果要重新编译 CUDA 扩展：需要 CUDA 11.7 的 nvcc，并先 `export TORCH_CUDA_ARCH_LIST="8.6+PTX"`（nvcc 11.7 不认识 4090 的 8.9 架构；8.6 的代码可以直接在 4090 上运行）。
   - 也可以把旧主机的 `~/.cache/torch_extensions/` 一起拷过去。

自检：

```bash
python -c "import torch, gsplat; print(torch.cuda.get_device_name(0), torch.__version__, gsplat.__version__)"
python -m unittest discover -s tests -v
```

## 0.0 第十一轮：测试时优化提速

第十轮结论：TTO 第一次让局部 GS 在两组上都超过纯光流（rife、`_al`、直接贴回、300 步：cam3 27.20 / 0.0470，6/6 不差；cam5 29.03 / 0.0390，3/4 不差）。1000 步留出视点反而变差（过拟合）。但耗时太长：前馈编码 0.5–0.8 s，TTO 到 1000 步 cam3 75 s、cam5 138 s，约每步 75 / 138 ms。慢的原因是每一步都把全部 3/5 个输入视图在约 911² 全图上渲染并反传，而真正优化的只是 token 区附近的高斯。

本轮只改速度，算法和其余配置不变（`gs.tto`）：
- **只到 300 步，加早期检查点**：`checkpoints: [25, 50, 100, 200, 300]`，用来找"最少几步就够"。
- **裁剪监督**（`gs.tto.crop`）：把可训练高斯投影到每个输入视图，取 0.2%–99.8% 分位外接框，外扩 32 px；各视图用同一尺寸的框（位置各自不同），平移主点后只渲染、只监督框内。框外高斯被视锥剔除，不参与计算。框是整幅视图的精确子窗口，渲染结果与全图渲染在框内一致。
- **逐检查点计时**：GPU 同步后记录从优化开始到每个检查点的累计墙钟时间，写入 `meta.json` 的 `tto.log[*].time_s`，监督框尺寸在 `tto.log[0].box`。

跑法（token 等仍沿用第七轮，GS 需重跑）：

```bash
export LGS_WORK_DIR=work/round11
mkdir -p $LGS_WORK_DIR && cp -r work/round7/{prepare,flow,poses,tokens,compat_packages} $LGS_WORK_DIR/
python scripts/run_anysplat.py --config configs/mofang_9cam.yaml
python scripts/evaluate.py --config configs/mofang_9cam.yaml
python scripts/validate_results.py configs/mofang_9cam.yaml
```

日志会打印一行 `TTO: … supervised box WxH (x% of view), …; it1: PSNR … , it25: PSNR … @ …s, …`。

`summary.md` 新增"测试时优化：步数–耗时–质量"表（rife、`_al`、直接贴回，不用真值）：每个检查点的累计耗时、监督区域、输入 PSNR、fg / object 指标和不差目标数。"颜色来源对比"表照旧（`_alv` 行现在带 `(alv)` 标记，不再与 `_al` 同名）。

判断标准：
1. 提速：300 步耗时相对第十轮（1000 步 75 / 138 s，300 步按比例约 22 / 41 s，未实测）降了多少；监督框占视图的比例。
2. 裁剪是否掉质量：本轮 TTO300 与第十轮 TTO300（cam3 27.20 / 0.0470，object 20.06 / 0.0926；cam5 29.03 / 0.0390，object 22.63 / 0.0556）应基本持平（跨轮重新编码有轻微数值差异）。
3. 最少步数：25 / 50 / 100 步里，最早在哪一步两组的 fg 均值就超过纯光流、不差目标数达到第十轮 300 步的水平；它对应的耗时就是每帧的实际额外开销。
4. 如果裁剪后仍然慢（比如监督框占视图 50% 以上），下一步考虑降监督分辨率（`max_res`）或每步只随机取一个输入视图。

## 0.00000 第十轮：局部 GS 测试时优化

第九轮结论：从输入原图取纹理，两条路线都明显差于直接用 GS 颜色（cam3 rife：GS 颜色 26.38 / 0.0537；GS 几何 + 输入纹理 23.48 / 0.0908；VGGT 几何 + 输入纹理 24.52 / 0.0997；object LPIPS 从 0.16 恶化到 0.40）。原因有两点：
- GS / VGGT 几何用来做对齐足够准，但按像素取色不够准。魔方色块边缘锐利，深度差 1–2 px（448 分辨率）就会取到相邻色块，出现"清晰但位置错"的纹理，而 LPIPS 对错位的惩罚比对模糊更重。
- VGGT 深度和 GS 深度在输入视点上只差 0.3–0.7%（两者同源），换成 VGGT 不会得到更好的几何。VGGT 路线另有 z-buffer 放大造成的条纹空洞（实现问题）。

结论是：GS 颜色虽然偏软，但它和各视点一致，几何小误差表现为模糊而不是错位。要让补丁变好，就得**让 GS 本身变清晰**。

第十轮以 AnySplat 前馈高斯为初值，在输入视点上做**测试时优化**（`lgs/tto.py`，`gs.tto`）：
- 监督用输入视点的**全分辨率裁剪**（约 911 px，而不是 448）。损失为 0.8·L1 + 0.2·(1−SSIM)。
- 默认只优化 token 重建区（各变体并集）里的高斯（`scope: local`），其余高斯固定但照常渲染。不做加密/剪枝，所以各变体的局部选择、贴回区都不变。
- 参数：位置（学习率乘以基线长度）、协方差（`cov = L0(I+A)(I+A)ᵀL0ᵀ`，L0 为前馈协方差的 Cholesky 因子，A 初值 0，第一次渲染与前馈完全相同）、不透明度、SH 直流颜色。高阶 SH 默认不优化（`sh_rest: false`），因为输入视点少，容易过拟合视角相关颜色。
- 检查点 300 和 1000 步各渲染一套：`gs_tok<变体>448_nat_dil_al_tto300_local` / `…_tto1000_local`（`_alv` 同理），与前馈版本共用同一 token 贴回区，是严格对照。
- 日志打印 TTO 前后输入视点 PSNR（只是拟合程度，越高不代表新视点越好）和耗时，写入 `meta.json` 的 `tto`。
- 第九轮的输入纹理重投影默认关闭（`gs.reproject.geometries: []`）。

`summary.md` 的"颜色来源对比"表按 token 变体列出前馈 / TTO300 / TTO1000 × 直接贴回 / oracle 的 fg、object、rest 和不差目标数。

判断标准：
1. TTO 是否让补丁变清晰：同一 token 下 cam3 object LPIPS 明显低于前馈的约 0.16，fg 高于 26.4。
2. cam5 能否超过光流：至少 oracle 行优于 28.00 / 0.0396。
3. 是否过拟合：输入视点 PSNR 很高、但留出视点变差（尤其 1000 步差于 300 步），说明 3/5 个视点约束不够，需要更少步数或更强正则。
4. 速度代价：日志中 TTO 耗时（每帧额外开销）。

## 0.000 第九轮：几何来自 3D，纹理取自输入图

第八轮结论：目标侧复核几乎没有改变贴回区。光流自己的风险信号在魔方边缘总会亮，不管合成得对不对，和 token 检测本质上是同一信息。更关键的是，**完美检测也只多出很少的收益**：

| | cam3 fg | cam5 fg |
|---|---|---|
| 纯光流 | 22.60 / 0.0629 | **28.00 / 0.0396** |
| rife 直接贴回 | 26.31 / 0.0536 | 27.71 / 0.0420 |
| rife，只贴真值失败区（oracle） | 26.51 / 0.0513 | 27.96 / 0.0406 |

cam3 上检测已接近饱和（只差 0.2 dB）。cam5 上即使只在光流真正出错的像素贴 GS，也仍然不如光流（目标 3 为 −0.95 dB）。瓶颈因此在 **GS 补丁本身的颜色**：AnySplat 前馈出的纹理偏软，贴回区内只有约 18–21 dB；而它的几何是可靠的（环带位移约 0，plane RMS 约 2 px）。

第九轮把几何和纹理拆开：3D 模型只提供虚拟视点的深度；颜色按深度反投影后，从两侧输入**原图**（全分辨率、清晰）采样。只取输入相机真正看得到的点（深度测试），两侧按 (1−t, t) 加权（`lgs/reproject.py`）。几何来源对比两条路线，其余流程完全相同：

| 后缀 | 虚拟视点深度 | 输入相机可见性 |
|---|---|---|
| `_rpgs` | 在虚拟视点渲染全部高斯的期望深度 | 高斯在各输入相机渲染的深度 |
| `_rpvggt` | AnySplat 内部 VGGT 深度图（两侧输入视点）前向投影到虚拟视点，z-buffer 取最近，空洞按较远邻居（背景）填补 | 同一 VGGT 深度图 |

- 两条路线用**同一个 token 贴回区**（`alpha_core` 与对应 GS 局部变体相同），只换颜色来源，所以和"GS 颜色贴回"是严格对照。
- VGGT 深度直接取自 AnySplat 编码时的深度头输出（`depth_dict`），与高斯在同一坐标系，不需要另跑模型。日志和 `meta.json` 中的 `vggt_vs_gs_depth_rel` 给出两者在输入视点上的深度差（中位相对误差）；`reproject_stats` 给出虚拟视点每个像素被两侧/一侧/零侧输入相机看到的比例。
- 只对可部署的 `aligned_int` 位姿做（`gs.reproject.sources`）。输出目录：`gs_tok<变体>448_nat_dil_al_rpgs_local`、`…_al_rpvggt_local`，评估自动生成 `hyb_token[…]` 和 `hyb_tokchk_oracle[…]`。
- 目标侧复核只保留 `w20` 和 oracle（第八轮其余阈值与不复核几乎相同）。
- `summary.md` 新增"纹理取自输入图"表：每个 token 变体 × 三种颜色来源（GS 颜色 / GS 几何 + 输入纹理 / VGGT 几何 + 输入纹理）× 直接贴回 / oracle，列出 fg、object、rest 和不差目标数。

判断标准：
1. 输入纹理是否比 GS 颜色更好：同一 token 下 `_rpgs` / `_rpvggt` 的 fg LPIPS 应明显低于 GS 颜色，cam3 的 PSNR 不低于 26.3。
2. cam5 是否终于超过光流：至少 oracle 行应优于光流 28.00 / 0.0396；直接贴回最好也不低于光流。
3. VGGT 能否替代 GS：`_rpvggt` 不差于 `_rpgs`，说明可以去掉 GS，框架简化为"光流主干 + VGGT 几何引导局部重投影"。
4. 风险点：深度边缘（魔方与头发交界）。448 分辨率深度在原图上会偏软，可能出现颜色渗出或飞点，注意看拼图和 object 区 LPIPS。

## 0.00 第八轮：目标侧光流风险复核（只改评估）

第七轮结论：固定窗口后所有变体共用一次编码（自渲染 29.1 / 29.4 dB），cam5 strict 的崩溃消失。cam3 上不加门限的 `rife` 是目前最好的结果（fg 26.31 dB / 0.0536，光流 22.60 / 0.0629；rest 33.49）。但按面积设门限两头都判错了。下表是 rife 的贴回区内 PSNR，光流 → GS：

| 目标 | 相机对 core | 贴回区内 光流 → GS | 门限 0.03 |
|---|---:|---|---|
| cam3 5 / 6 / 7（对 4–8） | 2.5% | 15.9→20.3 / 14.5→19.0 / 16.7→18.5 | 错关：头发与衬衫交界的遮挡带，光流确实错了 |
| cam5 1（对 0–2） | 5.9% | 16.7→18.7 | 触发，正确 |
| cam5 3（对 2–4） | 3.7% | 24.6→21.5 | 错开：光流本来就对 |
| cam5 5（对 4–6） | 0.8% | 19.0→17.0 | 关闭，正确 |

决定 GS 能不能赢的，是**这个虚拟视点里光流在贴回区错得有多严重**：GS 偏糊，在这些区域大约只能做到 18–21 dB；光流低于约 17 dB 时 GS 赢，高于约 19 dB 时 GS 输。输入相机上的 token 面积只说明这里可能有问题，回答不了这个问题。

第八轮在贴回前加一步**目标侧复核**：贴回区 = token core ∩ 参考光流在实际 t 下自己的风险区。风险区由 |W₀−W₁| 或共线残差超阈值、外扩、限制在光流结果抠像前景附近得到，第二轮测得这两个信号的 AUC 约 0.9，全程不用真值。实现上只改 `evaluate.py`，不需要重跑 AnySplat：

- `eval.target_check` 列出几组阈值（`w06` / `w08` / `w12` / `w20` / `w08d`），每组生成 `hyb_tokchk_<name>[v]`；
- 另加 `hyb_tokchk_oracle[v]` = core ∩ 真值失败区，作为"完美目标侧复核"的上限（诊断用）；
- 只对 `_al_local` 变体做（`target_check_suffix`）；每个目标的贴回面积写在 `tokchk.csv`；
- `summary.md` 新增"目标侧光流风险复核"表：每个 token 变体 × 复核方式的贴回面积、fg / object / rest，以及"不差目标数"（fg 的 PSNR 不低于、LPIPS 不高于纯光流的目标个数）。

判断标准：理想的复核方式在 cam3 保留不加门限 `rife` 的大部分收益（fg ≈ 26 dB），6/6 个目标不差；在 cam5 让目标 3、5 退回光流，fg 不低于 28.00 / 0.0396。再与 oracle 行比较，看目标侧信号离完美复核还差多少。

## 0.0 第七轮：逐对 token + 固定编码窗口

第六轮结论：收紧 token 后 cam3 的 `strict`/`rife` 第一次在 fg 的 PSNR、LPIPS 两项上都优于纯光流（strict 26.31 dB / 0.0558 vs 22.60 / 0.0629）。但逐目标拆开看，**收益全部来自目标 1–3**：

| cam3 目标 | 所在相机对 | 光流 object PSNR / LPIPS | strict object PSNR / LPIPS |
|---|---|---|---|
| 1, 2, 3 | 0–4 | 8.5–9.9 / ≈0.45 | 14.3–17.7 / ≈0.20 |
| 6, 7 | 4–8 | 21.8–22.5 / ≈0.05 | 16.4–18.5 / 0.17–0.20 |

光流的失败是**单侧**的：魔方在 0–4 这一对里严重失败，在 4–8 里基本正确（strict 在 cam0/4/8 的 core 分别为 12.3% / 12.7% / 0.1%）。但第六轮把所有输入相机的 token 取并集，cam4 上由 0–4 检出的魔方 token 也被贴到 4–8 之间的目标上，把清晰正确的光流魔方换成了偏软的 GS 魔方。另一个问题是每个变体按自己的重建区裁剪窗口，token 越少窗口越小：cam5 strict 窗口只有 448 px，AnySplat 输入自渲染掉到 21.2 dB（r5 窗口 917 px 时为 29.5 dB），贴回区内只有 15–17 dB。

第七轮的三处改动：

1. **逐对 token**（`detect_tokens.py`）：每个相机对 (a, b) 只用该对 RIFE t=0.5 的失败信号（以及该对的 DIS 遮挡）标记 a、b 两张图，输出到 `tokens/<setup>/<变体>/pair{a}-{b}/`。a、b 之间的虚拟视点只用这一对的 token 选高斯、算贴回足迹（`run_anysplat.py` 的 `local_keep`）。并集仍写在 `tokens/<setup>/<变体>/`，用于叠加图和编码窗口。
2. **固定编码窗口**（`gs.token_window: fixed`）：每个设置只编码一次 AnySplat，窗口取所有变体重建区的并集（实际由 r5 决定，约 910 px，自渲染约 29 dB）。各 token 变体共用同一组高斯，只决定选哪些高斯、贴到哪里，所以变体之间是干净的对照。命名：整窗口 `gs_tok448_nat_dil_al`，局部 `gs_tok<变体>448_nat_dil_al_local`。`per_variant` 可恢复第六轮的做法。
3. **逐对触发门限**（变体 `rife_g` / `strict_g`，`pair_gate: 0.03`）：某对的 core 面积（两台相机的平均值，占前景比例）低于门限时，这一对之间的虚拟视点不触发 GS，等于纯光流。0.03 是初值；`summary.md` 新增的"逐目标视点"表会列出每个目标所在相机对的 core 面积和 ΔPSNR / ΔLPIPS，可以据此事后确定门限。

本地真实数据上的逐对 core（占前景）：

| 设置 | 相机对 | r5 | rife | strict | 门限 0.03 |
|---|---|---:|---:|---:|---|
| cam3 | 0–4 | 30.2% | 14.5% | 12.4% | 触发 |
| cam3 | 4–8 | 25.2% | 2.5% | 0.4% | rife_g / strict_g 关闭 |
| cam5 | 0–2 | 18.8% | 5.9% | 4.0% | 触发 |
| cam5 | 2–4 | 13.0% | 3.7% | 2.1% | rife_g 触发，strict_g 关闭 |
| cam5 | 4–6 | 11.2% | 0.8% | 0.2% | 关闭 |
| cam5 | 6–8 | 7.1% | 0.4% | 0.0% | 关闭 |

预期：cam3 目标 1–3 保持第六轮的提升，目标 5–7 回到纯光流附近（`*_g` 恰好等于纯光流），6 个目标都不差于光流；cam5 大部分目标等于纯光流，fg 不再低于光流。

## 0.01 第六轮：收紧 token 检测

第五轮结论：对齐域渲染解决了几何错位（cam3 环带位移 0 px），token 贴回第一次超过纯光流（cam3 fg 25.27 vs 22.60 dB，魔方区 16.99 vs 14.53 dB）。但贴回区仍覆盖约 49% 前景，而光流真正失败的区域只有约 12%；其余部分把正确的光流换成更软的 GS，导致 fg LPIPS 和 rest 变差。用同一批 `_al` 高斯只贴 oracle 失败区时 fg 达到 27.03 dB / 0.057（PSNR 与 LPIPS 均优于光流），说明**瓶颈是检测精度**。小基线（cam5）即使完美检测也不如光流，应基本不触发。

第六轮一次跑多个 token 检测变体（`tokens.variants`），每个变体各自编码一次 AnySplat，便于直接比较：

| 变体 | 遮挡/匹配失败 | 深度突变 | 去小连通域 | 本地实测输入 core（占前景，各输入相机） |
|---|---|---|---|---|
| `r5`（第五轮，对照） | DIS 遮挡 + RIFE（\|W₀−W₁\|>0.08 或共线残差>2） | 任意 | 否 | cam3 26–38%；cam5 7–22% |
| `rife` | 仅 RIFE（>0.20 或 >6，阈值来自第二轮风险扫描最优 IoU） | 只保留紧挨失败处的 | ≥150 px | cam3 3–16%；cam5 0–9% |
| `strict` | 仅 \|W₀−W₁\|>0.25 | 不用 | ≥300 px | cam3 0–13%；cam5 0–7% |

`rife` 的标记集中在魔方、手和魔方扫过的头发上，脸和衣服基本不再被标（`tokens/<setup>/<variant>/cam{k}_overlay.png`）。基线越小标记越少，小基线下可能完全不触发（此时等价于纯光流）。

位姿只保留可部署的 `aligned_int`（光心按 t 插值）和对照 `aligned_vggt`；裁剪只保留 `token`。GS 变体命名：`gs_tok<变体>448_nat_dil_al_local`。`summary.md` 新增"Token 变体对比"表：输入 core、贴回区面积、fg / object / rest，并列出 oracle_fail 面积作对照。

另外合入第五轮 Linux 上的两处修复（RGB+深度渲染背景通道、aligned 位姿诊断回退），并修正 summary 中按位姿来源分类把 `_al`/`_alv` 归为 vggt 的问题。

## 0.1 第五轮：在对齐（离轴）域渲染 GS，贴回足迹改为可见表面

第四轮 token 贴回能把魔方补对，但整体变差，原因有两个，第五轮分别修正：

1. **统一坐标系 = 光流的对齐域。** 第四轮 `_int` 在两侧输入相机之间插值朝向，但本阵列每台相机是单独对准的，插值朝向与真实目标相机差 0.6–2.5°（f≈1820 px 时 20–80 px），再用真实相机的 H_k 贴回就整体错位。光流其实一直在一个统一坐标系里工作：单应对齐后的图像域，各相机各自的朝向已被消掉，只剩光心位置带来的视差，正是光场显示的**离轴相机**。第五轮让 GS 直接在这个域渲染（`lgs/aligned.py`，位姿来源 `aligned_int` / `aligned_vggt`，后缀 `_al` / `_alv`）：
   - 对齐参考相机：`H_ref · K_ref = K_al · Q`（RQ 分解），得到对齐域内参 `K_al` 和朝向；
   - 零视差平面：只用输入相机的 GS 位姿/内参和它们已知的单应 H，在高斯坐标系里拟合平面（`plane_rms_px` 是拟合残差，同时反映 AnySplat 输入相机几何与真实 H 的一致程度）；
   - 虚拟视点：参考朝向 + 对齐内参，放在目标光心 C 处渲染，再乘零视差平面单应 G_C。

   **依据**：同一光心的两台相机之间对所有点都只差一个单应，所以这样渲染出的图与"任意朝向、任意内参的真实相机在 C 处拍摄、再用它自己的 H 对齐"**完全相同**（不只是平面上的点）。真实相机的朝向和内参因此不再影响结果，只剩光心位置和高斯几何本身。单元测试在合成会聚阵列上验证了这一点（离平面的点误差 <1.5 px，独立数值检验 1e-5 px）。
   - `aligned_int`：光心在两侧输入相机光心间按光流同一个 t 插值（部署时生成虚拟视点就是这样）；
   - `aligned_vggt`：光心取 VGGT 联合估计（与真实目标相机对照，区分"光心位置误差"和"高斯几何误差"）。

2. **贴回足迹 = 输入相机里可见的表面。** 第四轮保留了视线穿过 token 的所有高斯，魔方后面的头发、绿幕上的大高斯也进来了，投影到虚拟视点后足迹约占前景的 61%，还有 40% 在前景外，把可靠的光流也换掉了（rest 从 33.5 dB 掉到 21.9 dB）。第五轮改为：渲染每台输入相机的期望深度，只保留投影落在 token 区域内、且深度通过测试（≤ 渲染深度×1.03）的高斯（`gs.keep_mode: visible`）。遮挡区要补的背景正是另一台输入相机 token 区域里的可见表面，所以不会丢。贴回区还会限制在光流结果前景（抠像）外扩 15 px 内（不用真值）。

## 0.2 第四轮：按原始框架做 token 局部 GS

前三轮的不可靠区域都定义在**虚拟视点（目标图）**上，GS 也是对整个裁剪窗口重建，这和原始框架不一致。第四轮改为：

```
3 台真实输入相机 ──► detect_tokens.py：在输入图像上标记不可靠区域（不使用真值）
                      · 遮挡 / 匹配失败：DIS 双向光流前后向不一致 且 光度不一致
                      · 光流失败：RIFE（合成用的同一网络）在 t=0.5 的 |W0−W1| 与双向共线残差，
                                   沿 RIFE 自身的对应关系映射回两张源图
                      · 深度突变：RIFE t=0.5 视差的大跳变（阈值 2.0，只取魔方-头发这类大跳变）
                      core  = 标记 + 6 px            → 贴回区
                      recon = core + 48 px 周围区域  → 重建区（保证 token 周围的场景也重建准确）
         │
         ▼
run_anysplat.py：窗口 = 各输入相机重建区外接框；AnySplat 编码后只保留视线穿过重建区的高斯
                 （前景与其后方背景都保留）= 局部 GS 点云（变体后缀 _local）
         │
         ▼
虚拟视点：光流生成完整图；局部 GS 在同一虚拟相机渲染，在 core 的投影范围内贴回（羽化在 recon 边距内）
         │
         ▼
与纯光流整图对比（真值只用于算指标）
```

**统一位姿坐标系（`_int` 变体）**：GS 不再使用独立估计的目标相机位姿，而是在高斯坐标系里用两侧输入相机的位姿，按**光流同一个 t** 插值（旋转 slerp、位置与内参线性插值）。高斯本身与这些输入相机位姿自洽，光流与 GS 描述的是同一个虚拟相机。对比 `''`（VGGT 联合估计的目标位姿）。

**环带配准（`hyb_token_ringreg`，不用真值）**：光流在 2D 对齐域插值、GS 是 3D 透视渲染，仍可能有残余错位。贴回前在 core 外的一圈（光流可靠）上把 GS 块配准到光流结果，再贴回；位移超过 20 px 视为不可信，退回直接贴回（`hyb_token`）。

**本地（CPU）已在真实数据上跑过 token 检测**（`work/tokens/<setup>/cam{k}_overlay.png` 可直接看）：cam3 的 core 占前景 26–37%，集中在魔方、手、前臂和少量衣服边缘；cam5 为 7–22%，基线越大 token 越多。两个修正：RIFE 在 t=0/1 时融合掩码为 0/1，另一侧的光流不受监督（本应约 24 px，实际只有 7 px），不能用来做源图间的前后向检查，因此改用 DIS；深度阈值 0.5 会把脸、衣服轮廓全部标出（光流在这些地方其实是对的），改为 2.0。

**第三轮更正**：用对齐图上真实匹配点的视差比例独立测得目标相机的 t，VGGT 与实测平均只差 0.011，棋盘格标定差 0.058。因此第三轮"cam3–cam4 间距只有一半"和"RIFE 的 t 对空间位置非线性"两条结论**撤回**：单块正对棋盘格下旋转与平移退化，标定的相机位置不可靠；RIFE 的 t 基本线性（oracle 与实测差 0.018），VGGT 的 t 是可靠的。标定脚本保留但默认不再使用。

## 0.3 第三轮改动（相对第二轮实测）

第二轮结论：cam3 物体区里，配准后的 GS 比最强光流（pose t + scale 0.5）高约 2.7 dB，LPIPS 也更低；但未配准时两者打平。GS 渲染与真值平均错位 8–10 px，说明 VGGT 估计的目标相机位姿和内参是主要瓶颈；滤波核影响不到 0.1 dB。第三轮针对这一点：

1. **离线阵列标定 `scripts/calibrate_rig.py`**。阵列在拍棋盘格和拍魔方之间没有动过，所以两次拍摄共享同一组相机参数，可以用一次光束平差（BA）把两类约束一起解出来：
   - 棋盘格（`北极星/biao_ding_9`，12×9 格 = 11×8 内角点）：9 台相机同时看到同一块平面板，88 个角点的亚像素位置作为已知 3D 点，同时固定世界坐标系（板平面，单位 = 1 格）；
   - 魔方场景：相邻相机对（序号差 ≤ 4）的 SIFT 匹配，经基础矩阵 RANSAC 过滤后三角化，作为未知 3D 点。

   未知量：每台相机的 R、t，共享焦距（9 台同型号；单独估计时焦距与深度耦合，散布 3%），共享径向畸变 k1、k2，主点固定在图像中心。第一阶段只用棋盘格做 `cv2.calibrateCamera`（共享内参）作初值，第二阶段 BA（先线性损失，再 Huber）。

   **本地（Windows CPU）实测**：棋盘 RMS 0.21 px（第一阶段）/ 0.37 px（BA），场景匹配重投影中位数 0.15 px、90% 分位 0.43 px，焦距 ≈1822–1836 px。相机间距明显不均匀，相邻光心间距（单位：格）为 2.19、2.18、1.89、**0.90**、1.82、2.34、1.69、2.56（cam3–cam4 只有其他间距的一半）。这一点也有旋转上的独立印证：cam3→4 的会聚角步长只有 2.3°，是所有相邻对里最小的。

2. **GS 用标定位姿渲染**（`gs.pose_sources: [vggt, calib, calibK]`，变体后缀 `''`、`_cal`、`_calK`）。标定位姿通过输入相机拟合的 Sim(3) 映射到 AnySplat 的高斯坐标系，**目标图像不参与**，因此是可部署的做法：
   - `_cal`：标定位姿 + VGGT 内参（与高斯本身的几何一致）；
   - `_calK`：标定位姿 + 标定内参 + 镜头畸变（先按针孔渲染，再按畸变模型重映射到原图）。

   `+reg` 的配准残差就是检验标准：如果 `_cal` 的残差从 8–10 px 降到 1–2 px、物体区接近第二轮的 `+reg` 上限，这条路线就在可部署条件下成立。

3. **光流的 t 加入 `posecal`**：t 取标定光心在两侧输入相机连线上的投影比例。本地已算出：标定的 t 与 oracle t 并不重合（例如 cam3 目标 3：标定 0.875、VGGT 0.80、oracle 0.76），说明 RIFE 的时间步对真实空间位置的响应本身是非线性的——这正是"时序插帧 ≠ 空间插帧"的直接证据。本轮用实验量化它。

4. **修正 3D Mip 滤波**：第二轮 `_nat_mip` 崩溃的原因是 Mip-Splatting 的不透明度能量补偿 `sqrt(det Σ / det(Σ+σ²I))` 要求高斯在训练时就带着这个滤波；AnySplat 的扁平面片高斯 det Σ ≈ 0，事后补偿会把它们压成透明。现在默认关闭补偿（`gs.filter3d_comp: false`）。第二轮 `in / nat / nat_dil / nat_mip2d` 相差 <0.1 dB，本轮只保留 `nat_dil` 和修正后的 `nat_mip`。

5. **评估**：参考光流改为 `flow_s1.0_pose`（第二轮在两种基线下都优于均匀 t）；`+reg` 的 GS 也构造 `hyb_risk` / `hyb_risktuned` 混合，可以单独量化检测器的损失；配准 `ok` 只在确实接受了非恒等变换时计为成功（第二轮恒等回退也计入了）；关键对比表按位姿来源分别列出最好的 GS。

## 0.4 第二轮改动（相对 2026-10-24 第一轮实测）

第一轮结论：GS 在光流失败区只比光流高 0.15 dB，且 SSIM/LPIPS 更差；但拼图显示 GS 把魔方补对了，误差主要是**全图几个像素的错位**和 **448 渲染再放大 2.4 倍的模糊**（`rest` 区 GS 19.6 dB vs 光流 33.2 dB）。第二轮针对这两点：

1. **渲染方式与滤波核（参考 Mip-Splatting）**。每个 (裁剪, 输入分辨率) 只编码一次，用同一组高斯比较 5 种渲染方式（配置 `gs.renders`）：

   | 变体后缀 | 渲染分辨率 | 滤波 | 说明 |
   |---|---|---|---|
   | `_in` | 输入分辨率 448，再放大到窗口 | 3DGS 原版膨胀 `Σ2D + 0.3I` | 第一轮的做法 |
   | `_nat` | 窗口原生分辨率（center 1080 / roi ≈540） | `Σ2D + 0.3I` | 直接高分辨率渲染，不加补偿（预期出现变细、孔洞、锯齿） |
   | `_nat_dil` | 原生 | `Σ2D + 0.3·(res/448)² I` | 按分辨率比例放大膨胀，相当于在输入网格上保持 0.3 px，对应你在 DirectLF 上"滤波核匹配渲染尺度"的结论 |
   | `_nat_mip2d` | 原生 | Mip-Splatting 2D Mip 滤波：`Σ2D + 0.1I`，不透明度乘 `sqrt(det Σ2D / det(Σ2D+0.1I))` 做能量补偿（gsplat `antialiased` 模式） | 只做屏幕空间抗混叠 |
   | `_nat_mip` | 原生 | 3D 平滑滤波 + 2D Mip 滤波 | **3D 滤波**：每个高斯按输入相机中最密的采样率 ν = f/d 限制最小尺度，`Σ3D + s/ν² I`（s=0.2），不透明度乘 `sqrt(det Σ3D / det Σ3D')`。可见性按输入相机视锥判断，不可见的取最大滤波 |

   Mip-Splatting 的 3D 滤波原本在训练时计算（训练相机的最大采样率）；这里"训练"相机就是 AnySplat 的输入相机，因此直接用输入相机在 448 网格上的 f/d 计算，无需训练。

2. **魔方 ROI 裁剪**（`crops: [center, roi]`）：在每台输入相机中用颜色检测魔方（高饱和、非绿幕色），取并集外接框四周加 80 px，得到一个正方形窗口（本场景约 540×540），放大倍数从 2.4 降到约 1.2。这才是"只在局部跑 GS"的设置。检测失败时可在 `gs.roi_window` 手动指定 `[x0, y0, side]`。

3. **物体级评价区域 `object`**：真值中的魔方掩码膨胀 40 px（含手和魔方周围的遮挡带）。上一轮的 `oracle_fail` 按"光流误差大"定义，里面有大量细轮廓边，GS 只要有亚像素错位就被惩罚，对光流有利；`object` 不依赖任何一方的误差。

4. **GS 配准诊断 `+reg`**：用真值对每个 GS 渲染做 ECC 配准（平移 → 仿射 → 单应逐级，在"前景除物体区"上拟合，每一级只有降低误差才保留）。`gs_x+reg` 与 `gs_x` 的差距 = 位姿/内参误差造成的损失；`registration.csv` 给出平均位移（px）。这是诊断用的上限，用到了真值，不是可部署的方法。

5. **风险阈值扫描**：对 |W₀−W₁| × 共线残差的阈值网格，计算风险区相对 `oracle_fail` 的精确率、召回率、IoU 和面积比；IoU 最优的阈值用于 `hyb_risktuned[...]`（在同一数据上调参，偏乐观）。

6. 合入第一轮 Linux 实测时的兼容修复：Python 3.9 的 AnySplat 注解兼容副本、联合位姿 aggregator 使用 bf16 autocast、AUC 并列值用平均秩。

---

## 1. 评价区域

| 区域 | 定义 | 作用 |
|---|---|---|
| `object` | 真值魔方掩码膨胀 40 px ∩ 前景 | **第二轮主结论区**：GS 能否把光流搞坏的物体补对 |
| `oracle_fail` | 参考光流与真值的误差 > 0.08，膨胀 3 px | 完美检测器假设下的失败区（上限） |
| `risk` | 光流自身信号、配置中的固定阈值 | 现实可得的简单检测器 |
| `edge` | 光流视差水平梯度 > 0.3 | 深度突变处 |
| `rest` | 前景中除去 `risk` 与 `oracle_fail` | 检查平坦区光流是否更好 |
| `fg` | 整个前景（绿幕抠像） | 整体质量 |

ROI 渲染只覆盖窗口内，表中 `覆盖` 列表示该方法在区域内有结果的比例；ROI 方法的 `fg`/`rest` 指标只针对窗口内部分，不能与整幅方法直接比较。**混合方法在窗口外回退到光流，覆盖始终为 1，可直接比较**。

混合方法命名：

- `hyb_oracle_fail[gs_x]`：在 `oracle_fail` 区内换成 GS（完美检测器上限）；
- `hyb_risk[gs_x]`：在配置默认阈值的 `risk` 区内换成 GS；
- `hyb_risktuned[gs_x]`：在扫描得到的最优阈值风险区内换成 GS；
- `hyb_oracle_fail[gs_x+reg]`：配准后的 GS + 完美检测器，即"位姿准确时这条路线的上限"。

---

## 2. 环境

在 Linux 3090 主机上使用能跑 AnySplat 的 conda 环境（含 torch、gsplat、lpips、scikit-image、scipy、opencv、pyyaml）。第一轮实测使用 `gspl`（Python 3.9.23、torch 2.0.1/CUDA 11.7、gsplat 1.5.3）：

```bash
conda activate gspl
python -m pip install --no-deps --target work/compat_packages e3nn==0.5.6   # gspl 的 e3nn 0.6.0 需要更新的 torch；只装到实验目录
```

- Python 3.9 下脚本会把 AnySplat 的 `src/` 复制到 `work/anysplat_py39/`，只转换类型注解里的 `|` 联合类型，不改原源码；
- 若使用 `LGS_WORK_DIR`，上面 e3nn 的目标目录也要改成该目录下的 `compat_packages`；
- 已缓存模型时可加 `HF_HUB_OFFLINE=1`。

需要的外部目录（相对于 `data_root`，可在配置中改为绝对路径）：`Practical-RIFE/train_log/{IFNet_HDv3.py,flownet.pkl}`、`AnySplat/`、`北极星/0/魔方_9/`、`北极星/1/魔方_9/`。

## 3. 运行

```bash
cd local-gs-eval && git pull
HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 bash run_all.sh configs/mofang_9cam.yaml
python -m unittest discover -s tests -v
```

**第十轮（复用第七轮的 token；GS 需要重跑以做测试时优化）**：

```bash
export LGS_WORK_DIR=work/round10
mkdir -p $LGS_WORK_DIR && cp -r work/round7/{prepare,flow,poses,tokens,compat_packages} $LGS_WORK_DIR/
python scripts/run_anysplat.py  --config configs/mofang_9cam.yaml
python scripts/evaluate.py      --config configs/mofang_9cam.yaml
python scripts/validate_results.py configs/mofang_9cam.yaml
python -m unittest discover -s tests -v
```

每个设置的日志会有一行 `TTO: … Gaussians at …px, …s; ctx PSNR it1=… it300=… it1000=…`。先看 `summary.md` 的"颜色来源对比"表。若显存不足，可把 `gs.tto.max_res` 调小（例如 672）。

**第九轮（复用第七轮的 token；GS 需要重跑，因为要输出深度和重投影结果）**：

```bash
export LGS_WORK_DIR=work/round9
mkdir -p $LGS_WORK_DIR && cp -r work/round7/{prepare,flow,poses,tokens,compat_packages} $LGS_WORK_DIR/
python scripts/run_anysplat.py  --config configs/mofang_9cam.yaml
python scripts/evaluate.py      --config configs/mofang_9cam.yaml
python scripts/validate_results.py configs/mofang_9cam.yaml
python -m unittest discover -s tests -v
```

日志里每个设置会打印 `VGGT vs GS depth on inputs`（两种几何在输入视点上的差异）和 `reproject[aligned_int/gs|vggt]: camK both=… none=…`（虚拟视点像素被两侧输入都看到/都看不到的比例）。重投影在 CPU 上用 numpy 计算，每个目标视点约数秒。先看 `summary.md` 的"纹理取自输入图"表。

**第八轮（复用第七轮的 token 和 GS，只重跑评估）**：

```bash
export LGS_WORK_DIR=work/round8
mkdir -p $LGS_WORK_DIR && cp -r work/round7/{prepare,flow,poses,tokens,gs,compat_packages} $LGS_WORK_DIR/
python scripts/evaluate.py      --config configs/mofang_9cam.yaml
python scripts/validate_results.py configs/mofang_9cam.yaml
python -m unittest discover -s tests -v
```

`gs/` 较大时也可以用软链接代替复制（`ln -s $(pwd)/work/round7/gs $LGS_WORK_DIR/gs`）。先看 `summary.md` 的"目标侧光流风险复核"表，再看 `tokchk.csv` 的逐目标贴回面积。

**第七轮（复用第六轮的工作目录）**：`prepare`、光流、`poses/` 不变；token 需要重新检测（新增逐对目录），GS 只编码一次：

```bash
export LGS_WORK_DIR=work/round7
mkdir -p $LGS_WORK_DIR && cp -r work/round6/{prepare,flow,poses,compat_packages} $LGS_WORK_DIR/
python scripts/detect_tokens.py --config configs/mofang_9cam.yaml
python scripts/run_anysplat.py  --config configs/mofang_9cam.yaml
python scripts/evaluate.py      --config configs/mofang_9cam.yaml
python scripts/validate_results.py configs/mofang_9cam.yaml
python -m unittest discover -s tests -v
```

日志里每对会打印 `pair a-b: core=…`（被门限关闭的会标出 `pure flow`），GS 编码后打印每个变体、每对选中的高斯数量。先看 `summary.md` 的"Token 变体对比"和"逐目标视点"两张表：

- cam3：`rife`/`strict` 的 fg 两项仍优于光流，且目标 5–7 不再变差（object LPIPS 不再从 0.05 升到 0.17+）；
- cam5：`rife_g`/`strict_g` 的 fg 不低于纯光流，`strict` 不再出现第六轮那样的 −4 dB；
- 固定窗口下输入自渲染 PSNR 应在 29 dB 左右（AnySplat 统计表）。

**第六轮（复用第五轮的工作目录）**：`prepare`、光流、`poses/` 不变；token 需要重新检测（新的变体目录结构）：

```bash
export LGS_WORK_DIR=work/round6
mkdir -p $LGS_WORK_DIR && cp -r work/round5/{prepare,flow,poses,compat_packages} $LGS_WORK_DIR/
python scripts/detect_tokens.py --config configs/mofang_9cam.yaml
python scripts/run_anysplat.py  --config configs/mofang_9cam.yaml
python scripts/evaluate.py      --config configs/mofang_9cam.yaml
python scripts/validate_results.py configs/mofang_9cam.yaml
python -m unittest discover -s tests -v
```

先看 `summary.md` 的"Token 变体对比"表：目标是 cam3 的 `rife`/`strict` 在 fg 的 PSNR 和 LPIPS 都优于纯光流（接近 oracle 上限 27.0 dB / 0.057），且 rest 回到约 33.5 dB；cam5 贴回区很小、指标接近纯光流。

**第五轮（复用第四轮的工作目录）**：`prepare`、光流、`poses/`、`tokens/` 都不变，只需重跑 GS 和评估：

```bash
export LGS_WORK_DIR=work/round5
mkdir -p $LGS_WORK_DIR && cp -r work/round4/{prepare,flow,poses,tokens,compat_packages} $LGS_WORK_DIR/
python scripts/run_anysplat.py  --config configs/mofang_9cam.yaml
python scripts/evaluate.py      --config configs/mofang_9cam.yaml
python scripts/validate_results.py configs/mofang_9cam.yaml
python -m unittest discover -s tests -v
```

日志里每次编码会打印 `zero-parallax plane fit rms`（越小说明 AnySplat 的输入相机几何与真实单应越一致）和局部高斯数量（visible 模式应明显少于第四轮）。

**第四轮（推荐复用第二轮的工作目录 `work/round2`）**：`prepare`、uniform/pose/oracle 光流和 `poses/<setup>.json`（光流用的 t）都不变，只需检测 token、重跑 GS 和评估。必须复制 `poses/`，这样 `_int` 插值用的 t 与参考光流 `flow_s1.0_pose` 的 t 完全一致：

```bash
export LGS_WORK_DIR=work/round4
mkdir -p $LGS_WORK_DIR && cp -r work/round2/{prepare,flow,poses,compat_packages} $LGS_WORK_DIR/
python scripts/detect_tokens.py --config configs/mofang_9cam.yaml     # GPU 上 1 分钟内
python scripts/run_anysplat.py  --config configs/mofang_9cam.yaml
python scripts/evaluate.py      --config configs/mofang_9cam.yaml
python scripts/validate_results.py configs/mofang_9cam.yaml
```

预计：AnySplat 2 个设置 × 2 个裁剪（token、center）各编码 1 次；每次 2 种位姿（vggt、int）× 整窗口/局部共 4 次渲染。先看 `work/tokens/<setup>/cam{k}_overlay.png`（红 = 遮挡/匹配失败，黄 = 深度突变，蓝 = 重建区）确认 token 是否合理，再看 `summary.md` 开头的"Token 局部 GS 贴回"两张表。

### 各步骤输出（`work/`）

| 步骤 | 主要输出 |
|---|---|
| `prepare.py` | `prepare/H_raw2aligned.npy`、`check_warp_*.png`、`prepare.json`、前景掩码 |
| `run_anysplat.py` | `gs/<setup>/gs_<crop><size>_<render>/cam{k}.png`（含 `_valid`、`_alpha`）、`meta.json`；`gs/<setup>/gs_<crop><size>_poses.json`；`poses/<setup>.json` |
| `run_flow.py` | `flow/<setup>/flow_s<scale>_<tmode>/cam{k}.png`、`cam{k}_signals.npz`、`meta.json` |
| `evaluate.py` | `eval/summary.md`（先看开头"关键对比"表）、`metrics.csv`、`auc.csv`、`registration.csv`、`risk_sweep.csv`、`<setup>/cam{k}_montage.png` |
| `validate_results.py` | `eval/validation.json` |

---

## 4. 需要得到的结果与判断标准

请回传 `work/eval/` 下的 `summary.md`、全部 csv、`validation.json`，`work/logs/`，以及 `cam3` 的 `cam1/2/3/5_montage.png`。

### 先检查

1. AnySplat 统计表中 `in` 变体的输入视点自渲染 PSNR 应与第一轮一致（center448 约 29 dB）；
2. `registration.csv`：`center448_in` 的平均位移就是第一轮"全图错位"的量化值；
3. 拼图中黄色轮廓（object）应套住魔方和手。

### 要回答的问题（第五轮）

| # | 问题 | 看哪里 | 支持框架的结果 |
|---|---|---|---|
| A1 | 对齐域渲染是否消除错位 | 环带位移（`token_ringreg.csv`）与 `+reg` 位移：`_al` / `_alv` vs 无后缀 | 明显小于第四轮（cam3 约 8–11 px） |
| A2 | 贴回足迹是否收紧 | 贴回区占前景比例；`rest` 区 PSNR | 足迹远小于第四轮的 61%，rest 接近纯光流（≈33.5 dB） |
| A3 | 整图是否超过纯光流 | "整图与物体区"表：`hyb_token*[*_al_local]` vs `flow_s1.0_pose` | fg 不降且 object 提升 |
| A4 | 光心插值是否够用 | `_al` vs `_alv` | 接近，说明部署时按 t 插值光心即可 |
| A5 | 零视差平面拟合质量 | 日志 `plane fit rms` | 小（几 px 以内）；若很大，AnySplat 输入相机几何本身与真实阵列不一致 |

### 要回答的问题（第四轮，token 框架）

对比基线：`flow_s1.0_pose`（与 `_int` 共用同一个 t）；cam3 另看 `flow_s0.5_pose`。

| # | 问题 | 看哪里 | 支持框架的结果 |
|---|---|---|---|
| T1 | token 是否标在该标的地方 | `tokens/<setup>/cam{k}_overlay.png`、`tokens.json` | 集中在魔方、手、遮挡带；cam3 多于 cam5 |
| T2 | 贴回区内局部 GS 是否优于光流 | "贴回区内"表：光流 vs `hyb_token` / `hyb_token_ringreg` | PSNR 更高且 LPIPS 更低 |
| T3 | 整图是否变好 | "整图与物体区"表 `fg`、`object` 列 | fg 不降、object 明显提升 |
| T4 | 统一位姿坐标系是否减少错位 | `*_int_local` vs `*_local`（无 `_int`），以及环带位移 | `_int` 更好、环带位移更小 |
| T5 | 环带配准是否有用 | `hyb_token_ringreg` vs `hyb_token` | ringreg 更好 |
| T6 | token 窗口 vs 整窗口 | `gs_token448_*_local` vs `gs_center448_*_local` | 相当或更好（只重建局部即可） |
| T7 | 小基线 | cam5 | 贴回区小、收益小或为负，说明 GS 只应在大基线触发 |

### 要回答的问题（第三轮，已完成）

对比基线一律用**最强光流**：cam3 看 `flow_s0.5_pose`，cam5 看 `flow_s1.0_pose`（第二轮各自最好），不再用均匀 t。

| # | 问题 | 看哪里 | 支持混合方案的结果 |
|---|---|---|---|
| H1 | 标定位姿能否消除错位 | `registration.csv` / 配准残差表：`_cal`、`_calK` vs 无后缀 | 平均位移从 8–10 px 降到 1–2 px |
| H2 | 可部署的 GS 是否优于最强光流 | cam3 关键对比表 `object` 列：`gs_*_cal(K)` vs `flow_s0.5_pose` | PSNR 更高且 LPIPS 更低，接近第二轮 `+reg` 的 17.4–17.6 dB |
| H3 | 可部署的混合是否有效 | `hyb_oracle_fail[gs_*_cal]`、`hyb_risktuned[gs_*_cal]` 在 `fg` | oracle 混合 fg 提升且 LPIPS 不变差；tuned 越接近 oracle，检测器越够用 |
| H4 | 标定内参与畸变是否必要 | `_calK` vs `_cal` | `_calK` 更好，说明 VGGT 内参也有误差；更差则说明高斯本身是按 VGGT 内参构建的，应保持一致 |
| H5 | 修正后的 3D 滤波 | `_nat_mip` vs `_nat_dil` | 不再崩溃；若仍无收益，滤波不是瓶颈 |
| H6 | 时间步与空间位置 | t 估计表、`flow_*_posecal` vs `flow_*_pose` | 若标定 t 比 VGGT t 更差，说明 RIFE 的 t 对空间位置的响应非线性，需要把 t 按网络响应重新标定（例如用留出相机建立查找表），而不是直接用几何比例 |
| H7 | 小基线 | cam5 | 预期光流仍优于 GS，说明 GS 只应在大基线、光流失败处触发 |

**决策**：
- H1 + H2 成立 → "标定一次 + 局部 GS 补全"在可部署条件下成立，下一步是实验 1 的风险检测和 token 局部推理；
- H1 成立但 H2 不成立 → 位姿已不是瓶颈，GS 外观质量不足：考虑以位姿为条件的前馈模型、在本阵列数据上微调，或几何一致的视差匹配 + 多视图 warp；
- H1 不成立 → 错位来自位姿以外（例如 AnySplat 自身输入相机几何与真实阵列不一致，看 AnySplat 统计表中 calib 行的 Sim(3) 残差），需要先定位。

---

## 5. 已知局限

- 只有魔方一个场景、单帧；
- 目标相机位姿由 VGGT 联合估计（目标图像只用于位姿，相当于阵列标定），没有标定板真值；
- `+reg`、`oracle_fail`、`oracle` t、`hyb_risktuned` 的阈值都用到了真值，只能作为诊断或上限；
- Mip-Splatting 的 3D 滤波在这里用输入相机在 448 网格上的采样率，没有经过训练阶段联合优化；
- 魔方检测是基于颜色的启发式，只适用于本场景；其他场景请手动指定 `gs.roi_window`。
