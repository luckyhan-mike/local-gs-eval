# local-gs-eval：大基线下局部前馈 GS 补全质量验证（实验 3）

本仓库用于回答一个问题：**在光流插视点失败的区域，前馈 3DGS（AnySplat）渲染的结果能不能比光流更好，并且好到值得做"光流为主 + 局部 GS 补全"的混合方案？**

采用留一法：从 9 路会聚相机中只取少数几台作为输入（例如 0/4/8 三台，对应之前 `北极星/2-3/魔方_9` 中魔方/头发出错的实验），被抽掉的真实相机作为真值，分别比较：

- **光流**：Practical-RIFE，用两侧最近的输入相机合成目标相机，逐时刻真实推理（不做 t=0.5 线性外推）；
- **局部 GS**：AnySplat 只用输入相机前馈生成高斯，在目标相机位姿处渲染；
- **混合**：光流结果为底，在指定区域内换成 GS，并羽化边界。

全部比较都在单应对齐后的图像域（即现有光流管线的输入/输出域）中进行。

> 数据说明：魔方场景的实际数据是 **9 路**相机（`北极星/0/魔方_9/grid_00~08` 原图、`北极星/1/魔方_9/aligned_grid_00~08` 对齐图），`北极星/0-3` 的三路就是第 0、4、8 号。相机数量和输入组合都在配置里设置。

---

## 0. 第四轮：按原始框架做 token 局部 GS（当前主实验）

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

## 0.1 第三轮改动（相对第二轮实测）

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

## 0.2 第二轮改动（相对 2026-10-24 第一轮实测）

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
