# local-gs-eval：大基线下局部前馈 GS 补全质量验证（实验 3）

本仓库用于回答一个问题：**在光流插视点失败的区域，前馈 3DGS（AnySplat）渲染的结果能不能比光流更好，并且好到值得做"光流为主 + 局部 GS 补全"的混合方案？**

采用留一法：从 9 路会聚相机中只取少数几台作为输入（例如 0/4/8 三台，对应之前 `北极星/2-3/魔方_9` 中魔方/头发出错的实验），被抽掉的真实相机作为真值，分别比较：

- **光流**：Practical-RIFE，用两侧最近的输入相机合成目标相机，逐时刻真实推理（不做 t=0.5 线性外推）；
- **局部 GS**：AnySplat 只用输入相机前馈生成高斯，在目标相机位姿处渲染；
- **混合**：光流结果为底，在指定区域内换成 GS，并羽化边界。

全部比较都在单应对齐后的图像域（即现有光流管线的输入/输出域）中进行。

> 数据说明：魔方场景的实际数据是 **9 路**相机（`北极星/0/魔方_9/grid_00~08` 原图、`北极星/1/魔方_9/aligned_grid_00~08` 对齐图），`北极星/0-3` 的三路就是第 0、4、8 号。相机数量和输入组合都在配置里设置。

---

## 0. 第二轮改动（相对 2026-10-24 第一轮实测）

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

**复用第一轮的 `work/`**：`prepare` 和光流结果与第一轮相同，可以只重跑 GS 和评估（约省一半时间）：

```bash
rm -rf work/gs work/eval            # 旧 GS 目录命名不同；即使不删，evaluate 也会自动跳过
python scripts/run_anysplat.py --config configs/mofang_9cam.yaml
python scripts/run_flow.py     --config configs/mofang_9cam.yaml --tmodes pose   # t_pose 来自新的 center448 编码，数值应与第一轮几乎相同
python scripts/evaluate.py     --config configs/mofang_9cam.yaml
python scripts/validate_results.py configs/mofang_9cam.yaml
```

预计：AnySplat 2 个设置 × 2 个裁剪各编码 1 次，每次 5 种渲染（1080² 原生渲染每视点几十毫秒）；evaluate 方法数约 66 个（6 光流 + 20 GS + 40 混合），约 10–20 分钟（ECC 配准与 SSIM 在 CPU 上）。

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

### 要回答的问题

| # | 问题 | 看哪里 | 支持混合方案的结果 |
|---|---|---|---|
| H1 | 物体区 GS 是否优于光流 | 关键对比表 `object` 列，cam3 | 最好的 GS（尤其 roi 变体）PSNR 高于 `flow_s1.0_uniform`，LPIPS 更低 |
| H2 | 滤波核是否有效 | `object`/`rest` 表中同一裁剪下 `_in / _nat / _nat_dil / _nat_mip2d / _nat_mip` 的对比 | `_nat_mip` 或 `_nat_dil` 优于 `_in`（更清晰）且不劣于 `_nat`（无孔洞/锯齿） |
| H3 | ROI 是否优于整幅裁剪 | `gs_roi448_*` vs `gs_center448_*` 在 `object` 区 | roi 更好（放大倍数更小） |
| H4 | 错位是不是主要损失 | `gs_x` vs `gs_x+reg`、`registration.csv` | 若 `+reg` 在 object 区提升数 dB，说明瓶颈是位姿/内参，下一步应换标定板位姿或在线配准 |
| H5 | 混合上限 | `hyb_oracle_fail[gs_x+reg]` vs `flow` 在 `fg` | 明显提升（+0.5 dB 以上且 LPIPS 下降）说明路线在位姿准确时成立 |
| H6 | 现实检测器够不够 | `hyb_risktuned` 与 `hyb_oracle_fail` 的差距；风险扫描表 | 差距小、IoU 高 |
| H7 | 光流侧改进 | `flow_s1.0_pose`、`flow_s0.5_pose` | 与第一轮一致：pose t 和 scale 0.5 有效 |

**决策**：
- H1 成立（尤其 roi + Mip 滤波） → 局部 GS 补全值得继续，下一步重点是位姿精度和检测器；
- H1 不成立但 H4/H5 成立（`+reg` 后成立） → 路线本身可行，瓶颈是目标相机位姿/内参，先解决标定；
- 配准后仍不成立 → GS 本身质量不足，应转向几何一致的视差匹配 + 多视图 warp，或以位姿为条件的前馈模型。

---

## 5. 已知局限

- 只有魔方一个场景、单帧；
- 目标相机位姿由 VGGT 联合估计（目标图像只用于位姿，相当于阵列标定），没有标定板真值；
- `+reg`、`oracle_fail`、`oracle` t、`hyb_risktuned` 的阈值都用到了真值，只能作为诊断或上限；
- Mip-Splatting 的 3D 滤波在这里用输入相机在 448 网格上的采样率，没有经过训练阶段联合优化；
- 魔方检测是基于颜色的启发式，只适用于本场景；其他场景请手动指定 `gs.roi_window`。
