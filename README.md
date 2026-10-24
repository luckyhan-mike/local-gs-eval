# local-gs-eval：大基线下局部前馈 GS 补全质量验证（实验 3）

本仓库用于回答一个问题：**在光流插视点失败的区域，前馈 3DGS（AnySplat）渲染的结果能不能比光流更好，并且好到值得做"光流为主 + 局部 GS 补全"的混合方案？**

采用留一法：从 9 路会聚相机中只取少数几台作为输入（例如 0/4/8 三台，对应之前 `北极星/2-3/魔方_9` 中魔方/头发出错的实验），被抽掉的真实相机作为真值，分别比较：

- **光流**：Practical-RIFE，用两侧最近的输入相机合成目标相机，逐时刻真实推理（不做 t=0.5 线性外推）；
- **局部 GS**：AnySplat 只用输入相机前馈生成高斯，在目标相机位姿处渲染；
- **混合**：光流结果为底，在风险区内换成 GS，并羽化边界（可选光度增益校正）。

全部比较都在单应对齐后的图像域（即现有光流管线的输入/输出域）中进行。

> 数据说明：魔方场景的实际数据是 **9 路**相机（`北极星/0/魔方_9/grid_00~08` 原图、`北极星/1/魔方_9/aligned_grid_00~08` 对齐图），`北极星/0-3` 的三路就是第 0、4、8 号。相机数量和输入组合都在配置里设置。

---

## 1. 风险区域怎么定义（不依赖实验 1）

实验 1（风险检测）的结论还没出来，所以本实验对每个区域**都分别评价**：

| 区域 | 定义 | 作用 |
|---|---|---|
| `oracle_fail` | 参考光流与真值的误差 > `eval.fail_err_thresh`（默认 0.08），膨胀 3 px | **核心结论区**：假设检测器完美，GS 能否补对光流算错的地方（上限） |
| `risk` | 只用光流自身信号、固定阈值：\|W₀−W₁\| > 0.08 或双向光流共线残差 > 2 px，膨胀 9 px | 现实可得的简单检测器下，混合方案的效果预览 |
| `edge` | 光流视差的水平梯度 > 0.3 | 深度突变处 |
| `rest` | 前景中除去上面两类的部分 | 检查"平坦区域光流更清晰，不应替换" |
| `fg` | 整个前景（绿幕抠像） | 整体质量 |

光流脚本会保存全部原始信号（\|W₀−W₁\|、共线残差、RIFE mask 熵/梯度、视差梯度）。实验 1 得出更好的信号或阈值后，**只需修改配置的 `risk` 段并重跑 `evaluate.py`**（几分钟），不必重跑 RIFE 和 AnySplat。`risk.extra_rules` 可以增加任意信号的阈值规则，例如 `{mask_ent: 0.9}`。

评估脚本还会顺带输出每个信号预测 `oracle_fail` 的 AUC，可以作为实验 1 的早期参考（仅单场景）。

---

## 2. 环境

在 Linux 3090 主机上，直接使用已经能跑 AnySplat 的 conda 环境（含 torch、gsplat、lpips、scikit-image、opencv、pyyaml）。RIFE 只需要 torch，不另装环境。

```bash
conda activate anysplat            # 换成你的 AnySplat 环境名
pip install lpips scikit-image pyyaml opencv-python   # 已有则跳过
```

需要的外部目录（相对于 `data_root`，可在配置中改为绝对路径）：

- `Practical-RIFE/train_log/IFNet_HDv3.py` 与 `train_log/flownet.pkl`
- `AnySplat/`（源码），权重 `lhjiang/anysplat`（HF 缓存或本地目录，填到 `anysplat_ckpt`）
- `北极星/0/魔方_9/`、`北极星/1/魔方_9/`

## 3. 配置

编辑 `configs/mofang_9cam.yaml`：

- `data_root`：Linux 上对应 Windows `D:\hb` 的目录，默认 `/home/han/linux-han`，**请确认**。也可以不改文件，用环境变量 `LGS_DATA_ROOT=/your/path` 覆盖；
- `setups`：输入相机组合，默认 `cam3: [0,4,8]`（大基线）与 `cam5: [0,2,4,6,8]`（中基线对照）；
- 其余阈值先保持默认。

## 4. 运行

```bash
git clone https://github.com/hhan70701/local-gs-eval.git
cd local-gs-eval
bash run_all.sh configs/mofang_9cam.yaml
```

也可以分步运行（顺序不能换：t_pose 来自 AnySplat 的位姿）：

```bash
python scripts/prepare.py      --config configs/mofang_9cam.yaml
python scripts/run_anysplat.py --config configs/mofang_9cam.yaml --sizes 448 518 --crops center fg
python scripts/run_flow.py     --config configs/mofang_9cam.yaml --scales 1.0 0.5 --tmodes uniform pose oracle
python scripts/evaluate.py     --config configs/mofang_9cam.yaml
```

预计耗时：prepare 约 1 分钟；AnySplat 共 2 个设置 × 4 个变体，每次几秒到几十秒；RIFE 的 oracle 搜索每个目标约 32 次推理，总共约几分钟；evaluate 约 5 分钟。显存方面，518 分辨率下 5 视点应该在 24 GB 以内；如果 OOM，先去掉 `--sizes 518` 或 `cam5`。

### 各步骤做了什么

| 步骤 | 内容 | 主要输出（`work/`） |
|---|---|---|
| `prepare.py` | 用 SIFT + MAGSAC 从"原图 ↔ 对齐图"反求每台相机的单应 H（原图→对齐图）；绿幕抠像得到前景 | `prepare/H_raw2aligned.npy`、`prepare/check_warp_*.png`、`prepare/prepare.json` |
| `run_anysplat.py` | 在原图上按同一窗口裁剪（`center`：中心 1080×1080；`fg`：输入相机前景外接正方形），缩放到 448/518，AnySplat 只用输入相机生成高斯；用一次 VGGT 相机头联合估计全部相机位姿（目标图像只用于这一步，等价于刚性阵列的离线标定），Sim(3) 对齐到高斯坐标系后渲染目标相机，再贴回原图坐标、用 H 变到对齐域 | `gs/<setup>/gs_<crop><size>/cam{k}.png`、`meta.json`、`poses.json`；`poses/<setup>.json` |
| `run_flow.py` | 对每个目标相机，用左右最近的输入相机做 RIFE 合成；t 取 `uniform`（按序号等分，现有做法）、`pose`（按估计光心在基线上的投影）、`oracle`（对真值网格搜索，上限）；scale 取 1.0 和 0.5（0.5 相当于扩大感受野，用于检验大位移） | `flow/<setup>/flow_s<scale>_<tmode>/cam{k}.png`、`cam{k}_signals.npz`、`cam{k}_risk.png` |
| `evaluate.py` | 生成各区域，计算 PSNR/SSIM/LPIPS（LPIPS 在区域外接框内计算，区域外像素用真值填充）；构造混合结果；计算信号 AUC；输出对比拼图 | `eval/summary.md`、`eval/metrics.csv`、`eval/auc.csv`、`eval/<setup>/cam{k}_montage.png` |

混合方法的命名：

- `hyb_oracle[gs_x]`：在 `oracle_fail` 区内换成 GS（完美检测器上限）；
- `hyb_risk[gs_x]`：在 `risk` 区内换成 GS；
- `hyb_risk_gain[gs_x]`：同上，并按外环均值做逐通道光度增益校正。

---

## 5. 需要得到的结果与判断标准

请把 `work/eval/summary.md`、`work/eval/*.csv`、`work/prepare/prepare.json`、`work/logs/` 和几张 `cam3` 的 `cam{k}_montage.png`（尤其是第 1、2、3 号相机，靠近魔方出错的一侧）发回来。

### 先检查（结果是否可信）

1. `prepare.json`：每台相机的 `warp_psnr_db` 应明显高于 30 dB。偏低说明对齐图不是原图的纯单应变换（例如做过锐化或裁剪），需要先看 `check_warp_*.png`；
2. AnySplat 统计表：`输入视点自渲染 PSNR` 反映 GS 拟合能力的上限；`位姿对齐最大旋转误差` 应小于约 1°，`最大光心误差/基线` 应小于约 0.05，否则目标相机位姿不可靠，GS 结果偏低可能是位姿问题，不是 GS 本身的问题；
3. 拼图中 GS 结果与真值没有整体平移或缩放。如果有，说明位姿或 H 出错。

### 要回答的问题

| # | 问题 | 看哪里 | 支持混合方案的结果 |
|---|---|---|---|
| H1 | 光流失败区内 GS 是否更好 | `cam3` 的 `oracle_fail` 表：`gs_*` 与 `flow_s1.0_uniform` | GS 的 PSNR/SSIM 更高、LPIPS 更低，差距越大越好 |
| H2 | 平坦区内光流是否更好 | `rest` 表 | 光流更好（否则应考虑直接全图用 GS） |
| H3 | 混合是否同时优于两者 | `fg` 表：`hyb_*` 与 `flow_*`、`gs_*` | `hyb_oracle` 明显优于两者；`hyb_risk` 越接近 `hyb_oracle`，说明简单检测器越够用 |
| H4 | 基线影响 | `cam3` 与 `cam5` 对比 | `cam3` 中 `oracle_fail` 面积更大、GS 收益更大 |
| H5 | GS 分辨率与裁剪的影响 | `gs_center448 / gs_center518 / gs_fg448 / gs_fg518` | 判断提高分辨率能否缓解"GS 偏软" |
| H6 | 大位移是不是失败主因 | `flow_s0.5_*` 与 `flow_s1.0_*` | 若 scale=0.5 在 `oracle_fail` 上明显更好，说明主因是位移超出感受野 |
| H7 | t 与空间位置是否对应（模块 A 的预验证） | t 估计对比表 | `t_pose` 比 `t_uniform` 更接近 `t_oracle`，且 `flow_*_pose` 优于 `flow_*_uniform` |

**决策**：H1 和 H2 同时成立，并且 `hyb_oracle` 在 `fg` 上有可见提升（如 PSNR +0.5 dB 以上、LPIPS 下降），说明局部 GS 补全值得继续做；如果 H1 不成立（GS 在失败区也不比光流好），应先改进 GS 一侧（以位姿为条件的前馈模型、ROI 高分辨率、微调），或者转向几何一致的视差匹配路线。

---

## 6. 已知局限

- 只有魔方一个场景、单帧，结论需要在挥手等场景上复核（新建一个配置文件即可）；
- 目标相机位姿由 VGGT 联合估计（用到了目标图像，但只用于位姿，相当于阵列标定），没有用标定板真值；
- 对齐图到原图的 H 是反求得到的，若原对齐流程不是纯单应会有残差（见 `prepare.json`）；
- AnySplat 输入为正方形裁剪，`center` 模式会裁掉画面两侧的背景，所有方法都只在 GS 有效区域内比较；
- `oracle_fail` 与 `oracle` t 用到了真值，只能作为上限，不能当作方法本身的结果。
