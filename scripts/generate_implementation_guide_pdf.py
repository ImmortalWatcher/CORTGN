#!/usr/bin/env python3
"""Generate DAMMFND implementation guide PDF (Chinese)."""

from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = ROOT / "DAMMFND_paper_implementation_guide_zh.pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")


class GuidePDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("SimHei", "", str(FONT_PATH))
        self.set_auto_page_break(auto=True, margin=18)

    def section_title(self, title: str):
        self.set_font("SimHei", "", 14)
        self.set_text_color(20, 60, 120)
        self.multi_cell(0, 9, title)
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def sub_title(self, title: str):
        self.set_font("SimHei", "", 12)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 8, title)
        self.ln(1)
        self.set_text_color(0, 0, 0)

    def body(self, text: str):
        self.set_font("SimHei", "", 10)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def code_block(self, text: str):
        self.set_font("SimHei", "", 9)
        self.set_fill_color(245, 245, 245)
        self.multi_cell(0, 5, text, fill=True)
        self.ln(2)


def build_pdf():
    pdf = GuidePDF()
    pdf.add_page()

    pdf.set_font("SimHei", "", 20)
    pdf.multi_cell(0, 12, "DAMMFND 论文实现讲解")
    pdf.ln(2)
    pdf.set_font("SimHei", "", 11)
    pdf.multi_cell(
        0,
        7,
        "论文：Domain-Aware Multimodal Multi-view Fake News Detection (AAAI-25)\n"
        "项目：DAMMFND 代码对照说明\n"
        "生成说明：本文档将论文方法与 src/ 目录代码一一对应。",
    )
    pdf.ln(4)

    sections = [
        (
            "一、论文在解决什么问题？",
            None,
            [
                "虚假新闻检测面临三个核心难点：",
                "1. 领域识别不准确：一条新闻可能同时属于多个领域（如科技+经济），硬贴单一标签会引入噪声。",
                "2. 跨领域数据不均衡：社会类远多于军事类，共享参数容易偏向数据多的领域，造成负迁移。",
                "3. 不同领域对各模态依赖不同：政治更重文字，娱乐更重图片，需要领域感知的决策权重。",
                "",
                "DAMMFND 核心思路：",
                "提取文本、图像、图文融合三个视角的特征 → 将「领域信息」与「新闻语义」解耦 → "
                "用领域信息辅助三视角分别判假 → 按领域动态加权三视角结果得到最终预测。",
            ],
        ),
        (
            "二、整体流程与代码入口",
            None,
            [
                "数据与标签：src/utils/clip_dataloader.py、src/data_pre.py（multi_category 多标签）",
                "图像预处理：src/weibo_build_pkl.py（MAE/CLIP 图像 pkl）",
                "训练启动：src/run_weibo.py → src/run.py",
                "模型与训练：src/model/dammfnd.py（DAMMFNDMODEL + Trainer）",
                "文本变体：src/model/dammfnd_text.py（仅文本三视角）",
                "评估指标：src/utils/utils.py（metricsTrueFalse）",
                "",
                "流程：原始新闻 → BERT/MAE/CLIP 编码 → 三视角 MCFA → 领域解耦 DD → "
                "DAMVD 三视角判别 → DEMD 加权决策 → 最终真假预测。",
            ],
        ),
        (
            "三、模块 0：数据与领域标签（论文 Table 1）",
            "对应：src/utils/clip_dataloader.py、src/run.py",
            [
                "论文将 Weibo/Weibo-21 划分为 9 个领域，支持多标签（一条新闻可属多个领域）。",
                "",
                "代码逻辑：",
                "  - category 列：主领域（如「政治」）",
                "  - 领域 列：第二领域（无则记为「无领域」）",
                "  - 构造 labels_multi_domain：形状 [N, 9] 的 0/1 向量",
                "",
                "领域字典定义在 src/run.py 的 category_dict（weibo 9 类 / weibo21 9 类）。",
                "run_weibo.py 使用 train_aligned.csv 及对应 pkl 文件加载图文数据。",
            ],
        ),
        (
            "三、模块 1：多视角特征提取（论文 §3.1，式 1-5）",
            "对应：src/model/dammfnd.py forward() 第 343-363 行",
            [
                "文本视角：BERT 编码 → MaskAttention 池化 → TextCNN（cnn_extractor）",
                "  代码：self.bert、self.text_attention、self.text_experts",
                "",
                "图像视角：MAE 编码 → TokenAttention 池化 → CNN（cnn_extractor）",
                "  代码：self.image_model（mae_vit_base）、self.image_experts",
                "",
                "多模态视角：CLIP 图文编码 → 拼接 → clip_fuion MLP",
                "  代码：self.ClipModel、self.clip_fusion",
                "",
                "TextCNN 实现：src/model/layers.py 的 cnn_extractor 类。",
                "",
                "与论文差异：论文式(3)(4)用 CLIP 余弦相似度加权融合；代码采用特征拼接 + MLP，思想相近。",
            ],
        ),
        (
            "三、模块 2：多通道特征聚合 MCFA（论文 §3.1，式 6-8）",
            "对应：src/model/dammfnd.py 第 370-475 行",
            [
                "论文：每个模态设 k=18 个通道，门控加权聚合。",
                "",
                "代码：num_expert=6 个私有专家 + num_expert*2=12 个共享专家 = 18 路门控。",
                "  text_gate_list 输出 18 维 softmax 权重，加权混合各专家输出。",
                "",
                "聚合后 att_mlp_text/img/mm 将特征拆为两支：",
                "  feature0 → 后续接 text_classifier（真假相关支路）",
                "  feature1 → 后续接 text_classifier_Mu（9 维领域多标签支路）",
                "",
                "forward 中注释 # Multi-view Features Extraction and Aggregation 即本模块。",
            ],
        ),
        (
            "三、模块 3：领域解耦 DD（论文 §3.2，式 9-12）",
            "对应：src/model/dammfnd.py 第 478-499 行；损失第 593-606 行",
            [
                "论文：领域支路预测多标签领域（L_dom）；语义支路预测应接近均匀分布（L_se，KL 散度）。",
                "",
                "代码：",
                "  text_classifier_Mu（9 维）+ BCE → loss11/21/31  （领域分类，对应 L_dom）",
                "  text_classifier（1 维）+ KL 均匀分布 → loss12/22/32 （语义约束，对应 L_se）",
                "",
                "MLP_Mu 定义在 src/model/layers.py（输出 9 类）。",
                "",
                "与论文差异：论文用 MLP 软掩码分离两支特征；代码用 att_mlp 注意力权重软分离 + 双分类头约束。",
            ],
        ),
        (
            "三、模块 4：领域感知多视角判别器 DAMVD（论文 §3.3，式 13-14）",
            "对应：src/model/dammfnd.py 第 506-517 行",
            [
                "论文：三门控 G_i 过滤领域特征，与语义特征拼接后，文本/图像/多模态各预测一次真假。",
                "",
                "代码：",
                "  gate_text_prefer(fake_news_feature) * text_domain_features  → 式(13)(14) 门控",
                "  domain_aware_text/image/fusion_classifier(语义 + 过滤后领域) → 三视角 sigmoid 预测",
                "",
                "fake_news_feature = 三模态语义支路特征之和（第 503 行）。",
                "各视角辅助 BCE 损失：loss12_aux / loss22_aux / loss32_auc（第 599-601 行）。",
            ],
        ),
        (
            "三、模块 5：领域增强多视角决策层 DEMD（论文 §3.4，式 15-19）",
            "对应：src/model/dammfnd.py 第 19-85 行、第 519-524 行",
            [
                "论文 DAFormer：全局特征 + 领域特征构成 Q，三模态语义特征作 K/V，输出三视角权重后加权融合。",
                "",
                "代码 DomainAwareTransformer 类：",
                "  global_rep = Mean(三模态领域特征)           → 式(15)",
                "  q = q_proj(domain_rep + global_rep)         → 式(16)",
                "  MultiHeadAttention + FFN + LayerNorm        → 式(17)",
                "  weight_proj + softmax → [w_text, w_img, w_mm] → 式(18)",
                "",
                "最终预测（式 19）：",
                "  fake_news_sigmoid = w0*text_view + w1*image_view + w2*fusion_view",
            ],
        ),
        (
            "三、模块 6：总损失（论文式 20）",
            "对应：src/model/dammfnd.py Trainer.train() 第 590-608 行",
            [
                "论文：L = L_final + α_fnd(L_text+L_img+L_mm) + α_dom·L_dom + α_se·L_se，α=0.25",
                "",
                "代码合并写法：",
                "  loss0 = BCE(最终预测, label)                    → L_final",
                "  loss11/21/31 = BCE(领域预测, multi_category)    → L_dom",
                "  loss12/22/32 = KL(语义支路, 均匀分布)          → L_se",
                "  loss12_aux 等 = BCE(各视角预测, label)          → 多视角真假损失",
                "  loss = loss0 + (领域+语义损失)/6 + (视角损失)/3",
            ],
        ),
        (
            "四、一次 forward 数据流（dammfnd.py）",
            None,
            [
                "输入：content/masks(BERT)、image(MAE)、clip_image/clip_text(CLIP)",
                "第 343-363 行：三模态预训练编码",
                "第 370-475 行：MCFA 专家混合 + 双支路拆分",
                "第 478-499 行：领域解耦双分类头",
                "第 506-517 行：DAMVD 三门控 + 三视角预测",
                "第 519-524 行：DEMD 加权融合",
                "第 528 行：返回最终预测及各中间结果（供 loss 计算）",
            ],
        ),
        (
            "五、训练与评估",
            None,
            [
                "run.py 导入：from model.dammfnd import Trainer",
                "run_weibo.py 配置：max_len=197、BERT 路径、batch、lr、save_param_dir",
                "训练循环：Trainer.train() → Adam + BCE + 多任务损失",
                "早停与保存：parameter_dammfnd.pkl",
                "测试：metricsTrueFalse 按领域统计 Accuracy/F1",
            ],
        ),
        (
            "六、论文 vs 代码主要差异",
            None,
            [
                "| 项目       | 论文              | 本仓库代码                    |",
                "| 图文融合   | 余弦相关加权      | CLIP 拼接 + MLP               |",
                "| 领域解耦   | 软掩码 MLP 拆分   | att_mlp 软权重 + 双分类头     |",
                "| 18 通道    | 18 个独立网络     | 6 私有 + 12 共享专家（18 门控）|",
                "| 损失权重   | 显式 α=0.25       | 多损失简单平均                |",
                "",
                "整体架构与论文图 2 一致：特征提取 → 解耦 → 判别 → 决策 四阶段一一对应。",
            ],
        ),
        (
            "七、相关文件速查",
            None,
            [
                "src/model/dammfnd.py      — 主模型 + 训练器（全模态）",
                "src/model/dammfnd_text.py — 文本三视角变体",
                "src/model/layers.py       — cnn_extractor、MLP、MLP_Mu、注意力层",
                "src/run_weibo.py          — Weibo 多模态训练入口",
                "src/run.py                — 数据集配置与 Run 基类",
                "src/weibo_build_pkl.py    — 图像 pkl 构建",
                "src/utils/clip_dataloader.py — 多模态 DataLoader",
                "论文 PDF                  — Domain-aware multimodal multi-view fake news detection.pdf",
            ],
        ),
    ]

    for title, subtitle, paragraphs in sections:
        pdf.section_title(title)
        if subtitle:
            pdf.sub_title(subtitle)
        for para in paragraphs:
            if para.strip() == "":
                pdf.ln(2)
            else:
                pdf.body(para)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT_PDF))
    return OUT_PDF


if __name__ == "__main__":
    path = build_pdf()
    print(f"Saved: {path}")
