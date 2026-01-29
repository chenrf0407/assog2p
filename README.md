# assoG2P

**assoG2P** 是一个全面的基因型-表型关联分析工具包，集成了GWAS（全基因组关联分析）和机器学习功能。

## 功能特性

- 🔬 **数据预处理**：支持VCF格式基因型数据和表型数据的预处理
- 🧬 **GWAS分析**：集成GEMMA进行全基因组关联分析
- 🔗 **LD过滤**：支持连锁不平衡（LD）过滤，减少冗余特征
- 🤖 **机器学习模型**：支持多种机器学习模型（LightGBM、XGBoost、RandomForest、SVM、CatBoost、Logistic）
- 📊 **结果可视化**：生成静态和交互式可视化图表
- 🎯 **特征选择**：支持多种特征选择模式（GWAS筛选、LD过滤、综合过滤）

## 系统要求

- Python 3.7+
- Linux系统（推荐）
- 足够的磁盘空间用于存储中间文件

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/assoG2P.git
cd assoG2P
```

### 2. 编译安装（推荐）

使用Makefile一键安装：

```bash
make
```

或者：

```bash
make install
```

这将自动完成：
- 检查Python和pip环境
- 安装所有Python依赖
- 安装项目本身
- 设置软件可执行文件权限
- 验证安装是否成功

### 3. 验证安装

```bash
association --help
```

如果看到帮助信息，说明安装成功。

### 其他安装方式

#### 手动安装

如果不想使用make，也可以手动安装：

```bash
# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
```

#### 仅安装依赖

```bash
pip install -r requirements.txt
```

## 使用示例

### 数据预处理

```bash
association preprocess \
    -g genotype.vcf \
    -p phenotype.csv \
    -o preprocessed_data \
    --threads 4
```

### 模型训练

```bash
# 训练单个模型（使用GWAS和LD综合过滤）
association train \
    -i preprocessed_data/train_data.txt \
    -m LightGBM \
    -f 4 \
    -o results \
    --gwas_genotype preprocessed_data/genotype \
    --gwas_pvalue 5e-8 \
    --ld_window_kb 50 \
    --ld_window_r2 0.2
```

### 训练所有模型并比较

```bash
association train-all \
    -i preprocessed_data/train_data.txt \
    -f 4 \
    -o results \
    --gwas_genotype preprocessed_data/genotype \
    --gwas_pvalue 5e-8
```

### 使用训练好的模型进行预测

```bash
association predict \
    -i new_data.csv \
    -m LightGBM \
    -o predictions
```

### 结果可视化

```bash
# 特征重要性可视化
association visualize \
    -i feature_importance.csv \
    -o plot

# 模型性能可视化
association visualize \
    -f results/LightGBM/plotting_data.npz

# 同时生成两种可视化
association visualize \
    -i feature_importance.csv \
    -o plot \
    -d results/LightGBM
```

## 特征选择模式说明

使用 `-f` 或 `--feature_selection_mode` 参数指定特征选择模式：

- **模式1**：空白对照（不使用GWAS和LD）
- **模式2**：GWAS筛选（仅使用GWAS）
- **模式3**：LD过滤（仅使用LD）
- **模式4**：GWAS和LD综合过滤（先GWAS后LD）

## 支持的模型

- LightGBM
- XGBoost
- RandomForest
- SVM
- CatBoost
- Logistic Regression

## 输入文件格式

### 基因型文件
- VCF格式（`.vcf`）
- PLINK二进制格式（`.bed`, `.bim`, `.fam`）
- PLINK文本格式（`.ped`, `.map`）

### 表型文件
- CSV格式，包含样本ID和表型值
- 默认表型列名为 `phenotype`，可通过 `--pheno-col` 指定其他列名

## 输出文件说明

### 预处理输出
- `train_data.txt`：预处理后的训练数据
- `*_metadata.json`：元数据文件，包含数据信息

### 训练输出
- `plotting_data.npz`：模型性能数据（用于可视化）
- `model.pkl`：训练好的模型文件
- `feature_importance.csv`：特征重要性文件（如果启用）
- `predictions.csv`：预测结果（如果进行预测）

## Makefile 使用说明

项目提供了Makefile来简化安装和管理：

```bash
make          # 安装项目（默认）
make install  # 安装项目
make test     # 测试安装
make clean    # 清理临时文件
make uninstall # 卸载项目
make help     # 显示帮助信息
```

## 常见问题

### Q: 如何指定表型列名？
A: 使用 `--pheno-col` 参数，例如：`--pheno-col trait_value`

### Q: 如何控制并行线程数？
A: 在预处理阶段使用 `--threads` 参数，例如：`--threads 8`

### Q: 如何生成交互式图表？
A: 可视化命令默认会生成静态和交互式图表。如果只需要交互式图表，使用 `--interactive-only` 参数。

### Q: 项目包含哪些外部软件？
A: 项目内置了PLINK和GEMMA的Linux版本，位于 `assoG2P/bin/software/` 目录。

### Q: 安装后找不到association命令？
A: 确保Python的bin目录在PATH环境变量中。可以运行 `which association` 检查命令位置，或使用 `python3 -m assoG2P.main` 作为替代。

## 依赖说明

### 必需依赖
- pandas
- numpy
- scikit-learn
- lightgbm
- xgboost
- catboost
- matplotlib
- seaborn
- scipy

### 可选依赖
- shap（用于特征重要性分析）
- tqdm（用于显示进度条）
- plotly（用于交互式可视化）
- kaleido（用于plotly导出图片）
- psutil（用于内存监控）
- datatable（用于快速读取大文件）

## 许可证

请查看 LICENSE 文件了解详情。

## 作者

- **chenrf** - 12024128035@stu.ynu.edu.cn

## 贡献

欢迎提交Issue和Pull Request！

## 更新日志

### v1.0.0
- 初始版本发布
- 支持数据预处理、模型训练、预测和可视化功能
- 集成GWAS和LD过滤功能
