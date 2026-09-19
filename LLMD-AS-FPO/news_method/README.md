# GDELT新闻批量获取脚本

## 功能说明

`fetch_instance_news.py` 根据 `time_range.xlsx` 中的每个instance，自动获取其对应asset中所有股票的GDELT新闻数据。

### 主要功能

1. **股票代码转公司名**：
   - 美股：使用 yfinance 获取公司长名称
   - 中国股票：优先使用预设的英文名映射，否则使用 akshare 获取中文名

2. **新闻获取**：
   - 数据源：GDELT Project API
   - 时间范围：每个instance开始时间的前一年到开始时间
   - 重试机制：失败自动重试3次，每次间隔1秒

3. **数据保存**：
   - 输出路径：`LLMD-AS-FPO\news_method\news\instance_{i}_news.csv`
   - 每个instance的所有股票新闻合并到一个CSV文件

### 文件结构

```
LLMD-AS-FPO/news_method/
├── asset_components.xlsx    # 每个asset对应的股票代码列表
├── time_range.xlsx          # 每个instance的asset和时间区间
├── fetch_instance_news.py   # 主脚本
└── news/                    # 输出目录
    ├── instance_1_news.csv
    ├── instance_2_news.csv
    └── ...
```

### 使用方法

```bash
cd "d:\郑皓元\life at DKU\July 7 - August 31, 2025\LLMD-AS-FPO\news_method"
python fetch_instance_news.py
```

### 输出CSV格式

每个CSV文件包含以下列：
- `URL`: 新闻链接
- `MobileURL`: 移动端链接
- `Date`: 发布日期
- `Title`: 新闻标题
- `keyword`: 搜索使用的公司名
- `ticker`: 股票代码

### 注意事项

1. **GDELT数据覆盖**：
   - 2015年之前的数据可能不完整
   - 中国公司的英文新闻报道可能较少
   - 脚本会处理所有instance，早期数据获取的新闻可能为空

2. **中国公司英文名映射**：
   目前已映射的公司：
   - 600519: Kweichow Moutai (贵州茅台)
   - 000333: Midea Group (美的集团)
   - 300750: Contemporary Amperex Technology (宁德时代)
   - 600030: CITIC Securities (中信证券)
   - 600036: China Merchants Bank (招商银行)
   - 600900: China Yangtze Power (长江电力)
   - 601166: Industrial Bank (兴业银行)
   - 601318: Ping An Insurance (中国平安)
   - 601899: Zijin Mining Group (紫金矿业)
   - 000858: Wuliangye Yibin (五粮液)

3. **错误处理**：
   - 单个股票获取失败会跳过并记录日志
   - 已处理的instance会跳过（检查文件是否存在）
   - 即使没有新闻也会创建空CSV文件

4. **性能**：
   - 每个股票请求后会sleep 1秒，避免API限流
   - 762个instance，平均每个10个股票，预计总耗时约2-3小时

### 扩展映射

如需添加更多中国公司英文名映射，编辑脚本中的 `CHINESE_COMPANY_ENGLISH_NAMES` 字典：

```python
CHINESE_COMPANY_ENGLISH_NAMES = {
    "股票代码": "英文公司名",
    # 例如：
    "000001": "Ping An Bank",
}
```

### 监控进度

脚本运行时会显示：
- 当前处理的instance编号
- 每个股票的处理状态
- 成功/失败统计
- 失败的股票列表

### 故障排查

如果遇到问题：

1. **ImportError**: 安装依赖
   ```bash
   pip install pandas yfinance akshare tqdm requests openpyxl
   ```

2. **网络错误**: 检查网络连接，GDELT API需要访问互联网

3. **编码问题**: 输出使用utf-8-sig编码，支持中文

4. **中断恢复**: 重新运行脚本会自动跳过已处理的instance
