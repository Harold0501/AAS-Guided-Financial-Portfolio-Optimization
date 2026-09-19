"""
快速开始指南
"""

print("="*80)
print("GDELT新闻批量获取脚本 - 快速开始")
print("="*80)

print("""
1. 环境准备
   确保已安装所需依赖：
   pip install pandas yfinance akshare tqdm requests openpyxl

2. 运行主脚本
   python fetch_instance_news.py

3. 脚本功能
   - 自动遍历所有762个instance
   - 为每个instance获取对应股票的GDELT新闻
   - 新闻时间范围：instance开始时间的前一年到开始时间
   - 输出文件：news/instance_{i}_news.csv

4. 预期耗时
   - 762个instance × 平均10个股票 × 1秒延迟
   - 预计总耗时：2-3小时

5. 断点续传
   - 脚本会自动跳过已处理的instance
   - 可以随时中断并重新运行

6. 注意事项
   - GDELT在2015年之前的数据可能不完整
   - 早期instance获取的新闻可能较少或为空
   - 中国公司使用英文名搜索，部分公司可能新闻较少

7. 监控进度
   脚本运行时会显示：
   - [Instance X] 当前处理的instance
   - [X/Y] 当前股票进度
   - [Summary] 成功/失败统计
   - [Failed Tickers] 失败的股票列表

8. 输出格式
   每个CSV包含列：
   - URL: 新闻链接
   - Date: 发布日期
   - Title: 新闻标题
   - keyword: 搜索的公司名
   - ticker: 股票代码

详细文档请查看 README.md
""")

print("="*80)
print("现在可以运行: python fetch_instance_news.py")
print("="*80)
