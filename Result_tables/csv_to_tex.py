import pandas as pd

# * This file is to convert csv files into tex tables
def format_val(x):
    if isinstance(x, float):
        return f"{x:.4f}"   # 控制小数位
    return str(x)

def csv_to_latex(
    csv_path,
    output_tex_path,
    caption="Your Caption",
    label=None
):
    # 读取 CSV
    df = pd.read_csv(csv_path)

    # 获取列名
    columns = df.columns.tolist()

    # 构造 tabular 列格式（自动根据列数）
    col_format = "|" + "|".join(["c"] * len(columns)) + "|"

    latex_lines = []

    # 表头
    latex_lines.append("\\begin{table}[H]")
    latex_lines.append("\\centering")
    latex_lines.append(f"\\caption{{{caption}}}")
    if label:
        latex_lines.append(f"\\label{{{label}}}")

    latex_lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    latex_lines.append("\\hline")

    # 表头行
    header = " & ".join(columns) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\hline")

    # 数据行
    for _, row in df.iterrows():
        row_str = " & ".join([format_val(x) for x in row]) + " \\\\"
        latex_lines.append(row_str)

    # 结束
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    # 写入文件
    with open(output_tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))

    print(f"LaTeX table saved to: {output_tex_path}")



if __name__ == "__main__":
    csv_to_latex(
        csv_path="RF_combined_merged_results.csv",
        output_tex_path="RF_result.tex",
        caption="Random Forest Algorithm Selection Result Summary",
        label="RF Selection Result"
    )

    csv_to_latex(
        csv_path="MLP_combined_merged_results.csv",
        output_tex_path="MLP_result.tex",
        caption="Multilayer Perceptron Algorithm Selection Result Summary",
        label="MLP Selection Result"
    )

    csv_to_latex(
        csv_path="PA_Summary.csv",
        output_tex_path="PA_Summary.tex",
        caption="Portfolio Candidate Algorithm Summary",
        label="Portfolio Algo Summary"
    )

    csv_to_latex(
        csv_path="RF_Algo_Count.csv",
        output_tex_path="RF_Algo_Count.tex",
        caption="Random Forest Algorithm Selection Counts across Feature Sets",
        label="RF Selection Count"
    )

    csv_to_latex(
        csv_path="MLP_Algo_Count.csv",
        output_tex_path="MLP_Algo_Count.tex",
        caption="Multilayer Perceptron Algorithm Selection Counts across Feature Sets",
        label="MLP Selection Count"
    )