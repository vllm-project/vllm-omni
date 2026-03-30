#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import re
import csv

# 覆盖率报告开始标记（可根据实际日志调整）
COVERAGE_START_MARKER = "================================ tests coverage ================================"

def remove_timestamp(line):
    """如果行以 [ 开头，提取 ] 之后的内容；否则返回原行。"""
    line = line.rstrip()
    if line.startswith('['):
        idx = line.find(']')
        if idx != -1:
            return line[idx+1:].lstrip()
    return line

def parse_coverage_report(lines):
    """
    解析覆盖率日志行，提取各字段。
    返回列表，每个元素为 [name, stmts, miss, branch, brpart, cover, missing]
    """
    # 查找覆盖率报告开始标记
    start_idx = None
    for i, line in enumerate(lines):
        if COVERAGE_START_MARKER in line:
            start_idx = i + 1  # 从标记的下一行开始
            break

    if start_idx is not None:
        # 只处理标记之后的内容
        lines = lines[start_idx:]

    # 正则表达式：文件名（非空白字符）、数字列、覆盖率百分比、可选Missing
    # 使用更通用的模式：^(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)%\s*(.*)$
    pattern = re.compile(
        r'^(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)%\s*(.*)$'
    )
    data = []
    for line in lines:
        # 去除时间戳
        line = remove_timestamp(line)
        if not line:
            continue

        # 跳过分隔线（全部由 - 组成）
        if line.replace('-', '').strip() == '':
            continue

        match = pattern.match(line)
        if match:
            groups = match.groups()
            name = groups[0].strip()
            stmts = groups[1]
            miss = groups[2]
            branch = groups[3]
            brpart = groups[4]
            cover = groups[5]
            missing = groups[6].strip() if groups[6] else ''
            data.append([name, stmts, miss, branch, brpart, cover, missing])
        # 可选的调试信息：输出无法匹配的行
        # else:
        #     print(f"忽略行: {line}", file=sys.stderr)
    return data

def write_csv(data, output_file):
    """将数据写入CSV文件"""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Stmts', 'Miss', 'Branch', 'BrPart', 'Cover', 'Missing'])
        writer.writerows(data)

if __name__ == '__main__':
    # 从命令行参数读取文件，或从标准输入读取
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    data = parse_coverage_report(lines)
    write_csv(data, 'coverage_report.csv')
    print(f"已生成 coverage_report.csv，共 {len(data)} 条记录。")