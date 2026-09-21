# -*- coding: utf-8 -*-
"""宁波大学科学技术学院招生网咨询爬虫。
抓取招生网"在线咨询"栏目的真实问答，输出为原始文本，可直接用 parse_qa.py 解析。

用法（在你自己的电脑上运行，需要网络）：
  python crawl_ndky.py                          # 默认抓 20 页（200 条）
  python crawl_ndky.py --pages 50               # 抓 50 页（500 条）
  python crawl_ndky.py --pages 100 --output my_raw.txt  # 抓 100 页，自定义输出

注意：
  - 请合理控制抓取页数和请求间隔，避免给学校服务器造成压力
  - 数据仅供个人学习/项目使用，请勿商用或大规模转载
  - 当前 DoubaoWork 运行时网络受限，请在你自己的 PowerShell / 终端中运行此脚本
"""

import argparse
import os
import re
import time

import requests
from bs4 import BeautifulSoup

# 清除可能误设的本地代理环境变量（目标为国内网站，无需代理；残留的死代理会导致 DNS/连接失败）
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_k, None)

BASE_URL = "https://zs.ndky.edu.cn/guestbookconsult"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; xpoi) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# 匹配日期时间行（实际页面中名字和日期分处两行：名字行 / 日期行 / 问题行 / 回答行）
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}$")


def fetch_page(page):
    """抓取第 page 页的 HTML。第 1 页 URL 不带页码，第 2 页起带 _页码。"""
    url = f"{BASE_URL}.jspx" if page == 1 else f"{BASE_URL}_{page}.jspx"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.encoding = resp.apparent_encoding
    return resp.text


def extract_items(html):
    """从页面 HTML 中提取咨询条目，返回 [{name, time, question, answer}, ...]。

    实际页面结构（BeautifulSoup get_text 后，每类信息各占一行）：
        名字
        YYYY-MM-DD HH:MM:SS
        问题（单行）
        回答（单行，偶尔跨多行）
    """
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text("\n")
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # 找到所有日期行的位置
    date_positions = [i for i, line in enumerate(lines) if DATE_RE.match(line)]

    items = []
    for n, pos in enumerate(date_positions):
        # 需要有名字行（pos-1）、问题行（pos+1）、回答行（pos+2）
        if pos < 1 or pos + 2 >= len(lines):
            continue
        name = lines[pos - 1]
        qtime = lines[pos]
        question = lines[pos + 1]
        # 回答从 pos+2 开始，到下一条的名字行（下一日期行-1）之前结束
        if n + 1 < len(date_positions):
            answer_end = date_positions[n + 1] - 1  # 下一条名字行的位置
        else:
            answer_end = len(lines)
        answer_lines = lines[pos + 2:answer_end]
        answer = "\n".join(a for a in answer_lines if a).strip()
        # 过滤异常：名字行不应是日期或过长文本；问题/回答不能为空
        if not name or not question or not answer:
            continue
        if DATE_RE.match(name) or len(name) > 30:
            continue
        items.append({"name": name, "time": qtime, "question": question, "answer": answer})
    return items


def save_raw(items, output_path):
    """保存为原始文本格式（与 parse_qa.py 的输入格式一致）。"""
    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item['name']} {item['time']}\n")
            f.write(f"{item['question']}\n")
            f.write(f"{item['answer']}\n\n")


def main():
    parser = argparse.ArgumentParser(description="宁波大学科学技术学院招生网咨询爬虫")
    parser.add_argument("--pages", type=int, default=20, help="抓取页数上限（默认 20，设大一点配合 --until-date 使用）")
    parser.add_argument("--output", default="raw_qa_full.txt", help="输出文件名")
    parser.add_argument("--delay", type=float, default=1.5, help="每页请求间隔秒数（默认 1.5）")
    parser.add_argument("--until-date", default=None,
                        help="只抓取该日期之后的咨询，格式 YYYY-MM-DD（如 2023-09-01），达到后自动停止")
    args = parser.parse_args()

    from datetime import datetime
    until_dt = datetime.strptime(args.until_date, "%Y-%m-%d") if args.until_date else None

    all_items = []
    stop = False
    print(f"开始抓取（最多 {args.pages} 页），输出到 {args.output}")
    if until_dt:
        print(f"  仅保留 {args.until_date} 之后的咨询，达到后自动停止")
    for page in range(1, args.pages + 1):
        if stop:
            break
        try:
            html = fetch_page(page)
            items = extract_items(html)
            # 按时间过滤
            if until_dt:
                filtered = []
                for item in items:
                    item_time = datetime.strptime(item["time"], "%Y-%m-%d %H:%M:%S")
                    if item_time >= until_dt:
                        filtered.append(item)
                    else:
                        stop = True  # 页面按时间倒序，遇到更早的说明后面都更早了
                items = filtered
            all_items.extend(items)
            print(f"  第 {page:3d} 页：{len(items)} 条（累计 {len(all_items)} 条）")
            time.sleep(args.delay)
        except Exception as e:
            print(f"  第 {page:3d} 页抓取失败：{e}")
            time.sleep(args.delay * 2)

    save_raw(all_items, args.output)
    print(f"\n完成！共抓取 {len(all_items)} 条咨询，已保存到 {args.output}")
    print(f"接下来运行：python parse_qa.py --input {args.output}")


if __name__ == "__main__":
    main()
