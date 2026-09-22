# -*- coding: utf-8 -*-
"""宁波大学科学技术学院 历年录取分数线爬虫。

抓取招生网"历年信息"栏目下的分省分专业录取表，解析成结构化 JSON。
两种表格结构：
  浙江表：专业名称 | 选考科目 | 最高分 | 最低分 | 平均分 | 位次号
  省外表：省份     | 录取专业 | 科类   | 最高分 | 最低分 | 平均分

输出 data/scores.json：
  [{"year":2025,"province":"浙江","category":"普通类","major":"计算机科学与技术",
    "subject":"物理","max":597,"min":572,"avg":584.5,"rank":null}, ...]

用法：python crawl_scores.py
"""

import json
import os
import re
import time

import requests
from bs4 import BeautifulSoup

for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_k, None)

BASE = "https://zs.ndky.edu.cn/history/"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"}

# (年份, 类别, 文章id, 默认省份标记)
ARTICLES = [
    (2025, "普通类", "18620", "浙江"),
    (2025, "普通类", "18619", "省外"),
    (2025, "专升本", "18613", "浙江"),
    (2024, "普通类", "17350", "浙江"),
    (2024, "普通类", "17349", "省外"),
    (2024, "专升本", "17348", "浙江"),
    (2024, "三位一体", "17347", "浙江"),
    (2023, "普通类", "16297", "浙江"),
    (2023, "普通类", "16296", "省外"),
    (2023, "专升本", "16201", "浙江"),
]


def col_map(header_cells):
    """根据表头关键词确定各列索引。"""
    m = {}
    for i, h in enumerate(header_cells):
        h = h.strip()
        if "专业" in h:
            m["major"] = i
        elif "省份" in h:
            m["province"] = i
        elif "选考" in h or "科类" in h or "类别" in h:
            m["subject"] = i
        elif "最高分" in h or h == "最高":
            m["max"] = i
        elif "最低分" in h or h == "最低":
            m["min"] = i
        elif "平均分" in h or h == "平均":
            m["avg"] = i
        elif "位次" in h:
            m["rank"] = i
    return m


def to_num(s):
    s = (s or "").replace(",", "").strip()
    if not s:
        return None
    try:
        return float(s) if "." in s else int(s)
    except ValueError:
        return None


def parse_table(table, year, category, default_province):
    """解析单个录取表格，返回记录列表。"""
    rows = table.find_all("tr")
    if not rows:
        return []
    header = [c.get_text(strip=True) for c in rows[0].find_all(["td", "th"])]
    m = col_map(header)
    if "major" not in m or "min" not in m:
        return []
    records = []
    for tr in rows[1:]:
        cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        if len(cells) <= max(m.values()):
            continue
        major = cells[m["major"]]
        if not major or "专业" in major:
            continue
        rec = {
            "year": year,
            "province": cells[m["province"]] if "province" in m else default_province,
            "category": category,
            "major": major,
            "subject": cells[m["subject"]] if "subject" in m else "",
            "max": to_num(cells[m["max"]]) if "max" in m else None,
            "min": to_num(cells[m["min"]]),
            "avg": to_num(cells[m["avg"]]) if "avg" in m else None,
            "rank": to_num(cells[m["rank"]]) if "rank" in m else None,
        }
        if rec["min"] is not None:
            records.append(rec)
    return records


def main():
    all_records = []
    for year, category, art_id, default_prov in ARTICLES:
        url = f"{BASE}{art_id}.jhtml"
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.encoding = r.apparent_encoding
            soup = BeautifulSoup(r.text, "lxml")
            n = 0
            for table in soup.find_all("table"):
                n += len(parse_table(table, year, category, default_prov))
            all_records.extend(
                [dict(r, _src=art_id) for r in []]  # placeholder
            )
            print(f"  {year} {default_prov} {category}: {n} 条")
            # 上面 parse 结果直接加入
            for table in soup.find_all("table"):
                for rec in parse_table(table, year, category, default_prov):
                    all_records.append(rec)
            time.sleep(1.0)
        except Exception as e:
            print(f"  {year} {default_prov} {category} 失败: {e}")

    out = os.path.join("data", "scores.json")
    os.makedirs("data", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    print(f"\n完成！共 {len(all_records)} 条录取记录，已保存到 {out}")


if __name__ == "__main__":
    main()
