"""
utils/nonmem_utils.py

NONMEM 控制流解析与提取工具。
所有 Skill / Guardrail 共享。
"""

from __future__ import annotations
import re


def extract_code(response: str) -> str:
    """
    从 LLM 响应中提取 NONMEM 代码块。

    支持三种情况：
      1. 完整 markdown 代码块 ```...``` 或 ```nonmem ...```
      2. 无代码块，但从 $PROBLEM 开始到响应结尾
      3. 无代码块无 $PROBLEM，返回全文
    """
    if not response:
        return ""

    # 1) markdown 代码块
    m = re.search(r'```(?:[a-zA-Z]+)?\s*\n?(.*?)\n?```',
                  response, re.DOTALL)
    if m:
        return _strip_fences(m.group(1).strip())

    # 2) $PROBLEM 起点
    m = re.search(r'\$PROBLEM', response, re.IGNORECASE)
    if m:
        return _strip_fences(response[m.start():].strip())

    # 3) 全文兜底
    return _strip_fences(response.strip())


def _strip_fences(code: str) -> str:
    """移除残留的 markdown 围栏行。"""
    if not code:
        return code
    lines = code.split('\n')
    clean = [ln for ln in lines if not re.match(r'^\s*```[a-zA-Z]*\s*$', ln)]
    return '\n'.join(clean).strip()


def validate_input_line(code: str) -> str:
    """
    校验 $INPUT 行是否存在 NONMEM 保留字符误用。
    返回错误描述；无错误返回空字符串。
    """
    m = re.search(r'\$INPUT\s+([^\n]+)', code, re.IGNORECASE)
    if not m:
        return "Missing $INPUT line"

    tokens = m.group(1).split()
    for i, tok in enumerate(tokens):
        # #ID 会丢弃 ID 列
        if tok.startswith('#') and tok[1:].upper() in ('ID', 'SUBJ', 'SUBJECT'):
            return (f"$INPUT contains '{tok}' — the '#' prefix DROPS this "
                    f"column in NONMEM. ID must NOT be dropped (individual "
                    f"ETA and IIV cannot be estimated). Remove the '#' prefix.")
        # DROP ID/TIME/DV/AMT/EVID/MDV
        if tok.upper() == 'DROP' and i + 1 < len(tokens):
            dropped = tokens[i + 1]
            if dropped.upper() in ('ID', 'TIME', 'DV', 'AMT', 'EVID', 'MDV'):
                return (f"$INPUT drops required column '{dropped}' — "
                        f"this column is essential for PopPK estimation.")
    return ""


# --------------------------------------------------------------------------- #
# THETA 索引 → 参数名映射
# --------------------------------------------------------------------------- #

# 参数名识别模式（按优先级：长名优先，避免 V 匹配 V1/V2/V3）
# 每个条目为 (正则片段, 参数名)
_THETA_PARAM_PATTERNS = [
    # TV 前缀（最常见）
    (r'TVCL',     'CL'),
    (r'TVV1',     'V1'),
    (r'TVV2',     'V2'),
    (r'TVV3',     'V3'),
    (r'TVV4',     'V4'),
    (r'TVV',      'V1'),       # TVV 单用视为 V1
    (r'TVQ2',     'Q2'),
    (r'TVQ3',     'Q3'),
    (r'TVQ4',     'Q4'),
    (r'TVQ',      'Q'),        # TVQ 单用视为 Q
    (r'TVKA',     'Ka'),
    (r'TVK',      'Ka'),       # 容错
    (r'TVALAG1',  'ALAG1'),
    (r'TVALAG',   'ALAG1'),
    (r'TVF1',     'F1'),
    (r'TVD1',     'D1'),
    # 无 TV 前缀
    (r'ALAG1',    'ALAG1'),
    (r'ALAG',     'ALAG1'),
    (r'CL',       'CL'),
    (r'V1',       'V1'),
    (r'V2',       'V2'),
    (r'V3',       'V3'),
    (r'V4',       'V4'),
    (r'Q2',       'Q2'),
    (r'Q3',       'Q3'),
    (r'Q4',       'Q4'),
    (r'Q',        'Q'),
    (r'KA',       'Ka'),
    (r'K',        'K'),
    (r'F1',       'F1'),
    (r'V',        'V'),        # 1cmt 中常见
]


def extract_theta_param_map(code: str) -> dict:
    """
    从 NONMEM 控制流中提取 THETA 索引 → 参数名映射。
    参数：
        code : NONMEM 控制流字符串
    返回：
        dict {theta_index (int) : param_name (str)}
        例如：{1: 'CL', 2: 'V1', 3: 'Ka'}
    匹配的赋值形式（大小写不敏感）：
        TVCL = THETA(1)
        TVCL=THETA(1)
        CL = THETA(1) * EXP(ETA(1))
        ALAG1 = THETA(9) * EXP(ETA(9))
        TVV1 = THETA(2) * (WT/70)**0.75

    处理策略：
        1. 优先从 $PK 块内匹配（语义最精确）
        2. 若 $PK 提取失败，回退到全文匹配
        3. 每个 THETA 索引只保留**首次**匹配到的参数名（避免重复）
        4. 参数名识别按**长名优先**，避免 'V' 误匹配 'V1'
    """


    if not code:
        return {}

    # ---- 步骤 1：优先提取 $PK 块 ----
    pk_match = re.search(
        r'\$PK\b\s*\n(.*?)(?=\n\s*\$[A-Z]|\Z)',
        code, re.DOTALL | re.IGNORECASE)
    search_text = pk_match.group(1) if pk_match else code

    # ---- 步骤 2：构建匹配正则 ----
    # 对每个参数名 pattern，生成两种形式：
    #   A) PATTERN = THETA(n)
    #   B) PATTERN = THETA(n) * ...      （后面跟随运算）
    # 使用 (?<![A-Za-z0-9_]) 和 (?![A-Za-z0-9_]) 保证整词匹配
    theta_map = {}

    for param_pattern, param_name in _THETA_PARAM_PATTERNS:
        # 匹配 "PATTERN = THETA(n)" 或 "PATTERN=THETA(n)"
        # 前缀不是字母/数字/下划线；后缀也不能紧跟字母数字（避免 V 命中 V1）
        regex = (
            rf'(?<![A-Za-z0-9_])'
            rf'{param_pattern}'
            rf'(?![A-Za-z0-9_])'          # 参数名后必须不是字母数字
            rf'\s*=\s*'
            rf'THETA\s*\(\s*(\d+)\s*\)'
        )
        for m in re.finditer(regex, search_text, re.IGNORECASE):
            idx = int(m.group(1))
            # 关键：先到先得，不覆盖
            if idx not in theta_map:
                theta_map[idx] = param_name

    return theta_map



def _count_omega(code: str) -> int:
    """
    统计 $OMEGA 块中实际的 OMEGA（随机效应）数量。

    规则：
      - BLOCK(n) 形式：返回 n（不论块体有几行）
      - DIAGONAL / 默认形式：逐行统计数值个数
      - 跳过注释（; 后内容）和空行
      - FIX 关键字不影响计数

    示例：
      $OMEGA BLOCK(3)        → 3
      0.1 0.05 0.1 0.02 0.03 0.1

      $OMEGA                 → 3
      0.1   ; ETA(1)
      0.2   ; ETA(2)
      0.3   ; ETA(3)

      $OMEGA                 → 2
      0.1 0.2 FIX            ; 两个值，FIX 不改变计数
    """
    import re

    m = re.search(r'\$OMEGA\s*(.*?)(?=\n\s*\$|\Z)', code,
                  re.DOTALL | re.IGNORECASE)
    if not m:
        return 0

    body = m.group(1)

    # BLOCK(n) 形式：直接返回维度 n
    block_m = re.search(r'BLOCK\s*\(\s*(\d+)\s*\)', body, re.IGNORECASE)
    if block_m:
        return int(block_m.group(1))

    # DIAGONAL / 默认形式：逐行统计数值个数
    count = 0
    for line in body.split('\n'):
        line = line.split(';')[0].strip()   # 去除注释
        if not line:
            continue
        # 匹配数值（含科学计数法、正负号、小数点）
        nums = re.findall(r'[-+]?\d+\.?\d*(?:[eE][+-]?\d+)?', line)
        count += len(nums)
    return count