"""
skills/phase4_optimize_iiv.py

Phase 4: 随机效应/IIV 优化。

职责：
1. 根据 shrinkage 取舍 ETA
2. 小样本优先 diagonal Ω
3. 调整 THETA bound / 初值
4. 用 _build_parameter_stabilization_guidance 的区间收紧搜索
5. 禁止加协变量、禁止改 ADVAN
"""
import re
from .base import Skill, SkillResult, CompartmentLockError
from modules.prompts.phase4_optimize import Phase4Optimize
from utils.nonmem_utils import extract_code as _extract_code
from utils.nonmem_utils import _count_omega

class OptimizeIIVSkill(Skill):
    name = 'phase4_optimize_iiv'
    phase = [4]
    # tools = ['omega_editor', 'theta_bound_tightener',
    #          'nonmem_runner', 'composite_scorer']
    tools = ['nonmem_runner', 'composite_scorer']

    prompt_template = 'phase4_optimize_iiv'

    def run(self, state, llm, tools) -> SkillResult:

        parsed = state.last_run.get('parsed_data', {})
        omega_count = _count_omega(state.current_code)

        # ★ 新增：若所有 ETA shrinkage >95%，直接简化 OMEGA 到 1（保留 CL）
        max_shrink = max(
            (s['shrinkage'] for s in parsed.get('eta_shrinkage', [])),
            default=None)

        if max_shrink is not None and max_shrink > 95 and omega_count > 1:
            print(f"  [PHASE4-OMEGA] All ETA collapse (max shrinkage "
                  f"{max_shrink:.1f}%) — reducing OMEGA to 1 (CL only)")
            new_code = _collapse_omega_to_cl_only(state.current_code)
            state.failed_strategies.append("PHASE4_OMEGA_COLLAPSED_TO_CL_ONLY")
            return SkillResult(updates={'current_code': new_code})

        # ★ 高 shrinkage 的 ETA 自动移除（在调用 LLM 之前）
        # 条件：shrinkage > 75% 且 OMEGA 数量 > 1
        # 动作：直接从 $OMEGA 和 $PK 中移除该 ETA
        removed = self._auto_remove_high_shrinkage_omega(
            state, parsed, omega_count)

        if removed:
            print(f"  [AUTO-REMOVE] Removed high-shrinkage ETA(s): {removed}")
            return SkillResult(updates={
                'current_code': state.current_code,
                'auto_action': {
                    'type': 'OMEGA_AUTO_REMOVED',
                    'removed': removed,
                },
            })


        prompt = Phase4Optimize.generate_prompt(
            iteration=state.iteration,
            current_code=state.current_code,
            parsed_results=state.last_run.get('parsed_data', {}),
            shrinkage_data=state.last_run.get('shrinkage', []),
            current_omega_count=_count_omega(state.current_code),
            n_subjects=state.data_profile['n_subjects'],
        )
        prompt = self._prepend_warning(prompt, state)  # ★
        response = llm.generate(prompt, model_type=state.model)
        new_code = _extract_code(response)

        # ★ 强制房室锁（替代原来的 _structure_unchanged 简易校验）
        try:
            new_code = self._enforce_compartment_lock(
                state, new_code, llm, prompt)
        except CompartmentLockError as e:
            print(f"  [PHASE4-LOCK-FAIL] {e}")
            return self.fallback(state, str(e))

        # 至少保留 1 个 OMEGA
        if _count_omega(new_code) < 1:
            return self.fallback(state, "Phase 4 would remove the last OMEGA — refused.")

        # 房室不得变化
        if not _structure_unchanged(state.current_code, new_code):
            return self.fallback(state, "Phase 4 changed ADVAN — refused.")

        return SkillResult(updates={'current_code': new_code})


    def _auto_remove_high_shrinkage_omega(self, state, parsed, omega_count):
        """
        当某个 ETA 的 shrinkage > 75% 且 OMEGA 数 > 1 时，自动移除该 ETA。

        返回被移除的 ETA 编号列表；无移除返回 []。
        """

        if omega_count <= 1:
            return []

        shrinkages = parsed.get('eta_shrinkage', []) or []
        # 收集高 shrinkage ETA 索引（1-based）
        high_shrink = []
        for entry in shrinkages:
            if not isinstance(entry, dict):
                continue
            v = entry.get('shrinkage')
            eta_idx = entry.get('eta')
            if v is None or eta_idx is None:
                continue
            if float(v) > 75.0:
                high_shrink.append(int(eta_idx))

        # ★ 新增：若所有 ETA 都 >90%，保留 CL（通常是第 1 个）
        if len(high_shrink) == omega_count and omega_count > 1:
            print(f"  [OMEGA-ALL-COLLAPSED] All {omega_count} ETA have >90% "
                  f"shrinkage — keeping only ETA(1) (typically CL)")
            high_shrink = list(range(2, omega_count + 1))

        if not high_shrink:
            return []

        # 移除所有高 shrinkage ETA 对应的 OMEGA 行
        code = state.current_code
        omega_block_match = re.search(
            r'(\$OMEGA[^\$]*?)(?=\n\s*\$|\Z)', code, re.DOTALL | re.IGNORECASE)
        if not omega_block_match:
            return []

        omega_block = omega_block_match.group(1)
        lines = omega_block.split('\n')
        kept_lines = []
        removed_indices = []
        current_omega_idx = 0
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.upper().startswith('$OMEGA'):
                kept_lines.append(line)
                continue
            # 数这一行的 OMEGA 值个数
            nums = re.findall(r'\d+\.?\d*(?:[eE][+-]?\d+)?',
                              stripped.split(';')[0])
            if not nums:
                kept_lines.append(line)
                continue
            # 简化为每行一个 OMEGA 的常见写法
            current_omega_idx += 1
            if current_omega_idx in high_shrink:
                removed_indices.append(current_omega_idx)
                continue
            kept_lines.append(line)

        if not removed_indices:
            return []

        new_omega_block = '\n'.join(kept_lines)
        new_code = code.replace(omega_block, new_omega_block)

        # 同步从 $PK 中移除对应的 ETA 项（若为 IIV on X 形式）
        for eta in removed_indices:
            new_code = re.sub(
                rf'\s*\*?\s*EXP\s*\(\s*ETA\s*\(\s*{eta}\s*\)\s*\)',
                '', new_code, count=1, flags=re.IGNORECASE)
            new_code = re.sub(
                rf'\s*\+\s*ETA\s*\(\s*{eta}\s*\)\s*\*\s*THETA\s*\([^\)]+\)',
                '', new_code, count=1, flags=re.IGNORECASE)

        state.current_code = new_code
        state.failed_strategies.append(
            f"OMEGA_AUTO_REMOVED_ETA_{'_'.join(map(str, removed_indices))}_HIGH_SHRINKAGE")
        return removed_indices




    def fallback(self, state, error: str) -> SkillResult:
        return SkillResult(ok=False, error=error,
                           updates={'current_code': state.current_code})

'''
def _collapse_omega_to_cl_only(code: str) -> str:
    """
    当所有 ETA 塌缩时，保留 CL 的 IIV，移除其余所有 ETA。

    策略：
      1. 将 $OMEGA 块替换为单个 CL 的方差
      2. $PK 中只保留 CL 上的 EXP(ETA(1))
      3. 其余参数的 IIV 项删除
    """
    # 1) $OMEGA 替换为单个 CL IIV
    omega_match = re.search(
        r'(\$OMEGA[^\$]*?)(?=\n\s*\$|\Z)', code, re.DOTALL | re.IGNORECASE)
    if omega_match:
        new_omega = "$OMEGA\n0.1   ; IIV on CL only (all others collapsed)\n"
        code = code.replace(omega_match.group(0), new_omega, 1)

    # 2) $PK 中保留 CL 的 IIV，删除其他 IIV
    #    CL = TVCL * EXP(ETA(1)) 保留
    #    V  = TVV  * EXP(ETA(2)) → V = TVV
    #    Ka = TVKA * EXP(ETA(3)) → Ka = TVKA
    pk_match = re.search(
        r'(\$PK[^\$]*?)(?=\n\s*\$|\Z)', code, re.DOTALL | re.IGNORECASE)
    if pk_match:
        pk_block = pk_match.group(0)
        new_pk = pk_block
        # 删除非 CL 参数的 IIV
        for param in ('V1', 'V2', 'V3', 'V', 'Q', 'Q2', 'Q3', 'Q4', 'Ka', 'KA'):
            new_pk = re.sub(
                rf'({param}\s*=\s*[A-Z0-9_]+)\s*\*\s*EXP\s*\(\s*ETA\s*\(\s*\d+\s*\)\s*\)',
                r'\1',
                new_pk, flags=re.IGNORECASE)
        code = code.replace(pk_block, new_pk, 1)

    return code

'''

'''
def _count_omega(code: str) -> int:
    """统计 $OMEGA 块中实际的 OMEGA 数量（BLOCK 形式按其维度计数）"""
    m = re.search(r'\$OMEGA\s*(.*?)(?=\n\s*\$|\Z)', code, re.DOTALL | re.IGNORECASE)
    if not m:
        return 0
    block_body = m.group(1)

    # 检测 BLOCK(n) 形式
    block_m = re.search(r'BLOCK\s*\(\s*(\d+)\s*\)', block_body, re.IGNORECASE)
    if block_m:
        return int(block_m.group(1))

    # DIAGONAL / 默认形式：逐行统计数值
    count = 0
    for line in block_body.split('\n'):
        line = line.split(';')[0].strip()
        if not line:
            continue
        # 匹配 FIX、SAME 等关键字后的数值
        nums = re.findall(r'[-+]?\d+\.?\d*(?:[eE][+-]?\d+)?', line)
        count += len(nums)
    return count
'''
'''
block = re.search(r'\$OMEGA\s*(.*?)(?=\n\s*\$|\Z)', code, re.DOTALL | re.IGNORECASE)
if not block:
    return 0
count = 0
for line in block.group(1).split('\n'):
    line = line.split(';')[0].strip()
    if line:
        count += len(re.findall(r'\d+\.?\d*(?:[eE][+-]?\d+)?', line))
return count
'''


def _structure_unchanged(old_code: str, new_code: str) -> bool:
    def _advan(code):
        m = re.search(r'ADVAN(\d+)', code, re.IGNORECASE)
        return m.group(1) if m else None
    return _advan(old_code) == _advan(new_code)

'''
def _collapse_omega_to_cl_only(code: str) -> str:
    """
    当所有 ETA 塌缩时，保留 CL 的 IIV，移除其余所有 ETA。

    策略：
      1. 解析 $PK，建立 ETA 编号 → LHS 变量名 的映射（跳过注释）
      2. 找到 CL 对应的 ETA 编号；若找不到，回退到最小 ETA 编号
      3. 删除 $PK 中所有非 CL 的 EXP(ETA(n)) 项（* / + / 三种形式）
      4. 把 CL 的 ETA 引用重新编号为 ETA(1)
      5. 从原 $OMEGA 中提取 CL ETA 的方差，替换为单个值
         （BLOCK 形式无法提取 → 用默认 0.1）
    """
    # ---- 1. 解析 ETA → LHS 变量名（跳过注释行） -------------------- #
    # 剥离注释（; 后内容），避免注释里的伪代码被误提取
    code_no_comments = re.sub(r';[^\n]*', '', code)
    eta_param_map = {}   # {eta_idx: LHS_UPPER}
    for m in re.finditer(
            r'^\s*(\w+)\s*=\s*[^=\n]*?EXP\s*\(\s*ETA\s*\(\s*(\d+)\s*\)',
            code_no_comments, re.IGNORECASE | re.MULTILINE):
        eta_idx = int(m.group(2))
        eta_param_map.setdefault(eta_idx, m.group(1).upper())

    if not eta_param_map:
        # $PK 里根本没有 ETA —— 无需简化
        print(f"  [OMEGA-COLLAPSE] No ETA found in $PK — skipping collapse")
        return code

    # ---- 2. 定位 CL 的 ETA（含兜底） ------------------------------ #
    cl_eta = None
    CL_ALIASES = ('CL', 'TVCL', 'CLPOP', 'CL_POP')
    for eta, param in eta_param_map.items():
        if param in CL_ALIASES:
            cl_eta = eta
            break

    if cl_eta is None:
        # 兜底：保留编号最小的 ETA（通常是 ETA(1)）
        cl_eta = min(eta_param_map.keys())
        print(f"  [OMEGA-COLLAPSE] No CL ETA found — anchoring on "
              f"ETA({cl_eta}) (LHS='{eta_param_map[cl_eta]}')")

    # ---- 3. 删除所有非 CL 的 EXP(ETA(n)) 项 ---------------------- #
    # 处理三种出现形式（按出现频次排序）
    deletion_patterns = [
        r'\s*\*\s*EXP\s*\(\s*ETA\s*\(\s*{eta}\s*\)\s*\)',   # * EXP(ETA(n))  [最常见]
        r'\s*\+\s*EXP\s*\(\s*ETA\s*\(\s*{eta}\s*\)\s*\)',   # + EXP(ETA(n))
        r'\s*/\s*EXP\s*\(\s*ETA\s*\(\s*{eta}\s*\)\s*\)',    # / EXP(ETA(n))
    ]
    removed_etas = []
    for eta in list(eta_param_map.keys()):
        if eta == cl_eta:
            continue
        removed_any = False
        for pat in deletion_patterns:
            new_code = re.sub(
                pat.format(eta=eta), '', code,
                count=1, flags=re.IGNORECASE)
            if new_code != code:
                code = new_code
                removed_any = True
                break   # 同一 ETA 只删一次
        if removed_any:
            removed_etas.append(eta)

    # ---- 4. CL 的 ETA 重新编号为 ETA(1) --------------------------- #
    # 保证 $PK 引用的 ETA 编号与 $OMEGA 的单值位置一致
    if cl_eta != 1:
        code = re.sub(
            rf'ETA\s*\(\s*{cl_eta}\s*\)',
            'ETA(1)', code, flags=re.IGNORECASE)
        print(f"  [OMEGA-COLLAPSE] Renumbered ETA({cl_eta}) → ETA(1) "
              f"(CL anchor)")

    # ---- 5. 从原 $OMEGA 中提取 CL ETA 的方差 ---------------------- #
    omega_match = re.search(
        r'(\$OMEGA[^\$]*?)(?=\n\s*\$|\Z)', code,
        re.DOTALL | re.IGNORECASE)
    cl_omega_val = '0.1'    # 默认值
    if omega_match:
        omega_body = omega_match.group(1)
        if 'BLOCK' in omega_body.upper():
            print(f"  [OMEGA-COLLAPSE] $OMEGA is BLOCK — using default 0.1 "
                  f"for ETA(1)")
        else:
            nums = re.findall(
                r'[-+]?\d+\.?\d*(?:[eE][+-]?\d+)?', omega_body)
            if len(nums) >= cl_eta:
                cl_omega_val = nums[cl_eta - 1]

    # ---- 6. 替换 $OMEGA 块为单个值 -------------------------------- #
    new_omega = (
        f"$OMEGA\n"
        f"{cl_omega_val}   "
        f"; IIV on CL only (ETA(1), collapsed from ETA({cl_eta}))\n"
    )
    if omega_match:
        code = code.replace(omega_match.group(0), new_omega, 1)

    if removed_etas:
        print(f"  [OMEGA-COLLAPSE] Removed ETA(s): {removed_etas}; "
              f"kept ETA(1) (CL), $OMEGA value = {cl_omega_val}")

    return code

'''

'''
def _collapse_omega_to_cl_only(code: str) -> str:
    # 1. 解析 $PK 中每个 ETA 对应的参数
    eta_param_map = {}
    for m in re.finditer(r'(\w+)\s*=\s*[^\n]*EXP\s*\(\s*ETA\s*\(\s*(\d+)\s*\)', code):
        eta_param_map[int(m.group(2))] = m.group(1)
    
    # 2. 找到 CL 对应的 ETA 编号
    cl_eta = None
    for eta, param in eta_param_map.items():
        if param.upper() in ('CL', 'TVCL'):
            cl_eta = eta
            break
    
    # 3. 删除所有非 CL 的 EXP(ETA(n)) 项
    for eta in list(eta_param_map.keys()):
        if eta != cl_eta:
            code = re.sub(
                rf'\s*\*?\s*EXP\s*\(\s*ETA\s*\(\s*{eta}\s*\)\s*\)',
                '', code, count=1)
    
    # 4. $OMEGA 替换为保留 CL ETA 的单个方差
    omega_match = re.search(r'(\$OMEGA[^\$]*?)(?=\n\s*\$|\Z)', code, 
                            re.DOTALL | re.IGNORECASE)
    if omega_match:
        new_omega = f"$OMEGA\n0.1   ; IIV on CL only (ETA({cl_eta}))\n"
        code = code.replace(omega_match.group(0), new_omega, 1)
    
    return code
'''

def _collapse_omega_to_cl_only(code: str) -> str:
    """
    当所有 ETA 塌缩时，保留 CL 的 IIV，移除其余所有 ETA。

    策略：
      1. 解析 $PK，建立 ETA 编号 → LHS 变量名 的映射（跳过注释）
      2. 找到 CL 对应的 ETA 编号；若找不到，回退到最小 ETA 编号
      3. 删除 $PK 中所有非 CL 的 EXP(ETA(n)) 项（* / + 三种形式）
      4. 把 CL 的 ETA 引用重新编号为 ETA(1)
      5. 从原 $OMEGA 中提取 CL ETA 的方差，替换为单个值
    """
    import re

    # ---- 1. 解析 ETA → LHS 变量名（跳过注释行） -------------------- #
    code_no_comments = re.sub(r';[^\n]*', '', code)

    eta_param_map = {}
    for m in re.finditer(
            r'^\s*(\w+)\s*=\s*[^=\n]*?EXP\s*\(\s*ETA\s*\(\s*(\d+)\s*\)',
            code_no_comments, re.IGNORECASE | re.MULTILINE):
        eta_idx = int(m.group(2))
        eta_param_map.setdefault(eta_idx, m.group(1).upper())

    if not eta_param_map:
        print(f"  [OMEGA-COLLAPSE] No ETA found in $PK — skipping collapse")
        return code

    # ---- 2. 定位 CL 的 ETA（含兜底） ------------------------------ #
    cl_eta = None
    CL_ALIASES = ('CL', 'TVCL', 'CLPOP', 'CL_POP')
    for eta, param in eta_param_map.items():
        if param in CL_ALIASES:
            cl_eta = eta
            break

    if cl_eta is None:
        cl_eta = min(eta_param_map.keys())
        print(f"  [OMEGA-COLLAPSE] No CL ETA found — anchoring on "
              f"ETA({cl_eta}) (LHS='{eta_param_map[cl_eta]}')")

    # ---- 3. 删除所有非 CL 的 EXP(ETA(n)) 项 ---------------------- #
    deletion_patterns = [
        r'\s*\*\s*EXP\s*\(\s*ETA\s*\(\s*{eta}\s*\)\s*\)',
        r'\s*\+\s*EXP\s*\(\s*ETA\s*\(\s*{eta}\s*\)\s*\)',
        r'\s*/\s*EXP\s*\(\s*ETA\s*\(\s*{eta}\s*\)\s*\)',
    ]
    removed_etas = []
    for eta in list(eta_param_map.keys()):
        if eta == cl_eta:
            continue
        removed_any = False
        for pat in deletion_patterns:
            new_code = re.sub(pat.format(eta=eta), '', code,
                              count=1, flags=re.IGNORECASE)
            if new_code != code:
                code = new_code
                removed_any = True
                break
        if removed_any:
            removed_etas.append(eta)

    # ---- 4. CL 的 ETA 重新编号为 ETA(1) --------------------------- #
    if cl_eta != 1:
        code = re.sub(
            rf'ETA\s*\(\s*{cl_eta}\s*\)',
            'ETA(1)', code, flags=re.IGNORECASE)
        print(f"  [OMEGA-COLLAPSE] Renumbered ETA({cl_eta}) → ETA(1)")

    # ---- 5. 从原 $OMEGA 中提取 CL ETA 的方差 ---------------------- #
    omega_match = re.search(
        r'(\$OMEGA[^\$]*?)(?=\n\s*\$|\Z)', code,
        re.DOTALL | re.IGNORECASE)
    cl_omega_val = '0.1'
    if omega_match:
        omega_body = omega_match.group(1)
        if 'BLOCK' in omega_body.upper():
            print(f"  [OMEGA-COLLAPSE] $OMEGA is BLOCK — using default 0.1")
        else:
            nums = re.findall(
                r'[-+]?\d+\.?\d*(?:[eE][+-]?\d+)?', omega_body)
            if len(nums) >= cl_eta:
                cl_omega_val = nums[cl_eta - 1]

    # ---- 6. 替换 $OMEGA 块为单个值 -------------------------------- #
    new_omega = (
        f"$OMEGA\n"
        f"{cl_omega_val}   "
        f"; IIV on CL only (ETA(1), collapsed from ETA({cl_eta}))\n"
    )
    if omega_match:
        code = code.replace(omega_match.group(0), new_omega, 1)

    if removed_etas:
        print(f"  [OMEGA-COLLAPSE] Removed ETA(s): {removed_etas}; "
              f"kept ETA(1) (CL), $OMEGA value = {cl_omega_val}")

    return code



