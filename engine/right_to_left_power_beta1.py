# coding: utf-8
"""自包含的周K线力量、趋势与预处理统一引擎。"""

from __future__ import annotations

import math
from numbers import Number
from typing import Iterable

import pandas as pd


_NEUTRAL_DIRECTION = {
    "traditional_direction": "none",
    "corrected_direction": "none",
    "is_direction_reversed": False,
    "outline_color": "",
}


def classify_weekly_kline_direction(open_price, high, low, close):
    """按实体方向和K线中点修正周K线方向。"""
    values = (open_price, high, low, close)
    if any(
        isinstance(value, bool) or not isinstance(value, Number)
        for value in values
    ):
        return _NEUTRAL_DIRECTION.copy()
    try:
        open_price, high, low, close = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError):
        return _NEUTRAL_DIRECTION.copy()
    if not all(math.isfinite(value) for value in (open_price, high, low, close)):
        return _NEUTRAL_DIRECTION.copy()
    if high < low:
        return _NEUTRAL_DIRECTION.copy()

    midpoint = (high + low) / 2.0
    # 中点规则是一个精确边界：只在收盘价严格等于 (H + L) / 2 时向上。
    # 不使用容差，避免略低于中点的阴线被误判为向上。
    at_midpoint = close == midpoint
    if at_midpoint:
        traditional = "up" if close > open_price else "down" if close < open_price else "none"
        corrected = "up"
    elif close == open_price:
        return _NEUTRAL_DIRECTION.copy()
    elif close > open_price:
        corrected = "up" if close > midpoint else "down"
        traditional = "up"
    else:
        corrected = "down" if close < midpoint else "up"
        traditional = "down"
    reversed_direction = corrected != traditional
    return {
        "traditional_direction": traditional,
        "corrected_direction": corrected,
        "is_direction_reversed": reversed_direction,
        "outline_color": (
            "#22c55e"
            if reversed_direction and traditional == "up"
            else "#ef4444" if reversed_direction else ""
        ),
    }


def _band_value_inside_kline(low, high, band):
    try:
        low_value, high_value, band_value = float(low), float(high), float(band)
    except (TypeError, ValueError, OverflowError):
        return False
    if any(pd.isna(value) for value in (low_value, high_value, band_value)):
        return False
    return low_value <= band_value <= high_value


def _high_touches_upper_band(high, b_up):
    try:
        high_value, band_value = float(high), float(b_up)
    except (TypeError, ValueError, OverflowError):
        return False
    return (
        not (pd.isna(high_value) or pd.isna(band_value))
        and high_value >= band_value
    )


def _low_touches_lower_band(low, b_down):
    try:
        low_value, band_value = float(low), float(b_down)
    except (TypeError, ValueError, OverflowError):
        return False
    return (
        not (pd.isna(low_value) or pd.isna(band_value))
        and low_value <= band_value
    )


def _row_has_valid_boll(row):
    """该行 BOLL(26) 三轨是否均已生成（非 NaN 且有限）。"""
    try:
        values = (
            float(row.get("b_up")),
            float(row.get("b_mid")),
            float(row.get("b_down")),
        )
    except (TypeError, ValueError, OverflowError):
        return False
    return all(math.isfinite(value) for value in values)


class PowerLine:
    def __init__(
        self,
        index,
        time,
        open_price,
        high,
        low,
        close,
        vol,
        vol_ma3,
        b_up,
        b_mid,
        b_down,
        source_timestamp=None,
    ):
        self.index = index
        self.time = time
        self.open = open_price
        self.high = high
        self.low = low
        self.close = close
        self.vol = vol
        self.vol_ma3 = vol_ma3
        self.b_up = b_up
        self.b_mid = b_mid
        self.b_down = b_down
        self.source_timestamp = source_timestamp
        self.direction = ""
        self.score = 0
        self.is_true = False
        self.is_combo = False
        self.has_formed_combo = False
        self.is_triple_combo = False
        self.break_level = 0.0
        self.touch_band = False
        self.key_line = False
        self.left_neighbor_idx = None
        self.right_neighbor_idx = None
        self.combo_partner_idx = None
        self.triple_combo_partners = []
        self.is_calculation_context = False
        self.is_incomplete_weekly = False


class Combo:
    def __init__(
        self,
        left_idx,
        right_idx,
        score,
        touch_band,
        direction,
        combo_type="double",
        middle_idx=None,
        block_left_upgrade=False,
    ):
        self.left_idx = left_idx
        self.right_idx = right_idx
        self.score = score
        self.touch_band = touch_band
        self.direction = direction
        self.combo_type = combo_type
        self.middle_idx = middle_idx
        self.block_left_upgrade = bool(block_left_upgrade)


class TrendLine:
    def __init__(
        self,
        index,
        time,
        direction,
        trend_phase,
        power_score,
        is_valid=True,
        break_level=0.0,
        close=0.0,
    ):
        self.index = index
        self.time = time
        self.direction = direction
        self.trend_phase = trend_phase
        self.initial_trend_phase = trend_phase
        self.power_score = power_score
        self.is_valid = is_valid
        self.break_level = break_level
        self.close = close


_CANONICAL_BOLL_NAMES = {
    "b_up": "boll_top",
    "b_mid": "boll_mid",
    "b_down": "boll_bottom",
    "boll_top": "boll_top",
    "boll_mid": "boll_mid",
    "boll_bottom": "boll_bottom",
}


def canonical_boll_name(name):
    """把旧缓存字段名转换成报告使用的标准布林轨名称。"""
    if name is None:
        return None
    return _CANONICAL_BOLL_NAMES.get(str(name))


class UnifiedPowerTrendAnalyzer:
    """统一完成预处理、右向左力量计分和趋势线分析。

    较新的组合先占用成员。较旧组合若被覆盖或关键线被极值破坏，其剩余
    成员会在扫描继续向左时自然按三线、双线、单线优先级重新计分。
    """

    REQUIRED_FIELDS = (
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vol_ma3",
        "b_up",
        "b_mid",
        "b_down",
    )

    def __init__(self, code=None):
        self.code = code
        self.all_lines: list[PowerLine] = []
        self.stack: list[int] = []
        self.combos: list[Combo] = []
        self.combo_history: list[Combo] = []
        self._combo_owner: dict[int, Combo] = {}

        self.reversed_lines: list[int] = []
        self.broken_lines: list[int] = []
        self.cleared_lines: list[int] = []
        self.invalidated_combo_members: set[int] = set()

        self.total_score = 0
        self.total_direction = "none"
        self.total_score_history: list[int] = []
        self.direction_history: list[str] = []
        self.power_change_time = None
        self.event_log: list[str] = []

        self.trend_lines: list[TrendLine] = []
        self.current_power_direction = None
        self.current_segment_start_idx = 0
        self.current_segment_start_time = None
        self.last_break_time = None

        self._source_df = pd.DataFrame()
        self._is_preprocessed = False
        self.analysis_start_idx = None
        self.analysis_start_time = None
        self.calculation_start_idx = None
        self.calculation_start_time = None
        self.power_context_start_idx = 0
        self.power_context_bar_count = 0
        self.last_touch = None
        self.last_touch_fields = ()
        self.last_touch_bolls = ()
        self.last_touch_boll = None
        self.last_mode = None

    def log(self, message):
        self.event_log.append(str(message))

    @staticmethod
    def _flag(value):
        if value is None:
            return False
        try:
            if pd.isna(value):
                return False
        except (TypeError, ValueError):
            pass
        return bool(value)

    def _append_bar_and_solve(self, bar_data, update_trend=True):
        """向当前计算区间追加一根K线，并同步更新力量历史。"""
        missing = [field for field in self.REQUIRED_FIELDS if field not in bar_data]
        if missing:
            raise ValueError(f"周K线数据缺失必填字段: {', '.join(missing)}")

        line = PowerLine(
            index=len(self.all_lines),
            time=pd.Timestamp(bar_data["time"]),
            open_price=float(bar_data["open"]),
            high=float(bar_data["high"]),
            low=float(bar_data["low"]),
            close=float(bar_data["close"]),
            vol=float(bar_data["volume"]),
            vol_ma3=float(bar_data["vol_ma3"]),
            b_up=float(bar_data["b_up"]),
            b_mid=float(bar_data["b_mid"]),
            b_down=float(bar_data["b_down"]),
            source_timestamp=bar_data.get("source_timestamp"),
        )
        line.is_incomplete_weekly = self._flag(
            bar_data.get("is_incomplete_weekly", False)
        )
        line.is_calculation_context = self._flag(
            bar_data.get("is_calculation_context", False)
        )
        if self.all_lines:
            line.left_neighbor_idx = line.index - 1
            self.all_lines[-1].right_neighbor_idx = line.index
        self.all_lines.append(line)

        self._solve_current_snapshot()
        self.total_score_history.append(int(self.total_score))
        self.direction_history.append(self.total_direction)
        self._recompute_power_change_time()
        if update_trend:
            self._solve_trends()
        return line

    def add_bar(self, bar_data):
        """添加新周K线；预处理实例会同步执行重算门控、力量和趋势更新。"""
        if not self._is_preprocessed:
            self.last_mode = "stream"
            return self._append_bar_and_solve(bar_data)

        row = self._bar_to_source_row(bar_data)
        previous_latest_time = (
            pd.Timestamp(self._source_df["time"].max())
            if not self._source_df.empty
            else None
        )
        updated = pd.concat([self._source_df, pd.DataFrame([row])], ignore_index=True)
        updated = (
            updated.sort_values("time")
            .drop_duplicates(subset=["time"], keep="last")
            .reset_index(drop=True)
        )
        touches = self._touches_mapping(row)
        requires_ordered_rebuild = (
            previous_latest_time is not None
            and pd.Timestamp(row["time"]) <= previous_latest_time
        )
        if (
            requires_ordered_rebuild
            or not self.last_touch_fields
            or any(touches.get(field, False) for field in self.last_touch_fields)
        ):
            self.preprocess(updated, _mode="recompute")
            return self.all_lines[-1] if self.all_lines else None

        self._source_df = updated
        self.last_mode = "incremental"
        return self._append_bar_and_solve(bar_data)

    def extend(self, bars: Iterable[dict]):
        for bar in bars:
            self.add_bar(bar)
        return self

    @classmethod
    def _bar_to_source_row(cls, bar_data):
        missing = [field for field in cls.REQUIRED_FIELDS if field not in bar_data]
        if missing:
            raise ValueError(f"周K线数据缺失必填字段: {', '.join(missing)}")
        row = {
            "time": pd.Timestamp(bar_data["time"]),
            "open": float(bar_data["open"]),
            "high": float(bar_data["high"]),
            "low": float(bar_data["low"]),
            "close": float(bar_data["close"]),
            "volume": float(bar_data["volume"]),
            "vol_ma3": float(bar_data["vol_ma3"]),
            "b_up": float(bar_data["b_up"]),
            "b_mid": float(bar_data["b_mid"]),
            "b_down": float(bar_data["b_down"]),
        }
        if cls._flag(bar_data.get("is_incomplete_weekly", False)):
            row["is_incomplete_weekly"] = True
        if bar_data.get("source_timestamp") is not None:
            row["source_timestamp"] = bar_data.get("source_timestamp")
        return row

    @staticmethod
    def _touches_mapping(row):
        return {
            "b_up": _high_touches_upper_band(row.get("high"), row.get("b_up")),
            "b_mid": _band_value_inside_kline(
                row.get("low"),
                row.get("high"),
                row.get("b_mid"),
            ),
            "b_down": _low_touches_lower_band(row.get("low"), row.get("b_down")),
        }

    @staticmethod
    def _resolve_last_touch_fields(touches):
        touches_up = bool((touches or {}).get("b_up", False))
        touches_mid = bool((touches or {}).get("b_mid", False))
        touches_down = bool((touches or {}).get("b_down", False))
        if touches_up and touches_down:
            return ("b_up", "b_down")
        if touches_up:
            return ("b_up",)
        if touches_down:
            return ("b_down",)
        if touches_mid:
            return ("b_mid",)
        return ()

    @classmethod
    def _normalize_source_frame(cls, frame, as_of=None):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("周线数据为空")
        normalized = frame.copy()
        if "datetime" in normalized.columns and "time" not in normalized.columns:
            normalized = normalized.rename(columns={"datetime": "time"})
        if "vol" in normalized.columns and "volume" not in normalized.columns:
            normalized = normalized.rename(columns={"vol": "volume"})
        missing = [field for field in cls.REQUIRED_FIELDS if field not in normalized.columns]
        if missing:
            raise ValueError(f"周线数据缺失必填字段: {', '.join(missing)}")
        normalized["time"] = pd.to_datetime(normalized["time"])
        if as_of is not None:
            normalized = normalized.loc[
                normalized["time"] <= pd.Timestamp(as_of)
            ].copy()
        normalized = (
            normalized.sort_values("time")
            .drop_duplicates(subset=["time"], keep="last")
            .reset_index(drop=True)
        )
        if normalized.empty:
            raise ValueError("截止时间之前没有周线数据")
        return normalized

    @staticmethod
    def _first_valid_boll_index(frame):
        """第一根满足 BOLL(26) 计算条件（三轨均已生成）的K线索引。"""
        for index in range(len(frame)):
            if _row_has_valid_boll(frame.iloc[index]):
                return index
        return None

    def _find_analysis_start(self, frame):
        latest_index = len(frame) - 1
        first_valid = self._first_valid_boll_index(frame)
        seen = {"b_up": False, "b_mid": False, "b_down": False}
        analysis_index = None
        for index in range(latest_index, -1, -1):
            # 力量/分析起点必须不早于第一根可计算 BOLL(26) 的K线：
            # 布林带未生成的K线不参与扫描（按值判断，而非假定固定在第26根）。
            if first_valid is None or index < first_valid:
                continue
            touches = self._touches_mapping(frame.iloc[index])
            for field in seen:
                seen[field] = seen[field] or bool(touches.get(field, False))
            if all(seen.values()):
                analysis_index = index
                break
        if analysis_index is None:
            # 找不到起点时，从最早满足 BOLL(26) 计算条件的K线开始。
            if first_valid is not None:
                return first_valid, (), True
            return 0, (), False
        touches = self._touches_mapping(frame.iloc[analysis_index])
        return analysis_index, self._resolve_last_touch_fields(touches), True

    def _find_calculation_start(self, frame, analysis_index):
        latest_index = len(frame) - 1
        region_start = analysis_index
        for index in range(analysis_index, latest_index + 1):
            if self._touches_mapping(frame.iloc[index]).get("b_mid", False):
                region_start = index
                break
        region = frame.iloc[region_start : latest_index + 1]
        if region.empty:
            return int(analysis_index)
        highest_index = int(region["high"].idxmax())
        lowest_index = int(region["low"].idxmin())
        return highest_index if highest_index < lowest_index else lowest_index

    def preprocess(self, frame, as_of=None, _mode=None):
        """在一个实例中完成起点、力量、趋势与快照预处理。"""
        source = self._normalize_source_frame(frame, as_of=as_of)
        mode = _mode or ("recompute" if self._is_preprocessed else "initial")
        analysis_index, touch_fields, has_full_boll = self._find_analysis_start(source)
        calculation_index = (
            self._find_calculation_start(source, analysis_index)
            if has_full_boll and touch_fields
            else int(analysis_index)
        )
        # 硬规则：力量计算起点必须不早于第一根满足 BOLL(26) 计算条件的K线。
        first_valid_boll = self._first_valid_boll_index(source)
        if first_valid_boll is not None:
            calculation_index = max(int(calculation_index), int(first_valid_boll))
        context_start = max(0, int(calculation_index) - 2)
        context_count = int(calculation_index) - context_start
        if not has_full_boll:
            context_start = 0
            context_count = len(source)

        code = self.code
        self.__init__(code=code)
        self._source_df = source
        self._is_preprocessed = True
        self.analysis_start_idx = int(analysis_index)
        self.analysis_start_time = pd.Timestamp(source.iloc[analysis_index]["time"])
        self.calculation_start_idx = int(calculation_index)
        self.calculation_start_time = pd.Timestamp(source.iloc[calculation_index]["time"])
        self.power_context_start_idx = int(context_start)
        self.power_context_bar_count = int(context_count)
        self.last_touch_fields = tuple(touch_fields)
        self.last_touch = touch_fields[0] if touch_fields else None
        self.last_touch_bolls = tuple(
            canonical_boll_name(field) for field in touch_fields
        )
        self.last_touch_boll = (
            self.last_touch_bolls[0]
            if len(self.last_touch_bolls) == 1
            else list(self.last_touch_bolls)
        )
        self.last_mode = mode

        calculation_frame = source.iloc[context_start:].reset_index(drop=True)
        for local_index, (_, row) in enumerate(calculation_frame.iterrows()):
            bar = self._bar_to_source_row(row.to_dict())
            if local_index < context_count:
                bar["is_calculation_context"] = True
            self._append_bar_and_solve(bar, update_trend=False)
        self._solve_trends()
        return self.get_snapshot()

    @staticmethod
    def _is_high_volume(line):
        values = (line.vol, line.vol_ma3)
        if any(pd.isna(value) for value in values):
            return False
        return float(line.vol) >= float(line.vol_ma3)

    @staticmethod
    def _get_direction(line):
        return classify_weekly_kline_direction(
            line.open,
            line.high,
            line.low,
            line.close,
        )["corrected_direction"]

    @staticmethod
    def _touches(line):
        return {
            "b_up": _high_touches_upper_band(line.high, line.b_up),
            "b_mid": _band_value_inside_kline(line.low, line.high, line.b_mid),
            "b_down": _low_touches_lower_band(line.low, line.b_down),
        }

    def _is_touching_band(self, line):
        return any(self._touches(line).values())

    def _is_scoring_bar(self, index):
        line = self.all_lines[index]
        return (
            not bool(getattr(line, "is_incomplete_weekly", False))
            and self._is_high_volume(line)
            and self._get_direction(line) in ("up", "down")
        )

    def _is_price_break(self, power_index, destroyer_index):
        power = self.all_lines[power_index]
        destroyer = self.all_lines[destroyer_index]
        direction = self._get_direction(power)
        if direction == "up":
            return float(destroyer.low) <= float(power.low)
        if direction == "down":
            return float(destroyer.high) >= float(power.high)
        return False

    def _build_suffix_extremes(self):
        """构建右侧最低/最高价表，使全右区间破坏判断为 O(1)。"""
        size = len(self.all_lines)
        self._suffix_low = [float("inf")] * (size + 1)
        self._suffix_high = [float("-inf")] * (size + 1)
        for index in range(size - 1, -1, -1):
            line = self.all_lines[index]
            self._suffix_low[index] = min(float(line.low), self._suffix_low[index + 1])
            self._suffix_high[index] = max(float(line.high), self._suffix_high[index + 1])

    def _is_broken_after(self, power_index, left_exclusive):
        """检查 ``(left_exclusive, latest]`` 是否破坏指定力量K线。"""
        right_start = int(left_exclusive) + 1
        if right_start >= len(self.all_lines):
            return False
        power = self.all_lines[power_index]
        direction = self._get_direction(power)
        if direction == "up":
            return self._suffix_low[right_start] <= float(power.low)
        if direction == "down":
            return self._suffix_high[right_start] >= float(power.high)
        return False

    @staticmethod
    def _combo_members(combo):
        members = [int(combo.left_idx)]
        if combo.middle_idx is not None:
            members.append(int(combo.middle_idx))
        members.append(int(combo.right_idx))
        return members

    def _reset_line_state(self, line):
        line.direction = self._get_direction(line)
        line.score = 0
        line.is_true = False
        line.is_combo = False
        line.has_formed_combo = False
        line.is_triple_combo = False
        line.break_level = 0.0
        line.touch_band = self._is_touching_band(line)
        line.key_line = False
        line.combo_partner_idx = None
        line.triple_combo_partners = []

    def _mark_broken(self, index):
        if index not in self.broken_lines:
            self.broken_lines.append(index)
        if index not in self.cleared_lines:
            self.cleared_lines.append(index)

    def _activate_single(self, index):
        line = self.all_lines[index]
        line.score = 1
        line.is_true = True
        line.key_line = True
        line.break_level = float(line.low if line.direction == "up" else line.high)
        self.stack.append(index)
        self.log(f"K{index} 单线: {line.direction}, 分数=1")

    def _activate_combo(self, combo, occupied):
        members = self._combo_members(combo)
        right_index = int(combo.right_idx)
        for index in members:
            line = self.all_lines[index]
            line.is_combo = True
            line.has_formed_combo = True
            line.is_triple_combo = combo.combo_type == "triple"
            line.key_line = index == right_index
            line.score = int(combo.score) if index == right_index else 0
            line.is_true = index == right_index
            line.combo_partner_idx = (
                int(combo.left_idx) if index == right_index else right_index
            )
            line.triple_combo_partners = (
                [member for member in members if member != index]
                if combo.combo_type == "triple"
                else []
            )
            line.break_level = (
                float(line.low if line.direction == "up" else line.high)
                if index == right_index
                else 0.0
            )
            self._combo_owner[index] = combo
        self.all_lines[right_index].touch_band = bool(combo.touch_band)
        self.stack.append(right_index)
        self.combos.append(combo)
        occupied.update(members)
        for index in members[:-1]:
            if index not in self.reversed_lines:
                self.reversed_lines.append(index)
        self.log(
            f"K{members} {combo.combo_type}: {combo.direction}, "
            f"分数={combo.score}, 触带={combo.touch_band}"
        )

    def _try_triple(self, right_index, occupied):
        """beta1 改版（2026-08-15 用户规则）：主K线三线组合成立前，先检查中间K线。

        主K线 R（right_index）满足三线组合前提后：
        1. 中间K线 M 被右侧（到计算终点）K线破坏 → 保持 R 三线组合；
        2. M 未被破坏且能与其左1、左2形成三线组合（M 为关键线）：
           - M 与 R 方向相反（M 被主K线反转）→ 保持 R 三线组合；
           - M 与 R 同向（不被反转）→ M 的三线组合保留计分，R 单独计分；
        3. M 无法三线 → M 能与左1形成双线且双线方向与 R 一致：
           - 一致 → R 单独计分，M 与左1双线计分；
           - 不一致 → 保持 R 三线组合。
        """
        if right_index < 2:
            return False
        left_index = right_index - 2
        middle_index = right_index - 1
        members = (left_index, middle_index, right_index)
        if any(index in occupied for index in members):
            return False
        if bool(
            getattr(
                self.all_lines[middle_index],
                "is_incomplete_weekly",
                False,
            )
        ):
            return False
        if not self._is_scoring_bar(left_index):
            return False
        left_direction = self.all_lines[left_index].direction
        right_direction = self.all_lines[right_index].direction
        if left_direction == right_direction:
            return False

        def form_main_triple():
            touch_band = any(
                self._is_touching_band(self.all_lines[index]) for index in members
            )
            combo = Combo(
                left_idx=left_index,
                middle_idx=middle_index,
                right_idx=right_index,
                score=2 if touch_band else 1,
                touch_band=touch_band,
                direction=right_direction,
                combo_type="triple",
            )
            self._activate_combo(combo, occupied)
            return True

        middle = self.all_lines[middle_index]
        # 1) 中间K线被右侧到计算终点的K线破坏 → 保持主K线三线组合
        if self._is_broken_after(middle_index, middle_index):
            return form_main_triple()
        # 2) 中间K线未被破坏：尝试它自己的三线组合（左1=left_index，左2=left_index-1）
        if self._middle_can_form_triple(middle_index, occupied):
            if middle.direction != right_direction:
                # 中间K线会被主K线反转 → 保持主K线三线组合
                return form_main_triple()
            # 不会被反转 → 中间K线三线组合保留计分；
            # 主K线按普通单线规则计分：触轨才计 1 分，不触轨则无分（2026-08-15 用户确认）
            self._form_middle_combo(middle_index, occupied, combo_type="triple")
            if self._is_touching_band(self.all_lines[right_index]):
                self._activate_single(right_index)
                occupied.add(right_index)
            return True
        # 3) 中间K线无法三线 → 尝试与左1（left_index）形成双线，方向须与主K线一致
        if (
            middle.direction in ("up", "down")
            and middle.direction == right_direction
            and self._is_scoring_bar(middle_index)
            and middle.direction != left_direction
        ):
            self._form_middle_combo(middle_index, occupied, combo_type="double")
            if self._is_touching_band(self.all_lines[right_index]):
                self._activate_single(right_index)
                occupied.add(right_index)
            return True
        # 方向不一致或无法成组 → 保持主K线三线组合
        return form_main_triple()

    def _middle_can_form_triple(self, middle_index, occupied):
        """中间K线作为关键线，与其左1、左2能否形成三线组合。

        （"中间K线右侧没有能破坏它的K线"由调用方在调用前保证。）
        """
        left1_index = middle_index - 1
        left2_index = middle_index - 2
        if left2_index < 0:
            return False
        if any(index in occupied for index in (left2_index, left1_index, middle_index)):
            return False
        middle = self.all_lines[middle_index]
        if middle.direction not in ("up", "down"):
            return False
        if not self._is_scoring_bar(middle_index):
            return False
        if bool(getattr(self.all_lines[left1_index], "is_incomplete_weekly", False)):
            return False
        if not self._is_scoring_bar(left2_index):
            return False
        return self.all_lines[left2_index].direction != middle.direction

    def _form_middle_combo(self, middle_index, occupied, combo_type):
        """以中间K线为关键线激活它自己的三线/双线组合（beta1 拆分计分用）。"""
        middle = self.all_lines[middle_index]
        if combo_type == "triple":
            members = (middle_index - 2, middle_index - 1, middle_index)
            combo = Combo(
                left_idx=members[0],
                middle_idx=members[1],
                right_idx=middle_index,
                score=0,
                touch_band=False,
                direction=middle.direction,
                combo_type="triple",
            )
        else:
            members = (middle_index - 1, middle_index)
            combo = Combo(
                left_idx=members[0],
                right_idx=middle_index,
                score=0,
                touch_band=False,
                direction=middle.direction,
                combo_type="double",
            )
        touch_band = any(self._is_touching_band(self.all_lines[index]) for index in members)
        combo.score = 2 if touch_band else 1
        combo.touch_band = touch_band
        self._activate_combo(combo, occupied)

    def _try_double(self, right_index, occupied):
        if right_index < 1:
            return False
        left_index = right_index - 1
        members = (left_index, right_index)
        if any(index in occupied for index in members):
            return False
        if not self._is_scoring_bar(left_index):
            return False
        left_direction = self.all_lines[left_index].direction
        right_direction = self.all_lines[right_index].direction
        if left_direction == right_direction:
            return False
        touch_band = any(self._is_touching_band(self.all_lines[index]) for index in members)
        combo = Combo(
            left_idx=left_index,
            right_idx=right_index,
            score=2 if touch_band else 1,
            touch_band=touch_band,
            direction=right_direction,
            combo_type="double",
        )
        self._activate_combo(combo, occupied)
        return True

    def _solve_current_snapshot(self):
        """按最新优先、从右向左重建当前有效力量栈。"""
        self.stack = []
        self.combos = []
        self.combo_history = []
        self._combo_owner = {}
        self.reversed_lines = []
        self.broken_lines = []
        self.cleared_lines = []
        self.invalidated_combo_members = set()
        self.event_log = []

        for line in self.all_lines:
            self._reset_line_state(line)
        self._build_suffix_extremes()

        if not self.all_lines:
            self.total_score = 0
            self.total_direction = "none"
            return

        latest_index = len(self.all_lines) - 1
        occupied: set[int] = set()
        for index in range(latest_index, -1, -1):
            line = self.all_lines[index]
            if (
                index in occupied
                or bool(getattr(line, "is_calculation_context", False))
                or not self._is_scoring_bar(index)
            ):
                continue

            if self._is_broken_after(index, index):
                self._mark_broken(index)
                self.log(f"K{index} 被右侧K线极值破坏")
                continue

            if self._try_triple(index, occupied):
                continue
            if self._try_double(index, occupied):
                continue
            if self._is_touching_band(line):
                self._activate_single(index)
                occupied.add(index)

        self.stack = sorted(set(self.stack))
        self.combos.sort(key=lambda combo: int(combo.right_idx))
        self.combo_history = list(self.combos)
        # 非关键组合成员不在有效力量栈，其后续极值破坏只做永久失效记录；
        # 只要最右关键线仍存活，已经形成的组合分数继续记在关键线上。
        for combo in self.combos:
            for member_index in self._combo_members(combo)[:-1]:
                if not self._is_scoring_bar(member_index):
                    continue
                is_broken = self._is_broken_after(
                    member_index,
                    member_index,
                )
                if not is_broken:
                    continue
                self.invalidated_combo_members.add(member_index)
                self._mark_broken(member_index)
                self.log(
                    f"组合非关键成员 K{member_index} 被关键线右侧K线极值破坏，"
                    f"关键线 K{combo.right_idx} 分数保持"
                )
        self.broken_lines.sort()
        self.cleared_lines.sort()
        self.reversed_lines.sort()
        self._update_total_score()

    def _update_total_score(self):
        total = 0
        valid = []
        for index in self.stack:
            line = self.all_lines[index]
            if not line.is_true or line.score <= 0:
                continue
            valid.append(line)
            total += int(line.score) if line.direction == "up" else -int(line.score)
        if total > 0:
            direction = "up"
        elif total < 0:
            direction = "down"
        elif valid:
            direction = max(valid, key=lambda line: int(line.index)).direction
        else:
            direction = "none"
        self.total_score = int(total)
        self.total_direction = direction

    def _recompute_power_change_time(self):
        # 反转新规则（2026-08-21 起）：上根周K方向与当前周K方向不一致即算反转，
        # "无"(none) 与 涨/跌 之间的相互切换同样计为反转（不再跳过 none）。
        self.power_change_time = None
        if not self.direction_history:
            return
        current_index = len(self.direction_history) - 1
        current_direction = self.direction_history[current_index]
        segment_start = current_index
        for index in range(current_index - 1, -1, -1):
            if self.direction_history[index] == current_direction:
                segment_start = index
                continue
            self.power_change_time = pd.Timestamp(self.all_lines[segment_start].time)
            return

    @staticmethod
    def _trend_body_ratio_ok(bar):
        bar_range = float(bar["high"]) - float(bar["low"])
        if bar_range <= 0:
            return False
        return abs(float(bar["close"]) - float(bar["open"])) / bar_range >= 0.5

    @staticmethod
    def _trend_body_mid(bar):
        return (float(bar["open"]) + float(bar["close"])) / 2.0

    @staticmethod
    def _trend_phase(line):
        phase = str(getattr(line, "initial_trend_phase", "") or "").upper()
        if phase in ("O1", "O2", "O3"):
            return phase
        phase = str(getattr(line, "trend_phase", "") or "").upper()
        return phase if phase in ("O1", "O2", "O3") else "none"

    @staticmethod
    def _next_trend_phase(phase):
        return {"O1": "O2", "O2": "O3", "O3": "done"}.get(
            str(phase or "").upper(),
            "O1",
        )

    def _trend_breakable_start(self, lines, minimum):
        start = int(minimum)
        last_o3 = None
        for line in lines:
            index = int(getattr(line, "index", -1))
            if index < start:
                continue
            phase = self._trend_phase(line)
            if phase == "O3":
                last_o3 = index
            elif phase == "O1" and last_o3 is not None and index > last_o3:
                start = max(start, index)
                last_o3 = None
        return start

    @staticmethod
    def _line_as_bar(line):
        return {
            "time": pd.Timestamp(line.time),
            "open": float(line.open),
            "high": float(line.high),
            "low": float(line.low),
            "close": float(line.close),
            "volume": float(line.vol),
            "vol_ma3": float(line.vol_ma3),
            "b_up": float(line.b_up),
            "b_mid": float(line.b_mid),
            "b_down": float(line.b_down),
            "is_incomplete_weekly": bool(
                getattr(line, "is_incomplete_weekly", False)
            ),
        }

    def _solve_trends(self):
        """用本实例的力量方向历史生成当前反转区段趋势线。"""
        context_count = int(self.power_context_bar_count or 0)
        formal_lines = self.all_lines[context_count:]
        formal_bars = [self._line_as_bar(line) for line in formal_lines]
        formal_directions = self.direction_history[context_count:]
        formal_scores = self.total_score_history[context_count:]

        trend_lines = []
        current_direction = None
        segment_start = 0
        self.last_break_time = None
        for index, bar in enumerate(formal_bars):
            direction = (
                formal_directions[index]
                if index < len(formal_directions)
                else (formal_directions[-1] if formal_directions else None)
            )
            if direction not in ("up", "down"):
                continue
            if current_direction != direction:
                current_direction = direction
                segment_start = index
                trend_lines = []
                self.current_power_direction = direction
                self.current_segment_start_idx = index
                self.current_segment_start_time = bar["time"]

            breakable_start = self._trend_breakable_start(
                trend_lines,
                segment_start,
            )
            for trend_line in trend_lines:
                if (
                    not getattr(trend_line, "is_valid", True)
                    or int(getattr(trend_line, "index", -1)) < breakable_start
                ):
                    continue
                broken = (
                    trend_line.direction == "up"
                    and float(bar["low"]) <= float(trend_line.break_level)
                ) or (
                    trend_line.direction == "down"
                    and float(bar["high"]) >= float(trend_line.break_level)
                )
                if broken:
                    trend_line.is_valid = False
                    trend_line.trend_phase = "none"
                    self.last_break_time = bar["time"]

            if bool(bar.get("is_incomplete_weekly", False)):
                continue
            bar_direction = classify_weekly_kline_direction(
                bar["open"],
                bar["high"],
                bar["low"],
                bar["close"],
            )["corrected_direction"]
            if (
                not self._trend_body_ratio_ok(bar)
                or bar_direction != current_direction
            ):
                continue

            last_valid = next(
                (
                    line
                    for line in reversed(trend_lines)
                    if getattr(line, "is_valid", True)
                    and line.direction == current_direction
                    and int(line.index) >= breakable_start
                ),
                None,
            )
            if last_valid is not None:
                body_mid = self._trend_body_mid(bar)
                if current_direction == "up" and body_mid < float(last_valid.close):
                    continue
                if current_direction == "down" and body_mid > float(last_valid.close):
                    continue
            phase = (
                self._next_trend_phase(self._trend_phase(last_valid))
                if last_valid is not None
                else "O1"
            )
            if phase == "done":
                continue
            power_score = (
                int(formal_scores[index])
                if index < len(formal_scores)
                else int(self.total_score)
            )
            trend_lines.append(
                TrendLine(
                    index=index,
                    time=bar["time"],
                    direction=current_direction,
                    trend_phase=phase,
                    power_score=power_score,
                    break_level=(
                        float(bar["low"])
                        if current_direction == "up"
                        else float(bar["high"])
                    ),
                    close=float(bar["close"]),
                )
            )

        self.trend_lines = trend_lines
        if current_direction not in ("up", "down"):
            self.current_power_direction = None
            self.current_segment_start_idx = 0
            self.current_segment_start_time = None

    def get_current_trend_lines(self):
        direction = self.current_power_direction
        if direction not in ("up", "down"):
            return []
        return [
            line
            for line in self.trend_lines
            if getattr(line, "is_valid", True)
            and line.direction == direction
            and self._trend_phase(line) in ("O1", "O2", "O3")
        ]

    def get_latest_trend_line(self):
        current = self.get_current_trend_lines()
        return current[-1] if current else None

    def _active_combos(self):
        return list(self.combos)

    def _active_combo_for_line(self, index):
        return self._combo_owner.get(int(index))

    def get_power_result(self):
        return self.total_direction

    def get_current_state(self):
        return {
            "total_week_power_score": self.total_score,
            "current_trend": self.total_direction,
            "active_week_lines": len(self.stack),
            "reversed_week_count": len(self.reversed_lines),
            "broken_week_count": len(self.broken_lines),
            "double_combo_count": sum(combo.combo_type == "double" for combo in self.combos),
            "triple_combo_count": sum(combo.combo_type == "triple" for combo in self.combos),
            "trend_change_time": self.power_change_time,
        }

    def get_snapshot(self):
        """返回同一状态机中的预处理、力量和趋势快照。"""
        current_score = int(self.all_lines[-1].score) if self.all_lines else 0
        active = []
        for index in self.stack:
            line = self.all_lines[index]
            if line.is_triple_combo:
                shape = "triple"
            elif line.is_combo:
                shape = "double"
            else:
                shape = "single"
            source_time = None
            if line.source_timestamp is not None:
                parsed_source_time = pd.to_datetime(
                    line.source_timestamp,
                    errors="coerce",
                )
                if not pd.isna(parsed_source_time):
                    source_time = (
                        pd.Timestamp(parsed_source_time) + pd.Timedelta(hours=8)
                    ).strftime("%Y-%m-%d %H:%M")
            active.append(
                {
                    "index": int(index),
                    "time": pd.Timestamp(line.time).strftime("%Y-%m-%d"),
                    "source_time": source_time,
                    "direction": line.direction,
                    "score": int(line.score),
                    "shape": shape,
                    "volume": float(line.vol),
                    "vol_ma3": float(line.vol_ma3),
                    "is_high_volume": self._is_high_volume(line),
                }
            )
        active_trends = [
            {
                "index": int(line.index),
                "time": pd.Timestamp(line.time).strftime("%Y-%m-%d"),
                "direction": line.direction,
                "phase": self._trend_phase(line),
                "break_level": float(line.break_level),
            }
            for line in self.get_current_trend_lines()
        ]
        change_time = (
            pd.Timestamp(self.power_change_time).strftime("%Y-%m-%d")
            if self.power_change_time is not None
            else None
        )
        current_time = (
            pd.Timestamp(self.all_lines[-1].time).strftime("%Y-%m-%d")
            if self.all_lines
            else None
        )
        return {
            "mode": self.last_mode,
            "as_of": current_time,
            "analysis_start_time": (
                pd.Timestamp(self.analysis_start_time).strftime("%Y-%m-%d")
                if self.analysis_start_time is not None
                else None
            ),
            "analysis_start_idx": self.analysis_start_idx,
            "calculation_start_time": (
                pd.Timestamp(self.calculation_start_time).strftime("%Y-%m-%d")
                if self.calculation_start_time is not None
                else None
            ),
            "calculation_start_idx": self.calculation_start_idx,
            "power_context_start_idx": self.power_context_start_idx,
            "power_context_bar_count": self.power_context_bar_count,
            "last_touch": self.last_touch,
            "last_touch_boll": self.last_touch_boll,
            "last_touch_bolls": list(self.last_touch_bolls),
            "total_power_score": int(self.total_score),
            "power_direction": self.total_direction,
            "current_bar_power_score": current_score,
            "active_power_lines": active,
            "power_change_time": change_time,
            "power_reversal": bool(change_time and change_time == current_time),
            "active_trend_lines": active_trends,
            "latest_trend": active_trends[-1] if active_trends else None,
        }
