import unittest

from right_to_left_power_beta1 import classify_weekly_kline_direction


class MidpointDirectionTests(unittest.TestCase):
    def test_hongzao_closing_at_midpoint_is_up(self):
        # 红枣加权 2026-08-28：C == L + (H - L) / 2。
        result = classify_weekly_kline_direction(8070, 8215, 7955, 8085)

        self.assertEqual(result["traditional_direction"], "up")
        self.assertEqual(result["corrected_direction"], "up")
        self.assertFalse(result["is_direction_reversed"])

    def test_bearish_candle_closing_at_midpoint_is_up(self):
        result = classify_weekly_kline_direction(8100, 8215, 7955, 8085)

        self.assertEqual(result["traditional_direction"], "down")
        self.assertEqual(result["corrected_direction"], "up")

    def test_flat_candle_closing_at_midpoint_is_up(self):
        result = classify_weekly_kline_direction(8085, 8215, 7955, 8085)

        self.assertEqual(result["corrected_direction"], "up")

    def test_bearish_close_below_midpoint_remains_down(self):
        result = classify_weekly_kline_direction(8100, 8215, 7955, 8084)

        self.assertEqual(result["corrected_direction"], "down")

    def test_bearish_close_just_below_midpoint_remains_down(self):
        # 仅严格相等时才适用中点向上规则，不能用容差扩大该边界。
        result = classify_weekly_kline_direction(8100, 8215, 7955, 8084.9999999995)

        self.assertEqual(result["corrected_direction"], "down")


if __name__ == "__main__":
    unittest.main()
