import unittest

from preprocess import preprocess_for_tts


class TestPreprocessForTTS(unittest.TestCase):
    def test_integer_conversion(self):
        self.assertEqual(preprocess_for_tts("今天25度"), "今天二十五度")

    def test_decimal_conversion(self):
        self.assertEqual(preprocess_for_tts("价格99.9元"), "价格九十九点九元")

    def test_remove_english(self):
        self.assertEqual(preprocess_for_tts("我用Python写APP"), "我用写")

    def test_punctuation_normalization(self):
        result = preprocess_for_tts("你好, world!")
        self.assertIn("，", result)
        self.assertIn("！", result)

    def test_large_number_conversion(self):
        self.assertEqual(preprocess_for_tts("订单10001号"), "订单一万零一号")


if __name__ == "__main__":
    unittest.main()
