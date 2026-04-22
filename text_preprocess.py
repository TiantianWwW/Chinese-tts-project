"""兼容层：统一复用 preprocess.py 的管道实现。"""

from preprocess import preprocess_for_tts


if __name__ == "__main__":
    test_cases = [
        "今天的温度是25度。",
        "价格：99.9元。",
        "我学会了使用Python编程。",
        "这个APP很好用。",
        "请注意：1. 准时；2. 带身份证。",
        "电话：13812345678。",
    ]

    for text in test_cases:
        result = preprocess_for_tts(text)
        print(f"\n原文: {text}")
        print(f"结果: {result}")
