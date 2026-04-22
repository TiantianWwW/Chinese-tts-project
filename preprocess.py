"""
中文 TTS 文本预处理模块（精简实用版）
功能：
- 数字转中文读法
- 移除英文字母
- 英文标点转中文标点
- 清理多余空白
"""

import re

# 数字映射表
NUM_MAP = {
    '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
    '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
}

# 标点映射表
PUNCT_MAP = {
    ',': '，', '.': '。', '?': '？', '!': '！',
    ':': '：', ';': '；', '(': '（', ')': '）',
}


def integer_to_chinese(num_str: str) -> str:
    """整数转中文读法"""
    try:
        num = int(num_str)
    except ValueError:
        return num_str

    if num == 0:
        return '零'
    if num < 10:
        return NUM_MAP.get(str(num), str(num))
    if num < 20:
        return '十' + (NUM_MAP.get(str(num % 10), '') if num % 10 != 0 else '')
    if num < 100:
        tens = num // 10
        ones = num % 10
        return NUM_MAP[str(tens)] + '十' + (NUM_MAP[str(ones)] if ones != 0 else '')
    if num < 1000:
        hundreds = num // 100
        rest = num % 100
        result = NUM_MAP[str(hundreds)] + '百'
        if rest != 0:
            if rest < 10:
                result += '零' + NUM_MAP[str(rest)]
            else:
                result += integer_to_chinese(str(rest))
        return result
    if num < 10000:
        thousands = num // 1000
        rest = num % 1000
        result = NUM_MAP[str(thousands)] + '千'
        if rest != 0:
            if rest < 100:
                result += '零' + integer_to_chinese(str(rest))
            else:
                result += integer_to_chinese(str(rest))
        return result

    # 大于等于1万
    wan = num // 10000
    rest = num % 10000
    result = integer_to_chinese(str(wan)) + '万'
    if rest != 0:
        if rest < 1000:
            result += '零' + integer_to_chinese(str(rest))
        else:
            result += integer_to_chinese(str(rest))
    return result


def decimal_to_chinese(match) -> str:
    """小数转中文读法"""
    parts = match.group(0).split('.')
    int_part = integer_to_chinese(parts[0])
    dec_part = ''.join(NUM_MAP.get(c, c) for c in parts[1])
    return f"{int_part}点{dec_part}"


def preprocess_for_tts(text: str) -> str:
    """
    中文 TTS 文本预处理

    处理规则：
    1. 数字转中文读法（保留）
    2. 英文字母直接移除
    3. 英文标点转中文标点
    4. 清理多余空白
    """
    # 1. 处理小数（优先，避免被整数规则干扰）
    text = re.sub(r'\d+\.\d+', decimal_to_chinese, text)

    # 2. 处理整数（排除年份等特殊情况）
    text = re.sub(r'(?<![年月日])\d+(?![年月日])',
                    lambda m: integer_to_chinese(m.group(0)), text)

    # 3. 移除英文字母
    text = re.sub(r'[a-zA-Z]+', '', text)

    # 4. 英文标点转中文标点
    for en_punct, zh_punct in PUNCT_MAP.items():
        text = text.replace(en_punct, zh_punct)

    # 5. 处理省略号
    text = re.sub(r'\.{3,}', '，', text)

    # 6. 清理多余空白
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


# ========== 测试代码 ==========
if __name__ == "__main__":
    test_cases = [
        "今天的温度是25度。",
        "价格：99.9元。",
        "我学会了使用Python编程。",
        "这个APP很好用。",
        "请注意：1. 准时；2. 带身份证。",
        "电话：13812345678。",
    ]

    print("=" * 50)
    print("中文 TTS 预处理测试")
    print("=" * 50)

    for text in test_cases:
        result = preprocess_for_tts(text)
        print(f"\n原文: {text}")
        print(f"结果: {result}")