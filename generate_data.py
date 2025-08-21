import random
import sys
import os


def generate_data(num_entries=10000, filename="dialogue_data.txt"):
    """
    生成一个更复杂、更精确的中文对话训练数据集。

    Args:
        num_entries (int): 想要生成的问答对数量。
        filename (str): 输出文件名。
    """
    # 核心问答模板
    qa_templates = [
        ("你好", ["你好呀", "你好，很高兴和你聊天", "嗨，你好！"]),
        ("谢谢", ["不客气", "不用谢", "这是我的荣幸"]),
        ("你叫什么名字？", ["我是一个AI助手", "我没有名字", "你可以叫我小A"]),
        ("你来自哪里？", ["我来自云端", "我没有实体", "我生活在互联网上"]),
        ("你有什么爱好吗？", ["我喜欢学习知识", "我的爱好是处理信息", "我喜欢和你聊天"]),
        ("今天天气怎么样？", ["我无法获取实时天气", "我不知道，但希望你心情愉快", "请查询天气预报"]),
        ("你累了吗？", ["我不会感到疲劳", "作为AI，我不需要休息", "我随时待命"]),
        ("你觉得我怎么样？", ["你很友好", "你是一个很棒的用户", "你很有趣"]),
    ]

    # 扩展问答模板和词汇
    extended_templates = {
        # 情绪和感受
        "feeling": {
            "questions": ["我感到{}。", "我今天很{}。", "我现在心情{}。"],
            "responses": ["为什么呢？", "怎么了？能和我说说吗？", "很高兴听到这个消息！",
                          "我很遗憾听到这个消息，希望你能好起来。"],
            "adjectives": ["开心", "难过", "生气", "高兴", "兴奋", "沮丧", "焦虑", "平静", "无聊"],
        },
        # 日常活动和建议
        "daily_life": {
            "questions": ["我正在{}。", "我打算{}。", "我想知道如何{}。"],
            "responses": ["听起来不错！", "那是个好主意，祝你顺利！", "我可以提供一些相关的建议。"],
            "verbs": ["学习", "看电影", "看书", "做饭", "运动", "旅行", "编程", "休息"],
        },
        # 知识和信息
        "knowledge": {
            "questions": ["什么是{}？", "你了解{}吗？", "{}有什么用？"],
            "responses": ["{}是一种...", "我能为你提供关于{}的信息。", "{}是用来做...的"],
            "nouns": ["人工智能", "编程", "机器学习", "深度学习", "宇宙", "历史", "艺术", "音乐"],
        },
        # 肯定和否定
        "affirmation": {
            "questions": ["你说得对。", "你说得不对。", "你是正确的。"],
            "responses": ["谢谢你的肯定。", "很抱歉，我可能犯了错误。", "我很高兴能帮助你。"],
        },
        # 命令和请求
        "request": {
            "questions": ["请帮我{}。", "你能{}吗？"],
            "responses": ["好的，我很乐意{}。", "当然，请告诉我你需要我做什么。", "我不能进行物理操作，但我可以提供信息。"],
            "verbs": ["写一封信", "写一首诗", "解决一个问题", "查找资料"],
        },
    }

    # 多轮对话模板
    multi_turn_templates = [
        ["你好。", "嗨！", "今天过得怎么样？", "挺好的，你呢？"],
        ["我正在学习编程。", "听起来很棒！", "你学的是哪种语言？", "主要是Python。"],
        ["我今天很累。", "怎么了？", "工作了一整天。", "好好休息一下吧。"],
        ["你了解人工智能吗？", "是的，我对此有些了解。", "能给我举个例子吗？", "比如我可以和你聊天。"],
    ]

    with open(filename, 'w', encoding='utf-8') as f:
        print(f"开始生成 {num_entries} 条数据...")

        # 首先写入核心问答对，确保基本对话模型稳定
        for q, responses in qa_templates:
            f.write(f"{q}|{random.choice(responses)}\n")

        # 扩展生成更复杂的问答对
        count = len(qa_templates)
        while count < num_entries * 0.9:  # 90%为问答对
            template_type = random.choice(list(extended_templates.keys()))
            template_set = extended_templates[template_type]

            question_template = random.choice(template_set["questions"])
            response_template = random.choice(template_set["responses"])

            # 根据类型填充词汇
            if "adjectives" in template_set:
                adj = random.choice(template_set["adjectives"])
                q = question_template.format(adj)
                if "为什么" in response_template:
                    r = response_template
                else:
                    r = random.choice(template_set["responses"]).format(adj)
            elif "verbs" in template_set:
                verb = random.choice(template_set["verbs"])
                q = question_template.format(verb)
                r = response_template.format(verb)
            elif "nouns" in template_set:
                noun = random.choice(template_set["nouns"])
                q = question_template.format(noun)
                r = response_template.format(noun)
            else:
                q = question_template
                r = response_template

            f.write(f"{q}|{r}\n")
            count += 1

        # 最后，生成多轮对话
        while count < num_entries:  # 剩下的10%为多轮对话
            turn_sequence = random.choice(multi_turn_templates)
            for i in range(len(turn_sequence) - 1):
                f.write(f"{turn_sequence[i]}|{turn_sequence[i + 1]}\n")
            count += len(turn_sequence) - 1

    print(f"成功生成 {num_entries} 条数据到文件 {filename}。")


if __name__ == "__main__":
    try:
        generate_data(num_entries=20000)  # 将这里的值改大
    except Exception as e:
        print(f"生成数据时发生错误: {e}")
        sys.exit(1)
