# QwenChat

## 1. 类的实现


```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

class QwenChat:
    def __init__(self, model_name_or_path, max_length=1024):
        """
        初始化模型和分词器，并创建对话历史记录。
        """
        self.model, self.tokenizer = self.load_model(model_name_or_path)
        self.history = []  # 保存对话历史
        self.max_length = max_length

    def load_model(self, model_name_or_path):
        """
        加载模型和分词器。
        """
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return model, tokenizer

    def chat(self, user_input, max_new_tokens=512):
        """
        处理用户输入，生成对话回复，并更新历史记录。
        """
        # 将用户输入添加到历史
        self.history.append({"role": "user", "content": user_input})
        if len(self.history) > self.max_length:
            self.history = self.history[-self.max_length:]

        # 构建模型输入（包括历史）
        messages = [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}] + self.history
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # 模型生成回复
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # 将模型回复添加到历史
        self.history.append({"role": "assistant", "content": response})

        return response

    def clear_history(self):
        """
        清空对话历史。
        """
        self.history = []

    def save_history(self, path):
        """
        保存对话历史。
        """
        with open(path, "w") as f:
            json.dump(self.history, f)

    def load_history(self, path):
        """
        加载对话历史。
        """
        with open(path, "r") as f:
            self.history = json.load(f)
```

    /home/wangxincheng/miniconda3/envs/llm/lib/python3.9/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


## 2. 交互示例


```python
# 初始化对话
chatbot = QwenChat("../HF/Qwen2.5-7B-Instruct")
```

    Loading checkpoint shards: 100%|██████████| 4/4 [00:05<00:00,  1.28s/it]



```python
# 进行一次对话
response = chatbot.chat("你好，请你介绍一下自己。")
print(response)
```

    你好！我是Qwen，由阿里云开发的AI助手。我的主要功能是提供语言理解和生成能力，帮助用户解答问题、创作文字内容等。无论你需要获取信息、编写文章还是进行创意构思，我都在这里为你提供支持。有什么我可以帮助你的吗？



```python
# 继续对话
response = chatbot.chat("你的名字是怎么来的？")
print(response)
```

    我的名字“Qwen”来源于“Question”的谐音，寓意着我能够像回答问题一样快速准确地提供帮助和支持。这个名字简洁明了，易于记忆，同时也体现了我的核心功能之一——回答和解决用户的问题。希望这个名字能够让你更容易地记住我，并且感受到我的友好与便捷。如果你有任何问题或需要帮助，随时可以来找我哦！



```python
# 继续对话
response = chatbot.chat("同济大学你知道吗？")
print(response)
```

    当然知道！同济大学是中国著名的高等学府，位于上海市。它是一所综合性大学，拥有悠久的历史和深厚的学术底蕴。同济大学在国内外享有很高的声誉，在工程、建筑、医学、汽车制造等多个领域都有着卓越的研究成果和广泛的影响力。
    
    同济大学设有多个学院和研究机构，涵盖了理、工、文、法、医、经济、管理、教育、艺术等学科门类。学校还与众多国际知名高校和企业有着密切的合作关系，为学生提供了丰富的国际交流和实践机会。
    
    如果你有关于同济大学的具体问题，比如校园生活、学术研究、招生信息等，都可以告诉我，我会尽力为你提供帮助。



```python
# 继续对话
response = chatbot.chat("我考 500 分能否上这一所大学？")
print(response)
```

    能否考上同济大学取决于多个因素，包括但不限于你的省份、具体的招生专业以及当年的招生计划和分数线。在中国，高考成绩是决定能否被大学录取的重要依据之一，但不是唯一的因素。以下几点可以帮助你更好地了解情况：
    
    1. **省份和批次**：不同省份的分数线不同，同济大学在各个省份的录取分数线也会有所差异。有些省份可能有多个批次（如一本、二本等），不同批次的分数线也有所不同。
    
    2. **具体专业**：同济大学的专业录取分数线也可能不同，一些热门专业或强势学科可能会有更高的录取要求。
    
    3. **招生计划**：每年的招生计划可能会有所调整，这会影响到实际录取人数和分数线。
    
    4. **综合素质评价**：部分省份和高校还会参考学生的综合素质评价，包括平时成绩、社会实践、特长等。
    
    如果你的高考成绩是500分，建议你：
    - 查看同济大学官网或招生办公室发布的最新招生信息。
    - 参考近几年同济大学在你所在省份的录取分数线。
    - 考虑是否参加提前批或特殊类型招生考试，这些可能会有不同的录取标准。
    - 咨询当地的教育考试院或学校老师，获取更具体的信息。
    
    如果你有具体的省份和专业信息，我可以帮你查找更详细的数据。



```python
# 继续对话
response = chatbot.chat(" 那如果是 650 分呢？")
print(response)
```

    如果高考成绩是650分，那么你被同济大学录取的可能性会大大增加，但仍需考虑以下几个因素：
    
    1. **省份和批次**：不同省份的分数线和录取规则可能有所不同。一般来说，650分在大多数省份已经超过了同济大学的一本线，但在一些竞争特别激烈或分数线较高的省份，可能仍需进一步评估。
    
    2. **具体专业**：同济大学的部分热门专业和强势学科可能会有更高的录取分数线。例如，建筑工程、机械工程、材料科学与工程等专业通常会有较高的录取要求。
    
    3. **历年数据**：你可以查阅同济大学过去几年在你所在省份的实际录取分数线，以获得更准确的参考。
    
    4. **招生政策**：每年的招生政策和计划可能会有所不同，确保了解最新的招生信息。
    
    以下是几个具体的步骤，帮助你更好地评估：
    
    1. **查询历年分数线**：
       - 访问同济大学官方网站，查找招生简章和历年录取分数线。
       - 查看你所在省份教育考试院发布的数据。
    
    2. **咨询学校**：
       - 直接联系同济大学招生办公室，获取最新的招生信息和专业分数线。
       - 咨询当地中学的招生办老师，他们通常会掌握最新的信息。
    
    3. **模拟填报志愿**：
       - 使用各地教育考试院提供的模拟填报系统，输入你的分数和意向专业，查看预估的录取可能性。
    
    4. **关注招生动态**：
       - 关注同济大学的官方社交媒体账号和招生公众号，及时获取最新的招生信息和动态。
    
    如果你能提供更多具体信息（如所在省份、目标专业），我可以为你提供更加详细的分析和建议。



```python
# 查看历史记录
print(chatbot.history)
```

    [{'role': 'user', 'content': '你好，请你介绍一下自己。'}, {'role': 'assistant', 'content': '你好！我是Qwen，由阿里云开发的AI助手。我的主要功能是提供语言理解和生成能力，帮助用户解答问题、创作文字内容等。无论你需要获取信息、编写文章还是进行创意构思，我都在这里为你提供支持。有什么我可以帮助你的吗？'}, {'role': 'user', 'content': '你的名字是怎么来的？'}, {'role': 'assistant', 'content': '我的名字“Qwen”来源于“Question”的谐音，寓意着我能够像回答问题一样快速准确地提供帮助和支持。这个名字简洁明了，易于记忆，同时也体现了我的核心功能之一——回答和解决用户的问题。希望这个名字能够让你更容易地记住我，并且感受到我的友好与便捷。如果你有任何问题或需要帮助，随时可以来找我哦！'}, {'role': 'user', 'content': '同济大学你知道吗？'}, {'role': 'assistant', 'content': '当然知道！同济大学是中国著名的高等学府，位于上海市。它是一所综合性大学，拥有悠久的历史和深厚的学术底蕴。同济大学在国内外享有很高的声誉，在工程、建筑、医学、汽车制造等多个领域都有着卓越的研究成果和广泛的影响力。\n\n同济大学设有多个学院和研究机构，涵盖了理、工、文、法、医、经济、管理、教育、艺术等学科门类。学校还与众多国际知名高校和企业有着密切的合作关系，为学生提供了丰富的国际交流和实践机会。\n\n如果你有关于同济大学的具体问题，比如校园生活、学术研究、招生信息等，都可以告诉我，我会尽力为你提供帮助。'}, {'role': 'user', 'content': '我考 500 分能否上这一所大学？'}, {'role': 'assistant', 'content': '能否考上同济大学取决于多个因素，包括但不限于你的省份、具体的招生专业以及当年的招生计划和分数线。在中国，高考成绩是决定能否被大学录取的重要依据之一，但不是唯一的因素。以下几点可以帮助你更好地了解情况：\n\n1. **省份和批次**：不同省份的分数线不同，同济大学在各个省份的录取分数线也会有所差异。有些省份可能有多个批次（如一本、二本等），不同批次的分数线也有所不同。\n\n2. **具体专业**：同济大学的专业录取分数线也可能不同，一些热门专业或强势学科可能会有更高的录取要求。\n\n3. **招生计划**：每年的招生计划可能会有所调整，这会影响到实际录取人数和分数线。\n\n4. **综合素质评价**：部分省份和高校还会参考学生的综合素质评价，包括平时成绩、社会实践、特长等。\n\n如果你的高考成绩是500分，建议你：\n- 查看同济大学官网或招生办公室发布的最新招生信息。\n- 参考近几年同济大学在你所在省份的录取分数线。\n- 考虑是否参加提前批或特殊类型招生考试，这些可能会有不同的录取标准。\n- 咨询当地的教育考试院或学校老师，获取更具体的信息。\n\n如果你有具体的省份和专业信息，我可以帮你查找更详细的数据。'}, {'role': 'user', 'content': ' 那如果是 650 分呢？'}, {'role': 'assistant', 'content': '如果高考成绩是650分，那么你被同济大学录取的可能性会大大增加，但仍需考虑以下几个因素：\n\n1. **省份和批次**：不同省份的分数线和录取规则可能有所不同。一般来说，650分在大多数省份已经超过了同济大学的一本线，但在一些竞争特别激烈或分数线较高的省份，可能仍需进一步评估。\n\n2. **具体专业**：同济大学的部分热门专业和强势学科可能会有更高的录取分数线。例如，建筑工程、机械工程、材料科学与工程等专业通常会有较高的录取要求。\n\n3. **历年数据**：你可以查阅同济大学过去几年在你所在省份的实际录取分数线，以获得更准确的参考。\n\n4. **招生政策**：每年的招生政策和计划可能会有所不同，确保了解最新的招生信息。\n\n以下是几个具体的步骤，帮助你更好地评估：\n\n1. **查询历年分数线**：\n   - 访问同济大学官方网站，查找招生简章和历年录取分数线。\n   - 查看你所在省份教育考试院发布的数据。\n\n2. **咨询学校**：\n   - 直接联系同济大学招生办公室，获取最新的招生信息和专业分数线。\n   - 咨询当地中学的招生办老师，他们通常会掌握最新的信息。\n\n3. **模拟填报志愿**：\n   - 使用各地教育考试院提供的模拟填报系统，输入你的分数和意向专业，查看预估的录取可能性。\n\n4. **关注招生动态**：\n   - 关注同济大学的官方社交媒体账号和招生公众号，及时获取最新的招生信息和动态。\n\n如果你能提供更多具体信息（如所在省份、目标专业），我可以为你提供更加详细的分析和建议。'}]



```python
# 清空历史记录
chatbot.clear_history()
```


```python
# 查看历史记录
print(chatbot.history)
```

    []



```python

```
