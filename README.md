<div align="center">
  <img src="./assets/logo.png" alt="AgentCPM-GUI Logo" width="400em"></img>
</div>
<p align="center">
    【English | <a href="README_zh.md">中文</a>】
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="https://huggingface.co/openbmb/AgentCPM-GUI">Model</a> •
  <a href="#evaluation-data">Evaluation Data</a> •
  <a href="https://arxiv.org/abs/2506.01391">Technical Report</a>
</p>

## News

* [2025-06-03] 📄📄📄 We have released the **technical report** of AgentCPM-GUI! Check it out [here](https://arxiv.org/abs/2506.01391).
* [2025-05-13] 🚀🚀🚀 We have open-sourced **AgentCPM-GUI**, an on-device GUI agent capable of operating Chinese & English apps and equipped with RFT-enhanced reasoning abilities.

## Overview

**AgentCPM-GUI** is an open-source on-device LLM agent model jointly developed by [THUNLP](https://nlp.csai.tsinghua.edu.cn), Renmin University of China and [ModelBest](https://modelbest.cn/en). Built on [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) with 8 billion parameters, it accepts smartphone screenshots as input and autonomously executes user-specified tasks. 

Key features include:

- **High-quality GUI grounding** — Pre-training on a large-scale bilingual Android dataset significantly boosts localization and comprehension of common GUI widgets (buttons, input boxes, labels, icons, etc.).
- **Chinese-app operation** — The first open-source GUI agent finely tuned for Chinese apps, covering 30 + popular titles such as Amap, Dianping, bilibili and Xiaohongshu.
- **Enhanced planning & reasoning** — Reinforcement fine-tuning (RFT) lets the model “think” before outputting an action, greatly improving success on complex tasks.
- **Compact action-space design** — An optimized action space and concise JSON format reduce the average action length to 9.7 tokens, boosting on-device inference efficiency.

Demo Case (1x speed):

https://github.com/user-attachments/assets/694d3c2c-12ce-4084-8feb-4937ca9ad247

## Quick Start

### Install dependencies

```bash
git clone https://github.com/OpenBMB/AgentCPM-GUI
cd AgentCPM-GUI
conda create -n gui_agent python=3.11
conda activate gui_agent
pip install -r requirements.txt
```

### Download the model

Download [AgentCPM-GUI](https://huggingface.co/openbmb/AgentCPM-GUI) from Hugging Face and place it in `model/AgentCPM-GUI`.

#### Huggingface Inference

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import json

# 1. Load the model and tokenizer
model_path = "model/AgentCPM-GUI"  # model path
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.to("cuda:0") 

# 2. Build the input
instruction = "请点击屏幕上的‘会员’按钮"
image_path = "assets/test.jpeg"
image = Image.open(image_path).convert("RGB")

# 3. Resize the longer side to 1120 px to save compute & memory
def __resize__(origin_img):
    resolution = origin_img.size
    w,h = resolution
    max_line_res = 1120
    if max_line_res is not None:
        max_line = max_line_res
        if h > max_line:
            w = int(w * max_line / h)
            h = max_line
        if w > max_line:
            h = int(h * max_line / w)
            w = max_line
    img = origin_img.resize((w,h),resample=Image.Resampling.LANCZOS)
    return img
image = __resize__(image)

# 4. Build the message format
messages = [{
    "role": "user",
    "content": [
        f"<Question>{instruction}</Question>\n当前屏幕截图：",
        image
    ]
}]

# 5. Inference
ACTION_SCHEMA = json.load(open('eval/utils/schema/schema.json', encoding="utf-8"))
items = list(ACTION_SCHEMA.items())
insert_index = 3
items.insert(insert_index, ("required", ["thought"])) # enable/disable thought by setting it to "required"/"optional"
ACTION_SCHEMA = dict(items)
SYSTEM_PROMPT = f'''# Role
你是一名熟悉安卓系统触屏GUI操作的智能体，将根据用户的问题，分析当前界面的GUI元素和布局，生成相应的操作。

# Task
针对用户问题，根据输入的当前屏幕截图，输出下一步的操作。

# Rule
- 以紧凑JSON格式输出
- 输出操作必须遵循Schema约束

# Schema
{json.dumps(ACTION_SCHEMA, indent=None, ensure_ascii=False, separators=(',', ':'))}'''

outputs = model.chat(
    image=None,
    msgs=messages,
    system_prompt=SYSTEM_PROMPT,
    tokenizer=tokenizer,
    temperature=0.1,
    top_p=0.3,
    n=1,
)

# 6. Output
print(outputs)
```

Expected output:

```JSON
{"thought":"任务目标是点击屏幕上的‘会员’按钮。当前界面显示了应用的推荐页面，顶部有一个导航栏。点击‘会员’按钮可以访问应用的会员相关内容。","POINT":[729,69]}
```

Note: AgentCPM-GUI outputs relative coordinates ranging from 0-1000. The conversions are as follows:
```python
rel_x, rel_y = [int(abs_x / width * 1000), int(abs_y / height * 1000)]
abs_x, abs_y = [int(rel_x / 1000 * width), int(rel_y / 1000 * height)]
```
where width and height refer to the original width and height of the image, respectively.

#### vLLM Inference

```bash
# Launch the vLLM server
vllm serve model/AgentCPM-GUI --served-model-name AgentCPM-GUI --tensor_parallel_size 1 --trust-remote-code --limit-mm-per-prompt image=10
```

```python
import base64
import io
import json
import requests
from PIL import Image

END_POINT = "http://localhost:8000/v1/chat/completions"  # Replace with actual endpoint

# system prompt
ACTION_SCHEMA = json.load(open('eval/utils/schema/schema.json', encoding="utf-8"))
items = list(ACTION_SCHEMA.items())
insert_index = 3
items.insert(insert_index, ("required", ["thought"])) # enable/disable thought by setting it to "required"/"optional"
ACTION_SCHEMA = dict(items)
SYSTEM_PROMPT = f'''# Role
你是一名熟悉安卓系统触屏GUI操作的智能体，将根据用户的问题，分析当前界面的GUI元素和布局，生成相应的操作。

# Task
针对用户问题，根据输入的当前屏幕截图，输出下一步的操作。

# Rule
- 以紧凑JSON格式输出
- 输出操作必须遵循Schema约束

# Schema
{json.dumps(ACTION_SCHEMA, indent=None, ensure_ascii=False, separators=(',', ':'))}'''

def encode_image(image: Image.Image) -> str:
    """Convert PIL Image to base64-encoded string."""
    with io.BytesIO() as in_mem_file:
        image.save(in_mem_file, format="JPEG")
        in_mem_file.seek(0)
        return base64.b64encode(in_mem_file.read()).decode("utf-8")

def __resize__(origin_img):
    resolution = origin_img.size
    w,h = resolution
    max_line_res = 1120
    if max_line_res is not None:
        max_line = max_line_res
        if h > max_line:
            w = int(w * max_line / h)
            h = max_line
        if w > max_line:
            h = int(h * max_line / w)
            w = max_line
    img = origin_img.resize((w,h),resample=Image.Resampling.LANCZOS)
    return img

def predict(text_prompt: str, image: Image.Image):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": f"<Question>{text_prompt}</Question>\n当前屏幕截图：(<image>./</image>)"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image)}"}}
        ]}
    ]

    payload = {
        "model": "AgentCPM-GUI",  # Your model name
        "temperature": 0.1,
        "messages": messages,
        "max_tokens": 2048,
    }

    headers = {
        "Content-Type": "application/json",
    }

    response = requests.post(END_POINT, headers=headers, json=payload)
    assistant_msg = response.json()["choices"][0]["message"]["content"]
    return assistant_msg

image = __resize__(Image.open("assets/test.jpeg"))
instruction = "请点击屏幕上的‘会员’按钮"
response = predict(instruction, image)
print(response)
```

### Action Space

At each step, the agent outputs is a single JSON object that contains:
- One (and only one) primitive action, chosen from the list below;
- Optional modifiers (`duration`, `thought`) and/or a task-level flag (`STATUS`).

Note that all keywords are **case-sensitive**, and we use **compact JSON** (i.e., no extra whitespace), which affects the tokenizer’s behavior.

| Action         | Required field(s)                                                                                           | Optional field(s)             | Purpose                                                                     |  Example                                  |
| --------------------- | ----------------------------------------------------------------------------------------------------------- | ----------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------ |
| **Click**               | `POINT:[x,y]`                                                                                               | `duration`,`thought`,`STATUS` | Single tap at the normalized screen coordinate (0–1000, origin = top-left). | `{"POINT":[480,320]}`                            |
| **Long Press**               | `POINT:[x,y]`<br>`duration:1000`                                                                                               | `duration`,`thought`,`STATUS` | Touch-and-hold at coordinate (set a longer duration, e.g. >200 ms). | `{"POINT":[480,320],"duration":1000}`                            |
| **Swipe**      | `POINT:[x,y]`<br>`to:"up" \| "down" \| "left" \| "right"` **or** `to:[x,y]`                                 | `duration`,`thought`,`STATUS` | Swipe from the start point toward a direction **or** another coordinate.     | `{"POINT":[500,200],"to":"down"}` |
| **Press key**         | `PRESS:"HOME" \| "BACK" \| "ENTER"`                                                                         | `duration`,`thought`,`STATUS` | Trigger a hardware / navigation button.                                     | `{"PRESS":"HOME"}`                |
| **Type text**         | `TYPE:"<text>"`                                                                    | `duration`,`thought`,`STATUS` | Insert the given text at the current input focus.                           | `{"TYPE":"Hello, world!"}`                       |
| **Wait**              | `duration`                                                                              | `thought`,`STATUS`            | Idle for the specified time without any other action.                       | `{"duration":500}`                               |
| **Task-level status** | `STATUS:"start" \| "continue" \| "finish" \| "satisfied" \| "impossible" \| "interrupt" \| "need_feedback"` | `duration`,`thought`          | Report task progress; may appear **alone** or **with a primitive action**.  | `{"STATUS":"finish"}`                        |


## Fine-tuning

Source code for SFT and RFT training is provided — see [SFT](sft/readme.md) and [RFT](rft/readme.md).

## Performance Evaluation

### Grounding Benchmark

| Model                     | fun2point      | text2point     | bbox2text      | average        |
| ------------------------- | -------------- | -------------- | -------------- | -------------- |
| **AgentCPM-GUI-8B**       | **79.1**       | **76.5**       | **58.2**       | **71.3**       |
| Qwen2.5-VL-7B             | 36.8           | 52.0           | 44.1           | 44.3           |
| Intern2.5-VL-8B           | 17.2           | 24.2           | 45.9           | 29.1           |
| Intern2.5-VL-26B          | 14.8           | 16.6           | 36.3           | 22.6           |
| OS-Genesis-7B             | 8.3            | 5.8            | 4.0            | 6.0            |
| UI-TARS-7B                | 56.8           | 66.7           | 1.4            | 41.6           |
| OS-Altas-7B               | 53.6           | 60.7           | 0.4            | 38.2           |
| Aguvis-7B                 | 60.8           | **76.5**       | 0.2            | 45.8           |
| GPT-4o                    | 22.1           | 19.9           | 14.3           | 18.8           |
| GPT-4o with Grounding     | 44.3           | 44.0           | 14.3           | 44.2           |

### Agent Benchmark

| Dataset                   | Android Control-Low TM | Android Control-Low EM | Android Control-High TM | Android Control-High EM | GUI-Odyssey TM  | GUI-Odyssey EM  | AITZ TM         | AITZ EM         | Chinese APP (CAGUI) TM  | Chinese APP (CAGUI) EM  |
| ------------------------- | ---------------------- | ---------------------- | ----------------------- | ----------------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| **AgentCPM-GUI-8B** | **94.39**        | **90.20**        | **77.70**         | **69.17**         | **90.85** | **74.96** | **85.71** | **76.38** | **96.86** | **91.28** |
| Qwen2.5-VL-7B             | 92.11                  | 82.12                  | 69.65                   | 57.36                   | 55.33           | 40.90           | 73.16           | 57.58           | 68.53           | 48.80           |
| UI-TARS-7B                | 93.52                  | 88.89                  | 68.53                   | 60.81                   | 78.79           | 57.33           | 71.74           | 55.31           | 71.01           | 53.92           |
| OS-Genesis-7B             | 90.74                  | 74.22                  | 65.92                   | 44.43                   | 11.67           | 3.63            | 19.98           | 8.45            | 38.10           | 14.50           |
| OS-Atlas-7B              | 73.03                  | 67.25                  | 70.36                   | 56.53                   | 91.83*           | 76.76*          | 74.13           | 58.45           | 81.53           | 55.89           |
| Aguvis-7B                 | 93.85                  | 89.40                  | 65.56                   | 54.18                   | 26.71           | 13.54           | 35.71           | 18.99           | 67.43           | 38.20           |
| OdysseyAgent-7B           | 65.10                  | 39.16                  | 58.80                   | 32.74                   | 90.83           | 73.67           | 59.17           | 31.60           | 67.56           | 25.44           |
| GPT-4o                    | -                      | 19.49                  | -                       | 20.80                   | -               | 20.39           | 70.00           | 35.30           | 3.67            | 3.67            |
| Gemini 2.0                | -                      | 28.50                  | -                       | 60.20                   | -               | 3.27            | -               | -               | -               | -               |
| Claude                    | -                      | 19.40                  | -                       | 12.50                   | 60.90           | -               | -               | -               | -               | -               |

> \*Different train/test splits

TM and EM stand for the **Type Match** and **Exact Match**, respectively. All evaluation data and code are open-sourced — see [here](eval) for details.

## Evaluation Data

We provide **CAGUI**, an evaluation benchmark for Chinese apps covering **grounding** and **agent** tasks.
See the dataset on [Hugging Face](https://huggingface.co/datasets/openbmb/CAGUI).

## FAQs

Click here to view the [FAQs](https://github.com/OpenBMB/AgentCPM-GUI/blob/main/eval/README.md#faqs).

## Trends

<a href="https://star-history.com/#OpenBMB/AgentCPM-GUI&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=OpenBMB/AgentCPM-GUI&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=OpenBMB/AgentCPM-GUI&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=OpenBMB/AgentCPM-GUI&type=Date" />
 </picture>
</a>

## License

* Code in this repository is released under the [Apache-2.0](./LICENSE) license.

## Citation

If **AgentCPM-GUI** is useful for your research, please cite:

```bibtex
@article{zhang2025agentcpmgui,
      title={Agent{CPM}-{GUI}: Building Mobile-Use Agents with Reinforcement Fine-Tuning}, 
      author={Zhong Zhang and Yaxi Lu and Yikun Fu and Yupeng Huo and Shenzhi Yang and Yesai Wu and Han Si and Xin Cong and Haotian Chen and Yankai Lin and Jie Xie and Wei Zhou and Wang Xu and Yuanheng Zhang and Zhou Su and Zhongwu Zhai and Xiaoming Liu and Yudong Mei and Jianming Xu and Hongyan Tian and Chongyi Wang and Chi Chen and Yuan Yao and Zhiyuan Liu and Maosong Sun},
      year={2025},
      journal={arXiv preprint arXiv:2506.01391},
}
```
