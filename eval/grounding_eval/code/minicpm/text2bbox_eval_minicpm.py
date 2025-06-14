import json
from PIL import Image
import base64
import re
from openai import AsyncClient
import os
import asyncio
import io

def read_jsonl(file_path):
    """
    读取 JSONL 文件并解析为 Python 字典列表
    :param file_path: JSONL 文件路径
    :return: 包含所有 JSON 对象的列表
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def load_image_from_path(image_path):
    """
    从指定路径加载图片
    :param image_path: 图片文件路径
    :return: PIL.Image 对象
    """
    try:
        image = Image.open(image_path)
        return image
    except Exception as e:
        print(f"Error loading image from {image_path}: {e}")
        return None
    
# Function to encode the image
def encode_image(image_path):
    image = Image.open(image_path)
    image = resize(image)
    w, h = image.width, image.height
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")  # 保存为 JPEG 格式
    image_bytes = buffered.getvalue()
    
    # 对字节流进行 Base64 编码
    base64_encoded = base64.b64encode(image_bytes).decode("utf-8")
    return base64_encoded, w, h

def resize(origin_img):
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
    
async def call_Qwenvl(item, client, model_name):
    """
    调用Qwen输出function的描述，输出bbox
    """
    # sys_prompt = '''你是一个GUI组件定位的专家，擅长输出图片上文本对应的坐标。你的任务是根据给定的GUI截图和图中某个文本输出该文本的坐标。
    # 输入：屏幕截图，文本描述
    # 输出：文本的绝对坐标的中心点，以<x,y>为格式，使用<>定位，其中不能存在任何非坐标字符，注意中心点应当是两个坐标而不是四个。
    # 示例输出一：我认为该文本在<600,1000>附近
    # 示例输出二：该文本的位置是<1238,430>'''
    sys_prompt = '''
    你是一个GUI组件定位的专家，擅长输出图片上文本对应的坐标。你的任务是根据给定的GUI截图和图中某个文本输出该文本的坐标。\n    输入：屏幕截图，文本描述\n    输出：文本的相对坐标的中心点,{\"POINT\":[...,...]}为格式
    '''
    image_path = item["image"]
    base64_image, w, h = encode_image(image_path)
    content = []
    # 动态添加base64_image部分到 content 列表
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}",
        },
    })
    content.append({
        "type": "text",
        "text": "屏幕上的文本：## Text\n{}\n".format(item['text'])
    })

    res = await client.chat.completions.create(
        messages=[
                        {
                "role": "system",
                "content": sys_prompt,
            },
            {
                "role":"user",
                "content": content,
            }
        ],
        model=model_name,
        temperature=0
    )

    response = res.choices[0].message.content
    return response.strip("\n").replace(" ","")

async def verify(response, ground_truth, w ,h):
    """
    接受模型的字符串输入，判断是否正确，当前判断方法：点是否落在bbox内
    """
    try:
        json_action = json.loads(response)
        bbox = json_action["POINT"]
        abs_bbox = [bbox[0]/1000*w, bbox[1]/1000*h]
        gt_bbox = list(map(int, ground_truth.strip('<>').split(',')))
        gt_x_min = gt_bbox[0]
        gt_x_max = gt_bbox[2]
        gt_y_min = gt_bbox[1]
        gt_y_max = gt_bbox[3]
        if gt_x_min<=abs_bbox[0]<=gt_x_max and gt_y_min<=abs_bbox[1]<=gt_y_max:
            return 1
    except Exception as e:
        print("wrong response: {}".format(response))
    return 0

async def process_item_async(item, client, model_name, semaphore):
    async with semaphore:
        image = Image.open(item["image"])
        w = image.width
        h = image.height
        bbox = await call_Qwenvl(item, client, model_name)
        correct = await verify(bbox,item["abs_position"],w,h)
        return correct

async def main():
    model_name = "minicpm"
    total = 0
    correct = 0
    json_data_path = "your/path/to/the/dataset"
    data = read_jsonl(json_data_path)
    client = AsyncClient(api_key="sk-123", base_url='http://localhost:8000/v1')
    semaphore = asyncio.Semaphore(8)
    tasks = []
    for item in data:
        task = asyncio.create_task(process_item_async(item, client, model_name,semaphore))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    for result in results:
        correct += result
        total += 1
    print(correct, total, correct / total)
    return 0

if __name__=="__main__":
    asyncio.run(main())
