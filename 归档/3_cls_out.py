import json
import torch
AdamW = torch.optim.AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

ft_2 = {
    'non-hate': 0,
    'Region': 1,
    'Sexism': 2,
    'Racism': 3,
    'LGBTQ': 4,
    'Hivism': 5,
}
# 修改模型预测函数，返回多个标签
def predict_hate_types(text, model, tokenizer, device, threshold=0.5, LLMhate=None):
    model.eval()
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # 使用sigmoid函数获取每个类别的概率
    predictions = torch.sigmoid(outputs.logits)
    predictions = (predictions > threshold).float()[0]  # 应用阈值并转换为一维
    # 获取预测为1的所有标签
    predicted_labels = []
    for i, pred in enumerate(predictions):
        if pred == 1:
            # 获取标签名称（通过反向查找ft_2字典）
            for key, val in ft_2.items():
                if val == i:
                    predicted_labels.append(key)
    if len(predicted_labels) == 0:
        # 寻找当前情况下哪一个标签的概率最大
        max_index = torch.argmax(predictions).item()
        for key, val in ft_2.items():
            if val == max_index:
                predicted_labels.append(key)
    if LLMhate=='是' and 'non-hate' in predicted_labels:
        # 从新寻找一个概率最大的标签 但是不包括non-hate
        max_index = torch.argmax(predictions[1:]).item() + 1
        for key, val in ft_2.items():
            if val == max_index:
                predicted_labels = [key]

    return predicted_labels

tokenizer = AutoTokenizer.from_pretrained('hate_speech_model+4hate_speech_model+3')
model = AutoModelForSequenceClassification.from_pretrained('hate_speech_model+4hate_speech_model+3', num_labels=len(ft_2))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

data = []
with open('test_2.json', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        temp = eval(line.strip())
        results = eval(temp['results'])
        for i in results:
            data.append({
                'content': temp['content'],
                'id': temp['id'],
                'Target': i['指代'],
                'Argument': i['评论'],
                'LLMhate': i['是否仇恨'],
                'tr': temp['content'] + '[SEP]' + i['评论'] + ':' +  i['指代'],
            })

with open('test_cls_3_out.txt', 'w', encoding='utf-8') as f:
    for i in data:
        hate_types = predict_hate_types(i['tr'], model, tokenizer, device,LLMhate=i['LLMhate'])
        temp = {}
        temp['content'] = i['content']
        temp['id'] = i['id']
        temp['Target'] = i['Target']
        temp['Argument'] = i['Argument']
        temp['LLMhate'] = i['LLMhate']
        temp['results_is'] = hate_types
        f.write(json.dumps(temp, ensure_ascii=False) + '\n')