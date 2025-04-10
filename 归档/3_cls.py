import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
AdamW = torch.optim.AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import json

# BERT = 'hfl/chinese-roberta-wwm-ext-large'
BERT = 'hate_speech_model+3'
BATCH_SIZE = 16
LR = 9e-6
MAX_LEN = 256
DELSIMGOTH = True
DELALLOTH = True
IFSEP = False
epochs = 5


print('BERT:', BERT)
print('BATCH_SIZE:', BATCH_SIZE)
print('LR:', LR)
print('DELSIMGOTH:', DELSIMGOTH, 'DELALLOTH:', DELALLOTH)
print('IFSEP:', IFSEP)

ft_2 = {
    'non-hate': 0,
    'Region': 1,
    'Sexism': 2,
    'Racism': 3,
    'LGBTQ': 4,
    'Hivism': 5,
}
if DELALLOTH == False:
    ft_2['others'] = 6
def output_tf(ssste, content):
    templist = ssste.replace(' ', '').replace('[END]', '').split('[SEP]')
    out = []
    for i in templist:
        temp = i.split('|')
        if len(temp) != 4:
            print('格式错误')
            return None
        tempss = temp[2].strip().split(',')
        cliss = []
        for i in tempss:
            if DELALLOTH and i == 'others':
                continue
            sssstemp = ft_2[i]
            cliss.append(sssstemp)
        SE = ''
        if IFSEP:
            SE = '[SEP]'
        else:
            SE = ':'
        out.append({
            'tr': content + '[SEP]' + temp[0].strip() + SE +  temp[1].strip(),
            'cls': cliss,
        })
    return out

# 修改数据集类，支持多标签分类
class HateDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_labels = len(ft_2)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label_indices = self.labels[idx]
        
        # 创建多标签向量（多热编码）
        label_vector = torch.zeros(self.num_labels)
        for label_idx in label_indices:
            label_vector[label_idx] = 1.0
        
        # 先获取完整的token长度（不截断）
        full_encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            padding=False
        )
        
        original_length = len(full_encoding['input_ids'])
        
        # 再获取截断后的encoding
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # # 计算被截断的token数量
        # truncated_tokens = max(0, original_length - self.max_len)
        # if truncated_tokens > 0:
        #     print('truncated_tokens:', truncated_tokens)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': label_vector,
            'original_length': original_length     # 原始token长度
        }

# 数据预处理函数
def preprocess_data(data):
    texts = []
    labels = []
    
    for item in data:
        temp = item['output']
        temp = output_tf(temp,item['content'])
        if DELSIMGOTH:
            if len(temp[0]['cls']) != 0 and len(temp[0]['cls']) == 1 and temp[0]['cls'][0] == 5:
                continue
        for i in temp:
            texts.append(i['tr'])
            labels.append(i['cls'])
    return texts, labels

# 加载数据
with open('train_hiv.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 预处理数据
# ==================================================================================================
texts, labels = preprocess_data(data)
# train_texts, val_texts, train_labels, val_labels = train_test_split(
    # texts, labels, test_size=0.1, random_state=222
# )
train_texts = texts
val_texts = texts
train_labels = labels
val_labels = labels
# ==================================================================================================
tokenizer = BertTokenizer.from_pretrained(BERT)
model = BertForSequenceClassification.from_pretrained(
    BERT,
    num_labels=len(ft_2),
    problem_type="multi_label_classification"  # 指定为多标签分类问题
)
train_dataset = HateDataset(train_texts, train_labels, tokenizer,max_len=MAX_LEN)
val_dataset = HateDataset(val_texts, val_labels, tokenizer,max_len=MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=LR, eps=1e-8)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    losses = []
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)  # 现在是多热编码向量
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    return np.mean(losses)
def evaluate(model, dataloader, device, threshold=0.5):
    model.eval()
    losses = []
    all_predictions = []
    all_labels = []
    exact_match_count = 0  # 添加完全匹配计数
    total_samples = 0      # 总样本数
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            losses.append(loss.item())
            predictions = torch.sigmoid(outputs.logits)
            predictions = (predictions > threshold).float()
            batch_size = labels.shape[0]
            total_samples += batch_size
            for i in range(batch_size):
                # print(f"预测: {predictions[i]}, 实际: {labels[i]}")
                # 检查是否完全匹配（所有标签都预测正确，没有多余也没有缺少）
                if torch.all(predictions[i] == labels[i]):
                    exact_match_count += 1
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    exact_match_ratio = exact_match_count / total_samples  # 计算完全匹配率
    return np.mean(losses), np.array(all_predictions), np.array(all_labels), exact_match_ratio
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    print('-' * 10)
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    print(f"当前学习率: {scheduler.get_last_lr()[0]}")
    print(f'Train loss: {train_loss}')
    val_loss, predictions, actual_labels, exact_match_ratio = evaluate(model, val_loader, device)
    print(f'Validation loss: {val_loss}')
    f1_micro = f1_score(actual_labels, predictions, average='micro')
    f1_macro = f1_score(actual_labels, predictions, average='macro')
    f1_per_class = f1_score(actual_labels, predictions, average=None)
    print(f'F1 Score (Micro): {f1_micro:.4f}')
    print(f'F1 Score (Macro): {f1_macro:.4f}')
    print(f'**Exact Match Ratio**: {exact_match_ratio:.4f}')  # 添加完全匹配率的输出
    print('F1 Score per class:')
    for i, label_name in enumerate(ft_2.keys()):
        print(f'  {label_name}: {f1_per_class[i]:.4f}')
    # 保存模型
    model.save_pretrained('./hate_speech_model+' + str(epoch)+ BERT)
    tokenizer.save_pretrained('./hate_speech_model+' + str(epoch)+ BERT)
    print()


# # 修改模型预测函数，返回多个标签
# def predict_hate_types(text, model, tokenizer, device, threshold=0.5):
#     model.eval()
#     encoding = tokenizer(
#         text,
#         add_special_tokens=True,
#         max_length=128,
#         truncation=True,
#         padding='max_length',
#         return_tensors='pt'
#     )
    
#     input_ids = encoding['input_ids'].to(device)
#     attention_mask = encoding['attention_mask'].to(device)
    
#     with torch.no_grad():
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
#     # 使用sigmoid函数获取每个类别的概率
#     predictions = torch.sigmoid(outputs.logits)
#     predictions = (predictions > threshold).float()[0]  # 应用阈值并转换为一维
    
#     # 获取预测为1的所有标签
#     predicted_labels = []
#     for i, pred in enumerate(predictions):
#         if pred == 1:
#             # 获取标签名称（通过反向查找ft_2字典）
#             for key, val in ft_2.items():
#                 if val == i:
#                     predicted_labels.append(key)
    
#     return predicted_labels

# # 测试模型
# test_texts = [
#     "这些人真是该死，他们不属于我们的国家。",
#     "女人就应该待在家里照顾孩子，不要去工作。",
#     "我觉得大家应该相互尊重，无论性别、种族或性取向。"
# ]

# for text in test_texts:
#     hate_types = predict_hate_types(text, model, tokenizer, device)
#     print(f"文本: '{text}'")
#     if hate_types:
#         print(f"预测仇恨类型: {', '.join(hate_types)}")
#     else:
#         print("预测结果: 无仇恨内容")
#     print()