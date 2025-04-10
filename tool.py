import difflib
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List, Tuple, Dict, Any
import requests
import json
import pandas as pd
from IPython.display import display
from openai import OpenAI
import json

import difflib
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List, Tuple, Dict, Any
import requests
import json
import pandas as pd
import json
from IPython.display import display
from openai import OpenAI
from tqdm import tqdm
import httpx


def soft_match_score(pred_Target , gold_Target, pred_Argument , gold_Argument):
    target_sim = difflib.SequenceMatcher(None, pred_Target, gold_Target).ratio()
    arg_sim = difflib.SequenceMatcher(None, pred_Argument, gold_Argument).ratio()
    return target_sim >= 0.5 and arg_sim >= 0.5

ft = {
    'non-hate': 'n',
    'hate': 'y',
}
def output_tf(ssste):
    templist = ssste.replace(' ', '').replace('[END]', '').split('[SEP]')
    out = []
    for i in templist:
        temp = i.split('|')
        if len(temp) != 4:
            print('格式错误')
            return None
        out.append({
            'Target': temp[0].strip(),
            'Argument': temp[1].strip(),
            'HateType': temp[2].strip(),
            'Is': ft[temp[3].strip()]
        })
    return out

def filter(pred_quads):
    for i in range(len(pred_quads)):
        temp = pred_quads[i]['Argument']
        # 删除所有中英文符号和空格
        temp = temp.replace(' ', '').replace('，', '').replace('。', '').replace('！', '').replace('？', '').replace('“', '').replace('”', '').replace('‘', '').replace('’', '').replace(',', '').replace('.', '').replace('!', '').replace('?', '')
        pred_quads[i]['Argument'] = temp
    return pred_quads
        

def merge(pred_quads,content):
    if len(pred_quads) <= 1:
        return pred_quads
    
     # 按Target值分组
    target_groups = {}
    for quad in pred_quads:
        target = quad['Target']
        if target not in target_groups:
            target_groups[target] = []
        target_groups[target].append(quad)
    # 只保留有重复Target的字典组
    result = []
    for target, quads in target_groups.items():
        if len(quads) > 1:  # 只处理重复出现的Target
            result.extend(quads)
    if len(result) == 0:
        return []
    # print('^^^^^',result)
    return result


[['默默','沉默'],[3907,4085,2595,1474]]


slang_replace = {
'🐶':'狗',
'xing奴':'性奴',
'腻哥':'nigger',
'黑棒':'黑人鸡巴',
'gun':'滚',
'默@沉默':'nigger ',
'easy girl':'婊子',
'黑s会':'黑社会',
'亩狗':'母狗',
'女quan':'女拳',
'郭嘉':'国家',
'执fa':'执法'
}


slang = {
"男拳":"谐音“男权”指极端男权主义者",
"国男":"中国男性的蔑称",
"金针菇":"对男性生殖器的蔑称",
"蝻":"对男性的蔑称",
"黄男":"对亚洲男性的蔑称",
"直男":"异性恋男性或是女性对男性的蔑称",
"龟男":"男性对积极追求女性的男性的蔑称",
"舔狗":"男性对积极追求女性的男性的蔑称",
"公猪":"对亚洲男性的蔑称",
"龟奴":"男性对积极追求女性的男性的蔑称",
"女拳":"谐音“女权”指极端女权主义者",
"黄女":"对亚洲女性的蔑称",
"幕刃":"对女性的蔑称",
"猪精":"对肥胖女性的蔑称",
"肥猪":"对肥胖女性的蔑称",
"坦克":"对肥胖女性的蔑称",
"母人":"对女性的蔑称",
"姆":"女性对家庭主妇的蔑称",
"幕刃":"对女性的蔑称",
"母朱":"对肥胖女性的蔑称",
"母狗":"对女性的蔑称",
"母拳":"对女权主义的蔑称",
"母苟":"对女性的蔑称",
"母猩猩":"对女性的蔑称",
"母畜":"对女性的蔑称",
"母√":"对女性的蔑称",
"母豖":"对女性的蔑称",
"妈宝":"对婚姻中听从母亲的伴侣的蔑称",
"玛利亚":"对不顾现实发表过高道德匹配的人的蔑称",
"圣母":"对不顾现实发表过高道德匹配的人的蔑称",
"芶":"谐音狗用于侮辱",
"反同":"指反对同性恋",
"掰弯":"指将异性恋转变为同性恋",
"弯":"指同性恋",
"B站":"指某个年轻人聚集的社交平台",
"默":"对黑人群体的歧视用语",
"直佬":"同性恋对异性恋的蔑称",
"黄皮":"对亚洲人的蔑称",
"白皮":"对白人的蔑称",
"黑皮":"对黑人的蔑称",
"学伴":"指一社会事件-某学校安排女学生与黑人留学生结伴",
"接盘":"指女孩年轻时追求浪漫开放的性关系之后想要结婚生子找一个愿意接盘的传统型男人",
"普信男":"指自以为是的男性",
"仙女":"指展现出极端自我中心双重标准的年轻女性",
"打拳":"指进行极端的女权主义活动",
"拳师":"指极端女权主义者",
"普信女":"指自以为是的女性",
"女quan":"指极端女权主义者",
"拳":"指进行性别主义活动",
"铁t":"指女同性恋",
"az":"指艾滋",
"艾":"指艾滋",
"通讯录":"指同性恋",
"txl":"指同性恋",
"洗头佬":"指希特勒",
"媚外":"指对外国人或外国文化的过度崇拜和迎合",
"媚黑":"指对黑人的过度崇拜和迎合",
"媚白":"指对白人的过度崇拜和迎合",
"媚洋":"指对国外人或事物的过度崇拜和迎合",
"奇趣蛋":"对肥胖女性的蔑称",
'撑同':'指支持同性恋',
'反同':'指反对同性恋',
'女⭕️':'谐音女权',
'eg':'Easy Girl的简写,容易上床的女孩的蔑称',
'饭桶':'谐音反同',
'反串':'指网络上假装成对立立场的角色',
}
# (其中‘掰弯’指将异性恋转变为同性恋)

def add_slang_prompt(content):
    print('已经废弃，已经废弃，已经废弃，已经废弃，已经废弃，已经废弃，已经废弃，已经废弃，已经废弃')
    exit(0)
    OUT = ''
    for key, value in slang.items():
        if key in content:
            OUT += "(其中‘" + key + '’是' + value + ")\n"
    return OUT

PRO = {}
PRO['第一任务'] = """提取句中最核心包含作者主观评论的指代（至少一个，至多两个），再提取句中对该指代评论的关键信息片段，再进行仇恨判断任务，输出‘指代’、‘评论’、‘是否仇恨’（是/否）
**示例**
输入：我在办公室倒是见过一个黑人找了个没脑子的中国妹子，还生了个孩子，有一天还把孩子带办公室来了。黑皮肤小卷毛的小孩，那个中国妹子长得也不咋样。那个黑人在办公室还主动跟我们打招呼，没人搭理他，连回应都不给那种，男同事们对他满眼全是鄙视。那个嫁老黑的中国妹推车里推个孩子，也没人搭理她。
输出：
[
{'指代': '没脑子的中国妹子','评论': '长得也不咋样','是否仇恨': '是'},
{'指代': '黑人','评论': '鄙视','是否仇恨': '是'}
]
"""


PRO['第二任务'] = """判断作者对‘评价对象’的主观‘评论内容’内容是否包含仇恨信息，同时提供了‘原句’用于参考。
如果‘仇恨判断’为是，再进一步判断其是针对何种群体的仇恨‘仇恨类型’（地域/种族/性别/同性恋/其他）,输出JSON格式{'仇恨判断'：‘是/否’，‘仇恨类型’：‘地域/种族/性别/同性恋/其他’}
"""

PRO['第三任务'] = """这里提供了一个句子成分提取任务介绍和示例，你不需要完成这个任务，根据示例的输入和输出去分析的提取过程，分析仅能包含三部分，俚语分析、语义分析、仇恨目标判断。
**任务介绍**
从句子中识别出作者表达仇恨的群体或个人。仇恨评论通常带有贬义、侮辱性或歧视性，针对特定群体或个人。
**示例**
"""


PRO['第四任务'] = """这里提供了一个'提取对某目标的仇恨语句'任务介绍和示例，你不需要完成这个任务，根据示例去分析的提取过程，分析仅能包含三部分，俚语分析、语义分析、语句片段提取。
**任务介绍**
从'句子'中抽取出作者对'仇恨目标'表达仇恨的关键'仇恨语句片段'
**示例**
"""

def assembly_prompt(content,work,isSlang):
    if work == 1:
        prompt = PRO['第一任务']
        prompt += "**任务**\n"
        if isSlang:
            prompt += add_slang_prompt(content)
        user += "\n输入：" + content + '\n输出：'
    elif work == 2:
        prompt = PRO['第二任务']
        prompt += "**任务**\n"
        if isSlang:
            prompt += add_slang_prompt(content['Target'])
        prompt += '评价对象：' + content['Target'] + '\n'
        prompt += '评论内容：' + content['Argument'] + '\n'
        prompt += '原句：' + content['content'] + '\n'
    elif work == 3:
        prompt = PRO['第三任务']
        prompt += ('输入:' + content + '\n' )
        prompt += ('输出:' + isSlang )
    elif work == 4:
        prompt = PRO['第四任务']
        prompt += ('句子:' + content['content'] + '\n' )
        prompt += ('仇恨目标:' + content['Target'] + '\n' )
        prompt += ('仇恨语句片段:' + content['Argument'] + '\n' )
    else:
        print('work error')
    return prompt
    

# 创建一个可配置代理的OpenAI客户端
def create_openai_client(api_key, base_url, proxy=None):
    client_kwargs = {"api_key": api_key, "base_url": base_url}
    if proxy:
        # 直接将代理URL字符串传递给httpx客户端
        http_client = httpx.Client(proxy=proxy)
        client_kwargs["http_client"] = http_client
    return OpenAI(**client_kwargs)

# 默认客户端（不使用代理）
def OpenAi_api(key,content, work, isSlang, log=False,proxy=None):
    prompt = assembly_prompt(content, work, isSlang)
    if proxy:
        client_to_use = create_openai_client(api_key=key,base_url="https://api.deepseek.com",proxy=proxy)
    else:
        client_to_use = create_openai_client(api_key=key,base_url="https://api.deepseek.com")

    response = client_to_use.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    
    if log:
        print(prompt, response)
    return response


def load_train_byWORKTYPE(filename,WORKTYPY):
    # 加载训练数据
    with open(filename, 'r', encoding='utf-8') as file:
        data = []
        if WORKTYPY == 1:
            data = json.load(file)
        elif WORKTYPY == 2:
            for line in file:
                temp = json.loads(line)
                try:
                    results = eval(temp['results'])
                except:
                    print('eval error')
                    print(temp['results'])
                for i in results:
                    data.append({
                        'content': temp['content'],
                        'id': temp['id'],
                        'Target': i['指代'],
                        'Argument': i['评论'],
                        'LLMhate': i['是否仇恨'],
                    })
        elif WORKTYPY == 3:
            datatemp = json.load(file)
            data = []
            for i_datatemp in datatemp:
                output = output_tf(i_datatemp['output'])
                Target = []
                # 仅包含仇恨言论######################################################
                if ' hate ' in i_datatemp['output'] and ' non-hate ' not in i_datatemp['output']:
                    for i in output:
                        Target.append(i['Target'])
                # 包含 仇恨 和 非仇恨
                elif ' hate ' in i_datatemp['output'] and ' non-hate ' in i_datatemp['output']:
                    for i in output:
                        if i['Is'] == 'y':
                            Target.append(i['Target'])
                # 仅包含非仇恨言论
                elif ' hate ' not in i_datatemp['output'] and ' non-hate ' in i_datatemp['output']:
                    Target = 0
                else:
                    print('error')
                # 对 Target 去重######################################################
                if Target != 0:
                    Target = list(set(Target))
                # 将 Target 转换为字符串######################################################
                if Target == 0:
                    Target = '(无明确仇恨目标)'
                elif len(Target) == 0:
                    print('error')
                elif len(Target) == 1:
                    Target = Target[0]
                else:
                    Target = ','.join(Target)
                    Target = '(' + Target + ')'
                data.append({
                    'content': i_datatemp['content'],
                    'id': i_datatemp['id'],
                    'Target': Target,
                })
        elif WORKTYPY == 4:
            datatemp = json.load(file)
            data = []
            for i_datatemp in datatemp:
                output = output_tf(i_datatemp['output'])
                for i_output in output:
                    if i_output['Is'] == 'y':
                        data.append({
                            'content': i_datatemp['content'],
                            'id': i_datatemp['id'],
                            'Target': i_output['Target'],
                            'Argument': i_output['Argument'],
                        })
        else:
            raise ValueError("WORKTYPY must be 1 or 2")
    return data