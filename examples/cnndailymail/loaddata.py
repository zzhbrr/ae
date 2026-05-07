from datasets import load_dataset


prompt = "Please generate a detailed and comprehensive response to the following request. Cover all possible aspects and provide multiple examples. Break down the topic into subtopics if needed, and avoid omitting any relevant details.\n"

def loaddata(num_sequence, min_len=0, max_len=2000, return_sum_length=False):
    ds = load_dataset("abisee/cnn_dailymail", "1.0.0", trust_remote_code=True, split="test")
    res = []
    lens_list = []
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    i = 0
    while len(res) < num_sequence:
        txt = ds[i]["article"]
        i += 1
        content = prompt + txt
        chat_input = [{"role": "user", "content": content}]
        content_after_template = tokenizer.apply_chat_template(chat_input, tokenize=False)
        inputs_id_len = len(tokenizer.encode(content_after_template))
        if inputs_id_len > max_len or inputs_id_len < min_len:
            continue
        res.append(content)
        lens_list.append(inputs_id_len)
    if return_sum_length:
        return res, sum(lens_list)
    else:
        return res

def loaddata_all(min_len=0, max_len=10000):
    ds = load_dataset("abisee/cnn_dailymail", "1.0.0", trust_remote_code=True, split="test")
    res = []
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    lens_list = []
    
    print("开始读取全部CNN DailyMail数据...")
    total_processed = 0
    
    print(f"数据集大小: {len(ds)}")
    for i in range(len(ds)):
        txt = ds[i]["article"]
        content = prompt + txt
        if i == 0:
            print(content)
        chat_input = [{"role": "user", "content": content}]
        content_after_template = tokenizer.apply_chat_template(chat_input, tokenize=False)
        inputs_id_len = len(tokenizer.encode(content_after_template))
        
        if inputs_id_len > max_len or inputs_id_len < min_len:
            continue
            
        res.append(content)
        lens_list.append(inputs_id_len)
        total_processed += 1
        
        if total_processed % 1000 == 0:
            print(f"已处理 {total_processed} 条数据")
    
    print(f"数据读取完成，共处理 {len(lens_list)} 条有效数据")
    return lens_list

'''
============================================================
总数据量: 11490
[0-1000) 区间: 6619 条数据 (57.61%), 平均长度: 668.97
[1000-2000) 区间: 4293 条数据 (37.36%), 平均长度: 1353.59
[2000+) 区间: 578 条数据 (5.03%), 平均长度: 2275.10
============================================================
最短长度: 125
最长长度: 3549
平均长度: 1005.56
中位数长度: 906
'''