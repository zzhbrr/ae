from datasets import load_dataset

prompt = "Please provide a comprehensive solution to the following programming problem from an online judge platform, write the detailed solution and C++ code. If the code has more than one solution, write all soluaiton. The output structure should be: {Solution Id[Solution id], Code[complete C++ implementation], Code Analysis[detailed technical breakdown of the solution]}. The problem is: \n"

def loaddata(num_sequence, min_len=0, max_len=2000, return_sum_length=False):
    res = []
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    ds_list = [load_dataset("codeparrot/apps", split="test", difficulties=["competition"]), load_dataset("codeparrot/apps", split="train", difficulties=["competition"]), load_dataset("codeparrot/apps", split="test", difficulties=["interview"]), load_dataset("codeparrot/apps", split="train", difficulties=["interview"]), load_dataset("codeparrot/apps", split="test", difficulties=["introductory"]), load_dataset("codeparrot/apps", split="train", difficulties=["introductory"])]
    lens_list = []
    for ds in ds_list:
        if len(res) >= num_sequence:
            break
        for i in range(len(ds)):
            
            txt = ds[i]["question"]
            i += 1
            content = prompt + txt
            chat_input = [{"role": "user", "content": content}]
            content_after_template = tokenizer.apply_chat_template(chat_input, tokenize=False)
            inputs_id_len = len(tokenizer.encode(content_after_template))
            if inputs_id_len > max_len or inputs_id_len < min_len:
                continue
            res.append(content)
            lens_list.append(inputs_id_len)
            if len(res) >= num_sequence:
                break
    if len(res) < num_sequence:
        print(f"Warning: only {len(res)} sequences are loaded, but {num_sequence} are required.")
        while len(res) < num_sequence:
            res += res[:num_sequence-len(res)]
            lens_list += lens_list[:num_sequence-len(lens_list)]
    if return_sum_length:
        return res, sum(lens_list)
    else:
        return res

def loaddata_all(min_len=0, max_len=10000):
    res = []
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    ds_list = [load_dataset("codeparrot/apps", split="test", difficulties=["competition"]), 
               load_dataset("codeparrot/apps", split="train", difficulties=["competition"]), 
               load_dataset("codeparrot/apps", split="test", difficulties=["interview"]), 
               load_dataset("codeparrot/apps", split="train", difficulties=["interview"]), 
               load_dataset("codeparrot/apps", split="test", difficulties=["introductory"]), 
               load_dataset("codeparrot/apps", split="train", difficulties=["introductory"])]
    lens_list = []
    
    print("开始读取全部数据...")
    total_processed = 0
    
    for ds_idx, ds in enumerate(ds_list):
        print(f"正在处理数据集 {ds_idx + 1}/{len(ds_list)}, 大小: {len(ds)}")
        for i in range(len(ds)):
            txt = ds[i]["question"]
            content = prompt + txt
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
总数据量: 29997
[0-1000) 区间: 28251 条数据 (94.18%), 平均长度: 524.82
[1000-2000) 区间: 1707 条数据 (5.69%), 平均长度: 1196.84
[2000+) 区间: 39 条数据 (0.13%), 平均长度: 2912.46
============================================================
最短长度: 99
最长长度: 5327
平均长度: 566.17
中位数长度: 533
'''