import os
import re

path = "/Users/wangconghao/Documents/毕业设计/数据结构人工/堆栈/"
dir = path + "text"

def read_json():
    json_file = path + path[41:-1] + ".json"    #我的json文件和文件夹命名一致
    with open(json_file, "r", encoding='utf-8') as jf:
        content_list = jf.readlines()
    all_content = ""
    for content in content_list:
        all_content += content
    return all_content

def read_list():
    list_file = path + "assemble_list.txt"
    with open(list_file) as lf:
        num_list = lf.readlines()
    return num_list

def extract_text(json_cont, ass_list):
    txt_list = []
    assembleId_pattern = 'assembleId":(\d+)'
    text_pattern = '"assembleText":"(.+?)"'
    # count = 0
    while 1:
        assID_res = re.search(assembleId_pattern, str(json_cont))
        if assID_res is None:
            break
        json_cont = json_cont[assID_res.span()[1]:]
        txt_res = re.search(text_pattern, json_cont)
        # count += 1
        # print(assID_res.group(1))
        if ass_list.count(assID_res.group(1) + "\n") > 0:
            # print(assID_res.group(1))
            ele = assID_res.group(1) + ":" + txt_res.group(1)
            # print(ele)
            txt_list.append(ele)
    return txt_list

def write_txt(txt_list):
    for ele in txt_list:
        name = ele[:7] + ".txt"
        # print(name)
        content = ele[8:]
        # print(content)
        with open(os.path.join(dir, name), "w+") as f:
            f.write(content)

def main():
    json_content = read_json()
    assemble_list = read_list()
    # print(json_content)
    # print(assemble_list)
    # print("yes?")
    text_list = extract_text(json_content, assemble_list)
    # print(text_list)
    if os.path.exists(dir) is False:
        os.makedirs(dir)
    write_txt(text_list)

if __name__ == "__main__":
    main()
