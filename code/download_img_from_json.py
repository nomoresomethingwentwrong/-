import re
import requests
import os

path = "/Users/wangconghao/Documents/毕业设计/数据结构人工/邻接矩阵/"

def main():
      all_content = read_json()
      assembleId_pattern = 'assembleId":(\d+)'
      image_pattern = 'img(.*)? src="(.+?)"'
      while 1:
            assembleId_result = re.search(assembleId_pattern, all_content)
            all_content = all_content[assembleId_result.span()[1]:]
            count = 0
            while 1:
                  assembleId_nex_result = re.search(assembleId_pattern, all_content)
                  image_result = re.search(image_pattern, all_content)
                  if image_result is None:
                        break
                  if assembleId_nex_result is None or assembleId_nex_result.span()[0] > image_result.span(2)[0]:
                        print(assembleId_result.group(1)+ " : "+ image_result.group(2))
                        download_image(image_result.group(2), assembleId_result.group(1),  count)
                        count += 1
                        all_content = all_content[image_result.span(2)[1]:]
                  else:
                        break
            if image_result is None:
                  break

def read_json():
      json_file = path + "邻接矩阵.json"
      with open(json_file, "r", encoding='utf-8') as f:
#            print("successfully open")
            content_list = f.readlines()
      all_content = ""
      for content in content_list:
            all_content += content
      return all_content

def download_image(url, assembleId, count):
      basedir = path + "image"
      if not os.path.exists(basedir):
            os.mkdir(basedir)
#      image = requests.get(url).content
      try:
            requ = requests.get(url)
#      if requ.status_code is True:
            image = requ.content
            image_name = "%s_%d.png" % (assembleId, count)
            image_dir = os.path.join(basedir, image_name)
            with open(image_dir, "wb") as f:
                f.write(image)
      except:
            print("Connection_error!")

if __name__ == "__main__":
      main()
