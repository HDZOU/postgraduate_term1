import jieba
import re

def stopwordList():
	with open("stopword.txt","r",encoding = "utf-8") as f:
		stopwordlist = f.readlines()
		for i in range(0,len(stopwordlist)):
			stopwordlist[i] = stopwordlist[i].strip()
		print("停用词载入完成")
		return tuple(stopwordlist)
def seg_sentence(rfilename,wfilename,userdict):
	stopwordlist = stopwordList()
	result_set = []
	with open(rfilename,'r',encoding = 'utf-8') as f:
		file = f.readlines()
	print("文本读取完成")
	with open(userdict,'r',encoding = 'utf-8') as f:
		user_dict = f.readlines()
	print("use-dict读取完成")
	docs_count = 0
	for line in file:
		# 每行是一个专利文本，逐行处理
		doc = []
		title_list = []
		line = line.strip()
		title = line.split()[0]
		# line = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[:：;+——()?【】“”！，。？、~@#￥%……&*（）]+", "",line)
		# 去各种半角全角英文字符和数字
		line = re.sub("[a-z|ａ-ｚ|Ａ-Ｚ|A-Z|0-9|　]+", "",line)
		title = re.sub("[a-z|ａ-ｚ|Ａ-Ｚ|A-Z|0-9|　]+", "",title)
		# 这儿添加自定义的词语
		for word in user_dict:
			jieba.add_word(word.strip())
		# 开始分词
		content_words = jieba.cut(line)
		title_words = jieba.cut(title)
		# jieba分词
		for word in content_words:
			if word not in stopwordlist:
				doc.append(word)
		for word in title_words:
			if word not in stopwordlist:
				title_list.append(word)
		temp = []
		count = 0
		while len(temp)<len(doc):
			# 将标题加入文本训练,直到标题与文本在一个数量集上
			for word in title_list:
				temp.append(word)
			count += 1
			if count>100:
				# 添加标题，超过100次，还没有到达内容长度的一半，也停止加入
				# 标题中全是无意义的虚词，或者有意义但是是英文缩写，意义被剔除了。
				# 有可能标题去停词后没有词了，考虑了这种情况，防止其无限循环。
				break
		for word in temp:
			doc.append(word)
		result_set.append(' '.join(doc))
		docs_count += 1
	print("分词与去停用词完成，准备写入文件")
	with open(wfilename,'w',encoding = 'utf-8') as f:
		for line in result_set:
			f.write(line + '\n')
	print("写入文件完成")
if __name__ == "__main__":
	seg_sentence('patent_content.txt','config_files/trainfile.txt','userdict.txt')