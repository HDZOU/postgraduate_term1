#-*- coding:utf-8 -*-  
import logging  
import logging.config  
import configparser  
import numpy as np  
import random  
import os  
  
from collections import OrderedDict  
# 获取当前路径  
path = os.getcwd()  
# 导入日志配置文件  
# logging.config.fileConfig("logging.conf")  
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='my.log', level=logging.INFO, format=LOG_FORMAT)  
# 创建日志对象  
logger = logging.getLogger()  
# loggerInfo = logging.getLogger("TimeInfoLogger")  
# Consolelogger = logging.getLogger("ConsoleLogger")  
# 导入配置文件  
conf = configparser.ConfigParser()  
conf.read("settings.conf")   
# 获取文件路径  
trainfile = os.path.join(path,os.path.normpath(conf.get("filepath", "trainfile")))  
wordidmapfile = os.path.join(path,os.path.normpath(conf.get("filepath","wordidmapfile")))  
authoridmapfile = os.path.join(path,os.path.normpath(conf.get("filepath","authoridmapfile")))  
companyidmapfile = os.path.join(path,os.path.normpath(conf.get("filepath","companyidmapfile")))  
thetafile = os.path.join(path,os.path.normpath(conf.get("filepath","thetafile")))  
theta_dkfile = os.path.join(path,os.path.normpath(conf.get("filepath","theta_dkfile")))  
phifile = os.path.join(path,os.path.normpath(conf.get("filepath","phifile")))  
psifile = os.path.join(path,os.path.normpath(conf.get("filepath","psifile")))  
paramfile = os.path.join(path,os.path.normpath(conf.get("filepath","paramfile")))  
topNfile = os.path.join(path,os.path.normpath(conf.get("filepath","topNfile")))  
tassignfile = os.path.join(path,os.path.normpath(conf.get("filepath","tassignfile")))  
pcafile = os.path.join(path,os.path.normpath(conf.get("filepath","pcafile"))) 
# 获取模型初始参数  
K = int(conf.get("model_args","K"))  
alpha = float(conf.get("model_args","alpha"))  
beta = float(conf.get("model_args","beta"))  
mu = float(conf.get("model_args","mu"))
gamma = float(conf.get("model_args","gamma"))
iter_times = int(conf.get("model_args","iter_times"))  
top_words_num = int(conf.get("model_args","top_words_num"))  
class Document(object):  
    # 每篇专利文本，包括文本中的各个单词和单词个数
    def __init__(self):  
        self.words = []  
        self.length = 0  
class DataPreProcessing(object):  
    # 数据预处理
    def __init__(self):  
        self.docs_count = 0  
        self.words_count = 0
        self.author_count = 0
        self.company_count = 0  
        #保存每个文档d的信息(单词序列，以及length)  
        self.docs = []  
        self.c_a_set = []
        #建立vocabulary表,author表和company表  
        self.word2id = OrderedDict()
        self.author2id = OrderedDict()
        self.company2id = OrderedDict()  
    def cachewordidmap(self):
        # 将词-id对应关系写入文本
        with open(wordidmapfile, 'w', encoding = 'utf-8') as f:  
            for word,id in self.word2id.items():  
                f.write(word +"\t"+str(id)+"\n") 
    def cacheauthoridmap(self):  
        # 将作者-id对应关系写入文本
        with open(authoridmapfile, 'w', encoding = 'utf-8') as f:  
            for author,id in self.author2id.items():  
                f.write(author +"\t"+str(id)+"\n")  
    def cachecompanyidmap(self):  
        # 将公司-id对应关系写入文本
        with open(companyidmapfile, 'w', encoding = 'utf-8') as f:  
            for company,id in self.company2id.items():  
                f.write(company +"\t"+str(id)+"\n")  

class ICTModel(object):  
    def __init__(self,dpre):  
        self.dpre = dpre #获取预处理参数  
        # 构建模型参数  
        # 聚类个数K，迭代次数iter_times,每个类特征词个数top_words_num,超参数α（alpha） β(beta) γ(gamma) μ(mu)  
        self.K = K  
        self.beta = beta  
        self.alpha = alpha  
        self.mu = mu
        self.gamma = gamma
        self.iter_times = iter_times  
        self.top_words_num = top_words_num   

        # 获取文件变量  
        # 专利文本分好词的文件trainfile  

        # 词对应id文件wordidmapfile
        # 作者对应id文件authormapfile
        # 公司对应id文件companymapfile

        # 文本-主题分布文件theta_dkfile
        # 作者-主题分布文件thetafile  
        # 词-主题分布文件phifile  
        # 公司-主体分布文件psifile
        # 每个主题topN词文件topNfile  
        # 最后分派结果文件tassignfile  
        # 模型训练选择的参数文件paramfile  

        self.wordidmapfile = wordidmapfile  
        self.authoridmapfile = authoridmapfile  
        self.companyidmapfile = companyidmapfile  
        self.trainfile = trainfile  
        self.thetafile = thetafile  
        self.theta_dkfile = theta_dkfile  
        self.phifile = phifile  
        self.psifile = psifile
        self.topNfile = topNfile  
        self.tassignfile = tassignfile  
        self.paramfile = paramfile  
        self.pcafile = pcafile

        # 下面以 doc 表示专利文本，word表示单词，topic表示主题

        # p,概率向量 ，存储采样的临时变量  
        self.p = np.zeros(self.K)  
        # nw,词word在主题topic上的分布  
        self.nw = np.zeros((self.dpre.words_count,self.K),dtype="int")  
        # nwsum,每各topic的词的总数  
        self.nwsum = np.zeros(self.K,dtype = "int")  
        # nd,每个doc中各个topic的词的总数  
        self.nd = np.zeros((self.dpre.docs_count,self.K),dtype = "int")  
        # ndsum,每各doc中词的总数  
        self.ndsum = np.zeros(self.dpre.docs_count,dtype = "int")  
        # at,作者在各主题topic上的分布
        self.at = np.zeros((self.dpre.author_count,self.K),dtype = "int")
        # atsum,作者所决定的单词总数
        self.atsum = np.zeros(self.dpre.author_count,dtype = "int")
        # ct,公司在主题topic上的分布
        self.ct = np.zeros((self.dpre.company_count,self.K),dtype = "int")
        # ctsum,公司名下单词的总数
        self.ctsum = np.zeros(self.dpre.company_count,dtype = "int")
        # Z，值为每个doc的每个word的主题，
        self.Z = np.array([ [0 for y in range(self.dpre.docs[x].length)] for x in range(self.dpre.docs_count)])
        # 为Z先随机分配类型，即为每个文档中的各个单词随机分配主题                                    ☝
        self.patent_author_queue =  [[] for i in range(self.dpre.docs_count)]
        for x in range(len(self.Z)):
        # len(self.Z)也就等于 self.dpre.docs_count文本数，因为行数就是dpre.docs_count       看上面 ☝  
            self.ndsum[x] = self.dpre.docs[x].length
            for y in range(self.dpre.docs[x].length):  
                topic = random.randint(0,self.K-1)#随机取一个主题  
                self.Z[x][y] = topic #第x篇文档中第y个词的主题为topic  
                self.nw[self.dpre.docs[x].words[y]][topic] += 1  
                self.nd[x][topic] += 1  
                self.nwsum[topic] += 1

                author = self.dpre.c_a_set[x][1][random.randint(0,self.dpre.c_a_set[x][2] - 1)]
                # 从这个doc的作者集中，随机选一个作者，表示该词由这个作者生成
                self.patent_author_queue[x].append(author)
                self.at[author][topic] += 1
                self.atsum[author] += 1

                company = self.dpre.c_a_set[x][0]
                self.ct[company][topic] += 1
                self.ctsum[company] += 1
        # 初始化各个分布
        self.theta = np.array([ [0.0 for y in range(self.K)] for x in range(self.dpre.author_count) ])  
        self.phi = np.array([ [ 0.0 for y in range(self.dpre.words_count) ] for x in range(self.K)]) 
        self.psi = np.array([[0.0 for y in range(self.K)] for x in range(self.dpre.company_count)])  
        self.theta_dk = np.array([ [0.0 for y in range(self.K)] for x in range(self.dpre.docs_count) ])  

    def sampling(self,i,j):  
        # Gibbs sampling
        # 为每篇专利的每个单词根据概率采样主题，输入 i:文档编号 ，j:单词在文本中的序号（注意不是编号id）  
        # 输出采样的主题topic
        topic = self.Z[i][j]  
        # topic 为单词分配的主题  
        word = self.dpre.docs[i].words[j]
        # word 为单词的编号id
        author = self.patent_author_queue[i][j]
        #获取之前为该单词选取的作者
        company = self.dpre.c_a_set[i][0]
        #获取单词所属文本对应的公司
        
        # 下面去除该单词对其他各个分布产生的影响。
        self.nw[word][topic] -= 1  
        self.nd[i][topic] -= 1  
        self.nwsum[topic] -= 1  
        self.ndsum[i] -= 1  
        self.at[author][topic] -= 1
        self.atsum[author] -= 1
        self.ct[company][topic] -= 1
        self.ctsum[company] -= 1

        # 计算采样所需概率
        Cmu = self.K * self.mu
        Vbeta = self.dpre.words_count * self.beta
        Kalpha = self.K * self.alpha
        Mgamma = self.K * self.gamma
        # 四个概率湘城
        self.p = (self.nw[word] + self.beta)/(self.nwsum + Vbeta) * \
                (self.nd[i] + self.gamma) / (self.ndsum[i] + Mgamma) * \
                (self.at[author] + self.alpha)/(self.atsum[author] + Kalpha) * \
                (self.ct[company] + self.mu)/(self.ctsum[company] + Cmu) * (1/self.dpre.c_a_set[i][2])

        # 随机更新主题代码
        # for k in range(1,self.K):  
        #     self.p[k] += self.p[k-1]  
        # u = random.uniform(0,self.p[self.K-1])  
        # for topic in range(self.K):  
        #     if self.p[topic]>u:  
        #         break  

        # 按这个更新主题更好理解，这个效果还不错  
        p = np.squeeze(np.asarray(self.p/np.sum(self.p)))  
        # 多项分布采样
        topic = np.argmax(np.random.multinomial(1, p))  

        # 用重新sample 的主题，重新填写该单词对分布的影响
        self.nw[word][topic] += 1  
        self.nwsum[topic] += 1  
        self.nd[i][topic] += 1  
        self.ndsum[i] += 1  
        self.at[author][topic] += 1
        self.atsum[author] += 1
        self.ct[company][topic] += 1
        self.ctsum[company] += 1
        return topic  
    def est(self):
        # 估计函数，直接对预处理的结果进行迭代，无标准输入输出。
        # 开始迭代，依次扫描每个文本的每个词，为其用sampling函数采样一个主题
        logger.info(u"迭代次数为%s 次" % self.iter_times)  
        for x in range(self.iter_times):  
            for i in range(self.dpre.docs_count):  
                for j in range(self.dpre.docs[i].length):  
                    topic = self.sampling(i,j)  
                    self.Z[i][j] = topic  
            logger.info(u"第%s次迭代完成" % x)
        logger.info(u"迭代完成。")  
        logger.debug(u"计算文章-主题分布")  
        self._theta_dk()  
        logger.debug(u"计算作者-主题分布")
        self._theta()
        logger.debug(u"计算单词-主题分布")  
        self._phi()  
        logger.debug(u"计算公司-主题分布")
        self._psi()

        logger.debug(u"保存模型")  
        self.save()  
    def _theta_dk(self):  
        # 填写 文档-词 分布
        for i in range(self.dpre.docs_count):  
            self.theta_dk[i] = (self.nd[i]+self.gamma)/(self.ndsum[i]+self.K * self.gamma)  
    def _theta(self):  
        # 填写 作者-主题 分布
        for i in range(self.dpre.author_count ):#遍历文档集合的所有作者  
            self.theta[i] = (self.at[i]+self.alpha)/(self.atsum[i]+self.K * self.alpha)  
    def _phi(self):  
        # 填写 主题-词 分布
        for i in range(self.K):  
            self.phi[i] = (self.nw.T[i] + self.beta)/(self.nwsum[i]+self.dpre.words_count * self.beta)  
    def _psi(self):  
        # 填写 公司-主题 分布
        for i in range(self.dpre.company_count ):
            self.psi[i] = (self.ct[i]+self.mu)/(self.ctsum[i]+self.K * self.mu)  

    def save(self):  
        # 保存theta_dk文章-主题分布  
        logger.info(u"文章-主题分布已保存到%s" % self.theta_dkfile)  
        with open(self.theta_dkfile,'w',encoding = 'utf-8') as f:  
            for x in range(self.dpre.docs_count):  
                for y in range(self.K):  
                    f.write(str(self.theta_dk[x][y]) + '\t')  
                f.write('\n')  
        # 保存phi词-主题分布  
        logger.info(u"词-主题分布已保存到%s" % self.phifile)  
        with open(self.phifile,'w',encoding = 'utf-8') as f:  
            for x in range(self.K):  
                for y in range(self.dpre.words_count):  
                    f.write(str(self.phi[x][y]) + '\t')  
                f.write('\n')  
        # 保存theta作者-主题分布  
        logger.info(u"作者-主题分布已保存到%s" % self.thetafile)  
        with open(self.thetafile,'w',encoding = 'utf-8') as f:  
            for x in range(self.dpre.author_count):  
                for y in range(self.K):  
                    f.write(str(self.theta[x][y]) + '\t')  
                f.write('\n')  
        # 保存psi文章-主题分布  
        logger.info(u"公司-主题分布已保存到%s" % self.psifile)  
        with open(self.psifile,'w',encoding = 'utf-8') as f:  
            for x in range(self.dpre.company_count):  
                for y in range(self.K):  
                    f.write(str(self.psi[x][y]) + '\t')  
                f.write('\n')  

        # 保存参数设置  
        logger.info(u"参数设置已保存到%s" % self.paramfile)  
        with open(self.paramfile,'w',encoding = 'utf-8') as f:  
            f.write('K=' + str(self.K) + '\n')  
            f.write('alpha=' + str(self.alpha) + '\n')  
            f.write('beta=' + str(self.beta) + '\n')  
            f.write('gamma=' + str(self.gamma) + '\n')  
            f.write('mu=' + str(self.mu) + '\n')  
            f.write(u'迭代次数  iter_times=' + str(self.iter_times) + '\n')  
            f.write(u'每个类的高频词显示个数  top_words_num=' + str(self.top_words_num) + '\n')  

        # 保存每个主题top_words_num的词  
        logger.info(u"主题topN词已保存到%s" % self.topNfile)  
        with open(self.topNfile,'w',encoding = 'utf-8') as f:  
            self.top_words_num = min(self.top_words_num,self.dpre.words_count)  
            for x in range(self.K):  
                f.write(u'第' + str(x) + u'类：' + '\n')  
                twords = []  
                twords = [(n,self.phi[x][n]) for n in range(self.dpre.words_count)]  
                twords.sort(key = lambda i:i[1], reverse= True)  
                for y in range(self.top_words_num):  
                    word = OrderedDict({value:key for key, value in self.dpre.word2id.items()})[twords[y][0]]  
                    f.write('\t'*2+ word +'\t' + str(twords[y][1])+ '\n')  

        # 保存每篇文章的每个词 最终分派的主题的结果  
        logger.info(u"文章-词-主题分派结果已保存到%s" % self.tassignfile)  
        with open(self.tassignfile,'w',encoding = 'utf-8') as f:  
            for x in range(self.dpre.docs_count):  
                for y in range(self.dpre.docs[x].length):  
                    f.write(str(self.dpre.docs[x].words[y])+':'+str(self.Z[x][y])+ '\t')  
                f.write('\n')  
        logger.info(u"模型训练完成。")  

  
def preprocessing():  
    # 数据预处理，输入为 trainfile 和 pcafile ，输出（函数返回）为 dpre 对象（处理结果）。
    logger.info(u'载入数据......')  
    with open(trainfile, 'r',encoding = 'utf-8') as f:  
        docs = f.readlines()  
    logger.debug(u"载入patent文本完成,准备生成字典对象和统计文本数据...")  
    # 大的文档集  
    with open(pcafile,'r',encoding = 'utf-8') as f:
        ca_set = f.readlines()
        # ca_set 的结构请看 pcafile文件，一看便知
    logger.debug(u'载入patent公司作者完成,准备统计公司作者数据')
    dpre = DataPreProcessing()  
    items_idx = 0  
    for line in docs:  
        if line != "":  
            tmp = line.strip().split()  
            # 生成一个文档对象：包含单词序列（w1,w2,w3,,,,,wn）可以重复的  
            doc = Document()  
            for item in tmp:  
                if item in dpre.word2id.keys():# 已有的话，只是当前文档追加  
                    doc.words.append(dpre.word2id[item])  
                else:  # 没有的话，要更新vocabulary中的单词词典及wordidmap  
                    dpre.word2id[item] = items_idx  
                    doc.words.append(items_idx)  
                    items_idx += 1  
            doc.length = len(tmp)  
            dpre.docs.append(doc) 
            # 将Document doc加入文本集 
        else:  
            continue

    comp_idx = 0
    author_idx = 0
    for line in ca_set:
        # 每个line 是一篇专利的公司和作者集
        c_a = []
        authors_id = []
        tmp = line.strip().split()
        # tmp是一个列表 ，第一项是公司，第二项是作者List [a1,a2,...]。
        # 将tmp中的各个公司和作者加入各自的id映射表中，然后将名称转化为他们的编号id，然后存在c_a 列表中。
        authors = tmp[1].split(',')
        if tmp[0] in dpre.company2id.keys():
            c_a.append(dpre.company2id[tmp[0]])
        else:
            dpre.company2id[tmp[0]] = comp_idx
            c_a.append(comp_idx)
            comp_idx +=1
        for author in authors:
            #作者转化为作者编号
            if author not in dpre.author2id.keys():#不在作者-id映射表中，加入映射表。
                dpre.author2id[author] = author_idx
                authors_id.append(author_idx)
                author_idx +=1
            else:
                authors_id.append(dpre.author2id[author])
        c_a.append(authors_id)
        c_a.append(len(authors_id))
        dpre.c_a_set.append(c_a)
    # dpre.c_a_set中为(编号id）[[company,[author,author,...],作者数],[company,[author,author...],作者数],...]
    #             举例        [[ c1   , [a1,a2,...] ,author_count] ,[c2   ,[a3,a5,a8,...],author_count],...]
    dpre.docs_count = len(dpre.docs) # 文档数  
    dpre.words_count = len(dpre.word2id) # 词汇数 
    dpre.author_count = len(dpre.author2id) #作者数 
    dpre.company_count = len(dpre.company2id) #公司数
    logger.info(u"共有%s个文档" % dpre.docs_count)  
    logger.info(u"共有%s个词汇" % dpre.words_count)  
    logger.info(u"共有%s个作者" % dpre.author_count)  
    logger.info(u"共有%s个公司" % dpre.company_count)  
    dpre.cachewordidmap()  #写入文件
    dpre.cacheauthoridmap()  #写入文件
    dpre.cachecompanyidmap()  #写入文件
    logger.info(u"词、公司和作者与序号的对应关系已保存")  
    return dpre  
def run():  
    # 主函数入口
    dpre = preprocessing()  
    ict = ICTModel(dpre)  
    ict.est()  
if __name__ == '__main__':  
    run()