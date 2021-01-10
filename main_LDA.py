# In[]: import 
from gensim import corpora, models
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# In[]: load data
gene_spot = []
for line in open("./data/HCC-3L-HVG_rank_expr.txt"):
    gene_spot.append([i for i in line.split()])
gene_spot = gene_spot[1:]

words_num_ls = []
for j in range(len(gene_spot[0])):
    words = [w[j] for w in gene_spot]
    words_num_ls.append(words)
genes = words_num_ls[0]
words_num_ls = words_num_ls[1:]

words_ls = []
word_max = 0
for i in range(len(words_num_ls)):
    tmp = []
    for j in range(len(words_num_ls[i])):
        if int(words_num_ls[i][j])> word_max:
            word_max = int(words_num_ls[i][j])
        for k in range(int(words_num_ls[i][j])):
            tmp.append(genes[j])
    words_ls.append(tmp)

print(word_max)
print(len(words_ls))
print(len(words_ls[0]))
print(words_ls[0][:10])
mmax = 0
for i in range(len(words_ls)):
    if len(words_ls[i])>mmax:
        mmax = len(words_ls[i])
print(mmax)

# 生成语料词典
dictionary = corpora.Dictionary(words_ls)
# 生成稀疏向量集
corpus = [dictionary.doc2bow(words) for words in words_ls]

# In[]: LDA & Cluster & Plot

num_topics = 40
# LDA模型，num_topics设置聚类数，即最终主题的数量
# lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
lda = models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics)

# # 展示每个主题的前5的基因
# for topic in lda.print_topics(num_words=5):
#     print(topic)

num_clusters = 4
kmean = KMeans(n_clusters=num_clusters)
res_cluster = kmean.fit_predict(lda.inference(corpus)[0])

# 推断每个语料库中的主题类别
# res_cluster = []
# for e, values in enumerate(lda.inference(corpus)[0]):
#     topic_val = 0
#     topic_id = 0
#     for tid, val in enumerate(values):
#         if val > topic_val:
#             topic_val = val
#             topic_id = tid
#     res_cluster.append(topic_id)

# 输出聚类结果
positions = []
for line in open("./data/HCC-3L-Position.txt"):
    positions.append([i for i in line.split()])
positions = positions[1:]
res_x = {}
res_y = {}
for i in range(len(res_cluster)):
    if res_cluster[i] in res_x:
        res_x[res_cluster[i]].append(float(positions[i][1]))
        res_y[res_cluster[i]].append(float(positions[i][2]))
    else:
        res_x[res_cluster[i]] = [float(positions[i][1])]
        res_y[res_cluster[i]] = [float(positions[i][2])]

colors = ['green','red','blue','yellow','cyan']
for topic in range(num_clusters):
    if topic in res_x:
        plt.scatter(res_x[topic], res_y[topic],c=colors[topic],label= topic,s=12, alpha=0.6)
    else:
        print('no such topic: '+str(topic))
plt.legend()
plt.show()
# In[]: Save result
plt.savefig('./result/result_LDA.png', dpi=300)
# %%
