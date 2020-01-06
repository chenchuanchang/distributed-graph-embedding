import hdfs
G = {}
negative = {}
biao = {}
flex = set()

f = open('datasets/email.txt', 'r')
for line in f.readlines():
    tem = line[:-1].split(' ')
    if len(tem)<2:
        break
    x = int(tem[0])
    y = int(tem[1])
    flex.add(x)
    flex.add(y)
cnt = 0
for i in flex:
    biao[str(i)] = str(cnt)
    cnt += 1

f = open('datasets/email.txt', 'r')
for line in f.readlines():
    tem = line[:-1].split(' ')
    if len(tem)<2:
        break
    x = biao[tem[0]]
    y = biao[tem[1]]
    if x not in G.keys():
        G[x] = {
            "edge": [y],
            "label": "-1"
        }
    elif y not in G[x]:
        G[x]["edge"].append(y)
    if y not in G.keys():
        G[y] = {
            "edge": [x],
            "label": "-1"
        }
    elif x not in G[y]:
        G[y]["edge"].append(x)
    if x not in negative.keys():
        negative[x] = 0
    negative[x] += 1
    if y not in negative.keys():
        negative[y] = 0
    negative[y] += 1

f = open('datasets/email-labels.txt', 'r')
for line in f.readlines():
    tem = line[:-1].split(' ')
    if len(tem)<2:
        break
    x = biao[tem[0]]
    G[x]["label"] = tem[1]

node_num = len(G.keys())
neg = [0 for i in range(node_num)]
for i, node  in enumerate(G.keys()):
    neg[i] = negative[node]**0.75
s = sum(neg)
for i in range(1,node_num):
    neg[i] = neg[i-1]+neg[i]
neg = [neg[i]/s for i in range(node_num)]
G['negative'] = neg
f.close()

f = open('graph.txt', 'w')
f.write(str(G))
f.close()

client = hdfs.Client("http://localhost:50070", timeout=100, session=False)

client.upload("/", "graph.txt", overwrite=True)
# with client.read('/graph.txt') as reader:
#     G = eval(reader.read())
# a = list(G.keys())[:-1]
# print(a)

