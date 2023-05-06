## TRTC文档AI机器人



### 需求：

demo需要你拥有OPENAI 的 API_KEY,  以及PINECON向量数据库

```
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
```


### embedding TRTC文档 


```
python3 process.py
```



### 开始问答 

```
python3 query.py
```