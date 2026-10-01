from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaLLM

template = PromptTemplate.from_template(
    "我在玩英雄联盟,我的英雄是{champion}, 玩的模式是{mode},帮我选一下出装，只把6个装备按顺序列,不要解释，只列出来"
)

prompt_text = template.format(champion="女警",mode="海克斯乱斗3.0")
model = OllamaLLM(model="qwen")

history_data = [
    ("human", "我在玩英雄联盟,我的英雄是女警, 玩的模式是海克斯乱斗3.0,帮我选一下出装，只把6个装备按顺序列,不要解释，只列出来"),
    ("ai", """1. 破甲拳套
        2. 冲锋盾牌
        3. 火焰之刃
        4. 狩猎者之刃
        5. 破碎魔药瓶
        6. 风暴之刃"""),
    ("human", "现在是13分钟， 我已经出了无尽和分裂弓，下一个出什么"),
    ("ai", "风暴之刃"),
]
chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a league of legend player"),
        MessagesPlaceholder("history"),
        ("human", "下一个呢")
    ]
)

chain = chat_prompt_template | model

res = chain.invoke({"history": history_data})
print(res)