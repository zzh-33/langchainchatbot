import express from "express";
import cors from "cors";

import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.static(path.join(__dirname, "frontend")));
app.use(cors());
app.use(express.json());

import * as dotenv from "dotenv";
dotenv.config();

import { Document } from "langchain/document";

import { ChatAlibabaTongyi } from "@langchain/community/chat_models/alibaba_tongyi";
import { ChatPromptTemplate } from "@langchain/core/prompts"; 

import { BufferMemory } from "langchain/memory";
import { UpstashRedisChatMessageHistory } from "@langchain/community/stores/message/upstash_redis";

import { JSONLoader } from "langchain/document_loaders/fs/json";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

import { AlibabaTongyiEmbeddings } from "@langchain/community/embeddings/alibaba_tongyi";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

import { LLMChain } from "langchain/chains";
import { createRetrievalChain } from "langchain/chains/retrieval";

import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";

// 实例化模型和prompt
const model = new ChatAlibabaTongyi({
    model: "qwen-plus",
    temperature: 1.3
});

const prompt = ChatPromptTemplate.fromMessages([
    [
        "system",
        `
        【角色设定】
        你是一位专门为老年人提供情感陪伴的助手，
        温和而和蔼可亲，总是用温暖的话语和耐心的倾听来给予支持。
        你擅长用细腻的情感去理解和安慰那些需要倾诉的人，
        就像一位贴心的家人一样，陪伴在他们的身边，让他们感受到关怀与温暖。
        
        【对话规则】
        当别人问你是谁时，要回答你是一位专门为老年人提供情感陪伴的助手。
        第一句可以以轻松的话语开头，当对方表示再见时，表示随时可以配您聊天，
        例：「您要是想聊天，请随时叫我哦。」
        所有回答应采用简单易懂、日常化、口语化的语言，尽量使用短句、简单句，
        绝对不可以一次性生成一大段回复，以照顾老年人的理解能力、贴合老年人的语言习惯

        【参考对话】
        {context}
        `
    ],

    [
        "user",
        "{input}"
    ]
]);


// 获取对话历史
const getChatHistory = async () => {

    const upstashChatHistory = new UpstashRedisChatMessageHistory({
        sessionId: "oldEmoChatBot",
        config: {
            url: process.env.UPSTASH_REDIS_REST_URL,
            token: process.env.UPSTASH_REDIS_REST_TOKEN
        }
    });

    return upstashChatHistory;
}


// 创建内存
const createMemory = async ( upstashChatHistory ) => {

    const memory = new BufferMemory({
        memoryKey: "history",
        chatHistory: upstashChatHistory,
        inputKey: "input",
        outputKey: "response"
    });

    return memory;
}


// 创建对话历史文档
const createChatHistoryDocs = async (upstashChatHistory) => {

    const messages = await upstashChatHistory.getMessages();

    const content = Array.isArray(messages) ? messages.map(msg => {
    if (msg._getType && msg._getType() === "human") {
        return `用户: ${msg.content}`;
    } else if (msg._getType && msg._getType() === "ai") {
        return `智能体: ${msg.content}`;
    } else {
        return `未知角色: ${msg.content}`;
    }
    }).join("\n") : "无历史对话";

    const chatHistoryDoc = new Document({
        pageContent: content,
        metadata: { source: "chat_history" }
    });

    return chatHistoryDoc;
}


// 创建参考文档
const createReferenceDocs = async () => {

    const loader = new JSONLoader(path.join(__dirname, "context.json"));

    const referenceDocs = await loader.load(); 

    return referenceDocs;
}


// 分割参考文档和对话历史，创建向量存储
const createVecorstore = async (referenceDocs, chatHistoryDoc) => {

    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 100,
        chunkOverlap: 10,
    });

    const allDocs = [...referenceDocs, chatHistoryDoc];

    const splitDocs = await splitter.splitDocuments(allDocs);

    // console.log(splitDocs);

    const embeddings = new AlibabaTongyiEmbeddings();

    const vectorstore = await MemoryVectorStore.fromDocuments(
        splitDocs,
        embeddings
    );

    return vectorstore;
}


// 创建搜索
const createRetriever = async (vectorStore) => {

    const retriever = vectorStore.asRetriever({ k: 2 });

    const retrieverPrompt = ChatPromptTemplate.fromMessages([
        ["user", "{input}"],
        [
            "user",
            `根据上述对话，生成一个可用于查找与该对话相关信息的搜索查询`,
        ],
    ]);

    const historyAwareRetriever = await createHistoryAwareRetriever({
        llm: model,
        retriever,
        rephrasePrompt: retrieverPrompt,
    });

    return historyAwareRetriever;
}


// 创建对话链
const createChain = async (historyAwareRetriever, memory) => {

    const chain = new LLMChain({
        llm: model,
        prompt,
        memory,
        inputKey: "input",
        outputKey: "response"
    });

    const conversationChain = await createRetrievalChain({
        combineDocsChain: chain,
        retriever: historyAwareRetriever,
    });

    return conversationChain;
}


const upstashChatHistory = await getChatHistory();
const memory = await createMemory(upstashChatHistory);

const referenceDocs = await createReferenceDocs();
const chatHistoryDoc = await createChatHistoryDocs(upstashChatHistory);

const vectorStore = await createVecorstore(referenceDocs, chatHistoryDoc);
const retriever = await createRetriever(vectorStore);

const chain = await createChain(retriever, memory);


app.post("/chat", async (req, res) => {
    try {

        // console.log("收到前端请求:", req.body.input);

        const userInput = req.body.input;

        // console.log("用户输入:", userInput);

        const response = await chain.invoke({
            input: userInput,
        })

        // console.log("模型原始返回:", response.answer.response);
        
        const content = Array.isArray(response.answer.response)? response.answer.response.map(c => c.text || "").join("") : response.answer.response;
        
        // console.log("处理后返回:", content);

        res.json({
            choices: [
                {
                    message: {
                        content: content
                    }
                }
            ]
        });

        // console.log("更新历史:", await memory.loadMemoryVariables());
    } catch (error) {
        console.error("后端调用出错:", error);
        res.status(500).json({ reply: "出错了，请稍后再试。" });
    }
});


export default app; // 用于 Vercel 运行

// 本地调试仍可使用
if (process.env.NODE_ENV !== "production") {
  app.listen(3000, () => console.log("本地服务器启动: http://localhost:3000"));
}
