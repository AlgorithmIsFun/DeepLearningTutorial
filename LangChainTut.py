import bs4
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv, find_dotenv
from typing import TypedDict
from langchain.tools import tool
from langchain.agents import create_agent, AgentState
from pydantic import BaseModel
from langchain_google_community import GmailToolkit
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.agents.structured_output import ToolStrategy
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents.middleware import wrap_tool_call, dynamic_prompt, ModelRequest, HumanInTheLoopMiddleware
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.7)
messages = [
    ("system", "You are a helpful assistant."),
    ("human", "Explain LLMs in one sentence.")
]
template = """
You are an expert data scientist with an expertise in building deep learning models.
Explain the concept of {concept} in a couple of lines
"""

class Context(TypedDict):
    user_role: str

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."

    return base_prompt

def agent():
    agent = create_agent(
        model=llm,
        tools=[search],
        middleware=[user_role_prompt],
        context_schema=Context
    )

    # Run the agent
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Explain machine learning"}]}, context={"user_role": "expert"}
    )
    print(response["messages"][1])

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

def agent_output_format():
    agent = create_agent(
        model=llm,
        tools=[search],
        response_format=ToolStrategy(ContactInfo)
    )
    # ContactInfo: name='John Doe', email='john@example.com', phone='(555) 123-4567'
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
    })
    print(result["structured_response"])

class CustomState(AgentState):
    user_preferences: dict

def state_agent():
    agent = create_agent(
        model=llm,
        tools=[search],
        state_schema=CustomState
    )
    # The agent can now track additional state beyond messages
    result = agent.invoke({
        "messages": [{"role": "user", "content": "I prefer technical explanations"}],
        "user_preferences": {"style": "technical", "verbosity": "detailed"},
    })
    print(result)

def agent_stream():
    agent = create_agent(
        model=llm,
        tools=[search],
    )
    for chunk in agent.stream({
        "messages": [{"role": "user", "content": "Search for AI news and summarize the findings"}]
    }, stream_mode="values"):
        # Each chunk contains the full state at that point
        latest_message = chunk["messages"][-1]
        if latest_message.content:
            print(f"Agent: {latest_message.content}")
        elif latest_message.tool_calls:
            print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")

def send_LLM_message(message):
    response = llm.invoke(message)
    print(response)

def prompt(template):
    prompt = PromptTemplate(input_variables=["concept"], template=template)
    return prompt.format(concept="regularization")

def chain():
    promt = ChatPromptTemplate.from_template("Tell me a about {topic}.")
    output_parser = StrOutputParser()
    chain = promt | llm | output_parser
    response = chain.invoke({"topic": "autoencoder"})
    print(response)

def past_convo_agent():
    #Add past conversations like this
    conversation = [
        {"role": "system", "content": "You are a helpful assistant that translates English to French."},
        {"role": "user", "content": "Translate: I love programming."},
        {"role": "assistant", "content": "J'adore la programmation."},
        {"role": "user", "content": "Translate: I love building applications."}
    ]
    agent = create_agent(
        model=llm
    )
    response = agent.invoke(conversation)

def batch_agent():
    agent = create_agent(
        model=llm
    )
    responses = agent.batch([
        "Why do parrots have colorful feathers?",
        "How do airplanes fly?",
        "What is quantum computing?"
    ])
    for response in responses:
        print(response)

#Send Gmail Email but get approval from user before completing email
def Human_Approval():
    toolkit = GmailToolkit()
    agent = create_agent(
        model=llm,
        tools=toolkit.get_tools(),
        checkpointer=InMemorySaver(),
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "gmail_send_message": {
                        "allowAccept": True,
                        "allowEdit": True,
                        "allowReject": True,
                    }
                },
            ),
        ],
    )
    user_input = "Please draft and send an email to user@example.com with the subject 'Quick Check' and body 'Did you finish the task?'"
    config = {"configurable": {"thread_id": "some_id"}}
    result = agent.invoke({"messages": [
                {
                    "role": "user",
                    "content": user_input,
                }
            ]}, config=config)
    print(result)

#RAG Implementation
def load_docs():
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    return loader.load()

def split_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    return text_splitter.split_documents(docs)


def prepare_docs():
    docs = load_docs()
    splitdocs = split_docs(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = InMemoryVectorStore(embeddings)
    document_ids = vector_store.add_documents(documents=splitdocs)
    return vector_store

vector_store = prepare_docs()
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def RAG():
    tools = [retrieve_context]
    # If desired, specify custom instructions
    prompt = (
        "You have access to a tool that retrieves context from a blog post. "
        "Use the tool to help answer user queries."
    )
    agent = create_agent(model=llm, tools=tools, system_prompt=prompt)
    query = (
        "What is the standard method for Task Decomposition?\n\n"
        "Once you get the answer, look up common extensions of that method."
    )

    for event in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
    ):
        event["messages"][-1].pretty_print()