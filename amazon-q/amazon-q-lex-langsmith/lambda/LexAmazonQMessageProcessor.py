from langchain_community.chat_models import BedrockChat
from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
from langchain.schema import HumanMessage, AIMessage
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import boto3
import os
import json
from tvm_client import TVMClient
from amazon_q import AmazonQ

model_id = "anthropic.claude-3-haiku-20240307-v1:0"

def lambda_handler(event, context):

    ##----------------------------------------------------------------------------------------
    ##ToDo [DO NOT DELETE]: Test wether Lex is not properly passing back session attributes,
    ## or if it is just an issue with the built-in test window in the console.
    # incoming_session_attributes = event.get('sessionState', {}).get('sessionAttributes', {})
    # conversation_id = incoming_session_attributes.get('conversationId')
    # parent_message_id = incoming_session_attributes.get('messageId')
    # print(conversation_id)
    ##----------------------------------------------------------------------------------------
    set_langchain_api_key()
    token_client = TVMClient(
        issuer=os.environ["TVM_ISSUER_URL"],
        client_id=os.environ["TVM_CLIENT_ID"],
        client_secret=os.environ["TVM_CLIENT_SECRET"],
        role_arn=os.environ["TVM_ROLE_ARN"],
        region=os.environ['AWS_REGION']
    )
    credentials_q = token_client.get_sigv4_credentials(email=os.environ["TVM_EMAIL"])
    q_client = boto3.client(
        'qbusiness',  # Using 'q' instead of 'qbusiness'
        region_name=os.environ['AWS_REGION'],
        **credentials_q
    )
    ##ToDo: Test initializing with different chat modes, as well as conversationId and parentMessageId
    llm = AmazonQ(
        region_name=os.environ['AWS_REGION'],  # specify your region
        streaming=False,
        client=q_client,
        application_id=os.environ['Q_APPLICATION_ID']
    )


    session_id = event['sessionId']
    history = DynamoDBChatMessageHistory(
        table_name=os.environ["CONVERSATION_TABLE_NAME"],
        session_id=session_id,
    )
    user_message = event["inputTranscript"]
    qa_system_prompt = """Answer using the context below. Keep to 3 sentences max. Say "I don't know" if unsure."""
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )


    def contextualized_question(input: dict):
        if input.get("chat_history"):
            return contextualize_q_chain
        else:
            return input["question"]


    rag_chain = (
        RunnablePassthrough()
        | qa_prompt
        | llm
    )
    #This is the string representing the AI generated answer the LLM returns
    response = rag_chain.invoke({"question": user_message, "chat_history": history.messages})

    #This is a dictionary representing the entire return object from the AmazonQ ChatSync API
    #This dictionary contains the LLM response above unders the "systemMessage" property.
    #Other properties such as conversationId and messageId are also availalbe in this dictionary for other uses.
    response_dictionary = llm.get_last_response()

    # # Add user message and AI message
    history.add_user_message(user_message)
    history.add_ai_message(response)
   
    session_attributes = event["sessionState"].get("sessionAttributes", {})
    session_attributes["conversationId"] = response_dictionary["conversationId"]  # Replace with actual value
    session_attributes["messageId"] = response_dictionary["systemMessageId"]  # Replace with actual value
    print("Returned Session Attributes to Amazon Lex:")
    print(session_attributes)

    return lex_response(event, response, session_attributes)


def lex_response(event, message, session_attributes):
    return {
        "sessionState":{
            "sessionAttributes":session_attributes,
            "dialogAction":{
                "type": "ElicitIntent",
            },
            'intent': {'name': event['sessionState']['intent']['name'], 'state': 'Fulfilled'}
        },
        'messages': [
            {
                'contentType': 'PlainText',
                'content': message
            }
        ]
    }

def set_langchain_api_key():
    ssm = boto3.client('ssm')
    response = ssm.get_parameter(Name=os.environ["LANGCHAIN_API_KEY_PARAMETER_NAME"])
    os.environ["LANGCHAIN_API_KEY"] = response['Parameter']['Value']


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
