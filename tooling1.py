from langchain import hub
from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Optional
from langchain.chains.openai_functions import (
    create_openai_fn_chain, create_structured_output_chain)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.tools import StructuredTool

class GetProcessUnits(BaseModel):
    """
    Pydantic arguments schema for get_process_unit
    """
    area: str = Field(description="Area description of the main element location in the diagram")
    tag: str = Field(description="Full element tag")
    description: str = Field(description="Description of the process unit")
    type: str = Field(description="Type of the process unit")
    name: str = Field(description="Name of the Process unit")
    notes: list = Field(description="Notes number and description next to the Process unit")

class GetLines(BaseModel):
    """
    Pydantic arguments schema for the lines
    """
    description: str = Field(description="Detailed line description, line specification is shown above the line e.g 123-xxxx-F23-JBNE, this is line tag")
    size: str = Field(description="Nominal size of the pipe, should be fetched as a first section of the line tag before the first dash")
    material_code: str = Field(description="The service code for the material that flows in the line represented by second section of the line tag between dashes, try to decipher where possible")
    tag: str = Field(description="the equipment tag from which the line originates followed by a unique sequential number, located in the third section of the line tag between dashes, no dashes, whatever is after dash goes to the next field")
    line_spec: str = Field(description="line specification for the pipe, including class and material type, valves etc, the last part of the line tag after the dash")


class GetFlowPaths(BaseModel):
    """
    Pydantic arguments schema for get_lines
    """
    connections: str = Field(description="what path process units path connects. specify each connection")
    tag: str = Field(..., description="Tag description")
    description: str = Field(description="Description of the flow path")
    type: str = Field(description="Type of the flow path")
    direction: str = Field(description="Direction of the flow path")

def get_process_units(area: str, tag: str, description: str, type: str, name: str, notes: list) -> str:
    #, qty:str
    response = GetProcessUnits(area=area, tag=tag, description=description, type=type, name=name, notes=notes)
    #, qty=qty
    
    return response.json()

def get_lines(description: str, size: str, material_code: str, tag: str, line_spec: str) -> str:
    #, qty:str
    response = GetLines(description=description, size=size, material_code=material_code, tag=tag, line_spec=line_spec)
    #, qty=qty
    
    return response.json()

def get_flow_paths(tag: str, description: str, type: str, direction: str, connections: str) -> str:
    # Создаем экземпляр модели с данными
    response = GetFlowPaths(tag=tag, description=description, type=type, direction=direction, connections=connections)
    # Возвращаем сериализованный в JSON объект
    return response.json()

# Define a main function to process text from Streamlit
def process_text_from_streamlit(text_output: str) -> str:
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4-0125-preview",
        response_format={"type": "json_object"}
    )

    # Initialize the tools
    tools = [
        StructuredTool.from_function(
            func=get_process_units,
            args_schema=GetProcessUnits,
            description="Function to get process units",
        ),
        StructuredTool.from_function(
            func=get_lines,
            args_schema=GetLines,
            description="Function to get lines",
        ),
        StructuredTool.from_function(
            func=get_flow_paths,
            args_schema=GetFlowPaths,
            description="Function to get flow paths",
        )
        ]
    llm_with_tools = llm.bind(
        functions=[format_tool_to_openai_function(t) for t in tools]
    )

    system_init_prompt =f"You are an expert in explainining pipeline and instrumentation diagrams in metallurgys"
    user_init_prompt = """ Find all process units, lines, flow paths and describe them in details. 
    Return answer in json. 
    The detailed text description is here:"""+f" {text_output}" 


    prompt = ChatPromptTemplate.from_messages([
        ("system", system_init_prompt),
        ("user", user_init_prompt),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = (
        {"input": lambda x: x["input"],
         "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"])
        }
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    response = agent_executor.invoke({"input": text_output})
    return response.get("output")
