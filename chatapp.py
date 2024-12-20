from datetime import datetime
import os
import streamlit as st
import argparse
import time
import json
from gepetto import gpt, gemini, ollama
from main import get_log_stats, load_config, scan_logfile, issues_list_to_report, resolutions_to_report, output_final_report
import logreader
import pandas as pd
from langchain.agents import AgentExecutor, create_react_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.agents.mrkl import prompt
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
import requests
from typing import List

def create_df_tools(df):
    df_tool = PythonAstREPLTool(locals={"df": df}, description="Use Python to analyze the DataFrame 'df'")
    return [
        Tool(
            name="python_repl_ast",
            func=df_tool._run,
            description="""Use this tool to analyze the pandas DataFrame 'df'. 
            Input should be a valid Python command. Output will be the result of running the command."""
        )
    ]

def setup_agent(df, provider="openai", ollama_url="http://localhost:11434", model_name="mistral"):
    if provider == "openai":
        agent = create_pandas_dataframe_agent(
            ChatOpenAI(model="gpt-4-mini", temperature=0),
            df,
            agent_type="openai-tools",
            verbose=True,
            allow_dangerous_code=True
        )
    else:
        llm = Ollama(
            model=model_name,
            base_url=ollama_url,
            temperature=0
        )
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            agent_type="zero-shot-react-description",
            verbose=True,
            allow_dangerous_code=True
        )

        # llm = Ollama(
        #     model=model_name,
        #     base_url=ollama_url,
        #     temperature=0
        # )
        
        # tools = create_df_tools(df)
        
        # prompt_template = """You are working with a pandas DataFrame 'df'.
        # Answer questions about the data using Python code.
        
        # Question: {input}
        
        # Let's approach this step by step:
        # 1) First, understand what is being asked
        # 2) Then, write Python code to analyze the DataFrame
        # 3) Finally, interpret the results
        
        # {agent_scratchpad}"""
        
        # prompt = PromptTemplate(
        #     input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
        #     template=prompt_template
        # )

        # agent = create_react_agent(llm, tools, prompt)
        
        # agent = AgentExecutor.from_agent_and_tools(
        #     agent=agent,
        #     tools=tools,
        #     verbose=True
        # )
    if 'df' not in st.session_state:
        st.session_state.df = df
    if 'agent' not in st.session_state:
        st.session_state.agent = agent
    if 'provider' not in st.session_state:
        st.session_state.provider = provider
    return agent

def get_ollama_models(ollama_url: str) -> List[str]:
    """Query available models from Ollama server"""
    try:
        response = requests.get(f"{ollama_url}/api/tags")
        if response.status_code == 200:
            models = [model['name'] for model in response.json()['models']]
            return models
        return ["mistral"]  # fallback default
    except:
        return ["mistral"]  # fallback if server unreachable

def get_syslog(file):
    # Pull syslog from the Linux system
    try:
        syslog_contents = os.popen('sudo dmesg').read()
        if not syslog_contents:
            return "Failed to retrieve system logs"
    except Exception as e:
        return f"An error occurred while retrieving system logs: {e}"

    # Save the syslog contents to the specified file
    with open(file, 'w') as f:
        f.write(syslog_contents)

    with open(file, 'r') as f:
        log_contents = f.read()
    return log_contents

def output_final_report_for_chat_app(cost, log_length, number_of_issues, model, total_time):
    today_string = datetime.now().strftime("%Y-%m-%d")
    seconds = round(total_time % 60)
    minutes = round((total_time // 60) % 60)
    final_report = f"## Log Report @ {today_string} {number_of_issues} (issues)\n\n"
    final_report += f"- Cost: US${cost:.3f} for {log_length} processed lines using {model} in {minutes:02d}m {seconds:02d}s_\n\n"
    return final_report

def generate_report(issue_model, suggestion_model, resolutions, dry_count,
                     remove_duplicates, config_file, show_log, overrides):
    start_time = time.time()
    # file = "/tmp/syslog.log"
    file = "syslog.log"
    # Read log contents
    log_contents = get_syslog(file)
    with open(file, 'r') as f:
        log_contents = f.read()
        
    config = load_config(config_file, overrides)

    log_contents = logreader.read_logfile(file, config.ignore_list, config.match_list, config.replacement_map, config.regex_ignore_list)
    if len(log_contents) == 0:
        return "No log entries found"

    if remove_duplicates:
        log_contents = logreader.filter_duplicate_logs(log_contents, max_occurrences=3, normalise_map=config.normalise_map)

    # if show_log:
    #     st.text_area("Log Contents", log_contents, height=300)

    #if dry_count:
    #         # st.warning("Dry count is enabled. No suggestions will be generated.")
    #     log_length, token_length = get_log_stats(log_contents, issue_model)
    #     st.text_area(f"Length: {log_length} lines")
    #     st.text_area(f"Tokens: {token_length} tokens")

    # Scan logfile
    issues, cost = scan_logfile(log_contents, config.log_scan_prompt, config.log_merge_prompt, line_chunk_size=500, model=issue_model)
    issues_list = [{"id": key, **value} for key, value in issues.items()]
    with open('issues.json', 'w') as json_file:
        json.dump(issues_list, json_file, indent=4)

    # Store issues in a pandas DataFrame
    df = pd.DataFrame(issues_list)
    if df.empty:
            return("No issues found. Fetch & Generate the report after some time")

    setup_agent(df)
    # report = agent.run("Summarize the issues with all the details in a tabular format")
   
    end_time = time.time()
    total_time = end_time - start_time
    number_of_issues = len(issues_list)
    final_report = output_final_report_for_chat_app(cost, len(log_contents),number_of_issues,issue_model, total_time)
    return final_report



def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

def main():
    st.title("Syslog Analyser Chat bot")
    
    # Settings in left sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Provider selection
        provider = st.selectbox(
            "Select Provider",
            ["openai", "ollama"],
            key="provider_select"
        )
        
        if provider == "openai":
            # Check if API key exists in environment variables
            if 'OPENAI_API_KEY' in st.secrets:
                st.success('API key already provided!', icon='✅')
                openai_api = st.secrets['OPENAI_API_KEY']
            else:
                openai_api = st.text_input('Enter OpenAI API token:', type='password')
            os.environ['OPENAI_API_KEY'] = openai_api
            api_valid = openai_api and openai_api.startswith('sk-')
        else:
            # Ollama server settings
            ollama_url = st.text_input(
                'Ollama Server URL:',
                value='http://localhost:11434',
                help='Enter the URL of your Ollama server'
            )
            
            # Add model selection for Ollama
            available_models = get_ollama_models(ollama_url)
            selected_model = st.selectbox(
                'Select Ollama Model',
                options=available_models,
                help='Choose a model available on your Ollama server'
            )
            api_valid = True
            
        resolutions = st.checkbox("Resolutions", value=False)
        dry_count = st.checkbox("Dry Count", value=False)
        remove_duplicates = st.checkbox("Remove Duplicates", value=True)
        
        generate_button = st.sidebar.button('Fetch & analyse syslog')
        st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    if provider == "openai" and not api_valid:
        st.warning('Please provide a valid OpenAI API key (should start with sk-)')
        return
        
    issue_model = gpt.Model.GPT_4_OMNI_MINI.value[0] if provider == "openai" else "mistral"
    suggestion_model = issue_model
    config_file = "prompts"
    show_log = False
    overrides = "local_overrides.py"

    if 'df' in st.session_state:
        st.dataframe(st.session_state.df, use_container_width=True)
    
    if 'agent' in st.session_state and st.session_state.provider == provider:
        agent = st.session_state.agent
    elif os.path.exists("issues.json"):
        date_list = json.load(open("issues.json"))
        df = pd.DataFrame(date_list)
        if df.empty:
            st.warning("No old issues found. Fetch & Generate the report")
            return
        else:
            agent = setup_agent(df, provider, 
                   ollama_url if provider == "ollama" else None,
                   selected_model if provider == "ollama" else None)
            st.dataframe(st.session_state.df, use_container_width=True)
    else:
        if not generate_button:
            st.warning("No old issues found. Fetch & Generate the report")
            return

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    
    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input(disabled=not api_valid):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if generate_button or st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            if generate_button:
                with st.spinner("Analysing... it will take around a minute"):
                    response = generate_report(issue_model, suggestion_model, resolutions, dry_count,
                     remove_duplicates, config_file, show_log, overrides)
            else:
                with st.spinner("Thinking.."):
                    response = agent.run(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
        if generate_button and 'df' in st.session_state:
            # Display the DataFrame using st.dataframe
            st.dataframe(st.session_state.df, use_container_width=True)

if __name__ == "__main__":
    main()