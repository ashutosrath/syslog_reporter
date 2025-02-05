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
from langchain_ollama import OllamaLLM
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)

from langchain_community.llms import Ollama
import requests
from typing import List
from streamlit_float import *



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
            ChatOpenAI(model=model_name, temperature=0),
            df,
            agent_type="openai-tools",
            verbose=True,
            allow_dangerous_code=True
        )
    else:
        llm = OllamaLLM(
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

def output_final_report_for_chat_app(cost, log_length, token_length,number_of_issues, model, total_time):
    today_string = datetime.now().strftime("%Y-%m-%d %H::%M::%S")
    seconds = round(total_time % 60)
    minutes = round((total_time // 60) % 60)
    final_report = f"### Report Summary\n\n"
    final_report += f"- Report generated on {today_string}\n"
    final_report += f"- Number of issues found: {number_of_issues}\n"
    final_report += f"- Cost: US${cost:.3f}, Time taken: {minutes:02d}m {seconds:02d}s, Model: {model}\n"
    final_report += f"- Log length: {log_length} lines, Token length: {token_length}\n"
    final_report += f"### Issue List in detail: \n\n"
    return final_report

def generate_report(issue_model, suggestion_model, provider, ollama_url, resolutions, dry_count,
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

    end_time = time.time()
    total_time = end_time - start_time
    number_of_issues = len(issues_list)
    log_length, token_length = get_log_stats(log_contents, issue_model)
    final_report = output_final_report_for_chat_app(cost, log_length,token_length,number_of_issues, issue_model, total_time)

    # Store issues and report in a JSON file
    report_data = {
        "report_info": final_report,
        "issues_list": issues_list
    }
    with open('issues.json', 'w') as json_file:
        json.dump(report_data, json_file, indent=4)
    # Store issues in a pandas DataFrame
    df = pd.DataFrame(issues_list)
    if df.empty:
        return "No issues found. Fetch & Generate the report after some time"
    setup_agent(df, provider, 
                ollama_url if provider == "ollama" else None,
                issue_model)
    st.session_state.report_info = final_report
    return



def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

def set_css():
    st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)

def move_focus():
    # inspect the html to determine which control to specify to receive focus (e.g. text or textarea).
    st.components.v1.html(
        f"""
            <script>
                var textarea = window.parent.document.querySelectorAll("textarea[type=textarea]");
                for (var i = 0; i < textarea.length; ++i) {{
                    textarea[i].focus();
                }}
            </script>
        """,
    )



def complete_messages(chat_container,agent_action,user_content,stream=False):
    if 'agent' in st.session_state:
        agent = st.session_state.agent
    else:
        return "No agent found to respond you. Fetch & Generate the report"
    
    if agent_action:
        st_callback = StreamlitCallbackHandler(chat_container)
        response = agent.invoke({"input": user_content}, {"callbacks": [st_callback]})
    else:
        response = agent.invoke({"input": user_content})
    if "output" in response:
        response_content = response["output"]
    else:
        response_content="No output found in the response."
    return response_content

def process_user_content():
    st.session_state.messages.append({"role": "user", "content": st.session_state.user_content})

def render_left_column():
    if 'df' in st.session_state:
        st.metric("Issues", len(st.session_state.df))
    else:
        st.metric("Issues", 0)

def render_center_column(generate_button, issue_model, suggestion_model, provider,ollama_url,
                         resolutions, dry_count, remove_duplicates, config_file, show_log, overrides):
    st.write("### Details of issues")
    center_container = st.container()
    center_container.markdown(
        """
        <style>
        .center_container {
            max-height: calc(100vh - 200px);
            overflow-y: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    with center_container:
        if generate_button:
                with st.spinner("Analysing... it will take around a minute"):
                    generate_report(issue_model, suggestion_model, provider, ollama_url, resolutions, dry_count,
                        remove_duplicates, config_file, show_log, overrides)
        center_container.empty()
        if 'report_info' in st.session_state and 'df' in st.session_state:
            st.markdown(st.session_state.report_info)
            st.dataframe(st.session_state.df, use_container_width=True)
        else:
            st.warning("No issues found. Fetch & Generate the report")

def render_right_column(api_valid, agent_action):
    st.write("### Chat")

    chat_container = st.container(height=550,border=True)
    chat_container.markdown(
        """
        <style>
        .chat-container {
            max-height: 550px;
            overflow-y: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    with chat_container:
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
        
        for i,message in enumerate(st.session_state.messages):
            chat_container.chat_message(message["role"]).write(message["content"])

        with st.container():
            st.chat_input(key='user_content', on_submit=process_user_content,disabled=not api_valid) 
            button_b_pos = "1rem"
            button_css = float_css_helper(width="3rem", bottom=button_b_pos, transition=0)
            float_parent(css=button_css)

        if user_content := st.session_state.user_content:
            with st.spinner("Thinking.."):
                assistant_content = complete_messages(chat_container,agent_action,user_content)
                chat_container.chat_message("assistant").write(assistant_content)
                st.session_state.messages.append({"role": "assistant", "content": assistant_content})


def main():
    ollama_url = "http://localhost:11434"
    st.set_page_config(layout="wide")
    float_init(theme=True, include_unstable_primary=False)
    # set_css()
    st.markdown("<h1 style='text-align: center;'>Syslog Analyser Chat bot</h1>", unsafe_allow_html=True)
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
            selected_model = gpt.Model.GPT_4_OMNI_MINI.value[0]
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
        agent_action = st.checkbox("Display Agentic Actions", value=False)
        
        generate_button = st.sidebar.button('Fetch & analyse syslog')
        st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    if provider == "openai" and not api_valid:
        st.warning('Please provide a valid OpenAI API key (should start with sk-)')
        return
    print(selected_model)
    issue_model = selected_model
    suggestion_model = issue_model
    config_file = "prompts"
    show_log = False
    overrides = "local_overrides.py"
    if os.path.exists("issues.json"):
        last_stored_issues = json.load(open("issues.json"))
        df = pd.DataFrame(last_stored_issues.get("issues_list", []))
        if df.empty:
            st.warning("No old issues found. Fetch & Generate the report")
            return
        else:
            print(f"Setting up agent with old issues: {df.shape[0]}")
            setup_agent(df, provider, 
                ollama_url if provider == "ollama" else None,
                selected_model)
            st.session_state.report_info = last_stored_issues.get("report_info", "")
    else:
        if not generate_button:
            st.warning("No old issues found. Fetch & Generate the report")
            return

    left,center,right = st.columns((1.5, 6, 3), gap='small', border=True)
    with left:
        render_left_column()
    with center:
        render_center_column(generate_button, issue_model, suggestion_model, provider,ollama_url,
                             resolutions, dry_count, remove_duplicates, config_file, show_log, overrides)
    with right:
        render_right_column(api_valid, agent_action)
        move_focus()
        
if __name__ == "__main__":
    main()