from utils.agent import Agent
from rich.console import Console
from rich.panel import Panel
from rich import box
from rich.text import Text
import re
from model_auditor import ModelAuditor
from model_auditor.metrics import *
from model_auditor.shifts import *
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import Subset
import warnings
import argparse
from pathlib import Path
import torch
from torchvision import datasets, models
from PIL import Image
import os
import datetime
import pandas as pd
import numpy as np

# Ignore all warnings
warnings.filterwarnings("ignore")

# Setup argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Model Auditor - Evaluate robustness of image classification models")
    
    # Model architecture
    parser.add_argument("--model", type=str, default="resnet18", 
                        help="Model architecture (any torchvision model, 'siim-isic' for SIIM-ISIC model, 'deepderm' for DeepDerm model, or 'other' for ONNX)")
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="CIFAR10",
                        help="Dataset name (torchvision, medmnist, or path to folder)")
    
    # Model weights
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to model weights (.pth or .onnx)")
    
    # Optional arguments
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (cpu, cuda, mps). Auto-detected if not specified")
    parser.add_argument("--subset", type=int, default=50,
                        help="Number of samples to use for evaluation")
    parser.add_argument("--no-debate", action="store_false", dest="show_debate",
                        help="Don't show debate details")
    parser.add_argument("--single-agent", action="store_false", dest="multi_agent_decision",
                        help="Use single agent instead of multi-agent debate")
    
    return parser.parse_args()

console = Console()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Determine device
    if args.device:
        device = args.device
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    console.print(f"[yellow]Using device: {device}")
    
    # Initialize conversation log
    conversation_log = []
    conversation_log.append(f"Conversation started at: {datetime.datetime.now()}")
    conversation_log.append(f"Model: {args.model}")
    conversation_log.append(f"Dataset: {args.dataset}")
    conversation_log.append(f"Weights: {args.weights}")
    conversation_log.append(f"Device: {device}")
    conversation_log.append(f"Subset: {args.subset}")
    conversation_log.append(f"Multi-agent debate: {args.multi_agent_decision}")
    conversation_log.append(f"Show debate: {args.show_debate}")
    conversation_log.append("\nConversation history:")
    
    # Get transform for the dataset
    transform = get_default_transform(args.dataset, args.model)
    
    # Load dataset
    try:
        test_set = load_dataset(args.dataset, transform)
        console.print(f"[green]Successfully loaded dataset: {args.dataset}")
    except Exception as e:
        console.print(f"[red]Error loading dataset: {str(e)}")
        return
    
    # Create subset for evaluation
    if args.subset and args.subset < len(test_set):
        subset = Subset(test_set, torch.randperm(len(test_set))[:args.subset])
        console.print(f"[yellow]Using subset of {args.subset} samples")
    else:
        subset = test_set
    
    # Determine number of classes
    num_classes = get_num_classes(args.dataset, args.model)
    if not num_classes and hasattr(test_set, 'classes'):
        num_classes = len(test_set.classes)
    elif not num_classes:
        console.print("[yellow]Could not determine number of classes, assuming 10")
        num_classes = 10
    
    # Add debugging information
    console.print(f"[yellow]Number of classes: {num_classes}")
    
    # Load model
    try:
        model = load_model(args.model, args.weights, num_classes, device)
        console.print(f"[green]Successfully loaded model: {args.model}")
    except Exception as e:
        console.print(f"[red]Error loading model: {str(e)}")
        return
    
    # Set multi-agent debate options
    multi_agent_decision = args.multi_agent_decision
    show_debate = args.show_debate
    
    # Display ASCII art header
    console.print(Text("""
    ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗     
    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║     
    ██╔████╔██║██║   ██║██║  ██║█████╗  ██║     
    ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║     
    ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗
    ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝
                                                
     █████╗ ██╗   ██╗██████╗ ██╗████████╗ ██████╗ ██████╗ 
    ██╔══██╗██║   ██║██╔══██╗██║╚══██╔══╝██╔═══██╗██╔══██╗
    ███████║██║   ██║██║  ██║██║   ██║   ██║   ██║██████╔╝
    ██╔══██║██║   ██║██║  ██║██║   ██║   ██║   ██║██╔══██╗
    ██║  ██║╚██████╔╝██████╔╝██║   ██║   ╚██████╔╝██║  ██║
    ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
    """, style="orange3"))
    
    # Initialize agent and auditor
    agent = Agent(model="anthropic/claude-3-7-sonnet-20250219", system_prompt=open("prompts/metrics_reloaded_prompt.txt").read())
    auditor = ModelAuditor(model, subset)
    
    ### METRIC SELECTION LOOP

    # Ask for task description once before starting the conversation loop
    console.print(Panel("✱ Welcome to the Model Auditor Agent! Enter a description of your image classification task", 
                       border_style="orange3", 
                       box=box.ROUNDED,
                       width=100))
    task_description = console.input("[yellow]> ")
    conversation_log.append(f"User: {task_description}")
    agent.add_event(task_description, display=False)

    selected_metrics = []
    while True:
        response = agent.ask()
        conversation_log.append(f"Assistant: {response}")
        
        # Check if the response contains metrics
        metric_match = re.search(r'<metric>(.*?)</metric>', response, re.DOTALL)
        if metric_match:
            if multi_agent_decision:
                # do a multi agent debate here
                memory_lst = agent.memory_lst.copy()
                debate_agent_1 = Agent(model="anthropic/claude-3-7-sonnet-20250219", system_prompt=open("prompts/metrics_reloaded_prompt.txt").read())
                debate_agent_2 = Agent(model="anthropic/claude-3-7-sonnet-20250219", system_prompt=open("prompts/metrics_reloaded_prompt.txt").read())
                moderator_agent = Agent(model="anthropic/claude-3-7-sonnet-20250219", system_prompt=open("prompts/metrics_reloaded_prompt.txt").read())

                debate_agent_1.memory_lst = memory_lst.copy()
                debate_agent_2.memory_lst = memory_lst.copy()
                moderator_agent.memory_lst = memory_lst.copy()

                prompt = f"""
                You got enough information to select the best metrics for the task.
                Please think step by step and justify your choice. And add the metrics at the end of your response.
                You will then debate those metrics with another assistant.
                """

                moderator_prompt = f"""
                You will be moderating a debate between two agents discussing which metrics are best for the task.
                Let the debate continue as long as new insights are being generated.
                When you feel the debate has reached a conclusion, output your final metric selection in the format:
                <metric>metric1\nmetric2\nmetric3</metric>
                """

                debate_agent_1.add_event(prompt, display=False)
                debate_agent_2.add_event(prompt, display=False)
                moderator_agent.add_event(moderator_prompt, display=False)

                debate_response_1 = debate_agent_1.ask()
                debate_response_2 = debate_agent_2.ask()

                conversation_log.append(f"Debate Agent 1: {debate_response_1}")
                conversation_log.append(f"Debate Agent 2: {debate_response_2}")

                debate_agent_1.add_memory(debate_response_1, display=False)
                debate_agent_2.add_memory(debate_response_2, display=False)

                console.print(Panel("✱ Starting multi-agent debate on metrics selection...", 
                               border_style="cyan", 
                               box=box.ROUNDED,
                               width=100))
                
                if show_debate:
                    console.print(Panel(debate_response_1, title="Agent 1", border_style="blue", box=box.ROUNDED, width=100))
                    console.print(Panel(debate_response_2, title="Agent 2", border_style="green", box=box.ROUNDED, width=100))

                debate_agent_1.add_event(f"Agent 2 said: {debate_response_2}", display=False)
                debate_agent_2.add_event(f"Agent 1 said: {debate_response_1}", display=False)
                
                # Add initial responses to moderator
                moderator_agent.add_event(f"Agent 1 initial proposal:\n{debate_response_1}", display=False)
                moderator_agent.add_event(f"Agent 2 initial proposal:\n{debate_response_2}", display=False)
                
                debate_round = 1
                debate_history = []
                debate_history.append((debate_response_1, debate_response_2))
                
                while True:
                    console.print(Panel(f"✱ Debate Round {debate_round}", border_style="cyan", box=box.ROUNDED, width=100))
                    
                    debate_response_1 = debate_agent_1.ask()
                    conversation_log.append(f"Debate Round {debate_round} - Agent 1: {debate_response_1}")
                    debate_agent_1.add_memory(debate_response_1, display=False)
                    console.print(Panel(debate_response_1, title="Agent 1", border_style="blue", box=box.ROUNDED, width=100))
                    
                    debate_agent_2.add_event(f"Agent 1 said: {debate_response_1}", display=False)
                    debate_response_2 = debate_agent_2.ask()
                    conversation_log.append(f"Debate Round {debate_round} - Agent 2: {debate_response_2}")
                    debate_agent_2.add_memory(debate_response_2, display=False)
                    console.print(Panel(debate_response_2, title="Agent 2", border_style="green", box=box.ROUNDED, width=100))
                    
                    debate_agent_1.add_event(f"Agent 2 said: {debate_response_2}", display=False)
                    
                    # Add this round to history
                    debate_history.append((debate_response_1, debate_response_2))
                    
                    # Update moderator with latest round
                    moderator_agent.add_event(f"Debate Round {debate_round}:\nAgent 1: {debate_response_1}\nAgent 2: {debate_response_2}", display=False)
                    
                    # Ask moderator if debate should continue
                    moderator_response = moderator_agent.ask()
                    conversation_log.append(f"Debate Round {debate_round} - Moderator: {moderator_response}")
                    
                    # Check if moderator has decided on final metrics
                    metric_match = re.search(r'<metric>(.*?)</metric>', moderator_response, re.DOTALL)
                    if metric_match:
                        console.print(Panel("✱ Moderator has decided the debate has reached a conclusion", 
                                   border_style="orange3", 
                                   box=box.ROUNDED,
                                   width=100))
                        
                        metrics = metric_match.group(1).strip().split('\n')
                        console.print(Panel("\n".join([f"✱ {metric.strip()}" for metric in metrics if metric.strip()]), 
                                    title="Selected Metrics (After Debate)",
                                    border_style="violet", 
                                    box=box.ROUNDED,
                                    width=100))
                        
                        # Store selected metrics
                        selected_metrics = [metric.strip() for metric in metrics if metric.strip()]
                        conversation_log.append(f"Selected Metrics (After Debate):\n" + "\n".join([f"- {metric}" for metric in selected_metrics]))
                        break
                    
                    # Display moderator's thoughts
                    console.print(Panel(moderator_response, title="Moderator", border_style="orange3", box=box.ROUNDED, width=100))
                    
                    debate_round += 1
                
                # Add the final decision to the main agent's memory
                agent.add_memory(f"After a debate between multiple agents, the following metrics were selected:\n" + 
                                "\n".join([f"- {metric}" for metric in selected_metrics]), display=False)
                
                break
            else:
                metrics = metric_match.group(1).strip().split('\n')
                console.print(Panel("\n".join([f"✱ {metric.strip()}" for metric in metrics if metric.strip()]), 
                                title="Selected Metrics",
                                border_style="violet", 
                                box=box.ROUNDED,
                                width=100))
                
                # Store selected metrics
                selected_metrics = [metric.strip() for metric in metrics if metric.strip()]
                conversation_log.append(f"Selected Metrics:\n" + "\n".join([f"- {metric}" for metric in selected_metrics]))
                agent.add_memory(response, display=False)
                break
        else:
            agent.add_memory(response)
        user_input = console.input("[yellow]> ")
        conversation_log.append(f"User: {user_input}")
        agent.add_event(user_input, display=False)

    # Add selected metrics to auditor
    for metric in selected_metrics:
        # Extract just the metric name without any descriptions
        metric_name = metric.split()[0].strip()
        try:
            # Dynamically create metric instance and add to auditor
            metric_instance = eval(f"{metric_name}()")
            auditor.add_metric(metric_instance)
        except Exception as e:
            console.print(f"[red]✗ Failed to add metric {metric_name}: {str(e)}")

    ### SHIFT SELECTION LOOP

    agent.add_event(open("prompts/shifts_prompt.txt").read(), display=False)

    selected_shifts = []
    while True:
        response = agent.ask()
        
        # Check if the response contains shifts
        shift_match = re.search(r'<shift>(.*?)</shift>', response, re.DOTALL)
        if shift_match:
            if multi_agent_decision:
                # do a multi agent debate here
                memory_lst = agent.memory_lst.copy()
                debate_agent_1 = Agent(model="anthropic/claude-3-7-sonnet-20250219", system_prompt=open("prompts/shifts_prompt.txt").read())
                debate_agent_2 = Agent(model="anthropic/claude-3-7-sonnet-20250219", system_prompt=open("prompts/shifts_prompt.txt").read())
                moderator_agent = Agent(model="anthropic/claude-3-7-sonnet-20250219", system_prompt=open("prompts/shifts_prompt.txt").read())

                debate_agent_1.memory_lst = memory_lst.copy()
                debate_agent_2.memory_lst = memory_lst.copy()
                moderator_agent.memory_lst = memory_lst.copy()

                prompt = f"""
                You got enough information to select the best shifts for the task.
                Please think step by step and justify your choice. And add the shifts at the end of your response.
                You will then debate those shifts with another assistant.
                """

                moderator_prompt = f"""
                You will be moderating a debate between two agents discussing which shifts are best for the task.
                Let the debate continue as long as new insights are being generated.
                When you feel the debate has reached a conclusion, output your final shift selection in the format:
                <shift>shift1\nshift2\nshift3</shift>
                """

                debate_agent_1.add_event(prompt, display=False)
                debate_agent_2.add_event(prompt, display=False)
                moderator_agent.add_event(moderator_prompt, display=False)

                debate_response_1 = debate_agent_1.ask()
                debate_response_2 = debate_agent_2.ask()

                conversation_log.append(f"Shifts Debate Agent 1: {debate_response_1}")
                conversation_log.append(f"Shifts Debate Agent 2: {debate_response_2}")

                debate_agent_1.add_memory(debate_response_1, display=False)
                debate_agent_2.add_memory(debate_response_2, display=False)

                console.print(Panel("✱ Starting multi-agent debate on shifts selection...", 
                               border_style="cyan", 
                               box=box.ROUNDED,
                               width=100))
                
                if show_debate:
                    console.print(Panel(debate_response_1, title="Agent 1", border_style="blue", box=box.ROUNDED, width=100))
                    console.print(Panel(debate_response_2, title="Agent 2", border_style="green", box=box.ROUNDED, width=100))

                debate_agent_1.add_event(f"Agent 2 said, please discuss: {debate_response_2}", display=False)
                debate_agent_2.add_event(f"Agent 1 said, please discuss: {debate_response_1}", display=False)
                
                # Add initial responses to moderator
                moderator_agent.add_event(f"Agent 1 initial proposal:\n{debate_response_1}", display=False)
                moderator_agent.add_event(f"Agent 2 initial proposal:\n{debate_response_2}", display=False)
                
                debate_round = 1
                debate_history = []
                debate_history.append((debate_response_1, debate_response_2))
                
                while True:
                    console.print(Panel(f"✱ Debate Round {debate_round}", border_style="cyan", box=box.ROUNDED, width=100))
                    
                    debate_response_1 = debate_agent_1.ask()
                    conversation_log.append(f"Shifts Debate Round {debate_round} - Agent 1: {debate_response_1}")
                    debate_agent_1.add_memory(debate_response_1, display=False)
                    console.print(Panel(debate_response_1, title="Agent 1", border_style="blue", box=box.ROUNDED, width=100))
                    
                    debate_agent_2.add_event(f"Agent 1 said: {debate_response_1}", display=False)
                    debate_response_2 = debate_agent_2.ask()
                    conversation_log.append(f"Shifts Debate Round {debate_round} - Agent 2: {debate_response_2}")
                    debate_agent_2.add_memory(debate_response_2, display=False)
                    console.print(Panel(debate_response_2, title="Agent 2", border_style="green", box=box.ROUNDED, width=100))
                    
                    debate_agent_1.add_event(f"Agent 2 said: {debate_response_2}", display=False)
                    
                    # Add this round to history
                    debate_history.append((debate_response_1, debate_response_2))
                    
                    # Update moderator with latest round
                    moderator_agent.add_event(f"Debate Round {debate_round}:\nAgent 1: {debate_response_1}\nAgent 2: {debate_response_2}", display=False)
                    
                    # Ask moderator if debate should continue
                    moderator_response = moderator_agent.ask()
                    conversation_log.append(f"Shifts Debate Round {debate_round} - Moderator: {moderator_response}")
                    
                    # Check if moderator has decided on final shifts
                    shift_match = re.search(r'<shift>(.*?)</shift>', moderator_response, re.DOTALL)
                    if shift_match:
                        console.print(Panel("✱ Moderator has decided the debate has reached a conclusion", 
                                   border_style="orange3", 
                                   box=box.ROUNDED,
                                   width=100))
                        
                        shifts = shift_match.group(1).strip().split('\n')
                        console.print(Panel("\n".join([f"✱ {shift.strip()}" for shift in shifts if shift.strip()]), 
                                    title="Selected Shifts (After Debate)",
                                    border_style="violet", 
                                    box=box.ROUNDED,
                                    width=100))
                        
                        # Store selected shifts
                        selected_shifts = [shift.strip() for shift in shifts if shift.strip()]
                        conversation_log.append(f"Selected Shifts:\n" + "\n".join([f"- {shift}" for shift in selected_shifts]))
                        break
                    
                    # Display moderator's thoughts
                    console.print(Panel(moderator_response, title="Moderator", border_style="orange3", box=box.ROUNDED, width=100))
                    
                    debate_round += 1
                
                # Add the final decision to the main agent's memory
                agent.add_memory(f"After a debate between multiple agents, the following shifts were selected:\n" + 
                                "\n".join([f"- {shift}" for shift in selected_shifts]), display=False)
                
                break
            else:
                shifts = shift_match.group(1).strip().split('\n')
                console.print(Panel("\n".join([f"✱ {shift.strip()}" for shift in shifts if shift.strip()]),
                               title="Selected Shifts",
                               border_style="violet", 
                               box=box.ROUNDED,
                               width=100))
                
                # Store selected shifts
                selected_shifts = [shift.strip() for shift in shifts if shift.strip()]
                conversation_log.append(f"Selected Shifts:\n" + "\n".join([f"- {shift}" for shift in selected_shifts]))
                agent.add_memory(response, display=False)
                break
        else:
            agent.add_memory(response)
        user_input = console.input("[yellow]> ")
        conversation_log.append(f"User: {user_input}")
        agent.add_event(user_input, display=False)

    # Add selected shifts to auditor
    for shift in selected_shifts:
        try:
            # Check if the shift includes parameters
            if '(' in shift and ')' in shift:
                # Extract shift name and parameters
                shift_name = shift.split('(')[0].strip()
                params_str = shift.split('(')[1].split(')')[0].strip()
                
                # Parse parameters if they exist
                if params_str:
                    # Convert string representation of list to actual list
                    params = eval(params_str)
                    shift_instance = eval(f"{shift_name}({params})")
                else:
                    shift_instance = eval(f"{shift_name}()")
            else:
                # No parameters case
                shift_name = shift.strip()
                shift_instance = eval(f"{shift_name}()")
                
            auditor.add_shift(shift_instance)
        except Exception as e:
            console.print(f"[red]✗ Failed to add shift {shift}: {str(e)}")

    # Show running status
    console.print(Panel("✱ Running model audit...", 
                       border_style="orange3",
                       box=box.ROUNDED,
                       width=100))

    results = auditor.run()

    # Format results as a string for the agent
    def format_results_for_agent(results):
        formatted_results = "# Model Audit Results\n\n"
        
        for shift_name, shift_results in results.items():
            formatted_results += f"## {shift_name}\n\n"
            
            # Create header row with severities
            severities = list(shift_results.keys())
            header = "| Metric |"
            separator = "|--------|"
            for severity in severities:
                header += f" Severity {severity} |"
                separator += "------------|"
            
            formatted_results += header + "\n" + separator + "\n"
            
            # Add metric rows
            metrics = list(next(iter(shift_results.values())).keys())
            for metric in metrics:
                row = f"| {metric} |"
                for severity in severities:
                    value = shift_results[severity][metric]
                    formatted_value = f"{value:.4f}"
                    row += f" {formatted_value} |"
                formatted_results += row + "\n"
            
            formatted_results += "\n\n"
        
        return formatted_results

    # Send formatted results to agent
    results_table = format_results_for_agent(results)
    conversation_log.append(f"Model Audit Results:\n{results_table}")
    agent.add_event(f"Here are the audit results:\n\n{results_table}", display=False)
    agent.add_event(open("prompts/analysis_prompt.txt").read(), display=False)

    # Ask the agent to analyze the results
    console.print(Panel("✱ Analyzing audit results...", 
                       border_style="orange3", 
                       box=box.ROUNDED,
                       width=100))

    if multi_agent_decision:
        # do a multi agent debate here
        memory_lst = agent.memory_lst.copy()
        debate_agent_1 = Agent(model="anthropic/claude-3-7-sonnet-20250219", system_prompt=open("prompts/analysis_prompt.txt").read())
        debate_agent_2 = Agent(model="anthropic/claude-3-7-sonnet-20250219", system_prompt=open("prompts/analysis_prompt.txt").read())
        moderator_agent = Agent(model="anthropic/claude-3-7-sonnet-20250219", system_prompt=open("prompts/analysis_prompt.txt").read())

        debate_agent_1.memory_lst = memory_lst.copy()
        debate_agent_2.memory_lst = memory_lst.copy()
        moderator_agent.memory_lst = memory_lst.copy()

        prompt = f"""
        You got enough information to analyze the audit results.
        Please think step by step and provide a thorough analysis of the results.
        You will then debate your analysis with another assistant.
        """

        moderator_prompt = f"""
        You will be moderating a debate between two agents discussing the analysis of audit results.
        Let the debate continue as long as new insights are being generated.
        When you feel the debate has reached a conclusion, output your final analysis with the tag <analysis></analysis>.
        """

        debate_agent_1.add_event(prompt, display=False)
        debate_agent_2.add_event(prompt, display=False)
        moderator_agent.add_event(moderator_prompt, display=False)

        debate_response_1 = debate_agent_1.ask()
        debate_response_2 = debate_agent_2.ask()

        debate_agent_1.add_memory(debate_response_1, display=False)
        debate_agent_2.add_memory(debate_response_2, display=False)

        console.print(Panel("✱ Starting multi-agent debate on results analysis...", 
                   border_style="cyan", 
                   box=box.ROUNDED,
                   width=100))
        
        console.print(Panel(debate_response_1, title="Agent 1", border_style="blue", box=box.ROUNDED, width=100))
        console.print(Panel(debate_response_2, title="Agent 2", border_style="green", box=box.ROUNDED, width=100))

        debate_agent_1.add_event(f"Agent 2 said: {debate_response_2}", display=False)
        debate_agent_2.add_event(f"Agent 1 said: {debate_response_1}", display=False)
        
        # Add initial responses to moderator
        moderator_agent.add_event(f"Agent 1 initial analysis:\n{debate_response_1}", display=False)
        moderator_agent.add_event(f"Agent 2 initial analysis:\n{debate_response_2}", display=False)
        
        debate_round = 1
        debate_history = []
        debate_history.append((debate_response_1, debate_response_2))
        
        while True:
            console.print(Panel(f"✱ Debate Round {debate_round}", border_style="cyan", box=box.ROUNDED, width=100))
            
            debate_response_1 = debate_agent_1.ask()
            debate_agent_1.add_memory(debate_response_1, display=False)
            console.print(Panel(debate_response_1, title="Agent 1", border_style="blue", box=box.ROUNDED, width=100))
            
            debate_agent_2.add_event(f"Agent 1 said: {debate_response_1}", display=False)
            debate_response_2 = debate_agent_2.ask()
            debate_agent_2.add_memory(debate_response_2, display=False)
            console.print(Panel(debate_response_2, title="Agent 2", border_style="green", box=box.ROUNDED, width=100))
            
            debate_agent_1.add_event(f"Agent 2 said: {debate_response_2}", display=False)
            
            # Add this round to history
            debate_history.append((debate_response_1, debate_response_2))
            
            # Update moderator with latest round
            moderator_agent.add_event(f"Debate Round {debate_round}:\nAgent 1: {debate_response_1}\nAgent 2: {debate_response_2}", display=False)
            
            # Ask moderator if debate should continue
            moderator_response = moderator_agent.ask()
            
            # Check if moderator has decided on final analysis
            analysis_match = re.search(r'<analysis>(.*?)</analysis>', moderator_response, re.DOTALL)
            if analysis_match:
                console.print(Panel("✱ Moderator has decided the debate has reached a conclusion", 
                       border_style="orange3", 
                       box=box.ROUNDED,
                       width=100))
                
                analysis = analysis_match.group(1).strip()
                console.print(Panel(analysis, 
                        title="Final Analysis (After Debate)",
                        border_style="violet", 
                        box=box.ROUNDED,
                        width=100))
                
                # Store the final analysis
                analysis_response = analysis
                break
            
            # Display moderator's thoughts
            console.print(Panel(moderator_response, title="Moderator", border_style="orange3", box=box.ROUNDED, width=100))
            
            debate_round += 1
        
        # Add the final analysis to the main agent's memory
        agent.add_memory(f"After a debate between multiple agents, the following analysis was produced:\n{analysis_response}", display=False)
        conversation_log.append(f"Final Analysis:\n{analysis_response}")
    else:
        analysis_response = agent.ask()
        conversation_log.append(f"Analysis:\n{analysis_response}")
        agent.add_memory(analysis_response)

    # Allow the user to ask follow-up questions about the results
    console.print(Panel("✱ You can now ask questions about the audit results", 
                       border_style="orange3", 
                       box=box.ROUNDED,
                       width=100))

    while True:
        user_input = console.input("[yellow]> ")
        conversation_log.append(f"User: {user_input}")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
        agent.add_event(user_input, display=False)
        
        if multi_agent_decision and show_debate:
            # Multi-agent debate for follow-up questions
            debate_agent_1 = Agent(model="anthropic/claude-3-7-sonnet-20250219", system_prompt=open("prompts/analysis_prompt.txt").read())
            debate_agent_2 = Agent(model="anthropic/claude-3-7-sonnet-20250219", system_prompt=open("prompts/analysis_prompt.txt").read())
            
            # Copy the main agent's memory to the debate agents
            debate_agent_1.memory_lst = agent.memory_lst.copy()
            debate_agent_2.memory_lst = agent.memory_lst.copy()
            
            # Add the user's question
            debate_agent_1.add_event(user_input, display=False)
            debate_agent_2.add_event(user_input, display=False)
            
            # Get initial responses
            response_1 = debate_agent_1.ask()
            response_2 = debate_agent_2.ask()
            
            conversation_log.append(f"Debate Agent 1 Response: {response_1}")
            conversation_log.append(f"Debate Agent 2 Response: {response_2}")
            
            console.print(Panel("✱ Multi-agent responses to your question:", 
                           border_style="cyan", 
                           box=box.ROUNDED,
                           width=100))
            
            if show_debate:
                console.print(Panel(response_1, title="Agent 1", border_style="blue", box=box.ROUNDED, width=100))
                console.print(Panel(response_2, title="Agent 2", border_style="green", box=box.ROUNDED, width=100))
            
            # Combine the responses for the main agent
            agent.add_memory(f"Agent 1 response: {response_1}\n\nAgent 2 response: {response_2}", display=False)
        else:
            response = agent.ask()
            conversation_log.append(f"Assistant: {response}")
            agent.add_memory(response)

    # Save the conversation log to a file
    results_folder = "results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{results_folder}/conversation_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        # Write header with run info
        f.write("=" * 80 + "\n")
        f.write("MODEL AUDITOR CONVERSATION LOG\n")
        f.write("=" * 80 + "\n\n")
        
        # Write configuration information
        f.write("RUN CONFIGURATION:\n")
        f.write("-" * 50 + "\n")
        for line in conversation_log[:8]:  # First 8 lines contain configuration
            f.write(f"{line}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Write conversation history
        f.write("CONVERSATION HISTORY:\n")
        f.write("-" * 50 + "\n\n")
        
        # Skip configuration lines
        for line in conversation_log[9:]:
            # Add separators between different types of entries for readability
            if line.startswith("Model Audit Results:"):
                f.write("\n" + "=" * 80 + "\n")
                f.write("AUDIT RESULTS:\n")
                f.write("-" * 50 + "\n\n")
            elif line.startswith("Final Analysis:") or line.startswith("Analysis:"):
                f.write("\n" + "=" * 80 + "\n")
                f.write("ANALYSIS:\n")
                f.write("-" * 50 + "\n\n")
            elif line.startswith("Selected Metrics:") or line.startswith("Selected Metrics (After Debate):"):
                f.write("\n" + "-" * 50 + "\n")
                f.write("SELECTED METRICS:\n")
            elif line.startswith("Selected Shifts:") or line.startswith("Selected Shifts (After Debate):"):
                f.write("\n" + "-" * 50 + "\n")
                f.write("SELECTED SHIFTS:\n")
            elif line.startswith("Debate Round") or line.startswith("Shifts Debate Round"):
                f.write("\n" + "-" * 30 + "\n")
            
            f.write(f"{line}\n\n")
            
    console.print(f"[green]Conversation saved to {filename}")

def load_dataset(dataset_name, transform):
    """Load dataset based on name or path"""
    if dataset_name.lower() == "cifar10":
        return datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset_name.lower() == "cifar100":
        return datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    elif dataset_name.lower() == "imagenet":
        # Assuming ImageNet validation set is in ./data/imagenet/val
        return datasets.ImageFolder(root='./data/imagenet/val', transform=transform)
    elif dataset_name.lower() == "camelyon17":
        try:
            from wilds import get_dataset
            from torch.utils.data import Dataset

            # Create a wrapper class to handle Camelyon17's tuple structure
            class Camelyon17Wrapper(Dataset):
                def __init__(self, dataset, transform=None):
                    self.dataset = dataset
                    self.transform = transform
                
                def __len__(self):
                    return len(self.dataset)
                
                def __getitem__(self, idx):
                    # Camelyon17 returns (x, y, metadata) but we only need (x, y)
                    x, y, _ = self.dataset[idx]
                    if self.transform:
                        x = self.transform(x)
                    return x, y
            
            # Get the original dataset
            dataset = get_dataset(dataset="camelyon17", download=True)
            original_test_data = dataset.get_subset('test')
            
            # Wrap it with our custom wrapper
            test_data = Camelyon17Wrapper(original_test_data, transform=transform)
            
            console.print(f"[green]Successfully loaded Camelyon17 WILDS test dataset with {len(test_data)} samples")
            return test_data
        except ImportError:
            console.print("[red]WILDS package not installed. Install with: pip install wilds")
            raise ImportError("WILDS package not installed. Install with: pip install wilds")
    elif dataset_name.lower() == "chexpert":
        from torch.utils.data import Dataset

        class ChexpertDataset(Dataset):
            def __init__(self, csv_path, transform=None, is_train=True):
                self.df = pd.read_csv(csv_path)
                self.df = self.df.replace(np.nan, 0.0)
                self.df = self.df.replace(-1.0, 0.0)
                self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']
                self.df = self.df[self.df['AP/PA'] == 'AP']
                
                # Remove path prefix
                self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0-small/train/', '')
                if not is_train:
                    self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0-small/valid/', '')
                    
                self.paths = self.df['Path'].to_numpy()
                self.labels = self.df[['Atelectasis', 'Consolidation', 'Cardiomegaly', 'Pleural Effusion', 'Edema']].to_numpy()
                self.transform = transform if transform is not None else transforms.ToTensor()
                self.base_path = 'data/chexpert/train/' if is_train else 'data/chexpert/valid/'

            def __len__(self):
                return len(self.paths)
            
            def __getitem__(self, idx):
                img_path = self.base_path + str(self.paths[idx])
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
                label = torch.tensor(self.labels[idx], dtype=torch.float32)
                return image, label

        test_data = ChexpertDataset(csv_path='data/chexpert/train.csv', transform=transform, is_train=True)
        # Use only the last 5000 samples
        test_data = Subset(test_data, range(len(test_data)-5000, len(test_data)))
        console.print(f"[green]Successfully loaded Chexpert WILDS train dataset with {len(test_data)} samples")
        return test_data
    elif "mnist" in dataset_name.lower():
        # Handle MedMNIST datasets
        try:
            from medmnist import INFO, Evaluator
            from medmnist import PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST
            
            dataset_map = {
                "pathmnist": PathMNIST,
                "chestmnist": ChestMNIST,
                "dermamnist": DermaMNIST,
                "octmnist": OCTMNIST,
                "pneumoniamnist": PneumoniaMNIST,
                "retinamnist": RetinaMNIST
            }
            
            dataset_class = dataset_map.get(dataset_name.lower())
            if dataset_class:
                return dataset_class(split='test', transform=transform, download=True)
        except ImportError:
            console.print("[red]MedMNIST not installed. Install with: pip install medmnist")
    elif dataset_name.lower() == "isic":
        # Use ImageFolder for ISIC dataset (data/isic/train with benign and malignant folders)
        try:
            test_data = datasets.ImageFolder(root='data/isic/train', transform=transform)
            console.print(f"[green]Successfully loaded ISIC dataset with {len(test_data)} samples")
            console.print(f"[yellow]Classes: {test_data.classes}")
            return test_data
        except Exception as e:
            console.print(f"[red]Error loading ISIC dataset: {str(e)}")
            console.print("[yellow]Make sure data/isic/train exists with 'benign' and 'malignant' subfolders")
            raise
    elif dataset_name.lower() == "ham10000":
        from torch.utils.data import Dataset
        
        class HAM10000Dataset(Dataset):
            def __init__(self, root_dir, transform=None):
                self.root_dir = root_dir
                self.transform = transform
                self.classes = ['bkl', 'mel']  # benign (0) and malignant (1)
                self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
                
                self.images = []
                self.labels = []
                
                # Load images from both classes
                for class_name in self.classes:
                    class_dir = os.path.join(root_dir, class_name)
                    for img_name in os.listdir(class_dir):
                        if img_name.endswith(('.jpg', '.jpeg', '.png')):
                            self.images.append(os.path.join(class_dir, img_name))
                            self.labels.append(self.class_to_idx[class_name])
            
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, idx):
                img_path = self.images[idx]
                label = self.labels[idx]
                
                # Load and transform image
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                    
                return image, label

        test_data = HAM10000Dataset(root_dir='data/ham10000/vidir_modern', transform=transform)
        console.print(f"[green]Successfully loaded HAM10000 dataset with {len(test_data)} samples")
        return test_data

    # If not a known dataset, try as a folder path
    if Path(dataset_name).is_dir():
        return datasets.ImageFolder(root=dataset_name, transform=transform)
    
    raise ValueError(f"Dataset {dataset_name} not recognized and not a valid folder path")

def load_model(model_name, weights_path, num_classes, device):
    """Load model based on architecture name and weights"""
    if model_name.lower() == "siim-isic":
        # Load SIIM-ISIC model
        try:
            from architectures.siim_isic_model import SIIMISICClassifier
            console.print(f"[yellow]Loading SIIM-ISIC model from {weights_path}")
            model = SIIMISICClassifier(weights_path=weights_path)
            model = model.to(device)
            model.eval()
            return model
        except ImportError as e:
            console.print(f"[red]Error importing SIIM-ISIC model: {str(e)}")
            console.print("[red]Make sure geffnet is installed: pip install geffnet")
            raise
        except Exception as e:
            console.print(f"[red]Error loading SIIM-ISIC model: {str(e)}")
            raise
    elif model_name.lower() == "deepderm":
        # Load DeepDerm model
        try:
            from architectures.deepderm_model import DeepDermClassifier
            console.print(f"[yellow]Loading DeepDerm model from {weights_path}")
            model = DeepDermClassifier(weights_path=weights_path)
            model = model.to(device)
            model.eval()
            return model
        except ImportError as e:
            console.print(f"[red]Error importing DeepDerm model: {str(e)}")
            raise
        except Exception as e:
            console.print(f"[red]Error loading DeepDerm model: {str(e)}")
            raise
    elif model_name.lower() == "other" or weights_path.endswith(".onnx"):
        # Load ONNX model
        try:
            import onnxruntime as ort
            console.print(f"[yellow]Loading ONNX model from {weights_path}")
            # Create a wrapper class that mimics PyTorch model interface
            class ONNXModel:
                def __init__(self, path, device):
                    providers = ['CPUExecutionProvider']
                    if device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
                        providers.insert(0, 'CUDAExecutionProvider')
                    self.session = ort.InferenceSession(path, providers=providers)
                    self.device = device
                    self.input_name = self.session.get_inputs()[0].name
                    
                def to(self, device):
                    self.device = device
                    return self
                    
                def eval(self):
                    # ONNX models are always in eval mode
                    return self
                    
                def __call__(self, x):
                    # Convert to numpy for ONNX
                    if isinstance(x, torch.Tensor):
                        x = x.cpu().numpy()
                    outputs = self.session.run(None, {self.input_name: x})
                    # Convert back to torch tensor
                    return torch.tensor(outputs[0], device=self.device)
            
            return ONNXModel(weights_path, device)
        except ImportError:
            console.print("[red]ONNX Runtime not installed. Install with: pip install onnxruntime")
            raise
    else:
        # Load PyTorch model
        try:
            # Get model constructor from torchvision.models
            model_fn = getattr(models, model_name.lower())
            
            # Special handling for ViT models
            if model_name.lower().startswith('vit_'):
                # Initialize model with correct number of classes
                model = model_fn(weights=None, num_classes=num_classes)
            else:
                model = model_fn(weights=None)
                
                # Modify final layer for the number of classes
                if hasattr(model, 'fc'):
                    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
                elif hasattr(model, 'classifier'):
                    if isinstance(model.classifier, torch.nn.Sequential):
                        in_features = model.classifier[-1].in_features
                        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
                    else:
                        in_features = model.classifier.in_features
                        model.classifier = torch.nn.Linear(in_features, num_classes)
            
            # Load weights
            model.load_state_dict(torch.load(weights_path, map_location='cpu'))
            model = model.to(device)
            model.eval()
            return model
        except AttributeError:
            console.print(f"[red]Model {model_name} not found in torchvision.models")
            raise

def get_default_transform(dataset_name, model_name=None):
    """Get default transform for known datasets"""
    # Special transform for SIIM-ISIC model - only resize and ToTensor (model handles normalization internally)
    if model_name and model_name.lower() == "siim-isic":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    # Special transform for DeepDerm model - resize to 299x299 and scale to (-1,1) range
    elif model_name and model_name.lower() == "deepderm":
        return transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Scale to (-1,1) range
        ])
    elif dataset_name.lower() == "cifar10" or dataset_name.lower() == "cifar100":
        return transforms.Compose([
            transforms.Resize(224), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif dataset_name.lower() == "camelyon17":
        return transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    elif "mnist" in dataset_name.lower():
        # Different MedMNIST datasets have different number of channels
        # PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, RetinaMNIST have RGB images (3 channels)
        if any(x in dataset_name.lower() for x in ["pathmnist", "chestmnist", "dermamnist", "octmnist", "retinamnist"]):
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # Other MedMNIST datasets are grayscale (1 channel)
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
    elif dataset_name.lower() == "chexpert":
        return transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    elif dataset_name.lower() == "ham10000":
        return transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif dataset_name.lower() == "isic":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    else:
        # Default ImageNet transforms
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_num_classes(dataset_name, model_name=None):
    """Get number of classes for known datasets"""
    dataset_classes = {
        "isic": 2,
        "cifar10": 10,
        "cifar100": 100,
        "imagenet": 1000,
        "camelyon17": 2, 
        "pathmnist": 9,
        "chestmnist": 14,
        "dermamnist": 7,
        "octmnist": 4,
        "pneumoniamnist": 2,
        "retinamnist": 5,
        "chexpert": 5,
        "ham10000": 2
    }
    return dataset_classes.get(dataset_name.lower(), None)

if __name__ == "__main__":
    main()


