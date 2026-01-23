from utils.agent import Agent
from rich.console import Console
from rich.panel import Panel
from rich import box
from rich.text import Text
from rich.spinner import Spinner
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.padding import Padding
import re
import time
import random
from model_auditor import ModelAuditor
from model_auditor.metrics import *
from model_auditor.shifts import *
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import Subset, random_split
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

# Central model configuration - change this to switch LLM providers
MODEL = "anthropic/claude-sonnet-4-5-20250929"
#MODEL = "gpt-5.2-2025-12-11"

console = Console()

def extract_thinking_message(response):
    """Extract a brief thinking message from agent response"""
    # Look for <thinking> tags first
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', response, re.DOTALL)
    if thinking_match:
        thinking = thinking_match.group(1).strip()
        # Take first sentence or first 60 chars
        first_sentence = thinking.split('.')[0] if '.' in thinking else thinking
        return first_sentence[:60] + "..." if len(first_sentence) > 60 else first_sentence

    # Fallback: take first line of response
    first_line = response.split('\n')[0].strip()
    if len(first_line) > 60:
        return first_line[:60] + "..."
    return first_line if first_line else "Thinking..."

def get_audit_message(shift_name):
    """Get a creative status message for a shift"""
    return f"Auditing {shift_name}..."

# Setup argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Model Auditor - Evaluate robustness of image classification, segmentation, and detection models")

    # Task type
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "segmentation", "detection"],
                        help="Task type: 'classification', 'segmentation', or 'detection'")

    # Model architecture
    parser.add_argument("--model", type=str, default="resnet18",
                        help="Model architecture (any torchvision model, 'unet' for segmentation, 'fasterrcnn' for detection, 'siim-isic', 'deepderm', or 'other' for ONNX)")

    # Dataset
    parser.add_argument("--dataset", type=str, default="CIFAR10",
                        help="Dataset name (torchvision, medmnist, 'kvasir-seg' for segmentation, 'polyp-c5-detection' for detection, or path to folder)")

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

def run_debate(memory_lst, system_prompt, decision_type, multi_agent_decision, show_debate, conversation_log):
    """
    Generic debate function for metrics/shifts/analysis decisions.
    Returns (selected_items, is_continue) where is_continue is only relevant for interpretation debates.
    """
    if not multi_agent_decision:
        return None, False

    debate_agent_1 = Agent(model=MODEL, system_prompt=system_prompt)
    debate_agent_2 = Agent(model=MODEL, system_prompt=system_prompt)
    moderator_agent = Agent(model=MODEL, system_prompt=system_prompt)

    debate_agent_1.memory_lst = memory_lst.copy()
    debate_agent_2.memory_lst = memory_lst.copy()
    moderator_agent.memory_lst = memory_lst.copy()

    if decision_type == "combined":
        prompt = """
        You have enough information to select the best metrics and shifts for the task.
        Please think step by step and justify your choices.
        You will debate these with another assistant.

        IMPORTANT: Start your response with a <thinking>brief summary of what you're considering</thinking> tag (5-8 words max).
        Example: <thinking>Prioritizing robustness metrics for medical imaging</thinking>
        """
        moderator_prompt = """
        You are moderating a debate between two agents discussing metrics and shifts selection.
        Let the debate continue as long as new insights are being generated.
        When you feel the debate has reached a conclusion, output both:
        <metric>metric1\\nmetric2\\nmetric3</metric>
        <shift>shift1\\nshift2\\nshift3</shift>

        IMPORTANT: Start your response with a <thinking>brief summary of current debate stage</thinking> tag (5-8 words max).
        Example: <thinking>Evaluating consensus on metric priorities</thinking>
        """
        tag_patterns = [("metric", r'<metric>(.*?)</metric>'), ("shift", r'<shift>(.*?)</shift>')]
    elif decision_type == "interpretation":
        prompt = """
        Analyze the audit results and determine if another round of testing is needed.
        Justify your reasoning.
        You will debate this with another assistant.

        IMPORTANT: Start your response with a <thinking>brief summary of what you're analyzing</thinking> tag (5-8 words max).
        Example: <thinking>Assessing model's contrast sensitivity gaps</thinking>
        """
        moderator_prompt = """
        You are moderating a debate about whether additional audit rounds are needed.
        Let the debate continue as long as new insights are being generated.
        When you feel the debate has reached a conclusion, output either:
        <continue><reasoning>...</reasoning><additional_metrics>...</additional_metrics><additional_shifts>...</additional_shifts></continue>
        OR
        <stop><reasoning>...</reasoning></stop>

        IMPORTANT: Start your response with a <thinking>brief summary of current debate stage</thinking> tag (5-8 words max).
        Example: <thinking>Weighing need for additional noise tests</thinking>
        """
        tag_patterns = [("stop", r'<stop>.*?</stop>'), ("continue", r'<continue>.*?</continue>')]

    debate_agent_1.add_event(prompt, display=False)
    debate_agent_2.add_event(prompt, display=False)
    moderator_agent.add_event(moderator_prompt, display=False)

    with console.status("[cyan]Initial perspectives forming...", spinner="dots") as status:
        debate_response_1 = debate_agent_1.ask()
        status.update(f"[blue]{extract_thinking_message(debate_response_1)}")
        debate_response_2 = debate_agent_2.ask()

    conversation_log.append(f"{decision_type.capitalize()} Debate Agent 1: {debate_response_1}")
    conversation_log.append(f"{decision_type.capitalize()} Debate Agent 2: {debate_response_2}")

    debate_agent_1.add_memory(debate_response_1, display=False)
    debate_agent_2.add_memory(debate_response_2, display=False)

    if show_debate:
        console.print(Padding(Panel(f"[cyan]✱ Multi-agent debate: {decision_type}",
                           border_style="cyan", box=box.ROUNDED, width=100), (0, 0, 0, 2)))
        # Show thinking messages from agents
        msg1 = extract_thinking_message(debate_response_1)
        msg2 = extract_thinking_message(debate_response_2)
        console.print(f"    [blue]▸[/blue] {msg1}")
        console.print(f"    [green]▸[/green] {msg2}")

    debate_agent_1.add_event(f"Agent 2 said: {debate_response_2}", display=False)
    debate_agent_2.add_event(f"Agent 1 said: {debate_response_1}", display=False)

    moderator_agent.add_event(f"Agent 1 initial proposal:\\n{debate_response_1}", display=False)
    moderator_agent.add_event(f"Agent 2 initial proposal:\\n{debate_response_2}", display=False)

    debate_round = 1

    with console.status("[cyan]Debating...", spinner="dots") as status:
        while True:
            debate_response_1 = debate_agent_1.ask()
            status.update(f"[blue]{extract_thinking_message(debate_response_1)}")
            conversation_log.append(f"{decision_type.capitalize()} Debate Round {debate_round} - Agent 1: {debate_response_1}")
            debate_agent_1.add_memory(debate_response_1, display=False)

            debate_agent_2.add_event(f"Agent 1 said: {debate_response_1}", display=False)
            debate_response_2 = debate_agent_2.ask()
            status.update(f"[green]{extract_thinking_message(debate_response_2)}")
            conversation_log.append(f"{decision_type.capitalize()} Debate Round {debate_round} - Agent 2: {debate_response_2}")
            debate_agent_2.add_memory(debate_response_2, display=False)

            debate_agent_1.add_event(f"Agent 2 said: {debate_response_2}", display=False)

            moderator_agent.add_event(f"Debate Round {debate_round}:\\nAgent 1: {debate_response_1}\\nAgent 2: {debate_response_2}", display=False)

            moderator_response = moderator_agent.ask()
            status.update(f"[orange3]{extract_thinking_message(moderator_response)}")
            conversation_log.append(f"{decision_type.capitalize()} Debate Round {debate_round} - Moderator: {moderator_response}")

            # Check if moderator has decided
            found_decision = False
            result = {}

            for tag_name, pattern in tag_patterns:
                match = re.search(pattern, moderator_response, re.DOTALL)
                if match:
                    found_decision = True
                    result[tag_name] = match.group(0) if tag_name in ["stop", "continue"] else match.group(1).strip()

            if found_decision:
                break

            if show_debate:
                console.print(f"    [dim]→ Round {debate_round} complete, continuing...[/dim]")

            debate_round += 1

    if show_debate:
        console.print(f"    [orange3]✓[/orange3] Consensus reached after {debate_round} rounds")

    return result, moderator_response

def format_results_for_agent(results):
    """Format results as a string for the agent"""
    formatted_results = "# Model Audit Results\\n\\n"

    for shift_name, shift_results in results.items():
        formatted_results += f"## {shift_name}\\n\\n"

        # Create header row with severities
        severities = list(shift_results.keys())
        header = "| Metric |"
        separator = "|--------|"
        for severity in severities:
            header += f" Severity {severity} |"
            separator += "------------|"

        formatted_results += header + "\\n" + separator + "\\n"

        # Add metric rows
        metrics = list(next(iter(shift_results.values())).keys())
        for metric in metrics:
            row = f"| {metric} |"
            for severity in severities:
                value = shift_results[severity][metric]
                formatted_value = f"{value:.4f}"
                row += f" {formatted_value} |"
            formatted_results += row + "\\n"

        formatted_results += "\\n\\n"

    return formatted_results

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
            device = "cpu"
        else:
            device = "cpu"

    console.print(f"  [yellow]Using device: {device}")

    # Initialize conversation log
    conversation_log = []
    conversation_log.append(f"Conversation started at: {datetime.datetime.now()}")
    conversation_log.append(f"Task: {args.task}")
    conversation_log.append(f"Model: {args.model}")
    conversation_log.append(f"Dataset: {args.dataset}")
    conversation_log.append(f"Weights: {args.weights}")
    conversation_log.append(f"Device: {device}")
    conversation_log.append(f"Subset: {args.subset}")
    conversation_log.append(f"Multi-agent debate: {args.multi_agent_decision}")
    conversation_log.append(f"Show debate: {args.show_debate}")
    conversation_log.append("\\nConversation history:")

    # Get transform for the dataset
    transform = get_default_transform(args.dataset, args.model)

    # Load dataset
    try:
        test_set = load_dataset(args.dataset, transform)
        console.print(f"  [green]Successfully loaded dataset: {args.dataset}")
    except Exception as e:
        console.print(f"  [red]Error loading dataset: {str(e)}")
        return

    # Create subset for evaluation
    if args.subset and args.subset < len(test_set):
        subset = Subset(test_set, torch.randperm(len(test_set))[:args.subset])
        console.print(f"  [yellow]Using subset of {args.subset} samples")
    else:
        subset = test_set

    # Determine number of classes
    num_classes = get_num_classes(args.dataset, args.model)
    if not num_classes and hasattr(test_set, 'classes'):
        num_classes = len(test_set.classes)
    elif not num_classes:
        console.print("[yellow]Could not determine number of classes, assuming 10")
        num_classes = 10

    console.print(f"  [yellow]Number of classes: {num_classes}")

    # Load model
    try:
        model = load_model(args.model, args.weights, num_classes, device)
        console.print(f"  [green]Successfully loaded model: {args.model}")
    except Exception as e:
        console.print(f"  [red]Error loading model: {str(e)}")
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


    # Initialize auditor
    auditor = ModelAuditor(model, subset, device=device, batch_size=64, task_type=args.task)

    # Maximum number of audit rounds
    MAX_AUDIT_ROUNDS = 1

    # Storage for all audit rounds
    all_audit_results = []
    audit_round_descriptions = []

    ### PHASE 1: COMBINED QUESTIONING (METRICS + SHIFTS)
    task_type_labels = {"classification": "classification", "segmentation": "segmentation", "detection": "detection"}
    task_type_label = task_type_labels.get(args.task, "classification")
    console.print(Padding(Panel(f"✱ Welcome to the Model Auditor! Please describe your {task_type_label} task to get started.",
                       border_style="orange3",
                       box=box.ROUNDED,
                       width=100), (0, 0, 0, 2)))

    # Initialize questioning agent - single prompt for all tasks
    questioning_agent = Agent(model=MODEL,
                             system_prompt=open("prompts/combined_initial_prompt.txt").read())

    task_description = console.input("  [yellow]> ")
    conversation_log.append(f"User: {task_description}")
    questioning_agent.add_event(task_description, display=False)

    selected_metrics = []
    selected_shifts = []

    # Ask the first question immediately
    response = questioning_agent.ask()
    conversation_log.append(f"Assistant: {response}")
    questioning_agent.add_memory(response, display=True)

    # Combined loop to get both metrics AND shifts
    while True:
        # Check if we have final selections
        metric_match = re.search(r'<metric>(.*?)</metric>', response, re.DOTALL)
        shift_match = re.search(r'<shift>(.*?)</shift>', response, re.DOTALL)

        if metric_match and shift_match:
            if multi_agent_decision:
                # Run debate
                debate_result, moderator_response = run_debate(
                    questioning_agent.memory_lst,
                    open("prompts/combined_initial_prompt.txt").read(),
                    "combined",
                    multi_agent_decision,
                    show_debate,
                    conversation_log
                )

                if debate_result and "metric" in debate_result and "shift" in debate_result:
                    metrics = debate_result["metric"].strip().split('\n')
                    shifts = debate_result["shift"].strip().split('\n')
                    selected_metrics = [m.strip() for m in metrics if m.strip()]
                    selected_shifts = [s.strip() for s in shifts if s.strip()]
            else:
                # No debate - use direct response
                metrics = metric_match.group(1).strip().split('\n')
                shifts = shift_match.group(1).strip().split('\n')
                selected_metrics = [m.strip() for m in metrics if m.strip()]
                selected_shifts = [s.strip() for s in shifts if s.strip()]

            # Display selections
            console.print(Padding(Panel("\n".join([f"✱ {metric}" for metric in selected_metrics]),
                            title="Selected Metrics",
                            border_style="violet",
                            box=box.ROUNDED,
                            width=100), (0, 0, 0, 2)))
            console.print(Padding(Panel("\n".join([f"✱ {shift}" for shift in selected_shifts]),
                            title="Selected Shifts",
                            border_style="violet",
                            box=box.ROUNDED,
                            width=100), (0, 0, 0, 2)))

            conversation_log.append(f"Selected Metrics:\n" + "\n".join([f"- {m}" for m in selected_metrics]))
            conversation_log.append(f"Selected Shifts:\n" + "\n".join([f"- {s}" for s in selected_shifts]))
            break

        # Get user input
        user_input = console.input("  [yellow]> ")
        conversation_log.append(f"User: {user_input}")
        questioning_agent.add_event(user_input, display=False)

        response = questioning_agent.ask()
        conversation_log.append(f"Assistant: {response}")
        questioning_agent.add_memory(response, display=True)

    # Add initial metrics to auditor
    for metric in selected_metrics:
        metric_name = metric.split()[0].strip()
        try:
            metric_instance = eval(f"{metric_name}()")
            auditor.add_metric(metric_instance)
        except Exception as e:
            console.print(f"  [red]✗ Failed to add metric {metric_name}: {str(e)}")

    # Add initial shifts to auditor
    for shift in selected_shifts:
        try:
            if '(' in shift and ')' in shift:
                shift_name = shift.split('(')[0].strip()
                params_str = shift.split('(')[1].split(')')[0].strip()
                if params_str:
                    params = eval(params_str)
                    shift_instance = eval(f"{shift_name}({params})")
                else:
                    shift_instance = eval(f"{shift_name}()")
            else:
                shift_name = shift.strip()
                shift_instance = eval(f"{shift_name}()")
            auditor.add_shift(shift_instance)
        except Exception as e:
            console.print(f"  [red]✗ Failed to add shift {shift}: {str(e)}")

    ### PHASE 2: MULTI-ROUND AUDITS
    audit_round = 1
    interpretation_agent = Agent(model=MODEL,
                                 system_prompt=open("prompts/interpretation_prompt.txt").read())

    # Add initial context to interpretation agent
    interpretation_agent.add_event(f"Task description: {task_description}", display=False)
    interpretation_agent.add_event(f"Initial metrics: {', '.join(selected_metrics)}", display=False)
    interpretation_agent.add_event(f"Initial shifts: {', '.join(selected_shifts)}", display=False)

    while audit_round <= MAX_AUDIT_ROUNDS:
        # Get shift names for display
        shift_names = [shift.__class__.__name__ for shift in auditor.shifts]
        shift_display = ", ".join(shift_names[:3]) + (f" +{len(shift_names)-3} more" if len(shift_names) > 3 else "")

        with console.status(f"[orange3]Running audit round {audit_round}/{MAX_AUDIT_ROUNDS}: {shift_display}...", spinner="aesthetic") as status:
            results = auditor.run()
            results_table = format_results_for_agent(results)

        console.print(f"  [green]✓[/green] Audit round {audit_round} complete ({len(auditor.metrics)} metrics × {len(auditor.shifts)} shifts)")

        all_audit_results.append(results)
        audit_round_descriptions.append(f"Round {audit_round}: {len(auditor.metrics)} metrics, {len(auditor.shifts)} shifts")
        conversation_log.append(f"\\n=== AUDIT ROUND {audit_round} ===\\n{results_table}")

        # Send results to interpretation agent
        interpretation_agent.add_event(f"Audit Round {audit_round} Results:\\n{results_table}", display=False)
        interpretation_agent.add_event(f"Should we run another audit round? (Current round: {audit_round}/{MAX_AUDIT_ROUNDS})", display=False)

        # Get interpretation
        if audit_round >= MAX_AUDIT_ROUNDS:
            console.print(f"  [orange3]✱[/orange3] Maximum rounds reached. Moving to final analysis...")
            break

        with console.status("[cyan]Analyzing results...", spinner="dots") as status:
            interpretation_response = interpretation_agent.ask()
            status.update(f"[cyan]{extract_thinking_message(interpretation_response)}")
        conversation_log.append(f"Interpretation Agent Round {audit_round}: {interpretation_response}")

        # Check if we should continue or stop
        continue_match = re.search(r'<continue>(.*?)</continue>', interpretation_response, re.DOTALL)
        stop_match = re.search(r'<stop>', interpretation_response, re.DOTALL)

        if stop_match:
            console.print(f"  [green]✓[/green] Analysis complete - sufficient information gathered")
            break
        elif continue_match:
            # Extract additional metrics and shifts
            additional_metrics_match = re.search(r'<additional_metrics>(.*?)</additional_metrics>', continue_match.group(1), re.DOTALL)
            additional_shifts_match = re.search(r'<additional_shifts>(.*?)</additional_shifts>', continue_match.group(1), re.DOTALL)

            if additional_metrics_match:
                new_metrics = [m.strip() for m in additional_metrics_match.group(1).strip().split('\\n') if m.strip()]
                console.print(f"  [cyan]➜[/cyan] Adding {len(new_metrics)} new metric(s) for round {audit_round + 1}")
                for m in new_metrics:
                    console.print(f"    [dim]•[/dim] {m}")
                for metric in new_metrics:
                    metric_name = metric.split()[0].strip()
                    try:
                        metric_instance = eval(f"{metric_name}()")
                        auditor.add_metric(metric_instance)
                    except Exception as e:
                        console.print(f"  [red]✗ Failed to add metric {metric_name}: {str(e)}")

            if additional_shifts_match:
                new_shifts = [s.strip() for s in additional_shifts_match.group(1).strip().split('\n') if s.strip()]
                console.print(f"  [cyan]➜[/cyan] Adding {len(new_shifts)} new shift(s) for round {audit_round + 1}")
                for s in new_shifts:
                    console.print(f"    [dim]•[/dim] {s}")
                for shift in new_shifts:
                    try:
                        if '(' in shift and ')' in shift:
                            shift_name = shift.split('(')[0].strip()
                            params_str = shift.split('(')[1].split(')')[0].strip()
                            if params_str:
                                params = eval(params_str)
                                shift_instance = eval(f"{shift_name}({params})")
                            else:
                                shift_instance = eval(f"{shift_name}()")
                        else:
                            shift_name = shift.strip()
                            shift_instance = eval(f"{shift_name}()")
                        auditor.add_shift(shift_instance)
                    except Exception as e:
                        console.print(f"  [red]✗ Failed to add shift {shift}: {str(e)}")

            audit_round += 1
        else:
            # No clear decision - default to continuing if not at max
            console.print(f"  [yellow]⚠[/yellow] Continuing to next round...")
            audit_round += 1

    ### PHASE 3: FINAL COMPREHENSIVE REPORT
    with console.status("[orange3]Synthesizing comprehensive report...", spinner="aesthetic") as status:
        report_agent = Agent(model=MODEL,
                            system_prompt=open("prompts/final_report_prompt.txt").read())

        # Add all context to report agent
        status.update("[orange3]Compiling audit results...")
        report_agent.add_event(f"Task description: {task_description}", display=False)
        report_agent.add_event(f"Total audit rounds completed: {len(all_audit_results)}", display=False)

        for i, (results, description) in enumerate(zip(all_audit_results, audit_round_descriptions), 1):
            results_table = format_results_for_agent(results)
            report_agent.add_event(f"{description}\\n{results_table}", display=False)

        report_agent.add_event("Please generate the comprehensive final report now.", display=False)

        status.update("[orange3]Generating insights and recommendations...")
        final_report = report_agent.ask()
        conversation_log.append(f"\\n=== FINAL COMPREHENSIVE REPORT ===\\n{final_report}")

    console.print(Padding(Panel(Markdown(final_report),
                       title="Final Audit Report",
                       border_style="green",
                       box=box.ROUNDED,
                       width=100), (0, 0, 0, 2)))

    # Interactive Q&A
    console.print(Padding(Panel("✱ You can now ask questions about the audit (type 'exit' to quit)",
                       border_style="orange3",
                       box=box.ROUNDED,
                       width=100), (0, 0, 0, 2)))

    while True:
        user_input = console.input("  [yellow]> ")
        conversation_log.append(f"User: {user_input}")
        if user_input.lower() in ["exit", "quit", "q"]:
            break

        report_agent.add_event(user_input, display=False)
        response = report_agent.ask()
        conversation_log.append(f"Assistant: {response}")
        console.print(Padding(Panel(Markdown(response), border_style="blue", box=box.ROUNDED, width=100), (0, 0, 0, 2)))

    # Save conversation log
    results_folder = "results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{results_folder}/conversation_{timestamp}.txt"

    with open(filename, 'w') as f:
        f.write("=" * 80 + "\\n")
        f.write("MODEL AUDITOR CONVERSATION LOG\\n")
        f.write("=" * 80 + "\\n\\n")
        for line in conversation_log:
            f.write(f"{line}\\n\\n")

    console.print(f"  [green]Conversation saved to {filename}")

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
            original_test_data = dataset.get_subset('val')
            
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
                return dataset_class(split='val', transform=transform, download=True)
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
    elif dataset_name.lower() == "kvasir-seg" or dataset_name.lower() == "kvasir":
        # Kvasir-SEG polyp segmentation dataset
        from torch.utils.data import Dataset
        
        class KvasirSEGDataset(Dataset):
            """Kvasir-SEG Polyp Dataset for segmentation - returns (image, mask) tuples"""
            def __init__(self, root_dir='data/kvasir-seg', img_transform=None, mask_transform=None):
                self.root_dir = Path(root_dir)
                self.img_transform = img_transform
                self.mask_transform = mask_transform
                
                images_dir = self.root_dir / "Kvasir-SEG" / "images"
                masks_dir = self.root_dir / "Kvasir-SEG" / "masks"
                
                if not images_dir.exists():
                    raise FileNotFoundError(
                        f"Kvasir-SEG dataset not found at {self.root_dir}. "
                        "Run examples/segmentation/train_segmentation.py to download it."
                    )
                
                self.images = sorted(images_dir.glob("*.jpg"))
                self.masks = sorted(masks_dir.glob("*.jpg"))
            
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, idx):
                image = Image.open(self.images[idx]).convert("RGB")
                mask = Image.open(self.masks[idx]).convert("L")
                
                if self.img_transform:
                    image = self.img_transform(image)
                if self.mask_transform:
                    mask = self.mask_transform(mask)
                
                mask = (mask > 0.5).float()
                return image, mask
        
        # Segmentation transforms
        img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        full_dataset = KvasirSEGDataset(root_dir='data/kvasir-seg', img_transform=img_transform, mask_transform=mask_transform)
        
        # Use same fixed split as training script (70/15/15) with same seed
        # Return only the TEST split for auditing (same held-out data)
        total = len(full_dataset)
        train_size = int(total * 0.70)
        val_size = int(total * 0.15)
        test_size = total - train_size - val_size
        
        _, _, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        console.print(f"[green]Successfully loaded Kvasir-SEG test split with {len(test_dataset)} samples")
        return test_dataset
    elif dataset_name.lower() == "polyp-c5" or dataset_name.lower() == "polyp_c5":
        # Polyp C5 dataset for segmentation validation/testing
        from torch.utils.data import Dataset

        class PolypDataset(Dataset):
            """Polyp Dataset - returns (image, mask) tuples. Supports C1-C6 datasets."""
            def __init__(self, dataset_id, base_dir='data', img_transform=None, mask_transform=None):
                self.dataset_id = dataset_id
                self.root_dir = Path(base_dir) / f"data_C{dataset_id}"
                self.img_transform = img_transform
                self.mask_transform = mask_transform

                images_dir = self.root_dir / f"images_C{dataset_id}"
                self.masks_dir = self.root_dir / f"masks_C{dataset_id}"

                if not images_dir.exists():
                    raise FileNotFoundError(
                        f"Polyp C{dataset_id} dataset not found at {self.root_dir}. "
                        f"Please ensure the data/data_C{dataset_id} directory contains "
                        f"images_C{dataset_id} and masks_C{dataset_id} subdirectories."
                    )

                self.samples = []
                all_images = sorted(images_dir.glob("*.jpg"))

                for img_path in all_images:
                    mask_name = img_path.stem + "_mask.jpg"
                    mask_path = self.masks_dir / mask_name

                    if not mask_path.exists() and img_path.stem.endswith(']'):
                        clean_stem = img_path.stem[:-1]
                        alt_mask_path = self.masks_dir / (clean_stem + "_mask.jpg")
                        if alt_mask_path.exists():
                            mask_path = alt_mask_path

                    if mask_path.exists():
                        self.samples.append((img_path, mask_path))

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                img_path, mask_path = self.samples[idx]

                image = Image.open(img_path).convert("RGB")
                mask = Image.open(mask_path).convert("L")

                if self.img_transform:
                    image = self.img_transform(image)
                if self.mask_transform:
                    mask = self.mask_transform(mask)

                mask = (mask > 0.5).float()
                return image, mask

        # Segmentation transforms - use 128x128 to match training
        img_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        mask_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        test_dataset = PolypDataset(
            dataset_id=5,
            base_dir='data',
            img_transform=img_transform,
            mask_transform=mask_transform
        )

        console.print(f"[green]Successfully loaded Polyp C5 dataset with {len(test_dataset)} samples")
        return test_dataset
    elif dataset_name.lower() == "polyp-c5-detection" or dataset_name.lower() == "polyp_c5_detection":
        # Polyp C5 dataset for object detection
        from torch.utils.data import Dataset

        def load_bboxes_from_file(bbox_path):
            """Load bounding boxes from a bbox text file."""
            bboxes = []
            if not bbox_path.exists():
                return bboxes
            with open(bbox_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5 and parts[0] == 'polyp':
                        x_min, y_min, x_max, y_max = map(int, parts[1:5])
                        if x_max > x_min and y_max > y_min:
                            bboxes.append([x_min, y_min, x_max, y_max])
            return bboxes

        class PolypDetectionDataset(Dataset):
            """Polyp Detection Dataset - returns (image, target) tuples for Faster R-CNN."""
            def __init__(self, dataset_id, base_dir='data', transform=None):
                self.dataset_id = dataset_id
                self.root_dir = Path(base_dir) / f"data_C{dataset_id}"
                self.transform = transform

                images_dir = self.root_dir / f"images_C{dataset_id}"
                self.bbox_dir = self.root_dir / f"bbox_C{dataset_id}"

                if not images_dir.exists():
                    raise FileNotFoundError(
                        f"Polyp C{dataset_id} dataset not found at {self.root_dir}."
                    )

                self.samples = []
                all_images = sorted(images_dir.glob("*.jpg"))

                for img_path in all_images:
                    bbox_name = img_path.stem + "_mask.txt"
                    bbox_path = self.bbox_dir / bbox_name
                    if bbox_path.exists():
                        self.samples.append((img_path, bbox_path))

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                img_path, bbox_path = self.samples[idx]

                image = Image.open(img_path).convert("RGB")
                orig_w, orig_h = image.size

                bboxes = load_bboxes_from_file(bbox_path)

                if self.transform:
                    image = self.transform(image)
                    _, new_h, new_w = image.shape

                    scale_x = new_w / orig_w
                    scale_y = new_h / orig_h

                    scaled_bboxes = []
                    for box in bboxes:
                        scaled_box = [
                            box[0] * scale_x,
                            box[1] * scale_y,
                            box[2] * scale_x,
                            box[3] * scale_y
                        ]
                        scaled_bboxes.append(scaled_box)
                    bboxes = scaled_bboxes

                if len(bboxes) > 0:
                    boxes = torch.as_tensor(bboxes, dtype=torch.float32)
                    labels = torch.ones((len(bboxes),), dtype=torch.int64)
                    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                else:
                    boxes = torch.zeros((0, 4), dtype=torch.float32)
                    labels = torch.zeros((0,), dtype=torch.int64)
                    area = torch.zeros((0,), dtype=torch.float32)

                target = {
                    "boxes": boxes,
                    "labels": labels,
                    "image_id": torch.tensor([idx]),
                    "area": area,
                    "iscrowd": torch.zeros((len(bboxes),), dtype=torch.int64)
                }

                return image, target

        # Detection transforms (320x320 to match MobileNet training)
        det_transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
        ])

        test_dataset = PolypDetectionDataset(
            dataset_id=5,
            base_dir='data',
            transform=det_transform
        )

        console.print(f"[green]Successfully loaded Polyp C5 detection dataset with {len(test_dataset)} samples")
        return test_dataset

    # If not a known dataset, try as a folder path
    if Path(dataset_name).is_dir():
        return datasets.ImageFolder(root=dataset_name, transform=transform)
    
    raise ValueError(f"Dataset {dataset_name} not recognized and not a valid folder path")

def load_model(model_name, weights_path, num_classes, device):
    """Load model based on architecture name and weights"""
    if model_name.lower() == "fasterrcnn":
        # Load Faster R-CNN for detection (MobileNet V3 backbone)
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        console.print(f"[yellow]Loading Faster R-CNN (MobileNet V3) detection model from {weights_path}")
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)

        # Replace the classifier head for num_classes (background + polyp = 2)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        model = model.to(device)
        model.eval()
        return model
    elif model_name.lower() == "unet":
        # Load U-Net for segmentation
        import torch.nn as nn
        
        class DoubleConv(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.double_conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            def forward(self, x):
                return self.double_conv(x)
        
        class UNet(nn.Module):
            def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
                super().__init__()
                self.downs = nn.ModuleList()
                self.ups = nn.ModuleList()
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                
                for feature in features:
                    self.downs.append(DoubleConv(in_channels, feature))
                    in_channels = feature
                
                self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
                
                for feature in reversed(features):
                    self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
                    self.ups.append(DoubleConv(feature * 2, feature))
                
                self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
            
            def forward(self, x):
                skip_connections = []
                for down in self.downs:
                    x = down(x)
                    skip_connections.append(x)
                    x = self.pool(x)
                
                x = self.bottleneck(x)
                skip_connections = skip_connections[::-1]
                
                for idx in range(0, len(self.ups), 2):
                    x = self.ups[idx](x)
                    skip = skip_connections[idx // 2]
                    if x.shape != skip.shape:
                        x = nn.functional.interpolate(x, size=skip.shape[2:])
                    x = torch.cat((skip, x), dim=1)
                    x = self.ups[idx + 1](x)
                
                return self.final_conv(x)
        
        console.print(f"[yellow]Loading U-Net segmentation model from {weights_path}")
        model = UNet(in_channels=3, out_channels=1)
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        model = model.to(device)
        model.eval()
        return model
    elif model_name.lower() == "siim-isic":
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
        "ham10000": 2,
        "polyp-c5-detection": 2,  # background + polyp
        "polyp_c5_detection": 2
    }
    return dataset_classes.get(dataset_name.lower(), None)

if __name__ == "__main__":
    main()


