from litellm import completion
from anthropic import Anthropic
import os
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich import box

os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class Agent:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.5, system_prompt: str = None):
        self.model = model
        self.temperature = temperature
        self.memory_lst = []
        self.is_anthropic = model.startswith("anthropic")
        self.console = Console()
        
        # Initialize memory with system prompt if provided
        if system_prompt:
            self.memory_lst = [{"role": "user", "content": system_prompt}]
        
        # Initialize Anthropic client if using Anthropic models
        if self.is_anthropic:
            self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    def generate_response(self, messages: list[dict], temperature: float = None, max_tokens: int = 5000) -> str:            
        if self.is_anthropic:
            # Convert the message format for Anthropic API
            anthropic_messages = []
            for msg in messages:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Determine which Claude model to use
            # If model is just "anthropic", default to claude-3-7-sonnet
            model_name = "claude-3-7-sonnet-20250219" if self.model == "anthropic" else self.model.replace("anthropic/", "")
            
            # Call Anthropic API
            response = self.anthropic_client.messages.create(
                model=model_name,
                messages=anthropic_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.content[0].text
        else:
            response = completion(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response["choices"][0]["message"]["content"]
            
    def add_memory(self, message: str, display: bool = True):
        self.memory_lst.append({"role": "assistant", "content": f"{message}"})
        if display:
            self._display_message(message, "assistant")
        
    def add_event(self, event: str, display: bool = True):
        self.memory_lst.append({"role": "user", "content": f"{event}"})
        if display:
            self._display_message(event, "user")
        
    def ask(self, temperature = None):
        if temperature is None:
            temperature = self.temperature
        response = self.generate_response(self.memory_lst, temperature)
        return response
    
    def _display_message(self, content: str, role: str):
        """Display formatted messages in the console using rich."""
        if role == "user":
            self.console.print(Panel(
                content, 
                title="User", 
                border_style="deep_sky_blue3", 
                box=box.ROUNDED,
                width=100
            ))
        else:
            # Try to render as markdown for assistant messages
            try:
                md = Markdown(content)
                self.console.print(Panel(
                    md, 
                    title="Assistant", 
                    border_style="orange3", 
                    box=box.ROUNDED,
                    width=100
                ))
            except Exception:
                # Fallback to plain text if markdown parsing fails
                self.console.print(Panel(
                    content, 
                    title="Assistant", 
                    border_style="orange3", 
                    box=box.ROUNDED,
                    width=100
                ))