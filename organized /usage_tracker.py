import os

class UsageTracker:
    def __init__(self):
        # Token prices (per 1000 tokens)
        self.prices = {
            "gpt-4": {
                "input": 0.03,    # $0.03 per 1K tokens
                "output": 0.06    # $0.06 per 1K tokens
            },
            "gpt-4-0125-preview": {  # GPT-4 Turbo
                "input": 0.01,    # $0.01 per 1K tokens
                "output": 0.03    # $0.03 per 1K tokens
            },
            "gpt-4-1106-preview": {  # GPT-4 Turbo (legacy)
                "input": 0.01,    # $0.01 per 1K tokens
                "output": 0.03    # $0.03 per 1K tokens
            },
            "gpt-3.5-turbo-0125": {
                "input": 0.0005,  # $0.0005 per 1K tokens
                "output": 0.0015  # $0.0015 per 1K tokens
            },
            "gpt-3.5-turbo-1106": {  # legacy
                "input": 0.001,   # $0.001 per 1K tokens
                "output": 0.002   # $0.002 per 1K tokens
            },
            "gpt-4o-mini": {      # New affordable model
                "input": 0.00015,  # $0.150 per 1M tokens = $0.00015 per 1K tokens
                "output": 0.0006   # $0.600 per 1M tokens = $0.0006 per 1K tokens
            }
        }
        
        self.reset()
    
    def reset(self):
        """Reset all counters"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        self.calls_count = 0
    
    def add_usage(self, response, model="gpt-4"):
        """
        Add usage from API response
        
        Args:
            response: OpenAI API response
            model: Model name used
        """
        self.calls_count += 1
        
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        # Calculate cost
        if model in self.prices:
            cost = (
                (input_tokens * self.prices[model]["input"] / 1000) +
                (output_tokens * self.prices[model]["output"] / 1000)
            )
            self.total_cost += cost
    
    def get_summary(self):
        """Returns usage summary"""
        return {
            "total_calls": self.calls_count,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": round(self.total_cost, 4)
        }
    
    def print_summary(self):
        """Prints detailed usage summary and writes to file"""
        summary = self.get_summary()
        summary_text = "\nUsage Summary:\n"
        summary_text += f"Total API calls: {summary['total_calls']}\n"
        summary_text += f"Total input tokens: {summary['input_tokens']}\n"
        summary_text += f"Total output tokens: {summary['output_tokens']}\n"
        summary_text += f"Total tokens: {summary['total_tokens']}\n"
        summary_text += f"Total cost: ${summary['total_cost']:.4f}\n"
        
        # Print to console
        print(summary_text)
        
        # Write to file
        result_dir = "./results/"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        # Get the current result file name pattern (e.g., "00000-00070V2.csv")
        files = [f for f in os.listdir(result_dir) if f.endswith('V2.csv')]
        if files:
            base_name = files[0].replace('.csv', '')
            summary_file = os.path.join(result_dir, f"{base_name}_summary.txt")
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary_text)

# Create single instance for use across all modules
tracker = UsageTracker() 