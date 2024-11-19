import re
import ast

def parse_function(input_str):
    # Regular expression to capture the function name and arguments
    pattern = r'(\w+)(?:\((.*)\))?'
    
    match = re.match(pattern, input_str)
    if match:
        function_name = match.group(1)  # Extract the function name
        raw_arguments = match.group(2)  # Extract the arguments as a string        
        # If there are arguments, attempt to parse them
        arguments = []
        if raw_arguments:
            try:
                # Use ast.literal_eval to safely evaluate and convert the arguments
                parsed_args = ast.literal_eval(f'({raw_arguments})')  # Wrap in tuple parentheses
                # Ensure it's always treated as a tuple even with a single argument
                arguments = list(parsed_args) if isinstance(parsed_args, tuple) else [parsed_args]
            except (ValueError, SyntaxError):
                # In case of failure to parse, return the raw argument string
                arguments = [raw_arguments.strip()]
        

        return {
            'function_name': function_name,
            'arguments': arguments
        }
    else:
        return None
    
