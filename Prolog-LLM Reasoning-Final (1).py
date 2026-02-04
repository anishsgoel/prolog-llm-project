#!/usr/bin/env python
# coding: utf-8

# In[10]:


get_ipython().system('brew install ollama')


# In[12]:


get_ipython().system('ollama pull gpt-oss:20b')


# In[ ]:





# In[20]:


get_ipython().system('pip install ollama')


# In[19]:


get_ipython().system('python3.9 -m pip install --upgrade pip')


# In[22]:


import pkg_resources
installed_packages = [pkg.key for pkg in pkg_resources.working_set]
print("ollama installed:", "ollama" in installed_packages)


# In[23]:


get_ipython().system('pip show ollama')


# In[24]:


import sys
print("Jupyter is using Python from:", sys.executable)
print("Python version:", sys.version)
print("Python path entries:")
for path in sys.path[:5]:
    print(f"  - {path}")


# In[25]:


import sys
# Install ollama in the Anaconda ml environment that Jupyter is using
get_ipython().system('{sys.executable} -m pip install ollama')


# In[26]:





# In[9]:


# Ollama Python Integration for Jupyter Notebook

## Installation
# !pip install ollama requests json

## Basic Setup and Usage

import ollama
import json
import requests
from typing import Dict, List, Optional

class OllamaClient:
    """
    A wrapper class for interacting with Ollama models in Jupyter
    """
    
    def __init__(self, model_name: str = "gpt-oss:20b", base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama client
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: URL where Ollama server is running
        """
        self.model_name = model_name
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)
        
    def query(self, 
              prompt: str, 
              system_prompt: Optional[str] = None,
              temperature: float = 0.7,
              max_tokens: int = 2000,
              stream: bool = False) -> str:
        """
        Query the Ollama model
        
        Args:
            prompt: User prompt
            system_prompt: System instructions for the model
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            stream: Whether to stream the response
            
        Returns:
            Model response as string
        """
        messages = []
        
        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })
            
        messages.append({
            'role': 'user',
            'content': prompt
        })
        
        response = self.client.chat(
            model=self.model_name,
            messages=messages,
            options={
                'temperature': temperature,
                'num_predict': max_tokens,
            },
            stream=stream
        )
        
        if stream:
            full_response = ""
            for chunk in response:
                chunk_content = chunk['message']['content']
                full_response += chunk_content
                print(chunk_content, end='', flush=True)
            return full_response
        else:
            return response['message']['content']
    
    def query_with_context(self, 
                          prompt: str,
                          context: str,
                          temperature: float = 0.7) -> str:
        """
        Query with additional context
        
        Args:
            prompt: User question
            context: Additional context to consider
            temperature: Sampling temperature
            
        Returns:
            Model response
        """
        full_prompt = f"""Context: {context}

Question: {prompt}

Please answer based on the provided context."""
        
        return self.query(full_prompt, temperature=temperature)
    
    def batch_query(self, 
                   prompts: List[str],
                   system_prompt: Optional[str] = None) -> List[str]:
        """
        Process multiple prompts
        
        Args:
            prompts: List of prompts to process
            system_prompt: Optional system prompt for all queries
            
        Returns:
            List of responses
        """
        responses = []
        for prompt in prompts:
            response = self.query(prompt, system_prompt=system_prompt)
            responses.append(response)
        return responses
    
    def tune_response(self,
                     prompt: str,
                     desired_format: str = "json",
                     examples: Optional[List[Dict]] = None) -> str:
        """
        Get a response in a specific format with few-shot examples
        
        Args:
            prompt: The user prompt
            desired_format: Format instruction (json, list, paragraph, etc.)
            examples: Optional few-shot examples
            
        Returns:
            Formatted response
        """
        system_prompt = f"You must respond in {desired_format} format."
        
        if examples:
            system_prompt += "\n\nHere are some examples:\n"
            for i, example in enumerate(examples, 1):
                system_prompt += f"\nExample {i}:\n"
                system_prompt += f"Input: {example.get('input', '')}\n"
                system_prompt += f"Output: {example.get('output', '')}\n"
        
        if desired_format == "json":
            system_prompt += "\nEnsure your response is valid JSON."
            
        return self.query(prompt, system_prompt=system_prompt, temperature=0.3)
    
    def check_model_status(self) -> Dict:
        """
        Check if the model is loaded and ready
        
        Returns:
            Status information dictionary
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json()
                return {
                    'status': 'connected',
                    'available_models': [m['name'] for m in models.get('models', [])],
                    'current_model': self.model_name
                }
            else:
                return {'status': 'error', 'message': 'Failed to get model list'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def custom_instruction_chain(self,
                                initial_prompt: str,
                                refinement_steps: List[str]) -> str:
        """
        Process a prompt through multiple refinement steps
        
        Args:
            initial_prompt: Starting prompt
            refinement_steps: List of refinement instructions
            
        Returns:
            Final refined response
        """
        current_output = self.query(initial_prompt)
        
        for step in refinement_steps:
            refinement_prompt = f"""Previous output: {current_output}

Refinement instruction: {step}

Please refine the previous output according to the instruction above."""
            current_output = self.query(refinement_prompt)
            
        return current_output

## Usage Examples

# Initialize the client
client = OllamaClient(model_name="gpt-oss:20b")

# Check model status
print("Model Status:")
print(client.check_model_status())

# Basic query
response = client.query("Explain quantum computing in simple terms")
print(response)

# Query with system prompt
response = client.query(
    prompt="Write a haiku about programming",
    system_prompt="You are a creative poet who loves technology"
)
print(response)

# Query with specific temperature
response = client.query(
    prompt="Generate creative names for a tech startup",
    temperature=0.9  # Higher temperature for more creativity
)
print(response)

# Streaming response
print("Streaming response:")
response = client.query(
    prompt="Tell me a short story",
    stream=True
)

# Query with context
context = """
The company reported Q3 revenue of $10.5 billion, 
up 15% from last year. Operating margins improved to 22%.
"""
response = client.query_with_context(
    prompt="What was the revenue growth?",
    context=context
)
print(response)

# Get JSON formatted response
response = client.tune_response(
    prompt="List 3 benefits of exercise",
    desired_format="json"
)
print(response)

# Few-shot learning example
examples = [
    {
        "input": "The sky is blue",
        "output": {"sentiment": "neutral", "topic": "nature"}
    },
    {
        "input": "I love this product!",
        "output": {"sentiment": "positive", "topic": "product"}
    }
]

response = client.tune_response(
    prompt="This movie was terrible",
    desired_format="json",
    examples=examples
)
print(response)

# Refinement chain
initial = "Write a product description for wireless headphones"
refinements = [
    "Make it more concise, under 50 words",
    "Add emphasis on battery life",
    "Make it sound more premium and sophisticated"
]

final_response = client.custom_instruction_chain(initial, refinements)
print(final_response)

## Advanced Response Tuning

class ResponseTuner:
    """
    Advanced response tuning capabilities
    """
    
    def __init__(self, client: OllamaClient):
        self.client = client
    
    def enforce_structure(self, 
                         prompt: str,
                         structure: Dict) -> str:
        """
        Enforce a specific response structure
        
        Args:
            prompt: User prompt
            structure: Dictionary defining the expected structure
            
        Returns:
            Structured response
        """
        structure_prompt = f"""You must respond with exactly this structure:
{json.dumps(structure, indent=2)}

Fill in the values appropriately based on the prompt."""
        
        return self.client.query(prompt, system_prompt=structure_prompt)
    
    def iterative_improvement(self,
                            prompt: str,
                            criteria: List[str],
                            max_iterations: int = 3) -> str:
        """
        Iteratively improve a response based on criteria
        
        Args:
            prompt: Initial prompt
            criteria: List of improvement criteria
            max_iterations: Maximum refinement iterations
            
        Returns:
            Improved response
        """
        current = self.client.query(prompt)
        
        for i in range(max_iterations):
            critique_prompt = f"""Response: {current}

Evaluate this response against these criteria:
{chr(10).join(f'- {c}' for c in criteria)}

Provide an improved version that better meets all criteria."""
            
            current = self.client.query(critique_prompt, temperature=0.5)
            
        return current
    
    def style_transfer(self,
                       text: str,
                       target_style: str) -> str:
        """
        Transform text to match a target style
        
        Args:
            text: Original text
            target_style: Description of target style
            
        Returns:
            Styled text
        """
        prompt = f"""Original text: {text}

Rewrite this text in the following style: {target_style}

Maintain the core meaning but completely transform the style."""
        
        return self.client.query(prompt, temperature=0.6)

# Example usage of ResponseTuner
tuner = ResponseTuner(client)

# Enforce specific structure
structure = {
    "summary": "...",
    "key_points": ["...", "..."],
    "recommendation": "..."
}

response = tuner.enforce_structure(
    prompt="Analyze the benefits of remote work",
    structure=structure
)
print(response)

# Iterative improvement
response = tuner.iterative_improvement(
    prompt="Write a marketing email for a new product",
    criteria=[
        "Under 100 words",
        "Include a clear call to action",
        "Professional but friendly tone",
        "Highlight key benefits"
    ]
)
print(response)

# Style transfer
original = "The quarterly results exceeded expectations with significant growth."
styled = tuner.style_transfer(
    text=original,
    target_style="casual conversation with a friend"
)
print(styled)


# In[11]:


get_ipython().system('ollama --version')
get_ipython().system('ollama ps')


# In[17]:


get_ipython().system('ollama run llama3.2:3b "ok"')


# In[18]:


get_ipython().system('pkill ollama || true')
get_ipython().system('ollama serve')


# In[16]:


get_ipython().system('pip show ollama')
get_ipython().system('pip install -U ollama')


# In[13]:


get_ipython().system('brew services list | grep ollama')
get_ipython().system('brew services restart ollama')


# In[35]:


# Initialize the client
import ollama

client = OllamaClient(model_name="gpt-oss:20b")

# Check model status
print("Model Status:")
print(client.check_model_status())

# Basic query
response = client.query("what is 2+2")
print(response)


# In[ ]:


"""
SLD Resolution Implementation using Ollama for Non-Monotonic Reasoning
This system implements backward chaining with SLD resolution for Prolog-like queries
"""

import ollama
import json
import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import deque

@dataclass
class Rule:
    """Represents a Prolog rule: head :- body"""
    head: str
    body: List[str]
    
    def __str__(self):
        if not self.body:
            return f"{self.head}."
        return f"{self.head} :- {', '.join(self.body)}."

@dataclass
class Substitution:
    """Represents variable substitutions (unification)"""
    mappings: Dict[str, str]
    
    def apply(self, term: str) -> str:
        """Apply substitution to a term"""
        result = term
        for var, val in self.mappings.items():
            result = result.replace(var, val)
        return result
    
    def compose(self, other: 'Substitution') -> 'Substitution':
        """Compose two substitutions"""
        new_mappings = {k: other.apply(v) for k, v in self.mappings.items()}
        new_mappings.update(other.mappings)
        return Substitution(new_mappings)

class SLDResolver:
    """
    SLD Resolution engine using Ollama for logical operations
    """
    
    def __init__(self, model_name: str = "gpt-oss:20b"):
        """
        Initialize SLD Resolver with Ollama
        
        Args:
            model_name: Name of the Ollama model to use
        """
        self.client = ollama.Client()
        self.model_name = model_name
        self.knowledge_base = []
        self.trace = []  # For debugging
        
    def add_rule(self, rule: Rule):
        """Add a rule to the knowledge base"""
        self.knowledge_base.append(rule)
    
    def parse_prolog_syntax(self, text: str) -> List[Rule]:
        """
        Use Ollama to parse Prolog syntax into rules
        
        Args:
            text: Prolog program text
            
        Returns:
            List of parsed rules
        """
        prompt = f"""Parse the following Prolog rules into a structured format.
For each rule, identify the head (conclusion) and body (conditions).
Facts (rules with no body) should have an empty body list.

Prolog rules:
{text}

Return the result as JSON with this format:
[
  {{"head": "predicate(args)", "body": ["condition1", "condition2"]}},
  ...
]

Important:
- Keep variable names (uppercase) and constants (lowercase) as-is
- For facts, use empty body array []
- Preserve all parentheses and arguments"""
        
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {'role': 'system', 'content': 'You are a Prolog parser. Return only valid JSON.'},
                {'role': 'user', 'content': prompt}
            ],
            options={'temperature': 0.1}
        )
        
        try:
            # Extract JSON from response
            json_str = response['message']['content']
            # Clean up the response to get just the JSON
            json_match = re.search(r'\[.*\]', json_str, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            
            parsed = json.loads(json_str)
            rules = [Rule(r['head'], r['body']) for r in parsed]
            return rules
        except Exception as e:
            print(f"Parsing error: {e}")
            print(f"Response: {response['message']['content']}")
            return []
    
    def unify(self, term1: str, term2: str) -> Optional[Substitution]:
        """
        Use Ollama to perform unification between two terms
        
        Args:
            term1: First term (e.g., "connected(X, Y)")
            term2: Second term (e.g., "connected(a, b)")
            
        Returns:
            Substitution if unification succeeds, None otherwise
        """
        prompt = f"""Perform Prolog unification between these two terms:
Term 1: {term1}
Term 2: {term2}

Rules for unification:
1. Variables (uppercase) can be unified with any term
2. Constants (lowercase) unify only with identical constants
3. Compound terms unify if predicates match and arguments unify pairwise
4. A variable cannot unify with a term containing itself (occurs check)

If unification succeeds, return JSON with variable substitutions:
{{"success": true, "substitutions": {{"X": "value", "Y": "value2"}}}}

If unification fails, return:
{{"success": false, "reason": "explanation"}}

Return ONLY the JSON, no other text."""
        
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {'role': 'system', 'content': 'You are a Prolog unification engine. Return only valid JSON.'},
                {'role': 'user', 'content': prompt}
            ],
            options={'temperature': 0.0}
        )
        
        try:
            json_str = response['message']['content']
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            
            result = json.loads(json_str)
            if result.get('success'):
                return Substitution(result.get('substitutions', {}))
            return None
        except Exception as e:
            print(f"Unification error: {e}")
            print(f"Response: {response['message']['content']}")
            return None
    
    def find_matching_rule(self, goal: str) -> List[Tuple[Rule, Substitution]]:
        """
        Find rules whose head can unify with the goal
        
        Args:
            goal: The goal to match (e.g., "connected(a, b)")
            
        Returns:
            List of (rule, substitution) pairs
        """
        prompt = f"""Given the goal: {goal}

And these rules from the knowledge base:
{chr(10).join(str(rule) for rule in self.knowledge_base)}

Find all rules whose head can unify with the goal.
For each matching rule:
1. Perform unification between the goal and the rule head
2. Apply the substitution to the rule body

Return JSON with this format:
[
  {{
    "rule_index": 0,
    "original_rule": "head :- body1, body2",
    "substitution": {{"X": "a", "Y": "b"}},
    "specialized_body": ["body1_specialized", "body2_specialized"]
  }}
]

If no rules match, return empty array []

Return ONLY valid JSON."""
        
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {'role': 'system', 'content': 'You are a Prolog rule matcher. Return only valid JSON.'},
                {'role': 'user', 'content': prompt}
            ],
            options={'temperature': 0.0}
        )
        
        try:
            json_str = response['message']['content']
            # Extract JSON array
            json_match = re.search(r'\[.*\]', json_str, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            
            matches = json.loads(json_str)
            results = []
            
            for match in matches:
                rule_idx = match['rule_index']
                if 0 <= rule_idx < len(self.knowledge_base):
                    rule = self.knowledge_base[rule_idx]
                    sub = Substitution(match.get('substitution', {}))
                    results.append((rule, sub))
            
            return results
        except Exception as e:
            print(f"Rule matching error: {e}")
            print(f"Response: {response['message']['content']}")
            return []
    
    def resolve(self, goals: List[str], max_depth: int = 20) -> Tuple[bool, List[Substitution]]:
        """
        Main SLD resolution algorithm
        
        Args:
            goals: List of goals to prove
            max_depth: Maximum resolution depth (prevent infinite loops)
            
        Returns:
            (success, list of substitutions that satisfy the goals)
        """
        # Use deque for goal stack (LIFO for depth-first search)
        stack = [(goals, Substitution({}), 0)]  # (remaining_goals, current_substitution, depth)
        solutions = []
        
        while stack and len(solutions) < 10:  # Limit solutions for efficiency
            current_goals, current_sub, depth = stack.pop()
            
            # Success: no more goals to prove
            if not current_goals:
                solutions.append(current_sub)
                continue
            
            # Depth limit reached
            if depth >= max_depth:
                continue
            
            # Take first goal (SLD resolution selects leftmost goal)
            first_goal = current_goals[0]
            remaining_goals = current_goals[1:]
            
            # Apply current substitution to the goal
            instantiated_goal = current_sub.apply(first_goal)
            
            # Find matching rules
            matches = self.find_matching_rule(instantiated_goal)
            
            if not matches:
                # No matching rules - this branch fails
                self.trace.append(f"No match for goal: {instantiated_goal}")
                continue
            
            # Try each matching rule (creates choice points)
            for rule, rule_sub in matches:
                # Compose substitutions
                new_sub = current_sub.compose(rule_sub)
                
                # Replace goal with rule body
                new_goals = [new_sub.apply(g) for g in rule.body]
                new_goals.extend([new_sub.apply(g) for g in remaining_goals])
                
                # Add to stack for exploration
                stack.append((new_goals, new_sub, depth + 1))
                
                self.trace.append(f"Applied rule: {rule} with substitution {rule_sub.mappings}")
        
        return (len(solutions) > 0, solutions)
    
    def query(self, goal_string: str) -> Dict:
        """
        High-level query interface
        
        Args:
            goal_string: Prolog-style query (e.g., "connected(X, Y), connected(Y, Z)")
            
        Returns:
            Dictionary with results and explanations
        """
        # Parse goals
        goals = [g.strip() for g in goal_string.split(',')]
        
        self.trace = []  # Reset trace
        success, solutions = self.resolve(goals, max_depth=20)
        
        return {
            'success': success,
            'query': goal_string,
            'solutions': [sub.mappings for sub in solutions],
            'trace': self.trace[:10]  # Limit trace output
        }
    
    def prove_with_explanation(self, goals: List[str]) -> str:
        """
        Use Ollama to generate natural language explanation of proof
        
        Args:
            goals: List of goals to prove
            
        Returns:
            Natural language explanation
        """
        result = self.query(', '.join(goals))
        
        prompt = f"""Explain the SLD resolution proof for the query: {', '.join(goals)}

Knowledge base:
{chr(10).join(str(rule) for rule in self.knowledge_base)}

Resolution result:
Success: {result['success']}
Solutions: {result['solutions']}
Trace: {result['trace']}

Provide a step-by-step explanation of how SLD resolution proved (or failed to prove) the goals.
Use clear, educational language suitable for a logic programming student."""
        
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {'role': 'system', 'content': 'You are a logic programming tutor explaining SLD resolution.'},
                {'role': 'user', 'content': prompt}
            ],
            options={'temperature': 0.3}
        )
        
        return response['message']['content']

# Example usage for Metro/Transport Network
def metro_example():
    """
    Example: Metro station connectivity using SLD resolution
    """
    resolver = SLDResolver(model_name="gpt-oss:20b")
    
    # Define metro network as Prolog rules
    metro_rules = """
    % Direct connections (facts)
    connected(union_square, times_square).
    connected(times_square, grand_central).
    connected(grand_central, bryant_park).
    connected(union_square, washington_square).
    connected(washington_square, west_4th).
    connected(west_4th, christopher_st).
    
    % Symmetric connection
    connection(X, Y) :- connected(X, Y).
    connection(X, Y) :- connected(Y, X).
    
    % Reachability (transitive closure)
    reachable(X, Y) :- connection(X, Y).
    reachable(X, Z) :- connection(X, Y), reachable(Y, Z).
    
    % Path finding
    path(X, Y, [X, Y]) :- connection(X, Y).
    path(X, Z, [X|Path]) :- connection(X, Y), path(Y, Z, Path).
    """
    
    # Parse and add rules to knowledge base
    rules = resolver.parse_prolog_syntax(metro_rules)
    for rule in rules:
        resolver.add_rule(rule)
        print(f"Added rule: {rule}")
    
    print("\n" + "="*60)
    print("METRO NETWORK SLD RESOLUTION EXAMPLES")
    print("="*60)
    
    # Test queries
    queries = [
        "connected(union_square, times_square)",  # Direct connection
        "reachable(union_square, bryant_park)",   # Multi-hop reachability
        "reachable(X, grand_central)",            # Find all stations connected to Grand Central
        "connection(times_square, union_square)",  # Symmetric connection test
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        result = resolver.query(query)
        
        if result['success']:
            print(f"✓ SUCCESS")
            if result['solutions']:
                print(f"Solutions found: {len(result['solutions'])}")
                for i, sol in enumerate(result['solutions'][:3], 1):
                    if sol:
                        print(f"  Solution {i}: {sol}")
                    else:
                        print(f"  Solution {i}: Query satisfied (no variables)")
        else:
            print(f"✗ FAILED - No proof found")
        
        if result['trace']:
            print(f"\nResolution trace (first 3 steps):")
            for step in result['trace'][:3]:
                print(f"  - {step}")
    
    # Get natural language explanation for a complex query
    print("\n" + "="*60)
    print("NATURAL LANGUAGE PROOF EXPLANATION")
    print("="*60)
    explanation = resolver.prove_with_explanation(["reachable(union_square, bryant_park)"])
    print(explanation)
    
    return resolver

# Non-monotonic reasoning example
def non_monotonic_example():
    """
    Example: Non-monotonic reasoning with defaults and exceptions
    """
    resolver = SLDResolver(model_name="gpt-oss:20b")
    
    # Define rules with defaults and exceptions
    rules = """
    % Default: Birds can fly
    can_fly(X) :- bird(X), not(abnormal_bird(X)).
    
    % Facts about birds
    bird(robin).
    bird(penguin).
    bird(ostrich).
    bird(eagle).
    
    % Exceptions
    abnormal_bird(penguin).
    abnormal_bird(ostrich).
    
    % Specific knowledge overrides defaults
    can_swim(X) :- penguin(X).
    penguin(penguin).
    
    % Categories
    predator(eagle).
    predator(X) :- hunts(X, Y).
    """
    
    parsed_rules = resolver.parse_prolog_syntax(rules)
    for rule in parsed_rules:
        resolver.add_rule(rule)
    
    print("\n" + "="*60)
    print("NON-MONOTONIC REASONING EXAMPLE")
    print("="*60)
    
    # Test non-monotonic queries
    queries = [
        "can_fly(robin)",      # Default case
        "can_fly(penguin)",    # Exception case
        "can_swim(penguin)",   # Specific knowledge
        "predator(eagle)",     # Direct fact
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = resolver.query(query)
        print(f"Result: {'✓ True' if result['success'] else '✗ False'}")
        if result['solutions']:
            print(f"Bindings: {result['solutions']}")

# Run examples
if __name__ == "__main__":
    print("SLD Resolution System with Ollama")
    print("=" * 60)
    
    # Test if Ollama is accessible
    try:
        client = ollama.Client()
        models = client.list()
        print("Available Ollama models:", [m['name'] for m in models.get('models', [])])
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Make sure Ollama is running (ollama serve)")
    
    # Run metro example
    print("\nRunning Metro Network Example...")
    metro_resolver = metro_example()
    
    # Run non-monotonic reasoning example
    print("\nRunning Non-Monotonic Reasoning Example...")
    non_monotonic_example()


# In[36]:


"""
Minimal test for Ollama's logic reasoning capabilities
Test incrementally before building full SLD resolution
"""

import ollama
import json
import re

class BasicLogicTest:
    def __init__(self, model_name="gpt-oss:20b"):
        self.client = ollama.Client()
        self.model = model_name
        print(f"Using model: {model_name}")
    
    def test_1_simple_unification(self):
        """Test 1: Can the model unify two simple terms?"""
        print("\n" + "="*50)
        print("TEST 1: Simple Unification")
        print("="*50)
        
        test_cases = [
            ("likes(X, pizza)", "likes(john, pizza)"),
            ("connected(A, B)", "connected(station1, station2)"),
            ("path(X, Y)", "path(X, paris)"),
        ]
        
        for term1, term2 in test_cases:
            prompt = f"""Unify these two Prolog terms:
            Term 1: {term1}
            Term 2: {term2}

            If they unify, what are the variable bindings?
            Reply with ONLY a JSON object like this:
            {{"unified": true, "bindings": {{"X": "value"}}}}
            or
            {{"unified": false}}"""

            response = self.client.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.0, 'num_predict': 100}
            )
            
            result = response['message']['content']
            print(f"\n{term1} ≟ {term2}")
            print(f"Result: {result}")
    
    def test_2_rule_matching(self):
        """Test 2: Can the model match a goal to a rule head?"""
        print("\n" + "="*50)
        print("TEST 2: Rule Matching")
        print("="*50)
        
        prompt = """Given these Prolog rules:
1. connected(a, b).
2. connected(b, c).
3. reachable(X, Y) :- connected(X, Y).
4. reachable(X, Z) :- connected(X, Y), reachable(Y, Z).

Which rule heads can unify with goal: reachable(a, c)

Reply with rule numbers that match, as JSON:
{"matches": [3, 4]}"""
        
        response = self.client.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.0, 'num_predict': 100}
        )
        
        print(f"Goal: reachable(a, c)")
        print(f"Matching rules: {response['message']['content']}")
    
    def test_3_simple_substitution(self):
        """Test 3: Can the model apply substitutions?"""
        print("\n" + "="*50)
        print("TEST 3: Apply Substitution")
        print("="*50)
        
        prompt = """Apply the substitution {X -> john, Y -> mary} to these terms:
1. likes(X, Y)
2. knows(X, bob)
3. friends(Y, X)

Reply with the substituted terms as JSON:
{"results": ["likes(john, mary)", "knows(john, bob)", "friends(mary, john)"]}"""
        
        response = self.client.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.0, 'num_predict': 150}
        )
        
        print(f"Substitution: {{X -> john, Y -> mary}}")
        print(f"Results: {response['message']['content']}")
    
    def test_4_single_resolution_step(self):
        """Test 4: Can the model perform ONE SLD resolution step?"""
        print("\n" + "="*50)
        print("TEST 4: Single SLD Resolution Step")
        print("="*50)
        
        prompt = """Perform ONE SLD resolution step:

Goal: connected(a, c)

Rules:
1. connected(a, b).
2. connected(b, c).
3. connected(X, Z) :- connected(X, Y), connected(Y, Z).

Find a rule that unifies with the goal.
Apply the unification.
Return the new subgoals.

Reply as JSON:
{"rule_used": 3, "substitution": {"X": "a", "Z": "c"}, "new_goals": ["connected(a, Y)", "connected(Y, c)"]}"""
        
        response = self.client.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.0, 'num_predict': 200}
        )
        
        print(f"Original goal: connected(a, c)")
        print(f"Resolution step: {response['message']['content']}")
    
    def run_all_tests(self):
        """Run all tests sequentially"""
        tests = [
            self.test_1_simple_unification,
            self.test_2_rule_matching,
            self.test_3_simple_substitution,
            self.test_4_single_resolution_step
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"Error in {test.__name__}: {e}")
                print("Continuing to next test...")

# Minimal working example
def quick_test():
    """Absolute minimal test - just check if model can handle logic"""
    client = ollama.Client()
    
    prompt = """Is this valid Prolog unification?
    Term1: likes(X, pizza)
    Term2: likes(john, pizza)
    
    Answer with just YES or NO and the binding:
    YES, X=john"""
    
    try:
        response = client.chat(
            model="gpt-oss:20b",
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.0, 'num_predict': 50}
        )
        print("Quick test response:", response['message']['content'])
        return True
    except Exception as e:
        print(f"Quick test failed: {e}")
        return False

# Simple SLD step-by-step implementation
class SimpleSLD:
    """Minimal SLD implementation - one step at a time"""
    
    def __init__(self, model="gpt-oss:20b"):
        self.client = ollama.Client()
        self.model = model
    
    def resolve_one_step(self, goal, rules):
        """Perform just ONE resolution step - no recursion"""
        
        prompt = f"""SLD Resolution - ONE STEP ONLY:
        
Current Goal: {goal}

Available Rules:
{rules}

Task:
1. Find a rule whose head unifies with the goal
2. If found, return the rule number and new subgoals
3. If not found, return "NO"

Output format:
MATCH: rule_number, new_subgoals
or
NO

Example: MATCH: 3, [connected(a,Y), connected(Y,c)]"""
        
        response = self.client.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.0, 'num_predict': 100}
        )
        
        return response['message']['content']

# Run the tests
if __name__ == "__main__":
    print("TESTING OLLAMA LOGIC CAPABILITIES")
    print("="*50)
    
    # First, run the quick sanity check
    print("\nRunning quick sanity check...")
    if quick_test():
        print("✓ Model can handle basic logic")
        
        # Now run systematic tests
        print("\nRunning systematic tests...")
        tester = BasicLogicTest(model_name="gpt-oss:20b")
        tester.run_all_tests()
        
        # Test simple SLD
        print("\n" + "="*50)
        print("SIMPLE SLD TEST")
        print("="*50)
        sld = SimpleSLD()
        
        rules = """
1. connected(a, b).
2. connected(b, c).
3. reachable(X, Y) :- connected(X, Y).
4. reachable(X, Z) :- connected(X, Y), reachable(Y, Z).
"""
        
        result = sld.resolve_one_step("reachable(a, b)", rules)
        print(f"Goal: reachable(a, b)")
        print(f"Result: {result}")
        
    else:
        print("✗ Model failed sanity check - check if Ollama is running")

tester = BasicLogicTest(model_name="gpt-oss:20b")
tester.run_all_tests()
# In[35]:


tester = BasicLogicTest(model_name="gpt-oss:20b")
tester.run_all_tests()


# In[37]:


"""
Minimal test for Ollama's logic reasoning capabilities
Test incrementally before building full SLD resolution
"""

import ollama
import json
import re

class BasicLogicTest:
    def __init__(self, model_name="gpt-oss:20b"):
        self.client = ollama.Client()
        self.model = model_name
        print(f"Using model: {model_name}")
    
    def test_1_simple_unification(self):
        """Test 1: Can the model unify two simple terms?"""
        print("\n" + "="*50)
        print("TEST 1: Simple Unification")
        print("="*50)
        
        test_cases = [
            ("likes(X, pizza)", "likes(john, pizza)"),
            ("connected(A, B)", "connected(station1, station2)"),
            ("path(X, Y)", "path(X, paris)"),
        ]
        
        for term1, term2 in test_cases:
            # Try simpler prompt first
            prompt = f"""Unify these Prolog terms:
{term1}
{term2}

What variable bindings make them equal?
Answer in this format: X=john, Y=mary
If they cannot unify, write: CANNOT UNIFY"""
            
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': 0.0, 'num_predict': 50}
                )
                
                result = response['message']['content']
                # Debug: show raw response
                print(f"\n{term1} ≟ {term2}")
                print(f"Raw result: '{result}'")
                print(f"Result length: {len(result)} chars")
                
                # If empty, try with different prompt style
                if not result or len(result.strip()) == 0:
                    print("Empty response, trying alternative prompt...")
                    simple_prompt = f"In Prolog, {term1} unifies with {term2}. What is X? Just write the answer."
                    response2 = self.client.chat(
                        model=self.model,
                        messages=[{'role': 'user', 'content': simple_prompt}],
                        options={'temperature': 0.0, 'num_predict': 30}
                    )
                    print(f"Alternative result: '{response2['message']['content']}'")
                    
            except Exception as e:
                print(f"Error: {e}")
    
    def test_2_rule_matching(self):
        """Test 2: Can the model match a goal to a rule head?"""
        print("\n" + "="*50)
        print("TEST 2: Rule Matching")
        print("="*50)
        
        # Simpler, more direct prompt
        prompt = """Look at these Prolog rules:
Rule 1: connected(a, b).
Rule 2: connected(b, c).
Rule 3: reachable(X, Y) :- connected(X, Y).
Rule 4: reachable(X, Z) :- connected(X, Y), reachable(Y, Z).

Question: Which rules have heads that match the pattern reachable(a, c)?

Answer: Just list the rule numbers that match, like: 3, 4"""
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.0, 'num_predict': 30}
            )
            
            result = response['message']['content']
            print(f"Goal: reachable(a, c)")
            print(f"Raw response: '{result}'")
            print(f"Response length: {len(result)} chars")
            
            # If empty, try even simpler
            if not result or len(result.strip()) == 0:
                print("\nEmpty response, trying simpler question...")
                simple_prompt = "Does the Prolog term reachable(X,Y) match reachable(a,c)? Answer YES or NO."
                response2 = self.client.chat(
                    model=self.model,
                    messages=[{'role': 'user', 'content': simple_prompt}],
                    options={'temperature': 0.0, 'num_predict': 10}
                )
                print(f"Simple test result: '{response2['message']['content']}'")
                
        except Exception as e:
            print(f"Error in test_2: {e}")
    
    def test_3_simple_substitution(self):
        """Test 3: Can the model apply substitutions?"""
        print("\n" + "="*50)
        print("TEST 3: Apply Substitution")
        print("="*50)
        
        prompt = """Apply the substitution {X -> john, Y -> mary} to these terms:
1. likes(X, Y)
2. knows(X, bob)
3. friends(Y, X)

Reply with the substituted terms as JSON:
{"results": ["likes(john, mary)", "knows(john, bob)", "friends(mary, john)"]}"""
        
        response = self.client.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.0, 'num_predict': 150}
        )
        
        print(f"Substitution: {{X -> john, Y -> mary}}")
        print(f"Results: {response['message']['content']}")
    
    def test_4_single_resolution_step(self):
        """Test 4: Can the model perform ONE SLD resolution step?"""
        print("\n" + "="*50)
        print("TEST 4: Single SLD Resolution Step")
        print("="*50)
        
        prompt = """Perform ONE SLD resolution step:

Goal: connected(a, c)

Rules:
1. connected(a, b).
2. connected(b, c).
3. connected(X, Z) :- connected(X, Y), connected(Y, Z).

Find a rule that unifies with the goal.
Apply the unification.
Return the new subgoals.

Reply as JSON:
{"rule_used": 3, "substitution": {"X": "a", "Z": "c"}, "new_goals": ["connected(a, Y)", "connected(Y, c)"]}"""
        
        response = self.client.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.0, 'num_predict': 200}
        )
        
        print(f"Original goal: connected(a, c)")
        print(f"Resolution step: {response['message']['content']}")
    
    def run_all_tests(self):
        """Run all tests sequentially"""
        tests = [
            self.test_1_simple_unification,
            self.test_2_rule_matching,
            self.test_3_simple_substitution,
            self.test_4_single_resolution_step
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"Error in {test.__name__}: {e}")
                print("Continuing to next test...")

# Minimal working example
def quick_test():
    """Absolute minimal test - just check if model can handle logic"""
    client = ollama.Client()
    
    prompt = """Is this valid Prolog unification?
    Term1: likes(X, pizza)
    Term2: likes(john, pizza)
    
    Answer with just YES or NO and the binding:
    YES, X=john"""
    
    try:
        response = client.chat(
            model="gpt-oss:20b",
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.0, 'num_predict': 50}
        )
        print("Quick test response:", response['message']['content'])
        return True
    except Exception as e:
        print(f"Quick test failed: {e}")
        return False

# Simple SLD step-by-step implementation
class SimpleSLD:
    """Minimal SLD implementation - one step at a time"""
    
    def __init__(self, model="gpt-oss:20b"):
        self.client = ollama.Client()
        self.model = model
    
    def resolve_one_step(self, goal, rules):
        """Perform just ONE resolution step - no recursion"""
        
        prompt = f"""SLD Resolution - ONE STEP ONLY:
        
Current Goal: {goal}

Available Rules:
{rules}

Task:
1. Find a rule whose head unifies with the goal
2. If found, return the rule number and new subgoals
3. If not found, return "NO"

Output format:
MATCH: rule_number, new_subgoals
or
NO

Example: MATCH: 3, [connected(a,Y), connected(Y,c)]"""
        
        response = self.client.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.0, 'num_predict': 100}
        )
        
        return response['message']['content']

# Run the tests
if __name__ == "__main__":
    print("TESTING OLLAMA LOGIC CAPABILITIES")
    print("="*50)
    
    # Add debug test first
    print("\nDEBUG: Testing basic model response...")
    client = ollama.Client()
    try:
        test_response = client.chat(
            model="gpt-oss:20b",
            messages=[{'role': 'user', 'content': 'Say "hello"'}],
            options={'temperature': 0.0, 'num_predict': 10}
        )
        print(f"Basic test response: '{test_response['message']['content']}'")
        print(f"Response type: {type(test_response['message']['content'])}")
    except Exception as e:
        print(f"Basic test failed: {e}")
    
    # First, run the quick sanity check
    print("\nRunning quick sanity check...")
    if quick_test():
        print("✓ Model can handle basic logic")
        
        # Now run systematic tests
        print("\nRunning systematic tests...")
        tester = BasicLogicTest(model_name="gpt-oss:20b")
        tester.run_all_tests()
        
        # Test simple SLD
        print("\n" + "="*50)
        print("SIMPLE SLD TEST")
        print("="*50)
        sld = SimpleSLD()
        
        rules = """
1. connected(a, b).
2. connected(b, c).
3. reachable(X, Y) :- connected(X, Y).
4. reachable(X, Z) :- connected(X, Y), reachable(Y, Z).
"""
        
        result = sld.resolve_one_step("reachable(a, b)", rules)
        print(f"Goal: reachable(a, b)")
        print(f"Result: {result}")
        
    else:
        print("✗ Model failed sanity check - check if Ollama is running")


# In[39]:


"""
Minimal Ollama Client with strict output control
"""

import ollama
import requests

class OllamaClient:
    def __init__(self, model_name="gpt-oss:20b"):
        self.model_name = model_name
        self.client = ollama.Client()
        
    def query(self, prompt, max_tokens=50):
        """Query with minimal response"""
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': 0.0,      # Deterministic
                    'num_predict': max_tokens,  # Limit output length
                    'top_k': 1,              # Only most likely token
                    'top_p': 0.1,            # Narrow probability range
                    'repeat_penalty': 1.0,    # No repeat penalty
                    'seed': 42,              # Fixed seed for consistency
                }
            )
            return response['message']['content']
        except Exception as e:
            return f"Error: {e}"
    
    def check_model_status(self):
        """Check if model is available"""
        try:
            # Check service
            service_check = requests.get("http://localhost:11434/api/tags", timeout=2)
            if service_check.status_code != 200:
                return {'status': 'error', 'message': 'Ollama not running'}
            
            # Check models
            models = self.client.list()
            model_list = [m['name'] for m in models.get('models', [])]
            
            if self.model_name in model_list:
                return {
                    'status': 'connected',
                    'available_models': model_list,
                    'current_model': self.model_name
                }
            else:
                return {
                    'status': 'error',
                    'message': f'Model {self.model_name} not found',
                    'available_models': model_list
                }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

# Test with minimal output
if __name__ == "__main__":
    # Initialize the client
    client = OllamaClient(model_name="gpt-oss:20b")
    
    # Check model status
    print("Model Status:")
    print(client.check_model_status())
    
    # Basic query - VERY LIMITED OUTPUT
    print("\nQuery: Explain quantum computing in simple terms")
    print("Response (max 50 tokens):")
    response = client.query("Explain quantum computing in simple terms", max_tokens=50)
    print(response)
    
    # Even shorter test
    print("\nQuery: What is 2+2?")
    print("Response (max 10 tokens):")
    response = client.query("What is 2+2?", max_tokens=10)
    print(response)
    
    # Logic test with minimal output
    print("\nQuery: In Prolog, does f(X) unify with f(a)? Answer yes or no.")
    print("Response (max 5 tokens):")
    response = client.query("In Prolog, does f(X) unify with f(a)? Answer yes or no.", max_tokens=5)
    print(response)


# In[40]:


"""
Simple Prolog logic tests without classes
Direct calls with parameters to control response
"""

import ollama

# Initialize client once
client = ollama.Client()
model_name = "gpt-oss:20b"

# Control parameters
OPTIONS = {
    'temperature': 0.0,  # Deterministic
    'num_predict': 50,   # Short responses
    'top_k': 1,          # Most likely token only
    'seed': 42,          # Fixed seed for consistency
    'stop': ['\n', '.']  # Stop at newline or period
}

def test_unification():
    """Test 1: Simple unification"""
    print("="*60)
    print("TEST 1: UNIFICATION")
    print("="*60)
    
    # Test case 1
    prompt1 = """Prolog unification:
Term1: likes(X, pizza)
Term2: likes(john, pizza)
What is X?
Answer: X="""
    
    response = client.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt1}],
        options=OPTIONS
    )
    print(f"Test: likes(X, pizza) = likes(john, pizza)")
    print(f"Response: {response['message']['content']}")
    
    # Test case 2
    prompt2 = """Prolog unification:
connected(A, B) = connected(station1, station2)
What are A and B?
Answer: A="""
    
    response = client.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt2}],
        options=OPTIONS
    )
    print(f"\nTest: connected(A, B) = connected(station1, station2)")
    print(f"Response: {response['message']['content']}")
    
    # Test case 3
    prompt3 = """Can these Prolog terms unify?
f(X, Y) and f(a, b)
Answer YES or NO:"""
    
    response = client.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt3}],
        options=OPTIONS
    )
    print(f"\nTest: Can f(X, Y) unify with f(a, b)?")
    print(f"Response: {response['message']['content']}")

def test_rule_matching():
    """Test 2: Rule matching"""
    print("\n" + "="*60)
    print("TEST 2: RULE MATCHING")
    print("="*60)
    
    prompt = """Prolog rules:
1. connected(a, b).
2. connected(b, c).
3. reachable(X, Y) :- connected(X, Y).

Which rule head matches goal: reachable(a, b)?
Answer with rule number:"""
    
    response = client.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt}],
        options=OPTIONS
    )
    print(f"Goal: reachable(a, b)")
    print(f"Response: {response['message']['content']}")

def test_substitution():
    """Test 3: Apply substitution"""
    print("\n" + "="*60)
    print("TEST 3: SUBSTITUTION")
    print("="*60)
    
    prompt = """Apply substitution X=john to term: likes(X, mary)
Result:"""
    
    response = client.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt}],
        options=OPTIONS
    )
    print(f"Substitute X=john in likes(X, mary)")
    print(f"Response: {response['message']['content']}")

def test_sld_step():
    """Test 4: One SLD resolution step"""
    print("\n" + "="*60)
    print("TEST 4: SLD RESOLUTION STEP")
    print("="*60)
    
    prompt = """SLD Resolution:
Goal: connected(a, c)
Rules:
1. connected(a, b).
2. connected(b, c).
3. connected(X, Z) :- connected(X, Y), connected(Y, Z).

Which rule matches the goal?
Answer: Rule"""
    
    response = client.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt}],
        options=OPTIONS
    )
    print(f"Find rule for goal: connected(a, c)")
    print(f"Response: {response['message']['content']}")
    
    # Follow-up: get new subgoals
    prompt2 = """If we use rule: connected(X, Z) :- connected(X, Y), connected(Y, Z)
With goal: connected(a, c)
What are the new subgoals?
Answer: connected(a,"""
    
    response2 = client.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt2}],
        options=OPTIONS
    )
    print(f"\nNew subgoals after resolution:")
    print(f"Response: {response2['message']['content']}")

def test_metro_example():
    """Test 5: Metro reachability"""
    print("\n" + "="*60)
    print("TEST 5: METRO EXAMPLE")
    print("="*60)
    
    prompt = """Metro network in Prolog:
connected(union_square, times_square).
connected(times_square, grand_central).
connected(grand_central, bryant_park).

Can we reach from union_square to grand_central?
Think step by step:
1. union_square connects to"""
    
    response = client.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt}],
        options={**OPTIONS, 'num_predict': 100}  # Allow longer response
    )
    print(f"Query: Can reach union_square to grand_central?")
    print(f"Response: {response['message']['content']}")

def simple_sld_resolution(goal, rules):
    """Simple SLD resolution - one function, no classes"""
    print(f"\nResolving goal: {goal}")
    print(f"Rules:\n{rules}")
    
    # Step 1: Find matching rule
    prompt1 = f"""SLD Resolution:
Goal: {goal}
Rules:
{rules}

Find the FIRST rule that matches this goal.
If a fact matches exactly, write: FACT MATCHES
If a rule head matches, write: RULE <number> MATCHES
If no match, write: NO MATCH

Answer:"""
    
    response1 = client.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt1}],
        options=OPTIONS
    )
    
    match_result = response1['message']['content']
    print(f"Matching result: {match_result}")
    
    # Step 2: If rule matches, get new subgoals
    if "RULE" in match_result and "MATCHES" in match_result:
        prompt2 = f"""The goal {goal} matched a rule.
After unification and substitution, what are the new subgoals?
List them separated by commas:"""
        
        response2 = client.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt2}],
            options=OPTIONS
        )
        
        print(f"New subgoals: {response2['message']['content']}")
        return response2['message']['content']
    
    elif "FACT MATCHES" in match_result:
        print("Goal proven by fact!")
        return "SUCCESS"
    
    else:
        print("No match found")
        return "FAIL"

# Run all tests
if __name__ == "__main__":
    print("SIMPLE PROLOG LOGIC TESTS")
    print("="*60)
    
    # Run individual tests
    test_unification()
    test_rule_matching()
    test_substitution()
    test_sld_step()
    test_metro_example()
    
    # Test complete SLD resolution
    print("\n" + "="*60)
    print("COMPLETE SLD RESOLUTION TEST")
    print("="*60)
    
    rules = """1. connected(a, b).
2. connected(b, c).
3. reachable(X, Y) :- connected(X, Y).
4. reachable(X, Z) :- connected(X, Y), reachable(Y, Z)."""
    
    # Test 1: Direct fact
    result1 = simple_sld_resolution("connected(a, b)", rules)
    
    # Test 2: One-step derivation
    result2 = simple_sld_resolution("reachable(a, b)", rules)
    
    # Test 3: Multi-step
    result3 = simple_sld_resolution("reachable(a, c)", rules)


# In[41]:


"""
Most basic possible test - find what works
"""

import ollama
import subprocess

client = ollama.Client()
model_name = "gpt-oss:20b"

print("="*60)
print("FINDING WHAT WORKS")
print("="*60)

# Test 1: generate() with minimal options
print("\n1. Testing generate() method:")
try:
    response = client.generate(
        model=model_name,
        prompt="What is 2+2?"
    )
    print(f"Generate response: {response.get('response', 'NO RESPONSE KEY')}")
except Exception as e:
    print(f"Generate failed: {e}")

# Test 2: generate() with no options at all
print("\n2. Testing generate() with simple prompt:")
try:
    response = client.generate(
        model=model_name,
        prompt="Hello"
    )
    print(f"Response: {response.get('response', str(response))}")
except Exception as e:
    print(f"Failed: {e}")

# Test 3: chat() with no options
print("\n3. Testing chat() with no options:")
try:
    response = client.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': 'Hi'}]
    )
    if response and 'message' in response:
        print(f"Chat response: {response['message'].get('content', 'NO CONTENT')}")
    else:
        print(f"Full response: {response}")
except Exception as e:
    print(f"Chat failed: {e}")

# Test 4: Try subprocess (most reliable)
print("\n4. Testing via subprocess (ollama CLI):")
try:
    result = subprocess.run(
        ['ollama', 'run', model_name, 'What is 2+2?'],
        capture_output=True,
        text=True,
        timeout=30
    )
    print(f"CLI response: {result.stdout}")
    if result.stderr:
        print(f"CLI stderr: {result.stderr}")
except Exception as e:
    print(f"Subprocess failed: {e}")

# Test 5: Check if model is loaded
print("\n5. Checking model status:")
try:
    import requests
    response = requests.get("http://localhost:11434/api/ps")
    if response.status_code == 200:
        loaded = response.json()
        if loaded.get('models'):
            print("Models in memory:")
            for m in loaded['models']:
                print(f"  - {m.get('name', 'unknown')}")
        else:
            print("No models loaded in memory")
            print("First call will be slow as model loads...")
except Exception as e:
    print(f"Status check failed: {e}")

print("\n" + "="*60)
print("SUBPROCESS-BASED PROLOG TEST")
print("="*60)

def ollama_run(prompt):
    """Use subprocess to call ollama directly"""
    try:
        result = subprocess.run(
            ['ollama', 'run', model_name, prompt],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

# Test Prolog via subprocess
print("\nTest 1: Unification")
response = ollama_run("In Prolog, if X=john, what is likes(X,pizza)? Give short answer.")
print(f"Response: {response}")

print("\nTest 2: Simple logic")
response = ollama_run("Prolog: Can f(X) unify with f(a)? Answer YES or NO only.")
print(f"Response: {response}")

print("\nTest 3: Rule matching")
prompt = """Given Prolog fact: connected(a,b).
Does this match goal: connected(a,b)?
Answer YES or NO only."""
response = ollama_run(prompt)
print(f"Response: {response}")

# If subprocess works, we can build everything using it
print("\n" + "="*60)
print("WORKING SOLUTION")
print("="*60)

def prolog_query(prompt):
    """Wrapper for Prolog queries via subprocess"""
    result = subprocess.run(
        ['ollama', 'run', model_name, prompt],
        capture_output=True,
        text=True,
        timeout=30
    )
    return result.stdout.strip()

def test_unification_via_cli():
    """Test unification using CLI"""
    prompt = "Prolog unification: likes(X,pizza) = likes(john,pizza). What is X? Answer: X="
    response = prolog_query(prompt)
    print(f"Unification test: {response}")

def test_sld_step_via_cli():
    """Test SLD step using CLI"""
    prompt = """Prolog SLD resolution:
Goal: connected(a,c)
Rules:
1. connected(a,b).
2. connected(b,c).
3. connected(X,Z) :- connected(X,Y), connected(Y,Z).

Which rule matches? Answer with number only:"""
    response = prolog_query(prompt)
    print(f"SLD step test: {response}")

# Run the working solution tests
if subprocess.run(['which', 'ollama'], capture_output=True).returncode == 0:
    print("\nOllama CLI is available. Testing Prolog via CLI:")
    test_unification_via_cli()
    test_sld_step_via_cli()
else:
    print("\nOllama CLI not found in PATH")


# In[42]:


"""
Working SLD Resolution System using Ollama
Based on the successful test results
"""

import ollama
import re

# Initialize client
client = ollama.Client()
model_name = "gpt-oss:20b"

def prolog_generate(prompt):
    """Use generate() which works reliably"""
    response = client.generate(
        model=model_name,
        prompt=prompt,
        options={'temperature': 0.0}  # Keep deterministic
    )
    return response.get('response', '').strip()

def extract_answer(response):
    """Extract the actual answer from the thinking process"""
    # Split by "...done thinking." if present
    if "...done thinking." in response:
        parts = response.split("...done thinking.")
        if len(parts) > 1:
            return parts[-1].strip()
    return response.strip()

def unify(term1, term2):
    """Perform unification between two Prolog terms"""
    prompt = f"""Prolog unification: {term1} = {term2}
What are the variable bindings?
If they unify, list bindings like: X=value, Y=value2
If they don't unify, write: CANNOT UNIFY
Answer:"""
    
    response = prolog_generate(prompt)
    answer = extract_answer(response)
    
    # Parse bindings
    if "CANNOT UNIFY" in answer.upper():
        return None
    
    # Extract variable bindings
    bindings = {}
    # Look for patterns like X=value
    matches = re.findall(r'([A-Z][A-Za-z0-9_]*)\s*=\s*([a-z][A-Za-z0-9_]*|\d+)', answer)
    for var, val in matches:
        bindings[var] = val
    
    return bindings

def find_matching_rule(goal, rules):
    """Find which rules can match a goal"""
    prompt = f"""SLD Resolution - Find matching rule:
Goal: {goal}

Rules:
{rules}

Which rules can match this goal? (considering unification)
List all matching rule numbers separated by commas.
If it's a fact that matches exactly, include its number.
If no rules match, write: NONE
Answer:"""
    
    response = prolog_generate(prompt)
    answer = extract_answer(response)
    
    if "NONE" in answer.upper():
        return []
    
    # Extract numbers from response
    numbers = re.findall(r'\d+', answer)
    return [int(n) for n in numbers]

def apply_rule(goal, rule, rule_body):
    """Apply a rule to a goal and get new subgoals"""
    if not rule_body:  # It's a fact
        prompt = f"""Prolog: Goal {goal} matches fact {rule}.
Does this prove the goal? Answer YES or NO:"""
        response = prolog_generate(prompt)
        answer = extract_answer(response)
        return [] if "YES" in answer.upper() else None
    
    # It's a rule with a body
    prompt = f"""SLD Resolution step:
Goal: {goal}
Rule: {rule}

After unifying the goal with the rule head and applying substitutions,
what are the new subgoals from the rule body?
List them separated by commas.
Answer:"""
    
    response = prolog_generate(prompt)
    answer = extract_answer(response)
    
    # Parse subgoals
    subgoals = [g.strip() for g in answer.split(',') if g.strip()]
    return subgoals

def sld_resolve(goal, knowledge_base, max_depth=10, verbose=True):
    """
    Main SLD Resolution algorithm
    
    Args:
        goal: String representing the goal to prove
        knowledge_base: String containing Prolog rules
        max_depth: Maximum resolution depth
        verbose: Print trace of resolution
    
    Returns:
        (success, trace)
    """
    # Parse rules to identify facts vs rules
    rules = knowledge_base.strip().split('\n')
    parsed_rules = []
    for i, rule in enumerate(rules, 1):
        rule = rule.strip()
        if not rule or rule.startswith('%'):
            continue
        
        if ':-' in rule:
            head, body = rule.split(':-')
            parsed_rules.append((i, head.strip(), body.strip().rstrip('.')))
        else:
            # It's a fact
            parsed_rules.append((i, rule.rstrip('.'), None))
    
    # Stack for depth-first search
    stack = [(goal, [], 0)]  # (current_goal, trace, depth)
    
    while stack:
        current_goal, trace, depth = stack.pop()
        
        if verbose:
            print(f"\nDepth {depth}: Trying to prove: {current_goal}")
        
        if depth >= max_depth:
            if verbose:
                print(f"  Max depth reached")
            continue
        
        # Find matching rules
        matching = find_matching_rule(current_goal, knowledge_base)
        
        if not matching:
            if verbose:
                print(f"  No matching rules found")
            continue
        
        if verbose:
            print(f"  Matching rules: {matching}")
        
        # Try each matching rule
        for rule_num in matching:
            # Find the actual rule
            rule_data = None
            for num, head, body in parsed_rules:
                if num == rule_num:
                    rule_data = (head, body)
                    break
            
            if not rule_data:
                continue
            
            head, body = rule_data
            
            if verbose:
                if body:
                    print(f"  Applying rule {rule_num}: {head} :- {body}")
                else:
                    print(f"  Matching fact {rule_num}: {head}")
            
            # Apply the rule
            new_subgoals = apply_rule(current_goal, head, body)
            
            if new_subgoals is None:
                continue
            
            if len(new_subgoals) == 0:
                # Goal proven!
                if verbose:
                    print(f"  ✓ Goal proven!")
                return True, trace + [f"Proved {current_goal} using rule {rule_num}"]
            
            # Add new subgoals to stack
            for subgoal in reversed(new_subgoals):  # Reverse for left-to-right processing
                stack.append((subgoal, trace + [f"Applied rule {rule_num} to {current_goal}"], depth + 1))
    
    return False, trace

# Test functions
def test_metro_network():
    """Test with metro network example"""
    print("="*60)
    print("METRO NETWORK SLD RESOLUTION")
    print("="*60)
    
    knowledge_base = """1. connected(union_square, times_square).
2. connected(times_square, grand_central).
3. connected(grand_central, bryant_park).
4. reachable(X, Y) :- connected(X, Y).
5. reachable(X, Z) :- connected(X, Y), reachable(Y, Z)."""
    
    # Test 1: Direct connection
    print("\nTest 1: Is union_square connected to times_square?")
    success, trace = sld_resolve("connected(union_square, times_square)", knowledge_base)
    print(f"Result: {'SUCCESS' if success else 'FAILED'}")
    
    # Test 2: One-hop reachability
    print("\nTest 2: Is union_square reachable to times_square?")
    success, trace = sld_resolve("reachable(union_square, times_square)", knowledge_base)
    print(f"Result: {'SUCCESS' if success else 'FAILED'}")
    
    # Test 3: Multi-hop reachability
    print("\nTest 3: Is union_square reachable to grand_central?")
    success, trace = sld_resolve("reachable(union_square, grand_central)", knowledge_base)
    print(f"Result: {'SUCCESS' if success else 'FAILED'}")
    
    # Test 4: Three-hop reachability
    print("\nTest 4: Is union_square reachable to bryant_park?")
    success, trace = sld_resolve("reachable(union_square, bryant_park)", knowledge_base)
    print(f"Result: {'SUCCESS' if success else 'FAILED'}")

def test_simple_logic():
    """Test with simple logical rules"""
    print("\n" + "="*60)
    print("SIMPLE LOGIC TEST")
    print("="*60)
    
    knowledge_base = """1. bird(robin).
2. bird(penguin).
3. can_fly(X) :- bird(X).
4. has_wings(X) :- bird(X)."""
    
    print("\nTest: Can robin fly?")
    success, trace = sld_resolve("can_fly(robin)", knowledge_base, verbose=False)
    print(f"Result: {'YES' if success else 'NO'}")
    
    print("\nTest: Does penguin have wings?")
    success, trace = sld_resolve("has_wings(penguin)", knowledge_base, verbose=False)
    print(f"Result: {'YES' if success else 'NO'}")

def interactive_prolog():
    """Interactive Prolog query interface"""
    print("\n" + "="*60)
    print("INTERACTIVE PROLOG (type 'quit' to exit)")
    print("="*60)
    
    # Default knowledge base
    kb = """1. parent(tom, bob).
2. parent(bob, ann).
3. parent(bob, pat).
4. grandparent(X, Z) :- parent(X, Y), parent(Y, Z)."""
    
    print("\nCurrent knowledge base:")
    print(kb)
    
    while True:
        query = input("\n?- ").strip()
        if query.lower() == 'quit':
            break
        
        if query.startswith('add '):
            # Add new rule
            new_rule = query[4:].strip()
            kb = kb + f"\n{len(kb.split(chr(10))) + 1}. {new_rule}"
            print("Rule added.")
            print("\nUpdated knowledge base:")
            print(kb)
        else:
            # Query the knowledge base
            success, trace = sld_resolve(query, kb, verbose=False)
            if success:
                print("true.")
            else:
                print("false.")

# Run tests
if __name__ == "__main__":
    # Test unification
    print("UNIFICATION TEST")
    print("="*60)
    result = unify("likes(X, pizza)", "likes(john, pizza)")
    print(f"Unify likes(X, pizza) with likes(john, pizza): {result}")
    
    result = unify("f(X, Y)", "f(a, b)")
    print(f"Unify f(X, Y) with f(a, b): {result}")
    
    # Test metro network
    test_metro_network()
    
    # Test simple logic
    test_simple_logic()
    
    # Optional: Run interactive mode
    print("\nWould you like to try interactive mode? (y/n)")
    if input().lower() == 'y':
        interactive_prolog()


# In[43]:


"""
Working SLD Resolution System using Ollama
Based on the successful test results
"""

import ollama
import re

# Initialize client
client = ollama.Client()
model_name = "gpt-oss:20b"

def prolog_generate(prompt):
    """Use generate() which works reliably"""
    response = client.generate(
        model=model_name,
        prompt=prompt,
        options={'temperature': 0.0}  # Keep deterministic
    )
    return response.get('response', '').strip()

def extract_answer(response):
    """Extract the actual answer from the thinking process"""
    # Split by "...done thinking." if present
    if "...done thinking." in response:
        parts = response.split("...done thinking.")
        if len(parts) > 1:
            return parts[-1].strip()
    return response.strip()

def unify(term1, term2):
    """Perform unification between two Prolog terms"""
    prompt = f"""Prolog unification: {term1} = {term2}
What are the variable bindings?
If they unify, list bindings like: X=value, Y=value2
If they don't unify, write: CANNOT UNIFY
Answer:"""
    
    response = prolog_generate(prompt)
    answer = extract_answer(response)
    
    # Parse bindings
    if "CANNOT UNIFY" in answer.upper():
        return None
    
    # Extract variable bindings
    bindings = {}
    # Look for patterns like X=value
    matches = re.findall(r'([A-Z][A-Za-z0-9_]*)\s*=\s*([a-z][A-Za-z0-9_]*|\d+)', answer)
    for var, val in matches:
        bindings[var] = val
    
    return bindings

def find_matching_rule(goal, rules):
    """Find which rules can match a goal"""
    prompt = f"""SLD Resolution - Find matching rule:
Goal: {goal}

Rules:
{rules}

Which rules can match this goal? (considering unification)
List all matching rule numbers separated by commas.
If it's a fact that matches exactly, include its number.
If no rules match, write: NONE
Answer:"""
    
    response = prolog_generate(prompt)
    answer = extract_answer(response)
    
    if "NONE" in answer.upper():
        return []
    
    # Extract numbers from response
    numbers = re.findall(r'\d+', answer)
    return [int(n) for n in numbers]

def apply_rule(goal, rule, rule_body):
    """Apply a rule to a goal and get new subgoals"""
    if not rule_body:  # It's a fact
        prompt = f"""Prolog: Goal {goal} matches fact {rule}.
Does this prove the goal? Answer YES or NO:"""
        response = prolog_generate(prompt)
        answer = extract_answer(response)
        return [] if "YES" in answer.upper() else None
    
    # It's a rule with a body
    prompt = f"""SLD Resolution:
Goal to prove: {goal}
Using rule: {rule} :- {rule_body}

Step 1: Unify the goal with the rule head
Step 2: Apply the substitution to the rule body
Step 3: Return the new subgoals

Example: If goal is reachable(a,c) and rule is reachable(X,Z) :- connected(X,Y), reachable(Y,Z)
Then X=a, Z=c, and new subgoals are: connected(a,Y), reachable(Y,c)

What are the new subgoals after substitution?
List ONLY the subgoals, separated by commas:"""
    
    response = prolog_generate(prompt)
    answer = extract_answer(response)
    
    # Clean up the answer - remove any explanatory text
    # Look for patterns like predicate(args)
    import re
    predicates = re.findall(r'\b[a-z_][a-z0-9_]*\([^)]+\)', answer)
    
    if predicates:
        return predicates
    
    # Fallback: split by comma and clean
    subgoals = []
    for part in answer.split(','):
        part = part.strip()
        # Check if it looks like a Prolog predicate
        if '(' in part and ')' in part:
            subgoals.append(part)
    
    return subgoals if subgoals else None

def sld_resolve(goal, knowledge_base, max_depth=10, verbose=True):
    """
    Main SLD Resolution algorithm
    
    Args:
        goal: String representing the goal to prove
        knowledge_base: String containing Prolog rules
        max_depth: Maximum resolution depth
        verbose: Print trace of resolution
    
    Returns:
        (success, trace)
    """
    # Parse rules to identify facts vs rules
    rules = knowledge_base.strip().split('\n')
    parsed_rules = []
    for i, rule in enumerate(rules, 1):
        rule = rule.strip()
        if not rule or rule.startswith('%'):
            continue
        
        # Extract rule number if present
        if re.match(r'^\d+\.\s*', rule):
            rule = re.sub(r'^\d+\.\s*', '', rule)
        
        if ':-' in rule:
            head, body = rule.split(':-')
            parsed_rules.append((i, head.strip(), body.strip().rstrip('.')))
        else:
            # It's a fact
            parsed_rules.append((i, rule.rstrip('.'), None))
    
    # Stack for depth-first search: (current_goals, trace, depth)
    stack = [([goal], [], 0)]
    
    while stack:
        current_goals, trace, depth = stack.pop()
        
        if not current_goals:
            # All goals proven!
            if verbose:
                print(f"  ✓ All goals proven!")
            return True, trace
        
        # Take first goal (leftmost in SLD)
        current_goal = current_goals[0]
        remaining_goals = current_goals[1:]
        
        if verbose:
            print(f"\nDepth {depth}: Trying to prove: {current_goal}")
            if remaining_goals:
                print(f"  Remaining goals: {remaining_goals}")
        
        if depth >= max_depth:
            if verbose:
                print(f"  Max depth reached")
            continue
        
        # Find matching rules
        matching = find_matching_rule(current_goal, knowledge_base)
        
        if not matching:
            if verbose:
                print(f"  No matching rules found")
            continue
        
        if verbose:
            print(f"  Matching rules: {matching}")
        
        # Try each matching rule
        for rule_num in matching:
            # Find the actual rule
            rule_data = None
            for num, head, body in parsed_rules:
                if num == rule_num:
                    rule_data = (head, body)
                    break
            
            if not rule_data:
                continue
            
            head, body = rule_data
            
            if verbose:
                if body:
                    print(f"  Applying rule {rule_num}: {head} :- {body}")
                else:
                    print(f"  Matching fact {rule_num}: {head}")
            
            # Apply the rule
            new_subgoals = apply_rule(current_goal, head, body)
            
            if new_subgoals is None:
                if verbose:
                    print(f"    Failed to apply rule")
                continue
            
            # Combine new subgoals with remaining goals
            all_new_goals = new_subgoals + remaining_goals
            
            if verbose:
                if new_subgoals:
                    print(f"    New subgoals: {new_subgoals}")
                else:
                    print(f"    Goal proven by fact!")
            
            # Add to stack for further exploration
            new_trace = trace + [f"Applied rule {rule_num} to {current_goal}"]
            stack.append((all_new_goals, new_trace, depth + 1))
    
    return False, trace

# Test functions
def test_metro_network():
    """Test with metro network example"""
    print("="*60)
    print("METRO NETWORK SLD RESOLUTION")
    print("="*60)
    
    knowledge_base = """1. connected(union_square, times_square).
2. connected(times_square, grand_central).
3. connected(grand_central, bryant_park).
4. reachable(X, Y) :- connected(X, Y).
5. reachable(X, Z) :- connected(X, Y), reachable(Y, Z)."""
    
    # Test 1: Direct connection
    print("\nTest 1: Is union_square connected to times_square?")
    success, trace = sld_resolve("connected(union_square, times_square)", knowledge_base)
    print(f"Result: {'SUCCESS' if success else 'FAILED'}")
    
    # Test 2: One-hop reachability
    print("\nTest 2: Is union_square reachable to times_square?")
    success, trace = sld_resolve("reachable(union_square, times_square)", knowledge_base)
    print(f"Result: {'SUCCESS' if success else 'FAILED'}")
    
    # Test 3: Multi-hop reachability
    print("\nTest 3: Is union_square reachable to grand_central?")
    success, trace = sld_resolve("reachable(union_square, grand_central)", knowledge_base)
    print(f"Result: {'SUCCESS' if success else 'FAILED'}")
    
    # Test 4: Three-hop reachability
    print("\nTest 4: Is union_square reachable to bryant_park?")
    success, trace = sld_resolve("reachable(union_square, bryant_park)", knowledge_base)
    print(f"Result: {'SUCCESS' if success else 'FAILED'}")

def test_simple_logic():
    """Test with simple logical rules"""
    print("\n" + "="*60)
    print("SIMPLE LOGIC TEST")
    print("="*60)
    
    knowledge_base = """1. bird(robin).
2. bird(penguin).
3. can_fly(X) :- bird(X).
4. has_wings(X) :- bird(X)."""
    
    print("\nTest: Can robin fly?")
    print("Knowledge base:")
    print(knowledge_base)
    success, trace = sld_resolve("can_fly(robin)", knowledge_base, verbose=True)
    print(f"Final Result: {'YES' if success else 'NO'}")
    
    print("\nTest: Does penguin have wings?")
    success, trace = sld_resolve("has_wings(penguin)", knowledge_base, verbose=True)
    print(f"Final Result: {'YES' if success else 'NO'}")

def interactive_prolog():
    """Interactive Prolog query interface"""
    print("\n" + "="*60)
    print("INTERACTIVE PROLOG (type 'quit' to exit)")
    print("="*60)
    
    # Default knowledge base
    kb = """1. parent(tom, bob).
2. parent(bob, ann).
3. parent(bob, pat).
4. grandparent(X, Z) :- parent(X, Y), parent(Y, Z)."""
    
    print("\nCurrent knowledge base:")
    print(kb)
    
    while True:
        query = input("\n?- ").strip()
        if query.lower() == 'quit':
            break
        
        if query.startswith('add '):
            # Add new rule
            new_rule = query[4:].strip()
            kb = kb + f"\n{len(kb.split(chr(10))) + 1}. {new_rule}"
            print("Rule added.")
            print("\nUpdated knowledge base:")
            print(kb)
        else:
            # Query the knowledge base
            success, trace = sld_resolve(query, kb, verbose=False)
            if success:
                print("true.")
            else:
                print("false.")

# Run tests
if __name__ == "__main__":
    # Test unification
    print("UNIFICATION TEST")
    print("="*60)
    result = unify("likes(X, pizza)", "likes(john, pizza)")
    print(f"Unify likes(X, pizza) with likes(john, pizza): {result}")
    
    result = unify("f(X, Y)", "f(a, b)")
    print(f"Unify f(X, Y) with f(a, b): {result}")
    
    # Test metro network
    test_metro_network()
    
    # Test simple logic
    test_simple_logic()
    
    # Optional: Run interactive mode
    print("\nWould you like to try interactive mode? (y/n)")
    if input().lower() == 'y':
        interactive_prolog()


# In[ ]:





# In[ ]:





# In[1]:


"""
Breadth-First SLD Resolution with Natural Language Rules
"""

import ollama
import re
from collections import deque

# Initialize client
client = ollama.Client()
model_name = "gpt-oss:20b"

def prolog_generate(prompt):
    """Use generate() which works reliably"""
    response = client.generate(
        model=model_name,
        prompt=prompt,
        options={'temperature': 0.0}
    )
    return response.get('response', '').strip()

def extract_answer(response):
    """Extract the actual answer from the thinking process"""
    if "...done thinking." in response:
        parts = response.split("...done thinking.")
        if len(parts) > 1:
            return parts[-1].strip()
    return response.strip()

def natural_to_prolog(natural_rules):
    """Convert natural language rules to Prolog format"""
    prompt = f"""Convert these natural language rules to Prolog format:

{natural_rules}

Rules for conversion:
- "X is Y's parent" becomes: parent(X, Y)
- "X can fly if X is a bird" becomes: can_fly(X) :- bird(X)
- "X is connected to Y" becomes: connected(X, Y)
- "X can reach Y if X is connected to Y" becomes: reachable(X, Y) :- connected(X, Y)
- Facts end with period, rules use :- for "if"

Write each rule on a new line in Prolog format:"""
    
    response = prolog_generate(prompt)
    answer = extract_answer(response)
    return answer

def find_matching_rules(goal, knowledge_base):
    """Find ALL rules that can match a goal"""
    prompt = f"""Prolog Pattern Matching:

Goal to prove: {goal}

Knowledge Base:
{knowledge_base}

Which rules can unify with this goal?
- Standardize the rules and goals apart
- Facts match if they unify exactly
- Rules match if their head unifies with the goal
- Consider variable unification

List ONLY the rule numbers, separated by commas.
If NO rules match, write: NONE

Answer:"""
    
    response = prolog_generate(prompt)
    answer = extract_answer(response)
    
    if "NONE" in answer.upper():
        return []
    
    numbers = re.findall(r'\d+', answer)
    return [int(n) for n in numbers]

def apply_rule_get_subgoals(goal, rule_head, rule_body):
    """Apply a rule and get new subgoals"""
    if not rule_body:  # It's a fact
        return []  # Goal proven
    
    prompt = f"""SLD Resolution Step:

Goal: {goal}
Rule head: {rule_head}
Rule body: {rule_body}

Unify goal with head, then apply substitutions to body.
Write each new subgoal on its own line:"""
    
    response = prolog_generate(prompt)
    answer = extract_answer(response)
    
    # Extract predicates
    subgoals = re.findall(r'[a-z_][a-zA-Z0-9_]*\([^)]+\)', answer)
    
    if not subgoals and answer:
        lines = answer.strip().split('\n')
        subgoals = [line.strip() for line in lines if '(' in line and ')' in line]
    
    return [sg.strip().rstrip('.,;') for sg in subgoals if sg.strip()]

def bfs_sld_resolve(goal, knowledge_base, max_iterations=100, verbose=True):
    """
    Breadth-First Search SLD Resolution
    
    Args:
        goal: Goal to prove
        knowledge_base: Prolog rules (can be from natural language)
        max_iterations: Maximum BFS iterations
        verbose: Print trace
    
    Returns:
        (success, proof_path)
    """
    # Parse knowledge base
    rules = []
    for line in knowledge_base.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        
        # Extract rule number if present
        if line[0].isdigit():
            parts = line.split('.', 1)
            if len(parts) > 1:
                num = int(parts[0])
                rule = parts[1].strip()
            else:
                continue
        else:
            num = len(rules) + 1
            rule = line
        
        if ':-' in rule:
            head, body = rule.split(':-')
            rules.append((num, head.strip(), body.strip().rstrip('.')))
        else:
            rules.append((num, rule.rstrip('.'), None))
    
    # BFS queue: (goals_list, proof_path, depth)
    queue = deque([([goal], [], 0)])
    visited = set()
    iteration = 0
    
    if verbose:
        print(f"Starting BFS Resolution for: {goal}")
        print(f"Knowledge base has {len(rules)} rules")
        print("="*40)
    
    while queue and iteration < max_iterations:
        iteration += 1
        goals, path, depth = queue.popleft()
        
        if not goals:
            # All goals proven!
            if verbose:
                print(f"\n✓ Proof found at depth {depth}!")
            return True, path
        
        current_goal = goals[0]
        remaining_goals = goals[1:]
        
        # Skip if we've seen this state
        state = (current_goal, tuple(remaining_goals))
        if state in visited:
            continue
        visited.add(state)
        
        if verbose:
            print(f"\nIteration {iteration} (depth {depth})")
            print(f"  Current goal: {current_goal}")
            if remaining_goals:
                print(f"  Remaining: {remaining_goals}")
        
        # Find matching rules
        matching = find_matching_rules(current_goal, knowledge_base)
        print(current_goal)
        print(matching)
        
        if not matching:
            if verbose:
                print(f"  No rules match")
            continue
        
        if verbose:
            print(f"  Matching rules: {matching}")
        
        # Try each matching rule
        for rule_num in matching:
            # Find the actual rule
            rule_data = None
            for num, head, body in rules:
                if num == rule_num:
                    rule_data = (head, body)
                    break
            
            if not rule_data:
                continue
            
            head, body = rule_data
            
            # Get new subgoals
            new_subgoals = apply_rule_get_subgoals(current_goal, head, body)
            
            # Create new goals list
            new_goals = new_subgoals + remaining_goals
            
            # Add to queue (breadth-first)
            new_path = path + [f"Rule {rule_num}: {current_goal} ← {head}"]
            queue.append((new_goals, new_path, depth + 1))
            
            if verbose:
                if body:
                    print(f"    Applied rule {rule_num}: {head} :- {body}")
                    print(f"    New subgoals: {new_subgoals}")
                else:
                    print(f"    Matched fact {rule_num}: {head}")
    
    if verbose:
        print(f"\nNo proof found after {iteration} iterations")
    return False, []

def test_natural_language():
    """Test with natural language rules"""
    print("="*60)
    print("NATURAL LANGUAGE RULES TEST")
    print("="*60)
    
    natural_rules = """
    1. Tom is Bob's parent
    2. Bob is Ann's parent
    3. Bob is Pat's parent
    4. X is Z's grandparent if X is Y's parent and Y is Z's parent
    5. X is Y's ancestor if X is Y's parent
    6. X is Z's ancestor if X is Y's parent and Y is Z's ancestor
    """
    
    print("Natural language rules:")
    print(natural_rules)
    
    # Convert to Prolog
    print("\nConverting to Prolog format...")
    prolog_rules = natural_to_prolog(natural_rules)
    print("Prolog rules:")
    print(prolog_rules)
    
    # Test queries
    queries = [
        ("grandparent(tom, ann)", "Is Tom Ann's grandparent?"),
        ("grandparent(tom, pat)", "Is Tom Pat's grandparent?"),
        ("ancestor(tom, ann)", "Is Tom Ann's ancestor?"),
    ]
    
    for goal, question in queries:
        print(f"\n{question}")
        success, path = bfs_sld_resolve(goal, prolog_rules, verbose=False)
        print(f"Answer: {'YES' if success else 'NO'}")
        if success and path:
            print(f"Proof: {' → '.join(path[:3])}")  # Show first 3 steps

def test_metro_bfs():
    """Test metro network with BFS"""
    print("\n" + "="*60)
    print("METRO NETWORK WITH BFS")
    print("="*60)
    
    knowledge_base = """1. connected(union_square, times_square).
2. connected(times_square, grand_central).
3. connected(grand_central, bryant_park).
4. connected(times_square, columbus_circle).
5. connected(columbus_circle, central_park).
6. reachable(X, Y) :- connected(X, Y).
7. reachable(X, Z) :- connected(X, Y), reachable(Y, Z)."""
    
    print("Metro network:")
    print(knowledge_base)
    
    # Test BFS finds shortest path
    print("\nTesting BFS (should find shortest proof):")
    
    queries = [
        "reachable(union_square, grand_central)",
        "reachable(union_square, central_park)",
        "reachable(times_square, bryant_park)",
    ]
    
    for goal in queries:
        print(f"\nQuery: {goal}")
        success, path = bfs_sld_resolve(goal, knowledge_base, verbose=False)
        print(f"Result: {'SUCCESS' if success else 'FAILED'}")
        if success:
            print(f"Proof depth: {len(path)}")
            print(f"Path: {' → '.join(path[:5])}")  # Show first 5 steps

def test_logic_puzzles():
    """Test with logic puzzle rules in natural language"""
    print("\n" + "="*60)
    print("LOGIC PUZZLE WITH NATURAL LANGUAGE")
    print("="*60)
    
    natural_rules = """
    1. All birds have wings
    2. Robins are birds
    3. Penguins are birds
    4. Eagles are birds
    5. Things with wings can fly unless they are penguins
    6. Eagles are predators
    7. Predators hunt prey
    """
    
    print("Natural language rules:")
    print(natural_rules)
    
    # Convert to Prolog-like format
    prolog_rules = """1. has_wings(X) :- bird(X).
2. bird(robin).
3. bird(penguin).
4. bird(eagle).
5. can_fly(X) :- has_wings(X), not_penguin(X).
6. not_penguin(X) :- bird(X), X \= penguin.
7. can_fly(X) :- bird(X), X \= penguin.
8. predator(eagle).
9. hunts(X, prey) :- predator(X)."""
    
    print("\nConverted rules (simplified):")
    print(prolog_rules)
    
    queries = [
        ("bird(robin)", "Is robin a bird?"),
        ("has_wings(penguin)", "Does penguin have wings?"),
        ("can_fly(eagle)", "Can eagle fly?"),
        ("predator(eagle)", "Is eagle a predator?"),
    ]
    
    for goal, question in queries:
        print(f"\n{question}")
        success, path = bfs_sld_resolve(goal, prolog_rules, verbose=False)
        print(f"Answer: {'YES' if success else 'NO'}")

def compare_dfs_bfs():
    """Compare DFS vs BFS search strategies"""
    print("\n" + "="*60)
    print("COMPARING BFS VS DFS")
    print("="*60)
    
    # Knowledge base with multiple paths
    kb = """1. path(a, d).
2. path(a, b).
3. path(b, c).
4. path(c, d).
5. route(X, Y) :- path(X, Y).
6. route(X, Z) :- path(X, Y), route(Y, Z)."""
    
    print("Knowledge base (multiple paths from a to d):")
    print(kb)
    print("\nDirect path: a → d (fact 1)")
    print("Indirect path: a → b → c → d (facts 2,3,4)")
    
    goal = "route(a, d)"
    
    print(f"\nQuery: {goal}")
    print("\nBFS will find the shortest proof (direct path)")
    success, path = bfs_sld_resolve(goal, kb, verbose=True)
    
    if success:
        print(f"\nBFS found proof with {len(path)} steps")
        for step in path:
            print(f"  {step}")

# Run all tests
if __name__ == "__main__":
    # Test natural language conversion
   # test_natural_language()
    
    # Test metro with BFS
    test_metro_bfs()
    
    # Test logic puzzles
    #test_logic_puzzles()
    
    # Compare BFS vs DFS
    #compare_dfs_bfs()


# In[ ]:





# In[45]:


"""
Simple Metro Network BFS - Natural Language Only
"""

import ollama
from collections import deque

client = ollama.Client()
model = "gpt-oss:20b"

def ask(prompt):
    """Short LLM query"""
    resp = client.generate(
        model=model, 
        prompt=prompt, 
        options={'temperature': 0.0, 'num_predict': 30}
    )
    answer = resp.get('response', '')
    if "...done thinking." in answer:
        return answer.split("...done thinking.")[-1].strip()
    return answer.strip()

def bfs_solve(goal, rules):
    """BFS to prove goal using natural language rules"""
    
    queue = deque([(goal, 0)])  # (current_goal, depth)
    visited = set()
    
    print(f"\nGoal: {goal}")
    print("-" * 40)
    
    while queue:
        current, depth = queue.popleft()
        
        if current in visited or depth > 5:
            continue
        visited.add(current)
        
        print(f"Level {depth}: {current}")
        
        # Check if current goal matches a fact
        prompt = f"""Rules:
{rules}

Does "{current}" match any fact exactly?
Answer: YES (rule number) or NO"""
        
        answer = ask(prompt)
        
        if "YES" in answer:
            print(f"  ✓ Proven by fact")
            return True
        
        # Get subgoals from rules
        prompt = f"""Rules:
{rules}

To prove "{current}", what needs to be proven?
List subgoals only, one per line:"""
        
        answer = ask(prompt)
        
        if answer and "cannot" not in answer.lower():
            subgoals = [s.strip() for s in answer.split('\n') if s.strip()]
            if subgoals:
                print(f"  → Requires: {subgoals}")
                for subgoal in subgoals:
                    queue.append((subgoal, depth + 1))
    
    print("  ✗ No proof found")
    return False

# Test metro network
rules = """
1. Union Square is connected to Times Square
2. Times Square is connected to Grand Central  
3. Grand Central is connected to Bryant Park
4. X can reach Y if X is connected to Y
5. X can reach Z if X is connected to Y and Y can reach Z
"""

print("METRO NETWORK - BFS")
print("=" * 50)
print("Rules:")
print(rules)

# Test cases
tests = [
    "Union Square is connected to Times Square",
    "Union Square can reach Grand Central",
    "Union Square can reach Bryant Park"
]

for test in tests:
    result = bfs_solve(test, rules)
    print(f"Result: {'SUCCESS' if result else 'FAILED'}\n")


# In[46]:


"""
BFS Resolution with Natural Language Rules
Based on working DFS version, converted to BFS with natural language
"""

import ollama
import re
from collections import deque

# Initialize client
client = ollama.Client()
model_name = "gpt-oss:20b"

def ask_llm(prompt):
    """Simple LLM query with answer extraction"""
    response = client.generate(
        model=model_name,
        prompt=prompt,
        options={'temperature': 0.0}
    )
    answer = response.get('response', '').strip()
    if "...done thinking." in answer:
        return answer.split("...done thinking.")[-1].strip()
    return answer

def match_fact(statement, facts):
    """Check if statement matches any fact exactly"""
    prompt = f"""Facts:
{facts}

Does the statement "{statement}" match any fact above exactly?
If yes, write the fact number (e.g., "1" or "2")
If no, write "NO"
Answer:"""
    
    answer = ask_llm(prompt)
    # Extract just the number or NO
    if "NO" in answer.upper():
        return None
    numbers = re.findall(r'\d+', answer)
    return int(numbers[0]) if numbers else None

def find_applicable_rules(goal, rules):
    """Find which rules could apply to prove the goal"""
    prompt = f"""Rules:
{rules}

Which rules could be used to prove: "{goal}"?

Only consider rules that have conclusions matching the pattern of the goal.
List just the rule numbers (e.g., "4,5")
If no rules apply, write "NONE"
Answer:"""
    
    answer = ask_llm(prompt)
    if "NONE" in answer.upper():
        return []
    numbers = re.findall(r'\d+', answer)
    return [int(n) for n in numbers]

def get_conditions(goal, rule_text):
    """Extract what conditions need to be proven for a rule"""
    prompt = f"""To prove: "{goal}"
Using rule: "{rule_text}"

What specific conditions must be proven?
Write each condition on a new line.
If the rule doesn't apply, write "CANNOT"
Answer:"""
    
    answer = ask_llm(prompt)
    if "CANNOT" in answer.upper():
        return None
    
    # Extract conditions (lines that describe what needs to be proven)
    conditions = []
    for line in answer.split('\n'):
        line = line.strip()
        # Filter out empty lines and meta-text
        if line and len(line) > 5 and not line.startswith('-'):
            conditions.append(line)
    return conditions

def bfs_resolve(goal, rules_text, verbose=True):
    """
    BFS Resolution for natural language rules
    
    Args:
        goal: Natural language goal to prove
        rules_text: Natural language rules
        verbose: Print trace
    
    Returns:
        (success, proof_path)
    """
    # Parse rules into facts and implications
    facts = []
    implications = []
    
    for line in rules_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Extract rule number and text
        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            text = match.group(2)
            
            # Simple heuristic: if it contains "if" it's an implication
            if " if " in text.lower():
                implications.append((num, text))
            else:
                facts.append((num, text))
    
    # BFS queue: (goals_to_prove, proof_path, depth)
    queue = deque([([goal], [], 0)])
    visited = set()
    max_depth = 10
    
    if verbose:
        print(f"\nGoal: {goal}")
        print("="*40)
    
    while queue:
        goals, path, depth = queue.popleft()
        
        if not goals:
            # All goals proven!
            if verbose:
                print(f"✓ SUCCESS at depth {depth}")
            return True, path
        
        if depth >= max_depth:
            continue
        
        current = goals[0]
        remaining = goals[1:]
        
        # Check visited state
        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)
        
        if verbose:
            print(f"\nDepth {depth}: Proving: {current}")
            if remaining:
                print(f"  Remaining: {remaining[:2]}...")  # Show first 2
        
        # First check if it's a fact
        facts_text = '\n'.join([f"{n}. {t}" for n, t in facts])
        fact_match = match_fact(current, facts_text)
        
        if fact_match:
            if verbose:
                print(f"  ✓ Fact {fact_match} matches")
            # Continue with remaining goals
            new_path = path + [f"Fact {fact_match}: {current}"]
            queue.append((remaining, new_path, depth + 1))
            continue
        
        # Find applicable rules
        rules_for_checking = '\n'.join([f"{n}. {t}" for n, t in implications])
        applicable = find_applicable_rules(current, rules_for_checking)
        
        if not applicable:
            if verbose:
                print(f"  No applicable rules")
            continue
        
        if verbose:
            print(f"  Applicable rules: {applicable}")
        
        # Try each applicable rule
        for rule_num in applicable:
            # Find the rule text
            rule_text = None
            for n, t in implications:
                if n == rule_num:
                    rule_text = t
                    break
            
            if not rule_text:
                continue
            
            # Get conditions for this rule
            conditions = get_conditions(current, rule_text)
            
            if conditions:
                if verbose:
                    print(f"  Rule {rule_num} requires: {conditions}")
                
                # Add conditions + remaining goals to queue
                new_goals = conditions + remaining
                new_path = path + [f"Rule {rule_num}: {current} needs {conditions}"]
                queue.append((new_goals, new_path, depth + 1))
    
    if verbose:
        print(f"\n✗ No proof found")
    return False, []

def test_metro_natural():
    """Test metro network with natural language rules"""
    print("="*60)
    print("METRO NETWORK - Natural Language BFS")
    print("="*60)
    
    rules = """
1. Union Square is connected to Times Square
2. Times Square is connected to Grand Central
3. Grand Central is connected to Bryant Park
4. X can reach Y if X is connected to Y
5. X can reach Z if X is connected to Y and Y can reach Z
"""
    
    print("Rules:")
    print(rules)
    
    tests = [
        "Union Square is connected to Times Square",
        "Union Square can reach Times Square", 
        "Union Square can reach Grand Central",
        "Union Square can reach Bryant Park"
    ]
    
    for test in tests:
        success, path = bfs_resolve(test, rules, verbose=True)
        print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
        if success and path:
            print("Proof:")
            for step in path[:5]:  # Show first 5 steps
                print(f"  - {step}")
        print()

def test_family_natural():
    """Test family relationships with natural language"""
    print("="*60)
    print("FAMILY RELATIONSHIPS - Natural Language BFS")
    print("="*60)
    
    rules = """
1. Tom is Bob's parent
2. Bob is Ann's parent
3. Bob is Pat's parent
4. X is Y's grandparent if X is Y's parent's parent
5. X is Y's grandparent if X is Z's parent and Z is Y's parent
"""
    
    print("Rules:")
    print(rules)
    
    tests = [
        "Tom is Bob's parent",
        "Tom is Ann's grandparent",
        "Tom is Pat's grandparent"
    ]
    
    for test in tests:
        success, path = bfs_resolve(test, rules, verbose=False)
        print(f"\nQuery: {test}")
        print(f"Result: {'YES' if success else 'NO'}")

if __name__ == "__main__":
    test_metro_natural()
    #test_family_natural()


# In[48]:


"""
Simplified BFS for Metro Network - With Output Validation
"""

import ollama
import re
from collections import deque

client = ollama.Client()
model_name = "gpt-oss:20b"

def ask_llm(prompt):
    """Simple LLM query"""
    response = client.generate(
        model=model_name,
        prompt=prompt,
        options={'temperature': 0.0}
    )
    answer = response.get('response', '').strip()
    if "...done thinking." in answer:
        return answer.split("...done thinking.")[-1].strip()
    return answer

def extract_conditions_from_rule(goal, rule_text):
    """Extract conditions in a very explicit way"""
    # Parse rule structure more explicitly
    if "if" in rule_text.lower():
        parts = rule_text.lower().split(" if ")
        if len(parts) == 2:
            conclusion = parts[0].strip()
            conditions = parts[1].strip()
            
            # Simple pattern matching
            prompt = f"""Match this pattern:
Rule pattern: "{conclusion}"  
Goal: "{goal}"

Rule conditions: "{conditions}"

Replace X, Y, Z in the conditions with the actual values from the goal.
Write each condition on a new line.
Keep it simple - just the conditions needed.

Example:
Rule pattern: "x can reach z"
Goal: "union square can reach grand central"
Conditions: "x is connected to y and y can reach z"
Output:
union square is connected to y
y can reach grand central

Now do it for the given rule and goal:"""
            
            answer = ask_llm(prompt)
            
            # Validate the answer contains relevant keywords
            conditions = []
            for line in answer.split('\n'):
                line = line.strip().lower()
                # Check if line is relevant (contains connection/reach keywords)
                if line and ('connected' in line or 'reach' in line or 'parent' in line):
                    conditions.append(line)
            
            return conditions if conditions else None
    
    return None

def bfs_metro_simple(goal, rules_text, verbose=True):
    """Simplified BFS specifically for metro network"""
    
    # Parse rules manually for clarity
    facts = {
        1: "union square is connected to times square",
        2: "times square is connected to grand central",
        3: "grand central is connected to bryant park"
    }
    
    # Normalize goal
    goal = goal.lower().strip()
    
    # BFS queue
    queue = deque([(goal, [], 0)])
    visited = set()
    max_depth = 10
    
    if verbose:
        print(f"\nGoal: {goal}")
        print("="*40)
    
    while queue:
        current, path, depth = queue.popleft()
        
        if depth >= max_depth:
            continue
            
        if current in visited:
            continue
        visited.add(current)
        
        if verbose:
            print(f"\nDepth {depth}: {current}")
        
        # Check facts directly
        for fact_num, fact_text in facts.items():
            if current == fact_text:
                if verbose:
                    print(f"  ✓ Fact {fact_num} matches!")
                return True, path + [f"Fact {fact_num}"]
        
        # Apply Rule 4: X can reach Y if X is connected to Y
        if "can reach" in current:
            # Extract X and Y from "X can reach Y"
            match = re.match(r"(.+) can reach (.+)", current)
            if match:
                x = match.group(1).strip()
                y = match.group(2).strip()
                
                # Rule 4: Need X is connected to Y
                new_goal = f"{x} is connected to {y}"
                if verbose:
                    print(f"  Rule 4: need '{new_goal}'")
                queue.append((new_goal, path + ["Rule 4"], depth + 1))
                
                # Rule 5: X can reach Z if X is connected to Y and Y can reach Z
                # Try each possible intermediate station
                for station in ["times square", "grand central", "bryant park"]:
                    if station not in x and station not in y:
                        goal1 = f"{x} is connected to {station}"
                        goal2 = f"{station} can reach {y}"
                        if verbose:
                            print(f"  Rule 5 (via {station}): need '{goal1}' and '{goal2}'")
                        
                        # Check first condition
                        queue.append((goal1, path + [f"Rule 5 via {station}"], depth + 1))
                        # If first succeeds, we'd need to check second
                        # For simplicity, add both to queue
                        queue.append((goal2, path + [f"Rule 5 via {station}"], depth + 1))
    
    if verbose:
        print(f"\n✗ Failed")
    return False, []

def test_metro_simplified():
    """Test with simplified approach"""
    print("="*60)
    print("SIMPLIFIED METRO NETWORK BFS")
    print("="*60)
    
    rules = """
Facts:
1. Union Square is connected to Times Square
2. Times Square is connected to Grand Central
3. Grand Central is connected to Bryant Park

Rules:
4. X can reach Y if X is connected to Y
5. X can reach Z if X is connected to Y and Y can reach Z
"""
    
    print(rules)
    
    tests = [
        "union square is connected to times square",
        "union square can reach times square",
        "union square can reach grand central",
        "times square can reach grand central",
        "union square can reach bryant park"
    ]
    
    for goal in tests:
        success, path = bfs_metro_simple(goal, rules, verbose=True)
        print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
        if path:
            print(f"Path: {' -> '.join(path)}")
        print()

if __name__ == "__main__":
    test_metro_simplified()


# In[1]:


"""
BFS Prolog Resolution with Standardization Apart
"""

import ollama
from collections import deque
import re

client = ollama.Client()
model = "gpt-oss:20b"

def ask_llm(prompt):
    resp = client.generate(model=model, prompt=prompt, options={'temperature': 0.0})
    answer = resp.get('response', '')
    if "...done thinking." in answer:
        return answer.split("...done thinking.")[-1].strip()
    return answer

def extract_answer(response):
    """Extract clean answer from response"""
    if "...done thinking." in response:
        return response.split("...done thinking.")[-1].strip()
    return response.strip()

def find_matching_rules(goal, knowledge_base):
    """Find ALL rules that can match a goal with standardization apart"""
    prompt = f"""Prolog Pattern Matching:

Goal to prove: {goal}

Knowledge Base:
{knowledge_base}

Which rules can unify with this goal?
- Standardize the rules and goals apart (rename variables to avoid conflicts)
- Facts match if they unify exactly
- Rules match if their head unifies with the goal
- Consider variable unification

List ONLY the rule numbers, separated by commas.
If NO rules match, write: NONE

Answer:"""
    
    response = ask_llm(prompt)
    answer = extract_answer(response)
    
    if "NONE" in answer.upper():
        return []
    
    numbers = re.findall(r'\d+', answer)
    return [int(n) for n in numbers]

def standardize_and_unify(goal, rule_head, rule_body):
    """Standardize rule apart and unify with goal"""
    prompt = f"""Prolog Resolution Step:

Goal: {goal}
Rule: {rule_head} :- {rule_body}

Step 1: Standardize the rule apart (rename variables to avoid conflicts with goal)
Example: If goal has X,Y and rule has X,Z, rename rule to use X1,Z1

Step 2: Unify the goal with the standardized rule head

Step 3: Apply the unification substitution to the standardized rule body

Write the resulting subgoals (one per line):"""
    
    response = ask_llm(prompt)
    answer = extract_answer(response)
    
    # Extract predicates from answer
    predicates = re.findall(r'[a-z_]+\([^)]+\)', answer)
    return predicates if predicates else None

def bfs_prolog_metro(goal, kb):
    """BFS for Prolog with proper standardization"""
    
    # Parse knowledge base
    facts = []
    rules = []
    
    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content = match.group(2).strip()
            
            if ':-' in content:
                head, body = content.split(':-')
                rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                facts.append((num, content.rstrip('.')))
    
    # BFS queue: (goals_list, path, depth)
    queue = deque([([goal], [], 0)])
    visited = set()
    max_depth = 10
    
    print(f"\nGoal: {goal}")
    print("-" * 40)
    
    while queue:
        goals, path, depth = queue.popleft()
        
        if not goals:
            print(f"✓ SUCCESS at depth {depth}")
            for step in path:
                print(f"  {step}")
            return True
        
        if depth >= max_depth:
            continue
        
        current = goals[0]
        remaining = goals[1:]
        
        # Check visited
        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)
        
        print(f"Depth {depth}: {current}")
        
        # Check facts
        fact_matched = False
        for num, fact in facts:
            if current == fact:
                print(f"  ✓ Fact {num} matches exactly")
                queue.append((remaining, path + [f"Fact {num}: {fact}"], depth + 1))
                fact_matched = True
                break
        
        if fact_matched:
            continue
        
        # Find matching rules
        matching_rules = find_matching_rules(current, kb)
        
        if not matching_rules:
            print(f"  No matching rules")
            continue
        
        print(f"  Matching rules: {matching_rules}")
        
        # Try each matching rule
        for rule_num in matching_rules:
            # Find the rule
            for num, head, body in rules:
                if num == rule_num:
                    print(f"  Trying rule {num}: {head} :- {body}")
                    
                    # Standardize apart and get subgoals
                    new_subgoals = standardize_and_unify(current, head, body)
                    
                    if new_subgoals:
                        print(f"    New subgoals: {new_subgoals}")
                        all_goals = new_subgoals + remaining
                        new_path = path + [f"Rule {num}: {current} → {new_subgoals}"]
                        queue.append((all_goals, new_path, depth + 1))
                    break
    
    print("✗ FAILED - No proof found")
    return False

# Metro test with Prolog rules
kb = """
1. connected(union_square, times_square).
2. connected(times_square, grand_central).
3. connected(grand_central, bryant_park).
4. reachable(X, Y) :- connected(X, Y).
5. reachable(X, Z) :- connected(X, Y), reachable(Y, Z).
"""

print("METRO BFS - PROLOG WITH STANDARDIZATION")
print("="*50)
print("Knowledge Base:")
print(kb)
print()

tests = [
    "connected(union_square, times_square)",
    "reachable(union_square, times_square)",
    "reachable(union_square, grand_central)",
    "reachable(union_square, bryant_park)"
]

for test in tests:
    result = bfs_prolog_metro(test, kb)
    print()


# In[1]:


"""
BFS Prolog with Proper Variable Binding
"""

import ollama
from collections import deque
import re

client = ollama.Client()
model = "gpt-oss:20b"

def ask_llm(prompt):
    resp = client.generate(model=model, prompt=prompt, options={'temperature': 0.0})
    answer = resp.get('response', '')
    if "...done thinking." in answer:
        return answer.split("...done thinking.")[-1].strip()
    return answer

def extract_answer(response):
    if "...done thinking." in response:
        return response.split("...done thinking.")[-1].strip()
    return response.strip()

def unify_with_fact(goal, fact):
    """Check if goal unifies with fact and return bindings"""
    prompt = f"""Prolog unification:
Goal: {goal}
Fact: {fact}

Can they unify? If yes, what are the variable bindings?
Answer: 
- If they unify, write bindings like: Y1=times_square
- If they don't unify, write: NO

Answer:"""
    
    response = ask_llm(prompt)
    answer = extract_answer(response)
    
    if "NO" in answer.upper():
        return None
    
    # Extract bindings
    bindings = {}
    matches = re.findall(r'([A-Z][A-Za-z0-9_]*)\s*=\s*([a-z_]+)', answer)
    for var, val in matches:
        bindings[var] = val
    
    return bindings if bindings else {}  # Empty dict means exact match

def apply_bindings_to_goals(remaining_goals, bindings):
    """Apply variable bindings to remaining goals"""
    if not bindings or not remaining_goals:
        return remaining_goals
    
    prompt = f"""Apply these bindings to the goals:
Bindings: {bindings}
Goals: {remaining_goals}

Write the instantiated goals (one per line):"""
    
    response = ask_llm(prompt)
    answer = extract_answer(response)
    
    # Extract predicates
    predicates = re.findall(r'[a-z_]+\([^)]+\)', answer)
    return predicates if predicates else remaining_goals

def find_matching_rules(goal, kb):
    """Find rules that can match a goal"""
    prompt = f"""Which rules can match this goal?
Goal: {goal}
Rules:
{kb}

List only the rule numbers that can unify with the goal.
Answer (just numbers separated by commas, or NONE):"""
    
    response = ask_llm(prompt)
    answer = extract_answer(response)
    
    if "NONE" in answer.upper():
        return []
    
    numbers = re.findall(r'\d+', answer)
    return [int(n) for n in numbers]

def get_rule_subgoals(goal, rule_head, rule_body):
    """Get subgoals after unifying goal with rule"""
    prompt = f"""SLD Resolution:
Goal: {goal}
Rule: {rule_head} :- {rule_body}

1. Unify goal with rule head (rename rule variables if needed)
2. Apply substitution to rule body
3. List the resulting subgoals

Write only the subgoals (one per line):"""
    
    response = ask_llm(prompt)
    answer = extract_answer(response)
    
    predicates = re.findall(r'[a-z_]+\([^)]+\)', answer)
    return predicates if predicates else None

def bfs_prolog_metro(goal, kb):
    """BFS with proper variable binding"""
    
    # Parse KB
    facts = []
    rules = []
    
    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content = match.group(2).strip()
            
            if ':-' in content:
                head, body = content.split(':-')
                rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                facts.append((num, content.rstrip('.')))
    
    # BFS queue: (current_goal, remaining_goals, path, depth)
    queue = deque([(goal, [], [], 0)])
    visited = set()
    max_depth = 10
    
    print(f"\nGoal: {goal}")
    print("-" * 40)
    
    while queue:
        current, remaining, path, depth = queue.popleft()
        
        if depth >= max_depth:
            continue
        
        # Check visited
        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)
        
        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")
        
        # Check facts
        for num, fact in facts:
            # Try to unify with fact
            bindings = unify_with_fact(current, fact)
            
            if bindings is not None:  # Unification succeeded
                print(f"  ✓ Fact {num}: {fact}")
                
                if bindings:  # Has variable bindings
                    print(f"    Bindings: {bindings}")
                    # Apply bindings to remaining goals
                    instantiated_remaining = apply_bindings_to_goals(remaining, bindings)
                    
                    if not instantiated_remaining:
                        print(f"✓ SUCCESS at depth {depth + 1}")
                        return True
                    
                    # Continue with instantiated remaining goals
                    next_goal = instantiated_remaining[0]
                    next_remaining = instantiated_remaining[1:]
                    queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))
                
                else:  # Exact match, no variables
                    if not remaining:
                        print(f"✓ SUCCESS at depth {depth + 1}")
                        return True
                    
                    # Continue with remaining goals
                    next_goal = remaining[0]
                    next_remaining = remaining[1:]
                    queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))
        
        # Try rules
        matching_rules = find_matching_rules(current, kb)
        
        if matching_rules:
            print(f"  Matching rules: {matching_rules}")
            
            for rule_num in matching_rules:
                for num, head, body in rules:
                    if num == rule_num:
                        subgoals = get_rule_subgoals(current, head, body)
                        
                        if subgoals:
                            print(f"  Rule {num}: → {subgoals}")
                            # Add subgoals before remaining goals
                            all_goals = subgoals + remaining
                            next_goal = all_goals[0]
                            next_remaining = all_goals[1:]
                            queue.append((next_goal, next_remaining, path + [f"Rule {num}"], depth + 1))
                        break
    
    print("✗ FAILED")
    return False

# Test
kb = """
1. connected(union_square, times_square).
2. connected(times_square, grand_central).
3. connected(grand_central, bryant_park).
4. reachable(X, Y) :- connected(X, Y).
5. reachable(X, Z) :- connected(X, Y), reachable(Y, Z).
"""

print("METRO BFS - PROLOG")
print("="*50)
print(kb)

tests = [
    "connected(union_square, times_square)",
    "reachable(union_square, times_square)",
    "reachable(union_square, grand_central)",
    "reachable(union_square, bryant_park)"
]

for test in tests:
    result = bfs_prolog_metro(test, kb)
    print()


# In[2]:


"""
Fixed BFS Prolog - Properly Distinguishes Facts from Rules
"""

import ollama
from collections import deque
import re

client = ollama.Client()
model = "gpt-oss:20b"

def ask_llm(prompt):
    resp = client.generate(model=model, prompt=prompt, options={'temperature': 0.0})
    answer = resp.get('response', '')
    if "...done thinking." in answer:
        return answer.split("...done thinking.")[-1].strip()
    return answer

def check_exact_match(goal, fact):
    """Check if goal matches fact exactly (no variables)"""
    return goal.strip() == fact.strip()

def unify_with_fact(goal, fact):
    """Check if goal unifies with fact and return bindings"""
    if check_exact_match(goal, fact):
        return {}  # Exact match, no bindings
    
    prompt = f"""Prolog unification:
Goal: {goal}
Fact: {fact}

Do they unify? If yes with variables, what are the bindings?
- If exact match (no variables): write EXACT
- If unify with variables: write bindings like Y=times_square
- If no unification: write NO

Answer:"""
    
    response = ask_llm(prompt)
    answer = response.strip()
    
    if "NO" in answer.upper():
        return None
    if "EXACT" in answer.upper():
        return {}
    
    # Extract bindings
    bindings = {}
    matches = re.findall(r'([A-Z][A-Za-z0-9_]*)\s*=\s*([a-z_]+)', answer)
    for var, val in matches:
        bindings[var] = val
    
    return bindings if bindings else None

def apply_bindings(goals, bindings):
    """Apply variable bindings to goals"""
    if not bindings or not goals:
        return goals
    
    prompt = f"""Apply bindings to goals:
Bindings: {bindings}
Goals: {goals}

Write the instantiated goals (one per line):"""
    
    response = ask_llm(prompt)
    predicates = re.findall(r'[a-z_]+\([^)]+\)', response)
    return predicates if predicates else goals

def find_matching_rules_only(goal, rules_list):
    """Find ONLY rules (not facts) that can match goal"""
    if not rules_list:
        return []
    
    rules_text = '\n'.join([f"{num}. {head} :- {body}" for num, head, body in rules_list])
    
    prompt = f"""Which RULES can match this goal?
Goal: {goal}

Rules (only consider these):
{rules_text}

List only the rule numbers that can unify with the goal.
Do NOT include facts. Only rules with :- 
Answer (numbers only, or NONE):"""
    
    response = ask_llm(prompt)
    
    if "NONE" in response.upper():
        return []
    
    numbers = re.findall(r'\d+', response)
    # Filter to only valid rule numbers
    valid_nums = [int(n) for n in numbers if any(int(n) == r[0] for r in rules_list)]
    return valid_nums

def get_subgoals(goal, rule_head, rule_body):
    """Get subgoals after unifying goal with rule"""
    prompt = f"""SLD Resolution:
Goal: {goal}
Rule: {rule_head} :- {rule_body}

1. Unify goal with head
2. Apply substitution to body
3. List resulting subgoals

Write only the subgoals (one per line):"""
    
    response = ask_llm(prompt)
    predicates = re.findall(r'[a-z_]+\([^)]+\)', response)
    return predicates if predicates else None

def bfs_prolog_metro(goal, kb):
    """BFS with correct fact/rule distinction"""
    
    # Parse KB - separate facts and rules
    facts = []
    rules = []
    
    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content = match.group(2).strip()
            
            if ':-' in content:
                head, body = content.split(':-')
                rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                facts.append((num, content.rstrip('.')))
    
    # BFS queue
    queue = deque([(goal, [], [], 0)])
    visited = set()
    max_depth = 10
    
    print(f"\nGoal: {goal}")
    print("-" * 40)
    
    while queue:
        current, remaining, path, depth = queue.popleft()
        
        if depth >= max_depth:
            continue
        
        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)
        
        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")
        
        # FIRST: Check for exact fact match
        fact_matched = False
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")
                
                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return True
                
                # Continue with remaining goals
                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))
                fact_matched = True
                break
        
        if fact_matched:
            continue
        
        # SECOND: Check for fact unification with variables
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)
            
            if bindings:  # Has variable bindings (not exact match)
                print(f"  ✓ Fact {num}: {fact}")
                print(f"    Bindings: {bindings}")
                
                # Apply bindings to remaining goals
                instantiated = apply_bindings(remaining, bindings)
                
                if not instantiated:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return True
                
                next_goal = instantiated[0]
                next_remaining = instantiated[1:]
                queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))
        
        # THIRD: Try rules
        matching_rules = find_matching_rules_only(current, rules)
        
        if matching_rules:
            print(f"  Matching rules: {matching_rules}")
            
            for rule_num in matching_rules:
                for num, head, body in rules:
                    if num == rule_num:
                        subgoals = get_subgoals(current, head, body)
                        
                        if subgoals:
                            print(f"  Rule {num}: → {subgoals}")
                            all_goals = subgoals + remaining
                            next_goal = all_goals[0]
                            next_remaining = all_goals[1:]
                            queue.append((next_goal, next_remaining, path + [f"Rule {num}"], depth + 1))
                        break
    
    print("✗ FAILED")
    return False

# Test
kb = """
1. connected(union_square, times_square).
2. connected(times_square, grand_central).
3. connected(grand_central, bryant_park).
4. reachable(X, Y) :- connected(X, Y).
5. reachable(X, Z) :- connected(X, Y), reachable(Y, Z).
"""

print("METRO BFS - PROLOG (FIXED)")
print("="*50)
print(kb)

tests = [
    "connected(union_square, times_square)",
    "reachable(union_square, times_square)",
    "reachable(union_square, grand_central)",
    "reachable(union_square, bryant_park)"
]

for test in tests:
    result = bfs_prolog_metro(test, kb)
    print()


# In[22]:


get_ipython().system('ollama run mistral')


# In[23]:


import ollama

client = ollama.Client()
model_name = "mistral:latest"
prompt = "hello, what model are you?"
response = client.generate(
        model=model_name,
        prompt=prompt,
        options={'temperature': 0.0}
    )
answer = response.get('response', '').strip()
if "...done thinking." in answer:
     print(answer.split("...done thinking.")[-1].strip())
print(answer)


# In[28]:


"""
Fixed BFS Prolog - Properly Distinguishes Facts from Rules
"""

import ollama
from collections import deque
import re

client = ollama.Client()
model = "mistral:latest"

def ask_llm(prompt):
    resp = client.generate(model=model, prompt=prompt, options={'temperature': 0.0})
    answer = resp.get('response', '')
    if "...done thinking." in answer:
        return answer.split("...done thinking.")[-1].strip()
    return answer

def check_exact_match(goal, fact):
    """Check if goal matches fact exactly (no variables)"""
    return goal.strip() == fact.strip()

def unify_with_fact(goal, fact):
    """Check if goal unifies with fact and return bindings"""
    if check_exact_match(goal, fact):
        return {}  # Exact match, no bindings
    
    prompt = f"""Prolog unification:
Goal: {goal}
Fact: {fact}

Do they unify? If yes with variables, what are the bindings?
- If exact match (no variables): write EXACT
- If unify with variables: write bindings like Y=times_square
- If no unification: write NO

Answer:"""
    
    response = ask_llm(prompt)
    answer = response.strip()
    
    if "NO" in answer.upper():
        return None
    if "EXACT" in answer.upper():
        return {}
    
    # Extract bindings
    bindings = {}
    matches = re.findall(r'([A-Z][A-Za-z0-9_]*)\s*=\s*([a-z_]+)', answer)
    for var, val in matches:
        bindings[var] = val
    
    return bindings if bindings else None

def apply_bindings(goals, bindings):
    """Apply variable bindings to goals"""
    if not bindings or not goals:
        return goals
    
    prompt = f"""Apply bindings to goals:
Bindings: {bindings}
Goals: {goals}

Write the instantiated goals (one per line):"""
    
    response = ask_llm(prompt)
    predicates = re.findall(r'[a-z_]+\([^)]+\)', response)
    return predicates if predicates else goals

def find_matching_rules_only(goal, rules_list):
    """Find ONLY rules (not facts) that can match goal"""
    if not rules_list:
        return []
    
    rules_text = '\n'.join([f"{num}. {head} :- {body}" for num, head, body in rules_list])
    
    prompt = f"""Which RULES can match this goal?
Goal: {goal}

Rules (only consider these):
{rules_text}

List only the rule numbers that can unify with the goal.
Do NOT include facts. Only rules with :- 
Answer (numbers only, or NONE):"""
    
    response = ask_llm(prompt)
    
    if "NONE" in response.upper():
        return []
    
    numbers = re.findall(r'\d+', response)
    # Filter to only valid rule numbers
    valid_nums = [int(n) for n in numbers if any(int(n) == r[0] for r in rules_list)]
    return valid_nums

def get_subgoals(goal, rule_head, rule_body):
    """Get subgoals after unifying goal with rule"""
    prompt = f"""SLD Resolution:
Goal: {goal}
Rule: {rule_head} :- {rule_body}

1. Unify goal with head
2. Apply substitution to body
3. List resulting subgoals

Write only the subgoals (one per line):"""
    
    response = ask_llm(prompt)
    predicates = re.findall(r'[a-z_]+\([^)]+\)', response)
    return predicates if predicates else None

def verify_subgoals(goal, rule_head, rule_body, subgoals):
    """Verify that the generated subgoals make sense"""
    prompt = f"""VERIFY SLD RESOLUTION

Original Goal: {goal}
Rule Used: {rule_head} :- {rule_body}
Generated Subgoals: {', '.join(subgoals)}

Check:
1. Do the subgoals make logical sense?
2. Are there any repeated variables that shouldn't be (like connected(Y,Y))?
3. Does the substitution look correct?

Answer YES if correct, or explain the error:"""
    
    response = ask_llm(prompt)
    return "YES" in response.upper()

def bfs_prolog_metro(goal, kb):
    """BFS with correct fact/rule distinction"""
    
    # Parse KB - separate facts and rules
    facts = []
    rules = []
    
    for line in kb_missing_3.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content = match.group(2).strip()
            
            if ':-' in content:
                head, body = content.split(':-')
                rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                facts.append((num, content.rstrip('.')))
    
    # BFS queue
    queue = deque([(goal, [], [], 0)])
    visited = set()
    max_depth = 10
    
    print(f"\nGoal: {goal}")
    print("-" * 40)
    
    while queue:
        current, remaining, path, depth = queue.popleft()
        
        if depth >= max_depth:
            continue
        
        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)
        
        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")
        
        # FIRST: Check for exact fact match
        fact_matched = False
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")
                
                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return True
                
                # Continue with remaining goals
                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))
                fact_matched = True
                break
        
        if fact_matched:
            continue
        
        # SECOND: Check for fact unification with variables
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)
            
            if bindings:  # Has variable bindings (not exact match)
                print(f"  ✓ Fact {num}: {fact}")
                print(f"    Bindings: {bindings}")
                
                # Apply bindings to remaining goals
                instantiated = apply_bindings(remaining, bindings)
                
                if not instantiated:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return True
                
                next_goal = instantiated[0]
                next_remaining = instantiated[1:]
                queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))
        
        # THIRD: Try rules
        matching_rules = find_matching_rules_only(current, rules)
        
        if matching_rules:
            print(f"  Matching rules: {matching_rules}")
            
            for rule_num in matching_rules:
                for num, head, body in rules:
                    if num == rule_num:
                        subgoals = get_subgoals(current, head, body)
                        
                        if subgoals:
                            print(f"  Rule {num}: → {subgoals}")
                            all_goals = subgoals + remaining
                            next_goal = all_goals[0]
                            next_remaining = all_goals[1:]
                            queue.append((next_goal, next_remaining, path + [f"Rule {num}"], depth + 1))
                        break
    
    print("✗ FAILED")
    return False

# Test
kb = """
1. connected(union_square, times_square).
2. connected(times_square, grand_central).
3. connected(grand_central, bryant_park).
4. reachable(X, Y) :- connected(X, Y).
5. reachable(X, Z) :- connected(X, Y), reachable(Y, Z).
"""

kb_missing_3 = """
1. connected(union_square, times_square).
2. connected(times_square, grand_central).
4. reachable(X, Y) :- connected(X, Y).
5. reachable(X, Z) :- connected(X, Y), reachable(Y, Z).
"""

print("METRO BFS - PROLOG (FIXED)")
print("="*50)
print(kb)

tests = [
    "connected(union_square, times_square)",
    "reachable(union_square, times_square)",
    "reachable(union_square, grand_central)",
    "reachable(union_square, bryant_park)"
]

for test in tests:
    result = bfs_prolog_metro(test, kb)
    print()


# # failed run

# In[13]:


"""
BFS Prolog SLD Resolution with LLM-backed Unification (JSON-structured for small models)
"""

import ollama
from collections import deque
import re
import json

# --- LLM setup ---

client = ollama.Client()

# Use whatever you want here:
# model = "gpt-oss:20b"
model = "gpt-oss:20b"  # for debugging Mistral 7B


def ask_llm(prompt: str) -> str:
    resp = client.generate(model=model, prompt=prompt, options={'temperature': 0.0})
    answer = resp.get('response', '')
    # Optional: strip off scratchpad if you use "...done thinking."
    if "...done thinking." in answer:
        return answer.split("...done thinking.")[-1].strip()
    return answer


# --- Helpers for parsing / JSON ---

def extract_first_json(text: str) -> str:
    """
    Extract the first {...} JSON object from possibly messy text.
    This makes the code robust to models that add a bit of extra junk.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in: {text!r}")
    return match.group(0)


# --- Core Prolog helpers ---

def check_exact_match(goal: str, fact: str) -> bool:
    """Check if goal matches fact exactly (no variables)."""
    return goal.strip() == fact.strip()


def unify_with_fact(goal: str, fact: str):
    """
    Check if goal unifies with fact and return bindings, using strict JSON output.

    Returns:
        None      -> NO unification
        {}        -> EXACT ground match (no variables)
        dict      -> bindings, e.g. {"Y": "times_square"}
    """
    # Cheap local check first
    if check_exact_match(goal, fact):
        return {}  # Exact match, no bindings

    prompt = f"""You are a STRICT Prolog unification engine.

Your ONLY job is to decide if the Prolog Goal unifies with the Prolog Fact.

Goal: {goal}
Fact: {fact}

Use ONLY the symbols that appear in Goal and Fact.
Do NOT invent new constants, variables, or predicates.
If you are uncertain, you MUST choose 'NO'.

Respond in EXACTLY ONE of these JSON formats, with NO extra text:

1) If they do NOT unify:
   {{ "result": "NO" }}

2) If they unify and there are NO variables (exact ground match):
   {{ "result": "EXACT" }}

3) If they unify and there ARE variables:
   {{ "result": "UNIFY", "bindings": {{"VarName1": "atom1", "VarName2": "atom2"}} }}

Rules:
- VarName keys MUST be exactly the variable names from Goal/Fact (uppercase or starting with uppercase).
- atom values MUST be lowercase_with_underscores atoms (Prolog atoms), but represented as JSON strings.
- NO explanation, NO prose. JSON ONLY.
"""

    response = ask_llm(prompt).strip()
    try:
        data = json.loads(extract_first_json(response))
    except Exception as e:
        print("Unify parse error:", e, "Raw:", response)
        return None

    res = str(data.get("result", "")).upper()
    if res == "NO":
        return None
    if res == "EXACT":
        return {}
    if res == "UNIFY":
        bindings = data.get("bindings", {})
        return bindings if bindings is not None else {}
    return None


def apply_bindings(goals, bindings):
    """
    Apply variable bindings to goals using the LLM and structured JSON.

    goals: list[str]   e.g. ["reachable(Y, Z)", "connected(Z, X)"]
    bindings: dict     e.g. {"Y": "times_square"}

    Returns: list[str] of instantiated goals.
    """
    if not bindings or not goals:
        return goals

    prompt = f"""You are a Prolog substitution engine.

Bindings (Python dict): {bindings}
Goals (list of Prolog goals): {goals}

Apply the bindings to EACH goal exactly as Prolog would:
- Replace each variable in the goals according to the bindings.
- Do NOT change predicate names.
- Do NOT add or remove goals.
- Do NOT introduce any new symbols.

Respond ONLY in this JSON format:

{{
  "goals": [
    "goal1(instantiated, here)",
    "goal2(...)", 
    ...
  ]
}}

If something is unclear, return the input goals unchanged in that JSON format.
NO explanation or extra text, JSON ONLY.
"""

    response = ask_llm(prompt).strip()
    try:
        data = json.loads(extract_first_json(response))
        new_goals = data.get("goals", [])
        # Sanity filter: only keep things that look like Prolog predicates
        preds = [g for g in new_goals if re.match(r'^[a-z_]+\([^)]*\)$', g)]
        return preds if preds else goals
    except Exception as e:
        print("Apply bindings parse error:", e, "Raw:", response)
        return goals


def find_matching_rules_only(goal, rules_list):
    """
    Find ONLY rules (not facts) whose HEAD can unify with the given goal.

    rules_list: list[(num, head, body)]
    Returns: list[int] of rule numbers.
    """
    if not rules_list:
        return []

    rules_text = '\n'.join([f"{num}. {head} :- {body}" for num, head, body in rules_list])

    prompt = f"""You are a Prolog rule matcher.

Goal: {goal}

Rules (numbered):
{rules_text}

Task:
Return the list of rule NUMBERS whose HEAD can unify with the Goal.
- Only consider rules shown above (they always contain ':-').
- Do NOT consider facts.
- Do NOT invent additional rules or modify heads.
- If you are uncertain, assume that a rule does NOT match.

Respond ONLY in this JSON format:

{{ "rules": [1, 3, 5] }}

If no rules match, respond:

{{ "rules": [] }}

No explanations, no extra text. JSON ONLY.
"""

    response = ask_llm(prompt).strip()
    try:
        data = json.loads(extract_first_json(response))
        numbers = data.get("rules", [])
        # Filter to only valid rule numbers present in rules_list
        valid_nums = [int(n) for n in numbers if any(int(n) == r[0] for r in rules_list)]
        return valid_nums
    except Exception as e:
        print("Rule match parse error:", e, "Raw:", response)
        return []


def get_subgoals(goal, rule_head, rule_body):
    """
    Get subgoals after unifying a goal with a rule, via JSON.

    Returns:
        list[str] subgoals, or None if rule cannot be applied.
    """
    prompt = f"""You are performing ONE step of SLD resolution in Prolog.

Goal: {goal}
Rule: {rule_head} :- {rule_body}

Steps:
1. Try to unify the Goal with the Rule head.
2. If unification FAILS, this rule CANNOT be used.
3. If unification SUCCEEDS, apply the most general unifier to the rule body.
4. Return the resulting subgoals (the instantiated body goals) in order.

Important constraints:
- Use ONLY the Goal and the given Rule. Do NOT use any other rules or facts.
- Do NOT invent new predicates, arguments, or constants.
- If unification fails OR you are unsure, treat it as failing.

Respond ONLY in one of these JSON forms:

a) If the rule CANNOT be used (no unification):
   {{ "subgoals": [] }}

b) If the rule CAN be used:
   {{
     "subgoals": [
       "first_subgoal(...)",
       "second_subgoal(...)",
       ...
     ]
   }}

No explanations or extra text. JSON ONLY.
"""

    response = ask_llm(prompt).strip()
    try:
        data = json.loads(extract_first_json(response))
        subs = data.get("subgoals", [])
        if not subs:
            return None
        preds = [g for g in subs if re.match(r'^[a-z_]+\([^)]*\)$', g)]
        return preds if preds else None
    except Exception as e:
        print("Subgoal parse error:", e, "Raw:", response)
        return None


"""
BFS Prolog SLD Resolution with LLM-backed Unification (JSON-structured for small models)
"""

import ollama
from collections import deque
import re
import json

# --- LLM setup ---

client = ollama.Client()

# Use whatever you want here:
# model = "gpt-oss:20b"
model = "gpt-oss:20b"  # for debugging Mistral 7B


def ask_llm(prompt: str) -> str:
    resp = client.generate(model=model, prompt=prompt, options={'temperature': 0.0})
    answer = resp.get('response', '')
    # Optional: strip off scratchpad if you use "...done thinking."
    if "...done thinking." in answer:
        return answer.split("...done thinking.")[-1].strip()
    return answer


# --- Helpers for parsing / JSON ---

def extract_first_json(text: str) -> str:
    """
    Extract the first {...} JSON object from possibly messy text.
    This makes the code robust to models that add a bit of extra junk.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in: {text!r}")
    return match.group(0)


# --- Core Prolog helpers ---

def check_exact_match(goal: str, fact: str) -> bool:
    """Check if goal matches fact exactly (no variables)."""
    return goal.strip() == fact.strip()


def unify_with_fact(goal: str, fact: str):
    """
    Check if goal unifies with fact and return bindings, using strict JSON output.

    Returns:
        None      -> NO unification
        {}        -> EXACT ground match (no variables)
        dict      -> bindings, e.g. {"Y": "times_square"}
    """
    # Cheap local check first
    if check_exact_match(goal, fact):
        return {}  # Exact match, no bindings

    prompt = f"""You are a STRICT Prolog unification engine.

Your ONLY job is to decide if the Prolog Goal unifies with the Prolog Fact.

Goal: {goal}
Fact: {fact}

Use ONLY the symbols that appear in Goal and Fact.
Do NOT invent new constants, variables, or predicates.
If you are uncertain, you MUST choose 'NO'.

Respond in EXACTLY ONE of these JSON formats, with NO extra text:

1) If they do NOT unify:
   {{ "result": "NO" }}

2) If they unify and there are NO variables (exact ground match):
   {{ "result": "EXACT" }}

3) If they unify and there ARE variables:
   {{ "result": "UNIFY", "bindings": {{"VarName1": "atom1", "VarName2": "atom2"}} }}

Rules:
- VarName keys MUST be exactly the variable names from Goal/Fact (uppercase or starting with uppercase).
- atom values MUST be lowercase_with_underscores atoms (Prolog atoms), but represented as JSON strings.
- NO explanation, NO prose. JSON ONLY.
"""

    response = ask_llm(prompt).strip()
    try:
        data = json.loads(extract_first_json(response))
    except Exception as e:
        print("Unify parse error:", e, "Raw:", response)
        return None

    res = str(data.get("result", "")).upper()
    if res == "NO":
        return None
    if res == "EXACT":
        return {}
    if res == "UNIFY":
        bindings = data.get("bindings", {})
        return bindings if bindings is not None else {}
    return None


def apply_bindings(goals, bindings):
    """
    Apply variable bindings to goals using the LLM and structured JSON.

    goals: list[str]   e.g. ["reachable(Y, Z)", "connected(Z, X)"]
    bindings: dict     e.g. {"Y": "times_square"}

    Returns: list[str] of instantiated goals.
    """
    if not bindings or not goals:
        return goals

    prompt = f"""You are a Prolog substitution engine.

Bindings (Python dict): {bindings}
Goals (list of Prolog goals): {goals}

Apply the bindings to EACH goal exactly as Prolog would:
- Replace each variable in the goals according to the bindings.
- Do NOT change predicate names.
- Do NOT add or remove goals.
- Do NOT introduce any new symbols.

Respond ONLY in this JSON format:

{{
  "goals": [
    "goal1(instantiated, here)",
    "goal2(...)", 
    ...
  ]
}}

If something is unclear, return the input goals unchanged in that JSON format.
NO explanation or extra text, JSON ONLY.
"""

    response = ask_llm(prompt).strip()
    try:
        data = json.loads(extract_first_json(response))
        new_goals = data.get("goals", [])
        # Sanity filter: only keep things that look like Prolog predicates
        preds = [g for g in new_goals if re.match(r'^[a-z_]+\([^)]*\)$', g)]
        return preds if preds else goals
    except Exception as e:
        print("Apply bindings parse error:", e, "Raw:", response)
        return goals


def find_matching_rules_only(goal, rules_list):
    """
    Find ONLY rules (not facts) whose HEAD can unify with the given goal.

    rules_list: list[(num, head, body)]
    Returns: list[int] of rule numbers.
    """
    if not rules_list:
        return []

    rules_text = '\n'.join([f"{num}. {head} :- {body}" for num, head, body in rules_list])

    prompt = f"""You are a Prolog rule matcher.

Goal: {goal}

Rules (numbered):
{rules_text}

Task:
Return the list of rule NUMBERS whose HEAD can unify with the Goal.
- Only consider rules shown above (they always contain ':-').
- Do NOT consider facts.
- Do NOT invent additional rules or modify heads.
- If you are uncertain, assume that a rule does NOT match.

Respond ONLY in this JSON format:

{{ "rules": [1, 3, 5] }}

If no rules match, respond:

{{ "rules": [] }}

No explanations, no extra text. JSON ONLY.
"""

    response = ask_llm(prompt).strip()
    try:
        data = json.loads(extract_first_json(response))
        numbers = data.get("rules", [])
        # Filter to only valid rule numbers present in rules_list
        valid_nums = [int(n) for n in numbers if any(int(n) == r[0] for r in rules_list)]
        return valid_nums
    except Exception as e:
        print("Rule match parse error:", e, "Raw:", response)
        return []


def get_subgoals(goal, rule_head, rule_body):
    """
    Get subgoals after unifying a goal with a rule, via JSON.

    Returns:
        list[str] subgoals, or None if rule cannot be applied.
    """
    prompt = f"""You are performing ONE step of SLD resolution in Prolog.

Goal: {goal}
Rule: {rule_head} :- {rule_body}

Steps:
1. Try to unify the Goal with the Rule head.
2. If unification FAILS, this rule CANNOT be used.
3. If unification SUCCEEDS, apply the most general unifier to the rule body.
4. Return the resulting subgoals (the instantiated body goals) in order.

Important constraints:
- Use ONLY the Goal and the given Rule. Do NOT use any other rules or facts.
- Do NOT invent new predicates, arguments, or constants.
- If unification fails OR you are unsure, treat it as failing.

Respond ONLY in one of these JSON forms:

a) If the rule CANNOT be used (no unification):
   {{ "subgoals": [] }}

b) If the rule CAN be used:
   {{
     "subgoals": [
       "first_subgoal(...)",
       "second_subgoal(...)",
       ...
     ]
   }}

No explanations or extra text. JSON ONLY.
"""

    response = ask_llm(prompt).strip()
    try:
        data = json.loads(extract_first_json(response))
        subs = data.get("subgoals", [])
        if not subs:
            return None
        preds = [g for g in subs if re.match(r'^[a-z_]+\([^)]*\)$', g)]
        return preds if preds else None
    except Exception as e:
        print("Subgoal parse error:", e, "Raw:", response)
        return None


# --- BFS Prolog engine ---

def bfs_prolog_metro(goal: str, kb: str, max_depth: int = 10) -> bool:
    """
    BFS with correct fact/rule distinction, using LLM to do:
    - unification with facts
    - applying bindings to remaining goals
    - selecting matching rules
    - generating instantiated subgoals
    """

    # Parse KB - separate facts and rules
    facts = []
    rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue j

        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content = match.group(2).strip()

            if ':-' in content:
                head, body = content.split(':-', 1)
                rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                facts.append((num, content.rstrip('.')))

    # BFS queue: (current_goal, remaining_goals, path, depth)
    queue = deque([(goal, [], [], 0)])
    visited = set()

    print(f"\nGoal: {goal}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        if depth >= max_depth:
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")

        # 1) FIRST: Check for exact fact match
        fact_matched = False
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")

                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return True

                # Continue with remaining goals
                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))
                fact_matched = True
                break

        if fact_matched:
            continue

        # 2) SECOND: Check for fact unification with variables (non-exact)
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)

            # None = no unification, {} could mean exact or unify without vars
            if bindings is None:
                continue

            # If unify_with_fact returned {} but the fact wasn't exact, it's still okay:
            print(f"  ✓ Fact {num}: {fact}")
            print(f"    Bindings: {bindings}")

            # Apply bindings to remaining goals
            instantiated = apply_bindings(remaining, bindings)

            if not instantiated:
                print(f"✓✓ SUCCESS at depth {depth + 1}")
                return True

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))

        # 3) THIRD: Try rules
        matching_rules = find_matching_rules_only(current, rules)

        if matching_rules:
            print(f"  Matching rules: {matching_rules}")

            for rule_num in matching_rules:
                for num, head, body in rules:
                    if num == rule_num:
                        subgoals = get_subgoals(current, head, body)

                        if subgoals:
                            print(f"  Rule {num}: → {subgoals}")
                            all_goals = subgoals + remaining
                            next_goal = all_goals[0]
                            next_remaining = all_goals[1:]
                            queue.append((next_goal, next_remaining, path + [f"Rule {num}"], depth + 1))
                        break

    print("✗ FAILED")
    return False


# --- Test ---

def omit_facts_from_kb(kb: str, omit_numbers):
    """
    Return a new KB string with any numbered lines in `omit_numbers` removed.
    `omit_numbers` should be a set/list of integers, e.g. {2, 3}.
    """
    omit_numbers = set(omit_numbers)
    new_lines = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        num = int(m.group(1))
        if num in omit_numbers:
            # skip this fact/rule
            continue
        new_lines.append(line)

    return '\n'.join(new_lines)
    
    
# --- BFS Prolog engine ---

def bfs_prolog_metro(goal: str, kb: str, max_depth: int = 10) -> bool:
    """
    BFS with correct fact/rule distinction, using LLM to do:
    - unification with facts
    - applying bindings to remaining goals
    - selecting matching rules
    - generating instantiated subgoals
    """

    # Parse KB - separate facts and rules
    facts = []
    rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content = match.group(2).strip()

            if ':-' in content:
                head, body = content.split(':-', 1)
                rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                facts.append((num, content.rstrip('.')))

    # BFS queue: (current_goal, remaining_goals, path, depth)
    queue = deque([(goal, [], [], 0)])
    visited = set()

    print(f"\nGoal: {goal}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        if depth >= max_depth:
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")

        # 1) FIRST: Check for exact fact match
        fact_matched = False
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")

                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return True

                # Continue with remaining goals
                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))
                fact_matched = True
                break

        if fact_matched:
            continue

        # 2) SECOND: Check for fact unification with variables (non-exact)
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)

            # None = no unification, {} could mean exact or unify without vars
            if bindings is None:
                continue

            # If unify_with_fact returned {} but the fact wasn't exact, it's still okay:
            print(f"  ✓ Fact {num}: {fact}")
            print(f"    Bindings: {bindings}")

            # Apply bindings to remaining goals
            instantiated = apply_bindings(remaining, bindings)

            if not instantiated:
                print(f"✓✓ SUCCESS at depth {depth + 1}")
                return True

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))

        # 3) THIRD: Try rules
        matching_rules = find_matching_rules_only(current, rules)

        if matching_rules:
            print(f"  Matching rules: {matching_rules}")

            for rule_num in matching_rules:
                for num, head, body in rules:
                    if num == rule_num:
                        subgoals = get_subgoals(current, head, body)

                        if subgoals:
                            print(f"  Rule {num}: → {subgoals}")
                            all_goals = subgoals + remaining
                            next_goal = all_goals[0]
                            next_remaining = all_goals[1:]
                            queue.append((next_goal, next_remaining, path + [f"Rule {num}"], depth + 1))
                        break

    print("✗ FAILED")
    return False


# --- Test ---

kb = """
1. connected(union_square, times_square).
2. connected(times_square, grand_central).
3. connected(grand_central, bryant_park).
4. reachable(X, Y) :- connected(X, Y).
5. reachable(X, Z) :- connected(X, Y), reachable(Y, Z).
"""

kb_missing_2 = omit_facts_from_kb(kb, omit_numbers={2})

print("=== KB with fact 2 removed ===")
print(kb_missing_2)

tests = [
    "reachable(union_square, times_square)",
    "reachable(union_square, grand_central)",
    "reachable(union_square, bryant_park)"
]

for t in tests:
    print(f"--- Test: {t} with missing fact 2 ---")
    bfs_prolog_metro(t, kb_missing_2)
    print()


# In[35]:


get_ipython().system('ollama run llama3.1:8b')


# In[30]:


get_ipython().system('ollama run qwen:14b')


# In[34]:


"""
BFS Prolog SLD Resolution with LLM-backed Unification (JSON-structured + functor checks)
"""

import ollama
from collections import deque
import re
import json
from typing import Optional

# --- Config / LLM setup ---

client = ollama.Client()

model = "gpt-oss:20b"
# model = "qwen:14b"

DEBUG = False  # set to True to print raw LLM outputs for debugging


def ask_llm(prompt: str) -> str:
    resp = client.generate(model=model, prompt=prompt, options={'temperature': 0.0})
    answer = resp.get('response', '')
    if "...done thinking." in answer:
        return answer.split("...done thinking.")[-1].strip()
    return answer


# --- Helpers for parsing / JSON ---

def extract_first_json(text: str) -> str:
    """
    Extract the first {...} JSON object from possibly messy text.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in: {text!r}")
    return match.group(0)


def parse_predicate(term: str):
    """
    Parse a simple Prolog predicate of the form:
        functor(arg1, arg2, ...)
    Returns (functor: str, args: list[str]) or None if parsing fails.
    """
    term = term.strip().rstrip('.')
    m = re.match(r'^([a-z_][a-zA-Z0-9_]*)\((.*)\)$', term)
    if not m:
        return None
    functor = m.group(1)
    args_raw = m.group(2).strip()
    if not args_raw:
        args = []
    else:
        # Simple arg split (no nested terms in this toy domain)
        args = [a.strip() for a in args_raw.split(',')]
    return functor, args


def is_variable(s: str) -> bool:
    """
    Prolog-ish variable check: starts with uppercase letter or '_'.
    """
    s = s.strip()
    return bool(s) and (s[0].isupper() or s[0] == '_')


# --- Core Prolog helpers ---

def check_exact_match(goal: str, fact: str) -> bool:
    """Check if goal matches fact exactly (no variables)."""
    return goal.strip().rstrip('.') == fact.strip().rstrip('.')


def unify_with_fact(goal: str, fact: str):
    """
    Check if goal unifies with fact and return bindings, using strict JSON output.

    Returns:
        None      -> NO unification
        {}        -> EXACT ground match (no variables)
        dict      -> bindings, e.g. {"Y": "times_square"}
    """

    # Quick syntactic predicate check
    parsed_goal = parse_predicate(goal)
    parsed_fact = parse_predicate(fact)
    if parsed_goal is None or parsed_fact is None:
        return None

    fun_g, args_g = parsed_goal
    fun_f, args_f = parsed_fact

    # Predicate name or arity mismatch => no unification
    if fun_g != fun_f or len(args_g) != len(args_f):
        return None

    # Cheap string equality check first
    if check_exact_match(goal, fact):
        return {}  # Exact match, no bindings

    prompt = f"""You are a STRICT Prolog unification engine.

Your ONLY job is to decide if the Prolog Goal unifies with the Prolog Fact.

Goal: {goal}
Fact: {fact}

Predicate names and arity are already known to MATCH.
Use ONLY the symbols that appear in Goal and Fact.
Do NOT invent new constants, variables, or predicates.
If you are uncertain, you MUST choose 'NO'.

Respond in EXACTLY ONE of these JSON formats, with NO extra text:

1) If they do NOT unify:
   {{ "result": "NO" }}

2) If they unify and there are NO variables (exact ground match):
   {{ "result": "EXACT" }}

3) If they unify and there ARE variables:
   {{ "result": "UNIFY", "bindings": {{"VarName1": "atom1", "VarName2": "atom2"}} }}

Rules:
- VarName keys MUST be exactly the variable names from Goal/Fact (uppercase or starting with uppercase).
- atom values MUST be lowercase_with_underscores atoms (Prolog atoms), but represented as JSON strings.
- NO explanation, NO prose. JSON ONLY.
"""

    response = ask_llm(prompt).strip()
    if DEBUG:
        print("\n[DEBUG unify_with_fact]")
        print("Prompt:\n", prompt)
        print("Raw response:\n", response)

    try:
        data = json.loads(extract_first_json(response))
    except Exception as e:
        print("Unify parse error:", e, "Raw:", response)
        return None

    res = str(data.get("result", "")).upper()
    if res == "NO":
        return None
    if res == "EXACT":
        return {}
    if res == "UNIFY":
        bindings = data.get("bindings", {})
        return bindings if bindings is not None else {}
    return None


def apply_bindings(goals, bindings):
    """
    Apply variable bindings to goals using the LLM and structured JSON.

    goals: list[str]   e.g. ["reachable(Y, Z)", "connected(Z, X)"]
    bindings: dict     e.g. {"Y": "times_square"}

    Returns: list[str] of instantiated goals.
    """
    if not bindings or not goals:
        return goals

    prompt = f"""You are a Prolog substitution engine.

Bindings (Python dict): {bindings}
Goals (list of Prolog goals): {goals}

Apply the bindings to EACH goal exactly as Prolog would:
- Replace each variable in the goals according to the bindings.
- Do NOT change predicate names.
- Do NOT add or remove goals.
- Do NOT introduce any new symbols.

Respond ONLY in this JSON format:

{{
  "goals": [
    "goal1(instantiated, here)",
    "goal2(...)", 
    ...
  ]
}}

If something is unclear, return the input goals unchanged in that JSON format.
NO explanation or extra text, JSON ONLY.
"""

    response = ask_llm(prompt).strip()
    if DEBUG:
        print("\n[DEBUG apply_bindings]")
        print("Prompt:\n", prompt)
        print("Raw response:\n", response)

    try:
        data = json.loads(extract_first_json(response))
        new_goals = data.get("goals", [])
        preds = [g for g in new_goals if re.match(r'^[a-z_]+\([^)]*\)$', g)]
        return preds if preds else goals
    except Exception as e:
        print("Apply bindings parse error:", e, "Raw:", response)
        return goals


def find_matching_rules_only(goal, rules_list):
    """
    Find ONLY rules (not facts) whose HEAD can unify with the given goal.

    IMPORTANT: This version is purely syntactic: it only checks functor and arity.
    We do NOT ask the LLM here, to avoid mismatched heads like reachable/2
    being applied to connected/2 goals.

    rules_list: list[(num, head, body)]
    Returns: list[int] of rule numbers.
    """
    parsed_goal = parse_predicate(goal)
    if parsed_goal is None:
        return []
    fun_g, args_g = parsed_goal
    arity_g = len(args_g)

    matching = []
    for num, head, body in rules_list:
        parsed_head = parse_predicate(head)
        if parsed_head is None:
            continue
        fun_h, args_h = parsed_head
        if fun_h == fun_g and len(args_h) == arity_g:
            matching.append(num)
    return matching


def get_subgoals(goal, rule_head, rule_body):
    """
    Get subgoals after unifying a goal with a rule, via JSON.

    Returns:
        list[str] subgoals, or None if rule cannot be applied.
    """
    prompt = f"""You are performing ONE step of SLD resolution in Prolog.

Goal: {goal}
Rule: {rule_head} :- {rule_body}

Steps:
1. Try to unify the Goal with the Rule head.
2. If unification FAILS, this rule CANNOT be used.
3. If unification SUCCEEDS, apply the most general unifier to the rule body.
4. Return the resulting subgoals (the instantiated body goals) in order.

Important constraints:
- Predicate name and arity of Goal and Rule head are already known to MATCH.
- Use ONLY the Goal and the given Rule. Do NOT use any other rules or facts.
- Do NOT invent new predicates, arguments, or constants.
- If unification fails OR you are unsure, treat it as failing.

Respond ONLY in one of these JSON forms:

a) If the rule CANNOT be used (no unification):
   {{ "subgoals": [] }}

b) If the rule CAN be used:
   {{
     "subgoals": [
       "first_subgoal(...)",
       "second_subgoal(...)",
       ...
     ]
   }}

No explanations or extra text. JSON ONLY.
"""

    response = ask_llm(prompt).strip()
    if DEBUG:
        print("\n[DEBUG get_subgoals]")
        print("Prompt:\n", prompt)
        print("Raw response:\n", response)

    try:
        data = json.loads(extract_first_json(response))
        subs = data.get("subgoals", [])
        if not subs:
            return None
        preds = [g for g in subs if re.match(r'^[a-z_]+\([^)]*\)$', g)]
        return preds if preds else None
    except Exception as e:
        print("Subgoal parse error:", e, "Raw:", response)
        return None


# --- BFS Prolog engine ---

def bfs_prolog_metro(goal: str, kb: str, max_depth: int = 10) -> bool:
    """
    BFS with correct fact/rule distinction, using LLM to do:
    - unification with facts (but guarded by functor/arity)
    - applying bindings to remaining goals
    - generating instantiated subgoals
    """

    # Parse KB - separate facts and rules
    facts = []
    rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content = match.group(2).strip()

            if ':-' in content:
                head, body = content.split(':-', 1)
                rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                facts.append((num, content.rstrip('.')))

    # BFS queue: (current_goal, remaining_goals, path, depth)
    queue = deque([(goal, [], [], 0)])
    visited = set()

    print(f"\nGoal: {goal}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        if depth >= max_depth:
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")

        # 1) FIRST: Check for exact fact match
        fact_matched = False
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")

                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return True

                # Continue with remaining goals
                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))
                fact_matched = True
                break

        if fact_matched:
            continue

        # 2) SECOND: Check for fact unification with variables (non-exact)
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)

            if bindings is None:
                continue

            print(f"  ✓ Fact {num}: {fact}")
            print(f"    Bindings: {bindings}")

            # Apply bindings to remaining goals
            instantiated = apply_bindings(remaining, bindings)

            if not instantiated:
                print(f"✓✓ SUCCESS at depth {depth + 1}")
                return True

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))

        # 3) THIRD: Try rules (matching by functor & arity only)
        matching_rules = find_matching_rules_only(current, rules)

        if matching_rules:
            print(f"  Matching rules: {matching_rules}")

            for rule_num in matching_rules:
                for num, head, body in rules:
                    if num == rule_num:
                        subgoals = get_subgoals(current, head, body)

                        if subgoals:
                            print(f"  Rule {num}: → {subgoals}")
                            all_goals = subgoals + remaining
                            next_goal = all_goals[0]
                            next_remaining = all_goals[1:]
                            queue.append((next_goal, next_remaining, path + [f"Rule {num}"], depth + 1))
                        break

    print("✗ FAILED")
    return False

def bfs_prolog_collect(goal: str, kb: str, max_depth: int = 10):
    """
    Near-clone of bfs_prolog_metro, but:

    - Still uses unify_with_fact → LLM
    - Still uses apply_bindings → LLM
    - Still uses get_subgoals → LLM
    - Still uses functor/arity filtering for rules
    - Still prints identical debug output

    Differences:
        • Returns a dict:
            {
              "success": bool,
              "proof_path": [...],
              "unresolved_atoms": set()
            }

        • Tracks unresolved atoms when:
            - no exact fact matched
            - no fact unified
            - no rule applied
    """

    # --- Parse KB exactly like bfs_prolog_metro ---
    facts = []
    rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content = match.group(2).strip()

            if ':-' in content:
                head, body = content.split(':-', 1)
                rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                facts.append((num, content.rstrip('.')))

    # BFS state: (current_goal, remaining_goals, path, depth)
    queue = deque([(goal, [], [], 0)])
    visited = set()
    unresolved_atoms = set()

    print(f"\n[COLLECT] Goal: {goal}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        if depth >= max_depth:
            unresolved_atoms.add(current)
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")

        # Track whether ANY progress was made on this goal
        progress = False

        # --- 1) Exact fact match ---
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")

                step = f"Fact {num}"
                new_path = path + [step]

                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return {
                        "success": True,
                        "proof_path": new_path,
                        "unresolved_atoms": set()
                    }

                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, new_path, depth + 1))

                progress = True
                break

        if progress:
            continue

        # --- 2) Fact unification (LLM) ---
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            progress = True

            print(f"  ✓ Fact {num}: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)

            step = f"Fact {num}"
            new_path = path + [step]

            if not instantiated:
                print(f"✓✓ SUCCESS at depth {depth + 1}")
                return {
                    "success": True,
                    "proof_path": new_path,
                    "unresolved_atoms": set()
                }

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, new_path, depth + 1))

        if progress:
            continue

        # --- 3) Rule attempts (LLM SLD step) ---
        matching_rules = find_matching_rules_only(current, rules)

        if matching_rules:
            print(f"  Matching rules: {matching_rules}")

        for rule_num in matching_rules:
            for num, head, body in rules:
                if num == rule_num:
                    subgoals = get_subgoals(current, head, body)
                    if subgoals:
                        print(f"  Rule {num}: → {subgoals}")
                        progress = True
                        all_goals = subgoals + remaining
                        next_goal = all_goals[0]
                        next_remaining = all_goals[1:]
                        step = f"Rule {num}"
                        new_path = path + [step]
                        queue.append((next_goal, next_remaining, new_path, depth + 1))
                    break

        # --- If absolutely nothing worked, record unresolved ---
        if not progress:
            print(f"  ✗ No facts or rules apply to: {current}")
            unresolved_atoms.add(current)

    # BFS fully exhausted → failure
    print("✗ FAILED (collect mode)")
    return {
        "success": False,
        "proof_path": [],
        "unresolved_atoms": unresolved_atoms
    }


import re
import json

def generate_background_hypotheses(goal: str, kb: str, unresolved_atoms, max_atoms: int = 5):
    """
    Use the LLM's background/world knowledge to propose additional Prolog clauses
    (facts or rules) that might make the GOAL provable.

    ALWAYS returns a list (possibly empty).
    NEVER returns None.
    """

    hypotheses = []

    # --- 1) Filter unresolved atoms to simple, ground atoms only ---
    # e.g. keep "connected(grand_central, bryant_park)" but drop "connected(grand_central, Y)"
    atom_list = list(unresolved_atoms)
    ground_atoms = []
    for atom in atom_list:
        atom = atom.strip()
        if not atom:
            continue
        # must look like functor(...)
        if '(' not in atom or ')' not in atom:
            continue
        # crude check: if any token inside parens starts with uppercase or '_', treat as variable → skip
        inside = atom.split('(', 1)[1].rsplit(')', 1)[0]
        if re.search(r'\b[A-Z_]\w*\b', inside):
            continue
        ground_atoms.append(atom)

    if max_atoms is not None and len(ground_atoms) > max_atoms:
        ground_atoms = ground_atoms[:max_atoms]

    if not ground_atoms:
        print("[generate_background_hypotheses] No suitable ground atoms to query.")
        return []  # <--- explicit list, not None

    # --- 2) For each ground unresolved atom, ask the LLM for hypotheses ---
    for atom in ground_atoms:
        prompt = f"""
You are a cautious Prolog expert with access to real-world background knowledge.

We attempted to prove the following GOAL using ONLY the numbered Prolog
knowledge base given below:

GOAL:
  {goal}

KNOWLEDGE BASE (numbered clauses):
{kb}

During breadth-first SLD resolution, the proof FAILED because we could not
prove the following subgoal:
  {atom}

Task:
Propose a SMALL set of additional Prolog clauses (facts or rules) that are
LIKELY to be true in the intended domain and that would help make the GOAL
provable. Think of these as "missing" facts or rules that could fill gaps in
the knowledge base.

Constraints:
- Each clause MUST be valid Prolog and MUST end with a period.
- You MUST NOT modify or delete any existing clauses in the KB.
- Use ONLY predicate names and arities that are compatible with the style of
  the existing KB (for example, connected/2, reachable/2, etc.).
- If you are uncertain about a clause, give it a lower confidence.
- You SHOULD prefer to return at least one plausible clause rather than an empty list.

Respond ONLY in this JSON format:

{{
  "hypotheses": [
    {{
      "clause": "connected(times_square, grand_central).",
      "confidence": 0.9
    }},
    {{
      "clause": "connected(times_square, bryant_park).",
      "confidence": 0.4
    }}
  ]
}}

If you truly have NO hypotheses, respond with:
{{ "hypotheses": [] }}
"""

        raw = ask_llm(prompt).strip()
        if DEBUG:
            print("\n[DEBUG generate_background_hypotheses]")
            print("Unresolved atom:", atom)
            print("Raw response:\n", raw)

        try:
            data = json.loads(extract_first_json(raw))
        except Exception as e:
            print("[generate_background_hypotheses] JSON parse error:", e)
            print("Raw LLM output:", raw)
            continue  # skip this atom, move on

        raw_hyps = data.get("hypotheses", [])
        if not isinstance(raw_hyps, list):
            print("[generate_background_hypotheses] 'hypotheses' not a list:", raw_hyps)
            continue

        for h in raw_hyps:
            clause = (h.get("clause") or "").strip()
            if not clause:
                continue

            # Normalize: ensure clause ends with a dot
            if not clause.endswith('.'):
                clause = clause + "."

            try:
                conf = float(h.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0

            hypotheses.append({
                "clause": clause,
                "confidence": conf,
                "from_atom": atom
            })

    # --- 3) Deduplicate clauses (keep highest-confidence version) ---
    dedup = {}
    for h in hypotheses:
        key = h["clause"]
        if key not in dedup or h["confidence"] > dedup[key]["confidence"]:
            dedup[key] = h

    return list(dedup.values())  # <--- ALWAYS returns a list

def _find_max_line_number_in_kb(kb: str) -> int:
    """
    Scan the numbered KB and return the maximum clause number seen.
    If no numbered lines are found, return 0.
    """
    max_num = 0
    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue
        num = int(m.group(1))
        if num > max_num:
            max_num = num
    return max_num


def _is_fact_clause(clause: str) -> bool:
    """
    Heuristic: a fact has no ':-' in it.
    Example: 'connected(times_square, grand_central).'
    """
    return ':-' not in clause


def _split_rule_clause(clause: str):
    """
    Split a rule clause 'Head :- Body.' into (head, body_str).

    Returns:
        head: 'reachable(X, Z)'
        body_str: 'connected(X, Y), reachable(Y, Z)'
    """
    # Remove trailing dot if present
    clause = clause.strip()
    if clause.endswith('.'):
        clause = clause[:-1]

    head_part, body_part = clause.split(':-', 1)
    head = head_part.strip()
    body_str = body_part.strip()
    return head, body_str


def attach_hypotheses_to_kb(kb: str, hypotheses):
    """
    Convert LLM-generated hypotheses into a 'soft KB' structure.

    Inputs:
        kb          : original numbered KB string
        hypotheses  : list[dict] from generate_background_hypotheses, where each dict has:
                        {
                          "clause": "prolog_clause_string_ending_with_dot.",
                          "confidence": float,
                          "from_atom": "unresolved atom"
                        }

    Returns:
        soft_kb: dict with two lists:
            {
              "facts": [
                  (num, atom_str, confidence),
                  ...
              ],
              "rules": [
                  (num, head_str, body_str, confidence),
                  ...
              ]
            }

        where:
          - num is a fresh line number (beyond any in the original KB),
          - atom_str is like "connected(times_square, grand_central)",
          - head_str is like "reachable(X, Z)",
          - body_str is like "connected(X, Y), reachable(Y, Z)".
    """

    soft_facts = []
    soft_rules = []

    # Start numbering hypotheses after the max existing KB line number
    max_num = _find_max_line_number_in_kb(kb)
    next_num = max_num + 1

    for h in hypotheses:
        clause = (h.get("clause") or "").strip()
        if not clause:
            continue

        conf = float(h.get("confidence", 0.0))

        # Normalize: ensure terminating dot
        if not clause.endswith('.'):
            clause = clause + '.'

        if _is_fact_clause(clause):
            # Example: "connected(times_square, grand_central)."
            atom = clause.rstrip('.').strip()
            soft_facts.append((next_num, atom, conf))
        else:
            # Example: "reachable(X, Z) :- connected(X, Y), reachable(Y, Z)."
            head, body_str = _split_rule_clause(clause)
            soft_rules.append((next_num, head, body_str, conf))

        next_num += 1

    soft_kb = {
        "facts": soft_facts,
        "rules": soft_rules,
    }
    return soft_kb

def bfs_prolog_metro_soft(
    goal: str,
    kb: str,
    soft_kb,
    max_depth: int = 10,
    max_soft: Optional[int] = None,
):
    """
    BFS SLD resolution that can use:
      - hard clauses from the original KB
      - soft clauses (hypotheses) from soft_kb

    soft_kb is the dict from attach_hypotheses_to_kb:
        {
          "facts": [(num, atom_str, confidence), ...],
          "rules": [(num, head_str, body_str, confidence), ...]
        }

    Returns a dict:
        {
          "success": bool,
          "proof_path": list,             # list of step labels
          "used_soft_clauses": list,      # list of (kind, num, confidence)
          "soft_cost": int,               # how many soft clauses used
          "min_conf": float | None        # min confidence over used soft clauses
        }
    """

    # --- Parse hard KB exactly like bfs_prolog_metro ---
    hard_facts = []
    hard_rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content = match.group(2).strip()

            if ':-' in content:
                head, body = content.split(':-', 1)
                hard_rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                hard_facts.append((num, content.rstrip('.')))

    # --- Soft KB unpack ---
    soft_facts = soft_kb.get("facts", [])  # list of (num, atom, conf)
    soft_rules = soft_kb.get("rules", [])  # list of (num, head, body_str, conf)

    # For rule matching, we want (num, head, body_str) lists
    soft_rules_for_match = [(num, head, body_str) for (num, head, body_str, conf) in soft_rules]

    # BFS queue: (current_goal, remaining_goals, path, depth, soft_cost, min_conf)
    #   path is a list of human-readable step labels (strings) like:
    #     "HardFact 1", "SoftFact 1001 (conf=0.90)", "HardRule 5", etc.
    queue = deque([(goal, [], [], 0, 0, 1.0)])  # min_conf starts at 1.0
    visited = set()

    print(f"\n[SOFT BFS] Goal: {goal}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth, soft_cost, min_conf = queue.popleft()

        if depth >= max_depth:
            continue

        # You can choose how much of state goes into 'visited'.
        # Here we include soft_cost to distinguish "cheaper" vs "more expensive" paths.
        state = (current, tuple(remaining), soft_cost)
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        print(f"  Soft cost: {soft_cost}, min_conf: {min_conf:.3f}")
        if remaining:
            print(f"  Remaining: {remaining}")

        # Helper to finalize success
        def make_success_result(final_path, final_soft_cost, final_min_conf):
            used_soft = []
            for step in final_path:
                if step.startswith("SoftFact"):
                    # step format: SoftFact <num> (conf=...)
                    parts = step.split()
                    if len(parts) >= 2:
                        try:
                            num = int(parts[1])
                        except ValueError:
                            num = None
                        used_soft.append(("fact", num))
                elif step.startswith("SoftRule"):
                    # step format: SoftRule <num> (conf=...)
                    parts = step.split()
                    if len(parts) >= 2:
                        try:
                            num = int(parts[1])
                        except ValueError:
                            num = None
                        used_soft.append(("rule", num))

            return {
                "success": True,
                "proof_path": final_path,
                "used_soft_clauses": used_soft,
                "soft_cost": final_soft_cost,
                "min_conf": final_min_conf if final_soft_cost > 0 else None
            }

        # --- 1) HARD facts: exact match ---
        for num, fact in hard_facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Hard Fact {num} matches exactly: {fact}")

                step_label = f"HardFact {num}"
                new_path = path + [step_label]

                if not remaining:
                    print(f"✓✓ SOFT-BFS SUCCESS (hard-only) at depth {depth + 1}")
                    return make_success_result(new_path, soft_cost, min_conf)

                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, new_path,
                              depth + 1, soft_cost, min_conf))
                # Since we found a hard fact, we can move to next BFS state
                # (We do not 'continue' here to allow exploring other options in this node too,
                #  but you could short-circuit if you want strict BFS semantics.)
                break

        # --- 2) HARD facts: unification with variables (LLM) ---
        for num, fact in hard_facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            print(f"  ✓ Hard Fact {num} unifies: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)

            step_label = f"HardFact {num}"
            new_path = path + [step_label]

            if not instantiated:
                print(f"✓✓ SOFT-BFS SUCCESS (hard-only) at depth {depth + 1}")
                return make_success_result(new_path, soft_cost, min_conf)

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, new_path,
                          depth + 1, soft_cost, min_conf))

        # --- 3) HARD rules (via SLD, LLM-backed) ---
        matching_hard_rules = find_matching_rules_only(current, hard_rules)

        if matching_hard_rules:
            print(f"  Matching hard rules: {matching_hard_rules}")

        for rule_num in matching_hard_rules:
            for num, head, body in hard_rules:
                if num != rule_num:
                    continue

                subgoals = get_subgoals(current, head, body)
                if not subgoals:
                    continue

                print(f"  Hard Rule {num}: {head} :- {body}")
                print(f"    → {subgoals}")

                all_goals = subgoals + remaining
                next_goal = all_goals[0]
                next_remaining = all_goals[1:]
                step_label = f"HardRule {num}"
                new_path = path + [step_label]

                queue.append((next_goal, next_remaining, new_path,
                              depth + 1, soft_cost, min_conf))
                break

        # --- 4) SOFT facts (hypotheses) ---
        for s_num, s_atom, s_conf in soft_facts:
            # If there's a limit on how many soft clauses we can use, enforce it
            if max_soft is not None and soft_cost >= max_soft:
                break

            bindings = unify_with_fact(current, s_atom)
            if bindings is None:
                continue

            new_soft_cost = soft_cost + 1
            new_min_conf = min(min_conf, s_conf)

            print(f"  ✓ Soft Fact {s_num} unifies: {s_atom}")
            print(f"    Bindings: {bindings}, conf={s_conf:.3f}")
            print(f"    New soft cost: {new_soft_cost}, new min_conf: {new_min_conf:.3f}")

            instantiated = apply_bindings(remaining, bindings)

            step_label = f"SoftFact {s_num} (conf={s_conf:.3f})"
            new_path = path + [step_label]

            if not instantiated:
                print(f"✓✓ SOFT-BFS SUCCESS (with soft facts) at depth {depth + 1}")
                return make_success_result(new_path, new_soft_cost, new_min_conf)

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, new_path,
                          depth + 1, new_soft_cost, new_min_conf))

        # --- 5) SOFT rules (hypotheses) ---
        # Use the syntactic matcher on the (num, head, body_str) projection
        matching_soft_rules = find_matching_rules_only(current, soft_rules_for_match)

        if matching_soft_rules:
            print(f"  Matching soft rules: {matching_soft_rules}")

        for rule_num in matching_soft_rules:
            # Enforce soft clause budget
            if max_soft is not None and soft_cost >= max_soft:
                break

            # Find full soft rule (with confidence)
            for s_num, s_head, s_body_str, s_conf in soft_rules:
                if s_num != rule_num:
                    continue

                subgoals = get_subgoals(current, s_head, s_body_str)
                if not subgoals:
                    continue

                new_soft_cost = soft_cost + 1
                new_min_conf = min(min_conf, s_conf)

                print(f"  Soft Rule {s_num}: {s_head} :- {s_body_str}")
                print(f"    → {subgoals}, conf={s_conf:.3f}")
                print(f"    New soft cost: {new_soft_cost}, new min_conf: {new_min_conf:.3f}")

                all_goals = subgoals + remaining
                next_goal = all_goals[0]
                next_remaining = all_goals[1:]
                step_label = f"SoftRule {s_num} (conf={s_conf:.3f})"
                new_path = path + [step_label]

                queue.append((next_goal, next_remaining, new_path,
                              depth + 1, new_soft_cost, new_min_conf))
                break

    # If we exhaust the queue, no proof was found even with soft clauses
    print("✗ SOFT-BFS FAILED (no proof found even with soft KB)")
    return {
        "success": False,
        "proof_path": [],
        "used_soft_clauses": [],
        "soft_cost": None,
        "min_conf": None
    }

def solve_with_background(
    goal: str,
    kb: str,
    max_depth: int = 10,
    max_soft=None,
    hard_result=None,
):
    """
    High-level pipeline:

      1) If hard_result is provided, use it (no extra BFS).
         Otherwise, run bfs_prolog_collect(goal, kb) once.

      2) If hard_result.success == True:
            -> HARD_SUCCESS, no background needed.

      3) If hard_result.success == False:
            -> generate_background_hypotheses
            -> attach_hypotheses_to_kb
            -> bfs_prolog_metro_soft

    Returns a dict:

        {
          "status": "HARD_SUCCESS" | "SOFT_SUCCESS" | "SOFT_FAILURE" | "FAILURE",
          "hard_result": {...},            # from bfs_prolog_collect
          "soft_result": {...} or None,    # from bfs_prolog_metro_soft
          "hypotheses": list[dict]
        }
    """

    print("\n========================================")
    print(f"SOLVE WITH BACKGROUND: {goal}")
    print("========================================\n")

    # 1) Hard-KB attempt (only if not provided)
    if hard_result is None:
        print(">>> Phase 1: Hard-KB BFS (bfs_prolog_collect)")
        hard_result = bfs_prolog_collect(goal, kb, max_depth=max_depth)
        print("Hard-KB result:", hard_result)
    else:
        print(">>> Phase 1: Hard-KB BFS result already computed, reusing it.")
        print("Hard-KB result:", hard_result)

    # If hard KB alone is enough, we are done
    if hard_result.get("success"):
        print("\n>>> Result: HARD_SUCCESS (no background hypotheses needed)\n")
        return {
            "status": "HARD_SUCCESS",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": []
        }

    # 2) If hard KB fails, generate background hypotheses for unresolved atoms
    unresolved_atoms = hard_result.get("unresolved_atoms", set())
    if not unresolved_atoms:
        print("\nNo unresolved atoms to explain; cannot generate hypotheses.")
        print(">>> Result: FAILURE\n")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": []
        }

    print("\n>>> Phase 2: Generate background hypotheses")
    print("Unresolved atoms:", unresolved_atoms)

    hypotheses = generate_background_hypotheses(
        goal=goal,
        kb=kb,
        unresolved_atoms=unresolved_atoms
    )

    # Robustness: if the function somehow returned None, treat as empty list
    if hypotheses is None:
        hypotheses = []

    if not hypotheses:
        print("Hypotheses returned by LLM: []")
        print("\nLLM returned NO hypotheses; cannot build soft KB.")
        print(">>> Result: FAILURE\n")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": []
        }

    print("Hypotheses returned by LLM:")
    for h in hypotheses:
        print("  - Clause:", h.get("clause"),
              "| Conf:", h.get("confidence"),
              "| From atom:", h.get("from_atom"))

    # 3) Attach hypotheses to KB as soft clauses
    print("\n>>> Phase 3: Attach hypotheses to soft KB")
    soft_kb = attach_hypotheses_to_kb(kb, hypotheses)
    print("Soft KB facts:", soft_kb.get("facts", []))
    print("Soft KB rules:", soft_kb.get("rules", []))

    # 4) Soft BFS using hard + soft KB
    print("\n>>> Phase 4: Soft BFS (bfs_prolog_metro_soft)")
    soft_result = bfs_prolog_metro_soft(
        goal=goal,
        kb=kb,
        soft_kb=soft_kb,
        max_depth=max_depth,
        max_soft=max_soft,
    )
    print("Soft-BFS result:", soft_result)

    if soft_result.get("success"):
        print("\n>>> Result: SOFT_SUCCESS (proof found using background hypotheses)\n")
        return {
            "status": "SOFT_SUCCESS",
            "hard_result": hard_result,
            "soft_result": soft_result,
            "hypotheses": hypotheses
        }

    print("\n>>> Result: SOFT_FAILURE (no proof even with background hypotheses)\n")
    return {
        "status": "SOFT_FAILURE",
        "hard_result": hard_result,
        "soft_result": soft_result,
        "hypotheses": hypotheses
    }


# Natural language to Prolog using LLM, algo SLD resolution

import re
import json

def nl_kb_to_prolog_kb(nl_kb_text: str, start_index: int = 1) -> list[str]:
    """
    Convert a *pure natural-language* description of a domain + rules
    into a numbered Prolog knowledge base.

    Example usage:
        nl_kb_text = '''
        There is a subway system with stations: Union Square, Times Square,
        Grand Central, and Bryant Park.

        Union Square is connected to Times Square.
        Times Square is connected to Grand Central.

        If one station is connected to another, then the first is reachable from
        the second. If X is connected to Y and Y is reachable to Z, then X is
        reachable to Z.
        '''

        prolog_kb = nl_kb_to_prolog_kb(nl_kb_text)
        # -> ["1. connected(union_square, times_square).",
        #     "2. connected(times_square, grand_central).",
        #     "3. reachable(X, Y) :- connected(X, Y).",
        #     "4. reachable(X, Z) :- connected(X, Y), reachable(Y, Z)."]

    Inputs:
        nl_kb_text : str
            Entire KB in natural language (facts + rules, any order).
        start_index: int
            Line number to start from (default 1). Useful if you want to
            merge multiple NL → KB calls later.

    Returns:
        list[str]: numbered Prolog clauses, each of the form:
            "<n>. clause(...)."
    """

    nl_kb_text = (nl_kb_text or "").strip()
    if not nl_kb_text:
        return []

    # --- Build LLM prompt (no reliance on existing Prolog KB) ---
    prompt = f"""
You are a Prolog formalization assistant.

The user will give you a natural-language description of a small domain,
including objects, relationships, and logical rules.

Your job is to convert that description into a set of Prolog clauses
(facts and rules).

Guidelines:
- Use lowercase atoms for concrete entities (e.g. union_square, times_square,
  grand_central, bryant_park).
- Use uppercase identifiers for variables (e.g. X, Y, Z).
- Choose predicate names that are short, descriptive, and consistent,
  for example: connected/2, reachable/2, located_in/2, etc.
- A fact must look like:
    connected(times_square, bryant_park).
- A rule must look like:
    reachable(X, Z) :- connected(X, Y), reachable(Y, Z).
- Every clause MUST end with a single period '.'.
- Do NOT include any line numbers in your output clauses.
- Do NOT add explanations or comments in the Prolog code.

Here is the NATURAL LANGUAGE description of the knowledge base:

\"\"\"{nl_kb_text}\"\"\"

Respond ONLY in this JSON format (and nothing else):

{{
  "clauses": [
    {{
      "clause": "connected(union_square, times_square)."
    }},
    {{
      "clause": "reachable(X, Y) :- connected(X, Y)."
    }}
  ]
}}
"""

    raw = ask_llm(prompt).strip()
    try:
        data = json.loads(extract_first_json(raw))
    except Exception as e:
        print("[nl_kb_to_prolog_kb] JSON parse error:", e)
        print("Raw LLM output:", raw)
        return []

    raw_clauses = data.get("clauses", [])
    if not isinstance(raw_clauses, list):
        print("[nl_kb_to_prolog_kb] 'clauses' field is not a list:", raw_clauses)
        return []

    cleaned_clauses = []

    for item in raw_clauses:
        # item may be a string or {"clause": "..."}
        if isinstance(item, str):
            clause = item.strip()
        elif isinstance(item, dict):
            clause = (item.get("clause") or "").strip()
        else:
            continue

        if not clause:
            continue

        # Strip any numbering if the model sneaks it in
        m_num = re.match(r'^\s*(\d+)\.\s*(.+)$', clause)
        if m_num:
            clause = m_num.group(2).strip()

        # Ensure it ends with a single dot
        clause = clause.rstrip()
        if not clause.endswith('.'):
            clause = clause + "."
        else:
            # avoid cases like '...)).' becoming '...)).'
            clause = re.sub(r'\.+$', '.', clause)

        # Very basic sanity: must look like head. or head :- body.
        body_str = clause[:-1].strip()  # drop final dot
        if ':-' in body_str:
            head_part, body_part = body_str.split(':-', 1)
            head = head_part.strip()
        else:
            head = body_str

        # Parse head as a predicate using your existing helper
        parsed_head = parse_predicate(head)
        if parsed_head is None:
            print("[nl_kb_to_prolog_kb] Discarding unparsable clause:", clause)
            continue

        cleaned_clauses.append(clause)

    # --- Number clauses sequentially starting from start_index ---
    numbered_clauses = []
    next_num = start_index
    for clause in cleaned_clauses:
        numbered_clauses.append(f"{next_num}. {clause}")
        next_num += 1

    return numbered_clauses

if __name__ == "__main__":
    # KB with fact 3 removed
    kb_missing_3 = """
    1. connected(union_square, times_square). #union square and times square are connected
    2. connected(times_square, grand_central). times square and grand central are connect
    4. reachable(X, Y) :- connected(X, Y). if X and Y are connected, then X is reachable from Y ; if two stations are connected, then the first is reachable from the second
    5. reachable(X, Z) :- connected(X, Y), reachable(Y, Z). if X and Z are connected, then Y is reachable from Z
    """

    print("===== METRO BFS - PROLOG WITH FACT 3 REMOVED =====")
    print(kb_missing_3)
    print("===================================================\n")

    tests = [
        "reachable(union_square, bryant_park)"
    ]

    for test in tests:
        print("\n==============================")
        print(f"TEST QUERY: {test}")
        print("==============================\n")

        # 1) Single hard-KB run using bfs_prolog_collect
        print(">>> Running bfs_prolog_collect (hard-KB BFS)...")
        collect_result = bfs_prolog_collect(test, kb_missing_3)
        print("Collect Result:", collect_result)
        print("\n----------------------------------------\n")

        if collect_result["success"]:
            print("Hard KB alone was enough. No background hypotheses needed.\n")
            continue

        # 2) Solve with background, reusing the hard_result (no re-run)
        print(">>> Running solve_with_background (full pipeline, reusing hard result)...")
        bg_result = solve_with_background(
            goal=test,
            kb=kb_missing_3,
            hard_result=collect_result   # <--- reuse, no extra BFS
        )
        print("Solve-with-background Result:")
        print(bg_result)

        print("\n========================================\n")


# In[35]:


nl_kb = """
We have stations: Union Square, Times Square, Grand Central, Bryant Park.
Union Square is connected to Times Square.
Times Square is connected to Grand Central.
If one station is connected to another, the first is reachable from the second.
If a station X is connected to Y and Y is reachable to Z, then X is reachable to Z.
"""

prolog_kb = nl_kb_to_prolog_kb(nl_kb)
for line in prolog_kb:
    print(line)


# In[55]:


# No LLM Prolog SLD Engine

def unify_args(args_goal, args_fact, env=None):
    """
    Unify two argument lists (flat terms, no nesting) under an environment.

    args_goal: list[str]  from the GOAL predicate
    args_fact: list[str]  from the FACT/RULE-HEAD predicate
    env      : dict or None   existing bindings, e.g. {"X": "times_square"}

    Returns:
        - None if unification fails
        - env (possibly modified) if unification succeeds
    """
    if env is None:
        env = {}

    if len(args_goal) != len(args_fact):
        return None

    for g, f in zip(args_goal, args_fact):
        g = g.strip()
        f = f.strip()

        g_is_var = is_variable(g)
        f_is_var = is_variable(f)

        # both constants
        if not g_is_var and not f_is_var:
            if g != f:
                return None
            continue

        # goal var, fact const
        if g_is_var and not f_is_var:
            if g in env:
                if env[g] != f:
                    return None
            else:
                env[g] = f
            continue

        # goal const, fact var  (treat fact vars as wildcards)
        if not g_is_var and f_is_var:
            if f in env:
                if env[f] != g:
                    return None
            else:
                env[f] = g
            continue

        # both variables
        if g_is_var and f_is_var:
            if g in env and f in env:
                if env[g] != env[f]:
                    return None
            elif g in env:
                env[f] = env[g]
            elif f in env:
                env[g] = env[f]
            # else both unbound → no constraint
            continue

    return env


def unify_with_fact(goal: str, fact: str):
    """
    Purely algorithmic unification between a GOAL and a FACT (or rule head).

    Returns:
        None      -> NO unification
        {}        -> EXACT ground match (no variables)
        dict      -> bindings, e.g. {"Y": "times_square"}
    """

    parsed_goal = parse_predicate(goal)
    parsed_fact = parse_predicate(fact)
    if parsed_goal is None or parsed_fact is None:
        return None

    fun_g, args_g = parsed_goal
    fun_f, args_f = parsed_fact

    # Functor or arity mismatch
    if fun_g != fun_f or len(args_g) != len(args_f):
        return None

    # If they are exactly the same string (ignoring trailing dot), treat as EXACT
    if check_exact_match(goal, fact):
        return {}

    env = unify_args(args_g, args_f, env={})
    if env is None:
        return None

    # If env ended up empty but they weren't literally identical strings, we
    # still treat this as a unification with no variable bindings.
    return env

def apply_bindings(goals, bindings):
    """
    Apply variable bindings to goals using pure string/term substitution.

    goals: list[str]   e.g. ["reachable(Y, Z)", "connected(Z, X)"]
    bindings: dict     e.g. {"Y": "times_square"}

    Returns: list[str] of instantiated goals.
    """
    if not bindings or not goals:
        return goals

    new_goals = []

    for g in goals:
        parsed = parse_predicate(g)
        if parsed is None:
            # If we can't parse it as a predicate, leave as-is
            new_goals.append(g)
            continue

        functor, args = parsed
        new_args = []
        for a in args:
            a_stripped = a.strip()
            if is_variable(a_stripped) and a_stripped in bindings:
                new_args.append(bindings[a_stripped])
            else:
                new_args.append(a_stripped)

        new_goal = f"{functor}({', '.join(new_args)})"
        new_goals.append(new_goal)

    return new_goals

def unify_arg_lists(head_args, goal_args):
    """
    Unify the arguments of a rule head with the arguments of a goal.

    head_args: list of strings from the rule head, e.g. ["X", "Y"]
    goal_args: list of strings from the goal,     e.g. ["union_square", "bryant_park"]

    Returns:
        - dict of bindings, e.g. {"X": "union_square", "Y": "bryant_park"}
        - or None if unification fails.
    """
    if len(head_args) != len(goal_args):
        return None

    bindings = {}

    for h_arg, g_arg in zip(head_args, goal_args):
        h_is_var = is_variable(h_arg)
        g_is_var = is_variable(g_arg)

        if h_is_var and g_is_var:
            # Two variables; we can just pick one direction (head → goal)
            # If head var already bound, enforce consistency.
            if h_arg in bindings:
                if bindings[h_arg] != g_arg:
                    return None
            else:
                bindings[h_arg] = g_arg

        elif h_is_var and not g_is_var:
            # Head has variable, goal has constant
            if h_arg in bindings:
                if bindings[h_arg] != g_arg:
                    return None
            else:
                bindings[h_arg] = g_arg

        elif not h_is_var and g_is_var:
            # Head has constant, goal has variable.
            # For our use-case (unifying rule head with goal), we only
            # really *need* bindings for head vars, but we can add this
            # for completeness.
            if g_arg in bindings:
                if bindings[g_arg] != h_arg:
                    return None
            else:
                bindings[g_arg] = h_arg

        else:
            # Both are constants; they must match
            if h_arg != g_arg:
                return None

    return bindings

def substitute_in_atom(atom: str, bindings: dict) -> str:
    """
    Apply variable bindings to a single Prolog atom, e.g.:

        atom     = "connected(X, Y)"
        bindings = {"X": "union_square", "Y": "bryant_park"}

    Returns:
        "connected(union_square, bryant_park)"
    """
    parsed = parse_predicate(atom)
    if parsed is None:
        return atom  # best-effort fallback

    functor, args = parsed
    new_args = []

    for a in args:
        a_stripped = a.strip()
        if is_variable(a_stripped) and a_stripped in bindings:
            new_args.append(bindings[a_stripped])
        else:
            new_args.append(a_stripped)

    return f"{functor}({', '.join(new_args)})"
def split_body_atoms(body_str: str):
    """
    Split a rule body like:
        "connected(X, Y), reachable(Y, Z)"
    into:
        ["connected(X, Y)", "reachable(Y, Z)"]

    It is parentheses-aware, so it will NOT split on commas that are
    inside argument lists.
    """
    body_str = body_str.strip()
    atoms = []
    current = []
    depth = 0  # parentheses nesting depth

    for ch in body_str:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth = max(depth - 1, 0)
            current.append(ch)
        elif ch == ',' and depth == 0:
            # top-level comma → split here
            atom = ''.join(current).strip()
            if atom:
                atoms.append(atom)
            current = []
        else:
            current.append(ch)

    # Flush the last atom
    atom = ''.join(current).strip()
    if atom:
        atoms.append(atom)

    return atoms


def get_subgoals(goal: str, rule_head: str, rule_body: str):
    """
    Algorithmic ONE-STEP SLD resolution:

    Given:
        goal      : e.g. "reachable(union_square, bryant_park)"
        rule_head : e.g. "reachable(X, Y)"
        rule_body : e.g. "connected(X, Y)"

    Steps:
      1. Parse goal and rule_head into (functor, args).
      2. If functor or arity differ -> rule cannot apply -> return None.
      3. Unify head args with goal args → bindings.
      4. If unification fails -> return None.
      5. Split rule_body on ',' into individual atoms.
      6. Apply bindings to each body atom.
      7. Return the list of instantiated subgoal atoms.

    Returns:
        - list[str] of subgoals, e.g. ["connected(union_square, bryant_park)"]
        - or None if the rule cannot be applied.
    """
    # 1) Parse goal and rule head
    parsed_goal = parse_predicate(goal)
    parsed_head = parse_predicate(rule_head)

    if parsed_goal is None or parsed_head is None:
        return None

    fun_g, args_g = parsed_goal
    fun_h, args_h = parsed_head

    # 2) Functor / arity mismatch => cannot use this rule
    if fun_g != fun_h or len(args_g) != len(args_h):
        return None

    # 3) Unify arguments (rule-head vars with goal terms)
    bindings = unify_arg_lists(args_h, args_g)
    if bindings is None:
        return None

    # 4) Split rule body into atoms
    body_str = rule_body.strip()
    if not body_str:
        # A rule with empty body would be strange, but handle it
        return []

    body_atoms = split_body_atoms(body_str)
    if not body_atoms:
        return []


    # 5) Apply bindings to each atom
    subgoals = [substitute_in_atom(atom, bindings) for atom in body_atoms]

    return subgoals if subgoals else None


# In[48]:


# --- Build KB string from the natural-language-converted KB ---

kb_str = "\n".join(prolog_kb)
print("Prolog KB:\n", kb_str, "\n")

# --- Parse KB into facts and rules (like bfs_prolog_metro does) ---

facts = []   # list of (num, fact_str)
rules = []   # list of (num, head_str, body_str)

for line in kb_str.strip().split("\n"):
    line = line.strip()
    if not line:
        continue

    m = re.match(r'^(\d+)\.\s*(.+)$', line)
    if not m:
        continue

    num = int(m.group(1))
    content = m.group(2).strip()

    if ':-' in content:
        head, body = content.split(':-', 1)
        rules.append((num, head.strip(), body.strip().rstrip('.')))
    else:
        facts.append((num, content.rstrip('.')))

print("Parsed facts:")
for num, f in facts:
    print(f"  {num}. {f}")
print("\nParsed rules:")
for num, h, b in rules:
    print(f"  {num}. {h} :- {b}")
print("\n")


# ============================================================
# 1) Test unify_with_fact
# ============================================================

print("=== TEST 1: unify_with_fact ===")

# Try to unify a concrete GOAL with the first connected/2 fact we find
goal1 = "connected(union_square, times_square)"
fact_candidate = None
for num, f in facts:
    if f.startswith("connected("):
        fact_candidate = (num, f)
        break

if fact_candidate is None:
    print("No connected/2 fact found in KB to test unification.")
else:
    num, fact_str = fact_candidate
    print(f"Goal: {goal1}")
    print(f"Fact {num}: {fact_str}")
    bindings = unify_with_fact(goal1, fact_str)
    print("Unification result (bindings):", bindings)
print("\n")


# ============================================================
# 2) Test apply_bindings
# ============================================================

print("=== TEST 2: apply_bindings ===")

bindings2 = {"X": "union_square", "Y": "bryant_park"}
goals2 = ["reachable(X, Y)", "connected(X, Y)"]

print("Original goals:", goals2)
print("Bindings:", bindings2)

instantiated_goals = apply_bindings(goals2, bindings2)

print("Instantiated goals:", instantiated_goals)
print("\n")


# ============================================================
# 3) Test get_subgoals
# ============================================================

print("=== TEST 3: get_subgoals ===")

# Find a reachable/2 rule, e.g.:
#   reachable(X, Y) :- connected(X, Y).
#   reachable(X, Z) :- connected(X, Y), reachable(Y, Z).
reachable_rule = None
for num, head, body in rules:
    parsed = parse_predicate(head)
    if parsed is None:
        continue
    functor, args = parsed
    if functor == "reachable" and len(args) == 2:
        reachable_rule = (num, head, body)
        break

if reachable_rule is None:
    print("No reachable/2 rule found in KB to test get_subgoals.")
else:
    num, head, body = reachable_rule
    # Use a concrete goal that should match, e.g. reachable(union_square, bryant_park)
    test_goal3 = "reachable(union_square, bryant_park)"

    print(f"Using rule {num}: {head} :- {body}")
    print(f"Goal: {test_goal3}")

    subgoals = get_subgoals(test_goal3, head, body)
    print("Resulting subgoals:", subgoals)

print("\n=== END OF SYMBOLIC TESTS ===")


# In[ ]:





# In[56]:


import ollama
from collections import deque
import re
import json
from typing import Optional

# --- Config / LLM setup ---

client = ollama.Client()

model = "gpt-oss:20b"
# model = "qwen:14b"

DEBUG = False  # set to True to print raw LLM outputs for debugging


def ask_llm(prompt: str) -> str:
    resp = client.generate(model=model, prompt=prompt, options={'temperature': 0.0})
    answer = resp.get('response', '')
    if "...done thinking." in answer:
        return answer.split("...done thinking.")[-1].strip()
    return answer


# --- Helpers for parsing / JSON ---

def extract_first_json(text: str) -> str:
    """
    Extract the first {...} JSON object from possibly messy text.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in: {text!r}")
    return match.group(0)




def parse_predicate(term: str):
    """
    Parse a simple Prolog predicate of the form:
        functor(arg1, arg2, ...)
    Returns (functor: str, args: list[str]) or None if parsing fails.
    """
    term = term.strip().rstrip('.')
    m = re.match(r'^([a-z_][a-zA-Z0-9_]*)\((.*)\)$', term)
    if not m:
        return None
    functor = m.group(1)
    args_raw = m.group(2).strip()
    if not args_raw:
        args = []
    else:
        # Simple arg split (no nested terms in this toy domain)
        args = [a.strip() for a in args_raw.split(',')]
    return functor, args


def is_variable(s: str) -> bool:
    """
    Prolog-ish variable check: starts with uppercase letter or '_'.
    """
    s = s.strip()
    return bool(s) and (s[0].isupper() or s[0] == '_')


# --- Core Prolog helpers ---

def check_exact_match(goal: str, fact: str) -> bool:
    """Check if goal matches fact exactly (no variables)."""
    return goal.strip().rstrip('.') == fact.strip().rstrip('.')

def unify_args(args_goal, args_fact, env=None):
    """
    Unify two argument lists (flat terms, no nesting) under an environment.

    args_goal: list[str]  from the GOAL predicate
    args_fact: list[str]  from the FACT/RULE-HEAD predicate
    env      : dict or None   existing bindings, e.g. {"X": "times_square"}

    Returns:
        - None if unification fails
        - env (possibly modified) if unification succeeds
    """
    if env is None:
        env = {}

    if len(args_goal) != len(args_fact):
        return None

    for g, f in zip(args_goal, args_fact):
        g = g.strip()
        f = f.strip()

        g_is_var = is_variable(g)
        f_is_var = is_variable(f)

        # both constants
        if not g_is_var and not f_is_var:
            if g != f:
                return None
            continue

        # goal var, fact const
        if g_is_var and not f_is_var:
            if g in env:
                if env[g] != f:
                    return None
            else:
                env[g] = f
            continue

        # goal const, fact var  (treat fact vars as wildcards)
        if not g_is_var and f_is_var:
            if f in env:
                if env[f] != g:
                    return None
            else:
                env[f] = g
            continue

        # both variables
        if g_is_var and f_is_var:
            if g in env and f in env:
                if env[g] != env[f]:
                    return None
            elif g in env:
                env[f] = env[g]
            elif f in env:
                env[g] = env[f]
            # else both unbound → no constraint
            continue

    return env


def unify_with_fact(goal: str, fact: str):
    """
    Purely algorithmic unification between a GOAL and a FACT (or rule head).

    Returns:
        None      -> NO unification
        {}        -> EXACT ground match (no variables)
        dict      -> bindings, e.g. {"Y": "times_square"}
    """

    parsed_goal = parse_predicate(goal)
    parsed_fact = parse_predicate(fact)
    if parsed_goal is None or parsed_fact is None:
        return None

    fun_g, args_g = parsed_goal
    fun_f, args_f = parsed_fact

    # Functor or arity mismatch
    if fun_g != fun_f or len(args_g) != len(args_f):
        return None

    # If they are exactly the same string (ignoring trailing dot), treat as EXACT
    if check_exact_match(goal, fact):
        return {}

    env = unify_args(args_g, args_f, env={})
    if env is None:
        return None

    return env

def apply_bindings(goals, bindings):
    """
    Apply variable bindings to goals using pure string/term substitution.

    goals: list[str]   e.g. ["reachable(Y, Z)", "connected(Z, X)"]
    bindings: dict     e.g. {"Y": "times_square"}

    Returns: list[str] of instantiated goals.
    """
    if not bindings or not goals:
        return goals

    new_goals = []

    for g in goals:
        parsed = parse_predicate(g)
        if parsed is None:
            # If we can't parse it as a predicate, leave as-is
            new_goals.append(g)
            continue

        functor, args = parsed
        new_args = []
        for a in args:
            a_stripped = a.strip()
            if is_variable(a_stripped) and a_stripped in bindings:
                new_args.append(bindings[a_stripped])
            else:
                new_args.append(a_stripped)

        new_goal = f"{functor}({', '.join(new_args)})"
        new_goals.append(new_goal)

    return new_goals




def find_matching_rules_only(goal, rules_list):
    """
    Find ONLY rules (not facts) whose HEAD can unify with the given goal.

    IMPORTANT: This version is purely syntactic: it only checks functor and arity.
    We do NOT ask the LLM here, to avoid mismatched heads like reachable/2
    being applied to connected/2 goals.

    rules_list: list[(num, head, body)]
    Returns: list[int] of rule numbers.
    """
    parsed_goal = parse_predicate(goal)
    if parsed_goal is None:
        return []
    fun_g, args_g = parsed_goal
    arity_g = len(args_g)

    matching = []
    for num, head, body in rules_list:
        parsed_head = parse_predicate(head)
        if parsed_head is None:
            continue
        fun_h, args_h = parsed_head
        if fun_h == fun_g and len(args_h) == arity_g:
            matching.append(num)
    return matching

def substitute_in_atom(atom: str, bindings: dict) -> str:
    """
    Apply variable bindings to a single Prolog atom, e.g.:

        atom     = "connected(X, Y)"
        bindings = {"X": "union_square", "Y": "bryant_park"}

    Returns:
        "connected(union_square, bryant_park)"
    """
    parsed = parse_predicate(atom)
    if parsed is None:
        return atom  # best-effort fallback

    functor, args = parsed
    new_args = []

    for a in args:
        a_stripped = a.strip()
        if is_variable(a_stripped) and a_stripped in bindings:
            new_args.append(bindings[a_stripped])
        else:
            new_args.append(a_stripped)

    return f"{functor}({', '.join(new_args)})"
def split_body_atoms(body_str: str):
    """
    Split a rule body like:
        "connected(X, Y), reachable(Y, Z)"
    into:
        ["connected(X, Y)", "reachable(Y, Z)"]

    It is parentheses-aware, so it will NOT split on commas that are
    inside argument lists.
    """
    body_str = body_str.strip()
    atoms = []
    current = []
    depth = 0  # parentheses nesting depth

    for ch in body_str:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth = max(depth - 1, 0)
            current.append(ch)
        elif ch == ',' and depth == 0:
            # top-level comma → split here
            atom = ''.join(current).strip()
            if atom:
                atoms.append(atom)
            current = []
        else:
            current.append(ch)

    # Flush the last atom
    atom = ''.join(current).strip()
    if atom:
        atoms.append(atom)

    return atoms


def get_subgoals(goal: str, rule_head: str, rule_body: str):
    """
    Algorithmic ONE-STEP SLD resolution:

    Given:
        goal      : e.g. "reachable(union_square, bryant_park)"
        rule_head : e.g. "reachable(X, Y)"
        rule_body : e.g. "connected(X, Y)"

    Steps:
      1. Parse goal and rule_head into (functor, args).
      2. If functor or arity differ -> rule cannot apply -> return None.
      3. Unify head args with goal args → bindings.
      4. If unification fails -> return None.
      5. Split rule_body on ',' into individual atoms.
      6. Apply bindings to each body atom.
      7. Return the list of instantiated subgoal atoms.

    Returns:
        - list[str] of subgoals, e.g. ["connected(union_square, bryant_park)"]
        - or None if the rule cannot be applied.
    """
    # 1) Parse goal and rule head
    parsed_goal = parse_predicate(goal)
    parsed_head = parse_predicate(rule_head)

    if parsed_goal is None or parsed_head is None:
        return None

    fun_g, args_g = parsed_goal
    fun_h, args_h = parsed_head

    # 2) Functor / arity mismatch => cannot use this rule
    if fun_g != fun_h or len(args_g) != len(args_h):
        return None

    # 3) Unify arguments (rule-head vars with goal terms)
    bindings = unify_arg_lists(args_h, args_g)
    if bindings is None:
        return None

    # 4) Split rule body into atoms
    body_str = rule_body.strip()
    if not body_str:
        # A rule with empty body would be strange, but handle it
        return []

    body_atoms = split_body_atoms(body_str)
    if not body_atoms:
        return []


    # 5) Apply bindings to each atom
    subgoals = [substitute_in_atom(atom, bindings) for atom in body_atoms]

    return subgoals if subgoals else None



# --- BFS Prolog engine ---

def bfs_prolog_metro(goal: str, kb: str, max_depth: int = 10) -> bool:
    """
    BFS with correct fact/rule distinction, using LLM to do:
    - unification with facts (but guarded by functor/arity)
    - applying bindings to remaining goals
    - generating instantiated subgoals
    """

    # Parse KB - separate facts and rules
    facts = []
    rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content = match.group(2).strip()

            if ':-' in content:
                head, body = content.split(':-', 1)
                rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                facts.append((num, content.rstrip('.')))

    # BFS queue: (current_goal, remaining_goals, path, depth)
    queue = deque([(goal, [], [], 0)])
    visited = set()

    print(f"\nGoal: {goal}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        if depth >= max_depth:
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")

        # 1) FIRST: Check for exact fact match
        fact_matched = False
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")

                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return True

                # Continue with remaining goals
                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))
                fact_matched = True
                break

        if fact_matched:
            continue

        # 2) SECOND: Check for fact unification with variables (non-exact)
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)

            if bindings is None:
                continue

            print(f"  ✓ Fact {num}: {fact}")
            print(f"    Bindings: {bindings}")

            # Apply bindings to remaining goals
            instantiated = apply_bindings(remaining, bindings)

            if not instantiated:
                print(f"✓✓ SUCCESS at depth {depth + 1}")
                return True

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))

        # 3) THIRD: Try rules (matching by functor & arity only)
        matching_rules = find_matching_rules_only(current, rules)

        if matching_rules:
            print(f"  Matching rules: {matching_rules}")

            for rule_num in matching_rules:
                for num, head, body in rules:
                    if num == rule_num:
                        subgoals = get_subgoals(current, head, body)

                        if subgoals:
                            print(f"  Rule {num}: → {subgoals}")
                            all_goals = subgoals + remaining
                            next_goal = all_goals[0]
                            next_remaining = all_goals[1:]
                            queue.append((next_goal, next_remaining, path + [f"Rule {num}"], depth + 1))
                        break

    print("✗ FAILED")
    return False

def bfs_prolog_collect(goal: str, kb: str, max_depth: int = 10):
    """
    Near-clone of bfs_prolog_metro, but:

    - Still uses unify_with_fact → LLM
    - Still uses apply_bindings → LLM
    - Still uses get_subgoals → LLM
    - Still uses functor/arity filtering for rules
    - Still prints identical debug output

    Differences:
        • Returns a dict:
            {
              "success": bool,
              "proof_path": [...],
              "unresolved_atoms": set()
            }

        • Tracks unresolved atoms when:
            - no exact fact matched
            - no fact unified
            - no rule applied
    """

    # --- Parse KB exactly like bfs_prolog_metro ---
    facts = []
    rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content = match.group(2).strip()

            if ':-' in content:
                head, body = content.split(':-', 1)
                rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                facts.append((num, content.rstrip('.')))

    # BFS state: (current_goal, remaining_goals, path, depth)
    queue = deque([(goal, [], [], 0)])
    visited = set()
    unresolved_atoms = set()

    print(f"\n[COLLECT] Goal: {goal}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        if depth >= max_depth:
            unresolved_atoms.add(current)
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")

        # Track whether ANY progress was made on this goal
        progress = False

        # --- 1) Exact fact match ---
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")

                step = f"Fact {num}"
                new_path = path + [step]

                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return {
                        "success": True,
                        "proof_path": new_path,
                        "unresolved_atoms": set()
                    }

                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, new_path, depth + 1))

                progress = True
                break

        if progress:
            continue

        # --- 2) Fact unification (LLM) ---
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            progress = True

            print(f"  ✓ Fact {num}: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)

            step = f"Fact {num}"
            new_path = path + [step]

            if not instantiated:
                print(f"✓✓ SUCCESS at depth {depth + 1}")
                return {
                    "success": True,
                    "proof_path": new_path,
                    "unresolved_atoms": set()
                }

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, new_path, depth + 1))

        if progress:
            continue

        # --- 3) Rule attempts (LLM SLD step) ---
        matching_rules = find_matching_rules_only(current, rules)

        if matching_rules:
            print(f"  Matching rules: {matching_rules}")

        for rule_num in matching_rules:
            for num, head, body in rules:
                if num == rule_num:
                    subgoals = get_subgoals(current, head, body)
                    if subgoals:
                        print(f"  Rule {num}: → {subgoals}")
                        progress = True
                        all_goals = subgoals + remaining
                        next_goal = all_goals[0]
                        next_remaining = all_goals[1:]
                        step = f"Rule {num}"
                        new_path = path + [step]
                        queue.append((next_goal, next_remaining, new_path, depth + 1))
                    break

        # --- If absolutely nothing worked, record unresolved ---
        if not progress:
            print(f"  ✗ No facts or rules apply to: {current}")
            unresolved_atoms.add(current)

    # BFS fully exhausted → failure
    print("✗ FAILED (collect mode)")
    return {
        "success": False,
        "proof_path": [],
        "unresolved_atoms": unresolved_atoms
    }


import re
import json

def generate_background_hypotheses(goal: str, kb: str, unresolved_atoms, max_atoms: int = 5):
    """
    Use the LLM's background/world knowledge to propose additional Prolog clauses
    (facts or rules) that might make the GOAL provable.

    ALWAYS returns a list (possibly empty).
    NEVER returns None.
    """

    hypotheses = []

    # --- 1) Filter unresolved atoms to simple, ground atoms only ---
    # e.g. keep "connected(grand_central, bryant_park)" but drop "connected(grand_central, Y)"
    atom_list = list(unresolved_atoms)
    ground_atoms = []
    for atom in atom_list:
        atom = atom.strip()
        if not atom:
            continue
        # must look like functor(...)
        if '(' not in atom or ')' not in atom:
            continue
        # crude check: if any token inside parens starts with uppercase or '_', treat as variable → skip
        inside = atom.split('(', 1)[1].rsplit(')', 1)[0]
        if re.search(r'\b[A-Z_]\w*\b', inside):
            continue
        ground_atoms.append(atom)

    if max_atoms is not None and len(ground_atoms) > max_atoms:
        ground_atoms = ground_atoms[:max_atoms]

    if not ground_atoms:
        print("[generate_background_hypotheses] No suitable ground atoms to query.")
        return []  # <--- explicit list, not None

    # --- 2) For each ground unresolved atom, ask the LLM for hypotheses ---
    for atom in ground_atoms:
        prompt = f"""
You are a cautious Prolog expert with access to real-world background knowledge.

We attempted to prove the following GOAL using ONLY the numbered Prolog
knowledge base given below:

GOAL:
  {goal}

KNOWLEDGE BASE (numbered clauses):
{kb}

During breadth-first SLD resolution, the proof FAILED because we could not
prove the following subgoal:
  {atom}

Task:
Propose a SMALL set of additional Prolog clauses (facts or rules) that are
LIKELY to be true in the intended domain and that would help make the GOAL
provable. Think of these as "missing" facts or rules that could fill gaps in
the knowledge base.

Constraints:
- Each clause MUST be valid Prolog and MUST end with a period.
- You MUST NOT modify or delete any existing clauses in the KB.
- Use ONLY predicate names and arities that are compatible with the style of
  the existing KB (for example, connected/2, reachable/2, etc.).
- If you are uncertain about the factual correctness of a clause, give it a lower confidence.
- You SHOULD prefer to return at least one plausible clause rather than an empty list.

Respond ONLY in this JSON format:

{{
  "hypotheses": [
    {{
      "clause": "connected(times_square, grand_central).",
      "confidence": 0.9
    }},
    {{
      "clause": "connected(times_square, bryant_park).",
      "confidence": 0.4
    }}
  ]
}}

If you truly have NO hypotheses, respond with:
{{ "hypotheses": [] }}
"""

        raw = ask_llm(prompt).strip()
        if DEBUG:
            print("\n[DEBUG generate_background_hypotheses]")
            print("Unresolved atom:", atom)
            print("Raw response:\n", raw)

        try:
            data = json.loads(extract_first_json(raw))
        except Exception as e:
            print("[generate_background_hypotheses] JSON parse error:", e)
            print("Raw LLM output:", raw)

            # Fix common Prolog backslash pattern like "\=" which is invalid JSON
            if "Invalid \\escape" in str(e):
                try:
                    fixed_raw = raw.replace("\\=", "\\\\=")
                    data = json.loads(extract_first_json(fixed_raw))
                except Exception as e2:
                    print("[generate_background_hypotheses] JSON parse error after fix:", e2)
                    continue
            else:
                continue  # different JSON error; skip this atom

        raw_hyps = data.get("hypotheses", [])
        if not isinstance(raw_hyps, list):
            print("[generate_background_hypotheses] 'hypotheses' not a list:", raw_hyps)
            continue

        for h in raw_hyps:
            clause = (h.get("clause") or "").strip()
            if not clause:
                continue

            # Normalize: ensure clause ends with a dot
            if not clause.endswith('.'):
                clause = clause + "."

            try:
                conf = float(h.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0

            hypotheses.append({
                "clause": clause,
                "confidence": conf,
                "from_atom": atom
            })

    # --- 3) Deduplicate clauses (keep highest-confidence version) ---
    dedup = {}
    for h in hypotheses:
        key = h["clause"]
        if key not in dedup or h["confidence"] > dedup[key]["confidence"]:
            dedup[key] = h

    return list(dedup.values())  # <--- ALWAYS returns a list

def _find_max_line_number_in_kb(kb: str) -> int:
    """
    Scan the numbered KB and return the maximum clause number seen.
    If no numbered lines are found, return 0.
    """
    max_num = 0
    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue
        num = int(m.group(1))
        if num > max_num:
            max_num = num
    return max_num


def _is_fact_clause(clause: str) -> bool:
    """
    Heuristic: a fact has no ':-' in it.
    Example: 'connected(times_square, grand_central).'
    """
    return ':-' not in clause


def _split_rule_clause(clause: str):
    """
    Split a rule clause 'Head :- Body.' into (head, body_str).

    Returns:
        head: 'reachable(X, Z)'
        body_str: 'connected(X, Y), reachable(Y, Z)'
    """
    # Remove trailing dot if present
    clause = clause.strip()
    if clause.endswith('.'):
        clause = clause[:-1]

    head_part, body_part = clause.split(':-', 1)
    head = head_part.strip()
    body_str = body_part.strip()
    return head, body_str


def attach_hypotheses_to_kb(kb: str, hypotheses):
    """
    Convert LLM-generated hypotheses into a 'soft KB' structure.

    Inputs:
        kb          : original numbered KB string
        hypotheses  : list[dict] from generate_background_hypotheses, where each dict has:
                        {
                          "clause": "prolog_clause_string_ending_with_dot.",
                          "confidence": float,
                          "from_atom": "unresolved atom"
                        }

    Returns:
        soft_kb: dict with two lists:
            {
              "facts": [
                  (num, atom_str, confidence),
                  ...
              ],
              "rules": [
                  (num, head_str, body_str, confidence),
                  ...
              ]
            }

        where:
          - num is a fresh line number (beyond any in the original KB),
          - atom_str is like "connected(times_square, grand_central)",
          - head_str is like "reachable(X, Z)",
          - body_str is like "connected(X, Y), reachable(Y, Z)".
    """

    soft_facts = []
    soft_rules = []

    # Start numbering hypotheses after the max existing KB line number
    max_num = _find_max_line_number_in_kb(kb)
    next_num = max_num + 1

    for h in hypotheses:
        clause = (h.get("clause") or "").strip()
        if not clause:
            continue

        conf = float(h.get("confidence", 0.0))

        # Normalize: ensure terminating dot
        if not clause.endswith('.'):
            clause = clause + '.'

        if _is_fact_clause(clause):
            # Example: "connected(times_square, grand_central)."
            atom = clause.rstrip('.').strip()
            soft_facts.append((next_num, atom, conf))
        else:
            # Example: "reachable(X, Z) :- connected(X, Y), reachable(Y, Z)."
            head, body_str = _split_rule_clause(clause)
            soft_rules.append((next_num, head, body_str, conf))

        next_num += 1

    soft_kb = {
        "facts": soft_facts,
        "rules": soft_rules,
    }
    return soft_kb

def bfs_prolog_metro_soft(
    goal: str,
    kb: str,
    soft_kb,
    max_depth: int = 10,
    max_soft: Optional[int] = None,
):
    """
    BFS SLD resolution that can use:
      - hard clauses from the original KB
      - soft clauses (hypotheses) from soft_kb

    soft_kb is the dict from attach_hypotheses_to_kb:
        {
          "facts": [(num, atom_str, confidence), ...],
          "rules": [(num, head_str, body_str, confidence), ...]
        }

    Returns a dict:
        {
          "success": bool,
          "proof_path": list,             # list of step labels
          "used_soft_clauses": list,      # list of (kind, num, confidence)
          "soft_cost": int,               # how many soft clauses used
          "min_conf": float | None        # min confidence over used soft clauses
        }
    """

    # --- Parse hard KB exactly like bfs_prolog_metro ---
    hard_facts = []
    hard_rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content = match.group(2).strip()

            if ':-' in content:
                head, body = content.split(':-', 1)
                hard_rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                hard_facts.append((num, content.rstrip('.')))

    # --- Soft KB unpack ---
    soft_facts = soft_kb.get("facts", [])  # list of (num, atom, conf)
    soft_rules = soft_kb.get("rules", [])  # list of (num, head, body_str, conf)

    # For rule matching, we want (num, head, body_str) lists
    soft_rules_for_match = [(num, head, body_str) for (num, head, body_str, conf) in soft_rules]

    # BFS queue: (current_goal, remaining_goals, path, depth, soft_cost, min_conf)
    #   path is a list of human-readable step labels (strings) like:
    #     "HardFact 1", "SoftFact 1001 (conf=0.90)", "HardRule 5", etc.
    queue = deque([(goal, [], [], 0, 0, 1.0)])  # min_conf starts at 1.0
    visited = set()

    print(f"\n[SOFT BFS] Goal: {goal}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth, soft_cost, min_conf = queue.popleft()

        if depth >= max_depth:
            continue

        # You can choose how much of state goes into 'visited'.
        # Here we include soft_cost to distinguish "cheaper" vs "more expensive" paths.
        state = (current, tuple(remaining), soft_cost)
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        print(f"  Soft cost: {soft_cost}, min_conf: {min_conf:.3f}")
        if remaining:
            print(f"  Remaining: {remaining}")

        # Helper to finalize success
        def make_success_result(final_path, final_soft_cost, final_min_conf):
            used_soft = []
            for step in final_path:
                if step.startswith("SoftFact"):
                    # step format: SoftFact <num> (conf=...)
                    parts = step.split()
                    if len(parts) >= 2:
                        try:
                            num = int(parts[1])
                        except ValueError:
                            num = None
                        used_soft.append(("fact", num))
                elif step.startswith("SoftRule"):
                    # step format: SoftRule <num> (conf=...)
                    parts = step.split()
                    if len(parts) >= 2:
                        try:
                            num = int(parts[1])
                        except ValueError:
                            num = None
                        used_soft.append(("rule", num))

            return {
                "success": True,
                "proof_path": final_path,
                "used_soft_clauses": used_soft,
                "soft_cost": final_soft_cost,
                "min_conf": final_min_conf if final_soft_cost > 0 else None
            }

        # --- 1) HARD facts: exact match ---
        for num, fact in hard_facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Hard Fact {num} matches exactly: {fact}")

                step_label = f"HardFact {num}"
                new_path = path + [step_label]

                if not remaining:
                    print(f"✓✓ SOFT-BFS SUCCESS (hard-only) at depth {depth + 1}")
                    return make_success_result(new_path, soft_cost, min_conf)

                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, new_path,
                              depth + 1, soft_cost, min_conf))
                # Since we found a hard fact, we can move to next BFS state
                # (We do not 'continue' here to allow exploring other options in this node too,
                #  but you could short-circuit if you want strict BFS semantics.)
                break

        # --- 2) HARD facts: unification with variables (LLM) ---
        for num, fact in hard_facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            print(f"  ✓ Hard Fact {num} unifies: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)

            step_label = f"HardFact {num}"
            new_path = path + [step_label]

            if not instantiated:
                print(f"✓✓ SOFT-BFS SUCCESS (hard-only) at depth {depth + 1}")
                return make_success_result(new_path, soft_cost, min_conf)

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, new_path,
                          depth + 1, soft_cost, min_conf))

        # --- 3) HARD rules (via SLD, LLM-backed) ---
        matching_hard_rules = find_matching_rules_only(current, hard_rules)

        if matching_hard_rules:
            print(f"  Matching hard rules: {matching_hard_rules}")

        for rule_num in matching_hard_rules:
            for num, head, body in hard_rules:
                if num != rule_num:
                    continue

                subgoals = get_subgoals(current, head, body)
                if not subgoals:
                    continue

                print(f"  Hard Rule {num}: {head} :- {body}")
                print(f"    → {subgoals}")

                all_goals = subgoals + remaining
                next_goal = all_goals[0]
                next_remaining = all_goals[1:]
                step_label = f"HardRule {num}"
                new_path = path + [step_label]

                queue.append((next_goal, next_remaining, new_path,
                              depth + 1, soft_cost, min_conf))
                break

        # --- 4) SOFT facts (hypotheses) ---
        for s_num, s_atom, s_conf in soft_facts:
            # If there's a limit on how many soft clauses we can use, enforce it
            if max_soft is not None and soft_cost >= max_soft:
                break

            bindings = unify_with_fact(current, s_atom)
            if bindings is None:
                continue

            new_soft_cost = soft_cost + 1
            new_min_conf = min(min_conf, s_conf)

            print(f"  ✓ Soft Fact {s_num} unifies: {s_atom}")
            print(f"    Bindings: {bindings}, conf={s_conf:.3f}")
            print(f"    New soft cost: {new_soft_cost}, new min_conf: {new_min_conf:.3f}")

            instantiated = apply_bindings(remaining, bindings)

            step_label = f"SoftFact {s_num} (conf={s_conf:.3f})"
            new_path = path + [step_label]

            if not instantiated:
                print(f"✓✓ SOFT-BFS SUCCESS (with soft facts) at depth {depth + 1}")
                return make_success_result(new_path, new_soft_cost, new_min_conf)

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, new_path,
                          depth + 1, new_soft_cost, new_min_conf))

        # --- 5) SOFT rules (hypotheses) ---
        # Use the syntactic matcher on the (num, head, body_str) projection
        matching_soft_rules = find_matching_rules_only(current, soft_rules_for_match)

        if matching_soft_rules:
            print(f"  Matching soft rules: {matching_soft_rules}")

        for rule_num in matching_soft_rules:
            # Enforce soft clause budget
            if max_soft is not None and soft_cost >= max_soft:
                break

            # Find full soft rule (with confidence)
            for s_num, s_head, s_body_str, s_conf in soft_rules:
                if s_num != rule_num:
                    continue

                subgoals = get_subgoals(current, s_head, s_body_str)
                if not subgoals:
                    continue

                new_soft_cost = soft_cost + 1
                new_min_conf = min(min_conf, s_conf)

                print(f"  Soft Rule {s_num}: {s_head} :- {s_body_str}")
                print(f"    → {subgoals}, conf={s_conf:.3f}")
                print(f"    New soft cost: {new_soft_cost}, new min_conf: {new_min_conf:.3f}")

                all_goals = subgoals + remaining
                next_goal = all_goals[0]
                next_remaining = all_goals[1:]
                step_label = f"SoftRule {s_num} (conf={s_conf:.3f})"
                new_path = path + [step_label]

                queue.append((next_goal, next_remaining, new_path,
                              depth + 1, new_soft_cost, new_min_conf))
                break

    # If we exhaust the queue, no proof was found even with soft clauses
    print("✗ SOFT-BFS FAILED (no proof found even with soft KB)")
    return {
        "success": False,
        "proof_path": [],
        "used_soft_clauses": [],
        "soft_cost": None,
        "min_conf": None
    }

def solve_with_background(
    goal: str,
    kb: str,
    max_depth: int = 10,
    max_soft=None,
    hard_result=None,
):
    """
    High-level pipeline:

      1) If hard_result is provided, use it (no extra BFS).
         Otherwise, run bfs_prolog_collect(goal, kb) once.

      2) If hard_result.success == True:
            -> HARD_SUCCESS, no background needed.

      3) If hard_result.success == False:
            -> generate_background_hypotheses
            -> attach_hypotheses_to_kb
            -> bfs_prolog_metro_soft

    Returns a dict:

        {
          "status": "HARD_SUCCESS" | "SOFT_SUCCESS" | "SOFT_FAILURE" | "FAILURE",
          "hard_result": {...},            # from bfs_prolog_collect
          "soft_result": {...} or None,    # from bfs_prolog_metro_soft
          "hypotheses": list[dict]
        }
    """

    print("\n========================================")
    print(f"SOLVE WITH BACKGROUND: {goal}")
    print("========================================\n")

    # 1) Hard-KB attempt (only if not provided)
    if hard_result is None:
        print(">>> Phase 1: Hard-KB BFS (bfs_prolog_collect)")
        hard_result = bfs_prolog_collect(goal, kb, max_depth=max_depth)
        print("Hard-KB result:", hard_result)
    else:
        print(">>> Phase 1: Hard-KB BFS result already computed, reusing it.")
        print("Hard-KB result:", hard_result)

    # If hard KB alone is enough, we are done
    if hard_result.get("success"):
        print("\n>>> Result: HARD_SUCCESS (no background hypotheses needed)\n")
        return {
            "status": "HARD_SUCCESS",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": []
        }

    # 2) If hard KB fails, generate background hypotheses for unresolved atoms
    unresolved_atoms = hard_result.get("unresolved_atoms", set())
    if not unresolved_atoms:
        print("\nNo unresolved atoms to explain; cannot generate hypotheses.")
        print(">>> Result: FAILURE\n")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": []
        }

    print("\n>>> Phase 2: Generate background hypotheses")
    print("Unresolved atoms:", unresolved_atoms)

    hypotheses = generate_background_hypotheses(
        goal=goal,
        kb=kb,
        unresolved_atoms=unresolved_atoms
    )

    # Robustness: if the function somehow returned None, treat as empty list
    if hypotheses is None:
        hypotheses = []

    if not hypotheses:
        print("Hypotheses returned by LLM: []")
        print("\nLLM returned NO hypotheses; cannot build soft KB.")
        print(">>> Result: FAILURE\n")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": []
        }

    print("Hypotheses returned by LLM:")
    for h in hypotheses:
        print("  - Clause:", h.get("clause"),
              "| Conf:", h.get("confidence"),
              "| From atom:", h.get("from_atom"))

    # 3) Attach hypotheses to KB as soft clauses
    print("\n>>> Phase 3: Attach hypotheses to soft KB")
    soft_kb = attach_hypotheses_to_kb(kb, hypotheses)
    print("Soft KB facts:", soft_kb.get("facts", []))
    print("Soft KB rules:", soft_kb.get("rules", []))

    # 4) Soft BFS using hard + soft KB
    print("\n>>> Phase 4: Soft BFS (bfs_prolog_metro_soft)")
    soft_result = bfs_prolog_metro_soft(
        goal=goal,
        kb=kb,
        soft_kb=soft_kb,
        max_depth=max_depth,
        max_soft=max_soft,
    )
    print("Soft-BFS result:", soft_result)

    if soft_result.get("success"):
        print("\n>>> Result: SOFT_SUCCESS (proof found using background hypotheses)\n")
        return {
            "status": "SOFT_SUCCESS",
            "hard_result": hard_result,
            "soft_result": soft_result,
            "hypotheses": hypotheses
        }

    print("\n>>> Result: SOFT_FAILURE (no proof even with background hypotheses)\n")
    return {
        "status": "SOFT_FAILURE",
        "hard_result": hard_result,
        "soft_result": soft_result,
        "hypotheses": hypotheses
    }

def omit_facts_from_kb(kb: str, omit_numbers):
    """
    Return a new KB string with numbered lines in `omit_numbers` removed.

    kb           : original Prolog KB string with numbered clauses
    omit_numbers : iterable of ints, e.g. {3} to omit clause 3.
    """
    omit_numbers = set(omit_numbers)
    new_lines = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        num = int(m.group(1))
        if num in omit_numbers:
            continue

        new_lines.append(line)

    return "\n".join(new_lines)


# Natural language to Prolog using LLM, algo SLD resolution

import re
import json

def nl_kb_to_prolog_kb(nl_kb_text: str, start_index: int = 1) -> list[str]:
    """
    Convert a *pure natural-language* description of a domain + rules
    into a numbered Prolog knowledge base.

    Example usage:
        nl_kb_text = '''
        There is a subway system with stations: Union Square, Times Square,
        Grand Central, and Bryant Park.

        Union Square is connected to Times Square.
        Times Square is connected to Grand Central.

        If one station is connected to another, then the first is reachable from
        the second. If X is connected to Y and Y is reachable to Z, then X is
        reachable to Z.
        '''

        prolog_kb = nl_kb_to_prolog_kb(nl_kb_text)
        # -> ["1. connected(union_square, times_square).",
        #     "2. connected(times_square, grand_central).",
        #     "3. reachable(X, Y) :- connected(X, Y).",
        #     "4. reachable(X, Z) :- connected(X, Y), reachable(Y, Z)."]

    Inputs:
        nl_kb_text : str
            Entire KB in natural language (facts + rules, any order).
        start_index: int
            Line number to start from (default 1). Useful if you want to
            merge multiple NL → KB calls later.

    Returns:
        list[str]: numbered Prolog clauses, each of the form:
            "<n>. clause(...)."
    """

    nl_kb_text = (nl_kb_text or "").strip()
    if not nl_kb_text:
        return []

    # --- Build LLM prompt (no reliance on existing Prolog KB) ---
    prompt = f"""
You are a Prolog formalization assistant.

The user will give you a natural-language description of a small domain,
including objects, relationships, and logical rules.

Your job is to convert that description into a set of Prolog clauses
(facts and rules).

Guidelines:
- Use lowercase atoms for concrete entities (e.g. union_square, times_square,
  grand_central, bryant_park).
- Use uppercase identifiers for variables (e.g. X, Y, Z).
- Choose predicate names that are short, descriptive, and consistent,
  for example: connected/2, reachable/2, located_in/2, etc.
- A fact must look like:
    connected(times_square, bryant_park).
- A rule must look like:
    reachable(X, Z) :- connected(X, Y), reachable(Y, Z).
- Every clause MUST end with a single period '.'.
- Do NOT include any line numbers in your output clauses.
- Do NOT add explanations or comments in the Prolog code.

Here is the NATURAL LANGUAGE description of the knowledge base:

\"\"\"{nl_kb_text}\"\"\"

Respond ONLY in this JSON format (and nothing else):

{{
  "clauses": [
    {{
      "clause": "connected(union_square, times_square)."
    }},
    {{
      "clause": "reachable(X, Y) :- connected(X, Y)."
    }}
  ]
}}
"""

    raw = ask_llm(prompt).strip()
    try:
        data = json.loads(extract_first_json(raw))
    except Exception as e:
        print("[nl_kb_to_prolog_kb] JSON parse error:", e)
        print("Raw LLM output:", raw)
        return []

    raw_clauses = data.get("clauses", [])
    if not isinstance(raw_clauses, list):
        print("[nl_kb_to_prolog_kb] 'clauses' field is not a list:", raw_clauses)
        return []

    cleaned_clauses = []

    for item in raw_clauses:
        # item may be a string or {"clause": "..."}
        if isinstance(item, str):
            clause = item.strip()
        elif isinstance(item, dict):
            clause = (item.get("clause") or "").strip()
        else:
            continue

        if not clause:
            continue

        # Strip any numbering if the model sneaks it in
        m_num = re.match(r'^\s*(\d+)\.\s*(.+)$', clause)
        if m_num:
            clause = m_num.group(2).strip()

        # Ensure it ends with a single dot
        clause = clause.rstrip()
        if not clause.endswith('.'):
            clause = clause + "."
        else:
            # avoid cases like '...)).' becoming '...)).'
            clause = re.sub(r'\.+$', '.', clause)

        # Very basic sanity: must look like head. or head :- body.
        body_str = clause[:-1].strip()  # drop final dot
        if ':-' in body_str:
            head_part, body_part = body_str.split(':-', 1)
            head = head_part.strip()
        else:
            head = body_str

        # Parse head as a predicate using your existing helper
        parsed_head = parse_predicate(head)
        if parsed_head is None:
            print("[nl_kb_to_prolog_kb] Discarding unparsable clause:", clause)
            continue

        cleaned_clauses.append(clause)

    # --- Number clauses sequentially starting from start_index ---
    numbered_clauses = []
    next_num = start_index
    for clause in cleaned_clauses:
        numbered_clauses.append(f"{next_num}. {clause}")
        next_num += 1

    return numbered_clauses



# In[57]:


if __name__ == "__main__":
    kb = """
    1. connected(union_square, times_square).
    2. connected(times_square, grand_central).
    3. connected(grand_central, bryant_park).
    4. reachable(X, Y) :- connected(X, Y).
    5. reachable(X, Z) :- connected(X, Y), reachable(Y, Z).
    """

    print("===== FULL METRO KB =====")
    print(kb)
    print("====================================\n")

    # Omit fact 3 (the last edge) to force background reasoning
    kb_missing_3 = omit_facts_from_kb(kb, omit_numbers={1})

    print("===== METRO KB WITH FACT 3 REMOVED =====")
    print(kb_missing_3)
    print("========================================\n")

    test_goal = "reachable(union_square, bryant_park)"

    print("==============================")
    print(f"TEST QUERY: {test_goal}")
    print("==============================\n")

    # 1) Symbolic hard-KB BFS + collection
    print(">>> Running bfs_prolog_collect (hard-KB BFS)...")
    collect_result = bfs_prolog_collect(test_goal, kb_missing_3)
    print("Collect Result:", collect_result)
    print("\n----------------------------------------\n")

    # 2) If hard KB failed, run full solve-with-background pipeline,
    #    reusing the hard result to avoid duplicate BFS.
    print(">>> Running solve_with_background (full pipeline, reusing hard result)...")
    bg_result = solve_with_background(
        goal=test_goal,
        kb=kb_missing_3,
        max_depth=10,
        max_soft=None,
        hard_result=collect_result,
    )
    print("Solve-with-background Result:")
    print(bg_result)
    print("\n========================================\n")


# In[ ]:





# In[66]:


import ollama
from collections import deque
import re
import json
from typing import Optional

# --- Config / LLM setup ---

client = ollama.Client()

model = "gpt-oss:20b"
# model = "qwen:14b"

DEBUG = False  # set to True to print raw LLM outputs for debugging


def ask_llm(prompt: str) -> str:
    resp = client.generate(model=model, prompt=prompt, options={'temperature': 0.0})
    answer = resp.get('response', '')
    if "...done thinking." in answer:
        return answer.split("...done thinking.")[-1].strip()
    return answer


# --- Helpers for parsing / JSON ---

def extract_first_json(text: str) -> str:
    """
    Extract the first {...} JSON object from possibly messy text.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in: {text!r}")
    return match.group(0)




def parse_predicate(term: str):
    """
    Parse a simple Prolog predicate of the form:
        functor(arg1, arg2, ...)
    Returns (functor: str, args: list[str]) or None if parsing fails.
    """
    term = term.strip().rstrip('.')
    m = re.match(r'^([a-z_][a-zA-Z0-9_]*)\((.*)\)$', term)
    if not m:
        return None
    functor = m.group(1)
    args_raw = m.group(2).strip()
    if not args_raw:
        args = []
    else:
        # Simple arg split (no nested terms in this toy domain)
        args = [a.strip() for a in args_raw.split(',')]
    return functor, args


def is_variable(s: str) -> bool:
    """
    Prolog-ish variable check: starts with uppercase letter or '_'.
    """
    s = s.strip()
    return bool(s) and (s[0].isupper() or s[0] == '_')


# --- Core Prolog helpers ---

def check_exact_match(goal: str, fact: str) -> bool:
    """Check if goal matches fact exactly (no variables)."""
    return goal.strip().rstrip('.') == fact.strip().rstrip('.')

def unify_args(args_goal, args_fact, env=None):
    """
    Unify two argument lists (flat terms, no nesting) under an environment.

    args_goal: list[str]  from the GOAL predicate
    args_fact: list[str]  from the FACT/RULE-HEAD predicate
    env      : dict or None   existing bindings, e.g. {"X": "times_square"}

    Returns:
        - None if unification fails
        - env (possibly modified) if unification succeeds
    """
    if env is None:
        env = {}

    if len(args_goal) != len(args_fact):
        return None

    for g, f in zip(args_goal, args_fact):
        g = g.strip()
        f = f.strip()

        g_is_var = is_variable(g)
        f_is_var = is_variable(f)

        # both constants
        if not g_is_var and not f_is_var:
            if g != f:
                return None
            continue

        # goal var, fact const
        if g_is_var and not f_is_var:
            if g in env:
                if env[g] != f:
                    return None
            else:
                env[g] = f
            continue

        # goal const, fact var  (treat fact vars as wildcards)
        if not g_is_var and f_is_var:
            if f in env:
                if env[f] != g:
                    return None
            else:
                env[f] = g
            continue

        # both variables
        if g_is_var and f_is_var:
            if g in env and f in env:
                if env[g] != env[f]:
                    return None
            elif g in env:
                env[f] = env[g]
            elif f in env:
                env[g] = env[f]
            # else both unbound → no constraint
            continue

    return env


def unify_with_fact(goal: str, fact: str):
    """
    Purely algorithmic unification between a GOAL and a FACT (or rule head).

    Returns:
        None      -> NO unification
        {}        -> EXACT ground match (no variables)
        dict      -> bindings, e.g. {"Y": "times_square"}
    """

    parsed_goal = parse_predicate(goal)
    parsed_fact = parse_predicate(fact)
    if parsed_goal is None or parsed_fact is None:
        return None

    fun_g, args_g = parsed_goal
    fun_f, args_f = parsed_fact

    # Functor or arity mismatch
    if fun_g != fun_f or len(args_g) != len(args_f):
        return None

    # If they are exactly the same string (ignoring trailing dot), treat as EXACT
    if check_exact_match(goal, fact):
        return {}

    env = unify_args(args_g, args_f, env={})
    if env is None:
        return None

    return env

def apply_bindings(goals, bindings):
    """
    Apply variable bindings to goals using pure string/term substitution.

    goals: list[str]   e.g. ["reachable(Y, Z)", "connected(Z, X)"]
    bindings: dict     e.g. {"Y": "times_square"}

    Returns: list[str] of instantiated goals.
    """
    if not bindings or not goals:
        return goals

    new_goals = []

    for g in goals:
        parsed = parse_predicate(g)
        if parsed is None:
            # If we can't parse it as a predicate, leave as-is
            new_goals.append(g)
            continue

        functor, args = parsed
        new_args = []
        for a in args:
            a_stripped = a.strip()
            if is_variable(a_stripped) and a_stripped in bindings:
                new_args.append(bindings[a_stripped])
            else:
                new_args.append(a_stripped)

        new_goal = f"{functor}({', '.join(new_args)})"
        new_goals.append(new_goal)

    return new_goals




def find_matching_rules_only(goal, rules_list):
    """
    Find ONLY rules (not facts) whose HEAD can unify with the given goal.

    IMPORTANT: This version is purely syntactic: it only checks functor and arity.
    We do NOT ask the LLM here, to avoid mismatched heads like reachable/2
    being applied to connected/2 goals.

    rules_list: list[(num, head, body)]
    Returns: list[int] of rule numbers.
    """
    parsed_goal = parse_predicate(goal)
    if parsed_goal is None:
        return []
    fun_g, args_g = parsed_goal
    arity_g = len(args_g)

    matching = []
    for num, head, body in rules_list:
        parsed_head = parse_predicate(head)
        if parsed_head is None:
            continue
        fun_h, args_h = parsed_head
        if fun_h == fun_g and len(args_h) == arity_g:
            matching.append(num)
    return matching

def substitute_in_atom(atom: str, bindings: dict) -> str:
    """
    Apply variable bindings to a single Prolog atom, e.g.:

        atom     = "connected(X, Y)"
        bindings = {"X": "union_square", "Y": "bryant_park"}

    Returns:
        "connected(union_square, bryant_park)"
    """
    parsed = parse_predicate(atom)
    if parsed is None:
        return atom  # best-effort fallback

    functor, args = parsed
    new_args = []

    for a in args:
        a_stripped = a.strip()
        if is_variable(a_stripped) and a_stripped in bindings:
            new_args.append(bindings[a_stripped])
        else:
            new_args.append(a_stripped)

    return f"{functor}({', '.join(new_args)})"
def split_body_atoms(body_str: str):
    """
    Split a rule body like:
        "connected(X, Y), reachable(Y, Z)"
    into:
        ["connected(X, Y)", "reachable(Y, Z)"]

    It is parentheses-aware, so it will NOT split on commas that are
    inside argument lists.
    """
    body_str = body_str.strip()
    atoms = []
    current = []
    depth = 0  # parentheses nesting depth

    for ch in body_str:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth = max(depth - 1, 0)
            current.append(ch)
        elif ch == ',' and depth == 0:
            # top-level comma → split here
            atom = ''.join(current).strip()
            if atom:
                atoms.append(atom)
            current = []
        else:
            current.append(ch)

    # Flush the last atom
    atom = ''.join(current).strip()
    if atom:
        atoms.append(atom)

    return atoms


def get_subgoals(goal: str, rule_head: str, rule_body: str):
    """
    Algorithmic ONE-STEP SLD resolution:

    Given:
        goal      : e.g. "reachable(union_square, bryant_park)"
        rule_head : e.g. "reachable(X, Y)"
        rule_body : e.g. "connected(X, Y)"

    Steps:
      1. Parse goal and rule_head into (functor, args).
      2. If functor or arity differ -> rule cannot apply -> return None.
      3. Unify head args with goal args → bindings.
      4. If unification fails -> return None.
      5. Split rule_body on ',' into individual atoms.
      6. Apply bindings to each body atom.
      7. Return the list of instantiated subgoal atoms.

    Returns:
        - list[str] of subgoals, e.g. ["connected(union_square, bryant_park)"]
        - or None if the rule cannot be applied.
    """
    # 1) Parse goal and rule head
    parsed_goal = parse_predicate(goal)
    parsed_head = parse_predicate(rule_head)

    if parsed_goal is None or parsed_head is None:
        return None

    fun_g, args_g = parsed_goal
    fun_h, args_h = parsed_head

    # 2) Functor / arity mismatch => cannot use this rule
    if fun_g != fun_h or len(args_g) != len(args_h):
        return None

    # 3) Unify arguments (rule-head vars with goal terms)
    bindings = unify_arg_lists(args_h, args_g)
    if bindings is None:
        return None

    # 4) Split rule body into atoms
    body_str = rule_body.strip()
    if not body_str:
        # A rule with empty body would be strange, but handle it
        return []

    body_atoms = split_body_atoms(body_str)
    if not body_atoms:
        return []


    # 5) Apply bindings to each atom
    subgoals = [substitute_in_atom(atom, bindings) for atom in body_atoms]

    return subgoals if subgoals else None



# --- BFS Prolog engine ---

def bfs_prolog_metro(goal: str, kb: str, max_depth: int = 10) -> bool:
    """
    BFS with correct fact/rule distinction, using LLM to do:
    - unification with facts (but guarded by functor/arity)
    - applying bindings to remaining goals
    - generating instantiated subgoals
    """

    # Parse KB - separate facts and rules
    facts = []
    rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content = match.group(2).strip()

            if ':-' in content:
                head, body = content.split(':-', 1)
                rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                facts.append((num, content.rstrip('.')))

    # BFS queue: (current_goal, remaining_goals, path, depth)
    queue = deque([(goal, [], [], 0)])
    visited = set()

    print(f"\nGoal: {goal}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        if depth >= max_depth:
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")

        # 1) FIRST: Check for exact fact match
        fact_matched = False
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")

                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return True

                # Continue with remaining goals
                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))
                fact_matched = True
                break

        if fact_matched:
            continue

        # 2) SECOND: Check for fact unification with variables (non-exact)
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)

            if bindings is None:
                continue

            print(f"  ✓ Fact {num}: {fact}")
            print(f"    Bindings: {bindings}")

            # Apply bindings to remaining goals
            instantiated = apply_bindings(remaining, bindings)

            if not instantiated:
                print(f"✓✓ SUCCESS at depth {depth + 1}")
                return True

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))

        # 3) THIRD: Try rules (matching by functor & arity only)
        matching_rules = find_matching_rules_only(current, rules)

        if matching_rules:
            print(f"  Matching rules: {matching_rules}")

            for rule_num in matching_rules:
                for num, head, body in rules:
                    if num == rule_num:
                        subgoals = get_subgoals(current, head, body)

                        if subgoals:
                            print(f"  Rule {num}: → {subgoals}")
                            all_goals = subgoals + remaining
                            next_goal = all_goals[0]
                            next_remaining = all_goals[1:]
                            queue.append((next_goal, next_remaining, path + [f"Rule {num}"], depth + 1))
                        break

    print("✗ FAILED")
    return False

def bfs_prolog_collect(goal: str, kb: str, max_depth: int = 10):
    """
    Near-clone of bfs_prolog_metro, but:

    - Still uses unify_with_fact → LLM
    - Still uses apply_bindings → LLM
    - Still uses get_subgoals → LLM
    - Still uses functor/arity filtering for rules
    - Still prints identical debug output

    Differences:
        • Returns a dict:
            {
              "success": bool,
              "proof_path": [...],
              "unresolved_atoms": set()
            }

        • Tracks unresolved atoms when:
            - no exact fact matched
            - no fact unified
            - no rule applied
    """

    # --- Parse KB exactly like bfs_prolog_metro ---
    facts = []
    rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content = match.group(2).strip()

            if ':-' in content:
                head, body = content.split(':-', 1)
                rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                facts.append((num, content.rstrip('.')))

    # BFS state: (current_goal, remaining_goals, path, depth)
    queue = deque([(goal, [], [], 0)])
    visited = set()
    unresolved_atoms = set()

    print(f"\n[COLLECT] Goal: {goal}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        if depth >= max_depth:
            unresolved_atoms.add(current)
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")

        # Track whether ANY progress was made on this goal
        progress = False

        # --- 1) Exact fact match ---
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")

                step = f"Fact {num}"
                new_path = path + [step]

                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return {
                        "success": True,
                        "proof_path": new_path,
                        "unresolved_atoms": set()
                    }

                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, new_path, depth + 1))

                progress = True
                break

        if progress:
            continue

        # --- 2) Fact unification (LLM) ---
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            progress = True

            print(f"  ✓ Fact {num}: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)

            step = f"Fact {num}"
            new_path = path + [step]

            if not instantiated:
                print(f"✓✓ SUCCESS at depth {depth + 1}")
                return {
                    "success": True,
                    "proof_path": new_path,
                    "unresolved_atoms": set()
                }

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, new_path, depth + 1))

        if progress:
            continue

        # --- 3) Rule attempts (LLM SLD step) ---
        matching_rules = find_matching_rules_only(current, rules)

        if matching_rules:
            print(f"  Matching rules: {matching_rules}")

        for rule_num in matching_rules:
            for num, head, body in rules:
                if num == rule_num:
                    subgoals = get_subgoals(current, head, body)
                    if subgoals:
                        print(f"  Rule {num}: → {subgoals}")
                        progress = True
                        all_goals = subgoals + remaining
                        next_goal = all_goals[0]
                        next_remaining = all_goals[1:]
                        step = f"Rule {num}"
                        new_path = path + [step]
                        queue.append((next_goal, next_remaining, new_path, depth + 1))
                    break

        # --- If absolutely nothing worked, record unresolved ---
        if not progress:
            print(f"  ✗ No facts or rules apply to: {current}")
            unresolved_atoms.add(current)

    # BFS fully exhausted → failure
    print("✗ FAILED (collect mode)")
    return {
        "success": False,
        "proof_path": [],
        "unresolved_atoms": unresolved_atoms
    }


import re
import json

def generate_background_hypotheses(goal: str, kb: str, unresolved_atoms, max_atoms: int = 5):
    """
    Use the LLM's background/world knowledge to propose additional Prolog clauses
    (facts or rules) that might make the GOAL provable.

    ALWAYS returns a list (possibly empty).
    NEVER returns None.
    """

    hypotheses = []

    # --- 1) Filter unresolved atoms to simple, ground atoms only ---
    # e.g. keep "connected(grand_central, bryant_park)" but drop "connected(grand_central, Y)"
    atom_list = list(unresolved_atoms)
    ground_atoms = []
    for atom in atom_list:
        atom = atom.strip()
        if not atom:
            continue
        # must look like functor(...)
        if '(' not in atom or ')' not in atom:
            continue
        # crude check: if any token inside parens starts with uppercase or '_', treat as variable → skip
        inside = atom.split('(', 1)[1].rsplit(')', 1)[0]
        if re.search(r'\b[A-Z_]\w*\b', inside):
            continue
        ground_atoms.append(atom)

    if max_atoms is not None and len(ground_atoms) > max_atoms:
        ground_atoms = ground_atoms[:max_atoms]

    if not ground_atoms:
        print("[generate_background_hypotheses] No suitable ground atoms to query.")
        return []  # <--- explicit list, not None

    # --- 2) For each ground unresolved atom, ask the LLM for hypotheses ---
    for atom in ground_atoms:
        prompt = f"""
You are a cautious Prolog expert with access to real-world background knowledge.

We attempted to prove the following GOAL using ONLY the numbered Prolog
knowledge base given below:

GOAL:
  {goal}

KNOWLEDGE BASE (numbered clauses):
{kb}

During breadth-first SLD resolution, the proof FAILED because we could not
prove the following subgoal:
  {atom}

DOMAIN DESCRIPTION (important):
- The intended domain is a metro / subway network with stations and lines.
- We are reasoning about which stations are directly connected and which
  stations are reachable by traveling along metro lines.

SEMANTIC HINTS ABOUT PREDICATES (very important, treat as ground truth):

1) connected/2:
   - connected(A, B) means that station A and station B are DIRECTLY ADJACENT
     stops on the SAME metro line (no intermediate stations).
   - Two stations are connected(A, B) IF AND ONLY IF they are right next to
     each other on the same line.
   - Do NOT propose connected(A, B) for stations that are far apart on a line
     or that would require passing through other stations.

2) reachable/2:
   - reachable(A, B) means there exists a path from station A to station B
     by traveling along one or more connected/2 edges (possibly with transfers).
   - If A and B are directly connected, then reachable(A, B) is true.
   - If A is connected to some C and reachable(C, B) is true, then reachable(A, B)
     is also true (reachability is the transitive closure of connected/2).

Task:
Propose a SMALL set of additional Prolog clauses (facts or rules) that are
LIKELY to be true in the intended metro domain and that would help make the GOAL
provable. Think of these as "missing" facts or rules that could fill gaps in
the knowledge base.

Constraints:
- Each clause MUST be valid Prolog and MUST end with a period.
- You MUST NOT modify or delete any existing clauses in the KB.
- Use ONLY predicate names and arities that are compatible with the style of
  the existing KB (for example, connected/2, reachable/2, etc.).
- Your proposed clauses MUST respect the semantic hints above, especially
  the meaning of connected/2 as direct adjacency.
- If you are uncertain about the factual correctness of a clause, give it a lower confidence.
- You SHOULD prefer to return at least one plausible clause rather than an empty list.

Respond ONLY in this JSON format:

{{
  "hypotheses": [
    {{
      "clause": "connected(times_square, grand_central).",
      "confidence": 0.9
    }},
    {{
      "clause": "connected(times_square, bryant_park).",
      "confidence": 0.4
    }}
  ]
}}

If you truly have NO hypotheses, respond with:
{{ "hypotheses": [] }}
"""
        raw = ask_llm(prompt).strip()
        if DEBUG:
            print("\n[DEBUG generate_background_hypotheses]")
            print("Unresolved atom:", atom)
            print("Raw response:\n", raw)

        try:
            data = json.loads(extract_first_json(raw))
        except Exception as e:
            print("[generate_background_hypotheses] JSON parse error:", e)
            print("Raw LLM output:", raw)

            # Fix common Prolog backslash pattern like "\=" which is invalid JSON
            if "Invalid \\escape" in str(e):
                try:
                    fixed_raw = raw.replace("\\=", "\\\\=")
                    data = json.loads(extract_first_json(fixed_raw))
                except Exception as e2:
                    print("[generate_background_hypotheses] JSON parse error after fix:", e2)
                    continue
            else:
                continue  # different JSON error; skip this atom

        raw_hyps = data.get("hypotheses", [])
        if not isinstance(raw_hyps, list):
            print("[generate_background_hypotheses] 'hypotheses' not a list:", raw_hyps)
            continue

        for h in raw_hyps:
            clause = (h.get("clause") or "").strip()
            if not clause:
                continue

            # Normalize: ensure clause ends with a dot
            if not clause.endswith('.'):
                clause = clause + "."

            try:
                conf = float(h.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0

            hypotheses.append({
                "clause": clause,
                "confidence": conf,
                "from_atom": atom
            })

    # --- 3) Deduplicate clauses (keep highest-confidence version) ---
    dedup = {}
    for h in hypotheses:
        key = h["clause"]
        if key not in dedup or h["confidence"] > dedup[key]["confidence"]:
            dedup[key] = h

    return list(dedup.values())  # <--- ALWAYS returns a list

def _find_max_line_number_in_kb(kb: str) -> int:
    """
    Scan the numbered KB and return the maximum clause number seen.
    If no numbered lines are found, return 0.
    """
    max_num = 0
    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue
        num = int(m.group(1))
        if num > max_num:
            max_num = num
    return max_num


def _is_fact_clause(clause: str) -> bool:
    """
    Heuristic: a fact has no ':-' in it.
    Example: 'connected(times_square, grand_central).'
    """
    return ':-' not in clause


def _split_rule_clause(clause: str):
    """
    Split a rule clause 'Head :- Body.' into (head, body_str).

    Returns:
        head: 'reachable(X, Z)'
        body_str: 'connected(X, Y), reachable(Y, Z)'
    """
    # Remove trailing dot if present
    clause = clause.strip()
    if clause.endswith('.'):
        clause = clause[:-1]

    head_part, body_part = clause.split(':-', 1)
    head = head_part.strip()
    body_str = body_part.strip()
    return head, body_str


def attach_hypotheses_to_kb(kb: str, hypotheses):
    """
    Convert LLM-generated hypotheses into a 'soft KB' structure.

    Inputs:
        kb          : original numbered KB string
        hypotheses  : list[dict] from generate_background_hypotheses, where each dict has:
                        {
                          "clause": "prolog_clause_string_ending_with_dot.",
                          "confidence": float,
                          "from_atom": "unresolved atom"
                        }

    Returns:
        soft_kb: dict with two lists:
            {
              "facts": [
                  (num, atom_str, confidence),
                  ...
              ],
              "rules": [
                  (num, head_str, body_str, confidence),
                  ...
              ]
            }

        where:
          - num is a fresh line number (beyond any in the original KB),
          - atom_str is like "connected(times_square, grand_central)",
          - head_str is like "reachable(X, Z)",
          - body_str is like "connected(X, Y), reachable(Y, Z)".
    """

    soft_facts = []
    soft_rules = []

    # Start numbering hypotheses after the max existing KB line number
    max_num = _find_max_line_number_in_kb(kb)
    next_num = max_num + 1

    for h in hypotheses:
        clause = (h.get("clause") or "").strip()
        if not clause:
            continue

        conf = float(h.get("confidence", 0.0))

        # Normalize: ensure terminating dot
        if not clause.endswith('.'):
            clause = clause + '.'

        if _is_fact_clause(clause):
            # Example: "connected(times_square, grand_central)."
            atom = clause.rstrip('.').strip()
            soft_facts.append((next_num, atom, conf))
        else:
            # Example: "reachable(X, Z) :- connected(X, Y), reachable(Y, Z)."
            head, body_str = _split_rule_clause(clause)
            soft_rules.append((next_num, head, body_str, conf))

        next_num += 1

    soft_kb = {
        "facts": soft_facts,
        "rules": soft_rules,
    }
    return soft_kb

def bfs_prolog_metro_soft(
    goal: str,
    kb: str,
    soft_kb,
    max_depth: int = 10,
    max_soft: Optional[int] = None,
):
    """
    BFS SLD resolution that can use:
      - hard clauses from the original KB
      - soft clauses (hypotheses) from soft_kb

    soft_kb is the dict from attach_hypotheses_to_kb:
        {
          "facts": [(num, atom_str, confidence), ...],
          "rules": [(num, head_str, body_str, confidence), ...]
        }

    Returns a dict:
        {
          "success": bool,
          "proof_path": list,             # list of step labels
          "used_soft_clauses": list,      # list of (kind, num, confidence)
          "soft_cost": int,               # how many soft clauses used
          "min_conf": float | None        # min confidence over used soft clauses
        }
    """

    # --- Parse hard KB exactly like bfs_prolog_metro ---
    hard_facts = []
    hard_rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content = match.group(2).strip()

            if ':-' in content:
                head, body = content.split(':-', 1)
                hard_rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                hard_facts.append((num, content.rstrip('.')))

    # --- Soft KB unpack ---
    soft_facts = soft_kb.get("facts", [])  # list of (num, atom, conf)
    soft_rules = soft_kb.get("rules", [])  # list of (num, head, body_str, conf)

    # For rule matching, we want (num, head, body_str) lists
    soft_rules_for_match = [(num, head, body_str) for (num, head, body_str, conf) in soft_rules]

    # BFS queue: (current_goal, remaining_goals, path, depth, soft_cost, min_conf)
    #   path is a list of human-readable step labels (strings) like:
    #     "HardFact 1", "SoftFact 1001 (conf=0.90)", "HardRule 5", etc.
    queue = deque([(goal, [], [], 0, 0, 1.0)])  # min_conf starts at 1.0
    visited = set()

    print(f"\n[SOFT BFS] Goal: {goal}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth, soft_cost, min_conf = queue.popleft()

        if depth >= max_depth:
            continue

        # You can choose how much of state goes into 'visited'.
        # Here we include soft_cost to distinguish "cheaper" vs "more expensive" paths.
        state = (current, tuple(remaining), soft_cost)
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        print(f"  Soft cost: {soft_cost}, min_conf: {min_conf:.3f}")
        if remaining:
            print(f"  Remaining: {remaining}")

        # Helper to finalize success
        def make_success_result(final_path, final_soft_cost, final_min_conf):
            used_soft = []
            for step in final_path:
                if step.startswith("SoftFact"):
                    # step format: SoftFact <num> (conf=...)
                    parts = step.split()
                    if len(parts) >= 2:
                        try:
                            num = int(parts[1])
                        except ValueError:
                            num = None
                        used_soft.append(("fact", num))
                elif step.startswith("SoftRule"):
                    # step format: SoftRule <num> (conf=...)
                    parts = step.split()
                    if len(parts) >= 2:
                        try:
                            num = int(parts[1])
                        except ValueError:
                            num = None
                        used_soft.append(("rule", num))

            return {
                "success": True,
                "proof_path": final_path,
                "used_soft_clauses": used_soft,
                "soft_cost": final_soft_cost,
                "min_conf": final_min_conf if final_soft_cost > 0 else None
            }

        # --- 1) HARD facts: exact match ---
        for num, fact in hard_facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Hard Fact {num} matches exactly: {fact}")

                step_label = f"HardFact {num}"
                new_path = path + [step_label]

                if not remaining:
                    print(f"✓✓ SOFT-BFS SUCCESS (hard-only) at depth {depth + 1}")
                    return make_success_result(new_path, soft_cost, min_conf)

                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, new_path,
                              depth + 1, soft_cost, min_conf))
                # Since we found a hard fact, we can move to next BFS state
                # (We do not 'continue' here to allow exploring other options in this node too,
                #  but you could short-circuit if you want strict BFS semantics.)
                break

        # --- 2) HARD facts: unification with variables (LLM) ---
        for num, fact in hard_facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            print(f"  ✓ Hard Fact {num} unifies: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)

            step_label = f"HardFact {num}"
            new_path = path + [step_label]

            if not instantiated:
                print(f"✓✓ SOFT-BFS SUCCESS (hard-only) at depth {depth + 1}")
                return make_success_result(new_path, soft_cost, min_conf)

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, new_path,
                          depth + 1, soft_cost, min_conf))

        # --- 3) HARD rules (via SLD, LLM-backed) ---
        matching_hard_rules = find_matching_rules_only(current, hard_rules)

        if matching_hard_rules:
            print(f"  Matching hard rules: {matching_hard_rules}")

        for rule_num in matching_hard_rules:
            for num, head, body in hard_rules:
                if num != rule_num:
                    continue

                subgoals = get_subgoals(current, head, body)
                if not subgoals:
                    continue

                print(f"  Hard Rule {num}: {head} :- {body}")
                print(f"    → {subgoals}")

                all_goals = subgoals + remaining
                next_goal = all_goals[0]
                next_remaining = all_goals[1:]
                step_label = f"HardRule {num}"
                new_path = path + [step_label]

                queue.append((next_goal, next_remaining, new_path,
                              depth + 1, soft_cost, min_conf))
                break

        # --- 4) SOFT facts (hypotheses) ---
        for s_num, s_atom, s_conf in soft_facts:
            # If there's a limit on how many soft clauses we can use, enforce it
            if max_soft is not None and soft_cost >= max_soft:
                break

            bindings = unify_with_fact(current, s_atom)
            if bindings is None:
                continue

            new_soft_cost = soft_cost + 1
            new_min_conf = min(min_conf, s_conf)

            print(f"  ✓ Soft Fact {s_num} unifies: {s_atom}")
            print(f"    Bindings: {bindings}, conf={s_conf:.3f}")
            print(f"    New soft cost: {new_soft_cost}, new min_conf: {new_min_conf:.3f}")

            instantiated = apply_bindings(remaining, bindings)

            step_label = f"SoftFact {s_num} (conf={s_conf:.3f})"
            new_path = path + [step_label]

            if not instantiated:
                print(f"✓✓ SOFT-BFS SUCCESS (with soft facts) at depth {depth + 1}")
                return make_success_result(new_path, new_soft_cost, new_min_conf)

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, new_path,
                          depth + 1, new_soft_cost, new_min_conf))

        # --- 5) SOFT rules (hypotheses) ---
        # Use the syntactic matcher on the (num, head, body_str) projection
        matching_soft_rules = find_matching_rules_only(current, soft_rules_for_match)

        if matching_soft_rules:
            print(f"  Matching soft rules: {matching_soft_rules}")

        for rule_num in matching_soft_rules:
            # Enforce soft clause budget
            if max_soft is not None and soft_cost >= max_soft:
                break

            # Find full soft rule (with confidence)
            for s_num, s_head, s_body_str, s_conf in soft_rules:
                if s_num != rule_num:
                    continue

                subgoals = get_subgoals(current, s_head, s_body_str)
                if not subgoals:
                    continue

                new_soft_cost = soft_cost + 1
                new_min_conf = min(min_conf, s_conf)

                print(f"  Soft Rule {s_num}: {s_head} :- {s_body_str}")
                print(f"    → {subgoals}, conf={s_conf:.3f}")
                print(f"    New soft cost: {new_soft_cost}, new min_conf: {new_min_conf:.3f}")

                all_goals = subgoals + remaining
                next_goal = all_goals[0]
                next_remaining = all_goals[1:]
                step_label = f"SoftRule {s_num} (conf={s_conf:.3f})"
                new_path = path + [step_label]

                queue.append((next_goal, next_remaining, new_path,
                              depth + 1, new_soft_cost, new_min_conf))
                break

    # If we exhaust the queue, no proof was found even with soft clauses
    print("✗ SOFT-BFS FAILED (no proof found even with soft KB)")
    return {
        "success": False,
        "proof_path": [],
        "used_soft_clauses": [],
        "soft_cost": None,
        "min_conf": None
    }

def solve_with_background(
    goal: str,
    kb: str,
    max_depth: int = 10,
    max_soft=None,
    hard_result=None,
):
    """
    High-level pipeline:

      1) If hard_result is provided, use it (no extra BFS).
         Otherwise, run bfs_prolog_collect(goal, kb) once.

      2) If hard_result.success == True:
            -> HARD_SUCCESS, no background needed.

      3) If hard_result.success == False:
            -> generate_background_hypotheses
            -> attach_hypotheses_to_kb
            -> bfs_prolog_metro_soft

    Returns a dict:

        {
          "status": "HARD_SUCCESS" | "SOFT_SUCCESS" | "SOFT_FAILURE" | "FAILURE",
          "hard_result": {...},            # from bfs_prolog_collect
          "soft_result": {...} or None,    # from bfs_prolog_metro_soft
          "hypotheses": list[dict]
        }
    """

    print("\n========================================")
    print(f"SOLVE WITH BACKGROUND: {goal}")
    print("========================================\n")

    # 1) Hard-KB attempt (only if not provided)
    if hard_result is None:
        print(">>> Phase 1: Hard-KB BFS (bfs_prolog_collect)")
        hard_result = bfs_prolog_collect(goal, kb, max_depth=max_depth)
        print("Hard-KB result:", hard_result)
    else:
        print(">>> Phase 1: Hard-KB BFS result already computed, reusing it.")
        print("Hard-KB result:", hard_result)

    # If hard KB alone is enough, we are done
    if hard_result.get("success"):
        print("\n>>> Result: HARD_SUCCESS (no background hypotheses needed)\n")
        return {
            "status": "HARD_SUCCESS",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": []
        }

    # 2) If hard KB fails, generate background hypotheses for unresolved atoms
    unresolved_atoms = hard_result.get("unresolved_atoms", set())
    if not unresolved_atoms:
        print("\nNo unresolved atoms to explain; cannot generate hypotheses.")
        print(">>> Result: FAILURE\n")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": []
        }

    print("\n>>> Phase 2: Generate background hypotheses")
    print("Unresolved atoms:", unresolved_atoms)

    hypotheses = generate_background_hypotheses(
        goal=goal,
        kb=kb,
        unresolved_atoms=unresolved_atoms
    )

    # Robustness: if the function somehow returned None, treat as empty list
    if hypotheses is None:
        hypotheses = []

    if not hypotheses:
        print("Hypotheses returned by LLM: []")
        print("\nLLM returned NO hypotheses; cannot build soft KB.")
        print(">>> Result: FAILURE\n")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": []
        }

    print("Hypotheses returned by LLM:")
    for h in hypotheses:
        print("  - Clause:", h.get("clause"),
              "| Conf:", h.get("confidence"),
              "| From atom:", h.get("from_atom"))

    # 3) Attach hypotheses to KB as soft clauses
    print("\n>>> Phase 3: Attach hypotheses to soft KB")
    soft_kb = attach_hypotheses_to_kb(kb, hypotheses)
    print("Soft KB facts:", soft_kb.get("facts", []))
    print("Soft KB rules:", soft_kb.get("rules", []))

    # 4) Soft BFS using hard + soft KB
    print("\n>>> Phase 4: Soft BFS (bfs_prolog_metro_soft)")
    soft_result = bfs_prolog_metro_soft(
        goal=goal,
        kb=kb,
        soft_kb=soft_kb,
        max_depth=max_depth,
        max_soft=max_soft,
    )
    print("Soft-BFS result:", soft_result)

    if soft_result.get("success"):
        print("\n>>> Result: SOFT_SUCCESS (proof found using background hypotheses)\n")
        return {
            "status": "SOFT_SUCCESS",
            "hard_result": hard_result,
            "soft_result": soft_result,
            "hypotheses": hypotheses
        }

    print("\n>>> Result: SOFT_FAILURE (no proof even with background hypotheses)\n")
    return {
        "status": "SOFT_FAILURE",
        "hard_result": hard_result,
        "soft_result": soft_result,
        "hypotheses": hypotheses
    }

def omit_facts_from_kb(kb: str, omit_numbers):
    """
    Return a new KB string with numbered lines in `omit_numbers` removed.

    kb           : original Prolog KB string with numbered clauses
    omit_numbers : iterable of ints, e.g. {3} to omit clause 3.
    """
    omit_numbers = set(omit_numbers)
    new_lines = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        num = int(m.group(1))
        if num in omit_numbers:
            continue

        new_lines.append(line)

    return "\n".join(new_lines)


# Natural language to Prolog using LLM, algo SLD resolution

import re
import json

def nl_kb_to_prolog_kb(nl_kb_text: str, start_index: int = 1) -> list[str]:
    """
    Convert a *pure natural-language* description of a domain + rules
    into a numbered Prolog knowledge base.

    Example usage:
        nl_kb_text = '''
        There is a subway system with stations: Union Square, Times Square,
        Grand Central, and Bryant Park.

        Union Square is connected to Times Square.
        Times Square is connected to Grand Central.

        If one station is connected to another, then the first is reachable from
        the second. If X is connected to Y and Y is reachable to Z, then X is
        reachable to Z.
        '''

        prolog_kb = nl_kb_to_prolog_kb(nl_kb_text)
        # -> ["1. connected(union_square, times_square).",
        #     "2. connected(times_square, grand_central).",
        #     "3. reachable(X, Y) :- connected(X, Y).",
        #     "4. reachable(X, Z) :- connected(X, Y), reachable(Y, Z)."]

    Inputs:
        nl_kb_text : str
            Entire KB in natural language (facts + rules, any order).
        start_index: int
            Line number to start from (default 1). Useful if you want to
            merge multiple NL → KB calls later.

    Returns:
        list[str]: numbered Prolog clauses, each of the form:
            "<n>. clause(...)."
    """

    nl_kb_text = (nl_kb_text or "").strip()
    if not nl_kb_text:
        return []

    # --- Build LLM prompt (no reliance on existing Prolog KB) ---
    prompt = f"""
You are a Prolog formalization assistant.

The user will give you a natural-language description of a small domain,
including objects, relationships, and logical rules.

Your job is to convert that description into a set of Prolog clauses
(facts and rules).

Guidelines:
- Use lowercase atoms for concrete entities (e.g. union_square, times_square,
  grand_central, bryant_park).
- Use uppercase identifiers for variables (e.g. X, Y, Z).
- Choose predicate names that are short, descriptive, and consistent,
  for example: connected/2, reachable/2, located_in/2, etc.
- A fact must look like:
    connected(times_square, bryant_park).
- A rule must look like:
    reachable(X, Z) :- connected(X, Y), reachable(Y, Z).
- Every clause MUST end with a single period '.'.
- Do NOT include any line numbers in your output clauses.
- Do NOT add explanations or comments in the Prolog code.

Here is the NATURAL LANGUAGE description of the knowledge base:

\"\"\"{nl_kb_text}\"\"\"

Respond ONLY in this JSON format (and nothing else):

{{
  "clauses": [
    {{
      "clause": "connected(union_square, times_square)."
    }},
    {{
      "clause": "reachable(X, Y) :- connected(X, Y)."
    }}
  ]
}}
"""

    raw = ask_llm(prompt).strip()
    try:
        data = json.loads(extract_first_json(raw))
    except Exception as e:
        print("[nl_kb_to_prolog_kb] JSON parse error:", e)
        print("Raw LLM output:", raw)
        return []

    raw_clauses = data.get("clauses", [])
    if not isinstance(raw_clauses, list):
        print("[nl_kb_to_prolog_kb] 'clauses' field is not a list:", raw_clauses)
        return []

    cleaned_clauses = []

    for item in raw_clauses:
        # item may be a string or {"clause": "..."}
        if isinstance(item, str):
            clause = item.strip()
        elif isinstance(item, dict):
            clause = (item.get("clause") or "").strip()
        else:
            continue

        if not clause:
            continue

        # Strip any numbering if the model sneaks it in
        m_num = re.match(r'^\s*(\d+)\.\s*(.+)$', clause)
        if m_num:
            clause = m_num.group(2).strip()

        # Ensure it ends with a single dot
        clause = clause.rstrip()
        if not clause.endswith('.'):
            clause = clause + "."
        else:
            # avoid cases like '...)).' becoming '...)).'
            clause = re.sub(r'\.+$', '.', clause)

        # Very basic sanity: must look like head. or head :- body.
        body_str = clause[:-1].strip()  # drop final dot
        if ':-' in body_str:
            head_part, body_part = body_str.split(':-', 1)
            head = head_part.strip()
        else:
            head = body_str

        # Parse head as a predicate using your existing helper
        parsed_head = parse_predicate(head)
        if parsed_head is None:
            print("[nl_kb_to_prolog_kb] Discarding unparsable clause:", clause)
            continue

        cleaned_clauses.append(clause)

    # --- Number clauses sequentially starting from start_index ---
    numbered_clauses = []
    next_num = start_index
    for clause in cleaned_clauses:
        numbered_clauses.append(f"{next_num}. {clause}")
        next_num += 1

    return numbered_clauses



# In[67]:


if __name__ == "__main__":
    kb = """
    1. connected(union_square, times_square). # Connected means directly adjacent on the same metro line
    2. connected(times_square, grand_central).
    3. connected(grand_central, bryant_park).
    4. reachable(X, Y) :- connected(X, Y). # Reachable means able to get from one metro station to another
    5. reachable(X, Z) :- connected(X, Y), reachable(Y, Z).
    """

    print("===== FULL METRO KB =====")
    print(kb)
    print("====================================\n")

    # Omit fact 3 (the last edge) to force background reasoning
    kb_missing_3 = omit_facts_from_kb(kb, omit_numbers={1})

    print("===== METRO KB WITH FACT 3 REMOVED =====")
    print(kb_missing_3)
    print("========================================\n")

    test_goal = "reachable(union_square, bryant_park)"

    print("==============================")
    print(f"TEST QUERY: {test_goal}")
    print("==============================\n")

    # 1) Symbolic hard-KB BFS + collection
    print(">>> Running bfs_prolog_collect (hard-KB BFS)...")
    collect_result = bfs_prolog_collect(test_goal, kb_missing_3)
    print("Collect Result:", collect_result)
    print("\n----------------------------------------\n")

    # 2) If hard KB failed, run full solve-with-background pipeline,
    #    reusing the hard result to avoid duplicate BFS.
    print(">>> Running solve_with_background (full pipeline, reusing hard result)...")
    bg_result = solve_with_background(
        goal=test_goal,
        kb=kb_missing_3,
        max_depth=10,
        max_soft=None,
        hard_result=collect_result,
    )
    print("Solve-with-background Result:")
    print(bg_result)
    print("\n========================================\n")


# In[73]:


import ollama
from collections import deque
import re
import json
from typing import Optional

# --- Config / LLM setup ---

client = ollama.Client()

model = "gpt-oss:20b"
# model = "qwen:14b"

DEBUG = False  # set to True to print raw LLM outputs for debugging


def ask_llm(prompt: str) -> str:
    resp = client.generate(model=model, prompt=prompt, options={'temperature': 0.0})
    answer = resp.get('response', '')
    if "...done thinking." in answer:
        return answer.split("...done thinking.")[-1].strip()
    return answer


# --- Helpers for parsing / JSON ---

def extract_first_json(text: str) -> str:
    """
    Extract the first {...} JSON object from possibly messy text.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in: {text!r}")
    return match.group(0)


def split_inline_comment(s: str):
    """
    Split a string into (code, comment) at the first '#'.
    Returns comment WITHOUT the '#'. If no comment, comment=None.
    """
    if "#" not in s:
        return s.strip(), None
    code, comment = s.split("#", 1)
    code = code.strip()
    comment = comment.strip()
    return code, (comment if comment else None)


def parse_predicate(term: str):
    """
    Parse a simple Prolog predicate of the form:
        functor(arg1, arg2, ...)
    Returns (functor: str, args: list[str]) or None if parsing fails.
    """
    term = term.strip().rstrip('.')
    m = re.match(r'^([a-z_][a-zA-Z0-9_]*)\((.*)\)$', term)
    if not m:
        return None
    functor = m.group(1)
    args_raw = m.group(2).strip()
    if not args_raw:
        args = []
    else:
        # Simple arg split (no nested terms in this toy domain)
        args = [a.strip() for a in args_raw.split(',')]
    return functor, args


def is_variable(s: str) -> bool:
    """
    Prolog-ish variable check: starts with uppercase letter or '_'.
    """
    s = s.strip()
    return bool(s) and (s[0].isupper() or s[0] == '_')


# --- Core Prolog helpers ---

def check_exact_match(goal: str, fact: str) -> bool:
    """Check if goal matches fact exactly (no variables)."""
    return goal.strip().rstrip('.') == fact.strip().rstrip('.')


def unify_args(args_goal, args_fact, env=None):
    """
    Unify two argument lists (flat terms, no nesting) under an environment.

    args_goal: list[str]  from the GOAL predicate
    args_fact: list[str]  from the FACT/RULE-HEAD predicate
    env      : dict or None   existing bindings, e.g. {"X": "times_square"}

    Returns:
        - None if unification fails
        - env (possibly modified) if unification succeeds
    """
    if env is None:
        env = {}

    if len(args_goal) != len(args_fact):
        return None

    for g, f in zip(args_goal, args_fact):
        g = g.strip()
        f = f.strip()

        g_is_var = is_variable(g)
        f_is_var = is_variable(f)

        # both constants
        if not g_is_var and not f_is_var:
            if g != f:
                return None
            continue

        # goal var, fact const
        if g_is_var and not f_is_var:
            if g in env:
                if env[g] != f:
                    return None
            else:
                env[g] = f
            continue

        # goal const, fact var  (treat fact vars as wildcards)
        if not g_is_var and f_is_var:
            if f in env:
                if env[f] != g:
                    return None
            else:
                env[f] = g
            continue

        # both variables
        if g_is_var and f_is_var:
            if g in env and f in env:
                if env[g] != env[f]:
                    return None
            elif g in env:
                env[f] = env[g]
            elif f in env:
                env[g] = env[f]
            # else both unbound → no constraint
            continue

    return env


def unify_arg_lists(args_rule_head, args_goal):
    """
    Wrapper used by get_subgoals: unify rule-head args with goal args.
    """
    return unify_args(args_rule_head, args_goal, env={})


def unify_with_fact(goal: str, fact: str):
    """
    Purely algorithmic unification between a GOAL and a FACT (or rule head).

    Returns:
        None      -> NO unification
        {}        -> EXACT ground match (no variables)
        dict      -> bindings, e.g. {"Y": "times_square"}
    """

    parsed_goal = parse_predicate(goal)
    parsed_fact = parse_predicate(fact)
    if parsed_goal is None or parsed_fact is None:
        return None

    fun_g, args_g = parsed_goal
    fun_f, args_f = parsed_fact

    # Functor or arity mismatch
    if fun_g != fun_f or len(args_g) != len(args_f):
        return None

    # If they are exactly the same string (ignoring trailing dot), treat as EXACT
    if check_exact_match(goal, fact):
        return {}

    env = unify_args(args_g, args_f, env={})
    if env is None:
        return None

    return env


def apply_bindings(goals, bindings):
    """
    Apply variable bindings to goals using pure string/term substitution.

    goals: list[str]   e.g. ["reachable(Y, Z)", "connected(Z, X)"]
    bindings: dict     e.g. {"Y": "times_square"}

    Returns: list[str] of instantiated goals.
    """
    if not bindings or not goals:
        return goals

    new_goals = []

    for g in goals:
        parsed = parse_predicate(g)
        if parsed is None:
            # If we can't parse it as a predicate, leave as-is
            new_goals.append(g)
            continue

        functor, args = parsed
        new_args = []
        for a in args:
            a_stripped = a.strip()
            if is_variable(a_stripped) and a_stripped in bindings:
                new_args.append(bindings[a_stripped])
            else:
                new_args.append(a_stripped)

        new_goal = f"{functor}({', '.join(new_args)})"
        new_goals.append(new_goal)

    return new_goals


def find_matching_rules_only(goal, rules_list):
    """
    Find ONLY rules (not facts) whose HEAD can unify with the given goal.

    IMPORTANT: This version is purely syntactic: it only checks functor and arity.
    We do NOT ask the LLM here, to avoid mismatched heads like reachable/2
    being applied to connected/2 goals.

    rules_list: list[(num, head, body)]
    Returns: list[int] of rule numbers.
    """
    parsed_goal = parse_predicate(goal)
    if parsed_goal is None:
        return []
    fun_g, args_g = parsed_goal
    arity_g = len(args_g)

    matching = []
    for num, head, body in rules_list:
        parsed_head = parse_predicate(head)
        if parsed_head is None:
            continue
        fun_h, args_h = parsed_head
        if fun_h == fun_g and len(args_h) == arity_g:
            matching.append(num)
    return matching


def substitute_in_atom(atom: str, bindings: dict) -> str:
    """
    Apply variable bindings to a single Prolog atom, e.g.:

        atom     = "connected(X, Y)"
        bindings = {"X": "union_square", "Y": "bryant_park"}

    Returns:
        "connected(union_square, bryant_park)"
    """
    parsed = parse_predicate(atom)
    if parsed is None:
        return atom  # best-effort fallback

    functor, args = parsed
    new_args = []

    for a in args:
        a_stripped = a.strip()
        if is_variable(a_stripped) and a_stripped in bindings:
            new_args.append(bindings[a_stripped])
        else:
            new_args.append(a_stripped)

    return f"{functor}({', '.join(new_args)})"


def split_body_atoms(body_str: str):
    """
    Split a rule body like:
        "connected(X, Y), reachable(Y, Z)"
    into:
        ["connected(X, Y)", "reachable(Y, Z)"]

    It is parentheses-aware, so it will NOT split on commas that are
    inside argument lists.
    """
    body_str = body_str.strip()
    atoms = []
    current = []
    depth = 0  # parentheses nesting depth

    for ch in body_str:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth = max(depth - 1, 0)
            current.append(ch)
        elif ch == ',' and depth == 0:
            # top-level comma → split here
            atom = ''.join(current).strip()
            if atom:
                atoms.append(atom)
            current = []
        else:
            current.append(ch)

    # Flush the last atom
    atom = ''.join(current).strip()
    if atom:
        atoms.append(atom)

    return atoms


def get_subgoals(goal: str, rule_head: str, rule_body: str):
    """
    Algorithmic ONE-STEP SLD resolution (purely symbolic).
    """
    parsed_goal = parse_predicate(goal)
    parsed_head = parse_predicate(rule_head)

    if parsed_goal is None or parsed_head is None:
        return None

    fun_g, args_g = parsed_goal
    fun_h, args_h = parsed_head

    if fun_g != fun_h or len(args_g) != len(args_h):
        return None

    bindings = unify_arg_lists(args_h, args_g)
    if bindings is None:
        return None

    body_str = rule_body.strip()
    if not body_str:
        return []

    body_atoms = split_body_atoms(body_str)
    if not body_atoms:
        return []

    subgoals = [substitute_in_atom(atom, bindings) for atom in body_atoms]
    return subgoals if subgoals else None


# --- KB comment extraction (inline + full-line) ---

def parse_kb_predicate_comments(kb: str):
    """
    Supports BOTH:
      - full-line comments starting with '#'
      - inline comments after a numbered clause: '... . # comment'

    Returns:
        dict mapping "predicate/arity" -> comment string
        e.g. { "connected/2": "Connected means directly adjacent ..." }
    """
    predicate_comments = {}
    pending_comments = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Full-line comment
        if line.startswith("#"):
            pending_comments.append(line.lstrip("#").strip())
            continue

        # Numbered clause
        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        content_raw = m.group(2).strip()
        content, inline_comment = split_inline_comment(content_raw)

        clause = content.strip().rstrip(".")
        head_str = clause.split(":-", 1)[0].strip()
        parsed = parse_predicate(head_str)
        if parsed is None:
            pending_comments = []
            continue

        functor, args = parsed
        key = f"{functor}/{len(args)}"

        combined = []
        if pending_comments:
            combined.append(" ".join(pending_comments))
        if inline_comment:
            combined.append(inline_comment)

        if combined:
            predicate_comments[key] = " ".join(combined).strip()

        pending_comments = []

    return predicate_comments


# --- BFS Prolog engine ---

def bfs_prolog_metro(goal: str, kb: str, max_depth: int = 10) -> bool:
    """
    BFS with correct fact/rule distinction.
    """

    facts = []
    rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content_raw = match.group(2).strip()
            content, _ = split_inline_comment(content_raw)

            if not content:
                continue

            if ':-' in content:
                head, body = content.split(':-', 1)
                rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                facts.append((num, content.rstrip('.')))

    queue = deque([(goal, [], [], 0)])
    visited = set()

    print(f"\nGoal: {goal}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        if depth >= max_depth:
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")

        # 1) Exact fact match
        fact_matched = False
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")

                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return True

                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))
                fact_matched = True
                break

        if fact_matched:
            continue

        # 2) Fact unification
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            print(f"  ✓ Fact {num}: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)

            if not instantiated:
                print(f"✓✓ SUCCESS at depth {depth + 1}")
                return True

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))

        # 3) Rules
        matching_rules = find_matching_rules_only(current, rules)
        if matching_rules:
            print(f"  Matching rules: {matching_rules}")

            for rule_num in matching_rules:
                for num, head, body in rules:
                    if num == rule_num:
                        subgoals = get_subgoals(current, head, body)
                        if subgoals:
                            print(f"  Rule {num}: → {subgoals}")
                            all_goals = subgoals + remaining
                            next_goal = all_goals[0]
                            next_remaining = all_goals[1:]
                            queue.append((next_goal, next_remaining, path + [f"Rule {num}"], depth + 1))
                        break

    print("✗ FAILED")
    return False


def bfs_prolog_collect(goal: str, kb: str, max_depth: int = 10):
    """
    Like bfs_prolog_metro, but returns unresolved atoms too.
    """

    facts = []
    rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content_raw = match.group(2).strip()
            content, _ = split_inline_comment(content_raw)

            if not content:
                continue

            if ':-' in content:
                head, body = content.split(':-', 1)
                rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                facts.append((num, content.rstrip('.')))

    queue = deque([(goal, [], [], 0)])
    visited = set()
    unresolved_atoms = set()

    print(f"\n[COLLECT] Goal: {goal}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        if depth >= max_depth:
            unresolved_atoms.add(current)
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")

        progress = False

        # 1) Exact fact match
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")

                step = f"Fact {num}"
                new_path = path + [step]

                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return {
                        "success": True,
                        "proof_path": new_path,
                        "unresolved_atoms": set()
                    }

                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, new_path, depth + 1))

                progress = True
                break

        if progress:
            continue

        # 2) Fact unification
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            progress = True

            print(f"  ✓ Fact {num}: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)

            step = f"Fact {num}"
            new_path = path + [step]

            if not instantiated:
                print(f"✓✓ SUCCESS at depth {depth + 1}")
                return {
                    "success": True,
                    "proof_path": new_path,
                    "unresolved_atoms": set()
                }

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, new_path, depth + 1))

        if progress:
            continue

        # 3) Rules
        matching_rules = find_matching_rules_only(current, rules)
        if matching_rules:
            print(f"  Matching rules: {matching_rules}")

        for rule_num in matching_rules:
            for num, head, body in rules:
                if num == rule_num:
                    subgoals = get_subgoals(current, head, body)
                    if subgoals:
                        print(f"  Rule {num}: → {subgoals}")
                        progress = True
                        all_goals = subgoals + remaining
                        next_goal = all_goals[0]
                        next_remaining = all_goals[1:]
                        step = f"Rule {num}"
                        new_path = path + [step]
                        queue.append((next_goal, next_remaining, new_path, depth + 1))
                    break

        if not progress:
            print(f"  ✗ No facts or rules apply to: {current}")
            unresolved_atoms.add(current)

    print("✗ FAILED (collect mode)")
    return {
        "success": False,
        "proof_path": [],
        "unresolved_atoms": unresolved_atoms
    }


def generate_background_hypotheses(goal: str, kb: str, unresolved_atoms, predicate_comments: dict, max_atoms: int = 5):
    """
    Keeps the LONGER, domain-specific prompt (close to your original),
    but injects predicate semantics from KB comments.
    """
    hypotheses = []

    # --- 1) Filter unresolved atoms to simple, ground atoms only ---
    atom_list = list(unresolved_atoms)
    ground_atoms = []
    for atom in atom_list:
        atom = atom.strip()
        if not atom:
            continue
        if '(' not in atom or ')' not in atom:
            continue
        inside = atom.split('(', 1)[1].rsplit(')', 1)[0]
        if re.search(r'\b[A-Z_]\w*\b', inside):
            continue
        ground_atoms.append(atom)

    if max_atoms is not None and len(ground_atoms) > max_atoms:
        ground_atoms = ground_atoms[:max_atoms]

    if not ground_atoms:
        print("[generate_background_hypotheses] No suitable ground atoms to query.")
        return []

    # --- 2) For each ground unresolved atom, ask the LLM for hypotheses ---
    for atom in ground_atoms:
        parsed = parse_predicate(atom)
        semantic_hint = ""
        if parsed is not None:
            functor, args = parsed
            pred_key = f"{functor}/{len(args)}"
            semantic_hint = predicate_comments.get(pred_key, "")

        prompt = f"""
You are a cautious Prolog expert with access to real-world background knowledge.

We attempted to prove the following GOAL using ONLY the numbered Prolog
knowledge base given below:

GOAL:
  {goal}

KNOWLEDGE BASE (numbered clauses):
{kb}

During breadth-first SLD resolution, the proof FAILED because we could not
prove the following subgoal:
  {atom}

DOMAIN DESCRIPTION (important):
- The intended domain is a metro / subway network with stations and lines.
- We are reasoning about which stations are directly connected and which
  stations are reachable by traveling along metro lines.

SEMANTIC HINTS ABOUT PREDICATES (author-provided; treat as ground truth):
{semantic_hint}

Task:
Propose a SMALL set of additional Prolog clauses (facts or rules) that are
LIKELY to be true in the intended metro domain and that would help make the GOAL
provable. Think of these as "missing" facts or rules that could fill gaps in
the knowledge base.

Constraints:
- Each clause MUST be valid Prolog and MUST end with a period.
- You MUST NOT modify or delete any existing clauses in the KB.
- Use ONLY predicate names and arities that are compatible with the style of
  the existing KB (for example, connected/2, reachable/2, etc.).
- Your proposed clauses MUST respect the semantic hints above.
- If you are uncertain about the factual correctness of a clause, give it a lower confidence.
- You SHOULD prefer to return at least one plausible clause rather than an empty list.

Respond ONLY in this JSON format:

{{
  "hypotheses": [
    {{
      "clause": "connected(times_square, grand_central).",
      "confidence": 0.9
    }},
    {{
      "clause": "connected(times_square, bryant_park).",
      "confidence": 0.4
    }}
  ]
}}

If you truly have NO hypotheses, respond with:
{{ "hypotheses": [] }}
"""
        raw = ask_llm(prompt).strip()
        if DEBUG:
            print("\n[DEBUG generate_background_hypotheses]")
            print("Unresolved atom:", atom)
            print("Raw response:\n", raw)

        try:
            data = json.loads(extract_first_json(raw))
        except Exception as e:
            print("[generate_background_hypotheses] JSON parse error:", e)
            print("Raw LLM output:", raw)

            if "Invalid \\escape" in str(e):
                try:
                    fixed_raw = raw.replace("\\=", "\\\\=")
                    data = json.loads(extract_first_json(fixed_raw))
                except Exception as e2:
                    print("[generate_background_hypotheses] JSON parse error after fix:", e2)
                    continue
            else:
                continue

        raw_hyps = data.get("hypotheses", [])
        if not isinstance(raw_hyps, list):
            print("[generate_background_hypotheses] 'hypotheses' not a list:", raw_hyps)
            continue

        for h in raw_hyps:
            clause = (h.get("clause") or "").strip()
            if not clause:
                continue

            if not clause.endswith('.'):
                clause = clause + "."

            try:
                conf = float(h.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0

            hypotheses.append({
                "clause": clause,
                "confidence": conf,
                "from_atom": atom
            })

    # --- 3) Deduplicate clauses (keep highest-confidence version) ---
    dedup = {}
    for h in hypotheses:
        key = h["clause"]
        if key not in dedup or h["confidence"] > dedup[key]["confidence"]:
            dedup[key] = h

    return list(dedup.values())


def _find_max_line_number_in_kb(kb: str) -> int:
    max_num = 0
    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        # strip inline comments here too
        content_raw = m.group(2).strip()
        content, _ = split_inline_comment(content_raw)
        if not content:
            continue

        num = int(m.group(1))
        if num > max_num:
            max_num = num
    return max_num


def _is_fact_clause(clause: str) -> bool:
    return ':-' not in clause


def _split_rule_clause(clause: str):
    clause = clause.strip()
    if clause.endswith('.'):
        clause = clause[:-1]
    head_part, body_part = clause.split(':-', 1)
    head = head_part.strip()
    body_str = body_part.strip()
    return head, body_str


def attach_hypotheses_to_kb(kb: str, hypotheses):
    soft_facts = []
    soft_rules = []

    max_num = _find_max_line_number_in_kb(kb)
    next_num = max_num + 1

    for h in hypotheses:
        clause = (h.get("clause") or "").strip()
        if not clause:
            continue

        conf = float(h.get("confidence", 0.0))

        if not clause.endswith('.'):
            clause = clause + '.'

        if _is_fact_clause(clause):
            atom = clause.rstrip('.').strip()
            soft_facts.append((next_num, atom, conf))
        else:
            head, body_str = _split_rule_clause(clause)
            soft_rules.append((next_num, head, body_str, conf))

        next_num += 1

    return {"facts": soft_facts, "rules": soft_rules}


def bfs_prolog_metro_soft(
    goal: str,
    kb: str,
    soft_kb,
    max_depth: int = 10,
    max_soft: Optional[int] = None,
):
    # --- Parse hard KB (strip inline comments) ---
    hard_facts = []
    hard_rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content_raw = match.group(2).strip()
            content, _ = split_inline_comment(content_raw)
            if not content:
                continue

            if ':-' in content:
                head, body = content.split(':-', 1)
                hard_rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                hard_facts.append((num, content.rstrip('.')))

    soft_facts = soft_kb.get("facts", [])
    soft_rules = soft_kb.get("rules", [])
    soft_rules_for_match = [(num, head, body_str) for (num, head, body_str, conf) in soft_rules]

    queue = deque([(goal, [], [], 0, 0, 1.0)])
    visited = set()

    print(f"\n[SOFT BFS] Goal: {goal}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth, soft_cost, min_conf = queue.popleft()

        if depth >= max_depth:
            continue

        state = (current, tuple(remaining), soft_cost)
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        print(f"  Soft cost: {soft_cost}, min_conf: {min_conf:.3f}")
        if remaining:
            print(f"  Remaining: {remaining}")

        def make_success_result(final_path, final_soft_cost, final_min_conf):
            used_soft = []
            for step in final_path:
                if step.startswith("SoftFact"):
                    parts = step.split()
                    if len(parts) >= 2:
                        try:
                            num = int(parts[1])
                        except ValueError:
                            num = None
                        used_soft.append(("fact", num))
                elif step.startswith("SoftRule"):
                    parts = step.split()
                    if len(parts) >= 2:
                        try:
                            num = int(parts[1])
                        except ValueError:
                            num = None
                        used_soft.append(("rule", num))

            return {
                "success": True,
                "proof_path": final_path,
                "used_soft_clauses": used_soft,
                "soft_cost": final_soft_cost,
                "min_conf": final_min_conf if final_soft_cost > 0 else None
            }

        # --- 1) HARD exact facts ---
        for num, fact in hard_facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Hard Fact {num} matches exactly: {fact}")

                step_label = f"HardFact {num}"
                new_path = path + [step_label]

                if not remaining:
                    print(f"✓✓ SOFT-BFS SUCCESS (hard-only) at depth {depth + 1}")
                    return make_success_result(new_path, soft_cost, min_conf)

                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, new_path,
                              depth + 1, soft_cost, min_conf))
                break

        # --- 2) HARD unification facts ---
        for num, fact in hard_facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            print(f"  ✓ Hard Fact {num} unifies: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)

            step_label = f"HardFact {num}"
            new_path = path + [step_label]

            if not instantiated:
                print(f"✓✓ SOFT-BFS SUCCESS (hard-only) at depth {depth + 1}")
                return make_success_result(new_path, soft_cost, min_conf)

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, new_path,
                          depth + 1, soft_cost, min_conf))

        # --- 3) HARD rules ---
        matching_hard_rules = find_matching_rules_only(current, hard_rules)
        if matching_hard_rules:
            print(f"  Matching hard rules: {matching_hard_rules}")

        for rule_num in matching_hard_rules:
            for num, head, body in hard_rules:
                if num != rule_num:
                    continue

                subgoals = get_subgoals(current, head, body)
                if not subgoals:
                    continue

                print(f"  Hard Rule {num}: {head} :- {body}")
                print(f"    → {subgoals}")

                all_goals = subgoals + remaining
                next_goal = all_goals[0]
                next_remaining = all_goals[1:]
                step_label = f"HardRule {num}"
                new_path = path + [step_label]

                queue.append((next_goal, next_remaining, new_path,
                              depth + 1, soft_cost, min_conf))
                break

        # --- 4) SOFT facts ---
        for s_num, s_atom, s_conf in soft_facts:
            if max_soft is not None and soft_cost >= max_soft:
                break

            bindings = unify_with_fact(current, s_atom)
            if bindings is None:
                continue

            new_soft_cost = soft_cost + 1
            new_min_conf = min(min_conf, s_conf)

            print(f"  ✓ Soft Fact {s_num} unifies: {s_atom}")
            print(f"    Bindings: {bindings}, conf={s_conf:.3f}")
            print(f"    New soft cost: {new_soft_cost}, new min_conf: {new_min_conf:.3f}")

            instantiated = apply_bindings(remaining, bindings)

            step_label = f"SoftFact {s_num} (conf={s_conf:.3f})"
            new_path = path + [step_label]

            if not instantiated:
                print(f"✓✓ SOFT-BFS SUCCESS (with soft facts) at depth {depth + 1}")
                return make_success_result(new_path, new_soft_cost, new_min_conf)

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, new_path,
                          depth + 1, new_soft_cost, new_min_conf))

        # --- 5) SOFT rules ---
        matching_soft_rules = find_matching_rules_only(current, soft_rules_for_match)
        if matching_soft_rules:
            print(f"  Matching soft rules: {matching_soft_rules}")

        for rule_num in matching_soft_rules:
            if max_soft is not None and soft_cost >= max_soft:
                break

            for s_num, s_head, s_body_str, s_conf in soft_rules:
                if s_num != rule_num:
                    continue

                subgoals = get_subgoals(current, s_head, s_body_str)
                if not subgoals:
                    continue

                new_soft_cost = soft_cost + 1
                new_min_conf = min(min_conf, s_conf)

                print(f"  Soft Rule {s_num}: {s_head} :- {s_body_str}")
                print(f"    → {subgoals}, conf={s_conf:.3f}")
                print(f"    New soft cost: {new_soft_cost}, new min_conf: {new_min_conf:.3f}")

                all_goals = subgoals + remaining
                next_goal = all_goals[0]
                next_remaining = all_goals[1:]
                step_label = f"SoftRule {s_num} (conf={s_conf:.3f})"
                new_path = path + [step_label]

                queue.append((next_goal, next_remaining, new_path,
                              depth + 1, new_soft_cost, new_min_conf))
                break

    print("✗ SOFT-BFS FAILED (no proof found even with soft KB)")
    return {
        "success": False,
        "proof_path": [],
        "used_soft_clauses": [],
        "soft_cost": None,
        "min_conf": None
    }


def solve_with_background(
    goal: str,
    kb: str,
    max_depth: int = 10,
    max_soft=None,
    hard_result=None,
):
    """
    High-level pipeline (unchanged), but now reads predicate comments from kb.
    """
    predicate_comments = parse_kb_predicate_comments(kb)

    print("\n========================================")
    print(f"SOLVE WITH BACKGROUND: {goal}")
    print("========================================\n")

    if hard_result is None:
        print(">>> Phase 1: Hard-KB BFS (bfs_prolog_collect)")
        hard_result = bfs_prolog_collect(goal, kb, max_depth=max_depth)
        print("Hard-KB result:", hard_result)
    else:
        print(">>> Phase 1: Hard-KB BFS result already computed, reusing it.")
        print("Hard-KB result:", hard_result)

    if hard_result.get("success"):
        print("\n>>> Result: HARD_SUCCESS (no background hypotheses needed)\n")
        return {
            "status": "HARD_SUCCESS",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": []
        }

    unresolved_atoms = hard_result.get("unresolved_atoms", set())
    if not unresolved_atoms:
        print("\nNo unresolved atoms to explain; cannot generate hypotheses.")
        print(">>> Result: FAILURE\n")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": []
        }

    print("\n>>> Phase 2: Generate background hypotheses")
    print("Unresolved atoms:", unresolved_atoms)

    hypotheses = generate_background_hypotheses(
        goal=goal,
        kb=kb,
        unresolved_atoms=unresolved_atoms,
        predicate_comments=predicate_comments
    )

    if hypotheses is None:
        hypotheses = []

    if not hypotheses:
        print("Hypotheses returned by LLM: []")
        print("\nLLM returned NO hypotheses; cannot build soft KB.")
        print(">>> Result: FAILURE\n")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": []
        }

    print("Hypotheses returned by LLM:")
    for h in hypotheses:
        print("  - Clause:", h.get("clause"),
              "| Conf:", h.get("confidence"),
              "| From atom:", h.get("from_atom"))

    print("\n>>> Phase 3: Attach hypotheses to soft KB")
    soft_kb = attach_hypotheses_to_kb(kb, hypotheses)
    print("Soft KB facts:", soft_kb.get("facts", []))
    print("Soft KB rules:", soft_kb.get("rules", []))

    print("\n>>> Phase 4: Soft BFS (bfs_prolog_metro_soft)")
    soft_result = bfs_prolog_metro_soft(
        goal=goal,
        kb=kb,
        soft_kb=soft_kb,
        max_depth=max_depth,
        max_soft=max_soft,
    )
    print("Soft-BFS result:", soft_result)

    if soft_result.get("success"):
        print("\n>>> Result: SOFT_SUCCESS (proof found using background hypotheses)\n")
        return {
            "status": "SOFT_SUCCESS",
            "hard_result": hard_result,
            "soft_result": soft_result,
            "hypotheses": hypotheses
        }

    print("\n>>> Result: SOFT_FAILURE (no proof even with background hypotheses)\n")
    return {
        "status": "SOFT_FAILURE",
        "hard_result": hard_result,
        "soft_result": soft_result,
        "hypotheses": hypotheses
    }


def omit_facts_from_kb(kb: str, omit_numbers):
    """
    Return a new KB string with numbered lines in `omit_numbers` removed.
    Preserves original line text (including comments), but matches by number.
    """
    omit_numbers = set(omit_numbers)
    new_lines = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        num = int(m.group(1))
        if num in omit_numbers:
            continue

        new_lines.append(line)

    return "\n".join(new_lines)


# Natural language to Prolog using LLM, algo SLD resolution

def nl_kb_to_prolog_kb(nl_kb_text: str, start_index: int = 1) -> list[str]:
    """
    Convert a *pure natural-language* description of a domain + rules
    into a numbered Prolog knowledge base.
    """

    nl_kb_text = (nl_kb_text or "").strip()
    if not nl_kb_text:
        return []

    prompt = f"""
You are a Prolog formalization assistant.

The user will give you a natural-language description of a small domain,
including objects, relationships, and logical rules.

Your job is to convert that description into a set of Prolog clauses
(facts and rules).

Guidelines:
- Use lowercase atoms for concrete entities (e.g. union_square, times_square,
  grand_central, bryant_park).
- Use uppercase identifiers for variables (e.g. X, Y, Z).
- Choose predicate names that are short, descriptive, and consistent,
  for example: connected/2, reachable/2, located_in/2, etc.
- A fact must look like:
    connected(times_square, bryant_park).
- A rule must look like:
    reachable(X, Z) :- connected(X, Y), reachable(Y, Z).
- Every clause MUST end with a single period '.'.
- Do NOT include any line numbers in your output clauses.
- Do NOT add explanations or comments in the Prolog code.

Here is the NATURAL LANGUAGE description of the knowledge base:

\"\"\"{nl_kb_text}\"\"\"

Respond ONLY in this JSON format (and nothing else):

{{
  "clauses": [
    {{
      "clause": "connected(union_square, times_square)."
    }},
    {{
      "clause": "reachable(X, Y) :- connected(X, Y)."
    }}
  ]
}}
"""

    raw = ask_llm(prompt).strip()
    try:
        data = json.loads(extract_first_json(raw))
    except Exception as e:
        print("[nl_kb_to_prolog_kb] JSON parse error:", e)
        print("Raw LLM output:", raw)
        return []

    raw_clauses = data.get("clauses", [])
    if not isinstance(raw_clauses, list):
        print("[nl_kb_to_prolog_kb] 'clauses' field is not a list:", raw_clauses)
        return []

    cleaned_clauses = []

    for item in raw_clauses:
        if isinstance(item, str):
            clause = item.strip()
        elif isinstance(item, dict):
            clause = (item.get("clause") or "").strip()
        else:
            continue

        if not clause:
            continue

        m_num = re.match(r'^\s*(\d+)\.\s*(.+)$', clause)
        if m_num:
            clause = m_num.group(2).strip()

        clause = clause.rstrip()
        if not clause.endswith('.'):
            clause = clause + "."
        else:
            clause = re.sub(r'\.+$', '.', clause)

        body_str = clause[:-1].strip()
        if ':-' in body_str:
            head_part, body_part = body_str.split(':-', 1)
            head = head_part.strip()
        else:
            head = body_str

        parsed_head = parse_predicate(head)
        if parsed_head is None:
            print("[nl_kb_to_prolog_kb] Discarding unparsable clause:", clause)
            continue

        cleaned_clauses.append(clause)

    numbered_clauses = []
    next_num = start_index
    for clause in cleaned_clauses:
        numbered_clauses.append(f"{next_num}. {clause}")
        next_num += 1

    return numbered_clauses


# In[74]:


if __name__ == "__main__":
    kb = """
    1. connected(union_square, times_square). # connected/2: directly adjacent on the same metro line (no intermediate stops)
    2. connected(times_square, grand_central).
    3. connected(grand_central, bryant_park).
    4. reachable(X, Y) :- connected(X, Y). # reachable/2: there exists a path via one or more connected/2 edges
    5. reachable(X, Z) :- connected(X, Y), reachable(Y, Z).
    """

    print("===== FULL METRO KB =====")
    print(kb)
    print("====================================\n")

    # Omit fact 1 to force background reasoning (removes union_square -> times_square)
    kb_missing_1 = omit_facts_from_kb(kb, omit_numbers={1})

    print("===== METRO KB WITH FACT 1 REMOVED =====")
    print(kb_missing_1)
    print("========================================\n")

    test_goal = "reachable(union_square, bryant_park)"

    print("==============================")
    print(f"TEST QUERY: {test_goal}")
    print("==============================\n")

    print(">>> Running bfs_prolog_collect (hard-KB BFS)...")
    collect_result = bfs_prolog_collect(test_goal, kb_missing_1)
    print("Collect Result:", collect_result)
    print("\n----------------------------------------\n")

    print(">>> Running solve_with_background (full pipeline, reusing hard result)...")
    bg_result = solve_with_background(
        goal=test_goal,
        kb=kb_missing_1,
        max_depth=10,
        max_soft=None,
        hard_result=collect_result,
    )
    print("Solve-with-background Result:")
    print(bg_result)
    print("\n========================================\n")


# In[77]:


import ollama
import heapq
from itertools import count
from collections import deque
import re
import json
from typing import Optional


# --- Config / LLM setup ---

client = ollama.Client()

model = "gpt-oss:20b"
# model = "qwen:14b"

DEBUG = False  # set to True to print raw LLM outputs for debugging


def ask_llm(prompt: str) -> str:
    resp = client.generate(model=model, prompt=prompt, options={'temperature': 0.0})
    answer = resp.get('response', '')
    if "...done thinking." in answer:
        return answer.split("...done thinking.")[-1].strip()
    return answer


# --- Helpers for parsing / JSON ---

def extract_first_json(text: str) -> str:
    """
    Extract the first {...} JSON object from possibly messy text.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in: {text!r}")
    return match.group(0)


def split_inline_comment(s: str):
    """
    Split a string into (code, comment) at the first '#'.
    Returns comment WITHOUT the '#'. If no comment, comment=None.
    """
    if "#" not in s:
        return s.strip(), None
    code, comment = s.split("#", 1)
    code = code.strip()
    comment = comment.strip()
    return code, (comment if comment else None)


def parse_predicate(term: str):
    """
    Parse a simple Prolog predicate of the form:
        functor(arg1, arg2, ...)
    Returns (functor: str, args: list[str]) or None if parsing fails.
    """
    term = term.strip().rstrip('.')
    m = re.match(r'^([a-z_][a-zA-Z0-9_]*)\((.*)\)$', term)
    if not m:
        return None
    functor = m.group(1)
    args_raw = m.group(2).strip()
    if not args_raw:
        args = []
    else:
        # Simple arg split (no nested terms in this toy domain)
        args = [a.strip() for a in args_raw.split(',')]
    return functor, args


def is_variable(s: str) -> bool:
    """
    Prolog-ish variable check: starts with uppercase letter or '_'.
    """
    s = s.strip()
    return bool(s) and (s[0].isupper() or s[0] == '_')


# --- Core Prolog helpers ---

def check_exact_match(goal: str, fact: str) -> bool:
    """Check if goal matches fact exactly (no variables)."""
    return goal.strip().rstrip('.') == fact.strip().rstrip('.')


def unify_args(args_goal, args_fact, env=None):
    """
    Unify two argument lists (flat terms, no nesting) under an environment.

    args_goal: list[str]  from the GOAL predicate
    args_fact: list[str]  from the FACT/RULE-HEAD predicate
    env      : dict or None   existing bindings, e.g. {"X": "times_square"}

    Returns:
        - None if unification fails
        - env (possibly modified) if unification succeeds
    """
    if env is None:
        env = {}

    if len(args_goal) != len(args_fact):
        return None

    for g, f in zip(args_goal, args_fact):
        g = g.strip()
        f = f.strip()

        g_is_var = is_variable(g)
        f_is_var = is_variable(f)

        # both constants
        if not g_is_var and not f_is_var:
            if g != f:
                return None
            continue

        # goal var, fact const
        if g_is_var and not f_is_var:
            if g in env:
                if env[g] != f:
                    return None
            else:
                env[g] = f
            continue

        # goal const, fact var  (treat fact vars as wildcards)
        if not g_is_var and f_is_var:
            if f in env:
                if env[f] != g:
                    return None
            else:
                env[f] = g
            continue

        # both variables
        if g_is_var and f_is_var:
            if g in env and f in env:
                if env[g] != env[f]:
                    return None
            elif g in env:
                env[f] = env[g]
            elif f in env:
                env[g] = env[f]
            # else both unbound → no constraint
            continue

    return env


def unify_arg_lists(args_rule_head, args_goal):
    """
    Wrapper used by get_subgoals: unify rule-head args with goal args.
    """
    return unify_args(args_rule_head, args_goal, env={})


def unify_with_fact(goal: str, fact: str):
    """
    Purely algorithmic unification between a GOAL and a FACT (or rule head).

    Returns:
        None      -> NO unification
        {}        -> EXACT ground match (no variables)
        dict      -> bindings, e.g. {"Y": "times_square"}
    """

    parsed_goal = parse_predicate(goal)
    parsed_fact = parse_predicate(fact)
    if parsed_goal is None or parsed_fact is None:
        return None

    fun_g, args_g = parsed_goal
    fun_f, args_f = parsed_fact

    # Functor or arity mismatch
    if fun_g != fun_f or len(args_g) != len(args_f):
        return None

    # If they are exactly the same string (ignoring trailing dot), treat as EXACT
    if check_exact_match(goal, fact):
        return {}

    env = unify_args(args_g, args_f, env={})
    if env is None:
        return None

    return env


def apply_bindings(goals, bindings):
    """
    Apply variable bindings to goals using pure string/term substitution.

    goals: list[str]   e.g. ["reachable(Y, Z)", "connected(Z, X)"]
    bindings: dict     e.g. {"Y": "times_square"}

    Returns: list[str] of instantiated goals.
    """
    if not bindings or not goals:
        return goals

    new_goals = []

    for g in goals:
        parsed = parse_predicate(g)
        if parsed is None:
            # If we can't parse it as a predicate, leave as-is
            new_goals.append(g)
            continue

        functor, args = parsed
        new_args = []
        for a in args:
            a_stripped = a.strip()
            if is_variable(a_stripped) and a_stripped in bindings:
                new_args.append(bindings[a_stripped])
            else:
                new_args.append(a_stripped)

        new_goal = f"{functor}({', '.join(new_args)})"
        new_goals.append(new_goal)

    return new_goals


def find_matching_rules_only(goal, rules_list):
    """
    Find ONLY rules (not facts) whose HEAD can unify with the given goal.

    IMPORTANT: This version is purely syntactic: it only checks functor and arity.
    We do NOT ask the LLM here, to avoid mismatched heads like reachable/2
    being applied to connected/2 goals.

    rules_list: list[(num, head, body)]
    Returns: list[int] of rule numbers.
    """
    parsed_goal = parse_predicate(goal)
    if parsed_goal is None:
        return []
    fun_g, args_g = parsed_goal
    arity_g = len(args_g)

    matching = []
    for num, head, body in rules_list:
        parsed_head = parse_predicate(head)
        if parsed_head is None:
            continue
        fun_h, args_h = parsed_head
        if fun_h == fun_g and len(args_h) == arity_g:
            matching.append(num)
    return matching


def substitute_in_atom(atom: str, bindings: dict) -> str:
    """
    Apply variable bindings to a single Prolog atom, e.g.:

        atom     = "connected(X, Y)"
        bindings = {"X": "union_square", "Y": "bryant_park"}

    Returns:
        "connected(union_square, bryant_park)"
    """
    parsed = parse_predicate(atom)
    if parsed is None:
        return atom  # best-effort fallback

    functor, args = parsed
    new_args = []

    for a in args:
        a_stripped = a.strip()
        if is_variable(a_stripped) and a_stripped in bindings:
            new_args.append(bindings[a_stripped])
        else:
            new_args.append(a_stripped)

    return f"{functor}({', '.join(new_args)})"


def split_body_atoms(body_str: str):
    """
    Split a rule body like:
        "connected(X, Y), reachable(Y, Z)"
    into:
        ["connected(X, Y)", "reachable(Y, Z)"]

    It is parentheses-aware, so it will NOT split on commas that are
    inside argument lists.
    """
    body_str = body_str.strip()
    atoms = []
    current = []
    depth = 0  # parentheses nesting depth

    for ch in body_str:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth = max(depth - 1, 0)
            current.append(ch)
        elif ch == ',' and depth == 0:
            # top-level comma → split here
            atom = ''.join(current).strip()
            if atom:
                atoms.append(atom)
            current = []
        else:
            current.append(ch)

    # Flush the last atom
    atom = ''.join(current).strip()
    if atom:
        atoms.append(atom)

    return atoms


def get_subgoals(goal: str, rule_head: str, rule_body: str):
    """
    Algorithmic ONE-STEP SLD resolution (purely symbolic).
    """
    parsed_goal = parse_predicate(goal)
    parsed_head = parse_predicate(rule_head)

    if parsed_goal is None or parsed_head is None:
        return None

    fun_g, args_g = parsed_goal
    fun_h, args_h = parsed_head

    if fun_g != fun_h or len(args_g) != len(args_h):
        return None

    bindings = unify_arg_lists(args_h, args_g)
    if bindings is None:
        return None

    body_str = rule_body.strip()
    if not body_str:
        return []

    body_atoms = split_body_atoms(body_str)
    if not body_atoms:
        return []

    subgoals = [substitute_in_atom(atom, bindings) for atom in body_atoms]
    return subgoals if subgoals else None


# --- KB comment extraction (inline + full-line) ---

def parse_kb_predicate_comments(kb: str):
    """
    Supports BOTH:
      - full-line comments starting with '#'
      - inline comments after a numbered clause: '... . # comment'

    Returns:
        dict mapping "predicate/arity" -> comment string
        e.g. { "connected/2": "Connected means directly adjacent ..." }
    """
    predicate_comments = {}
    pending_comments = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Full-line comment
        if line.startswith("#"):
            pending_comments.append(line.lstrip("#").strip())
            continue

        # Numbered clause
        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        content_raw = m.group(2).strip()
        content, inline_comment = split_inline_comment(content_raw)

        clause = content.strip().rstrip(".")
        head_str = clause.split(":-", 1)[0].strip()
        parsed = parse_predicate(head_str)
        if parsed is None:
            pending_comments = []
            continue

        functor, args = parsed
        key = f"{functor}/{len(args)}"

        combined = []
        if pending_comments:
            combined.append(" ".join(pending_comments))
        if inline_comment:
            combined.append(inline_comment)

        if combined:
            predicate_comments[key] = " ".join(combined).strip()

        pending_comments = []

    return predicate_comments


# --- BFS Prolog engine ---

def bfs_prolog_metro(goal: str, kb: str, max_depth: int = 10) -> bool:
    """
    BFS with correct fact/rule distinction.
    """

    facts = []
    rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        line = strip_inline_comment(line).strip()
        if not line:
            continue
        
        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content_raw = match.group(2).strip()
            content, _ = split_inline_comment(content_raw)

            if not content:
                continue

            if ':-' in content:
                head, body = content.split(':-', 1)
                rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                facts.append((num, content.rstrip('.')))

    queue = deque([(goal, [], [], 0)])
    visited = set()

    print(f"\nGoal: {goal}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        if depth >= max_depth:
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")

        # 1) Exact fact match
        fact_matched = False
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")

                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return True

                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))
                fact_matched = True
                break

        if fact_matched:
            continue

        # 2) Fact unification
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            print(f"  ✓ Fact {num}: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)

            if not instantiated:
                print(f"✓✓ SUCCESS at depth {depth + 1}")
                return True

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))

        # 3) Rules
        matching_rules = find_matching_rules_only(current, rules)
        if matching_rules:
            print(f"  Matching rules: {matching_rules}")

            for rule_num in matching_rules:
                for num, head, body in rules:
                    if num == rule_num:
                        subgoals = get_subgoals(current, head, body)
                        if subgoals:
                            print(f"  Rule {num}: → {subgoals}")
                            all_goals = subgoals + remaining
                            next_goal = all_goals[0]
                            next_remaining = all_goals[1:]
                            queue.append((next_goal, next_remaining, path + [f"Rule {num}"], depth + 1))
                        break

    print("✗ FAILED")
    return False


def bfs_prolog_collect(goal: str, kb: str, max_depth: int = 10):
    """
    Like bfs_prolog_metro, but returns unresolved atoms too.
    """

    facts = []
    rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        line = strip_inline_comment(line).strip()
        if not line:
            continue
        
        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content_raw = match.group(2).strip()
            content, _ = split_inline_comment(content_raw)

            if not content:
                continue

            if ':-' in content:
                head, body = content.split(':-', 1)
                rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                facts.append((num, content.rstrip('.')))

    queue = deque([(goal, [], [], 0)])
    visited = set()
    unresolved_atoms = set()

    print(f"\n[COLLECT] Goal: {goal}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        if depth >= max_depth:
            unresolved_atoms.add(current)
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")

        progress = False

        # 1) Exact fact match
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")

                step = f"Fact {num}"
                new_path = path + [step]

                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return {
                        "success": True,
                        "proof_path": new_path,
                        "unresolved_atoms": set()
                    }

                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, new_path, depth + 1))

                progress = True
                break

        if progress:
            continue

        # 2) Fact unification
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            progress = True

            print(f"  ✓ Fact {num}: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)

            step = f"Fact {num}"
            new_path = path + [step]

            if not instantiated:
                print(f"✓✓ SUCCESS at depth {depth + 1}")
                return {
                    "success": True,
                    "proof_path": new_path,
                    "unresolved_atoms": set()
                }

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, new_path, depth + 1))

        if progress:
            continue

        # 3) Rules
        matching_rules = find_matching_rules_only(current, rules)
        if matching_rules:
            print(f"  Matching rules: {matching_rules}")

        for rule_num in matching_rules:
            for num, head, body in rules:
                if num == rule_num:
                    subgoals = get_subgoals(current, head, body)
                    if subgoals:
                        print(f"  Rule {num}: → {subgoals}")
                        progress = True
                        all_goals = subgoals + remaining
                        next_goal = all_goals[0]
                        next_remaining = all_goals[1:]
                        step = f"Rule {num}"
                        new_path = path + [step]
                        queue.append((next_goal, next_remaining, new_path, depth + 1))
                    break

        if not progress:
            print(f"  ✗ No facts or rules apply to: {current}")
            unresolved_atoms.add(current)

    print("✗ FAILED (collect mode)")
    return {
        "success": False,
        "proof_path": [],
        "unresolved_atoms": unresolved_atoms
    }


def generate_background_hypotheses(goal: str, kb: str, unresolved_atoms, predicate_comments: dict, max_atoms: int = 5):
    """
    Keeps the LONGER, domain-specific prompt (close to your original),
    but injects predicate semantics from KB comments.
    """
    hypotheses = []

    # --- 1) Filter unresolved atoms to simple, ground atoms only ---
    atom_list = list(unresolved_atoms)
    ground_atoms = []
    for atom in atom_list:
        atom = atom.strip()
        if not atom:
            continue
        if '(' not in atom or ')' not in atom:
            continue
        inside = atom.split('(', 1)[1].rsplit(')', 1)[0]
        if re.search(r'\b[A-Z_]\w*\b', inside):
            continue
        ground_atoms.append(atom)

    if max_atoms is not None and len(ground_atoms) > max_atoms:
        ground_atoms = ground_atoms[:max_atoms]

    if not ground_atoms:
        print("[generate_background_hypotheses] No suitable ground atoms to query.")
        return []

    # --- 2) For each ground unresolved atom, ask the LLM for hypotheses ---
    for atom in ground_atoms:
        parsed = parse_predicate(atom)
        semantic_hint = ""
        if parsed is not None:
            functor, args = parsed
            pred_key = f"{functor}/{len(args)}"
            semantic_hint = predicate_comments.get(pred_key, "")

        prompt = f"""
You are a cautious Prolog expert with access to real-world background knowledge.

We attempted to prove the following GOAL using ONLY the numbered Prolog
knowledge base given below:

GOAL:
  {goal}

KNOWLEDGE BASE (numbered clauses):
{kb}

During breadth-first SLD resolution, the proof FAILED because we could not
prove the following subgoal:
  {atom}

DOMAIN DESCRIPTION (important):
- The intended domain is a metro / subway network with stations and lines.
- We are reasoning about which stations are directly connected and which
  stations are reachable by traveling along metro lines.

SEMANTIC HINTS ABOUT PREDICATES (author-provided; treat as ground truth):
{semantic_hint}

Task:
Propose a SMALL set of additional Prolog clauses (facts or rules) that are
LIKELY to be true in the intended metro domain and that would help make the GOAL
provable. Think of these as "missing" facts or rules that could fill gaps in
the knowledge base.

Constraints:
- Each clause MUST be valid Prolog and MUST end with a period.
- You MUST NOT modify or delete any existing clauses in the KB.
- Use ONLY predicate names and arities that are compatible with the style of
  the existing KB (for example, connected/2, reachable/2, etc.).
- Your proposed clauses MUST respect the semantic hints above.
- If you are uncertain about the factual correctness of a clause, give it a lower confidence.
- You SHOULD prefer to return at least one plausible clause rather than an empty list.

Respond ONLY in this JSON format:

{{
  "hypotheses": [
    {{
      "clause": "connected(times_square, grand_central).",
      "confidence": 0.9
    }},
    {{
      "clause": "connected(times_square, bryant_park).",
      "confidence": 0.4
    }}
  ]
}}

If you truly have NO hypotheses, respond with:
{{ "hypotheses": [] }}
"""
        raw = ask_llm(prompt).strip()
        if DEBUG:
            print("\n[DEBUG generate_background_hypotheses]")
            print("Unresolved atom:", atom)
            print("Raw response:\n", raw)

        try:
            data = json.loads(extract_first_json(raw))
        except Exception as e:
            print("[generate_background_hypotheses] JSON parse error:", e)
            print("Raw LLM output:", raw)

            if "Invalid \\escape" in str(e):
                try:
                    fixed_raw = raw.replace("\\=", "\\\\=")
                    data = json.loads(extract_first_json(fixed_raw))
                except Exception as e2:
                    print("[generate_background_hypotheses] JSON parse error after fix:", e2)
                    continue
            else:
                continue

        raw_hyps = data.get("hypotheses", [])
        if not isinstance(raw_hyps, list):
            print("[generate_background_hypotheses] 'hypotheses' not a list:", raw_hyps)
            continue

        for h in raw_hyps:
            clause = (h.get("clause") or "").strip()
            if not clause:
                continue

            if not clause.endswith('.'):
                clause = clause + "."

            try:
                conf = float(h.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0

            hypotheses.append({
                "clause": clause,
                "confidence": conf,
                "from_atom": atom
            })

    # --- 3) Deduplicate clauses (keep highest-confidence version) ---
    dedup = {}
    for h in hypotheses:
        key = h["clause"]
        if key not in dedup or h["confidence"] > dedup[key]["confidence"]:
            dedup[key] = h

    return list(dedup.values())


def _find_max_line_number_in_kb(kb: str) -> int:
    max_num = 0
    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        # strip inline comments here too
        content_raw = m.group(2).strip()
        content, _ = split_inline_comment(content_raw)
        if not content:
            continue

        num = int(m.group(1))
        if num > max_num:
            max_num = num
    return max_num


def _is_fact_clause(clause: str) -> bool:
    return ':-' not in clause


def _split_rule_clause(clause: str):
    clause = clause.strip()
    if clause.endswith('.'):
        clause = clause[:-1]
    head_part, body_part = clause.split(':-', 1)
    head = head_part.strip()
    body_str = body_part.strip()
    return head, body_str


def attach_hypotheses_to_kb(kb: str, hypotheses):
    soft_facts = []
    soft_rules = []

    max_num = _find_max_line_number_in_kb(kb)
    next_num = max_num + 1

    for h in hypotheses:
        clause = (h.get("clause") or "").strip()
        if not clause:
            continue

        conf = float(h.get("confidence", 0.0))

        if not clause.endswith('.'):
            clause = clause + '.'

        if _is_fact_clause(clause):
            atom = clause.rstrip('.').strip()
            soft_facts.append((next_num, atom, conf))
        else:
            head, body_str = _split_rule_clause(clause)
            soft_rules.append((next_num, head, body_str, conf))

        next_num += 1

    return {"facts": soft_facts, "rules": soft_rules}


def strip_inline_comment(s: str) -> str:
    # Everything after '#' is non-executable annotation
    return s.split('#', 1)[0].rstrip()


def bfs_prolog_metro_soft(
    goal: str,
    kb: str,
    soft_kb,
    max_depth: int = 10,
    max_soft: Optional[int] = None,
):
    """
    Priority-guided SLD resolution using a heap (best-first search).

    Priority key:
        (soft_cost, -min_conf, depth)

    Meaning:
      - prefer fewer soft clauses
      - among those, prefer higher min confidence over used soft clauses
      - among those, prefer smaller depth
    """

    # --- Parse hard KB exactly like bfs_prolog_metro ---
    hard_facts = []
    hard_rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        # NEW: remove inline comments like "...). # reachable/2: ..."
        line = strip_inline_comment(line).strip()
        if not line:
            continue

        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content = match.group(2).strip()

            if ':-' in content:
                head, body = content.split(':-', 1)
                hard_rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                hard_facts.append((num, content.rstrip('.')))

    # --- Soft KB unpack ---
    soft_facts = soft_kb.get("facts", [])  # list of (num, atom, conf)
    soft_rules = soft_kb.get("rules", [])  # list of (num, head, body_str, conf)

    # For rule matching, we want (num, head, body_str) lists
    soft_rules_for_match = [(num, head, body_str) for (num, head, body_str, conf) in soft_rules]

    print(f"\n[SOFT BFS - PRIORITY] Goal: {goal}")
    print("-" * 40)

    # ------------------------------------------------------------
    # Priority queue item:
    #   (soft_cost, -min_conf, depth, tie, current, remaining, path, min_conf)
    # ------------------------------------------------------------
    pq = []
    tie = count()

    def push_state(current, remaining, path, depth, soft_cost, min_conf):
        # Enforce depth bound here to avoid pushing junk
        if depth >= max_depth:
            return
        heapq.heappush(
            pq,
            (soft_cost, -min_conf, depth, next(tie), current, remaining, path, min_conf)
        )

    # Start state
    push_state(goal, [], [], 0, 0, 1.0)

    # Dominance / best-known pruning:
    # For each (current, remaining) keep best (soft_cost, -min_conf) seen so far.
    # If a new state is worse or equal, skip it.
    best_seen = {}  # (current, tuple(remaining)) -> (soft_cost, neg_min_conf)

    def dominated(current, remaining, soft_cost, min_conf):
        key = (current, tuple(remaining))
        new = (soft_cost, -min_conf)

        old = best_seen.get(key)
        if old is None:
            best_seen[key] = new
            return False

        # If old is <= new in lexicographic order, old is better or equal => new dominated
        if old <= new:
            return True

        # Otherwise new is strictly better
        best_seen[key] = new
        return False

    # Helper to finalize success
    def make_success_result(final_path, final_soft_cost, final_min_conf):
        used_soft = []
        for step in final_path:
            if step.startswith("SoftFact"):
                parts = step.split()
                if len(parts) >= 2:
                    try:
                        num = int(parts[1])
                    except ValueError:
                        num = None
                    used_soft.append(("fact", num))
            elif step.startswith("SoftRule"):
                parts = step.split()
                if len(parts) >= 2:
                    try:
                        num = int(parts[1])
                    except ValueError:
                        num = None
                    used_soft.append(("rule", num))

        return {
            "success": True,
            "proof_path": final_path,
            "used_soft_clauses": used_soft,
            "soft_cost": final_soft_cost,
            "min_conf": final_min_conf if final_soft_cost > 0 else None
        }

    while pq:
        soft_cost, neg_min_conf, depth, _, current, remaining, path, min_conf = heapq.heappop(pq)

        # Optional: print “best-first” trace
        print(f"Depth {depth}: {current}")
        print(f"  Priority key: (soft_cost={soft_cost}, -min_conf={neg_min_conf:.3f}, depth={depth})")
        if remaining:
            print(f"  Remaining: {remaining}")

        if dominated(current, remaining, soft_cost, min_conf):
            continue

        # --- 1) HARD facts: exact match ---
        for num, fact in hard_facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Hard Fact {num} matches exactly: {fact}")

                step_label = f"HardFact {num}"
                new_path = path + [step_label]

                if not remaining:
                    print(f"✓✓ SUCCESS (hard-only) at depth {depth + 1}")
                    return make_success_result(new_path, soft_cost, min_conf)

                next_goal = remaining[0]
                next_remaining = remaining[1:]
                push_state(next_goal, next_remaining, new_path, depth + 1, soft_cost, min_conf)

                # As in your old code, we can break after first exact match
                break

        # --- 2) HARD facts: unification ---
        for num, fact in hard_facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            print(f"  ✓ Hard Fact {num} unifies: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)

            step_label = f"HardFact {num}"
            new_path = path + [step_label]

            if not instantiated:
                print(f"✓✓ SUCCESS (hard-only) at depth {depth + 1}")
                return make_success_result(new_path, soft_cost, min_conf)

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            push_state(next_goal, next_remaining, new_path, depth + 1, soft_cost, min_conf)

        # --- 3) HARD rules ---
        matching_hard_rules = find_matching_rules_only(current, hard_rules)
        if matching_hard_rules:
            print(f"  Matching hard rules: {matching_hard_rules}")

        for rule_num in matching_hard_rules:
            for num, head, body in hard_rules:
                if num != rule_num:
                    continue

                subgoals = get_subgoals(current, head, body)
                if not subgoals:
                    continue

                print(f"  Hard Rule {num}: {head} :- {body}")
                print(f"    → {subgoals}")

                all_goals = subgoals + remaining
                next_goal = all_goals[0]
                next_remaining = all_goals[1:]
                step_label = f"HardRule {num}"
                new_path = path + [step_label]

                push_state(next_goal, next_remaining, new_path, depth + 1, soft_cost, min_conf)
                break

        # --- 4) SOFT facts ---
        for s_num, s_atom, s_conf in soft_facts:
            if max_soft is not None and soft_cost >= max_soft:
                break

            bindings = unify_with_fact(current, s_atom)
            if bindings is None:
                continue

            new_soft_cost = soft_cost + 1
            new_min_conf = min(min_conf, s_conf)

            print(f"  ✓ Soft Fact {s_num} unifies: {s_atom}")
            print(f"    Bindings: {bindings}, conf={s_conf:.3f}")
            print(f"    New soft cost: {new_soft_cost}, new min_conf: {new_min_conf:.3f}")

            instantiated = apply_bindings(remaining, bindings)

            step_label = f"SoftFact {s_num} (conf={s_conf:.3f})"
            new_path = path + [step_label]

            if not instantiated:
                print(f"✓✓ SUCCESS (with soft facts) at depth {depth + 1}")
                return make_success_result(new_path, new_soft_cost, new_min_conf)

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            push_state(next_goal, next_remaining, new_path, depth + 1, new_soft_cost, new_min_conf)

        # --- 5) SOFT rules ---
        matching_soft_rules = find_matching_rules_only(current, soft_rules_for_match)
        if matching_soft_rules:
            print(f"  Matching soft rules: {matching_soft_rules}")

        for rule_num in matching_soft_rules:
            if max_soft is not None and soft_cost >= max_soft:
                break

            for s_num, s_head, s_body_str, s_conf in soft_rules:
                if s_num != rule_num:
                    continue

                subgoals = get_subgoals(current, s_head, s_body_str)
                if not subgoals:
                    continue

                new_soft_cost = soft_cost + 1
                new_min_conf = min(min_conf, s_conf)

                print(f"  Soft Rule {s_num}: {s_head} :- {s_body_str}")
                print(f"    → {subgoals}, conf={s_conf:.3f}")
                print(f"    New soft cost: {new_soft_cost}, new min_conf: {new_min_conf:.3f}")

                all_goals = subgoals + remaining
                next_goal = all_goals[0]
                next_remaining = all_goals[1:]
                step_label = f"SoftRule {s_num} (conf={s_conf:.3f})"
                new_path = path + [step_label]

                push_state(next_goal, next_remaining, new_path, depth + 1, new_soft_cost, new_min_conf)
                break

    print("✗ PRIORITY SOFT-BFS FAILED (no proof found even with soft KB)")
    return {
        "success": False,
        "proof_path": [],
        "used_soft_clauses": [],
        "soft_cost": None,
        "min_conf": None
    }
def solve_with_background(
    goal: str,
    kb: str,
    max_depth: int = 10,
    max_soft=None,
    hard_result=None,
):
    """
    High-level pipeline (unchanged), but now reads predicate comments from kb.
    """
    predicate_comments = parse_kb_predicate_comments(kb)

    print("\n========================================")
    print(f"SOLVE WITH BACKGROUND: {goal}")
    print("========================================\n")

    if hard_result is None:
        print(">>> Phase 1: Hard-KB BFS (bfs_prolog_collect)")
        hard_result = bfs_prolog_collect(goal, kb, max_depth=max_depth)
        print("Hard-KB result:", hard_result)
    else:
        print(">>> Phase 1: Hard-KB BFS result already computed, reusing it.")
        print("Hard-KB result:", hard_result)

    if hard_result.get("success"):
        print("\n>>> Result: HARD_SUCCESS (no background hypotheses needed)\n")
        return {
            "status": "HARD_SUCCESS",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": []
        }

    unresolved_atoms = hard_result.get("unresolved_atoms", set())
    if not unresolved_atoms:
        print("\nNo unresolved atoms to explain; cannot generate hypotheses.")
        print(">>> Result: FAILURE\n")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": []
        }

    print("\n>>> Phase 2: Generate background hypotheses")
    print("Unresolved atoms:", unresolved_atoms)

    hypotheses = generate_background_hypotheses(
        goal=goal,
        kb=kb,
        unresolved_atoms=unresolved_atoms,
        predicate_comments=predicate_comments
    )

    if hypotheses is None:
        hypotheses = []

    if not hypotheses:
        print("Hypotheses returned by LLM: []")
        print("\nLLM returned NO hypotheses; cannot build soft KB.")
        print(">>> Result: FAILURE\n")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": []
        }

    print("Hypotheses returned by LLM:")
    for h in hypotheses:
        print("  - Clause:", h.get("clause"),
              "| Conf:", h.get("confidence"),
              "| From atom:", h.get("from_atom"))

    print("\n>>> Phase 3: Attach hypotheses to soft KB")
    soft_kb = attach_hypotheses_to_kb(kb, hypotheses)
    print("Soft KB facts:", soft_kb.get("facts", []))
    print("Soft KB rules:", soft_kb.get("rules", []))

    print("\n>>> Phase 4: Soft BFS (bfs_prolog_metro_soft)")
    soft_result = bfs_prolog_metro_soft(
        goal=goal,
        kb=kb,
        soft_kb=soft_kb,
        max_depth=max_depth,
        max_soft=max_soft,
    )
    print("Soft-BFS result:", soft_result)

    if soft_result.get("success"):
        print("\n>>> Result: SOFT_SUCCESS (proof found using background hypotheses)\n")
        return {
            "status": "SOFT_SUCCESS",
            "hard_result": hard_result,
            "soft_result": soft_result,
            "hypotheses": hypotheses
        }

    print("\n>>> Result: SOFT_FAILURE (no proof even with background hypotheses)\n")
    return {
        "status": "SOFT_FAILURE",
        "hard_result": hard_result,
        "soft_result": soft_result,
        "hypotheses": hypotheses
    }


def omit_facts_from_kb(kb: str, omit_numbers):
    """
    Return a new KB string with numbered lines in `omit_numbers` removed.
    Preserves original line text (including comments), but matches by number.
    """
    omit_numbers = set(omit_numbers)
    new_lines = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        num = int(m.group(1))
        if num in omit_numbers:
            continue

        new_lines.append(line)

    return "\n".join(new_lines)


# Natural language to Prolog using LLM, algo SLD resolution

def nl_kb_to_prolog_kb(nl_kb_text: str, start_index: int = 1) -> list[str]:
    """
    Convert a *pure natural-language* description of a domain + rules
    into a numbered Prolog knowledge base.
    """

    nl_kb_text = (nl_kb_text or "").strip()
    if not nl_kb_text:
        return []

    prompt = f"""
You are a Prolog formalization assistant.

The user will give you a natural-language description of a small domain,
including objects, relationships, and logical rules.

Your job is to convert that description into a set of Prolog clauses
(facts and rules).

Guidelines:
- Use lowercase atoms for concrete entities (e.g. union_square, times_square,
  grand_central, bryant_park).
- Use uppercase identifiers for variables (e.g. X, Y, Z).
- Choose predicate names that are short, descriptive, and consistent,
  for example: connected/2, reachable/2, located_in/2, etc.
- A fact must look like:
    connected(times_square, bryant_park).
- A rule must look like:
    reachable(X, Z) :- connected(X, Y), reachable(Y, Z).
- Every clause MUST end with a single period '.'.
- Do NOT include any line numbers in your output clauses.
- Do NOT add explanations or comments in the Prolog code.

Here is the NATURAL LANGUAGE description of the knowledge base:

\"\"\"{nl_kb_text}\"\"\"

Respond ONLY in this JSON format (and nothing else):

{{
  "clauses": [
    {{
      "clause": "connected(union_square, times_square)."
    }},
    {{
      "clause": "reachable(X, Y) :- connected(X, Y)."
    }}
  ]
}}
"""

    raw = ask_llm(prompt).strip()
    try:
        data = json.loads(extract_first_json(raw))
    except Exception as e:
        print("[nl_kb_to_prolog_kb] JSON parse error:", e)
        print("Raw LLM output:", raw)
        return []

    raw_clauses = data.get("clauses", [])
    if not isinstance(raw_clauses, list):
        print("[nl_kb_to_prolog_kb] 'clauses' field is not a list:", raw_clauses)
        return []

    cleaned_clauses = []

    for item in raw_clauses:
        if isinstance(item, str):
            clause = item.strip()
        elif isinstance(item, dict):
            clause = (item.get("clause") or "").strip()
        else:
            continue

        if not clause:
            continue

        m_num = re.match(r'^\s*(\d+)\.\s*(.+)$', clause)
        if m_num:
            clause = m_num.group(2).strip()

        clause = clause.rstrip()
        if not clause.endswith('.'):
            clause = clause + "."
        else:
            clause = re.sub(r'\.+$', '.', clause)

        body_str = clause[:-1].strip()
        if ':-' in body_str:
            head_part, body_part = body_str.split(':-', 1)
            head = head_part.strip()
        else:
            head = body_str

        parsed_head = parse_predicate(head)
        if parsed_head is None:
            print("[nl_kb_to_prolog_kb] Discarding unparsable clause:", clause)
            continue

        cleaned_clauses.append(clause)

    numbered_clauses = []
    next_num = start_index
    for clause in cleaned_clauses:
        numbered_clauses.append(f"{next_num}. {clause}")
        next_num += 1

    return numbered_clauses


# In[78]:


if __name__ == "__main__":
    kb = """
    1. connected(union_square, times_square). # connected/2: directly adjacent on the same metro line (no intermediate stops)
    2. connected(times_square, grand_central).
    3. connected(grand_central, bryant_park).
    4. reachable(X, Y) :- connected(X, Y). # reachable/2: there exists a path via one or more connected/2 edges
    5. reachable(X, Z) :- connected(X, Y), reachable(Y, Z).
    """

    print("===== FULL METRO KB =====")
    print(kb)
    print("====================================\n")

    # Omit fact 1 to force background reasoning (removes union_square -> times_square)
    kb_missing_1 = omit_facts_from_kb(kb, omit_numbers={1})

    print("===== METRO KB WITH FACT 1 REMOVED =====")
    print(kb_missing_1)
    print("========================================\n")

    test_goal = "reachable(union_square, bryant_park)"

    print("==============================")
    print(f"TEST QUERY: {test_goal}")
    print("==============================\n")

    print(">>> Running bfs_prolog_collect (hard-KB BFS)...")
    collect_result = bfs_prolog_collect(test_goal, kb_missing_1)
    print("Collect Result:", collect_result)
    print("\n----------------------------------------\n")

    print(">>> Running solve_with_background (full pipeline, reusing hard result)...")
    bg_result = solve_with_background(
        goal=test_goal,
        kb=kb_missing_1,
        max_depth=10,
        max_soft=None,
        hard_result=collect_result,
    )
    print("Solve-with-background Result:")
    print(bg_result)
    print("\n========================================\n")


# In[4]:


import ollama
import heapq
from itertools import count
from collections import deque
import re
import json
from typing import Optional


# --- Config / LLM setup ---

client = ollama.Client()

model = "gpt-oss:20b"
# model = "qwen:14b"

DEBUG = False  # set to True to print raw LLM outputs for debugging


def ask_llm(prompt: str) -> str:
    resp = client.generate(model=model, prompt=prompt, options={'temperature': 0.0})
    answer = resp.get('response', '')
    if "...done thinking." in answer:
        return answer.split("...done thinking.")[-1].strip()
    return answer


# --- Helpers for parsing / JSON ---

def extract_first_json(text: str) -> str:
    """
    Extract the first {...} JSON object from possibly messy text.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in: {text!r}")
    return match.group(0)


def split_inline_comment(s: str):
    """
    Split a string into (code, comment) at the first '#'.
    Returns comment WITHOUT the '#'. If no comment, comment=None.
    """
    if "#" not in s:
        return s.strip(), None
    code, comment = s.split("#", 1)
    code = code.strip()
    comment = comment.strip()
    return code, (comment if comment else None)


def parse_predicate(term: str):
    """
    Parse a simple Prolog predicate of the form:
        functor(arg1, arg2, ...)
    Returns (functor: str, args: list[str]) or None if parsing fails.
    """
    term = term.strip().rstrip('.')
    m = re.match(r'^([a-z_][a-zA-Z0-9_]*)\((.*)\)$', term)
    if not m:
        return None
    functor = m.group(1)
    args_raw = m.group(2).strip()
    if not args_raw:
        args = []
    else:
        # Simple arg split (no nested terms in this toy domain)
        args = [a.strip() for a in args_raw.split(',')]
    return functor, args


def is_variable(s: str) -> bool:
    """
    Prolog-ish variable check: starts with uppercase letter or '_'.
    """
    s = s.strip()
    return bool(s) and (s[0].isupper() or s[0] == '_')


# --- Core Prolog helpers ---

def check_exact_match(goal: str, fact: str) -> bool:
    """Check if goal matches fact exactly (no variables)."""
    return goal.strip().rstrip('.') == fact.strip().rstrip('.')


def unify_args(args_goal, args_fact, env=None):
    """
    Unify two argument lists (flat terms, no nesting) under an environment.

    args_goal: list[str]  from the GOAL predicate
    args_fact: list[str]  from the FACT/RULE-HEAD predicate
    env      : dict or None   existing bindings, e.g. {"X": "times_square"}

    Returns:
        - None if unification fails
        - env (possibly modified) if unification succeeds
    """
    if env is None:
        env = {}

    if len(args_goal) != len(args_fact):
        return None

    for g, f in zip(args_goal, args_fact):
        g = g.strip()
        f = f.strip()

        g_is_var = is_variable(g)
        f_is_var = is_variable(f)

        # both constants
        if not g_is_var and not f_is_var:
            if g != f:
                return None
            continue

        # goal var, fact const
        if g_is_var and not f_is_var:
            if g in env:
                if env[g] != f:
                    return None
            else:
                env[g] = f
            continue

        # goal const, fact var  (treat fact vars as wildcards)
        if not g_is_var and f_is_var:
            if f in env:
                if env[f] != g:
                    return None
            else:
                env[f] = g
            continue

        # both variables
        if g_is_var and f_is_var:
            if g in env and f in env:
                if env[g] != env[f]:
                    return None
            elif g in env:
                env[f] = env[g]
            elif f in env:
                env[g] = env[f]
            # else both unbound → no constraint
            continue

    return env


def unify_arg_lists(args_rule_head, args_goal):
    """
    Wrapper used by get_subgoals: unify rule-head args with goal args.
    """
    return unify_args(args_rule_head, args_goal, env={})


def unify_with_fact(goal: str, fact: str):
    """
    Purely algorithmic unification between a GOAL and a FACT (or rule head).

    Returns:
        None      -> NO unification
        {}        -> EXACT ground match (no variables)
        dict      -> bindings, e.g. {"Y": "times_square"}
    """

    parsed_goal = parse_predicate(goal)
    parsed_fact = parse_predicate(fact)
    if parsed_goal is None or parsed_fact is None:
        return None

    fun_g, args_g = parsed_goal
    fun_f, args_f = parsed_fact

    # Functor or arity mismatch
    if fun_g != fun_f or len(args_g) != len(args_f):
        return None

    # If they are exactly the same string (ignoring trailing dot), treat as EXACT
    if check_exact_match(goal, fact):
        return {}

    env = unify_args(args_g, args_f, env={})
    if env is None:
        return None

    return env


def apply_bindings(goals, bindings):
    """
    Apply variable bindings to goals using pure string/term substitution.

    goals: list[str]   e.g. ["reachable(Y, Z)", "connected(Z, X)"]
    bindings: dict     e.g. {"Y": "times_square"}

    Returns: list[str] of instantiated goals.
    """
    if not bindings or not goals:
        return goals

    new_goals = []

    for g in goals:
        parsed = parse_predicate(g)
        if parsed is None:
            # If we can't parse it as a predicate, leave as-is
            new_goals.append(g)
            continue

        functor, args = parsed
        new_args = []
        for a in args:
            a_stripped = a.strip()
            if is_variable(a_stripped) and a_stripped in bindings:
                new_args.append(bindings[a_stripped])
            else:
                new_args.append(a_stripped)

        new_goal = f"{functor}({', '.join(new_args)})"
        new_goals.append(new_goal)

    return new_goals


def find_matching_rules_only(goal, rules_list):
    """
    Find ONLY rules (not facts) whose HEAD can unify with the given goal.

    IMPORTANT: This version is purely syntactic: it only checks functor and arity.
    We do NOT ask the LLM here, to avoid mismatched heads like reachable/2
    being applied to connected/2 goals.

    rules_list: list[(num, head, body)]
    Returns: list[int] of rule numbers.
    """
    parsed_goal = parse_predicate(goal)
    if parsed_goal is None:
        return []
    fun_g, args_g = parsed_goal
    arity_g = len(args_g)

    matching = []
    for num, head, body in rules_list:
        parsed_head = parse_predicate(head)
        if parsed_head is None:
            continue
        fun_h, args_h = parsed_head
        if fun_h == fun_g and len(args_h) == arity_g:
            matching.append(num)
    return matching


def substitute_in_atom(atom: str, bindings: dict) -> str:
    """
    Apply variable bindings to a single Prolog atom, e.g.:

        atom     = "connected(X, Y)"
        bindings = {"X": "union_square", "Y": "bryant_park"}

    Returns:
        "connected(union_square, bryant_park)"
    """
    parsed = parse_predicate(atom)
    if parsed is None:
        return atom  # best-effort fallback

    functor, args = parsed
    new_args = []

    for a in args:
        a_stripped = a.strip()
        if is_variable(a_stripped) and a_stripped in bindings:
            new_args.append(bindings[a_stripped])
        else:
            new_args.append(a_stripped)

    return f"{functor}({', '.join(new_args)})"


def split_body_atoms(body_str: str):
    """
    Split a rule body like:
        "connected(X, Y), reachable(Y, Z)"
    into:
        ["connected(X, Y)", "reachable(Y, Z)"]

    It is parentheses-aware, so it will NOT split on commas that are
    inside argument lists.
    """
    body_str = body_str.strip()
    atoms = []
    current = []
    depth = 0  # parentheses nesting depth

    for ch in body_str:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth = max(depth - 1, 0)
            current.append(ch)
        elif ch == ',' and depth == 0:
            # top-level comma → split here
            atom = ''.join(current).strip()
            if atom:
                atoms.append(atom)
            current = []
        else:
            current.append(ch)

    # Flush the last atom
    atom = ''.join(current).strip()
    if atom:
        atoms.append(atom)

    return atoms


def get_subgoals(goal: str, rule_head: str, rule_body: str):
    """
    Algorithmic ONE-STEP SLD resolution (purely symbolic).
    """
    parsed_goal = parse_predicate(goal)
    parsed_head = parse_predicate(rule_head)

    if parsed_goal is None or parsed_head is None:
        return None

    fun_g, args_g = parsed_goal
    fun_h, args_h = parsed_head

    if fun_g != fun_h or len(args_g) != len(args_h):
        return None

    bindings = unify_arg_lists(args_h, args_g)
    if bindings is None:
        return None

    body_str = rule_body.strip()
    if not body_str:
        return []

    body_atoms = split_body_atoms(body_str)
    if not body_atoms:
        return []

    subgoals = [substitute_in_atom(atom, bindings) for atom in body_atoms]
    return subgoals if subgoals else None


# --- KB comment extraction (inline + full-line) ---

def parse_kb_predicate_comments(kb: str):
    """
    Supports BOTH:
      - full-line comments starting with '#'
      - inline comments after a numbered clause: '... . # comment'

    Returns:
        dict mapping "predicate/arity" -> comment string
        e.g. { "connected/2": "Connected means directly adjacent ..." }
    """
    predicate_comments = {}
    pending_comments = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Full-line comment
        if line.startswith("#"):
            pending_comments.append(line.lstrip("#").strip())
            continue

        # Numbered clause
        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        content_raw = m.group(2).strip()
        content, inline_comment = split_inline_comment(content_raw)

        clause = content.strip().rstrip(".")
        head_str = clause.split(":-", 1)[0].strip()
        parsed = parse_predicate(head_str)
        if parsed is None:
            pending_comments = []
            continue

        functor, args = parsed
        key = f"{functor}/{len(args)}"

        combined = []
        if pending_comments:
            combined.append(" ".join(pending_comments))
        if inline_comment:
            combined.append(inline_comment)

        if combined:
            predicate_comments[key] = " ".join(combined).strip()

        pending_comments = []

    return predicate_comments


# --- BFS Prolog engine ---

def bfs_prolog_metro(goal: str, kb: str, max_depth: int = 10) -> bool:
    """
    BFS with correct fact/rule distinction.
    """

    facts = []
    rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        line = strip_inline_comment(line).strip()
        if not line:
            continue
        
        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content_raw = match.group(2).strip()
            content, _ = split_inline_comment(content_raw)

            if not content:
                continue

            if ':-' in content:
                head, body = content.split(':-', 1)
                rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                facts.append((num, content.rstrip('.')))

    queue = deque([(goal, [], [], 0)])
    visited = set()

    print(f"\nGoal: {goal}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        if depth >= max_depth:
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")

        # 1) Exact fact match
        fact_matched = False
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")

                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return True

                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))
                fact_matched = True
                break

        if fact_matched:
            continue

        # 2) Fact unification
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            print(f"  ✓ Fact {num}: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)

            if not instantiated:
                print(f"✓✓ SUCCESS at depth {depth + 1}")
                return True

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))

        # 3) Rules
        matching_rules = find_matching_rules_only(current, rules)
        if matching_rules:
            print(f"  Matching rules: {matching_rules}")

            for rule_num in matching_rules:
                for num, head, body in rules:
                    if num == rule_num:
                        subgoals = get_subgoals(current, head, body)
                        if subgoals:
                            print(f"  Rule {num}: → {subgoals}")
                            all_goals = subgoals + remaining
                            next_goal = all_goals[0]
                            next_remaining = all_goals[1:]
                            queue.append((next_goal, next_remaining, path + [f"Rule {num}"], depth + 1))
                        break

    print("✗ FAILED")
    return False


def bfs_prolog_collect(goal: str, kb: str, max_depth: int = 10):
    """
    Like bfs_prolog_metro, but returns unresolved atoms too.
    """

    facts = []
    rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        line = strip_inline_comment(line).strip()
        if not line:
            continue
        
        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content_raw = match.group(2).strip()
            content, _ = split_inline_comment(content_raw)

            if not content:
                continue

            if ':-' in content:
                head, body = content.split(':-', 1)
                rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                facts.append((num, content.rstrip('.')))

    queue = deque([(goal, [], [], 0)])
    visited = set()
    unresolved_atoms = set()

    print(f"\n[COLLECT] Goal: {goal}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        if depth >= max_depth:
            unresolved_atoms.add(current)
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")

        progress = False

        # 1) Exact fact match
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")

                step = f"Fact {num}"
                new_path = path + [step]

                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return {
                        "success": True,
                        "proof_path": new_path,
                        "unresolved_atoms": set()
                    }

                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, new_path, depth + 1))

                progress = True
                break

        if progress:
            continue

        # 2) Fact unification
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            progress = True

            print(f"  ✓ Fact {num}: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)

            step = f"Fact {num}"
            new_path = path + [step]

            if not instantiated:
                print(f"✓✓ SUCCESS at depth {depth + 1}")
                return {
                    "success": True,
                    "proof_path": new_path,
                    "unresolved_atoms": set()
                }

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, new_path, depth + 1))

        if progress:
            continue

        # 3) Rules
        matching_rules = find_matching_rules_only(current, rules)
        if matching_rules:
            print(f"  Matching rules: {matching_rules}")

        for rule_num in matching_rules:
            for num, head, body in rules:
                if num == rule_num:
                    subgoals = get_subgoals(current, head, body)
                    if subgoals:
                        print(f"  Rule {num}: → {subgoals}")
                        progress = True
                        all_goals = subgoals + remaining
                        next_goal = all_goals[0]
                        next_remaining = all_goals[1:]
                        step = f"Rule {num}"
                        new_path = path + [step]
                        queue.append((next_goal, next_remaining, new_path, depth + 1))
                    break

        if not progress:
            print(f"  ✗ No facts or rules apply to: {current}")
            unresolved_atoms.add(current)

    print("✗ FAILED (collect mode)")
    return {
        "success": False,
        "proof_path": [],
        "unresolved_atoms": unresolved_atoms
    }


def generate_background_hypotheses(goal: str, kb: str, unresolved_atoms, predicate_comments: dict, max_atoms: int = 5):
    """
    Keeps the LONGER, domain-specific prompt (close to your original),
    but injects predicate semantics from KB comments.
    """
    hypotheses = []

    # --- 1) Filter unresolved atoms to simple, ground atoms only ---
    atom_list = list(unresolved_atoms)
    ground_atoms = []
    for atom in atom_list:
        atom = atom.strip()
        if not atom:
            continue
        if '(' not in atom or ')' not in atom:
            continue
        inside = atom.split('(', 1)[1].rsplit(')', 1)[0]
        if re.search(r'\b[A-Z_]\w*\b', inside):
            continue
        ground_atoms.append(atom)

    if max_atoms is not None and len(ground_atoms) > max_atoms:
        ground_atoms = ground_atoms[:max_atoms]

    if not ground_atoms:
        print("[generate_background_hypotheses] No suitable ground atoms to query.")
        return []

    # --- 2) For each ground unresolved atom, ask the LLM for hypotheses ---
    for atom in ground_atoms:
        parsed = parse_predicate(atom)
        semantic_hint = ""
        if parsed is not None:
            functor, args = parsed
            pred_key = f"{functor}/{len(args)}"
            semantic_hint = predicate_comments.get(pred_key, "")

        prompt = f"""
You are a cautious Prolog expert with access to real-world background knowledge.

We attempted to prove the following GOAL using ONLY the numbered Prolog
knowledge base given below:

GOAL:
  {goal}

KNOWLEDGE BASE (numbered clauses):
{kb}

During breadth-first SLD resolution, the proof FAILED because we could not
prove the following subgoal:
  {atom}

DOMAIN DESCRIPTION (important):
- The intended domain is a metro / subway network with stations and lines.
- We are reasoning about which stations are directly connected and which
  stations are reachable by traveling along metro lines.

SEMANTIC HINTS ABOUT PREDICATES (author-provided; treat as ground truth):
{semantic_hint}

Task:
Propose a SMALL set of additional Prolog clauses (facts or rules) that are
LIKELY to be true in the intended metro domain and that would help make the GOAL
provable. Think of these as "missing" facts or rules that could fill gaps in
the knowledge base.

Constraints:
- Each clause MUST be valid Prolog and MUST end with a period.
- You MUST NOT modify or delete any existing clauses in the KB.
- Use ONLY predicate names and arities that are compatible with the style of
  the existing KB (for example, connected/2, reachable/2, etc.).
- Your proposed clauses MUST respect the semantic hints above.
- If you are uncertain about the factual correctness of a clause, give it a lower confidence.
- You SHOULD prefer to return at least one plausible clause rather than an empty list.

Respond ONLY in this JSON format:

{{
  "hypotheses": [
    {{
      "clause": "connected(times_square, grand_central).",
      "confidence": 0.9
    }},
    {{
      "clause": "connected(times_square, bryant_park).",
      "confidence": 0.4
    }}
  ]
}}

If you truly have NO hypotheses, respond with:
{{ "hypotheses": [] }}
"""
        raw = ask_llm(prompt).strip()
        if DEBUG:
            print("\n[DEBUG generate_background_hypotheses]")
            print("Unresolved atom:", atom)
            print("Raw response:\n", raw)

        try:
            data = json.loads(extract_first_json(raw))
        except Exception as e:
            print("[generate_background_hypotheses] JSON parse error:", e)
            print("Raw LLM output:", raw)

            if "Invalid \\escape" in str(e):
                try:
                    fixed_raw = raw.replace("\\=", "\\\\=")
                    data = json.loads(extract_first_json(fixed_raw))
                except Exception as e2:
                    print("[generate_background_hypotheses] JSON parse error after fix:", e2)
                    continue
            else:
                continue

        raw_hyps = data.get("hypotheses", [])
        if not isinstance(raw_hyps, list):
            print("[generate_background_hypotheses] 'hypotheses' not a list:", raw_hyps)
            continue

        for h in raw_hyps:
            clause = (h.get("clause") or "").strip()
            if not clause:
                continue

            if not clause.endswith('.'):
                clause = clause + "."

            try:
                conf = float(h.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0

            hypotheses.append({
                "clause": clause,
                "confidence": conf,
                "from_atom": atom
            })

    # --- 3) Deduplicate clauses (keep highest-confidence version) ---
    dedup = {}
    for h in hypotheses:
        key = h["clause"]
        if key not in dedup or h["confidence"] > dedup[key]["confidence"]:
            dedup[key] = h

    return list(dedup.values())


def _find_max_line_number_in_kb(kb: str) -> int:
    max_num = 0
    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        # strip inline comments here too
        content_raw = m.group(2).strip()
        content, _ = split_inline_comment(content_raw)
        if not content:
            continue

        num = int(m.group(1))
        if num > max_num:
            max_num = num
    return max_num


def _is_fact_clause(clause: str) -> bool:
    return ':-' not in clause


def _split_rule_clause(clause: str):
    clause = clause.strip()
    if clause.endswith('.'):
        clause = clause[:-1]
    head_part, body_part = clause.split(':-', 1)
    head = head_part.strip()
    body_str = body_part.strip()
    return head, body_str


def attach_hypotheses_to_kb(kb: str, hypotheses):
    soft_facts = []
    soft_rules = []

    max_num = _find_max_line_number_in_kb(kb)
    next_num = max_num + 1

    for h in hypotheses:
        clause = (h.get("clause") or "").strip()
        if not clause:
            continue

        conf = float(h.get("confidence", 0.0))

        if not clause.endswith('.'):
            clause = clause + '.'

        if _is_fact_clause(clause):
            atom = clause.rstrip('.').strip()
            soft_facts.append((next_num, atom, conf))
        else:
            head, body_str = _split_rule_clause(clause)
            soft_rules.append((next_num, head, body_str, conf))

        next_num += 1

    return {"facts": soft_facts, "rules": soft_rules}


def strip_inline_comment(s: str) -> str:
    # Everything after '#' is non-executable annotation
    return s.split('#', 1)[0].rstrip()


def bfs_prolog_metro_soft(
    goal: str,
    kb: str,
    soft_kb,
    max_depth: int = 10,
    max_soft: Optional[int] = None,
):
    """
    Priority-guided SLD resolution using a heap (best-first search).

    Priority key:
        (soft_cost, -min_conf, depth)

    Meaning:
      - prefer fewer soft clauses
      - among those, prefer higher min confidence over used soft clauses
      - among those, prefer smaller depth
    """

    # --- Parse hard KB exactly like bfs_prolog_metro ---
    hard_facts = []
    hard_rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        # NEW: remove inline comments like "...). # reachable/2: ..."
        line = strip_inline_comment(line).strip()
        if not line:
            continue

        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content = match.group(2).strip()

            if ':-' in content:
                head, body = content.split(':-', 1)
                hard_rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                hard_facts.append((num, content.rstrip('.')))

    # --- Soft KB unpack ---
    soft_facts = soft_kb.get("facts", [])  # list of (num, atom, conf)
    soft_rules = soft_kb.get("rules", [])  # list of (num, head, body_str, conf)

    # For rule matching, we want (num, head, body_str) lists
    soft_rules_for_match = [(num, head, body_str) for (num, head, body_str, conf) in soft_rules]

    print(f"\n[SOFT BFS - PRIORITY] Goal: {goal}")
    print("-" * 40)

    # ------------------------------------------------------------
    # Priority queue item:
    #   (soft_cost, -min_conf, depth, tie, current, remaining, path, min_conf)
    # ------------------------------------------------------------
    pq = []
    tie = count()

    def push_state(current, remaining, path, depth, soft_cost, min_conf):
        # Enforce depth bound here to avoid pushing junk
        if depth >= max_depth:
            return
        heapq.heappush(
            pq,
            (soft_cost, -min_conf, depth, next(tie), current, remaining, path, min_conf)
        )

    # Start state
    push_state(goal, [], [], 0, 0, 1.0)

    # Dominance / best-known pruning:
    # For each (current, remaining) keep best (soft_cost, -min_conf) seen so far.
    # If a new state is worse or equal, skip it.
    best_seen = {}  # (current, tuple(remaining)) -> (soft_cost, neg_min_conf)

    def dominated(current, remaining, soft_cost, min_conf):
        key = (current, tuple(remaining))
        new = (soft_cost, -min_conf)

        old = best_seen.get(key)
        if old is None:
            best_seen[key] = new
            return False

        # If old is <= new in lexicographic order, old is better or equal => new dominated
        if old <= new:
            return True

        # Otherwise new is strictly better
        best_seen[key] = new
        return False

    # Helper to finalize success
    def make_success_result(final_path, final_soft_cost, final_min_conf):
        used_soft = []
        for step in final_path:
            if step.startswith("SoftFact"):
                parts = step.split()
                if len(parts) >= 2:
                    try:
                        num = int(parts[1])
                    except ValueError:
                        num = None
                    used_soft.append(("fact", num))
            elif step.startswith("SoftRule"):
                parts = step.split()
                if len(parts) >= 2:
                    try:
                        num = int(parts[1])
                    except ValueError:
                        num = None
                    used_soft.append(("rule", num))

        return {
            "success": True,
            "proof_path": final_path,
            "used_soft_clauses": used_soft,
            "soft_cost": final_soft_cost,
            "min_conf": final_min_conf if final_soft_cost > 0 else None
        }

    while pq:
        soft_cost, neg_min_conf, depth, _, current, remaining, path, min_conf = heapq.heappop(pq)

        # Optional: print “best-first” trace
        print(f"Depth {depth}: {current}")
        print(f"  Priority key: (soft_cost={soft_cost}, -min_conf={neg_min_conf:.3f}, depth={depth})")
        if remaining:
            print(f"  Remaining: {remaining}")

        if dominated(current, remaining, soft_cost, min_conf):
            continue

        # --- 1) HARD facts: exact match ---
        for num, fact in hard_facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Hard Fact {num} matches exactly: {fact}")

                step_label = f"HardFact {num}"
                new_path = path + [step_label]

                if not remaining:
                    print(f"✓✓ SUCCESS (hard-only) at depth {depth + 1}")
                    return make_success_result(new_path, soft_cost, min_conf)

                next_goal = remaining[0]
                next_remaining = remaining[1:]
                push_state(next_goal, next_remaining, new_path, depth + 1, soft_cost, min_conf)

                # As in your old code, we can break after first exact match
                break

        # --- 2) HARD facts: unification ---
        for num, fact in hard_facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            print(f"  ✓ Hard Fact {num} unifies: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)

            step_label = f"HardFact {num}"
            new_path = path + [step_label]

            if not instantiated:
                print(f"✓✓ SUCCESS (hard-only) at depth {depth + 1}")
                return make_success_result(new_path, soft_cost, min_conf)

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            push_state(next_goal, next_remaining, new_path, depth + 1, soft_cost, min_conf)

        # --- 3) HARD rules ---
        matching_hard_rules = find_matching_rules_only(current, hard_rules)
        if matching_hard_rules:
            print(f"  Matching hard rules: {matching_hard_rules}")

        for rule_num in matching_hard_rules:
            for num, head, body in hard_rules:
                if num != rule_num:
                    continue

                subgoals = get_subgoals(current, head, body)
                if not subgoals:
                    continue

                print(f"  Hard Rule {num}: {head} :- {body}")
                print(f"    → {subgoals}")

                all_goals = subgoals + remaining
                next_goal = all_goals[0]
                next_remaining = all_goals[1:]
                step_label = f"HardRule {num}"
                new_path = path + [step_label]

                push_state(next_goal, next_remaining, new_path, depth + 1, soft_cost, min_conf)
                break

        # --- 4) SOFT facts ---
        for s_num, s_atom, s_conf in soft_facts:
            if max_soft is not None and soft_cost >= max_soft:
                break

            bindings = unify_with_fact(current, s_atom)
            if bindings is None:
                continue

            new_soft_cost = soft_cost + 1
            new_min_conf = min(min_conf, s_conf)

            print(f"  ✓ Soft Fact {s_num} unifies: {s_atom}")
            print(f"    Bindings: {bindings}, conf={s_conf:.3f}")
            print(f"    New soft cost: {new_soft_cost}, new min_conf: {new_min_conf:.3f}")

            instantiated = apply_bindings(remaining, bindings)

            step_label = f"SoftFact {s_num} (conf={s_conf:.3f})"
            new_path = path + [step_label]

            if not instantiated:
                print(f"✓✓ SUCCESS (with soft facts) at depth {depth + 1}")
                return make_success_result(new_path, new_soft_cost, new_min_conf)

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            push_state(next_goal, next_remaining, new_path, depth + 1, new_soft_cost, new_min_conf)

        # --- 5) SOFT rules ---
        matching_soft_rules = find_matching_rules_only(current, soft_rules_for_match)
        if matching_soft_rules:
            print(f"  Matching soft rules: {matching_soft_rules}")

        for rule_num in matching_soft_rules:
            if max_soft is not None and soft_cost >= max_soft:
                break

            for s_num, s_head, s_body_str, s_conf in soft_rules:
                if s_num != rule_num:
                    continue

                subgoals = get_subgoals(current, s_head, s_body_str)
                if not subgoals:
                    continue

                new_soft_cost = soft_cost + 1
                new_min_conf = min(min_conf, s_conf)

                print(f"  Soft Rule {s_num}: {s_head} :- {s_body_str}")
                print(f"    → {subgoals}, conf={s_conf:.3f}")
                print(f"    New soft cost: {new_soft_cost}, new min_conf: {new_min_conf:.3f}")

                all_goals = subgoals + remaining
                next_goal = all_goals[0]
                next_remaining = all_goals[1:]
                step_label = f"SoftRule {s_num} (conf={s_conf:.3f})"
                new_path = path + [step_label]

                push_state(next_goal, next_remaining, new_path, depth + 1, new_soft_cost, new_min_conf)
                break

    print("✗ PRIORITY SOFT-BFS FAILED (no proof found even with soft KB)")
    return {
        "success": False,
        "proof_path": [],
        "used_soft_clauses": [],
        "soft_cost": None,
        "min_conf": None
    }
def solve_with_background(
    goal: str,
    kb: str,
    max_depth: int = 10,
    max_soft=None,
    hard_result=None,
):
    """
    High-level pipeline (unchanged), but now reads predicate comments from kb.
    """
    predicate_comments = parse_kb_predicate_comments(kb)

    print("\n========================================")
    print(f"SOLVE WITH BACKGROUND: {goal}")
    print("========================================\n")

    if hard_result is None:
        print(">>> Phase 1: Hard-KB BFS (bfs_prolog_collect)")
        hard_result = bfs_prolog_collect(goal, kb, max_depth=max_depth)
        print("Hard-KB result:", hard_result)
    else:
        print(">>> Phase 1: Hard-KB BFS result already computed, reusing it.")
        print("Hard-KB result:", hard_result)

    if hard_result.get("success"):
        print("\n>>> Result: HARD_SUCCESS (no background hypotheses needed)\n")
        return {
            "status": "HARD_SUCCESS",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": []
        }

    unresolved_atoms = hard_result.get("unresolved_atoms", set())
    if not unresolved_atoms:
        print("\nNo unresolved atoms to explain; cannot generate hypotheses.")
        print(">>> Result: FAILURE\n")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": []
        }

    print("\n>>> Phase 2: Generate background hypotheses")
    print("Unresolved atoms:", unresolved_atoms)

    hypotheses = generate_background_hypotheses(
        goal=goal,
        kb=kb,
        unresolved_atoms=unresolved_atoms,
        predicate_comments=predicate_comments
    )

    if hypotheses is None:
        hypotheses = []

    if not hypotheses:
        print("Hypotheses returned by LLM: []")
        print("\nLLM returned NO hypotheses; cannot build soft KB.")
        print(">>> Result: FAILURE\n")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": []
        }

    print("Hypotheses returned by LLM:")
    for h in hypotheses:
        print("  - Clause:", h.get("clause"),
              "| Conf:", h.get("confidence"),
              "| From atom:", h.get("from_atom"))

    print("\n>>> Phase 3: Attach hypotheses to soft KB")
    soft_kb = attach_hypotheses_to_kb(kb, hypotheses)
    print("Soft KB facts:", soft_kb.get("facts", []))
    print("Soft KB rules:", soft_kb.get("rules", []))

    print("\n>>> Phase 4: Soft BFS (bfs_prolog_metro_soft)")
    soft_result = bfs_prolog_metro_soft(
        goal=goal,
        kb=kb,
        soft_kb=soft_kb,
        max_depth=max_depth,
        max_soft=max_soft,
    )
    print("Soft-BFS result:", soft_result)

    if soft_result.get("success"):
        print("\n>>> Result: SOFT_SUCCESS (proof found using background hypotheses)\n")
        return {
            "status": "SOFT_SUCCESS",
            "hard_result": hard_result,
            "soft_result": soft_result,
            "hypotheses": hypotheses
        }

    print("\n>>> Result: SOFT_FAILURE (no proof even with background hypotheses)\n")
    return {
        "status": "SOFT_FAILURE",
        "hard_result": hard_result,
        "soft_result": soft_result,
        "hypotheses": hypotheses
    }


def omit_facts_from_kb(kb: str, omit_numbers):
    """
    Return a new KB string with numbered lines in `omit_numbers` removed.
    Preserves original line text (including comments), but matches by number.
    """
    omit_numbers = set(omit_numbers)
    new_lines = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        num = int(m.group(1))
        if num in omit_numbers:
            continue

        new_lines.append(line)

    return "\n".join(new_lines)


# Natural language to Prolog using LLM, algo SLD resolution

def nl_kb_to_prolog_kb(nl_kb_text: str, start_index: int = 1) -> list[str]:
    """
    Convert a *pure natural-language* description of a domain + rules
    into a numbered Prolog knowledge base.
    """

    nl_kb_text = (nl_kb_text or "").strip()
    if not nl_kb_text:
        return []

    prompt = f"""
You are a Prolog formalization assistant.

The user will give you a natural-language description of a small domain,
including objects, relationships, and logical rules.

Your job is to convert that description into a set of Prolog clauses
(facts and rules).

Guidelines:
- Use lowercase atoms for concrete entities (e.g. union_square, times_square,
  grand_central, bryant_park).
- Use uppercase identifiers for variables (e.g. X, Y, Z).
- Choose predicate names that are short, descriptive, and consistent,
  for example: connected/2, reachable/2, located_in/2, etc.
- A fact must look like:
    connected(times_square, bryant_park).
- A rule must look like:
    reachable(X, Z) :- connected(X, Y), reachable(Y, Z).
- Every clause MUST end with a single period '.'.
- Do NOT include any line numbers in your output clauses.
- Do NOT add explanations or comments in the Prolog code.

Here is the NATURAL LANGUAGE description of the knowledge base:

\"\"\"{nl_kb_text}\"\"\"

Respond ONLY in this JSON format (and nothing else):

{{
  "clauses": [
    {{
      "clause": "connected(union_square, times_square)."
    }},
    {{
      "clause": "reachable(X, Y) :- connected(X, Y)."
    }}
  ]
}}
"""

    raw = ask_llm(prompt).strip()
    try:
        data = json.loads(extract_first_json(raw))
    except Exception as e:
        print("[nl_kb_to_prolog_kb] JSON parse error:", e)
        print("Raw LLM output:", raw)
        return []

    raw_clauses = data.get("clauses", [])
    if not isinstance(raw_clauses, list):
        print("[nl_kb_to_prolog_kb] 'clauses' field is not a list:", raw_clauses)
        return []

    cleaned_clauses = []

    for item in raw_clauses:
        if isinstance(item, str):
            clause = item.strip()
        elif isinstance(item, dict):
            clause = (item.get("clause") or "").strip()
        else:
            continue

        if not clause:
            continue

        m_num = re.match(r'^\s*(\d+)\.\s*(.+)$', clause)
        if m_num:
            clause = m_num.group(2).strip()

        clause = clause.rstrip()
        if not clause.endswith('.'):
            clause = clause + "."
        else:
            clause = re.sub(r'\.+$', '.', clause)

        body_str = clause[:-1].strip()
        if ':-' in body_str:
            head_part, body_part = body_str.split(':-', 1)
            head = head_part.strip()
        else:
            head = body_str

        parsed_head = parse_predicate(head)
        if parsed_head is None:
            print("[nl_kb_to_prolog_kb] Discarding unparsable clause:", clause)
            continue

        cleaned_clauses.append(clause)

    numbered_clauses = []
    next_num = start_index
    for clause in cleaned_clauses:
        numbered_clauses.append(f"{next_num}. {clause}")
        next_num += 1

    return numbered_clauses


# In[5]:


if __name__ == "__main__":
    kb = """
    1. connected(union_square, times_square). # connected/2: directly adjacent on the same metro line (no intermediate stops)
    2. connected(times_square, grand_central).
    3. connected(grand_central, bryant_park).
    4. reachable(X, Y) :- connected(X, Y). # reachable/2: there exists a path via one or more connected/2 edges
    5. reachable(X, Z) :- connected(X, Y), reachable(Y, Z).
    """

    print("===== FULL METRO KB =====")
    print(kb)
    print("====================================\n")

    # Omit fact 1 to force background reasoning (removes union_square -> times_square)
    kb_missing_1 = omit_facts_from_kb(kb, omit_numbers={1})

    print("===== METRO KB WITH FACT 1 REMOVED =====")
    print(kb_missing_1)
    print("========================================\n")

    test_goal = "reachable(union_square, bryant_park)"

    print("==============================")
    print(f"TEST QUERY: {test_goal}")
    print("==============================\n")

    print(">>> Running bfs_prolog_collect (hard-KB BFS)...")
    collect_result = bfs_prolog_collect(test_goal, kb_missing_1)
    print("Collect Result:", collect_result)
    print("\n----------------------------------------\n")

    print(">>> Running solve_with_background (full pipeline, reusing hard result)...")
    bg_result = solve_with_background(
        goal=test_goal,
        kb=kb_missing_1,
        max_depth=10,
        max_soft=None,
        hard_result=collect_result,
    )
    print("Solve-with-background Result:")
    print(bg_result)
    print("\n========================================\n")


# In[37]:


import ollama
import heapq
from itertools import count
from collections import deque
import re
import json
from typing import Optional
import math


# --- Config / LLM setup ---

client = ollama.Client()

model = "gpt-oss:20b"
# model = "qwen:14b"

DEBUG = False  # set to True to print raw LLM outputs for debugging


def ask_llm(prompt: str) -> str:
    resp = client.generate(model=model, prompt=prompt, options={'temperature': 0.0})
    answer = resp.get('response', '')
    if "...done thinking." in answer:
        return answer.split("...done thinking.")[-1].strip()
    return answer


# --- Helpers for parsing / JSON ---

def extract_first_json(text: str) -> str:
    """
    Extract the first {...} JSON object from possibly messy text.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in: {text!r}")
    return match.group(0)


def split_inline_comment(s: str):
    """
    Split a string into (code, comment) at the first '#'.
    Returns comment WITHOUT the '#'. If no comment, comment=None.
    """
    if "#" not in s:
        return s.strip(), None
    code, comment = s.split("#", 1)
    code = code.strip()
    comment = comment.strip()
    return code, (comment if comment else None)


def parse_predicate(term: str):
    """
    Parse a simple Prolog predicate of the form:
        functor(arg1, arg2, ...)
    Returns (functor: str, args: list[str]) or None if parsing fails.
    """
    term = term.strip().rstrip('.')
    m = re.match(r'^([a-z_][a-zA-Z0-9_]*)\((.*)\)$', term)
    if not m:
        return None
    functor = m.group(1)
    args_raw = m.group(2).strip()
    if not args_raw:
        args = []
    else:
        # Simple arg split (no nested terms in this toy domain)
        args = [a.strip() for a in args_raw.split(',')]
    return functor, args


def is_variable(s: str) -> bool:
    """
    Prolog-ish variable check: starts with uppercase letter or '_'.
    """
    s = s.strip()
    return bool(s) and (s[0].isupper() or s[0] == '_')


# --- Core Prolog helpers ---

def check_exact_match(goal: str, fact: str) -> bool:
    """Check if goal matches fact exactly (no variables)."""
    return goal.strip().rstrip('.') == fact.strip().rstrip('.')


def unify_args(args_goal, args_fact, env=None):
    """
    Unify two argument lists (flat terms, no nesting) under an environment.

    args_goal: list[str]  from the GOAL predicate
    args_fact: list[str]  from the FACT/RULE-HEAD predicate
    env      : dict or None   existing bindings, e.g. {"X": "times_square"}

    Returns:
        - None if unification fails
        - env (possibly modified) if unification succeeds
    """
    if env is None:
        env = {}

    if len(args_goal) != len(args_fact):
        return None

    for g, f in zip(args_goal, args_fact):
        g = g.strip()
        f = f.strip()

        g_is_var = is_variable(g)
        f_is_var = is_variable(f)

        # both constants
        if not g_is_var and not f_is_var:
            if g != f:
                return None
            continue

        # goal var, fact const
        if g_is_var and not f_is_var:
            if g in env:
                if env[g] != f:
                    return None
            else:
                env[g] = f
            continue

        # goal const, fact var  (treat fact vars as wildcards)
        if not g_is_var and f_is_var:
            if f in env:
                if env[f] != g:
                    return None
            else:
                env[f] = g
            continue

        # both variables
        if g_is_var and f_is_var:
            if g in env and f in env:
                if env[g] != env[f]:
                    return None
            elif g in env:
                env[f] = env[g]
            elif f in env:
                env[g] = env[f]
            # else both unbound → no constraint
            continue

    return env


def unify_arg_lists(args_rule_head, args_goal):
    """
    Wrapper used by get_subgoals: unify rule-head args with goal args.
    """
    return unify_args(args_rule_head, args_goal, env={})


def unify_with_fact(goal: str, fact: str):
    """
    Purely algorithmic unification between a GOAL and a FACT (or rule head).

    Returns:
        None      -> NO unification
        {}        -> EXACT ground match (no variables)
        dict      -> bindings, e.g. {"Y": "times_square"}
    """

    parsed_goal = parse_predicate(goal)
    parsed_fact = parse_predicate(fact)
    if parsed_goal is None or parsed_fact is None:
        return None

    fun_g, args_g = parsed_goal
    fun_f, args_f = parsed_fact

    # Functor or arity mismatch
    if fun_g != fun_f or len(args_g) != len(args_f):
        return None

    # If they are exactly the same string (ignoring trailing dot), treat as EXACT
    if check_exact_match(goal, fact):
        return {}

    env = unify_args(args_g, args_f, env={})
    if env is None:
        return None

    return env


def apply_bindings(goals, bindings):
    """
    Apply variable bindings to goals using pure string/term substitution.

    goals: list[str]   e.g. ["reachable(Y, Z)", "connected(Z, X)"]
    bindings: dict     e.g. {"Y": "times_square"}

    Returns: list[str] of instantiated goals.
    """
    if not bindings or not goals:
        return goals

    new_goals = []

    for g in goals:
        parsed = parse_predicate(g)
        if parsed is None:
            # If we can't parse it as a predicate, leave as-is
            new_goals.append(g)
            continue

        functor, args = parsed
        new_args = []
        for a in args:
            a_stripped = a.strip()
            if is_variable(a_stripped) and a_stripped in bindings:
                new_args.append(bindings[a_stripped])
            else:
                new_args.append(a_stripped)

        new_goal = f"{functor}({', '.join(new_args)})"
        new_goals.append(new_goal)

    return new_goals


def find_matching_rules_only(goal, rules_list):
    """
    Find ONLY rules (not facts) whose HEAD can unify with the given goal.

    IMPORTANT: This version is purely syntactic: it only checks functor and arity.
    We do NOT ask the LLM here, to avoid mismatched heads like reachable/2
    being applied to connected/2 goals.

    rules_list: list[(num, head, body)]
    Returns: list[int] of rule numbers.
    """
    parsed_goal = parse_predicate(goal)
    if parsed_goal is None:
        return []
    fun_g, args_g = parsed_goal
    arity_g = len(args_g)

    matching = []
    for num, head, body in rules_list:
        parsed_head = parse_predicate(head)
        if parsed_head is None:
            continue
        fun_h, args_h = parsed_head
        if fun_h == fun_g and len(args_h) == arity_g:
            matching.append(num)
    return matching


def substitute_in_atom(atom: str, bindings: dict) -> str:
    """
    Apply variable bindings to a single Prolog atom, e.g.:

        atom     = "connected(X, Y)"
        bindings = {"X": "union_square", "Y": "bryant_park"}

    Returns:
        "connected(union_square, bryant_park)"
    """
    parsed = parse_predicate(atom)
    if parsed is None:
        return atom  # best-effort fallback

    functor, args = parsed
    new_args = []

    for a in args:
        a_stripped = a.strip()
        if is_variable(a_stripped) and a_stripped in bindings:
            new_args.append(bindings[a_stripped])
        else:
            new_args.append(a_stripped)

    return f"{functor}({', '.join(new_args)})"


def split_body_atoms(body_str: str):
    """
    Split a rule body like:
        "connected(X, Y), reachable(Y, Z)"
    into:
        ["connected(X, Y)", "reachable(Y, Z)"]

    It is parentheses-aware, so it will NOT split on commas that are
    inside argument lists.
    """
    body_str = body_str.strip()
    atoms = []
    current = []
    depth = 0  # parentheses nesting depth

    for ch in body_str:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth = max(depth - 1, 0)
            current.append(ch)
        elif ch == ',' and depth == 0:
            # top-level comma → split here
            atom = ''.join(current).strip()
            if atom:
                atoms.append(atom)
            current = []
        else:
            current.append(ch)

    # Flush the last atom
    atom = ''.join(current).strip()
    if atom:
        atoms.append(atom)

    return atoms


def get_subgoals(goal: str, rule_head: str, rule_body: str):
    """
    Algorithmic ONE-STEP SLD resolution (purely symbolic).
    """
    parsed_goal = parse_predicate(goal)
    parsed_head = parse_predicate(rule_head)

    if parsed_goal is None or parsed_head is None:
        return None

    fun_g, args_g = parsed_goal
    fun_h, args_h = parsed_head

    if fun_g != fun_h or len(args_g) != len(args_h):
        return None

    bindings = unify_arg_lists(args_h, args_g)
    if bindings is None:
        return None

    body_str = rule_body.strip()
    if not body_str:
        return []

    body_atoms = split_body_atoms(body_str)
    if not body_atoms:
        return []

    subgoals = [substitute_in_atom(atom, bindings) for atom in body_atoms]
    return subgoals if subgoals else None


# --- KB comment extraction (inline + full-line) ---

def parse_kb_predicate_comments(kb: str):
    """
    Supports BOTH:
      - full-line comments starting with '#'
      - inline comments after a numbered clause: '... . # comment'

    Returns:
        dict mapping "predicate/arity" -> comment string
        e.g. { "connected/2": "Connected means directly adjacent ..." }
    """
    predicate_comments = {}
    pending_comments = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Full-line comment
        if line.startswith("#"):
            pending_comments.append(line.lstrip("#").strip())
            continue

        # Numbered clause
        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        content_raw = m.group(2).strip()
        content, inline_comment = split_inline_comment(content_raw)

        clause = content.strip().rstrip(".")
        head_str = clause.split(":-", 1)[0].strip()
        parsed = parse_predicate(head_str)
        if parsed is None:
            pending_comments = []
            continue

        functor, args = parsed
        key = f"{functor}/{len(args)}"

        combined = []
        if pending_comments:
            combined.append(" ".join(pending_comments))
        if inline_comment:
            combined.append(inline_comment)

        if combined:
            predicate_comments[key] = " ".join(combined).strip()

        pending_comments = []

    return predicate_comments


# --- BFS Prolog engine ---

def bfs_prolog_metro(goal: str, kb: str, max_depth: int = 10) -> bool:
    """
    BFS with correct fact/rule distinction.
    """

    facts = []
    rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        line = strip_inline_comment(line).strip()
        if not line:
            continue
        
        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content_raw = match.group(2).strip()
            content, _ = split_inline_comment(content_raw)

            if not content:
                continue

            if ':-' in content:
                head, body = content.split(':-', 1)
                rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                facts.append((num, content.rstrip('.')))

    queue = deque([(goal, [], [], 0)])
    visited = set()

    print(f"\nGoal: {goal}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        if depth >= max_depth:
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")

        # 1) Exact fact match
        fact_matched = False
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")

                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return True

                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))
                fact_matched = True
                break

        if fact_matched:
            continue

        # 2) Fact unification
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            print(f"  ✓ Fact {num}: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)

            if not instantiated:
                print(f"✓✓ SUCCESS at depth {depth + 1}")
                return True

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))

        # 3) Rules
        matching_rules = find_matching_rules_only(current, rules)
        if matching_rules:
            print(f"  Matching rules: {matching_rules}")

            for rule_num in matching_rules:
                for num, head, body in rules:
                    if num == rule_num:
                        subgoals = get_subgoals(current, head, body)
                        if subgoals:
                            print(f"  Rule {num}: → {subgoals}")
                            all_goals = subgoals + remaining
                            next_goal = all_goals[0]
                            next_remaining = all_goals[1:]
                            queue.append((next_goal, next_remaining, path + [f"Rule {num}"], depth + 1))
                        break

    print("✗ FAILED")
    return False


def bfs_prolog_collect(goal: str, kb: str, max_depth: int = 10):
    """
    Like bfs_prolog_metro, but returns unresolved atoms too.
    """

    facts = []
    rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        line = strip_inline_comment(line).strip()
        if not line:
            continue
        
        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content_raw = match.group(2).strip()
            content, _ = split_inline_comment(content_raw)

            if not content:
                continue

            if ':-' in content:
                head, body = content.split(':-', 1)
                rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                facts.append((num, content.rstrip('.')))

    queue = deque([(goal, [], [], 0)])
    visited = set()
    unresolved_atoms = set()

    print(f"\n[COLLECT] Goal: {goal}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        if depth >= max_depth:
            unresolved_atoms.add(current)
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")

        progress = False

        # 1) Exact fact match
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")

                step = f"Fact {num}"
                new_path = path + [step]

                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return {
                        "success": True,
                        "proof_path": new_path,
                        "unresolved_atoms": set()
                    }

                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, new_path, depth + 1))

                progress = True
                break

        if progress:
            continue

        # 2) Fact unification
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            progress = True

            print(f"  ✓ Fact {num}: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)

            step = f"Fact {num}"
            new_path = path + [step]

            if not instantiated:
                print(f"✓✓ SUCCESS at depth {depth + 1}")
                return {
                    "success": True,
                    "proof_path": new_path,
                    "unresolved_atoms": set()
                }

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, new_path, depth + 1))

        if progress:
            continue

        # 3) Rules
        matching_rules = find_matching_rules_only(current, rules)
        if matching_rules:
            print(f"  Matching rules: {matching_rules}")

        for rule_num in matching_rules:
            for num, head, body in rules:
                if num == rule_num:
                    subgoals = get_subgoals(current, head, body)
                    if subgoals:
                        print(f"  Rule {num}: → {subgoals}")
                        progress = True
                        all_goals = subgoals + remaining
                        next_goal = all_goals[0]
                        next_remaining = all_goals[1:]
                        step = f"Rule {num}"
                        new_path = path + [step]
                        queue.append((next_goal, next_remaining, new_path, depth + 1))
                    break

        if not progress:
            print(f"  ✗ No facts or rules apply to: {current}")
            unresolved_atoms.add(current)

    print("✗ FAILED (collect mode)")
    return {
        "success": False,
        "proof_path": [],
        "unresolved_atoms": unresolved_atoms
    }


def generate_background_hypotheses(goal: str, kb: str, unresolved_atoms, predicate_comments: dict, max_atoms: int = 5):
    """
    Keeps the LONGER, domain-specific prompt (close to your original),
    but injects predicate semantics from KB comments.
    """
    hypotheses = []

    # --- 1) Filter unresolved atoms to simple, ground atoms only ---
    atom_list = list(unresolved_atoms)
    ground_atoms = []
    for atom in atom_list:
        atom = atom.strip()
        if not atom:
            continue
        if '(' not in atom or ')' not in atom:
            continue
        inside = atom.split('(', 1)[1].rsplit(')', 1)[0]
        if re.search(r'\b[A-Z_]\w*\b', inside):
            continue
        ground_atoms.append(atom)

    if max_atoms is not None and len(ground_atoms) > max_atoms:
        ground_atoms = ground_atoms[:max_atoms]

    if not ground_atoms:
        print("[generate_background_hypotheses] No suitable ground atoms to query.")
        return []

    # --- 2) For each ground unresolved atom, ask the LLM for hypotheses ---
    for atom in ground_atoms:
        parsed = parse_predicate(atom)
        semantic_hint = ""
        if parsed is not None:
            functor, args = parsed
            pred_key = f"{functor}/{len(args)}"
            semantic_hint = predicate_comments.get(pred_key, "")

        prompt = f"""
You are a cautious Prolog expert with access to real-world background knowledge.

We attempted to prove the following GOAL using ONLY the numbered Prolog
knowledge base given below:

GOAL:
  {goal}

KNOWLEDGE BASE (numbered clauses):
{kb}

During breadth-first SLD resolution, the proof FAILED because we could not
prove the following subgoal:
  {atom}

DOMAIN DESCRIPTION (important):
- The intended domain is a metro / subway network with stations and lines.
- We are reasoning about which stations are directly connected and which
  stations are reachable by traveling along metro lines.

SEMANTIC HINTS ABOUT PREDICATES (author-provided; treat as ground truth):
{semantic_hint}

Task:
Propose a SMALL set of additional Prolog clauses (facts or rules) that are
LIKELY to be true in the intended metro domain and that would help make the GOAL
provable. Think of these as "missing" facts or rules that could fill gaps in
the knowledge base.

Constraints:
- Each clause MUST be valid Prolog and MUST end with a period.
- You MUST NOT modify or delete any existing clauses in the KB.
- Use ONLY predicate names and arities that are compatible with the style of
  the existing KB (for example, connected/2, reachable/2, etc.).
- Your proposed clauses MUST respect the semantic hints above.
- If you are uncertain about the factual correctness of a clause, give it a lower confidence.
- You SHOULD prefer to return at least one plausible clause rather than an empty list.

Respond ONLY in this JSON format:

{{
  "hypotheses": [
    {{
      "clause": "connected(times_square, grand_central).",
      "confidence": 0.9
    }},
    {{
      "clause": "connected(times_square, bryant_park).",
      "confidence": 0.4
    }}
  ]
}}

If you truly have NO hypotheses, respond with:
{{ "hypotheses": [] }}
"""
        raw = ask_llm(prompt).strip()
        if DEBUG:
            print("\n[DEBUG generate_background_hypotheses]")
            print("Unresolved atom:", atom)
            print("Raw response:\n", raw)

        try:
            data = json.loads(extract_first_json(raw))
        except Exception as e:
            print("[generate_background_hypotheses] JSON parse error:", e)
            print("Raw LLM output:", raw)

            if "Invalid \\escape" in str(e):
                try:
                    fixed_raw = raw.replace("\\=", "\\\\=")
                    data = json.loads(extract_first_json(fixed_raw))
                except Exception as e2:
                    print("[generate_background_hypotheses] JSON parse error after fix:", e2)
                    continue
            else:
                continue

        raw_hyps = data.get("hypotheses", [])
        if not isinstance(raw_hyps, list):
            print("[generate_background_hypotheses] 'hypotheses' not a list:", raw_hyps)
            continue

        for h in raw_hyps:
            clause = (h.get("clause") or "").strip()
            if not clause:
                continue

            if not clause.endswith('.'):
                clause = clause + "."

            try:
                conf = float(h.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0

            hypotheses.append({
                "clause": clause,
                "confidence": conf,
                "from_atom": atom
            })

    # --- 3) Deduplicate clauses (keep highest-confidence version) ---
    dedup = {}
    for h in hypotheses:
        key = h["clause"]
        if key not in dedup or h["confidence"] > dedup[key]["confidence"]:
            dedup[key] = h

    return list(dedup.values())


def _find_max_line_number_in_kb(kb: str) -> int:
    max_num = 0
    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        # strip inline comments here too
        content_raw = m.group(2).strip()
        content, _ = split_inline_comment(content_raw)
        if not content:
            continue

        num = int(m.group(1))
        if num > max_num:
            max_num = num
    return max_num


def _is_fact_clause(clause: str) -> bool:
    return ':-' not in clause


def _split_rule_clause(clause: str):
    clause = clause.strip()
    if clause.endswith('.'):
        clause = clause[:-1]
    head_part, body_part = clause.split(':-', 1)
    head = head_part.strip()
    body_str = body_part.strip()
    return head, body_str


def attach_hypotheses_to_kb(kb: str, hypotheses):
    soft_facts = []
    soft_rules = []

    max_num = _find_max_line_number_in_kb(kb)
    next_num = max_num + 1

    for h in hypotheses:
        clause = (h.get("clause") or "").strip()
        if not clause:
            continue

        conf = float(h.get("confidence", 0.0))

        if not clause.endswith('.'):
            clause = clause + '.'

        if _is_fact_clause(clause):
            atom = clause.rstrip('.').strip()
            soft_facts.append((next_num, atom, conf))
        else:
            head, body_str = _split_rule_clause(clause)
            soft_rules.append((next_num, head, body_str, conf))

        next_num += 1

    return {"facts": soft_facts, "rules": soft_rules}


def strip_inline_comment(s: str) -> str:
    # Everything after '#' is non-executable annotation
    return s.split('#', 1)[0].rstrip()


def bfs_prolog_metro_soft(
    goal: str,
    kb: str,
    soft_kb,
    max_depth: int = 10,
    max_soft: Optional[int] = None,
):
    """
    Priority-guided SLD resolution using a heap (best-first search).

    Priority key:
        (soft_cost, -min_conf, depth)

    Meaning:
      - prefer fewer soft clauses
      - among those, prefer higher min confidence over used soft clauses
      - among those, prefer smaller depth
    """

    # --- Parse hard KB exactly like bfs_prolog_metro ---
    hard_facts = []
    hard_rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        # NEW: remove inline comments like "...). # reachable/2: ..."
        line = strip_inline_comment(line).strip()
        if not line:
            continue

        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content = match.group(2).strip()

            if ':-' in content:
                head, body = content.split(':-', 1)
                hard_rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                hard_facts.append((num, content.rstrip('.')))

    # --- Soft KB unpack ---
    soft_facts = soft_kb.get("facts", [])  # list of (num, atom, conf)
    soft_rules = soft_kb.get("rules", [])  # list of (num, head, body_str, conf)

    # For rule matching, we want (num, head, body_str) lists
    soft_rules_for_match = [(num, head, body_str) for (num, head, body_str, conf) in soft_rules]

    print(f"\n[SOFT BFS - PRIORITY] Goal: {goal}")
    print("-" * 40)

    # ------------------------------------------------------------
    # Priority queue item:
    #   (soft_cost, -min_conf, depth, tie, current, remaining, path, min_conf)
    # ------------------------------------------------------------
    pq = []
    tie = count()

    def push_state(current, remaining, path, depth, soft_cost, min_conf):
        # Enforce depth bound here to avoid pushing junk
        if depth >= max_depth:
            return
        heapq.heappush(
            pq,
            (soft_cost, -min_conf, depth, next(tie), current, remaining, path, min_conf)
        )

    # Start state
    push_state(goal, [], [], 0, 0, 1.0)

    # Dominance / best-known pruning:
    # For each (current, remaining) keep best (soft_cost, -min_conf) seen so far.
    # If a new state is worse or equal, skip it.
    best_seen = {}  # (current, tuple(remaining)) -> (soft_cost, neg_min_conf)

    def dominated(current, remaining, soft_cost, min_conf):
        key = (current, tuple(remaining))
        new = (soft_cost, -min_conf)

        old = best_seen.get(key)
        if old is None:
            best_seen[key] = new
            return False

        # If old is <= new in lexicographic order, old is better or equal => new dominated
        if old <= new:
            return True

        # Otherwise new is strictly better
        best_seen[key] = new
        return False

    # Helper to finalize success
    def make_success_result(final_path, final_soft_cost, final_min_conf):
        used_soft = []
        for step in final_path:
            if step.startswith("SoftFact"):
                parts = step.split()
                if len(parts) >= 2:
                    try:
                        num = int(parts[1])
                    except ValueError:
                        num = None
                    used_soft.append(("fact", num))
            elif step.startswith("SoftRule"):
                parts = step.split()
                if len(parts) >= 2:
                    try:
                        num = int(parts[1])
                    except ValueError:
                        num = None
                    used_soft.append(("rule", num))

        return {
            "success": True,
            "proof_path": final_path,
            "used_soft_clauses": used_soft,
            "soft_cost": final_soft_cost,
            "min_conf": final_min_conf if final_soft_cost > 0 else None
        }

    while pq:
        soft_cost, neg_min_conf, depth, _, current, remaining, path, min_conf = heapq.heappop(pq)

        # Optional: print “best-first” trace
        print(f"Depth {depth}: {current}")
        print(f"  Priority key: (soft_cost={soft_cost}, -min_conf={neg_min_conf:.3f}, depth={depth})")
        if remaining:
            print(f"  Remaining: {remaining}")

        if dominated(current, remaining, soft_cost, min_conf):
            continue

        # --- 1) HARD facts: exact match ---
        for num, fact in hard_facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Hard Fact {num} matches exactly: {fact}")

                step_label = f"HardFact {num}"
                new_path = path + [step_label]

                if not remaining:
                    print(f"✓✓ SUCCESS (hard-only) at depth {depth + 1}")
                    return make_success_result(new_path, soft_cost, min_conf)

                next_goal = remaining[0]
                next_remaining = remaining[1:]
                push_state(next_goal, next_remaining, new_path, depth + 1, soft_cost, min_conf)

                # As in your old code, we can break after first exact match
                break

        # --- 2) HARD facts: unification ---
        for num, fact in hard_facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            print(f"  ✓ Hard Fact {num} unifies: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)

            step_label = f"HardFact {num}"
            new_path = path + [step_label]

            if not instantiated:
                print(f"✓✓ SUCCESS (hard-only) at depth {depth + 1}")
                return make_success_result(new_path, soft_cost, min_conf)

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            push_state(next_goal, next_remaining, new_path, depth + 1, soft_cost, min_conf)

        # --- 3) HARD rules ---
        matching_hard_rules = find_matching_rules_only(current, hard_rules)
        if matching_hard_rules:
            print(f"  Matching hard rules: {matching_hard_rules}")

        for rule_num in matching_hard_rules:
            for num, head, body in hard_rules:
                if num != rule_num:
                    continue

                subgoals = get_subgoals(current, head, body)
                if not subgoals:
                    continue

                print(f"  Hard Rule {num}: {head} :- {body}")
                print(f"    → {subgoals}")

                all_goals = subgoals + remaining
                next_goal = all_goals[0]
                next_remaining = all_goals[1:]
                step_label = f"HardRule {num}"
                new_path = path + [step_label]

                push_state(next_goal, next_remaining, new_path, depth + 1, soft_cost, min_conf)
                break

        # --- 4) SOFT facts ---
        for s_num, s_atom, s_conf in soft_facts:
            if max_soft is not None and soft_cost >= max_soft:
                break

            bindings = unify_with_fact(current, s_atom)
            if bindings is None:
                continue

            new_soft_cost = soft_cost + 1
            new_min_conf = min(min_conf, s_conf)

            print(f"  ✓ Soft Fact {s_num} unifies: {s_atom}")
            print(f"    Bindings: {bindings}, conf={s_conf:.3f}")
            print(f"    New soft cost: {new_soft_cost}, new min_conf: {new_min_conf:.3f}")

            instantiated = apply_bindings(remaining, bindings)

            step_label = f"SoftFact {s_num} (conf={s_conf:.3f})"
            new_path = path + [step_label]

            if not instantiated:
                print(f"✓✓ SUCCESS (with soft facts) at depth {depth + 1}")
                return make_success_result(new_path, new_soft_cost, new_min_conf)

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            push_state(next_goal, next_remaining, new_path, depth + 1, new_soft_cost, new_min_conf)

        # --- 5) SOFT rules ---
        matching_soft_rules = find_matching_rules_only(current, soft_rules_for_match)
        if matching_soft_rules:
            print(f"  Matching soft rules: {matching_soft_rules}")

        for rule_num in matching_soft_rules:
            if max_soft is not None and soft_cost >= max_soft:
                break

            for s_num, s_head, s_body_str, s_conf in soft_rules:
                if s_num != rule_num:
                    continue

                subgoals = get_subgoals(current, s_head, s_body_str)
                if not subgoals:
                    continue

                new_soft_cost = soft_cost + 1
                new_min_conf = min(min_conf, s_conf)

                print(f"  Soft Rule {s_num}: {s_head} :- {s_body_str}")
                print(f"    → {subgoals}, conf={s_conf:.3f}")
                print(f"    New soft cost: {new_soft_cost}, new min_conf: {new_min_conf:.3f}")

                all_goals = subgoals + remaining
                next_goal = all_goals[0]
                next_remaining = all_goals[1:]
                step_label = f"SoftRule {s_num} (conf={s_conf:.3f})"
                new_path = path + [step_label]

                push_state(next_goal, next_remaining, new_path, depth + 1, new_soft_cost, new_min_conf)
                break

    print("✗ PRIORITY SOFT-BFS FAILED (no proof found even with soft KB)")
    return {
        "success": False,
        "proof_path": [],
        "used_soft_clauses": [],
        "soft_cost": None,
        "min_conf": None
    }
def solve_with_background(
    goal: str,
    kb: str,
    max_depth: int = 10,
    max_soft=None,
    hard_result=None,
):
    """
    High-level pipeline (unchanged), but now reads predicate comments from kb.
    """
    predicate_comments = parse_kb_predicate_comments(kb)

    print("\n========================================")
    print(f"SOLVE WITH BACKGROUND: {goal}")
    print("========================================\n")

    if hard_result is None:
        print(">>> Phase 1: Hard-KB BFS (bfs_prolog_collect)")
        hard_result = bfs_prolog_collect(goal, kb, max_depth=max_depth)
        print("Hard-KB result:", hard_result)
    else:
        print(">>> Phase 1: Hard-KB BFS result already computed, reusing it.")
        print("Hard-KB result:", hard_result)

    if hard_result.get("success"):
        print("\n>>> Result: HARD_SUCCESS (no background hypotheses needed)\n")
        return {
            "status": "HARD_SUCCESS",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": []
        }

    unresolved_atoms = hard_result.get("unresolved_atoms", set())
    if not unresolved_atoms:
        print("\nNo unresolved atoms to explain; cannot generate hypotheses.")
        print(">>> Result: FAILURE\n")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": []
        }

    print("\n>>> Phase 2: Generate background hypotheses")
    print("Unresolved atoms:", unresolved_atoms)

    hypotheses = generate_background_hypotheses(
        goal=goal,
        kb=kb,
        unresolved_atoms=unresolved_atoms,
        predicate_comments=predicate_comments
    )

    if hypotheses is None:
        hypotheses = []

    if not hypotheses:
        print("Hypotheses returned by LLM: []")
        print("\nLLM returned NO hypotheses; cannot build soft KB.")
        print(">>> Result: FAILURE\n")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": []
        }

    print("Hypotheses returned by LLM:")
    for h in hypotheses:
        print("  - Clause:", h.get("clause"),
              "| Conf:", h.get("confidence"),
              "| From atom:", h.get("from_atom"))

    print("\n>>> Phase 3: Attach hypotheses to soft KB")
    soft_kb = attach_hypotheses_to_kb(kb, hypotheses)
    print("Soft KB facts:", soft_kb.get("facts", []))
    print("Soft KB rules:", soft_kb.get("rules", []))

    print("\n>>> Phase 4: Soft BFS (bfs_prolog_metro_soft)")
    soft_result = bfs_prolog_metro_soft(
        goal=goal,
        kb=kb,
        soft_kb=soft_kb,
        max_depth=max_depth,
        max_soft=max_soft,
    )
    print("Soft-BFS result:", soft_result)

    if soft_result.get("success"):
        print("\n>>> Result: SOFT_SUCCESS (proof found using background hypotheses)\n")
        return {
            "status": "SOFT_SUCCESS",
            "hard_result": hard_result,
            "soft_result": soft_result,
            "hypotheses": hypotheses
        }

    print("\n>>> Result: SOFT_FAILURE (no proof even with background hypotheses)\n")
    return {
        "status": "SOFT_FAILURE",
        "hard_result": hard_result,
        "soft_result": soft_result,
        "hypotheses": hypotheses
    }


def omit_facts_from_kb(kb: str, omit_numbers):
    """
    Return a new KB string with numbered lines in `omit_numbers` removed.
    Preserves original line text (including comments), but matches by number.
    """
    omit_numbers = set(omit_numbers)
    new_lines = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        num = int(m.group(1))
        if num in omit_numbers:
            continue

        new_lines.append(line)

    return "\n".join(new_lines)


# Natural language to Prolog using LLM, algo SLD resolution

def nl_kb_to_prolog_kb(nl_kb_text: str, start_index: int = 1) -> list[str]:
    """
    Convert a *pure natural-language* description of a domain + rules
    into a numbered Prolog knowledge base.
    """

    nl_kb_text = (nl_kb_text or "").strip()
    if not nl_kb_text:
        return []

    prompt = f"""
You are a Prolog formalization assistant.

The user will give you a natural-language description of a small domain,
including objects, relationships, and logical rules.

Your job is to convert that description into a set of Prolog clauses
(facts and rules).

Guidelines:
- Use lowercase atoms for concrete entities (e.g. union_square, times_square,
  grand_central, bryant_park).
- Use uppercase identifiers for variables (e.g. X, Y, Z).
- Choose predicate names that are short, descriptive, and consistent,
  for example: connected/2, reachable/2, located_in/2, etc.
- A fact must look like:
    connected(times_square, bryant_park).
- A rule must look like:
    reachable(X, Z) :- connected(X, Y), reachable(Y, Z).
- Every clause MUST end with a single period '.'.
- Do NOT include any line numbers in your output clauses.
- Do NOT add explanations or comments in the Prolog code.

Here is the NATURAL LANGUAGE description of the knowledge base:

\"\"\"{nl_kb_text}\"\"\"

Respond ONLY in this JSON format (and nothing else):

{{
  "clauses": [
    {{
      "clause": "connected(union_square, times_square)."
    }},
    {{
      "clause": "reachable(X, Y) :- connected(X, Y)."
    }}
  ]
}}
"""

    raw = ask_llm(prompt).strip()
    try:
        data = json.loads(extract_first_json(raw))
    except Exception as e:
        print("[nl_kb_to_prolog_kb] JSON parse error:", e)
        print("Raw LLM output:", raw)
        return []

    raw_clauses = data.get("clauses", [])
    if not isinstance(raw_clauses, list):
        print("[nl_kb_to_prolog_kb] 'clauses' field is not a list:", raw_clauses)
        return []

    cleaned_clauses = []

    for item in raw_clauses:
        if isinstance(item, str):
            clause = item.strip()
        elif isinstance(item, dict):
            clause = (item.get("clause") or "").strip()
        else:
            continue

        if not clause:
            continue

        m_num = re.match(r'^\s*(\d+)\.\s*(.+)$', clause)
        if m_num:
            clause = m_num.group(2).strip()

        clause = clause.rstrip()
        if not clause.endswith('.'):
            clause = clause + "."
        else:
            clause = re.sub(r'\.+$', '.', clause)

        body_str = clause[:-1].strip()
        if ':-' in body_str:
            head_part, body_part = body_str.split(':-', 1)
            head = head_part.strip()
        else:
            head = body_str

        parsed_head = parse_predicate(head)
        if parsed_head is None:
            print("[nl_kb_to_prolog_kb] Discarding unparsable clause:", clause)
            continue

        cleaned_clauses.append(clause)

    numbered_clauses = []
    next_num = start_index
    for clause in cleaned_clauses:
        numbered_clauses.append(f"{next_num}. {clause}")
        next_num += 1

    return numbered_clauses


# In[24]:


if __name__ == "__main__":
    kb = """
    1. connected(union_square, 14th_street).        
    2. connected(14th_street, 23rd_street).
    3. connected(23rd_street, 34th_street).
    4. connected(34th_street, times_square).
    5. connected(times_square, 42nd_street).
    6. connected(42nd_street, grand_central).
    7. connected(grand_central, bryant_park).
    8. reachable(X, Y) :- connected(X, Y).         
    9. reachable(X, Z) :- connected(X, Y), reachable(Y, Z).  
    """

    print("===== FULL METRO KB =====")
    print(kb)
    print("====================================\n")

    # Omit fact 1 to force background reasoning (removes union_square -> times_square)
    kb_missing_fact = omit_facts_from_kb(kb, omit_numbers={7})

    print("===== METRO KB WITH FACT REMOVED =====")
    print(kb_missing_fact)
    print("========================================\n")

    test_goal = "reachable(union_square, bryant_park)"

    print("==============================")
    print(f"TEST QUERY: {test_goal}")
    print("==============================\n")

    print(">>> Running bfs_prolog_collect (hard-KB BFS)...")
    collect_result = bfs_prolog_collect(test_goal, kb_missing_fact)
    print("Collect Result:", collect_result)
    print("\n----------------------------------------\n")

    print(">>> Running solve_with_background (full pipeline, reusing hard result)...")
    bg_result = solve_with_background(
        goal=test_goal,
        kb=kb_missing_fact,
        max_depth=10,
        max_soft=None,
        hard_result=collect_result,
    )
    print("Solve-with-background Result:")
    print(bg_result)
    print("\n========================================\n")


# WITH HINTS

# In[38]:


if __name__ == "__main__":
    kb = """
    1. connected(union_square, 14th_street). # connected/2 = ADJACENT STOPS ONLY. Use only when two stations are immediate neighbors on the same line. Do NOT add shortcut edges that skip intermediate stations.
    2. connected(14th_street, 23rd_street).
    3. connected(23rd_street, 34th_street).
    4. connected(34th_street, times_square).
    5. connected(times_square, 42nd_street).
    6. connected(42nd_street, grand_central).
    7. connected(grand_central, bryant_park).
    8. reachable(X, Y) :- connected(X, Y). # reachable/2 = PATH EXISTS. One-hop reachable comes only from connected/2.
    9. reachable(X, Z) :- connected(X, Y), reachable(Y, Z). # reachable/2 is the transitive closure of connected/2 (multi-hop path following only adjacent edges).

    """

    print("===== FULL METRO KB =====")
    print(kb)
    print("====================================\n")

    # Omit fact 1 to force background reasoning (removes union_square -> times_square)
    kb_missing_fact = omit_facts_from_kb(kb, omit_numbers={7})

    print("===== METRO KB WITH FACT REMOVED =====")
    print(kb_missing_fact)
    print("========================================\n")

    test_goal = "reachable(union_square, bryant_park)"

    print("==============================")
    print(f"TEST QUERY: {test_goal}")
    print("==============================\n")

    print(">>> Running bfs_prolog_collect (hard-KB BFS)...")
    collect_result = bfs_prolog_collect(test_goal, kb_missing_fact)
    print("Collect Result:", collect_result)
    print("\n----------------------------------------\n")

    print(">>> Running solve_with_background (full pipeline, reusing hard result)...")
    bg_result = solve_with_background(
        goal=test_goal,
        kb=kb_missing_fact,
        max_depth=10,
        max_soft=None,
        hard_result=collect_result,
    )
    print("Solve-with-background Result:")
    print(bg_result)
    print("\n========================================\n")


# In[ ]:


WITH HINTS + NEW PROMPT


# In[63]:


import ollama
import heapq
from itertools import count
from collections import deque
import re
import json
from typing import Optional
import math


# --- Config / LLM setup ---

client = ollama.Client()

model = "gpt-oss:20b"
# model = "qwen:14b"

DEBUG = False  # set to True to print raw LLM outputs for debugging


def ask_llm(prompt: str) -> str:
    resp = client.generate(model=model, prompt=prompt, options={'temperature': 0.0})
    answer = resp.get('response', '')
    if "...done thinking." in answer:
        return answer.split("...done thinking.")[-1].strip()
    return answer


# --- Helpers for parsing / JSON ---

def extract_first_json(text: str) -> str:
    """
    Extract the first {...} JSON object from possibly messy text.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in: {text!r}")
    return match.group(0)


def split_inline_comment(s: str):
    """
    Split a string into (code, comment) at the first '#'.
    Returns comment WITHOUT the '#'. If no comment, comment=None.
    """
    if "#" not in s:
        return s.strip(), None
    code, comment = s.split("#", 1)
    code = code.strip()
    comment = comment.strip()
    return code, (comment if comment else None)


def parse_predicate(term: str):
    """
    Parse a simple Prolog predicate of the form:
        functor(arg1, arg2, ...)
    Returns (functor: str, args: list[str]) or None if parsing fails.
    """
    term = term.strip().rstrip('.')
    m = re.match(r'^([a-z_][a-zA-Z0-9_]*)\((.*)\)$', term)
    if not m:
        return None
    functor = m.group(1)
    args_raw = m.group(2).strip()
    if not args_raw:
        args = []
    else:
        # Simple arg split (no nested terms in this toy domain)
        args = [a.strip() for a in args_raw.split(',')]
    return functor, args


def is_variable(s: str) -> bool:
    """
    Prolog-ish variable check: starts with uppercase letter or '_'.
    """
    s = s.strip()
    return bool(s) and (s[0].isupper() or s[0] == '_')


# --- Core Prolog helpers ---

def check_exact_match(goal: str, fact: str) -> bool:
    """Check if goal matches fact exactly (no variables)."""
    return goal.strip().rstrip('.') == fact.strip().rstrip('.')


def unify_args(args_goal, args_fact, env=None):
    """
    Unify two argument lists (flat terms, no nesting) under an environment.

    args_goal: list[str]  from the GOAL predicate
    args_fact: list[str]  from the FACT/RULE-HEAD predicate
    env      : dict or None   existing bindings, e.g. {"X": "times_square"}

    Returns:
        - None if unification fails
        - env (possibly modified) if unification succeeds
    """
    if env is None:
        env = {}

    if len(args_goal) != len(args_fact):
        return None

    for g, f in zip(args_goal, args_fact):
        g = g.strip()
        f = f.strip()

        g_is_var = is_variable(g)
        f_is_var = is_variable(f)

        # both constants
        if not g_is_var and not f_is_var:
            if g != f:
                return None
            continue

        # goal var, fact const
        if g_is_var and not f_is_var:
            if g in env:
                if env[g] != f:
                    return None
            else:
                env[g] = f
            continue

        # goal const, fact var  (treat fact vars as wildcards)
        if not g_is_var and f_is_var:
            if f in env:
                if env[f] != g:
                    return None
            else:
                env[f] = g
            continue

        # both variables
        if g_is_var and f_is_var:
            if g in env and f in env:
                if env[g] != env[f]:
                    return None
            elif g in env:
                env[f] = env[g]
            elif f in env:
                env[g] = env[f]
            # else both unbound → no constraint
            continue

    return env


def unify_arg_lists(args_rule_head, args_goal):
    """
    Wrapper used by get_subgoals: unify rule-head args with goal args.
    """
    return unify_args(args_rule_head, args_goal, env={})


def unify_with_fact(goal: str, fact: str):
    """
    Purely algorithmic unification between a GOAL and a FACT (or rule head).

    Returns:
        None      -> NO unification
        {}        -> EXACT ground match (no variables)
        dict      -> bindings, e.g. {"Y": "times_square"}
    """

    parsed_goal = parse_predicate(goal)
    parsed_fact = parse_predicate(fact)
    if parsed_goal is None or parsed_fact is None:
        return None

    fun_g, args_g = parsed_goal
    fun_f, args_f = parsed_fact

    # Functor or arity mismatch
    if fun_g != fun_f or len(args_g) != len(args_f):
        return None

    # If they are exactly the same string (ignoring trailing dot), treat as EXACT
    if check_exact_match(goal, fact):
        return {}

    env = unify_args(args_g, args_f, env={})
    if env is None:
        return None

    return env


def apply_bindings(goals, bindings):
    """
    Apply variable bindings to goals using pure string/term substitution.

    goals: list[str]   e.g. ["reachable(Y, Z)", "connected(Z, X)"]
    bindings: dict     e.g. {"Y": "times_square"}

    Returns: list[str] of instantiated goals.
    """
    if not bindings or not goals:
        return goals

    new_goals = []

    for g in goals:
        parsed = parse_predicate(g)
        if parsed is None:
            # If we can't parse it as a predicate, leave as-is
            new_goals.append(g)
            continue

        functor, args = parsed
        new_args = []
        for a in args:
            a_stripped = a.strip()
            if is_variable(a_stripped) and a_stripped in bindings:
                new_args.append(bindings[a_stripped])
            else:
                new_args.append(a_stripped)

        new_goal = f"{functor}({', '.join(new_args)})"
        new_goals.append(new_goal)

    return new_goals


def find_matching_rules_only(goal, rules_list):
    """
    Find ONLY rules (not facts) whose HEAD can unify with the given goal.

    IMPORTANT: This version is purely syntactic: it only checks functor and arity.
    We do NOT ask the LLM here, to avoid mismatched heads like reachable/2
    being applied to connected/2 goals.

    rules_list: list[(num, head, body)]
    Returns: list[int] of rule numbers.
    """
    parsed_goal = parse_predicate(goal)
    if parsed_goal is None:
        return []
    fun_g, args_g = parsed_goal
    arity_g = len(args_g)

    matching = []
    for num, head, body in rules_list:
        parsed_head = parse_predicate(head)
        if parsed_head is None:
            continue
        fun_h, args_h = parsed_head
        if fun_h == fun_g and len(args_h) == arity_g:
            matching.append(num)
    return matching


def substitute_in_atom(atom: str, bindings: dict) -> str:
    """
    Apply variable bindings to a single Prolog atom, e.g.:

        atom     = "connected(X, Y)"
        bindings = {"X": "union_square", "Y": "bryant_park"}

    Returns:
        "connected(union_square, bryant_park)"
    """
    parsed = parse_predicate(atom)
    if parsed is None:
        return atom  # best-effort fallback

    functor, args = parsed
    new_args = []

    for a in args:
        a_stripped = a.strip()
        if is_variable(a_stripped) and a_stripped in bindings:
            new_args.append(bindings[a_stripped])
        else:
            new_args.append(a_stripped)

    return f"{functor}({', '.join(new_args)})"


def split_body_atoms(body_str: str):
    """
    Split a rule body like:
        "connected(X, Y), reachable(Y, Z)"
    into:
        ["connected(X, Y)", "reachable(Y, Z)"]

    It is parentheses-aware, so it will NOT split on commas that are
    inside argument lists.
    """
    body_str = body_str.strip()
    atoms = []
    current = []
    depth = 0  # parentheses nesting depth

    for ch in body_str:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth = max(depth - 1, 0)
            current.append(ch)
        elif ch == ',' and depth == 0:
            # top-level comma → split here
            atom = ''.join(current).strip()
            if atom:
                atoms.append(atom)
            current = []
        else:
            current.append(ch)

    # Flush the last atom
    atom = ''.join(current).strip()
    if atom:
        atoms.append(atom)

    return atoms


def get_subgoals(goal: str, rule_head: str, rule_body: str):
    """
    Algorithmic ONE-STEP SLD resolution (purely symbolic).
    """
    parsed_goal = parse_predicate(goal)
    parsed_head = parse_predicate(rule_head)

    if parsed_goal is None or parsed_head is None:
        return None

    fun_g, args_g = parsed_goal
    fun_h, args_h = parsed_head

    if fun_g != fun_h or len(args_g) != len(args_h):
        return None

    bindings = unify_arg_lists(args_h, args_g)
    if bindings is None:
        return None

    body_str = rule_body.strip()
    if not body_str:
        return []

    body_atoms = split_body_atoms(body_str)
    if not body_atoms:
        return []

    subgoals = [substitute_in_atom(atom, bindings) for atom in body_atoms]
    return subgoals if subgoals else None


# --- KB comment extraction (inline + full-line) ---

def parse_kb_predicate_comments(kb: str):
    """
    Supports BOTH:
      - full-line comments starting with '#'
      - inline comments after a numbered clause: '... . # comment'

    Returns:
        dict mapping "predicate/arity" -> comment string
        e.g. { "connected/2": "Connected means directly adjacent ..." }
    """
    predicate_comments = {}
    pending_comments = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Full-line comment
        if line.startswith("#"):
            pending_comments.append(line.lstrip("#").strip())
            continue

        # Numbered clause
        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        content_raw = m.group(2).strip()
        content, inline_comment = split_inline_comment(content_raw)

        clause = content.strip().rstrip(".")
        head_str = clause.split(":-", 1)[0].strip()
        parsed = parse_predicate(head_str)
        if parsed is None:
            pending_comments = []
            continue

        functor, args = parsed
        key = f"{functor}/{len(args)}"

        combined = []
        if pending_comments:
            combined.append(" ".join(pending_comments))
        if inline_comment:
            combined.append(inline_comment)

        if combined:
            predicate_comments[key] = " ".join(combined).strip()

        pending_comments = []

    return predicate_comments


# --- BFS Prolog engine ---

def bfs_prolog_metro(goal: str, kb: str, max_depth: int = 10) -> bool:
    """
    BFS with correct fact/rule distinction.
    """

    facts = []
    rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        line = strip_inline_comment(line).strip()
        if not line:
            continue
        
        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content_raw = match.group(2).strip()
            content, _ = split_inline_comment(content_raw)

            if not content:
                continue

            if ':-' in content:
                head, body = content.split(':-', 1)
                rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                facts.append((num, content.rstrip('.')))

    queue = deque([(goal, [], [], 0)])
    visited = set()

    print(f"\nGoal: {goal}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        if depth >= max_depth:
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")

        # 1) Exact fact match
        fact_matched = False
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")

                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return True

                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))
                fact_matched = True
                break

        if fact_matched:
            continue

        # 2) Fact unification
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            print(f"  ✓ Fact {num}: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)

            if not instantiated:
                print(f"✓✓ SUCCESS at depth {depth + 1}")
                return True

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))

        # 3) Rules
        matching_rules = find_matching_rules_only(current, rules)
        if matching_rules:
            print(f"  Matching rules: {matching_rules}")

            for rule_num in matching_rules:
                for num, head, body in rules:
                    if num == rule_num:
                        subgoals = get_subgoals(current, head, body)
                        if subgoals:
                            print(f"  Rule {num}: → {subgoals}")
                            all_goals = subgoals + remaining
                            next_goal = all_goals[0]
                            next_remaining = all_goals[1:]
                            queue.append((next_goal, next_remaining, path + [f"Rule {num}"], depth + 1))
                        break

    print("✗ FAILED")
    return False


def bfs_prolog_collect(goal: str, kb: str, max_depth: int = 10):
    """
    Like bfs_prolog_metro, but returns unresolved atoms too.
    """

    facts = []
    rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        line = strip_inline_comment(line).strip()
        if not line:
            continue
        
        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content_raw = match.group(2).strip()
            content, _ = split_inline_comment(content_raw)

            if not content:
                continue

            if ':-' in content:
                head, body = content.split(':-', 1)
                rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                facts.append((num, content.rstrip('.')))

    queue = deque([(goal, [], [], 0)])
    visited = set()
    unresolved_atoms = set()

    print(f"\n[COLLECT] Goal: {goal}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        if depth >= max_depth:
            unresolved_atoms.add(current)
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")

        progress = False

        # 1) Exact fact match
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")

                step = f"Fact {num}"
                new_path = path + [step]

                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return {
                        "success": True,
                        "proof_path": new_path,
                        "unresolved_atoms": set()
                    }

                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, new_path, depth + 1))

                progress = True
                break

        if progress:
            continue

        # 2) Fact unification
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            progress = True

            print(f"  ✓ Fact {num}: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)

            step = f"Fact {num}"
            new_path = path + [step]

            if not instantiated:
                print(f"✓✓ SUCCESS at depth {depth + 1}")
                return {
                    "success": True,
                    "proof_path": new_path,
                    "unresolved_atoms": set()
                }

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, new_path, depth + 1))

        if progress:
            continue

        # 3) Rules
        matching_rules = find_matching_rules_only(current, rules)
        if matching_rules:
            print(f"  Matching rules: {matching_rules}")

        for rule_num in matching_rules:
            for num, head, body in rules:
                if num == rule_num:
                    subgoals = get_subgoals(current, head, body)
                    if subgoals:
                        print(f"  Rule {num}: → {subgoals}")
                        progress = True
                        all_goals = subgoals + remaining
                        next_goal = all_goals[0]
                        next_remaining = all_goals[1:]
                        step = f"Rule {num}"
                        new_path = path + [step]
                        queue.append((next_goal, next_remaining, new_path, depth + 1))
                    break

        if not progress:
            print(f"  ✗ No facts or rules apply to: {current}")
            unresolved_atoms.add(current)

    print("✗ FAILED (collect mode)")
    return {
        "success": False,
        "proof_path": [],
        "unresolved_atoms": unresolved_atoms
    }


def generate_background_hypotheses(goal: str, kb: str, unresolved_atoms, predicate_comments: dict, max_atoms: int = 5):
    """
    Generates hypotheses but rigorously filters out 'shortcuts' that contradict
    the topology of the existing metro map.
    """
    hypotheses = []

    # --- 0) Build hard connected/2 graph from KB for validation ---
    hard_adj = {}
    hard_edges = set()
    stations = set()

    # Extract all existing hard connections
    for line in kb.strip().split("\n"):
        line = line.strip()
        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m: continue
        
        content_raw = m.group(2).strip()
        content, _ = split_inline_comment(content_raw)
        content = (content or "").strip()
        
        # Only parse ground facts, ignore rules for graph building
        if ':-' in content: continue
        
        atom0 = content.rstrip('.').strip()
        p = parse_predicate(atom0)
        if p and p[0] == "connected" and len(p[1]) == 2:
            a, b = p[1][0].strip(), p[1][1].strip()
            hard_adj.setdefault(a, []).append(b)
            hard_edges.add((a, b))
            stations.add(a); stations.add(b)

    def shortest_path_len(src: str, dst: str, limit: int = 6):
        """BFS to find distance in existing Hard KB."""
        if src == dst: return 0
        q = deque([(src, 0)])
        seen = {src}
        while q:
            curr, d = q.popleft()
            if d >= limit: continue
            for nxt in hard_adj.get(curr, []):
                if nxt == dst: return d + 1
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, d + 1))
        return None

    def is_shortcut_edge(u: str, v: str) -> bool:
        """
        Returns True if the Hard KB implies that u and v are connected 
        indirectly (distance >= 2), meaning a direct connection suggests
        the LLM is skipping stations (hallucinating).
        """
        # Check forward path u -> ... -> v
        d_fwd = shortest_path_len(u, v)
        if d_fwd is not None and d_fwd > 1:
            return True
        return False

    # --- 1) Filter unresolved atoms to simple, ground atoms ---
    atom_list = list(unresolved_atoms)
    ground_atoms = []
    for atom in atom_list:
        if '(' not in atom or ')' not in atom: continue
        if re.search(r'\b[A-Z_]\w*\b', atom.split('(', 1)[1]): continue # Skip vars
        ground_atoms.append(atom)

    if not ground_atoms:
        return []

    # Prioritize atoms (simple heuristic)
    ordered = ground_atoms[:max_atoms]

    # --- 2) Query LLM ---
    for atom in ordered:
        parsed = parse_predicate(atom)
        semantic_hint = ""
        if parsed:
            semantic_hint = predicate_comments.get(f"{parsed[0]}/{len(parsed[1])}", "")

        prompt = f"""
You are a cautious Prolog expert with access to real-world background knowledge.
GOAL: {goal}
KNOWLEDGE BASE:
{kb}
FAILED SUBGOAL: {atom}
SEMANTIC HINTS: {semantic_hint}

Task:
Propose a SMALL set of additional Prolog facts (connected/2) that are LIKELY true
and help prove the goal.
- IMPORTANT: Do NOT propose connections that skip stations.
- If A is connected to B, and B to C, do NOT say A is connected to C.
- Only propose IMMEDIATE neighbors.

Respond in JSON: {{ "hypotheses": [ {{ "clause": "connected(a,b).", "confidence": 0.9 }} ] }}
"""
        raw = ask_llm(prompt).strip()
        
        try:
            data = json.loads(extract_first_json(raw))
            raw_hyps = data.get("hypotheses", [])
        except:
            continue

        for h in raw_hyps:
            clause = (h.get("clause") or "").strip()
            if not clause.endswith('.'): clause += "."
            
            try:
                conf = float(h.get("confidence", 0.0))
            except:
                conf = 0.0

            # Parse to validate
            atom_str = clause.rstrip('.').strip()
            p2 = parse_predicate(atom_str)
            
            # 2.5) VALIDATION LAYER
            if not (p2 and p2[0] == "connected" and len(p2[1]) == 2):
                continue # Only accept connected/2

            u, v = p2[1][0].strip(), p2[1][1].strip()

            # Reject duplicates
            if (u, v) in hard_edges: continue
            
            # REJECT SHORTCUTS (The Fix)
            if is_shortcut_edge(u, v):
                print(f"!!! REJECTING HALLUCINATION: {u} -> {v} (Path already exists via other nodes)")
                continue

            hypotheses.append({
                "clause": clause,
                "confidence": conf,
                "from_atom": atom
            })

    # --- 3) Deduplicate ---
    dedup = {}
    for h in hypotheses:
        key = h["clause"]
        if key not in dedup or h["confidence"] > dedup[key]["confidence"]:
            dedup[key] = h

    return list(dedup.values())

def _find_max_line_number_in_kb(kb: str) -> int:
    max_num = 0
    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        # strip inline comments here too
        content_raw = m.group(2).strip()
        content, _ = split_inline_comment(content_raw)
        if not content:
            continue

        num = int(m.group(1))
        if num > max_num:
            max_num = num
    return max_num


def _is_fact_clause(clause: str) -> bool:
    return ':-' not in clause


def _split_rule_clause(clause: str):
    clause = clause.strip()
    if clause.endswith('.'):
        clause = clause[:-1]
    head_part, body_part = clause.split(':-', 1)
    head = head_part.strip()
    body_str = body_part.strip()
    return head, body_str


def attach_hypotheses_to_kb(kb: str, hypotheses):
    soft_facts = []
    soft_rules = []

    max_num = _find_max_line_number_in_kb(kb)
    next_num = max_num + 1

    for h in hypotheses:
        clause = (h.get("clause") or "").strip()
        if not clause:
            continue

        conf = float(h.get("confidence", 0.0))

        if not clause.endswith('.'):
            clause = clause + '.'

        if _is_fact_clause(clause):
            atom = clause.rstrip('.').strip()
            soft_facts.append((next_num, atom, conf))
        else:
            head, body_str = _split_rule_clause(clause)
            soft_rules.append((next_num, head, body_str, conf))

        next_num += 1

    return {"facts": soft_facts, "rules": soft_rules}


def strip_inline_comment(s: str) -> str:
    # Everything after '#' is non-executable annotation
    return s.split('#', 1)[0].rstrip()


def generate_background_hypotheses(goal: str, kb: str, unresolved_atoms, predicate_comments: dict, max_atoms: int = 5):
    """
    Generates hypotheses with O(1) shortcut checking using pre-computed distances.
    """
    hypotheses = []

    # --- HELPER: Pre-compute All-Pairs Shortest Paths (Floyd-Warshall / Repeated BFS) ---
    # We do this ONCE per function call, not inside the loop.
    hard_adj = {}
    hard_edges = set()
    stations = set()
    
    # 1. Parse KB for Graph
    for line in kb.strip().split("\n"):
        line = line.strip()
        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m: continue
        content_raw = m.group(2).strip()
        content, _ = split_inline_comment(content_raw)
        content = (content or "").strip()
        if ':-' in content: continue # Skip rules
        
        atom0 = content.rstrip('.').strip()
        p = parse_predicate(atom0)
        if p and p[0] == "connected" and len(p[1]) == 2:
            a, b = p[1][0].strip(), p[1][1].strip()
            hard_adj.setdefault(a, []).append(b)
            hard_edges.add((a, b))
            stations.add(a); stations.add(b)

    # 2. Compute Distances Matrix
    # distance_map[(start, end)] = int_distance
    distance_map = {}
    
    for start_node in stations:
        # Run local BFS from every node to fill the map
        q = deque([(start_node, 0)])
        visited = {start_node}
        distance_map[(start_node, start_node)] = 0
        
        while q:
            curr, d = q.popleft()
            if d > 5: continue # Optimization: We only care about short paths for validation
            
            for nxt in hard_adj.get(curr, []):
                if nxt not in visited:
                    visited.add(nxt)
                    distance_map[(start_node, nxt)] = d + 1
                    q.append((nxt, d + 1))

    # --- END HELPER ---

    # --- 1) Filter unresolved atoms ---
    atom_list = list(unresolved_atoms)
    ground_atoms = []
    for atom in atom_list:
        if '(' not in atom or ')' not in atom: continue
        if re.search(r'\b[A-Z_]\w*\b', atom.split('(', 1)[1]): continue 
        ground_atoms.append(atom)

    if not ground_atoms: return []
    ordered = ground_atoms[:max_atoms]

    # --- 2) Query LLM ---
    for atom in ordered:
        parsed = parse_predicate(atom)
        semantic_hint = ""
        if parsed:
            semantic_hint = predicate_comments.get(f"{parsed[0]}/{len(parsed[1])}", "")

        prompt = f"""
You are a cautious Prolog expert.
GOAL: {goal}
KNOWLEDGE BASE:
{kb}
FAILED SUBGOAL: {atom}
SEMANTIC HINTS: {semantic_hint}

Task:
Propose a SMALL set of missing connected/2 facts.
- IMPORTANT: Do NOT propose connections that skip stations.
- If the KB implies A->...->B (distance > 1), do NOT add connected(A,B).

Respond in JSON: {{ "hypotheses": [ {{ "clause": "connected(a,b).", "confidence": 0.9 }} ] }}
"""
        raw = ask_llm(prompt).strip()
        
        try:
            data = json.loads(extract_first_json(raw))
            raw_hyps = data.get("hypotheses", [])
        except:
            continue

        for h in raw_hyps:
            clause = (h.get("clause") or "").strip()
            if not clause.endswith('.'): clause += "."
            try: conf = float(h.get("confidence", 0.0))
            except: conf = 0.0

            atom_str = clause.rstrip('.').strip()
            p2 = parse_predicate(atom_str)
            
            if not (p2 and p2[0] == "connected" and len(p2[1]) == 2): continue

            u, v = p2[1][0].strip(), p2[1][1].strip()

            if (u, v) in hard_edges: continue
            if u == v: continue

            # --- O(1) SHORTCUT CHECK ---
            # If distance is known and > 1, it's a shortcut
            dist = distance_map.get((u, v), None)
            
            if dist is not None and dist > 1:
                if DEBUG: print(f"!!! Rejecting Shortcut: {u}->{v} (Dist {dist})")
                continue
            # ---------------------------

            hypotheses.append({
                "clause": clause,
                "confidence": conf,
                "from_atom": atom
            })

    dedup = {}
    for h in hypotheses:
        key = h["clause"]
        if key not in dedup or h["confidence"] > dedup[key]["confidence"]:
            dedup[key] = h

    return list(dedup.values())

def solve_with_background(
    goal: str,
    kb: str,
    max_depth: int = 15,
    max_soft=None,
    hard_result=None,
):
    """
    High-level pipeline (unchanged), but now reads predicate comments from kb.
    """
    predicate_comments = parse_kb_predicate_comments(kb)

    print("\n========================================")
    print(f"SOLVE WITH BACKGROUND: {goal}")
    print("========================================\n")

    if hard_result is None:
        print(">>> Phase 1: Hard-KB BFS (bfs_prolog_collect)")
        hard_result = bfs_prolog_collect(goal, kb, max_depth=max_depth)
        print("Hard-KB result:", hard_result)
    else:
        print(">>> Phase 1: Hard-KB BFS result already computed, reusing it.")
        print("Hard-KB result:", hard_result)

    if hard_result.get("success"):
        print("\n>>> Result: HARD_SUCCESS (no background hypotheses needed)\n")
        return {
            "status": "HARD_SUCCESS",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": []
        }

    unresolved_atoms = hard_result.get("unresolved_atoms", set())
    if not unresolved_atoms:
        print("\nNo unresolved atoms to explain; cannot generate hypotheses.")
        print(">>> Result: FAILURE\n")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": []
        }

    print("\n>>> Phase 2: Generate background hypotheses")
    print("Unresolved atoms:", unresolved_atoms)

    hypotheses = generate_background_hypotheses(
        goal=goal,
        kb=kb,
        unresolved_atoms=unresolved_atoms,
        predicate_comments=predicate_comments
    )

    if hypotheses is None:
        hypotheses = []

    if not hypotheses:
        print("Hypotheses returned by LLM: []")
        print("\nLLM returned NO hypotheses; cannot build soft KB.")
        print(">>> Result: FAILURE\n")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": []
        }

    print("Hypotheses returned by LLM:")
    for h in hypotheses:
        print("  - Clause:", h.get("clause"),
              "| Conf:", h.get("confidence"),
              "| From atom:", h.get("from_atom"))

    print("\n>>> Phase 3: Attach hypotheses to soft KB")
    soft_kb = attach_hypotheses_to_kb(kb, hypotheses)
    print("Soft KB facts:", soft_kb.get("facts", []))
    print("Soft KB rules:", soft_kb.get("rules", []))

    print("\n>>> Phase 4: Soft BFS (bfs_prolog_metro_soft)")
    soft_result = bfs_prolog_metro_soft(
        goal=goal,
        kb=kb,
        soft_kb=soft_kb,
        max_depth=max_depth,
        max_soft=max_soft,
    )
    print("Soft-BFS result:", soft_result)

    if soft_result.get("success"):
        print("\n>>> Result: SOFT_SUCCESS (proof found using background hypotheses)\n")
        return {
            "status": "SOFT_SUCCESS",
            "hard_result": hard_result,
            "soft_result": soft_result,
            "hypotheses": hypotheses
        }

    print("\n>>> Result: SOFT_FAILURE (no proof even with background hypotheses)\n")
    return {
        "status": "SOFT_FAILURE",
        "hard_result": hard_result,
        "soft_result": soft_result,
        "hypotheses": hypotheses
    }


def omit_facts_from_kb(kb: str, omit_numbers):
    """
    Return a new KB string with numbered lines in `omit_numbers` removed.
    Preserves original line text (including comments), but matches by number.
    """
    omit_numbers = set(omit_numbers)
    new_lines = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        num = int(m.group(1))
        if num in omit_numbers:
            continue

        new_lines.append(line)

    return "\n".join(new_lines)


# Natural language to Prolog using LLM, algo SLD resolution

def nl_kb_to_prolog_kb(nl_kb_text: str, start_index: int = 1) -> list[str]:
    """
    Convert a *pure natural-language* description of a domain + rules
    into a numbered Prolog knowledge base.
    """

    nl_kb_text = (nl_kb_text or "").strip()
    if not nl_kb_text:
        return []

    prompt = f"""
You are a Prolog formalization assistant.

The user will give you a natural-language description of a small domain,
including objects, relationships, and logical rules.

Your job is to convert that description into a set of Prolog clauses
(facts and rules).

Guidelines:
- Use lowercase atoms for concrete entities (e.g. union_square, times_square,
  grand_central, bryant_park).
- Use uppercase identifiers for variables (e.g. X, Y, Z).
- Choose predicate names that are short, descriptive, and consistent,
  for example: connected/2, reachable/2, located_in/2, etc.
- A fact must look like:
    connected(times_square, bryant_park).
- A rule must look like:
    reachable(X, Z) :- connected(X, Y), reachable(Y, Z).
- Every clause MUST end with a single period '.'.
- Do NOT include any line numbers in your output clauses.
- Do NOT add explanations or comments in the Prolog code.

Here is the NATURAL LANGUAGE description of the knowledge base:

\"\"\"{nl_kb_text}\"\"\"

Respond ONLY in this JSON format (and nothing else):

{{
  "clauses": [
    {{
      "clause": "connected(union_square, times_square)."
    }},
    {{
      "clause": "reachable(X, Y) :- connected(X, Y)."
    }}
  ]
}}
"""

    raw = ask_llm(prompt).strip()
    try:
        data = json.loads(extract_first_json(raw))
    except Exception as e:
        print("[nl_kb_to_prolog_kb] JSON parse error:", e)
        print("Raw LLM output:", raw)
        return []

    raw_clauses = data.get("clauses", [])
    if not isinstance(raw_clauses, list):
        print("[nl_kb_to_prolog_kb] 'clauses' field is not a list:", raw_clauses)
        return []

    cleaned_clauses = []

    for item in raw_clauses:
        if isinstance(item, str):
            clause = item.strip()
        elif isinstance(item, dict):
            clause = (item.get("clause") or "").strip()
        else:
            continue

        if not clause:
            continue

        m_num = re.match(r'^\s*(\d+)\.\s*(.+)$', clause)
        if m_num:
            clause = m_num.group(2).strip()

        clause = clause.rstrip()
        if not clause.endswith('.'):
            clause = clause + "."
        else:
            clause = re.sub(r'\.+$', '.', clause)

        body_str = clause[:-1].strip()
        if ':-' in body_str:
            head_part, body_part = body_str.split(':-', 1)
            head = head_part.strip()
        else:
            head = body_str

        parsed_head = parse_predicate(head)
        if parsed_head is None:
            print("[nl_kb_to_prolog_kb] Discarding unparsable clause:", clause)
            continue

        cleaned_clauses.append(clause)

    numbered_clauses = []
    next_num = start_index
    for clause in cleaned_clauses:
        numbered_clauses.append(f"{next_num}. {clause}")
        next_num += 1

    return numbered_clauses


# In[64]:


if __name__ == "__main__":
    kb = """
    1. connected(union_square, 14th_street). # connected/2 = ADJACENT STOPS ONLY. Use only when two stations are immediate neighbors on the same line. Do NOT add shortcut edges that skip intermediate stations.
    2. connected(14th_street, 23rd_street).
    3. connected(23rd_street, 34th_street).
    4. connected(34th_street, times_square).
    5. connected(times_square, 42nd_street).
    6. connected(42nd_street, grand_central).
    7. connected(grand_central, bryant_park).
    8. reachable(X, Y) :- connected(X, Y). # reachable/2 = PATH EXISTS. One-hop reachable comes only from connected/2.
    9. reachable(X, Z) :- connected(X, Y), reachable(Y, Z). # reachable/2 is the transitive closure of connected/2 (multi-hop path following only adjacent edges).

    """

    print("===== FULL METRO KB =====")
    print(kb)
    print("====================================\n")

    # Omit fact 1 to force background reasoning (removes union_square -> times_square)
    kb_missing_fact = omit_facts_from_kb(kb, omit_numbers={7})

    print("===== METRO KB WITH FACT REMOVED =====")
    print(kb_missing_fact)
    print("========================================\n")

    test_goal = "reachable(union_square, bryant_park)"

    print("==============================")
    print(f"TEST QUERY: {test_goal}")
    print("==============================\n")

    print(">>> Running bfs_prolog_collect (hard-KB BFS)...")
    collect_result = bfs_prolog_collect(test_goal, kb_missing_fact)
    print("Collect Result:", collect_result)
    print("\n----------------------------------------\n")

    print(">>> Running solve_with_background (full pipeline, reusing hard result)...")
    bg_result = solve_with_background(
        goal=test_goal,
        kb=kb_missing_fact,
        max_depth=15,
        max_soft=None,
        hard_result=collect_result,
    )
    print("Solve-with-background Result:")
    print(bg_result)
    print("\n========================================\n")


# In[71]:


import ollama
import heapq
from itertools import count
from collections import deque
import re
import json
from typing import Optional
import math


# --- Config / LLM setup ---

client = ollama.Client()

model = "gpt-oss:20b"
# model = "qwen:14b"

DEBUG = False  # set to True to print raw LLM outputs for debugging


def ask_llm(prompt: str) -> str:
    resp = client.generate(model=model, prompt=prompt, options={'temperature': 0.0})
    answer = resp.get('response', '')
    if "...done thinking." in answer:
        return answer.split("...done thinking.")[-1].strip()
    return answer


# --- Helpers for parsing / JSON ---

def extract_first_json(text: str) -> str:
    """
    Extract the first {...} JSON object from possibly messy text.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in: {text!r}")
    return match.group(0)


def split_inline_comment(s: str):
    """
    Split a string into (code, comment) at the first '#'.
    Returns comment WITHOUT the '#'. If no comment, comment=None.
    """
    if "#" not in s:
        return s.strip(), None
    code, comment = s.split("#", 1)
    code = code.strip()
    comment = comment.strip()
    return code, (comment if comment else None)


def parse_predicate(term: str):
    """
    Parse a simple Prolog predicate of the form:
        functor(arg1, arg2, ...)
    Returns (functor: str, args: list[str]) or None if parsing fails.
    """
    term = term.strip().rstrip('.')
    m = re.match(r'^([a-z_][a-zA-Z0-9_]*)\((.*)\)$', term)
    if not m:
        return None
    functor = m.group(1)
    args_raw = m.group(2).strip()
    if not args_raw:
        args = []
    else:
        # Simple arg split (no nested terms in this toy domain)
        args = [a.strip() for a in args_raw.split(',')]
    return functor, args


def is_variable(s: str) -> bool:
    """
    Prolog-ish variable check: starts with uppercase letter or '_'.
    """
    s = s.strip()
    return bool(s) and (s[0].isupper() or s[0] == '_')


def strip_inline_comment(s: str) -> str:
    # Everything after '#' is non-executable annotation
    return s.split('#', 1)[0].rstrip()


# --- Core Prolog helpers ---

def check_exact_match(goal: str, fact: str) -> bool:
    """Check if goal matches fact exactly (no variables)."""
    return goal.strip().rstrip('.') == fact.strip().rstrip('.')


def unify_args(args_goal, args_fact, env=None):
    """
    Unify two argument lists (flat terms, no nesting) under an environment.

    args_goal: list[str]  from the GOAL predicate
    args_fact: list[str]  from the FACT/RULE-HEAD predicate
    env      : dict or None   existing bindings, e.g. {"X": "times_square"}

    Returns:
        - None if unification fails
        - env (possibly modified) if unification succeeds
    """
    if env is None:
        env = {}

    if len(args_goal) != len(args_fact):
        return None

    for g, f in zip(args_goal, args_fact):
        g = g.strip()
        f = f.strip()

        g_is_var = is_variable(g)
        f_is_var = is_variable(f)

        # both constants
        if not g_is_var and not f_is_var:
            if g != f:
                return None
            continue

        # goal var, fact const
        if g_is_var and not f_is_var:
            if g in env:
                if env[g] != f:
                    return None
            else:
                env[g] = f
            continue

        # goal const, fact var  (treat fact vars as wildcards)
        if not g_is_var and f_is_var:
            if f in env:
                if env[f] != g:
                    return None
            else:
                env[f] = g
            continue

        # both variables
        if g_is_var and f_is_var:
            if g in env and f in env:
                if env[g] != env[f]:
                    return None
            elif g in env:
                env[f] = env[g]
            elif f in env:
                env[g] = env[f]
            # else both unbound → no constraint
            continue

    return env


def unify_arg_lists(args_rule_head, args_goal):
    """
    Wrapper used by get_subgoals: unify rule-head args with goal args.
    """
    return unify_args(args_rule_head, args_goal, env={})


def unify_with_fact(goal: str, fact: str):
    """
    Purely algorithmic unification between a GOAL and a FACT (or rule head).

    Returns:
        None      -> NO unification
        {}        -> EXACT ground match (no variables)
        dict      -> bindings, e.g. {"Y": "times_square"}
    """
    parsed_goal = parse_predicate(goal)
    parsed_fact = parse_predicate(fact)
    if parsed_goal is None or parsed_fact is None:
        return None

    fun_g, args_g = parsed_goal
    fun_f, args_f = parsed_fact

    # Functor or arity mismatch
    if fun_g != fun_f or len(args_g) != len(args_f):
        return None

    # If they are exactly the same string (ignoring trailing dot), treat as EXACT
    if check_exact_match(goal, fact):
        return {}

    env = unify_args(args_g, args_f, env={})
    if env is None:
        return None

    return env


def apply_bindings(goals, bindings):
    """
    Apply variable bindings to goals using pure string/term substitution.
    """
    if not bindings or not goals:
        return goals

    new_goals = []

    for g in goals:
        parsed = parse_predicate(g)
        if parsed is None:
            new_goals.append(g)
            continue

        functor, args = parsed
        new_args = []
        for a in args:
            a_stripped = a.strip()
            if is_variable(a_stripped) and a_stripped in bindings:
                new_args.append(bindings[a_stripped])
            else:
                new_args.append(a_stripped)

        new_goal = f"{functor}({', '.join(new_args)})"
        new_goals.append(new_goal)

    return new_goals


def find_matching_rules_only(goal, rules_list):
    """
    Find ONLY rules (not facts) whose HEAD can unify with the given goal.
    Purely syntactic: functor + arity.
    """
    parsed_goal = parse_predicate(goal)
    if parsed_goal is None:
        return []
    fun_g, args_g = parsed_goal
    arity_g = len(args_g)

    matching = []
    for num, head, body in rules_list:
        parsed_head = parse_predicate(head)
        if parsed_head is None:
            continue
        fun_h, args_h = parsed_head
        if fun_h == fun_g and len(args_h) == arity_g:
            matching.append(num)
    return matching


def substitute_in_atom(atom: str, bindings: dict) -> str:
    parsed = parse_predicate(atom)
    if parsed is None:
        return atom

    functor, args = parsed
    new_args = []

    for a in args:
        a_stripped = a.strip()
        if is_variable(a_stripped) and a_stripped in bindings:
            new_args.append(bindings[a_stripped])
        else:
            new_args.append(a_stripped)

    return f"{functor}({', '.join(new_args)})"


def split_body_atoms(body_str: str):
    """
    Parentheses-aware split on top-level commas.
    """
    body_str = body_str.strip()
    atoms = []
    current = []
    depth = 0

    for ch in body_str:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth = max(depth - 1, 0)
            current.append(ch)
        elif ch == ',' and depth == 0:
            atom = ''.join(current).strip()
            if atom:
                atoms.append(atom)
            current = []
        else:
            current.append(ch)

    atom = ''.join(current).strip()
    if atom:
        atoms.append(atom)

    return atoms


def get_subgoals(goal: str, rule_head: str, rule_body: str):
    """
    Algorithmic ONE-STEP SLD resolution (purely symbolic).
    """
    parsed_goal = parse_predicate(goal)
    parsed_head = parse_predicate(rule_head)

    if parsed_goal is None or parsed_head is None:
        return None

    fun_g, args_g = parsed_goal
    fun_h, args_h = parsed_head

    if fun_g != fun_h or len(args_g) != len(args_h):
        return None

    bindings = unify_arg_lists(args_h, args_g)
    if bindings is None:
        return None

    body_str = rule_body.strip()
    if not body_str:
        return []

    body_atoms = split_body_atoms(body_str)
    if not body_atoms:
        return []

    subgoals = [substitute_in_atom(atom, bindings) for atom in body_atoms]
    return subgoals if subgoals else None


# --- KB comment extraction (inline + full-line) ---

def parse_kb_predicate_comments(kb: str):
    predicate_comments = {}
    pending_comments = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("#"):
            pending_comments.append(line.lstrip("#").strip())
            continue

        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        content_raw = m.group(2).strip()
        content, inline_comment = split_inline_comment(content_raw)

        clause = content.strip().rstrip(".")
        head_str = clause.split(":-", 1)[0].strip()
        parsed = parse_predicate(head_str)
        if parsed is None:
            pending_comments = []
            continue

        functor, args = parsed
        key = f"{functor}/{len(args)}"

        combined = []
        if pending_comments:
            combined.append(" ".join(pending_comments))
        if inline_comment:
            combined.append(inline_comment)

        if combined:
            predicate_comments[key] = " ".join(combined).strip()

        pending_comments = []

    return predicate_comments


# --- BFS Prolog engine ---

def bfs_prolog_metro(goal: str, kb: str, max_depth: int = 10) -> bool:
    """
    BFS with correct fact/rule distinction.
    """
    facts = []
    rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        line = strip_inline_comment(line).strip()
        if not line:
            continue

        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content_raw = match.group(2).strip()
            content, _ = split_inline_comment(content_raw)

            if not content:
                continue

            if ':-' in content:
                head, body = content.split(':-', 1)
                rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                facts.append((num, content.rstrip('.')))

    queue = deque([(goal, [], [], 0)])
    visited = set()

    print(f"\nGoal: {goal}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        if depth >= max_depth:
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")

        # 1) Exact fact match
        fact_matched = False
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")

                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return True

                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))
                fact_matched = True
                break

        if fact_matched:
            continue

        # 2) Fact unification
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            print(f"  ✓ Fact {num}: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)

            if not instantiated:
                print(f"✓✓ SUCCESS at depth {depth + 1}")
                return True

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, path + [f"Fact {num}"], depth + 1))

        # 3) Rules
        matching_rules = find_matching_rules_only(current, rules)
        if matching_rules:
            print(f"  Matching rules: {matching_rules}")

            for rule_num in matching_rules:
                for num, head, body in rules:
                    if num == rule_num:
                        subgoals = get_subgoals(current, head, body)
                        if subgoals:
                            print(f"  Rule {num}: → {subgoals}")
                            all_goals = subgoals + remaining
                            next_goal = all_goals[0]
                            next_remaining = all_goals[1:]
                            queue.append((next_goal, next_remaining, path + [f"Rule {num}"], depth + 1))
                        break

    print("✗ FAILED")
    return False


def bfs_prolog_collect(goal: str, kb: str, max_depth: int = 10):
    """
    Like bfs_prolog_metro, but returns unresolved atoms too.
    """
    facts = []
    rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        line = strip_inline_comment(line).strip()
        if not line:
            continue

        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content_raw = match.group(2).strip()
            content, _ = split_inline_comment(content_raw)

            if not content:
                continue

            if ':-' in content:
                head, body = content.split(':-', 1)
                rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                facts.append((num, content.rstrip('.')))

    queue = deque([(goal, [], [], 0)])
    visited = set()
    unresolved_atoms = set()

    print(f"\n[COLLECT] Goal: {goal}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        if depth >= max_depth:
            unresolved_atoms.add(current)
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")

        progress = False

        # 1) Exact fact match
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")

                step = f"Fact {num}"
                new_path = path + [step]

                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return {
                        "success": True,
                        "proof_path": new_path,
                        "unresolved_atoms": set()
                    }

                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, new_path, depth + 1))

                progress = True
                break

        if progress:
            continue

        # 2) Fact unification
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            progress = True

            print(f"  ✓ Fact {num}: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)

            step = f"Fact {num}"
            new_path = path + [step]

            if not instantiated:
                print(f"✓✓ SUCCESS at depth {depth + 1}")
                return {
                    "success": True,
                    "proof_path": new_path,
                    "unresolved_atoms": set()
                }

            next_goal = instantiated[0]
            next_remaining = instantiated[1:]
            queue.append((next_goal, next_remaining, new_path, depth + 1))

        if progress:
            continue

        # 3) Rules
        matching_rules = find_matching_rules_only(current, rules)
        if matching_rules:
            print(f"  Matching rules: {matching_rules}")

        for rule_num in matching_rules:
            for num, head, body in rules:
                if num == rule_num:
                    subgoals = get_subgoals(current, head, body)
                    if subgoals:
                        print(f"  Rule {num}: → {subgoals}")
                        progress = True
                        all_goals = subgoals + remaining
                        next_goal = all_goals[0]
                        next_remaining = all_goals[1:]
                        step = f"Rule {num}"
                        new_path = path + [step]
                        queue.append((next_goal, next_remaining, new_path, depth + 1))
                    break

        if not progress:
            print(f"  ✗ No facts or rules apply to: {current}")
            unresolved_atoms.add(current)

    print("✗ FAILED (collect mode)")
    return {
        "success": False,
        "proof_path": [],
        "unresolved_atoms": unresolved_atoms
    }


# ============================================================
# FAST background hypothesis generation (single-call + cache)
# ============================================================

_BG_HYP_CACHE = {}  # (kb_sig, goal, tuple(atoms), max_hyp_per_atom) -> list[hypothesis]


def _kb_signature_for_bg(kb: str) -> str:
    """
    Cheap signature: only connected/2 facts + reachable clauses (ignores comments).
    """
    parts = []
    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        content_raw = m.group(2).strip()
        content, _ = split_inline_comment(content_raw)
        content = (content or "").strip()
        if not content:
            continue

        content = content.rstrip('.').strip()
        if content.startswith("connected(") or content.startswith("reachable("):
            parts.append(content)

    return str(hash("\n".join(parts)))


def _extract_connected_facts_and_stations(kb: str):
    hard_adj = {}
    hard_edges = set()
    stations = set()
    connected_facts = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        content_raw = m.group(2).strip()
        content, _ = split_inline_comment(content_raw)
        content = (content or "").strip()
        if not content:
            continue

        if ':-' in content:
            continue

        atom0 = content.rstrip('.').strip()
        p = parse_predicate(atom0)
        if p and p[0] == "connected" and len(p[1]) == 2:
            a, b = p[1][0].strip(), p[1][1].strip()
            hard_adj.setdefault(a, []).append(b)
            hard_edges.add((a, b))
            stations.add(a)
            stations.add(b)
            connected_facts.append(f"connected({a}, {b}).")

    return hard_adj, hard_edges, stations, connected_facts


def _shortest_path_len_bounded(adj: dict, src: str, dst: str, max_depth: int = 25):
    if src == dst:
        return 0
    q = deque([(src, 0)])
    seen = {src}
    while q:
        node, d = q.popleft()
        if d >= max_depth:
            continue
        for nb in adj.get(node, []):
            if nb == dst:
                return d + 1
            if nb not in seen:
                seen.add(nb)
                q.append((nb, d + 1))
    return None


def _infer_preferred_atom_for_chain(goal: str, hard_adj: dict) -> Optional[str]:
    gp = parse_predicate(goal)
    if not gp or len(gp[1]) != 2:
        return None
    goal_src, goal_dst = gp[1][0].strip(), gp[1][1].strip()

    last = goal_src
    seen = set()
    while last in hard_adj and len(hard_adj[last]) == 1 and last not in seen:
        seen.add(last)
        last = hard_adj[last][0]

    if last and goal_dst:
        return f"connected({last}, {goal_dst})"
    return None


def _select_atoms_for_bg(unresolved_atoms, preferred_atom: Optional[str], max_atoms: int):
    atom_list = list(unresolved_atoms)
    ground = []
    for atom in atom_list:
        atom = atom.strip()
        if not atom or '(' not in atom or ')' not in atom:
            continue
        inside = atom.split('(', 1)[1].rsplit(')', 1)[0]
        if re.search(r'\b[A-Z_]\w*\b', inside):
            continue
        ground.append(atom)

    if preferred_atom:
        ground = [preferred_atom] + [a for a in ground if a != preferred_atom]

    if max_atoms is not None:
        ground = ground[:max_atoms]
    return ground


def generate_background_hypotheses_fast(
    goal: str,
    kb: str,                      # must be the same KB used for hard BFS
    hard_result: dict,
    predicate_comments: dict,
    max_atoms: int = 5,
    max_hyp_per_atom: int = 2,
    prompt_fact_limit: int = 60,
):
    unresolved_atoms = hard_result.get("unresolved_atoms", set()) if hard_result else set()
    if not unresolved_atoms:
        return []

    hard_adj, hard_edges, stations, connected_facts = _extract_connected_facts_and_stations(kb)
    preferred_atom = _infer_preferred_atom_for_chain(goal, hard_adj)
    atoms = _select_atoms_for_bg(unresolved_atoms, preferred_atom, max_atoms=max_atoms)
    if not atoms:
        print("[generate_background_hypotheses_fast] No suitable ground atoms.")
        return []

    kb_sig = _kb_signature_for_bg(kb)
    cache_key = (kb_sig, goal, tuple(atoms), max_hyp_per_atom)
    if cache_key in _BG_HYP_CACHE:
        return _BG_HYP_CACHE[cache_key]

    facts_for_prompt = connected_facts[:prompt_fact_limit]

    needed = {"connected/2"}
    for a in atoms:
        p = parse_predicate(a)
        if p:
            needed.add(f"{p[0]}/{len(p[1])}")

    semantic_lines = []
    for k in sorted(needed):
        if k in predicate_comments:
            semantic_lines.append(f"- {k}: {predicate_comments[k]}")
    semantic_hint_block = "\n".join(semantic_lines) if semantic_lines else "(none)"

    atoms_block = "\n".join([f"- {a}" for a in atoms])

    prompt = f"""
You are a cautious Prolog expert. You are proposing missing FACTS only.

GOAL:
  {goal}

HARD KB SNAPSHOT (partial; do NOT assume any other edges exist):
Connected facts (sample):
{chr(10).join(facts_for_prompt)}

Predicate semantics (ground truth):
{semantic_hint_block}

The proof failed because these subgoals could not be proven:
{atoms_block}

Task:
For each failed subgoal above, propose up to {max_hyp_per_atom} additional Prolog FACTS
that are likely true and would help prove the GOAL.

Constraints:
- Output ONLY connected/2 FACTS. No rules.
- Each fact MUST end with a period.
- Prefer facts that connect existing stations seen in the snapshot.
- If a suggested edge is uncertain or could be a "shortcut", lower confidence.

Return ONLY valid JSON:

{{
  "by_atom": {{
    "connected(42nd_street, bryant_park)": [
      {{"clause":"connected(grand_central, bryant_park).","confidence":0.95}},
      {{"clause":"connected(42nd_street, bryant_park).","confidence":0.55}}
    ]
  }}
}}
""".strip()

    raw = ask_llm(prompt).strip()
    if DEBUG:
        print("\n[DEBUG generate_background_hypotheses_fast] raw:\n", raw)

    try:
        data = json.loads(extract_first_json(raw))
    except Exception as e:
        print("[generate_background_hypotheses_fast] JSON parse error:", e)
        print("Raw LLM output:", raw)
        _BG_HYP_CACHE[cache_key] = []
        return []

    by_atom = data.get("by_atom", {})
    if not isinstance(by_atom, dict):
        _BG_HYP_CACHE[cache_key] = []
        return []

    def norm_clause(cl: str) -> Optional[str]:
        if not cl:
            return None
        cl = cl.strip()
        if not cl.endswith('.'):
            cl += '.'
        atom_str = cl.rstrip('.').strip()
        p = parse_predicate(atom_str)
        if not (p and p[0] == "connected" and len(p[1]) == 2):
            return None
        u, v = p[1][0].strip(), p[1][1].strip()
        if u == v:
            return None
        return f"connected({u}, {v})."

    def shortcut_meta(u: str, v: str):
        d = _shortest_path_len_bounded(hard_adj, u, v, max_depth=30)
        is_shortcut = (d is not None and d >= 2)
        return is_shortcut, d
    if is_shortcut_edge(u, v):
    continue  # DO NOT add to hypotheses


    out = []
    for atom, proposals in by_atom.items():
        if not isinstance(proposals, list):
            continue
        for item in proposals[:max_hyp_per_atom]:
            if not isinstance(item, dict):
                continue
            clause = norm_clause(item.get("clause", ""))
            if clause is None:
                continue

            try:
                conf = float(item.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            conf = max(0.0, min(1.0, conf))

            p = parse_predicate(clause.rstrip('.'))
            u, v = p[1][0].strip(), p[1][1].strip()

            if (u, v) in hard_edges:
                continue

            is_shortcut, d = shortcut_meta(u, v)
            unknown_station = bool(stations) and (u not in stations or v not in stations)

            out.append({
                "clause": clause,
                "confidence": conf,
                "from_atom": atom,
                "is_shortcut": is_shortcut,
                "shortcut_len": d,
                "unknown_station": unknown_station,
            })

    # Dedup keep best confidence
    dedup = {}
    for h in out:
        key = h["clause"]
        if key not in dedup or h["confidence"] > dedup[key]["confidence"]:
            dedup[key] = h

    result = list(dedup.values())
    _BG_HYP_CACHE[cache_key] = result
    return result


# --- Hypothesis attachment ---

def _find_max_line_number_in_kb(kb: str) -> int:
    max_num = 0
    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        content_raw = m.group(2).strip()
        content, _ = split_inline_comment(content_raw)
        if not content:
            continue

        num = int(m.group(1))
        if num > max_num:
            max_num = num
    return max_num


def _is_fact_clause(clause: str) -> bool:
    return ':-' not in clause


def _split_rule_clause(clause: str):
    clause = clause.strip()
    if clause.endswith('.'):
        clause = clause[:-1]
    head_part, body_part = clause.split(':-', 1)
    head = head_part.strip()
    body_str = body_part.strip()
    return head, body_str


def attach_hypotheses_to_kb(kb: str, hypotheses):
    """
    Soft facts now include a penalty field:
      (num, atom, conf, penalty)

    penalty is used to deprioritize (but not ban) shortcuts / hallucinations.
    """
    soft_facts = []
    soft_rules = []

    max_num = _find_max_line_number_in_kb(kb)
    next_num = max_num + 1

    for h in hypotheses:
        clause = (h.get("clause") or "").strip()
        if not clause:
            continue

        conf = float(h.get("confidence", 0.0))
        if not clause.endswith('.'):
            clause = clause + '.'

        penalty = 0.0
        if h.get("is_shortcut"):
            penalty += 1.0
        if h.get("unknown_station"):
            penalty += 0.5

        if _is_fact_clause(clause):
            atom = clause.rstrip('.').strip()
            soft_facts.append((next_num, atom, conf, penalty))
        else:
            head, body_str = _split_rule_clause(clause)
            # keep compatibility if you ever allow soft rules
            soft_rules.append((next_num, head, body_str, conf, penalty))

        next_num += 1

    return {"facts": soft_facts, "rules": soft_rules}


# --- SOFT BFS with confidence-first priority + penalty ---

def bfs_prolog_metro_soft(
    goal: str,
    kb: str,
    soft_kb,
    max_depth: int = 10,
    max_soft: Optional[int] = None,
    max_solutions: int = 25,
):
    """
    Best-first SLD with objective:
        (soft_cost, -min_conf, penalty_sum)
    Depth is only a bound + final tiebreak.
    """

    # --- Parse hard KB exactly like bfs_prolog_metro ---
    hard_facts = []
    hard_rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        line = strip_inline_comment(line).strip()
        if not line:
            continue

        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            num = int(match.group(1))
            content_raw = match.group(2).strip()
            content, _ = split_inline_comment(content_raw)
            content = (content or "").strip()
            if not content:
                continue

            if ':-' in content:
                head, body = content.split(':-', 1)
                hard_rules.append((num, head.strip(), body.strip().rstrip('.')))
            else:
                hard_facts.append((num, content.rstrip('.')))

    # --- Soft KB unpack (with penalty) ---
    soft_facts = soft_kb.get("facts", [])  # (num, atom, conf, penalty)
    soft_rules = soft_kb.get("rules", [])  # (num, head, body_str, conf, penalty)
    soft_rules_for_match = [(num, head, body_str) for (num, head, body_str, conf, penalty) in soft_rules]

    print(f"\n[SOFT BFS - BEST-PROOF (conf-priority + penalty)] Goal: {goal}")
    print("-" * 40)

    # PQ item:
    #   (soft_cost, -min_conf, penalty_sum, depth, tie, current, remaining, path, min_conf, penalty_sum)
    pq = []
    tie = count()

    def push_state(current, remaining, path, depth, soft_cost, min_conf, penalty_sum):
        if depth > max_depth:
            return
        heapq.heappush(
            pq,
            (soft_cost, -min_conf, penalty_sum, depth, next(tie),
             current, remaining, path, min_conf, penalty_sum)
        )

    push_state(goal, [], [], 0, 0, 1.0, 0.0)

    # Dominance pruning on (soft_cost, -min_conf, penalty_sum), depth as tiebreak
    best_seen = {}  # (current, tuple(remaining)) -> (soft_cost, neg_min_conf, penalty_sum, depth)

    def dominated(current, remaining, soft_cost, min_conf, penalty_sum, depth):
        key = (current, tuple(remaining))
        new = (soft_cost, -min_conf, penalty_sum, depth)
        old = best_seen.get(key)
        if old is None:
            best_seen[key] = new
            return False

        if (old[0], old[1], old[2]) < (new[0], new[1], new[2]):
            return True

        if (old[0], old[1], old[2]) == (new[0], new[1], new[2]) and old[3] <= new[3]:
            return True

        best_seen[key] = new
        return False

    def make_success_result(final_path, final_soft_cost, final_min_conf, final_depth, final_penalty_sum):
        used_soft = []
        for step in final_path:
            if step.startswith("SoftFact"):
                parts = step.split()
                if len(parts) >= 2:
                    try:
                        num = int(parts[1])
                    except ValueError:
                        num = None
                    used_soft.append(("fact", num))
            elif step.startswith("SoftRule"):
                parts = step.split()
                if len(parts) >= 2:
                    try:
                        num = int(parts[1])
                    except ValueError:
                        num = None
                    used_soft.append(("rule", num))

        return {
            "success": True,
            "proof_path": final_path,
            "used_soft_clauses": used_soft,
            "soft_cost": final_soft_cost,
            "min_conf": final_min_conf if final_soft_cost > 0 else None,
            "penalty_sum": final_penalty_sum,
            "depth": final_depth,
        }

    solutions = []  # (soft_cost, neg_min_conf, penalty_sum, res)

    def maybe_add_solution(res, soft_cost, min_conf, penalty_sum):
        solutions.append((soft_cost, -min_conf, penalty_sum, res))
        solutions.sort(key=lambda x: (x[0], x[1], x[2]))
        if len(solutions) > max_solutions:
            solutions.pop()

    while pq:
        soft_cost, neg_min_conf, penalty_sum, depth, _, current, remaining, path, min_conf, penalty_sum = heapq.heappop(pq)

        print(f"Depth {depth}: {current}")
        print(f"  Priority key: (soft_cost={soft_cost}, -min_conf={neg_min_conf:.3f}, penalty={penalty_sum:.3f}, depth={depth})")
        if remaining:
            print(f"  Remaining: {remaining}")

        if dominated(current, remaining, soft_cost, min_conf, penalty_sum, depth):
            continue

        # --- 1) HARD facts: exact match ---
        for num, fact in hard_facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Hard Fact {num} matches exactly: {fact}")
                new_path = path + [f"HardFact {num}"]

                if not remaining:
                    res = make_success_result(new_path, soft_cost, min_conf, depth + 1, penalty_sum)
                    maybe_add_solution(res, soft_cost, min_conf, penalty_sum)
                    break

                next_goal = remaining[0]
                next_remaining = remaining[1:]
                push_state(next_goal, next_remaining, new_path, depth + 1, soft_cost, min_conf, penalty_sum)
                break

        # --- 2) HARD facts: unification ---
        for num, fact in hard_facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            print(f"  ✓ Hard Fact {num} unifies: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)
            new_path = path + [f"HardFact {num}"]

            if not instantiated:
                res = make_success_result(new_path, soft_cost, min_conf, depth + 1, penalty_sum)
                maybe_add_solution(res, soft_cost, min_conf, penalty_sum)
                continue

            push_state(instantiated[0], instantiated[1:], new_path, depth + 1, soft_cost, min_conf, penalty_sum)

        # --- 3) HARD rules ---
        matching_hard_rules = find_matching_rules_only(current, hard_rules)
        if matching_hard_rules:
            print(f"  Matching hard rules: {matching_hard_rules}")

        for rule_num in matching_hard_rules:
            for num, head, body in hard_rules:
                if num != rule_num:
                    continue

                subgoals = get_subgoals(current, head, body)
                if not subgoals:
                    continue

                print(f"  Hard Rule {num}: {head} :- {body}")
                print(f"    → {subgoals}")

                all_goals = subgoals + remaining
                push_state(
                    all_goals[0],
                    all_goals[1:],
                    path + [f"HardRule {num}"],
                    depth + 1,
                    soft_cost,
                    min_conf,
                    penalty_sum
                )
                break

        # --- 4) SOFT facts ---
        for s_num, s_atom, s_conf, s_penalty in soft_facts:
            if max_soft is not None and soft_cost >= max_soft:
                break

            bindings = unify_with_fact(current, s_atom)
            if bindings is None:
                continue

            new_soft_cost = soft_cost + 1
            new_min_conf = min(min_conf, s_conf)
            new_penalty_sum = penalty_sum + float(s_penalty)

            print(f"  ✓ Soft Fact {s_num} unifies: {s_atom}")
            print(f"    Bindings: {bindings}, conf={s_conf:.3f}, penalty={float(s_penalty):.3f}")
            print(f"    New soft cost: {new_soft_cost}, new min_conf: {new_min_conf:.3f}, new penalty: {new_penalty_sum:.3f}")

            instantiated = apply_bindings(remaining, bindings)
            new_path = path + [f"SoftFact {s_num} (conf={s_conf:.3f},pen={float(s_penalty):.3f})"]

            if not instantiated:
                res = make_success_result(new_path, new_soft_cost, new_min_conf, depth + 1, new_penalty_sum)
                maybe_add_solution(res, new_soft_cost, new_min_conf, new_penalty_sum)
                continue

            push_state(instantiated[0], instantiated[1:], new_path, depth + 1, new_soft_cost, new_min_conf, new_penalty_sum)

        # --- 5) SOFT rules (optional) ---
        matching_soft_rules = find_matching_rules_only(current, soft_rules_for_match)
        if matching_soft_rules:
            print(f"  Matching soft rules: {matching_soft_rules}")

        for rule_num in matching_soft_rules:
            if max_soft is not None and soft_cost >= max_soft:
                break

            for s_num, s_head, s_body_str, s_conf, s_penalty in soft_rules:
                if s_num != rule_num:
                    continue

                subgoals = get_subgoals(current, s_head, s_body_str)
                if not subgoals:
                    continue

                new_soft_cost = soft_cost + 1
                new_min_conf = min(min_conf, s_conf)
                new_penalty_sum = penalty_sum + float(s_penalty)

                print(f"  Soft Rule {s_num}: {s_head} :- {s_body_str}")
                print(f"    → {subgoals}, conf={s_conf:.3f}, penalty={float(s_penalty):.3f}")
                print(f"    New soft cost: {new_soft_cost}, new min_conf: {new_min_conf:.3f}, new penalty: {new_penalty_sum:.3f}")

                all_goals = subgoals + remaining
                new_path = path + [f"SoftRule {s_num} (conf={s_conf:.3f},pen={float(s_penalty):.3f})"]

                push_state(all_goals[0], all_goals[1:], new_path, depth + 1, new_soft_cost, new_min_conf, new_penalty_sum)
                break

    if solutions:
        return solutions[0][3]

    print("✗ PRIORITY SOFT-BFS FAILED (no proof found even with soft KB)")
    return {
        "success": False,
        "proof_path": [],
        "used_soft_clauses": [],
        "soft_cost": None,
        "min_conf": None,
        "penalty_sum": None,
        "depth": None,
    }


# --- Orchestration ---

def solve_with_background(
    goal: str,
    kb: str,
    max_depth: int = 10,
    max_soft=None,
    hard_result=None,
):
    """
    High-level pipeline.
    IMPORTANT: `kb` must be the exact KB used for hard BFS and for hypothesis generation.
    """
    predicate_comments = parse_kb_predicate_comments(kb)

    print("\n========================================")
    print(f"SOLVE WITH BACKGROUND: {goal}")
    print("========================================\n")

    if hard_result is None:
        print(">>> Phase 1: Hard-KB BFS (bfs_prolog_collect)")
        hard_result = bfs_prolog_collect(goal, kb, max_depth=max_depth)
        print("Hard-KB result:", hard_result)
    else:
        print(">>> Phase 1: Hard-KB BFS result already computed, reusing it.")
        print("Hard-KB result:", hard_result)

    if hard_result.get("success"):
        print("\n>>> Result: HARD_SUCCESS (no background hypotheses needed)\n")
        return {
            "status": "HARD_SUCCESS",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": []
        }

    unresolved_atoms = hard_result.get("unresolved_atoms", set())
    if not unresolved_atoms:
        print("\nNo unresolved atoms to explain; cannot generate hypotheses.")
        print(">>> Result: FAILURE\n")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": []
        }

    print("\n>>> Phase 2: Generate background hypotheses (FAST)")
    print("Unresolved atoms:", unresolved_atoms)

    hypotheses = generate_background_hypotheses_fast(
        goal=goal,
        kb=kb,                        # same KB everywhere
        hard_result=hard_result,       # reuse unresolved atoms
        predicate_comments=predicate_comments,
        max_atoms=5,
        max_hyp_per_atom=2,
        prompt_fact_limit=60,
    )

    if hypotheses is None:
        hypotheses = []

    if not hypotheses:
        print("Hypotheses returned by LLM: []")
        print("\nLLM returned NO hypotheses; cannot build soft KB.")
        print(">>> Result: FAILURE\n")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": []
        }

    print("Hypotheses returned by LLM:")
    for h in hypotheses:
        print("  - Clause:", h.get("clause"),
              "| Conf:", h.get("confidence"),
              "| From atom:", h.get("from_atom"),
              "| Shortcut:", h.get("is_shortcut"),
              "| UnknownStation:", h.get("unknown_station"))

    print("\n>>> Phase 3: Attach hypotheses to soft KB")
    soft_kb = attach_hypotheses_to_kb(kb, hypotheses)
    print("Soft KB facts:", soft_kb.get("facts", []))
    print("Soft KB rules:", soft_kb.get("rules", []))

    print("\n>>> Phase 4: Soft BFS (bfs_prolog_metro_soft)")
    soft_result = bfs_prolog_metro_soft(
        goal=goal,
        kb=kb,
        soft_kb=soft_kb,
        max_depth=max_depth,
        max_soft=max_soft,
    )
    print("Soft-BFS result:", soft_result)

    if soft_result.get("success"):
        print("\n>>> Result: SOFT_SUCCESS (proof found using background hypotheses)\n")
        return {
            "status": "SOFT_SUCCESS",
            "hard_result": hard_result,
            "soft_result": soft_result,
            "hypotheses": hypotheses
        }

    print("\n>>> Result: SOFT_FAILURE (no proof even with background hypotheses)\n")
    return {
        "status": "SOFT_FAILURE",
        "hard_result": hard_result,
        "soft_result": soft_result,
        "hypotheses": hypotheses
    }


def omit_facts_from_kb(kb: str, omit_numbers):
    """
    Return a new KB string with numbered lines in `omit_numbers` removed.
    Preserves original line text (including comments), but matches by number.
    """
    omit_numbers = set(omit_numbers)
    new_lines = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        num = int(m.group(1))
        if num in omit_numbers:
            continue

        new_lines.append(line)

    return "\n".join(new_lines)


# Natural language to Prolog using LLM, algo SLD resolution

def nl_kb_to_prolog_kb(nl_kb_text: str, start_index: int = 1) -> list[str]:
    """
    Convert a *pure natural-language* description of a domain + rules
    into a numbered Prolog knowledge base.
    """
    nl_kb_text = (nl_kb_text or "").strip()
    if not nl_kb_text:
        return []

    prompt = f"""
You are a Prolog formalization assistant.

The user will give you a natural-language description of a small domain,
including objects, relationships, and logical rules.

Your job is to convert that description into a set of Prolog clauses
(facts and rules).

Guidelines:
- Use lowercase atoms for concrete entities (e.g. union_square, times_square,
  grand_central, bryant_park).
- Use uppercase identifiers for variables (e.g. X, Y, Z).
- Choose predicate names that are short, descriptive, and consistent,
  for example: connected/2, reachable/2, located_in/2, etc.
- A fact must look like:
    connected(times_square, bryant_park).
- A rule must look like:
    reachable(X, Z) :- connected(X, Y), reachable(Y, Z).
- Every clause MUST end with a single period '.'.
- Do NOT include any line numbers in your output clauses.
- Do NOT add explanations or comments in the Prolog code.

Here is the NATURAL LANGUAGE description of the knowledge base:

\"\"\"{nl_kb_text}\"\"\"

Respond ONLY in this JSON format (and nothing else):

{{
  "clauses": [
    {{
      "clause": "connected(union_square, times_square)."
    }},
    {{
      "clause": "reachable(X, Y) :- connected(X, Y)."
    }}
  ]
}}
"""
    raw = ask_llm(prompt).strip()
    try:
        data = json.loads(extract_first_json(raw))
    except Exception as e:
        print("[nl_kb_to_prolog_kb] JSON parse error:", e)
        print("Raw LLM output:", raw)
        return []

    raw_clauses = data.get("clauses", [])
    if not isinstance(raw_clauses, list):
        print("[nl_kb_to_prolog_kb] 'clauses' field is not a list:", raw_clauses)
        return []

    cleaned_clauses = []

    for item in raw_clauses:
        if isinstance(item, str):
            clause = item.strip()
        elif isinstance(item, dict):
            clause = (item.get("clause") or "").strip()
        else:
            continue

        if not clause:
            continue

        m_num = re.match(r'^\s*(\d+)\.\s*(.+)$', clause)
        if m_num:
            clause = m_num.group(2).strip()

        clause = clause.rstrip()
        if not clause.endswith('.'):
            clause = clause + "."
        else:
            clause = re.sub(r'\.+$', '.', clause)

        body_str = clause[:-1].strip()
        if ':-' in body_str:
            head_part, body_part = body_str.split(':-', 1)
            head = head_part.strip()
        else:
            head = body_str

        parsed_head = parse_predicate(head)
        if parsed_head is None:
            print("[nl_kb_to_prolog_kb] Discarding unparsable clause:", clause)
            continue

        cleaned_clauses.append(clause)

    numbered_clauses = []
    next_num = start_index
    for clause in cleaned_clauses:
        numbered_clauses.append(f"{next_num}. {clause}")
        next_num += 1

    return numbered_clauses


# In[72]:


if __name__ == "__main__":
    kb = """
    1. connected(union_square, 14th_street). # connected/2 = ADJACENT STOPS ONLY. Use only when two stations are immediate neighbors on the same line. Do NOT add shortcut edges that skip intermediate stations.
    2. connected(14th_street, 23rd_street).
    3. connected(23rd_street, 34th_street).
    4. connected(34th_street, times_square).
    5. connected(times_square, 42nd_street).
    6. connected(42nd_street, grand_central).
    7. connected(grand_central, bryant_park).
    8. reachable(X, Y) :- connected(X, Y). # reachable/2 = PATH EXISTS. One-hop reachable comes only from connected/2.
    9. reachable(X, Z) :- connected(X, Y), reachable(Y, Z). # reachable/2 is the transitive closure of connected/2 (multi-hop path following only adjacent edges).

    """

    print("===== FULL METRO KB =====")
    print(kb)
    print("====================================\n")

    # Omit fact 1 to force background reasoning (removes union_square -> times_square)
    kb_missing_fact = omit_facts_from_kb(kb, omit_numbers={7})

    print("===== METRO KB WITH FACT REMOVED =====")
    print(kb_missing_fact)
    print("========================================\n")

    test_goal = "reachable(union_square, bryant_park)"

    print("==============================")
    print(f"TEST QUERY: {test_goal}")
    print("==============================\n")

    print(">>> Running bfs_prolog_collect (hard-KB BFS)...")
    collect_result = bfs_prolog_collect(
    goal=test_goal,
    kb=kb_missing_fact,          
    max_depth=20    
)
    print("Collect Result:", collect_result)
    print("\n----------------------------------------\n")

    print(">>> Running solve_with_background (full pipeline, reusing hard result)...")
    bg_result = solve_with_background(
        goal=test_goal,
        kb=kb_missing_fact,
        max_depth=20,
        max_soft=None,
        hard_result=collect_result,
    )
    print("Solve-with-background Result:")
    print(bg_result)
    print("\n========================================\n")


# In[6]:


import ollama
from collections import deque
import re
import json
from typing import Optional, Dict, Any, List, Tuple

# ============================================================
# Config / LLM setup
# ============================================================

client = ollama.Client()
model = "gpt-oss:20b"
# model = "qwen:14b"

DEBUG = False  # set to True to print raw LLM outputs for debugging


def ask_llm(prompt: str) -> str:
    resp = client.generate(model=model, prompt=prompt, options={'temperature': 0.0})
    answer = resp.get('response', '')
    if "...done thinking." in answer:
        return answer.split("...done thinking.")[-1].strip()
    return answer


# ============================================================
# Helpers for parsing / JSON
# ============================================================

def extract_first_json(text: str) -> str:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in: {text!r}")
    return match.group(0)


def split_inline_comment(s: str):
    if "#" not in s:
        return s.strip(), None
    code, comment = s.split("#", 1)
    code = code.strip()
    comment = comment.strip()
    return code, (comment if comment else None)


def parse_predicate(term: str):
    term = term.strip().rstrip('.')
    m = re.match(r'^([a-z_][a-zA-Z0-9_]*)\((.*)\)$', term)
    if not m:
        return None
    functor = m.group(1)
    args_raw = m.group(2).strip()
    if not args_raw:
        args = []
    else:
        args = [a.strip() for a in args_raw.split(',')]
    return functor, args


def is_variable(s: str) -> bool:
    s = s.strip()
    return bool(s) and (s[0].isupper() or s[0] == '_')


def strip_inline_comment(s: str) -> str:
    return s.split('#', 1)[0].rstrip()


# ============================================================
# Core Prolog helpers (unification + SLD one-step)
# ============================================================

def check_exact_match(goal: str, fact: str) -> bool:
    return goal.strip().rstrip('.') == fact.strip().rstrip('.')


def unify_args(args_goal, args_fact, env=None):
    if env is None:
        env = {}

    if len(args_goal) != len(args_fact):
        return None

    for g, f in zip(args_goal, args_fact):
        g = g.strip()
        f = f.strip()

        g_is_var = is_variable(g)
        f_is_var = is_variable(f)

        if not g_is_var and not f_is_var:
            if g != f:
                return None
            continue

        if g_is_var and not f_is_var:
            if g in env:
                if env[g] != f:
                    return None
            else:
                env[g] = f
            continue

        if not g_is_var and f_is_var:
            if f in env:
                if env[f] != g:
                    return None
            else:
                env[f] = g
            continue

        if g_is_var and f_is_var:
            if g in env and f in env:
                if env[g] != env[f]:
                    return None
            elif g in env:
                env[f] = env[g]
            elif f in env:
                env[g] = env[f]
            continue

    return env


def unify_arg_lists(args_rule_head, args_goal):
    return unify_args(args_rule_head, args_goal, env={})


def unify_with_fact(goal: str, fact: str):
    parsed_goal = parse_predicate(goal)
    parsed_fact = parse_predicate(fact)
    if parsed_goal is None or parsed_fact is None:
        return None

    fun_g, args_g = parsed_goal
    fun_f, args_f = parsed_fact

    if fun_g != fun_f or len(args_g) != len(args_f):
        return None

    if check_exact_match(goal, fact):
        return {}

    env = unify_args(args_g, args_f, env={})
    if env is None:
        return None
    return env


def apply_bindings(goals, bindings):
    if not bindings or not goals:
        return goals

    new_goals = []
    for g in goals:
        parsed = parse_predicate(g)
        if parsed is None:
            new_goals.append(g)
            continue

        functor, args = parsed
        new_args = []
        for a in args:
            a_stripped = a.strip()
            if is_variable(a_stripped) and a_stripped in bindings:
                new_args.append(bindings[a_stripped])
            else:
                new_args.append(a_stripped)

        new_goal = f"{functor}({', '.join(new_args)})"
        new_goals.append(new_goal)

    return new_goals


def find_matching_rules_only(goal, rules_list):
    parsed_goal = parse_predicate(goal)
    if parsed_goal is None:
        return []
    fun_g, args_g = parsed_goal
    arity_g = len(args_g)

    matching = []
    for num, head, body in rules_list:
        parsed_head = parse_predicate(head)
        if parsed_head is None:
            continue
        fun_h, args_h = parsed_head
        if fun_h == fun_g and len(args_h) == arity_g:
            matching.append(num)
    return matching


def substitute_in_atom(atom: str, bindings: dict) -> str:
    parsed = parse_predicate(atom)
    if parsed is None:
        return atom

    functor, args = parsed
    new_args = []
    for a in args:
        a_stripped = a.strip()
        if is_variable(a_stripped) and a_stripped in bindings:
            new_args.append(bindings[a_stripped])
        else:
            new_args.append(a_stripped)

    return f"{functor}({', '.join(new_args)})"


def split_body_atoms(body_str: str):
    body_str = body_str.strip()
    atoms = []
    current = []
    depth = 0

    for ch in body_str:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth = max(depth - 1, 0)
            current.append(ch)
        elif ch == ',' and depth == 0:
            atom = ''.join(current).strip()
            if atom:
                atoms.append(atom)
            current = []
        else:
            current.append(ch)

    atom = ''.join(current).strip()
    if atom:
        atoms.append(atom)

    return atoms


def get_subgoals(goal: str, rule_head: str, rule_body: str):
    parsed_goal = parse_predicate(goal)
    parsed_head = parse_predicate(rule_head)

    if parsed_goal is None or parsed_head is None:
        return None

    fun_g, args_g = parsed_goal
    fun_h, args_h = parsed_head

    if fun_g != fun_h or len(args_g) != len(args_h):
        return None

    bindings = unify_arg_lists(args_h, args_g)
    if bindings is None:
        return None

    body_str = rule_body.strip()
    if not body_str:
        return []

    body_atoms = split_body_atoms(body_str)
    if not body_atoms:
        return []

    subgoals = [substitute_in_atom(atom, bindings) for atom in body_atoms]
    return subgoals if subgoals else None


# ============================================================
# KB comment extraction (inline + full-line)
# ============================================================

def parse_kb_predicate_comments(kb: str):
    predicate_comments = {}
    pending_comments = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("#"):
            pending_comments.append(line.lstrip("#").strip())
            continue

        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        content_raw = m.group(2).strip()
        content, inline_comment = split_inline_comment(content_raw)

        clause = content.strip().rstrip(".")
        head_str = clause.split(":-", 1)[0].strip()
        parsed = parse_predicate(head_str)
        if parsed is None:
            pending_comments = []
            continue

        functor, args = parsed
        key = f"{functor}/{len(args)}"

        combined = []
        if pending_comments:
            combined.append(" ".join(pending_comments))
        if inline_comment:
            combined.append(inline_comment)

        if combined:
            predicate_comments[key] = " ".join(combined).strip()

        pending_comments = []

    return predicate_comments


# ============================================================
# KB parsing (facts + rules)
# ============================================================

def parse_kb_facts_rules(kb: str):
    facts = []
    rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        line = strip_inline_comment(line).strip()
        if not line:
            continue

        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not match:
            continue

        num = int(match.group(1))
        content_raw = match.group(2).strip()
        content, _ = split_inline_comment(content_raw)
        content = (content or "").strip()
        if not content:
            continue

        if ':-' in content:
            head, body = content.split(':-', 1)
            rules.append((num, head.strip(), body.strip().rstrip('.')))
        else:
            facts.append((num, content.rstrip('.')))

    return facts, rules


# ============================================================
# BFS Prolog engine (collect + full resume support)
# ============================================================

def bfs_prolog_collect(goal: str, kb: str, max_depth: int = 10):
    """
    BFS SLD (facts + rules). Returns:
      - success
      - proof_path (if success)
      - unresolved_atoms (set)
      - failed_state (a representative failed state)
      - failed_states (map: atom -> state info)
    """
    facts, rules = parse_kb_facts_rules(kb)

    queue = deque([(goal, [], [], 0)])
    visited = set()
    unresolved_atoms = set()

    failed_states = {}  # current_atom -> {current, remaining, path, depth}

    print(f"\n[COLLECT] Goal: {goal}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        if depth >= max_depth:
            unresolved_atoms.add(current)
            if current not in failed_states or depth > failed_states[current]["depth"]:
                failed_states[current] = {"current": current, "remaining": remaining, "path": path, "depth": depth}
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")

        progress = False

        # 1) Exact fact match
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")

                step = f"Fact {num}"
                new_path = path + [step]

                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return {
                        "success": True,
                        "proof_path": new_path,
                        "unresolved_atoms": set(),
                        "failed_state": None,
                        "failed_states": {}
                    }

                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, new_path, depth + 1))

                progress = True
                break

        if progress:
            continue

        # 2) Fact unification
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            progress = True
            print(f"  ✓ Fact {num}: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)
            step = f"Fact {num}"
            new_path = path + [step]

            if not instantiated:
                print(f"✓✓ SUCCESS at depth {depth + 1}")
                return {
                    "success": True,
                    "proof_path": new_path,
                    "unresolved_atoms": set(),
                    "failed_state": None,
                    "failed_states": {}
                }

            queue.append((instantiated[0], instantiated[1:], new_path, depth + 1))

        if progress:
            continue

        # 3) Rules
        matching_rules = find_matching_rules_only(current, rules)
        if matching_rules:
            print(f"  Matching rules: {matching_rules}")

        for rule_num in matching_rules:
            for num, head, body in rules:
                if num != rule_num:
                    continue

                subgoals = get_subgoals(current, head, body)
                if subgoals:
                    print(f"  Rule {num}: → {subgoals}")
                    progress = True
                    all_goals = subgoals + remaining
                    step = f"Rule {num}"
                    new_path = path + [step]
                    queue.append((all_goals[0], all_goals[1:], new_path, depth + 1))
                break

        if not progress:
            print(f"  ✗ No facts or rules apply to: {current}")
            unresolved_atoms.add(current)
            if current not in failed_states or depth > failed_states[current]["depth"]:
                failed_states[current] = {"current": current, "remaining": remaining, "path": path, "depth": depth}

    print("✗ FAILED (collect mode)")

    # Representative failed_state: choose deepest failed atom (more likely “closest” to completion)
    rep = None
    if failed_states:
        rep = max(failed_states.values(), key=lambda s: s["depth"])

    return {
        "success": False,
        "proof_path": [],
        "unresolved_atoms": unresolved_atoms,
        "failed_state": rep,
        "failed_states": failed_states
    }


def bfs_prolog_resume_from_state(
    kb: str,
    start_current: str,
    start_remaining: List[str],
    start_path_prefix: List[str],
    start_depth: int,
    max_depth: int = 10
):
    """
    Resume BFS SLD from an intermediate state and return a FULL proof path
    (prefix + continuation) that corresponds to proving the ROOT goal.

    IMPORTANT:
    - start_path_prefix is the path that led to (start_current, start_remaining).
    - On success, returns full proof steps including prefix.
    """
    facts, rules = parse_kb_facts_rules(kb)

    queue = deque([(start_current, list(start_remaining), list(start_path_prefix), start_depth)])
    visited = set()

    print(f"\n[RESUME BFS] Starting from failed node:")
    print(f"  Depth {start_depth}: {start_current}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        if depth >= max_depth:
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")

        # 1) Exact fact match
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")
                new_path = path + [f"Fact {num}"]

                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return {"success": True, "proof_path": new_path, "final_depth": depth + 1}

                queue.append((remaining[0], remaining[1:], new_path, depth + 1))
                break  # exact match: single transition

        # 2) Fact unification
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            print(f"  ✓ Fact {num}: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)
            new_path = path + [f"Fact {num}"]

            if not instantiated:
                print(f"✓✓ SUCCESS at depth {depth + 1}")
                return {"success": True, "proof_path": new_path, "final_depth": depth + 1}

            queue.append((instantiated[0], instantiated[1:], new_path, depth + 1))

        # 3) Rules
        matching_rules = find_matching_rules_only(current, rules)
        if matching_rules:
            print(f"  Matching rules: {matching_rules}")

        for rule_num in matching_rules:
            for num, head, body in rules:
                if num != rule_num:
                    continue
                subgoals = get_subgoals(current, head, body)
                if not subgoals:
                    break
                print(f"  Rule {num}: → {subgoals}")
                all_goals = subgoals + remaining
                new_path = path + [f"Rule {num}"]
                queue.append((all_goals[0], all_goals[1:], new_path, depth + 1))
                break

    print("✗ RESUME FAILED")
    return {"success": False, "proof_path": [], "final_depth": None}


# ============================================================
# Background hypothesis generation (FAST) + shortcut filtering
# ============================================================

_BG_HYP_CACHE = {}  # (kb_sig, goal, tuple(atoms), max_hyp_per_atom) -> list[hypothesis]


def _kb_signature_for_bg(kb: str) -> str:
    parts = []
    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        content_raw = m.group(2).strip()
        content, _ = split_inline_comment(content_raw)
        content = (content or "").strip()
        if not content:
            continue

        content = content.rstrip('.').strip()
        if content.startswith("connected(") or content.startswith("reachable("):
            parts.append(content)

    return str(hash("\n".join(parts)))


def _extract_connected_facts_and_stations(kb: str):
    hard_adj = {}
    hard_edges = set()
    stations = set()
    connected_facts = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        content_raw = m.group(2).strip()
        content, _ = split_inline_comment(content_raw)
        content = (content or "").strip()
        if not content:
            continue

        if ':-' in content:
            continue

        atom0 = content.rstrip('.').strip()
        p = parse_predicate(atom0)
        if p and p[0] == "connected" and len(p[1]) == 2:
            a, b = p[1][0].strip(), p[1][1].strip()
            hard_adj.setdefault(a, []).append(b)
            hard_edges.add((a, b))
            stations.add(a)
            stations.add(b)
            connected_facts.append(f"connected({a}, {b}).")

    return hard_adj, hard_edges, stations, connected_facts


def _shortest_path_len_bounded(adj: dict, src: str, dst: str, max_depth: int = 25):
    if src == dst:
        return 0
    q = deque([(src, 0)])
    seen = {src}
    while q:
        node, d = q.popleft()
        if d >= max_depth:
            continue
        for nb in adj.get(node, []):
            if nb == dst:
                return d + 1
            if nb not in seen:
                seen.add(nb)
                q.append((nb, d + 1))
    return None


def _infer_preferred_atom_for_chain(goal: str, hard_adj: dict) -> Optional[str]:
    gp = parse_predicate(goal)
    if not gp or len(gp[1]) != 2:
        return None
    goal_src, goal_dst = gp[1][0].strip(), gp[1][1].strip()

    last = goal_src
    seen = set()
    while last in hard_adj and len(hard_adj[last]) == 1 and last not in seen:
        seen.add(last)
        last = hard_adj[last][0]

    if last and goal_dst:
        return f"connected({last}, {goal_dst})"
    return None


def _select_atoms_for_bg(unresolved_atoms, preferred_atom: Optional[str], max_atoms: int):
    atom_list = list(unresolved_atoms)
    ground = []
    for atom in atom_list:
        atom = atom.strip()
        if not atom or '(' not in atom or ')' not in atom:
            continue
        inside = atom.split('(', 1)[1].rsplit(')', 1)[0]
        if re.search(r'\b[A-Z_]\w*\b', inside):
            continue
        ground.append(atom)

    if preferred_atom:
        ground = [preferred_atom] + [a for a in ground if a != preferred_atom]

    if max_atoms is not None:
        ground = ground[:max_atoms]
    return ground


def generate_background_hypotheses_fast(
    goal: str,
    kb: str,
    hard_result: dict,
    predicate_comments: dict,
    max_atoms: int = 6,
    max_hyp_per_atom: int = 2,
    prompt_fact_limit: int = 60,
):
    unresolved_atoms = hard_result.get("unresolved_atoms", set()) if hard_result else set()
    if not unresolved_atoms:
        return []

    hard_adj, hard_edges, stations, connected_facts = _extract_connected_facts_and_stations(kb)
    preferred_atom = _infer_preferred_atom_for_chain(goal, hard_adj)
    atoms = _select_atoms_for_bg(unresolved_atoms, preferred_atom, max_atoms=max_atoms)
    if not atoms:
        print("[generate_background_hypotheses_fast] No suitable ground atoms.")
        return []

    kb_sig = _kb_signature_for_bg(kb)
    cache_key = (kb_sig, goal, tuple(atoms), max_hyp_per_atom)
    if cache_key in _BG_HYP_CACHE:
        return _BG_HYP_CACHE[cache_key]

    facts_for_prompt = connected_facts[:prompt_fact_limit]

    needed = {"connected/2"}
    for a in atoms:
        p = parse_predicate(a)
        if p:
            needed.add(f"{p[0]}/{len(p[1])}")

    semantic_lines = []
    for k in sorted(needed):
        if k in predicate_comments:
            semantic_lines.append(f"- {k}: {predicate_comments[k]}")
    semantic_hint_block = "\n".join(semantic_lines) if semantic_lines else "(none)"

    atoms_block = "\n".join([f"- {a}" for a in atoms])

    prompt = f"""
You are a cautious Prolog expert. You are proposing missing FACTS only.

GOAL:
  {goal}

HARD KB SNAPSHOT (partial; do NOT assume any other edges exist):
Connected facts (sample):
{chr(10).join(facts_for_prompt)}

Predicate semantics (ground truth):
{semantic_hint_block}

The proof failed because these subgoals could not be proven:
{atoms_block}

Task:
For each failed subgoal above, propose up to {max_hyp_per_atom} additional Prolog FACTS
that are likely true and would help prove the GOAL.

Constraints:
- Output ONLY connected/2 FACTS. No rules.
- Each fact MUST end with a period.
- Prefer facts that connect existing stations seen in the snapshot.
- If a suggested edge is uncertain or could be a "shortcut", lower confidence.

Return ONLY valid JSON:

{{
  "by_atom": {{
    "connected(42nd_street, bryant_park)": [
      {{"clause":"connected(grand_central, bryant_park).","confidence":0.95}},
      {{"clause":"connected(42nd_street, bryant_park).","confidence":0.55}}
    ]
  }}
}}
""".strip()

    print("[LLM] about to call ask_llm; prompt chars =", len(prompt))
    raw = ask_llm(prompt).strip()
    print("[LLM] returned from ask_llm; response chars =", len(raw))
    
    if DEBUG:
        print("\n[DEBUG generate_background_hypotheses_fast] raw:\n", raw)

    try:
        data = json.loads(extract_first_json(raw))
    except Exception as e:
        print("[generate_background_hypotheses_fast] JSON parse error:", e)
        print("Raw LLM output:", raw)
        _BG_HYP_CACHE[cache_key] = []
        return []

    by_atom = data.get("by_atom", {})
    if not isinstance(by_atom, dict):
        _BG_HYP_CACHE[cache_key] = []
        return []

    def norm_clause(cl: str) -> Optional[str]:
        if not cl:
            return None
        cl = cl.strip()
        if not cl.endswith('.'):
            cl += '.'
        atom_str = cl.rstrip('.').strip()
        p = parse_predicate(atom_str)
        if not (p and p[0] == "connected" and len(p[1]) == 2):
            return None
        u, v = p[1][0].strip(), p[1][1].strip()
        if u == v:
            return None
        return f"connected({u}, {v})."

    def is_shortcut_edge(u: str, v: str) -> Tuple[bool, Optional[int]]:
        d = _shortest_path_len_bounded(hard_adj, u, v, max_depth=30)
        # If there is already a path of length >=2, then u->v is a shortcut edge
        return (d is not None and d >= 2), d

    out = []
    for atom, proposals in by_atom.items():
        if not isinstance(proposals, list):
            continue
        for item in proposals[:max_hyp_per_atom]:
            if not isinstance(item, dict):
                continue

            clause = norm_clause(item.get("clause", ""))
            if clause is None:
                continue

            try:
                conf = float(item.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            conf = max(0.0, min(1.0, conf))

            p = parse_predicate(clause.rstrip('.'))
            u, v = p[1][0].strip(), p[1][1].strip()

            # already in hard KB
            if (u, v) in hard_edges:
                continue

            # HARD BAN shortcut edges (your requirement)
            shortcut, d = is_shortcut_edge(u, v)
            if shortcut:
                continue  # DO NOT add to hypotheses

            unknown_station = bool(stations) and (u not in stations or v not in stations)

            out.append({
                "clause": clause,
                "confidence": conf,
                "from_atom": atom,
                "is_shortcut": shortcut,
                "shortcut_len": d,
                "unknown_station": unknown_station,
            })

    # Dedup keep best confidence
    dedup = {}
    for h in out:
        key = h["clause"]
        if key not in dedup or h["confidence"] > dedup[key]["confidence"]:
            dedup[key] = h

    result = list(dedup.values())
    _BG_HYP_CACHE[cache_key] = result
    return result


# ============================================================
# Inject fact into KB (preserve numbering)
# ============================================================

def _find_max_line_number_in_kb(kb: str) -> int:
    max_num = 0
    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue
        num = int(m.group(1))
        if num > max_num:
            max_num = num
    return max_num


def inject_fact_into_kb(kb: str, fact_clause: str) -> str:
    """
    fact_clause must be like: connected(a,b). OR connected(a,b)
    This function appends a NEW numbered fact line.
    """
    fact_clause = (fact_clause or "").strip()
    if not fact_clause:
        return kb
    if not fact_clause.endswith('.'):
        fact_clause += '.'

    max_num = _find_max_line_number_in_kb(kb)
    new_num = max_num + 1

    kb_lines = [ln.rstrip() for ln in kb.strip().split("\n") if ln.strip()]
    kb_lines.append(f"{new_num}. {fact_clause}")
    return "\n".join(kb_lines)


# ============================================================
# Orchestration: probabilistic logic = try hypotheses by confidence
# ============================================================

def _best_resume_state_for_hypothesis(h: dict, hard_result: dict):
    """
    Prefer resuming from the failed state that matches the hypothesis's from_atom,
    otherwise fallback to deepest failure (representative).
    """
    failed_states = hard_result.get("failed_states", {}) or {}
    from_atom = (h.get("from_atom") or "").strip()

    if from_atom and from_atom in failed_states:
        return failed_states[from_atom]

    # If hypothesis is a clause connected(u,v), also try matching on current=connected(u,v)
    clause = (h.get("clause") or "").strip().rstrip(".")
    if clause and clause in failed_states:
        return failed_states[clause]

    rep = hard_result.get("failed_state")
    if rep:
        return rep

    # last resort: choose deepest
    if failed_states:
        return max(failed_states.values(), key=lambda s: s.get("depth", -1))

    return None


def solve_with_background(
    goal: str,
    kb: str,
    max_depth: int = 10,
    hard_result=None,
    max_atoms: int = 6,
    max_hyp_per_atom: int = 2,
):
    """
    Pipeline:
      1) Hard BFS collect failures
      2) LLM hypotheses
      3) Sort hypotheses by confidence desc
      4) For each hypothesis:
           - inject into KB
           - resume BFS from best matching failed state
           - if success: return full proof path (root proven)
    """
    predicate_comments = parse_kb_predicate_comments(kb)

    print("\n========================================")
    print(f"SOLVE WITH BACKGROUND (TOP-K FACT INJECTION w/ RESUME): {goal}")
    print("========================================\n")

    if hard_result is None:
        print(">>> Phase 1: Hard-KB BFS (bfs_prolog_collect)")
        hard_result = bfs_prolog_collect(goal, kb, max_depth=max_depth)
        print("Hard-KB result:", hard_result)
    else:
        print(">>> Phase 1: Hard-KB BFS result already computed, reusing it.")
        print("Hard-KB result:", hard_result)

    if hard_result.get("success"):
        print("\n>>> Result: HARD_SUCCESS (no background hypotheses needed)\n")
        return {
            "status": "HARD_SUCCESS",
            "hard_result": hard_result,
            "hypotheses": [],
            "injected_fact": None,
            "final_proof_path": hard_result.get("proof_path", []),
            "kb_with_injection": kb,
        }

    unresolved_atoms = hard_result.get("unresolved_atoms", set())
    if not unresolved_atoms:
        print("\nNo unresolved atoms to explain; cannot generate hypotheses.")
        print(">>> Result: FAILURE\n")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "hypotheses": [],
            "injected_fact": None,
            "final_proof_path": [],
            "kb_with_injection": kb,
        }

    print("\n>>> Phase 2: Generate background hypotheses (FAST)")
    print("Unresolved atoms:", unresolved_atoms)

    hypotheses = generate_background_hypotheses_fast(
        goal=goal,
        kb=kb,
        hard_result=hard_result,
        predicate_comments=predicate_comments,
        max_atoms=max_atoms,
        max_hyp_per_atom=max_hyp_per_atom,
        prompt_fact_limit=60,
    ) or []

    print("Hypotheses returned by LLM:")
    for h in hypotheses:
        print("  - Clause:", h.get("clause"),
              "| Conf:", h.get("confidence"),
              "| From atom:", h.get("from_atom"),
              "| Shortcut:", h.get("is_shortcut"),
              "| UnknownStation:", h.get("unknown_station"))

    if not hypotheses:
        print("\nLLM returned NO usable hypotheses; cannot proceed.")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "hypotheses": [],
            "injected_fact": None,
            "final_proof_path": [],
            "kb_with_injection": kb,
        }

    # Phase 3: probabilistic logic = try highest-confidence first, iterate if resume fails
    hypotheses_sorted = sorted(hypotheses, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)

    print("\n>>> Phase 3: Try hypotheses by confidence (inject + resume)")
    for idx, h in enumerate(hypotheses_sorted, start=1):
        clause = (h.get("clause") or "").strip()
        conf = float(h.get("confidence", 0.0))

        print(f"\n--- Attempt {idx}/{len(hypotheses_sorted)} ---")
        print(f"Candidate: {clause} | conf={conf:.3f}")

        kb2 = inject_fact_into_kb(kb, clause)

        resume_state = _best_resume_state_for_hypothesis(h, hard_result)
        if not resume_state:
            print("No resume state available; cannot resume from failure. Skipping.")
            continue

        # Resume from the *state that actually failed* (so we continue the original proof),
        # not from the root. On success, the returned path includes the prefix proof steps.
        res = bfs_prolog_resume_from_state(
            kb=kb2,
            start_current=resume_state["current"],
            start_remaining=resume_state.get("remaining", []),
            start_path_prefix=resume_state.get("path", []),
            start_depth=resume_state.get("depth", 0),
            max_depth=max_depth
        )

        if res.get("success"):
            print("\n>>> Result: SOFT_SUCCESS (root goal proven via injected fact)\n")
            return {
                "status": "SOFT_SUCCESS",
                "hard_result": hard_result,
                "hypotheses": hypotheses_sorted,
                "injected_fact": clause,
                "final_proof_path": res.get("proof_path", []),
                "kb_with_injection": kb2,
                "resumed_from": resume_state,
            }

        print("Resume failed for this hypothesis; trying next...")

    print("\n>>> Result: SOFT_FAILURE (no injected hypothesis led to a full proof)\n")
    return {
        "status": "SOFT_FAILURE",
        "hard_result": hard_result,
        "hypotheses": hypotheses_sorted,
        "injected_fact": None,
        "final_proof_path": [],
        "kb_with_injection": kb,
        "resumed_from": None,
    }


# ============================================================
# Utilities
# ============================================================

def omit_facts_from_kb(kb: str, omit_numbers):
    omit_numbers = set(omit_numbers)
    new_lines = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        num = int(m.group(1))
        if num in omit_numbers:
            continue

        new_lines.append(line)

    return "\n".join(new_lines)


# In[7]:



# ============================================================
# Main (exactly compatible with your provided main)
# ============================================================

if __name__ == "__main__":
    kb = """
    1. connected(union_square, 14th_street). # connected/2 = ADJACENT STOPS ONLY. Use only when two stations are immediate neighbors on the same line. Do NOT add shortcut edges that skip intermediate stations.
    2. connected(14th_street, 23rd_street).
    3. connected(23rd_street, 34th_street).
    4. connected(34th_street, times_square).
    5. connected(times_square, 42nd_street).
    6. connected(42nd_street, grand_central).
    7. connected(grand_central, bryant_park).
    8. reachable(X, Y) :- connected(X, Y). # reachable/2 = PATH EXISTS. One-hop reachable comes only from connected/2.
    9. reachable(X, Z) :- connected(X, Y), reachable(Y, Z). # reachable/2 is the transitive closure of connected/2 (multi-hop path following only adjacent edges).

    """

    print("===== FULL METRO KB =====")
    print(kb)
    print("====================================\n")

    # Omit fact 7 to force background reasoning
    kb_missing_fact = omit_facts_from_kb(kb, omit_numbers={7})

    print("===== METRO KB WITH FACT REMOVED =====")
    print(kb_missing_fact)
    print("========================================\n")

    test_goal = "reachable(union_square, bryant_park)"

    print("==============================")
    print(f"TEST QUERY: {test_goal}")
    print("==============================\n")

    print(">>> Running bfs_prolog_collect (hard-KB BFS)...")
    collect_result = bfs_prolog_collect(
        goal=test_goal,
        kb=kb_missing_fact,
        max_depth=20
    )
    print("Collect Result:", collect_result)
    print("\n----------------------------------------\n")

    print(">>> Running solve_with_background (full pipeline, reusing hard result)...")
    bg_result = solve_with_background(
        goal=test_goal,
        kb=kb_missing_fact,
        max_depth=20,
        hard_result=collect_result,
    )
    print("Solve-with-background Result:")
    print(bg_result)

    if bg_result.get("status") == "SOFT_SUCCESS":
        print("\n===== FULL PROOF PATH (ROOT GOAL PROVEN) =====")
        for step in bg_result.get("final_proof_path", []):
            print(step)
        print("============================================\n")


# In[1]:


import ollama
from collections import deque
import re
import json
from typing import Optional, Dict, Any, List, Tuple

# ============================================================
# Config / LLM setup
# ============================================================

client = ollama.Client()
model = "gpt-oss:20b"
# model = "qwen:14b"

DEBUG = False  # set to True to print raw LLM outputs for debugging


def ask_llm(prompt: str) -> str:
    resp = client.generate(model=model, prompt=prompt, options={'temperature': 0.0})
    answer = resp.get('response', '')
    if "...done thinking." in answer:
        return answer.split("...done thinking.")[-1].strip()
    return answer


# ============================================================
# Helpers for parsing / JSON
# ============================================================

def extract_first_json(text: str) -> str:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in: {text!r}")
    return match.group(0)


def split_inline_comment(s: str):
    if "#" not in s:
        return s.strip(), None
    code, comment = s.split("#", 1)
    code = code.strip()
    comment = comment.strip()
    return code, (comment if comment else None)


def parse_predicate(term: str):
    term = term.strip().rstrip('.')
    m = re.match(r'^([a-z_][a-zA-Z0-9_]*)\((.*)\)$', term)
    if not m:
        return None
    functor = m.group(1)
    args_raw = m.group(2).strip()
    if not args_raw:
        args = []
    else:
        args = [a.strip() for a in args_raw.split(',')]
    return functor, args


def is_variable(s: str) -> bool:
    s = s.strip()
    return bool(s) and (s[0].isupper() or s[0] == '_')


def strip_inline_comment(s: str) -> str:
    return s.split('#', 1)[0].rstrip()


# ============================================================
# Core Prolog helpers (unification + SLD one-step)
# ============================================================

def check_exact_match(goal: str, fact: str) -> bool:
    return goal.strip().rstrip('.') == fact.strip().rstrip('.')


def unify_args(args_goal, args_fact, env=None):
    if env is None:
        env = {}

    if len(args_goal) != len(args_fact):
        return None

    for g, f in zip(args_goal, args_fact):
        g = g.strip()
        f = f.strip()

        g_is_var = is_variable(g)
        f_is_var = is_variable(f)

        if not g_is_var and not f_is_var:
            if g != f:
                return None
            continue

        if g_is_var and not f_is_var:
            if g in env:
                if env[g] != f:
                    return None
            else:
                env[g] = f
            continue

        if not g_is_var and f_is_var:
            if f in env:
                if env[f] != g:
                    return None
            else:
                env[f] = g
            continue

        if g_is_var and f_is_var:
            if g in env and f in env:
                if env[g] != env[f]:
                    return None
            elif g in env:
                env[f] = env[g]
            elif f in env:
                env[g] = env[f]
            continue

    return env


def unify_arg_lists(args_rule_head, args_goal):
    return unify_args(args_rule_head, args_goal, env={})


def unify_with_fact(goal: str, fact: str):
    parsed_goal = parse_predicate(goal)
    parsed_fact = parse_predicate(fact)
    if parsed_goal is None or parsed_fact is None:
        return None

    fun_g, args_g = parsed_goal
    fun_f, args_f = parsed_fact

    if fun_g != fun_f or len(args_g) != len(args_f):
        return None

    if check_exact_match(goal, fact):
        return {}

    env = unify_args(args_g, args_f, env={})
    if env is None:
        return None
    return env


def apply_bindings(goals, bindings):
    if not bindings or not goals:
        return goals

    new_goals = []
    for g in goals:
        parsed = parse_predicate(g)
        if parsed is None:
            new_goals.append(g)
            continue

        functor, args = parsed
        new_args = []
        for a in args:
            a_stripped = a.strip()
            if is_variable(a_stripped) and a_stripped in bindings:
                new_args.append(bindings[a_stripped])
            else:
                new_args.append(a_stripped)

        new_goal = f"{functor}({', '.join(new_args)})"
        new_goals.append(new_goal)

    return new_goals


def find_matching_rules_only(goal, rules_list):
    parsed_goal = parse_predicate(goal)
    if parsed_goal is None:
        return []
    fun_g, args_g = parsed_goal
    arity_g = len(args_g)

    matching = []
    for num, head, body in rules_list:
        parsed_head = parse_predicate(head)
        if parsed_head is None:
            continue
        fun_h, args_h = parsed_head
        if fun_h == fun_g and len(args_h) == arity_g:
            matching.append(num)
    return matching


def substitute_in_atom(atom: str, bindings: dict) -> str:
    parsed = parse_predicate(atom)
    if parsed is None:
        return atom

    functor, args = parsed
    new_args = []
    for a in args:
        a_stripped = a.strip()
        if is_variable(a_stripped) and a_stripped in bindings:
            new_args.append(bindings[a_stripped])
        else:
            new_args.append(a_stripped)

    return f"{functor}({', '.join(new_args)})"


def split_body_atoms(body_str: str):
    body_str = body_str.strip()
    atoms = []
    current = []
    depth = 0

    for ch in body_str:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth = max(depth - 1, 0)
            current.append(ch)
        elif ch == ',' and depth == 0:
            atom = ''.join(current).strip()
            if atom:
                atoms.append(atom)
            current = []
        else:
            current.append(ch)

    atom = ''.join(current).strip()
    if atom:
        atoms.append(atom)

    return atoms


def get_subgoals(goal: str, rule_head: str, rule_body: str):
    parsed_goal = parse_predicate(goal)
    parsed_head = parse_predicate(rule_head)

    if parsed_goal is None or parsed_head is None:
        return None

    fun_g, args_g = parsed_goal
    fun_h, args_h = parsed_head

    if fun_g != fun_h or len(args_g) != len(args_h):
        return None

    bindings = unify_arg_lists(args_h, args_g)
    if bindings is None:
        return None

    body_str = rule_body.strip()
    if not body_str:
        return []

    body_atoms = split_body_atoms(body_str)
    if not body_atoms:
        return []

    subgoals = [substitute_in_atom(atom, bindings) for atom in body_atoms]
    return subgoals if subgoals else None


# ============================================================
# KB comment extraction (inline + full-line)
# ============================================================

def parse_kb_predicate_comments(kb: str):
    predicate_comments = {}
    pending_comments = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("#"):
            pending_comments.append(line.lstrip("#").strip())
            continue

        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        content_raw = m.group(2).strip()
        content, inline_comment = split_inline_comment(content_raw)

        clause = content.strip().rstrip(".")
        head_str = clause.split(":-", 1)[0].strip()
        parsed = parse_predicate(head_str)
        if parsed is None:
            pending_comments = []
            continue

        functor, args = parsed
        key = f"{functor}/{len(args)}"

        combined = []
        if pending_comments:
            combined.append(" ".join(pending_comments))
        if inline_comment:
            combined.append(inline_comment)

        if combined:
            predicate_comments[key] = " ".join(combined).strip()

        pending_comments = []

    return predicate_comments


# ============================================================
# KB parsing (facts + rules)
# ============================================================

def parse_kb_facts_rules(kb: str):
    facts = []
    rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        line = strip_inline_comment(line).strip()
        if not line:
            continue

        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not match:
            continue

        num = int(match.group(1))
        content_raw = match.group(2).strip()
        content, _ = split_inline_comment(content_raw)
        content = (content or "").strip()
        if not content:
            continue

        if ':-' in content:
            head, body = content.split(':-', 1)
            rules.append((num, head.strip(), body.strip().rstrip('.')))
        else:
            facts.append((num, content.rstrip('.')))

    return facts, rules


# ============================================================
# BFS Prolog engine (collect + full resume support)
# ============================================================

def bfs_prolog_collect(goal: str, kb: str, max_depth: int = 10):
    """
    BFS SLD (facts + rules). Returns:
      - success
      - proof_path (if success)
      - unresolved_atoms (set)
      - failed_state (a representative failed state)
      - failed_states (map: atom -> state info)
    """
    facts, rules = parse_kb_facts_rules(kb)

    queue = deque([(goal, [], [], 0)])
    visited = set()
    unresolved_atoms = set()

    failed_states = {}  # current_atom -> {current, remaining, path, depth}

    print(f"\n[COLLECT] Goal: {goal}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        if depth >= max_depth:
            unresolved_atoms.add(current)
            if current not in failed_states or depth > failed_states[current]["depth"]:
                failed_states[current] = {"current": current, "remaining": remaining, "path": path, "depth": depth}
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")

        progress = False

        # 1) Exact fact match
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")

                step = f"Fact {num}"
                new_path = path + [step]

                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return {
                        "success": True,
                        "proof_path": new_path,
                        "unresolved_atoms": set(),
                        "failed_state": None,
                        "failed_states": {}
                    }

                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, new_path, depth + 1))

                progress = True
                break

        if progress:
            continue

        # 2) Fact unification
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            progress = True
            print(f"  ✓ Fact {num}: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)
            step = f"Fact {num}"
            new_path = path + [step]

            if not instantiated:
                print(f"✓✓ SUCCESS at depth {depth + 1}")
                return {
                    "success": True,
                    "proof_path": new_path,
                    "unresolved_atoms": set(),
                    "failed_state": None,
                    "failed_states": {}
                }

            queue.append((instantiated[0], instantiated[1:], new_path, depth + 1))

        if progress:
            continue

        # 3) Rules
        matching_rules = find_matching_rules_only(current, rules)
        if matching_rules:
            print(f"  Matching rules: {matching_rules}")

        for rule_num in matching_rules:
            for num, head, body in rules:
                if num != rule_num:
                    continue

                subgoals = get_subgoals(current, head, body)
                if subgoals:
                    print(f"  Rule {num}: → {subgoals}")
                    progress = True
                    all_goals = subgoals + remaining
                    step = f"Rule {num}"
                    new_path = path + [step]
                    queue.append((all_goals[0], all_goals[1:], new_path, depth + 1))
                break

        if not progress:
            print(f"  ✗ No facts or rules apply to: {current}")
            unresolved_atoms.add(current)
            if current not in failed_states or depth > failed_states[current]["depth"]:
                failed_states[current] = {"current": current, "remaining": remaining, "path": path, "depth": depth}

    print("✗ FAILED (collect mode)")

    # Representative failed_state: choose deepest failed atom (more likely “closest” to completion)
    rep = None
    if failed_states:
        rep = max(failed_states.values(), key=lambda s: s["depth"])

    return {
        "success": False,
        "proof_path": [],
        "unresolved_atoms": unresolved_atoms,
        "failed_state": rep,
        "failed_states": failed_states
    }


def bfs_prolog_resume_from_state(
    kb: str,
    start_current: str,
    start_remaining: List[str],
    start_path_prefix: List[str],
    start_depth: int,
    max_depth: int = 10
):
    """
    Resume BFS SLD from an intermediate state and return a FULL proof path
    (prefix + continuation) that corresponds to proving the ROOT goal.

    IMPORTANT:
    - start_path_prefix is the path that led to (start_current, start_remaining).
    - On success, returns full proof steps including prefix.
    """
    facts, rules = parse_kb_facts_rules(kb)

    queue = deque([(start_current, list(start_remaining), list(start_path_prefix), start_depth)])
    visited = set()

    print(f"\n[RESUME BFS] Starting from failed node:")
    print(f"  Depth {start_depth}: {start_current}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        if depth >= max_depth:
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")

        # 1) Exact fact match
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")
                new_path = path + [f"Fact {num}"]

                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return {"success": True, "proof_path": new_path, "final_depth": depth + 1}

                queue.append((remaining[0], remaining[1:], new_path, depth + 1))
                break  # exact match: single transition

        # 2) Fact unification
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            print(f"  ✓ Fact {num}: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)
            new_path = path + [f"Fact {num}"]

            if not instantiated:
                print(f"✓✓ SUCCESS at depth {depth + 1}")
                return {"success": True, "proof_path": new_path, "final_depth": depth + 1}

            queue.append((instantiated[0], instantiated[1:], new_path, depth + 1))

        # 3) Rules
        matching_rules = find_matching_rules_only(current, rules)
        if matching_rules:
            print(f"  Matching rules: {matching_rules}")

        for rule_num in matching_rules:
            for num, head, body in rules:
                if num != rule_num:
                    continue
                subgoals = get_subgoals(current, head, body)
                if not subgoals:
                    break
                print(f"  Rule {num}: → {subgoals}")
                all_goals = subgoals + remaining
                new_path = path + [f"Rule {num}"]
                queue.append((all_goals[0], all_goals[1:], new_path, depth + 1))
                break

    print("✗ RESUME FAILED")
    return {"success": False, "proof_path": [], "final_depth": None}


# ============================================================
# Background hypothesis generation (FAST) + shortcut filtering
# ============================================================

_BG_HYP_CACHE = {}  # (kb_sig, goal, tuple(atoms), max_hyp_per_atom) -> list[hypothesis]


def _kb_signature_for_bg(kb: str) -> str:
    parts = []
    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        content_raw = m.group(2).strip()
        content, _ = split_inline_comment(content_raw)
        content = (content or "").strip()
        if not content:
            continue

        content = content.rstrip('.').strip()
        if content.startswith("connected(") or content.startswith("reachable("):
            parts.append(content)

    return str(hash("\n".join(parts)))


def _extract_connected_facts_and_stations(kb: str):
    hard_adj = {}
    hard_edges = set()
    stations = set()
    connected_facts = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        content_raw = m.group(2).strip()
        content, _ = split_inline_comment(content_raw)
        content = (content or "").strip()
        if not content:
            continue

        if ':-' in content:
            continue

        atom0 = content.rstrip('.').strip()
        p = parse_predicate(atom0)
        if p and p[0] == "connected" and len(p[1]) == 2:
            a, b = p[1][0].strip(), p[1][1].strip()
            hard_adj.setdefault(a, []).append(b)
            hard_edges.add((a, b))
            stations.add(a)
            stations.add(b)
            connected_facts.append(f"connected({a}, {b}).")

    return hard_adj, hard_edges, stations, connected_facts


def _shortest_path_len_bounded(adj: dict, src: str, dst: str, max_depth: int = 25):
    if src == dst:
        return 0
    q = deque([(src, 0)])
    seen = {src}
    while q:
        node, d = q.popleft()
        if d >= max_depth:
            continue
        for nb in adj.get(node, []):
            if nb == dst:
                return d + 1
            if nb not in seen:
                seen.add(nb)
                q.append((nb, d + 1))
    return None


def _infer_preferred_atom_for_chain(goal: str, hard_adj: dict) -> Optional[str]:
    gp = parse_predicate(goal)
    if not gp or len(gp[1]) != 2:
        return None
    goal_src, goal_dst = gp[1][0].strip(), gp[1][1].strip()

    last = goal_src
    seen = set()
    while last in hard_adj and len(hard_adj[last]) == 1 and last not in seen:
        seen.add(last)
        last = hard_adj[last][0]

    if last and goal_dst:
        return f"connected({last}, {goal_dst})"
    return None


def _select_atoms_for_bg(unresolved_atoms, preferred_atom: Optional[str], max_atoms: int):
    atom_list = list(unresolved_atoms)
    ground = []
    for atom in atom_list:
        atom = atom.strip()
        if not atom or '(' not in atom or ')' not in atom:
            continue
        inside = atom.split('(', 1)[1].rsplit(')', 1)[0]
        if re.search(r'\b[A-Z_]\w*\b', inside):
            continue
        ground.append(atom)

    if preferred_atom:
        ground = [preferred_atom] + [a for a in ground if a != preferred_atom]

    if max_atoms is not None:
        ground = ground[:max_atoms]
    return ground


def generate_background_hypotheses_fast(
    goal: str,
    kb: str,
    hard_result: dict,
    predicate_comments: dict,
    max_atoms: int = 6,
    max_hyp_per_atom: int = 2,
    prompt_fact_limit: int = 60,
):
    unresolved_atoms = hard_result.get("unresolved_atoms", set()) if hard_result else set()
    if not unresolved_atoms:
        return []

    hard_adj, hard_edges, stations, connected_facts = _extract_connected_facts_and_stations(kb)
    preferred_atom = _infer_preferred_atom_for_chain(goal, hard_adj)
    atoms = _select_atoms_for_bg(unresolved_atoms, preferred_atom, max_atoms=max_atoms)
    if not atoms:
        print("[generate_background_hypotheses_fast] No suitable ground atoms.")
        return []

    kb_sig = _kb_signature_for_bg(kb)
    cache_key = (kb_sig, goal, tuple(atoms), max_hyp_per_atom)
    if cache_key in _BG_HYP_CACHE:
        return _BG_HYP_CACHE[cache_key]

    facts_for_prompt = connected_facts[:prompt_fact_limit]

    needed = {"connected/2"}
    for a in atoms:
        p = parse_predicate(a)
        if p:
            needed.add(f"{p[0]}/{len(p[1])}")

    semantic_lines = []
    for k in sorted(needed):
        if k in predicate_comments:
            semantic_lines.append(f"- {k}: {predicate_comments[k]}")
    semantic_hint_block = "\n".join(semantic_lines) if semantic_lines else "(none)"

    atoms_block = "\n".join([f"- {a}" for a in atoms])

    prompt = f"""
You are a cautious Prolog expert. You are proposing missing FACTS only.

GOAL:
  {goal}

HARD KB SNAPSHOT (partial; do NOT assume any other edges exist):
Connected facts (sample):
{chr(10).join(facts_for_prompt)}

Predicate semantics (ground truth):
{semantic_hint_block}

The proof failed because these subgoals could not be proven:
{atoms_block}

Task:
For each failed subgoal above, propose up to {max_hyp_per_atom} additional Prolog FACTS
that are likely true and would help prove the GOAL.

Constraints:
- Output ONLY connected/2 FACTS. No rules.
- Each fact MUST end with a period.
- Prefer facts that connect existing stations seen in the snapshot.
- If a suggested edge is uncertain or could be a "shortcut", lower confidence.

Return ONLY valid JSON:

{{
  "by_atom": {{
    "connected(42nd_street, bryant_park)": [
      {{"clause":"connected(grand_central, bryant_park).","confidence":0.95}},
      {{"clause":"connected(42nd_street, bryant_park).","confidence":0.55}}
    ]
  }}
}}
""".strip()

    raw = ask_llm(prompt).strip()
    if DEBUG:
        print("\n[DEBUG generate_background_hypotheses_fast] raw:\n", raw)

    try:
        data = json.loads(extract_first_json(raw))
    except Exception as e:
        print("[generate_background_hypotheses_fast] JSON parse error:", e)
        print("Raw LLM output:", raw)
        _BG_HYP_CACHE[cache_key] = []
        return []

    by_atom = data.get("by_atom", {})
    if not isinstance(by_atom, dict):
        _BG_HYP_CACHE[cache_key] = []
        return []

    def norm_clause(cl: str) -> Optional[str]:
        if not cl:
            return None
        cl = cl.strip()
        if not cl.endswith('.'):
            cl += '.'
        atom_str = cl.rstrip('.').strip()
        p = parse_predicate(atom_str)
        if not (p and p[0] == "connected" and len(p[1]) == 2):
            return None
        u, v = p[1][0].strip(), p[1][1].strip()
        if u == v:
            return None
        return f"connected({u}, {v})."

    def is_shortcut_edge(u: str, v: str) -> Tuple[bool, Optional[int]]:
        d = _shortest_path_len_bounded(hard_adj, u, v, max_depth=30)
        # If there is already a path of length >=2, then u->v is a shortcut edge
        return (d is not None and d >= 2), d

    out = []
    for atom, proposals in by_atom.items():
        if not isinstance(proposals, list):
            continue
        for item in proposals[:max_hyp_per_atom]:
            if not isinstance(item, dict):
                continue

            clause = norm_clause(item.get("clause", ""))
            if clause is None:
                continue

            try:
                conf = float(item.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            conf = max(0.0, min(1.0, conf))

            p = parse_predicate(clause.rstrip('.'))
            u, v = p[1][0].strip(), p[1][1].strip()

            # already in hard KB
            if (u, v) in hard_edges:
                continue

            # HARD BAN shortcut edges (your requirement)
            shortcut, d = is_shortcut_edge(u, v)
            if shortcut:
                continue  # DO NOT add to hypotheses

            unknown_station = bool(stations) and (u not in stations or v not in stations)

            out.append({
                "clause": clause,
                "confidence": conf,
                "from_atom": atom,
                "is_shortcut": shortcut,
                "shortcut_len": d,
                "unknown_station": unknown_station,
            })

    # Dedup keep best confidence
    dedup = {}
    for h in out:
        key = h["clause"]
        if key not in dedup or h["confidence"] > dedup[key]["confidence"]:
            dedup[key] = h

    result = list(dedup.values())
    _BG_HYP_CACHE[cache_key] = result
    return result


# ============================================================
# Inject fact into KB (preserve numbering)
# ============================================================

def _find_max_line_number_in_kb(kb: str) -> int:
    max_num = 0
    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue
        num = int(m.group(1))
        if num > max_num:
            max_num = num
    return max_num


def inject_fact_into_kb(kb: str, fact_clause: str) -> str:
    """
    fact_clause must be like: connected(a,b). OR connected(a,b)
    This function appends a NEW numbered fact line.
    """
    fact_clause = (fact_clause or "").strip()
    if not fact_clause:
        return kb
    if not fact_clause.endswith('.'):
        fact_clause += '.'

    max_num = _find_max_line_number_in_kb(kb)
    new_num = max_num + 1

    kb_lines = [ln.rstrip() for ln in kb.strip().split("\n") if ln.strip()]
    kb_lines.append(f"{new_num}. {fact_clause}")
    return "\n".join(kb_lines)


# ============================================================
# Orchestration: probabilistic logic = try hypotheses by confidence
# ============================================================

def _best_resume_state_for_hypothesis(h: dict, hard_result: dict):
    """
    Prefer resuming from the failed state that matches the hypothesis's from_atom,
    otherwise fallback to deepest failure (representative).
    """
    failed_states = hard_result.get("failed_states", {}) or {}
    from_atom = (h.get("from_atom") or "").strip()

    if from_atom and from_atom in failed_states:
        return failed_states[from_atom]

    # If hypothesis is a clause connected(u,v), also try matching on current=connected(u,v)
    clause = (h.get("clause") or "").strip().rstrip(".")
    if clause and clause in failed_states:
        return failed_states[clause]

    rep = hard_result.get("failed_state")
    if rep:
        return rep

    # last resort: choose deepest
    if failed_states:
        return max(failed_states.values(), key=lambda s: s.get("depth", -1))

    return None


def solve_with_background(
    goal: str,
    kb: str,
    max_depth: int = 10,
    hard_result=None,
    max_atoms: int = 6,
    max_hyp_per_atom: int = 2,
):
    """
    Pipeline:
      1) Hard BFS collect failures
      2) LLM hypotheses
      3) Sort hypotheses by confidence desc
      4) For each hypothesis:
           - inject into KB
           - resume BFS from best matching failed state
           - if success: return full proof path (root proven)
    """
    predicate_comments = parse_kb_predicate_comments(kb)

    print("\n========================================")
    print(f"SOLVE WITH BACKGROUND (TOP-K FACT INJECTION w/ RESUME): {goal}")
    print("========================================\n")

    if hard_result is None:
        print(">>> Phase 1: Hard-KB BFS (bfs_prolog_collect)")
        hard_result = bfs_prolog_collect(goal, kb, max_depth=max_depth)
        print("Hard-KB result:", hard_result)
    else:
        print(">>> Phase 1: Hard-KB BFS result already computed, reusing it.")
        print("Hard-KB result:", hard_result)

    if hard_result.get("success"):
        print("\n>>> Result: HARD_SUCCESS (no background hypotheses needed)\n")
        return {
            "status": "HARD_SUCCESS",
            "hard_result": hard_result,
            "hypotheses": [],
            "injected_fact": None,
            "final_proof_path": hard_result.get("proof_path", []),
            "kb_with_injection": kb,
        }

    unresolved_atoms = hard_result.get("unresolved_atoms", set())
    if not unresolved_atoms:
        print("\nNo unresolved atoms to explain; cannot generate hypotheses.")
        print(">>> Result: FAILURE\n")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "hypotheses": [],
            "injected_fact": None,
            "final_proof_path": [],
            "kb_with_injection": kb,
        }

    print("\n>>> Phase 2: Generate background hypotheses (FAST)")
    print("Unresolved atoms:", unresolved_atoms)

    hypotheses = generate_background_hypotheses_fast(
        goal=goal,
        kb=kb,
        hard_result=hard_result,
        predicate_comments=predicate_comments,
        max_atoms=max_atoms,
        max_hyp_per_atom=max_hyp_per_atom,
        prompt_fact_limit=60,
    ) or []

    print("Hypotheses returned by LLM:")
    for h in hypotheses:
        print("  - Clause:", h.get("clause"),
              "| Conf:", h.get("confidence"),
              "| From atom:", h.get("from_atom"),
              "| Shortcut:", h.get("is_shortcut"),
              "| UnknownStation:", h.get("unknown_station"))

    if not hypotheses:
        print("\nLLM returned NO usable hypotheses; cannot proceed.")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "hypotheses": [],
            "injected_fact": None,
            "final_proof_path": [],
            "kb_with_injection": kb,
        }

    # Phase 3: probabilistic logic = try highest-confidence first, iterate if resume fails
    hypotheses_sorted = sorted(hypotheses, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)

    print("\n>>> Phase 3: Try hypotheses by confidence (inject + resume)")
    for idx, h in enumerate(hypotheses_sorted, start=1):
        clause = (h.get("clause") or "").strip()
        conf = float(h.get("confidence", 0.0))

        print(f"\n--- Attempt {idx}/{len(hypotheses_sorted)} ---")
        print(f"Candidate: {clause} | conf={conf:.3f}")

        kb2 = inject_fact_into_kb(kb, clause)

        resume_state = _best_resume_state_for_hypothesis(h, hard_result)
        if not resume_state:
            print("No resume state available; cannot resume from failure. Skipping.")
            continue

        # Resume from the *state that actually failed* (so we continue the original proof),
        # not from the root. On success, the returned path includes the prefix proof steps.
        res = bfs_prolog_resume_from_state(
            kb=kb2,
            start_current=resume_state["current"],
            start_remaining=resume_state.get("remaining", []),
            start_path_prefix=resume_state.get("path", []),
            start_depth=resume_state.get("depth", 0),
            max_depth=max_depth
        )

        if res.get("success"):
            print("\n>>> Result: SOFT_SUCCESS (root goal proven via injected fact)\n")
            return {
                "status": "SOFT_SUCCESS",
                "hard_result": hard_result,
                "hypotheses": hypotheses_sorted,
                "injected_fact": clause,
                "final_proof_path": res.get("proof_path", []),
                "kb_with_injection": kb2,
                "resumed_from": resume_state,
            }

        print("Resume failed for this hypothesis; trying next...")

    print("\n>>> Result: SOFT_FAILURE (no injected hypothesis led to a full proof)\n")
    return {
        "status": "SOFT_FAILURE",
        "hard_result": hard_result,
        "hypotheses": hypotheses_sorted,
        "injected_fact": None,
        "final_proof_path": [],
        "kb_with_injection": kb,
        "resumed_from": None,
    }


# ============================================================
# Utilities
# ============================================================

def omit_facts_from_kb(kb: str, omit_numbers):
    omit_numbers = set(omit_numbers)
    new_lines = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        num = int(m.group(1))
        if num in omit_numbers:
            continue

        new_lines.append(line)

    return "\n".join(new_lines)


# In[3]:


import ollama
from collections import deque
import re
import json
from typing import Optional, Dict, Any, List, Tuple

# ============================================================
# Config / LLM setup
# ============================================================

client = ollama.Client()
model = "gpt-oss:20b"
# model = "qwen:14b"

DEBUG = False  # set to True to print raw LLM outputs for debugging


def ask_llm(prompt: str) -> str:
    resp = client.generate(model=model, prompt=prompt, options={'temperature': 0.0})
    answer = resp.get('response', '')
    if "...done thinking." in answer:
        return answer.split("...done thinking.")[-1].strip()
    return answer


# ============================================================
# Helpers for parsing / JSON
# ============================================================

def extract_first_json(text: str) -> str:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in: {text!r}")
    return match.group(0)


def split_inline_comment(s: str):
    if "#" not in s:
        return s.strip(), None
    code, comment = s.split("#", 1)
    code = code.strip()
    comment = comment.strip()
    return code, (comment if comment else None)


def parse_predicate(term: str):
    term = term.strip().rstrip('.')
    m = re.match(r'^([a-z_][a-zA-Z0-9_]*)\((.*)\)$', term)
    if not m:
        return None
    functor = m.group(1)
    args_raw = m.group(2).strip()
    if not args_raw:
        args = []
    else:
        args = [a.strip() for a in args_raw.split(',')]
    return functor, args


def is_variable(s: str) -> bool:
    s = s.strip()
    return bool(s) and (s[0].isupper() or s[0] == '_')


def strip_inline_comment(s: str) -> str:
    return s.split('#', 1)[0].rstrip()


# ============================================================
# Core Prolog helpers (unification + SLD one-step)
# ============================================================

def check_exact_match(goal: str, fact: str) -> bool:
    return goal.strip().rstrip('.') == fact.strip().rstrip('.')


def unify_args(args_goal, args_fact, env=None):
    if env is None:
        env = {}

    if len(args_goal) != len(args_fact):
        return None

    for g, f in zip(args_goal, args_fact):
        g = g.strip()
        f = f.strip()

        g_is_var = is_variable(g)
        f_is_var = is_variable(f)

        if not g_is_var and not f_is_var:
            if g != f:
                return None
            continue

        if g_is_var and not f_is_var:
            if g in env:
                if env[g] != f:
                    return None
            else:
                env[g] = f
            continue

        if not g_is_var and f_is_var:
            if f in env:
                if env[f] != g:
                    return None
            else:
                env[f] = g
            continue

        if g_is_var and f_is_var:
            if g in env and f in env:
                if env[g] != env[f]:
                    return None
            elif g in env:
                env[f] = env[g]
            elif f in env:
                env[g] = env[f]
            continue

    return env


def unify_arg_lists(args_rule_head, args_goal):
    return unify_args(args_rule_head, args_goal, env={})


def unify_with_fact(goal: str, fact: str):
    parsed_goal = parse_predicate(goal)
    parsed_fact = parse_predicate(fact)
    if parsed_goal is None or parsed_fact is None:
        return None

    fun_g, args_g = parsed_goal
    fun_f, args_f = parsed_fact

    if fun_g != fun_f or len(args_g) != len(args_f):
        return None

    if check_exact_match(goal, fact):
        return {}

    env = unify_args(args_g, args_f, env={})
    if env is None:
        return None
    return env


def apply_bindings(goals, bindings):
    if not bindings or not goals:
        return goals

    new_goals = []
    for g in goals:
        parsed = parse_predicate(g)
        if parsed is None:
            new_goals.append(g)
            continue

        functor, args = parsed
        new_args = []
        for a in args:
            a_stripped = a.strip()
            if is_variable(a_stripped) and a_stripped in bindings:
                new_args.append(bindings[a_stripped])
            else:
                new_args.append(a_stripped)

        new_goal = f"{functor}({', '.join(new_args)})"
        new_goals.append(new_goal)

    return new_goals


def find_matching_rules_only(goal, rules_list):
    parsed_goal = parse_predicate(goal)
    if parsed_goal is None:
        return []
    fun_g, args_g = parsed_goal
    arity_g = len(args_g)

    matching = []
    for num, head, body in rules_list:
        parsed_head = parse_predicate(head)
        if parsed_head is None:
            continue
        fun_h, args_h = parsed_head
        if fun_h == fun_g and len(args_h) == arity_g:
            matching.append(num)
    return matching


def substitute_in_atom(atom: str, bindings: dict) -> str:
    parsed = parse_predicate(atom)
    if parsed is None:
        return atom

    functor, args = parsed
    new_args = []
    for a in args:
        a_stripped = a.strip()
        if is_variable(a_stripped) and a_stripped in bindings:
            new_args.append(bindings[a_stripped])
        else:
            new_args.append(a_stripped)

    return f"{functor}({', '.join(new_args)})"


def split_body_atoms(body_str: str):
    body_str = body_str.strip()
    atoms = []
    current = []
    depth = 0

    for ch in body_str:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth = max(depth - 1, 0)
            current.append(ch)
        elif ch == ',' and depth == 0:
            atom = ''.join(current).strip()
            if atom:
                atoms.append(atom)
            current = []
        else:
            current.append(ch)

    atom = ''.join(current).strip()
    if atom:
        atoms.append(atom)

    return atoms


def get_subgoals(goal: str, rule_head: str, rule_body: str):
    parsed_goal = parse_predicate(goal)
    parsed_head = parse_predicate(rule_head)

    if parsed_goal is None or parsed_head is None:
        return None

    fun_g, args_g = parsed_goal
    fun_h, args_h = parsed_head

    if fun_g != fun_h or len(args_g) != len(args_h):
        return None

    bindings = unify_arg_lists(args_h, args_g)
    if bindings is None:
        return None

    body_str = rule_body.strip()
    if not body_str:
        return []

    body_atoms = split_body_atoms(body_str)
    if not body_atoms:
        return []

    subgoals = [substitute_in_atom(atom, bindings) for atom in body_atoms]
    return subgoals if subgoals else None


# ============================================================
# KB comment extraction (inline + full-line)
# ============================================================

def parse_kb_predicate_comments(kb: str):
    predicate_comments = {}
    pending_comments = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("#"):
            pending_comments.append(line.lstrip("#").strip())
            continue

        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        content_raw = m.group(2).strip()
        content, inline_comment = split_inline_comment(content_raw)

        clause = content.strip().rstrip(".")
        head_str = clause.split(":-", 1)[0].strip()
        parsed = parse_predicate(head_str)
        if parsed is None:
            pending_comments = []
            continue

        functor, args = parsed
        key = f"{functor}/{len(args)}"

        combined = []
        if pending_comments:
            combined.append(" ".join(pending_comments))
        if inline_comment:
            combined.append(inline_comment)

        if combined:
            predicate_comments[key] = " ".join(combined).strip()

        pending_comments = []

    return predicate_comments


# ============================================================
# KB parsing (facts + rules)
# ============================================================

def parse_kb_facts_rules(kb: str):
    facts = []
    rules = []

    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        line = strip_inline_comment(line).strip()
        if not line:
            continue

        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not match:
            continue

        num = int(match.group(1))
        content_raw = match.group(2).strip()
        content, _ = split_inline_comment(content_raw)
        content = (content or "").strip()
        if not content:
            continue

        if ':-' in content:
            head, body = content.split(':-', 1)
            rules.append((num, head.strip(), body.strip().rstrip('.')))
        else:
            facts.append((num, content.rstrip('.')))

    return facts, rules


# ============================================================
# BFS Prolog engine (collect + full resume support)
# ============================================================

def bfs_prolog_collect(goal: str, kb: str, max_depth: int = 10):
    """
    BFS SLD (facts + rules). Returns:
      - success
      - proof_path (if success)
      - unresolved_atoms (set)
      - failed_state (a representative failed state)
      - failed_states (map: atom -> state info)
    """
    facts, rules = parse_kb_facts_rules(kb)

    queue = deque([(goal, [], [], 0)])
    visited = set()
    unresolved_atoms = set()

    failed_states = {}  # current_atom -> {current, remaining, path, depth}

    print(f"\n[COLLECT] Goal: {goal}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        if depth >= max_depth:
            unresolved_atoms.add(current)
            if current not in failed_states or depth > failed_states[current]["depth"]:
                failed_states[current] = {"current": current, "remaining": remaining, "path": path, "depth": depth}
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")

        progress = False

        # 1) Exact fact match
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")

                step = f"Fact {num}"
                new_path = path + [step]

                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return {
                        "success": True,
                        "proof_path": new_path,
                        "unresolved_atoms": set(),
                        "failed_state": None,
                        "failed_states": {}
                    }

                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append((next_goal, next_remaining, new_path, depth + 1))

                progress = True
                break

        if progress:
            continue

        # 2) Fact unification
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            progress = True
            print(f"  ✓ Fact {num}: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)
            step = f"Fact {num}"
            new_path = path + [step]

            if not instantiated:
                print(f"✓✓ SUCCESS at depth {depth + 1}")
                return {
                    "success": True,
                    "proof_path": new_path,
                    "unresolved_atoms": set(),
                    "failed_state": None,
                    "failed_states": {}
                }

            queue.append((instantiated[0], instantiated[1:], new_path, depth + 1))

        if progress:
            continue

        # 3) Rules
        matching_rules = find_matching_rules_only(current, rules)
        if matching_rules:
            print(f"  Matching rules: {matching_rules}")

        for rule_num in matching_rules:
            for num, head, body in rules:
                if num != rule_num:
                    continue

                subgoals = get_subgoals(current, head, body)
                if subgoals:
                    print(f"  Rule {num}: → {subgoals}")
                    progress = True
                    all_goals = subgoals + remaining
                    step = f"Rule {num}"
                    new_path = path + [step]
                    queue.append((all_goals[0], all_goals[1:], new_path, depth + 1))
                break

        if not progress:
            print(f"  ✗ No facts or rules apply to: {current}")
            unresolved_atoms.add(current)
            if current not in failed_states or depth > failed_states[current]["depth"]:
                failed_states[current] = {"current": current, "remaining": remaining, "path": path, "depth": depth}

    print("✗ FAILED (collect mode)")

    # Representative failed_state: choose deepest failed atom (more likely “closest” to completion)
    rep = None
    if failed_states:
        rep = max(failed_states.values(), key=lambda s: s["depth"])

    return {
        "success": False,
        "proof_path": [],
        "unresolved_atoms": unresolved_atoms,
        "failed_state": rep,
        "failed_states": failed_states
    }


def bfs_prolog_resume_from_state(
    kb: str,
    start_current: str,
    start_remaining: List[str],
    start_path_prefix: List[str],
    start_depth: int,
    max_depth: int = 10
):
    """
    Resume BFS SLD from an intermediate state and return a FULL proof path
    (prefix + continuation) that corresponds to proving the ROOT goal.

    IMPORTANT:
    - start_path_prefix is the path that led to (start_current, start_remaining).
    - On success, returns full proof steps including prefix.
    """
    facts, rules = parse_kb_facts_rules(kb)

    queue = deque([(start_current, list(start_remaining), list(start_path_prefix), start_depth)])
    visited = set()

    print(f"\n[RESUME BFS] Starting from failed node:")
    print(f"  Depth {start_depth}: {start_current}")
    print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        if depth >= max_depth:
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        print(f"Depth {depth}: {current}")
        if remaining:
            print(f"  Remaining: {remaining}")

        # 1) Exact fact match
        for num, fact in facts:
            if check_exact_match(current, fact):
                print(f"  ✓ Fact {num} matches exactly!")
                new_path = path + [f"Fact {num}"]

                if not remaining:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return {"success": True, "proof_path": new_path, "final_depth": depth + 1}

                queue.append((remaining[0], remaining[1:], new_path, depth + 1))
                break  # exact match: single transition

        # 2) Fact unification
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            print(f"  ✓ Fact {num}: {fact}")
            print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)
            new_path = path + [f"Fact {num}"]

            if not instantiated:
                print(f"✓✓ SUCCESS at depth {depth + 1}")
                return {"success": True, "proof_path": new_path, "final_depth": depth + 1}

            queue.append((instantiated[0], instantiated[1:], new_path, depth + 1))

        # 3) Rules
        matching_rules = find_matching_rules_only(current, rules)
        if matching_rules:
            print(f"  Matching rules: {matching_rules}")

        for rule_num in matching_rules:
            for num, head, body in rules:
                if num != rule_num:
                    continue
                subgoals = get_subgoals(current, head, body)
                if not subgoals:
                    break
                print(f"  Rule {num}: → {subgoals}")
                all_goals = subgoals + remaining
                new_path = path + [f"Rule {num}"]
                queue.append((all_goals[0], all_goals[1:], new_path, depth + 1))
                break

    print("✗ RESUME FAILED")
    return {"success": False, "proof_path": [], "final_depth": None}


# ============================================================
# Background hypothesis generation (FAST) + shortcut filtering
# ============================================================

_BG_HYP_CACHE = {}  # (kb_sig, goal, tuple(atoms), max_hyp_per_atom) -> list[hypothesis]


def _kb_signature_for_bg(kb: str) -> str:
    parts = []
    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        content_raw = m.group(2).strip()
        content, _ = split_inline_comment(content_raw)
        content = (content or "").strip()
        if not content:
            continue

        content = content.rstrip('.').strip()
        if content.startswith("connected(") or content.startswith("reachable("):
            parts.append(content)

    return str(hash("\n".join(parts)))


def _extract_connected_facts_and_stations(kb: str):
    hard_adj = {}
    hard_edges = set()
    stations = set()
    connected_facts = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        content_raw = m.group(2).strip()
        content, _ = split_inline_comment(content_raw)
        content = (content or "").strip()
        if not content:
            continue

        if ':-' in content:
            continue

        atom0 = content.rstrip('.').strip()
        p = parse_predicate(atom0)
        if p and p[0] == "connected" and len(p[1]) == 2:
            a, b = p[1][0].strip(), p[1][1].strip()
            hard_adj.setdefault(a, []).append(b)
            hard_edges.add((a, b))
            stations.add(a)
            stations.add(b)
            connected_facts.append(f"connected({a}, {b}).")

    return hard_adj, hard_edges, stations, connected_facts


def _shortest_path_len_bounded(adj: dict, src: str, dst: str, max_depth: int = 25):
    if src == dst:
        return 0
    q = deque([(src, 0)])
    seen = {src}
    while q:
        node, d = q.popleft()
        if d >= max_depth:
            continue
        for nb in adj.get(node, []):
            if nb == dst:
                return d + 1
            if nb not in seen:
                seen.add(nb)
                q.append((nb, d + 1))
    return None


def _infer_preferred_atom_for_chain(goal: str, hard_adj: dict) -> Optional[str]:
    gp = parse_predicate(goal)
    if not gp or len(gp[1]) != 2:
        return None
    goal_src, goal_dst = gp[1][0].strip(), gp[1][1].strip()

    last = goal_src
    seen = set()
    while last in hard_adj and len(hard_adj[last]) == 1 and last not in seen:
        seen.add(last)
        last = hard_adj[last][0]

    if last and goal_dst:
        return f"connected({last}, {goal_dst})"
    return None


def _select_atoms_for_bg(unresolved_atoms, preferred_atom: Optional[str], max_atoms: int):
    atom_list = list(unresolved_atoms)
    ground = []
    for atom in atom_list:
        atom = atom.strip()
        if not atom or '(' not in atom or ')' not in atom:
            continue
        inside = atom.split('(', 1)[1].rsplit(')', 1)[0]
        if re.search(r'\b[A-Z_]\w*\b', inside):
            continue
        ground.append(atom)

    if preferred_atom:
        ground = [preferred_atom] + [a for a in ground if a != preferred_atom]

    if max_atoms is not None:
        ground = ground[:max_atoms]
    return ground


def generate_background_hypotheses_fast(
    goal: str,
    kb: str,
    hard_result: dict,
    predicate_comments: dict,
    max_atoms: int = 6,
    max_hyp_per_atom: int = 2,
    prompt_fact_limit: int = 60,
):
    unresolved_atoms = hard_result.get("unresolved_atoms", set()) if hard_result else set()
    if not unresolved_atoms:
        return []

    hard_adj, hard_edges, stations, connected_facts = _extract_connected_facts_and_stations(kb)
    preferred_atom = _infer_preferred_atom_for_chain(goal, hard_adj)
    atoms = _select_atoms_for_bg(unresolved_atoms, preferred_atom, max_atoms=max_atoms)
    if not atoms:
        print("[generate_background_hypotheses_fast] No suitable ground atoms.")
        return []

    kb_sig = _kb_signature_for_bg(kb)
    cache_key = (kb_sig, goal, tuple(atoms), max_hyp_per_atom)
    if cache_key in _BG_HYP_CACHE:
        return _BG_HYP_CACHE[cache_key]

    facts_for_prompt = connected_facts[:prompt_fact_limit]

    needed = {"connected/2"}
    for a in atoms:
        p = parse_predicate(a)
        if p:
            needed.add(f"{p[0]}/{len(p[1])}")

    semantic_lines = []
    for k in sorted(needed):
        if k in predicate_comments:
            semantic_lines.append(f"- {k}: {predicate_comments[k]}")
    semantic_hint_block = "\n".join(semantic_lines) if semantic_lines else "(none)"

    atoms_block = "\n".join([f"- {a}" for a in atoms])

    facts_for_prompt = connected_facts[:min(prompt_fact_limit, 12)]   # was 60
    atoms = atoms[:min(len(atoms), 4)]                               # was 6
    atoms_block = "\n".join(atoms)

    prompt = f"""
    Return JSON only.

    Goal: {goal}

    Known true connected/2 facts (directed):
    {chr(10).join(facts_for_prompt)}

    Unproved subgoals (each is a connected/2 atom):
    {atoms_block}

    Task: For EACH unproved atom above, propose up to {max_hyp_per_atom} missing FACTS of the form:
    connected(a, b).
    Use ONLY station names that appear in the known facts or the unproved atoms.
    If a fact is likely true in the NYC subway (general knowledge), give higher confidence (0..1).
    If unsure, still propose but lower confidence.

    Output schema (MUST include every atom key, even if you give an empty list):
    {{"by_atom": {{"<atom>":[{{"clause":"connected(x,y).","confidence":0.9}}]}}}}
    """.strip()

    raw = ask_llm(prompt).strip()
    if DEBUG:
        print("\n[DEBUG generate_background_hypotheses_fast] raw:\n", raw)

    try:
        data = json.loads(extract_first_json(raw))
    except Exception as e:
        print("[generate_background_hypotheses_fast] JSON parse error:", e)
        print("Raw LLM output:", raw)
        _BG_HYP_CACHE[cache_key] = []
        return []

    by_atom = data.get("by_atom", {})
    if not isinstance(by_atom, dict):
        _BG_HYP_CACHE[cache_key] = []
        return []

    def norm_clause(cl: str) -> Optional[str]:
        if not cl:
            return None
        cl = cl.strip()
        if not cl.endswith('.'):
            cl += '.'
        atom_str = cl.rstrip('.').strip()
        p = parse_predicate(atom_str)
        if not (p and p[0] == "connected" and len(p[1]) == 2):
            return None
        u, v = p[1][0].strip(), p[1][1].strip()
        if u == v:
            return None
        return f"connected({u}, {v})."

    def is_shortcut_edge(u: str, v: str) -> Tuple[bool, Optional[int]]:
        d = _shortest_path_len_bounded(hard_adj, u, v, max_depth=30)
        # If there is already a path of length >=2, then u->v is a shortcut edge
        return (d is not None and d >= 2), d

    out = []
    for atom, proposals in by_atom.items():
        if not isinstance(proposals, list):
            continue
        for item in proposals[:max_hyp_per_atom]:
            if not isinstance(item, dict):
                continue

            clause = norm_clause(item.get("clause", ""))
            if clause is None:
                continue

            try:
                conf = float(item.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            conf = max(0.0, min(1.0, conf))

            p = parse_predicate(clause.rstrip('.'))
            u, v = p[1][0].strip(), p[1][1].strip()

            # already in hard KB
            if (u, v) in hard_edges:
                continue

            # HARD BAN shortcut edges (your requirement)
            shortcut, d = is_shortcut_edge(u, v)
            if shortcut:
                continue  # DO NOT add to hypotheses

            unknown_station = bool(stations) and (u not in stations or v not in stations)

            out.append({
                "clause": clause,
                "confidence": conf,
                "from_atom": atom,
                "is_shortcut": shortcut,
                "shortcut_len": d,
                "unknown_station": unknown_station,
            })

    # Dedup keep best confidence
    dedup = {}
    for h in out:
        key = h["clause"]
        if key not in dedup or h["confidence"] > dedup[key]["confidence"]:
            dedup[key] = h

    result = list(dedup.values())
    _BG_HYP_CACHE[cache_key] = result
    return result


# ============================================================
# Inject fact into KB (preserve numbering)
# ============================================================

def _find_max_line_number_in_kb(kb: str) -> int:
    max_num = 0
    for line in kb.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue
        num = int(m.group(1))
        if num > max_num:
            max_num = num
    return max_num


def inject_fact_into_kb(kb: str, fact_clause: str) -> str:
    """
    fact_clause must be like: connected(a,b). OR connected(a,b)
    This function appends a NEW numbered fact line.
    """
    fact_clause = (fact_clause or "").strip()
    if not fact_clause:
        return kb
    if not fact_clause.endswith('.'):
        fact_clause += '.'

    max_num = _find_max_line_number_in_kb(kb)
    new_num = max_num + 1

    kb_lines = [ln.rstrip() for ln in kb.strip().split("\n") if ln.strip()]
    kb_lines.append(f"{new_num}. {fact_clause}")
    return "\n".join(kb_lines)


# ============================================================
# Orchestration: probabilistic logic = try hypotheses by confidence
# ============================================================

def _best_resume_state_for_hypothesis(h: dict, hard_result: dict):
    """
    Prefer resuming from the failed state that matches the hypothesis's from_atom,
    otherwise fallback to deepest failure (representative).
    """
    failed_states = hard_result.get("failed_states", {}) or {}
    from_atom = (h.get("from_atom") or "").strip()

    if from_atom and from_atom in failed_states:
        return failed_states[from_atom]

    # If hypothesis is a clause connected(u,v), also try matching on current=connected(u,v)
    clause = (h.get("clause") or "").strip().rstrip(".")
    if clause and clause in failed_states:
        return failed_states[clause]

    rep = hard_result.get("failed_state")
    if rep:
        return rep

    # last resort: choose deepest
    if failed_states:
        return max(failed_states.values(), key=lambda s: s.get("depth", -1))

    return None


def solve_with_background(
    goal: str,
    kb: str,
    max_depth: int = 10,
    hard_result=None,
    max_atoms: int = 6,
    max_hyp_per_atom: int = 2,
):
    """
    Pipeline:
      1) Hard BFS collect failures
      2) LLM hypotheses
      3) Sort hypotheses by confidence desc
      4) For each hypothesis:
           - inject into KB
           - resume BFS from best matching failed state
           - if success: return full proof path (root proven)
    """
    predicate_comments = parse_kb_predicate_comments(kb)

    print("\n========================================")
    print(f"SOLVE WITH BACKGROUND (TOP-K FACT INJECTION w/ RESUME): {goal}")
    print("========================================\n")

    if hard_result is None:
        print(">>> Phase 1: Hard-KB BFS (bfs_prolog_collect)")
        hard_result = bfs_prolog_collect(goal, kb, max_depth=max_depth)
        print("Hard-KB result:", hard_result)
    else:
        print(">>> Phase 1: Hard-KB BFS result already computed, reusing it.")
        print("Hard-KB result:", hard_result)

    if hard_result.get("success"):
        print("\n>>> Result: HARD_SUCCESS (no background hypotheses needed)\n")
        return {
            "status": "HARD_SUCCESS",
            "hard_result": hard_result,
            "hypotheses": [],
            "injected_fact": None,
            "final_proof_path": hard_result.get("proof_path", []),
            "kb_with_injection": kb,
        }

    unresolved_atoms = hard_result.get("unresolved_atoms", set())
    if not unresolved_atoms:
        print("\nNo unresolved atoms to explain; cannot generate hypotheses.")
        print(">>> Result: FAILURE\n")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "hypotheses": [],
            "injected_fact": None,
            "final_proof_path": [],
            "kb_with_injection": kb,
        }

    print("\n>>> Phase 2: Generate background hypotheses (FAST)")
    print("Unresolved atoms:", unresolved_atoms)

    hypotheses = generate_background_hypotheses_fast(
        goal=goal,
        kb=kb,
        hard_result=hard_result,
        predicate_comments=predicate_comments,
        max_atoms=max_atoms,
        max_hyp_per_atom=max_hyp_per_atom,
        prompt_fact_limit=60,
    ) or []

    print("Hypotheses returned by LLM:")
    for h in hypotheses:
        print("  - Clause:", h.get("clause"),
              "| Conf:", h.get("confidence"),
              "| From atom:", h.get("from_atom"),
              "| Shortcut:", h.get("is_shortcut"),
              "| UnknownStation:", h.get("unknown_station"))

    if not hypotheses:
        print("\nLLM returned NO usable hypotheses; cannot proceed.")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "hypotheses": [],
            "injected_fact": None,
            "final_proof_path": [],
            "kb_with_injection": kb,
        }

    # Phase 3: probabilistic logic = try highest-confidence first, iterate if resume fails
    hypotheses_sorted = sorted(hypotheses, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)

    print("\n>>> Phase 3: Try hypotheses by confidence (inject + resume)")
    for idx, h in enumerate(hypotheses_sorted, start=1):
        clause = (h.get("clause") or "").strip()
        conf = float(h.get("confidence", 0.0))

        print(f"\n--- Attempt {idx}/{len(hypotheses_sorted)} ---")
        print(f"Candidate: {clause} | conf={conf:.3f}")

        kb2 = inject_fact_into_kb(kb, clause)

        resume_state = _best_resume_state_for_hypothesis(h, hard_result)
        if not resume_state:
            print("No resume state available; cannot resume from failure. Skipping.")
            continue

        # Resume from the *state that actually failed* (so we continue the original proof),
        # not from the root. On success, the returned path includes the prefix proof steps.
        res = bfs_prolog_resume_from_state(
            kb=kb2,
            start_current=resume_state["current"],
            start_remaining=resume_state.get("remaining", []),
            start_path_prefix=resume_state.get("path", []),
            start_depth=resume_state.get("depth", 0),
            max_depth=max_depth
        )

        if res.get("success"):
            print("\n>>> Result: SOFT_SUCCESS (root goal proven via injected fact)\n")
            return {
                "status": "SOFT_SUCCESS",
                "hard_result": hard_result,
                "hypotheses": hypotheses_sorted,
                "injected_fact": clause,
                "final_proof_path": res.get("proof_path", []),
                "kb_with_injection": kb2,
                "resumed_from": resume_state,
            }

        print("Resume failed for this hypothesis; trying next...")

    print("\n>>> Result: SOFT_FAILURE (no injected hypothesis led to a full proof)\n")
    return {
        "status": "SOFT_FAILURE",
        "hard_result": hard_result,
        "hypotheses": hypotheses_sorted,
        "injected_fact": None,
        "final_proof_path": [],
        "kb_with_injection": kb,
        "resumed_from": None,
    }


# ============================================================
# Utilities
# ============================================================

def omit_facts_from_kb(kb: str, omit_numbers):
    omit_numbers = set(omit_numbers)
    new_lines = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue

        num = int(m.group(1))
        if num in omit_numbers:
            continue

        new_lines.append(line)

    return "\n".join(new_lines)


# In[4]:



# ============================================================
# Main (exactly compatible with your provided main)
# ============================================================

if __name__ == "__main__":
    kb = """
    1. connected(union_square, 14th_street). # connected/2 = ADJACENT STOPS ONLY. Use only when two stations are immediate neighbors on the same line. Do NOT add shortcut edges that skip intermediate stations.
    2. connected(14th_street, 23rd_street).
    3. connected(23rd_street, 34th_street).
    4. connected(34th_street, times_square).
    5. connected(times_square, 42nd_street).
    6. connected(42nd_street, grand_central).
    7. connected(grand_central, bryant_park).
    8. reachable(X, Y) :- connected(X, Y). # reachable/2 = PATH EXISTS. One-hop reachable comes only from connected/2.
    9. reachable(X, Z) :- connected(X, Y), reachable(Y, Z). # reachable/2 is the transitive closure of connected/2 (multi-hop path following only adjacent edges).

    """

    print("===== FULL METRO KB =====")
    print(kb)
    print("====================================\n")

    # Omit fact 7 to force background reasoning
    kb_missing_fact = omit_facts_from_kb(kb, omit_numbers={7})

    print("===== METRO KB WITH FACT REMOVED =====")
    print(kb_missing_fact)
    print("========================================\n")

    test_goal = "reachable(union_square, bryant_park)"

    print("==============================")
    print(f"TEST QUERY: {test_goal}")
    print("==============================\n")

    print(">>> Running bfs_prolog_collect (hard-KB BFS)...")
    collect_result = bfs_prolog_collect(
        goal=test_goal,
        kb=kb_missing_fact,
        max_depth=20
    )
    print("Collect Result:", collect_result)
    print("\n----------------------------------------\n")

    print(">>> Running solve_with_background (full pipeline, reusing hard result)...")
    bg_result = solve_with_background(
        goal=test_goal,
        kb=kb_missing_fact,
        max_depth=20,
        hard_result=collect_result,
    )
    print("Solve-with-background Result:")
    print(bg_result)

    if bg_result.get("status") == "SOFT_SUCCESS":
        print("\n===== FULL PROOF PATH (ROOT GOAL PROVEN) =====")
        for step in bg_result.get("final_proof_path", []):
            print(step)
        print("============================================\n")


# In[ ]:





# In[2]:



# ============================================================
# Main (exactly compatible with your provided main)
# ============================================================

if __name__ == "__main__":
    kb = """
    1. connected(union_square, 14th_street). # connected/2 = ADJACENT STOPS ONLY. Use only when two stations are immediate neighbors on the same line. Do NOT add shortcut edges that skip intermediate stations.
    2. connected(14th_street, 23rd_street).
    3. connected(23rd_street, 34th_street).
    4. connected(34th_street, times_square).
    5. connected(times_square, 42nd_street).
    6. connected(42nd_street, grand_central).
    7. connected(grand_central, bryant_park).
    8. reachable(X, Y) :- connected(X, Y). # reachable/2 = PATH EXISTS. One-hop reachable comes only from connected/2.
    9. reachable(X, Z) :- connected(X, Y), reachable(Y, Z). # reachable/2 is the transitive closure of connected/2 (multi-hop path following only adjacent edges).

    """

    print("===== FULL METRO KB =====")
    print(kb)
    print("====================================\n")

    # Omit fact 7 to force background reasoning
    kb_missing_fact = omit_facts_from_kb(kb, omit_numbers={7})

    print("===== METRO KB WITH FACT REMOVED =====")
    print(kb_missing_fact)
    print("========================================\n")

    test_goal = "reachable(union_square, bryant_park)"

    print("==============================")
    print(f"TEST QUERY: {test_goal}")
    print("==============================\n")

    print(">>> Running bfs_prolog_collect (hard-KB BFS)...")
    collect_result = bfs_prolog_collect(
        goal=test_goal,
        kb=kb_missing_fact,
        max_depth=20
    )
    print("Collect Result:", collect_result)
    print("\n----------------------------------------\n")

    print(">>> Running solve_with_background (full pipeline, reusing hard result)...")
    bg_result = solve_with_background(
        goal=test_goal,
        kb=kb_missing_fact,
        max_depth=20,
        hard_result=collect_result,
    )
    print("Solve-with-background Result:")
    print(bg_result)

    if bg_result.get("status") == "SOFT_SUCCESS":
        print("\n===== FULL PROOF PATH (ROOT GOAL PROVEN) =====")
        for step in bg_result.get("final_proof_path", []):
            print(step)
        print("============================================\n")


# In[1]:



# ============================================================
# Main (exactly compatible with your provided main)
# ============================================================

if __name__ == "__main__":
    kb = """
    1. connected(union_square, 14th_street). # connected/2 = ADJACENT STOPS ONLY. Use only when two stations are immediate neighbors on the same line. Do NOT add shortcut edges that skip intermediate stations.
    2. connected(14th_street, 23rd_street).
    3. connected(23rd_street, 34th_street).
    4. connected(34th_street, times_square).
    5. connected(times_square, 42nd_street).
    6. connected(42nd_street, grand_central).
    7. connected(grand_central, bryant_park).
    8. reachable(X, Y) :- connected(X, Y). # reachable/2 = PATH EXISTS. One-hop reachable comes only from connected/2.
    9. reachable(X, Z) :- connected(X, Y), reachable(Y, Z). # reachable/2 is the transitive closure of connected/2 (multi-hop path following only adjacent edges).

    """

    print("===== FULL METRO KB =====")
    print(kb)
    print("====================================\n")

    # Omit fact 7 to force background reasoning
    kb_missing_fact = omit_facts_from_kb(kb, omit_numbers={7})

    print("===== METRO KB WITH FACT REMOVED =====")
    print(kb_missing_fact)
    print("========================================\n")

    test_goal = "reachable(union_square, bryant_park)"

    print("==============================")
    print(f"TEST QUERY: {test_goal}")
    print("==============================\n")

    print(">>> Running bfs_prolog_collect (hard-KB BFS)...")
    collect_result = bfs_prolog_collect(
        goal=test_goal,
        kb=kb_missing_fact,
        max_depth=20
    )
    print("Collect Result:", collect_result)
    print("\n----------------------------------------\n")

    print(">>> Running solve_with_background (full pipeline, reusing hard result)...")
    bg_result = solve_with_background(
        goal=test_goal,
        kb=kb_missing_fact,
        max_depth=20,
        hard_result=collect_result,
    )
    print("Solve-with-background Result:")
    print(bg_result)

    if bg_result.get("status") == "SOFT_SUCCESS":
        print("\n===== FULL PROOF PATH (ROOT GOAL PROVEN) =====")
        for step in bg_result.get("final_proof_path", []):
            print(step)
        print("============================================\n")


# In[5]:


print(ask_llm("Return exactly: {\"ok\": true}"))


# In[27]:


def safe_generate_hypotheses(goal, kb, unresolved_atoms):
    """
    A safety wrapper around generate_background_hypotheses.
    Ensures:
        - Return value is ALWAYS a list (never None)
        - LLM failures or JSON parsing errors do NOT crash pipeline
        - Empty or invalid output becomes []
    """

    try:
        hyps = generate_background_hypotheses(
            goal=goal,
            kb=kb,
            unresolved_atoms=unresolved_atoms
        )
    except Exception as e:
        print("[safe_generate_hypotheses] ERROR in generate_background_hypotheses:", e)
        return []

    # Case 1: LLM returned None
    if hyps is None:
        print("[safe_generate_hypotheses] No hypotheses returned (None).")
        return []

    # Case 2: LLM returned something not iterable
    if not isinstance(hyps, list):
        print("[safe_generate_hypotheses] Hypotheses not a list:", hyps)
        return []

    # Case 3: LLM returned a list but it's empty
    if len(hyps) == 0:
        print("[safe_generate_hypotheses] Empty hypothesis list.")
        return []

    # Otherwise it's valid
    return hyps


hypotheses = safe_generate_hypotheses(
    goal=test,
    kb=kb_missing_3,
    unresolved_atoms=collect_result["unresolved_atoms"]
)


# In[ ]:





# In[2]:


get_ipython().system('pip install wordnet')


# In[5]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install python-prover9')


# In[15]:


# Cell 1: Setup and Install WordNet data
import nltk
import ollama
from typing import List, Dict, Tuple
import random
import subprocess
import tempfile
import json

# Download WordNet data (run this once)
nltk.download('wordnet')
nltk.download('omw-1.4')  # For lemmatization

from nltk.corpus import wordnet

# Cell 2: Helper functions for sampling
def sample_name():
    """Sample from a list of names"""
    # You can load from the GitHub repos mentioned in the paper
    # For now, using a simple list
    names = ["Sawyer", "Amy", "Jack", "Bella", "Moriarty", "Colt", "Anakin", "Buster"]
    return random.choice(names)

def sample_keyword():
    """Sample characteristic keywords from WordNet"""
    # Get random adjectives from WordNet
    all_adjectives = list(wordnet.all_synsets(pos='a'))
    if all_adjectives:
        synset = random.choice(all_adjectives)
        # Get the first lemma (word) from the synset
        return synset.lemmas()[0].name().replace('_', ' ')
    else:
        # Fallback list
        return random.choice(['swift', 'loyal', 'elegant', 'fierce', 'gentle'])

# Cell 3: Background Story Generator
class BackgroundStoryGenerator:
    def __init__(self, llm_client, model="gpt-oss:20b"):
        self.llm = llm_client
        self.model = model
    
    def generate(self, name, keyword):
        prompt = f"""You will be given a keyword and a name. Generate a background story with no more than 150 words.
        
keyword: {keyword}
name: {name}

Your answer should be in JSON format with keys: category, story.
The category should be 'human', 'animal', or 'object' based on the nature of the character."""

        response = self.llm.generate(
            model=self.model,
            prompt=prompt
        )
        
        try:
            return json.loads(response['response'])
        except:
            # Fallback if JSON parsing fails
            return {
                "category": "human",
                "story": response['response']
            }

# Cell 4: Prover9 Interface
class Prover9Interface:
    def __init__(self):
        self.prover_path = "prover9"  # Assumes prover9 is in PATH
        
    def prove(self, premises, goal):
        """Use Prover9 to validate logical inference"""
        input_text = self.format_prover9_input(premises, goal)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
            f.write(input_text)
            temp_file = f.name
        
        try:
            result = subprocess.run(
                [self.prover_path, '-f', temp_file],
                capture_output=True,
                text=True,
                timeout=5
            )
            return self.parse_result(result.stdout)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "proven": False}
        except FileNotFoundError:
            print("Prover9 not found. Please install it or update the path.")
            return {"status": "error", "proven": False}
        finally:
            import os
            os.unlink(temp_file)
    
    def format_prover9_input(self, premises, goal):
        """Format input for Prover9"""
        input_text = "formulas(assumptions).\n"
        for premise in premises:
            input_text += f"  {premise}.\n"
        input_text += "end_of_list.\n\n"
        input_text += "formulas(goals).\n"
        input_text += f"  {goal}.\n"
        input_text += "end_of_list.\n"
        return input_text
    
    def parse_result(self, output):
        """Parse Prover9 output"""
        if "THEOREM PROVED" in output:
            return {"status": "proven", "proven": True}
        elif "SOS empty" in output:
            return {"status": "failed", "proven": False}
        else:
            return {"status": "unknown", "proven": False}

# Cell 5: Main ProverGen Framework
class ProverGen:
    def __init__(self, llm_model="gpt-oss:20b"):
        self.llm = ollama.Client()
        self.model = llm_model
        self.story_generator = BackgroundStoryGenerator(self.llm, llm_model)
        self.prover = Prover9Interface()
        
    def generate_fol_problem(self, difficulty="medium"):
        # 1. Sample name and keyword
        name = sample_name()
        keyword = sample_keyword()
        
        print(f"Generating problem for {name} with keyword '{keyword}'")
        
        # 2. Generate background story
        story_data = self.story_generator.generate(name, keyword)
        
        # 3. Generate logic skeleton (simplified for now)
        skeleton = self.generate_logic_skeleton(name, difficulty)
        
        # 4. Translate to natural language
        problem = self.translate_skeleton(skeleton, story_data)
        
        return problem
    
    def generate_logic_skeleton(self, subject, difficulty):
        """Simplified logic skeleton generation"""
        # This is a placeholder - the full implementation would be more complex
        if difficulty == "easy":
            return {
                "facts": [f"domesticated({subject})", f"elephant({subject})"],
                "rules": [f"all x (elephant(x) -> (domesticated(x) | wild(x)) & -(domesticated(x) & wild(x)))"],
                "goal": f"-wild({subject})"
            }
        # Add more complexity for medium and hard
        return skeleton
    
    def translate_skeleton(self, skeleton, story_data):
        """Translate logical skeleton to natural language"""
        # Placeholder for translation logic
        return {
            "background": story_data["story"],
            "context": skeleton,
            "question": "Based on the above information, is the following statement true, false, or uncertain?"
        }

# Cell 6: Test the system
generator = ProverGen()
problem = generator.generate_fol_problem(difficulty="easy")
print(json.dumps(problem, indent=2))


# In[12]:





# In[13]:





# In[16]:





# In[14]:





# In[17]:





# In[18]:





# In[19]:





# In[20]:





# In[21]:





# In[22]:





# In[23]:





# In[26]:





# In[15]:


"""
SLD Resolution with BFS + LLM-backed Unification / Rule Application
- Includes:
  * parse_KB       (from 1.1)
  * bfs_sld        (1.2 BFS engine)
  * unify_with_fact,
    apply_bindings,
    find_matching_rules_only,
    get_subgoals   (1.3 LLM helpers)
"""

import ollama
from collections import deque
import re
import json

# --- LLM setup ---

client = ollama.Client()
model = "gpt-oss:20b"  # adjust as needed


def ask_llm(prompt: str) -> str:
    resp = client.generate(model=model, prompt=prompt, options={"temperature": 0.0})
    answer = resp.get("response", "")
    if "...done thinking." in answer:
        return answer.split("...done thinking.")[-1].strip()
    return answer


# --- JSON helper ---

def extract_first_json(text: str) -> str:
    """
    Extract the first {...} JSON object from possibly messy text.
    This makes the code robust to models that add a bit of extra junk.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in: {text!r}")
    return match.group(0)


# --- 1.1: Parse KB into facts + rules ---

def parse_KB(kb_text: str):
    """
    Parse a numbered Prolog-style KB into:
        - facts: [(num, atom_string)]
        - rules: [(num, head_string, [body_atom_strings])]
    """

    facts = []
    rules = []

    for raw_line in kb_text.strip().split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        m = re.match(r"^(\d+)\.\s*(.+)$", line)
        if not m:
            continue

        num = int(m.group(1))
        content = m.group(2).strip()

        # Rule vs fact
        if ":-" in content:
            head_part, body_part = content.split(":-", 1)
            head = head_part.strip()
            body = body_part.strip().rstrip(".")

            # Split body on commas into a list of atoms
            body_atoms = [b.strip() for b in body.split(",")]

            rules.append((num, head, body_atoms))
        else:
            # Fact: remove trailing dot
            atom = content.rstrip(".")
            facts.append((num, atom))

    return facts, rules


# --- Small structural helper ---

def check_exact_match(goal: str, fact: str) -> bool:
    """Check if goal matches fact exactly (no variables, string equality)."""
    return goal.strip() == fact.strip()


# --- 1.3: LLM-backed helpers -----------------------------------------------

def unify_with_fact(goal: str, fact: str):
    """
    Check if goal unifies with fact and return bindings, using strict JSON.

    Returns:
        None      -> NO unification
        {}        -> EXACT ground match (no variables)
        dict      -> bindings, e.g. {"Y": "times_square"}
    """
    # Cheap local check first
    if check_exact_match(goal, fact):
        return {}  # Exact match, no bindings

    prompt = f"""You are a STRICT Prolog unification engine.

Your ONLY job is to decide if the Prolog Goal unifies with the Prolog Fact.

Goal: {goal}
Fact: {fact}

Use ONLY the symbols that appear in Goal and Fact.
Do NOT invent new constants, variables, or predicates.
If you are uncertain, you MUST choose 'NO'.

Respond in EXACTLY ONE of these JSON formats, with NO extra text:

1) If they do NOT unify:
   {{ "result": "NO" }}

2) If they unify and there are NO variables (exact ground match):
   {{ "result": "EXACT" }}

3) If they unify and there ARE variables:
   {{ "result": "UNIFY", "bindings": {{"VarName1": "atom1", "VarName2": "atom2"}} }}

Rules:
- VarName keys MUST be exactly the variable names from Goal/Fact (uppercase or starting with uppercase).
- atom values MUST be lowercase_with_underscores atoms (Prolog atoms), but represented as JSON strings.
- NO explanation, NO prose. JSON ONLY.
"""

    response = ask_llm(prompt).strip()
    try:
        data = json.loads(extract_first_json(response))
    except Exception as e:
        print("Unify parse error:", e, "Raw:", response)
        return None

    res = str(data.get("result", "")).upper()
    if res == "NO":
        return None
    if res == "EXACT":
        return {}
    if res == "UNIFY":
        bindings = data.get("bindings", {})
        return bindings if bindings is not None else {}
    return None


def apply_bindings(goals, bindings):
    """
    Apply variable bindings to goals using the LLM and structured JSON.

    goals: list[str]   e.g. ["reachable(Y, Z)", "connected(Z, X)"]
    bindings: dict     e.g. {"Y": "times_square"}

    Returns: list[str] of instantiated goals.
    """
    if not bindings or not goals:
        return goals

    prompt = f"""You are a Prolog substitution engine.

Bindings (Python dict): {bindings}
Goals (list of Prolog goals): {goals}

Apply the bindings to EACH goal exactly as Prolog would:
- Replace each variable in the goals according to the bindings.
- Do NOT change predicate names.
- Do NOT add or remove goals.
- Do NOT introduce any new symbols.

Respond ONLY in this JSON format:

{{
  "goals": [
    "goal1(instantiated, here)",
    "goal2(...)", 
    ...
  ]
}}

If something is unclear, return the input goals unchanged in that JSON format.
NO explanation or extra text, JSON ONLY.
"""

    response = ask_llm(prompt).strip()
    try:
        data = json.loads(extract_first_json(response))
        new_goals = data.get("goals", [])
        # Sanity filter: only keep things that look like Prolog predicates
        preds = [g for g in new_goals if re.match(r"^[a-z_]+\([^)]*\)$", g)]
        return preds if preds else goals
    except Exception as e:
        print("Apply bindings parse error:", e, "Raw:", response)
        return goals


def find_matching_rules_only(goal, rules_list):
    """
    Find ONLY rules (not facts) whose HEAD can unify with the given goal.

    rules_list: list[(num, head, body_list)]
    Returns: list[int] of rule numbers.
    """
    if not rules_list:
        return []

    rules_text = "\n".join(
        [f"{num}. {head} :- {', '.join(body)}" for num, head, body in rules_list]
    )

    prompt = f"""You are a Prolog rule matcher.

Goal: {goal}

Rules (numbered):
{rules_text}

Task:
Return the list of rule NUMBERS whose HEAD can unify with the Goal.
- Only consider rules shown above (they always contain ':-').
- Do NOT consider facts.
- Do NOT invent additional rules or modify heads.
- If you are uncertain, assume that a rule does NOT match.

Respond ONLY in this JSON format:

{{ "rules": [1, 3, 5] }}

If no rules match, respond:

{{ "rules": [] }}

No explanations, no extra text. JSON ONLY.
"""

    response = ask_llm(prompt).strip()
    try:
        data = json.loads(extract_first_json(response))
        numbers = data.get("rules", [])
        # Filter to only valid rule numbers present in rules_list
        valid_nums = [int(n) for n in numbers if any(int(n) == r[0] for r in rules_list)]
        return valid_nums
    except Exception as e:
        print("Rule match parse error:", e, "Raw:", response)
        return []


def get_subgoals(goal, rule_head, rule_body_atoms):
    """
    Get subgoals after unifying a goal with a rule, via JSON.

    rule_body_atoms: list[str], e.g. ["connected(X, Y)", "reachable(Y, Z)"]

    Returns:
        list[str] subgoals, or None if rule cannot be applied.
    """
    body_str = ", ".join(rule_body_atoms)

    prompt = f"""You are performing ONE step of SLD resolution in Prolog.

Goal: {goal}
Rule: {rule_head} :- {body_str}

Steps:
1. Try to unify the Goal with the Rule head.
2. If unification FAILS, this rule CANNOT be used.
3. If unification SUCCEEDS, apply the most general unifier to the rule body.
4. Return the resulting subgoals (the instantiated body goals) in order.

Important constraints:
- Use ONLY the Goal and the given Rule. Do NOT use any other rules or facts.
- Do NOT invent new predicates, arguments, or constants.
- If unification fails OR you are unsure, treat it as failing.

Respond ONLY in one of these JSON forms:

a) If the rule CANNOT be used (no unification):
   {{ "subgoals": [] }}

b) If the rule CAN be used:
   {{
     "subgoals": [
       "first_subgoal(...)",
       "second_subgoal(...)",
       ...
     ]
   }}

No explanations or extra text. JSON ONLY.
"""

    response = ask_llm(prompt).strip()
    try:
        data = json.loads(extract_first_json(response))
        subs = data.get("subgoals", [])
        if not subs:
            return None
        preds = [g for g in subs if re.match(r"^[a-z_]+\([^)]*\)$", g)]
        return preds if preds else None
    except Exception as e:
        print("Subgoal parse error:", e, "Raw:", response)
        return None


# --- 1.2: BFS SLD Engine ----------------------------------------------------

def bfs_sld(goal: str, kb_text: str, max_depth: int = 10, verbose: bool = True):
    """
    Generic BFS SLD resolution engine over a Prolog-style KB.

    Uses:
      - exact fact matching
      - LLM-backed unification with facts
      - LLM-backed rule matching and SLD step

    Returns a dict:
    {
      "success": bool,
      "proof_path": list of step descriptions (if success),
      "unresolved_atoms": set of atoms that couldn't be proven (if failure)
    }
    """

    facts, rules = parse_KB(kb_text)

    # BFS queue: (current_goal, remaining_goals, path, depth)
    queue = deque([(goal, [], [], 0)])
    visited = set()
    unresolved_atoms = set()

    if verbose:
        print(f"\nGoal: {goal}")
        print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        if depth > max_depth:
            unresolved_atoms.add(current)
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        if verbose:
            print(f"Depth {depth}: {current}")
            if remaining:
                print(f"  Remaining: {remaining}")

        # 1) Exact fact match
        fact_matched = False
        for num, fact_atom in facts:
            if check_exact_match(current, fact_atom):
                if verbose:
                    print(f"  ✓ Exact fact match: Fact {num}: {fact_atom}")

                step_desc = ("Fact", num, fact_atom, {})
                if not remaining:
                    if verbose:
                        print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return {
                        "success": True,
                        "proof_path": path + [step_desc],
                        "unresolved_atoms": set()
                    }

                next_goal = remaining[0]
                next_remaining = remaining[1:]
                queue.append(
                    (next_goal, next_remaining, path + [step_desc], depth + 1)
                )
                fact_matched = True
                break

        if fact_matched:
            continue

        # 2) Unification with facts (variables allowed)
        unified_any_fact = False
        for num, fact_atom in facts:
            bindings = unify_with_fact(current, fact_atom)
            if bindings is None:
                continue

            unified_any_fact = True
            if verbose:
                print(f"  ✓ Unify with Fact {num}: {fact_atom}")
                print(f"    Bindings: {bindings}")

            instantiated_remaining = apply_bindings(remaining, bindings)

            step_desc = ("Fact", num, fact_atom, bindings)

            if not instantiated_remaining:
                if verbose:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                return {
                    "success": True,
                    "proof_path": path + [step_desc],
                    "unresolved_atoms": set()
                }

            next_goal = instantiated_remaining[0]
            next_remaining = instantiated_remaining[1:]
            queue.append(
                (next_goal, next_remaining, path + [step_desc], depth + 1)
            )

        if unified_any_fact:
            continue

        # 3) Rules (SLD step)
        matching = find_matching_rules_only(current, rules)
        if matching and verbose:
            print(f"  Matching rules: {matching}")

        any_rule_applied = False
        for rule_num in matching:
            # find the rule tuple
            for num, head, body_atoms in rules:
                if num != rule_num:
                    continue

                subgoals = get_subgoals(current, head, body_atoms)
                if not subgoals:
                    continue

                any_rule_applied = True
                if verbose:
                    print(f"  Rule {num}: {head} :- {', '.join(body_atoms)}")
                    print(f"    → {subgoals}")

                all_goals = subgoals + remaining
                next_goal = all_goals[0]
                next_remaining = all_goals[1:]
                step_desc = ("Rule", num, head, body_atoms)
                queue.append(
                    (next_goal, next_remaining, path + [step_desc], depth + 1)
                )

        if not any_rule_applied and not unified_any_fact and not fact_matched:
            # Nothing could be done with this goal
            unresolved_atoms.add(current)

    if verbose:
        print("✗ FAILED")

    return {
        "success": False,
        "proof_path": [],
        "unresolved_atoms": unresolved_atoms
    }


# --- Simple test harness (optional) -----------------------------------------
if __name__ == "__main__":
    kb = """
    1. connected(union_square, times_square).
    2. connected(times_square, grand_central).
    3. connected(grand_central, bryant_park).
    4. reachable(X, Y) :- connected(X, Y).
    5. reachable(X, Z) :- connected(X, Y), reachable(Y, Z).
    """

    result = bfs_sld("reachable(union_square, grand_central)", kb, max_depth=10)
    print("\nResult:", result)

